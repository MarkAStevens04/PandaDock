"""
Parallel search algorithms for PandaDock.
This module provides parallel implementations of search algorithms for molecular docking
that leverage multi-core CPUs for improved performance.

Optimized version with:
- Better code organization using mixins
- Numba JIT compilation for performance-critical functions
- Improved error handling and robustness
- Support for flexible ligand docking
- Hybrid search algorithms
- Better documentation
"""

import numpy as np
import copy
import random
import time
import multiprocessing as mp
from scipy.spatial.transform import Rotation, Slerp
import os
from scipy.optimize import minimize
import logging
from pathlib import Path
from scipy.spatial import cKDTree
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from tqdm import tqdm

# Try to import numba for JIT compilation, fall back to a no-op decorator if not available
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    # Define a no-op JIT decorator if numba is not available
    def jit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        else:
            def decorator(func):
                return func
            return decorator
    
    # Define a fallback for parallel range
    prange = range
    HAS_NUMBA = False
    logging.warning("Numba not available. Install with 'pip install numba' for improved performance.")

from .search import DockingSearch
from .search import GeneticAlgorithm, RandomSearch
from .utils import (
    calculate_rmsd, is_within_grid, detect_steric_clash, 
    generate_spherical_grid, generate_cartesian_grid, 
    is_inside_sphere, random_point_in_sphere, local_optimize_pose, enforce_sphere_boundary, is_fully_inside_sphere, 
    save_intermediate_result, update_status, reposition_inside_sphere, setup_logging
)

# ------------------------------------------------------------------------------
# Performance-critical functions with JIT compilation
# ------------------------------------------------------------------------------

@jit(nopython=True, parallel=True, fastmath=True)
def fast_clash_detection(ligand_coords, protein_coords, clash_threshold=1.5):
    """
    Fast clash detection using Numba JIT compilation.
    
    Parameters:
    -----------
    ligand_coords : numpy.ndarray
        Coordinates of ligand atoms (N x 3)
    protein_coords : numpy.ndarray
        Coordinates of protein atoms (M x 3)
    clash_threshold : float
        Distance threshold for clash detection (Å)
        
    Returns:
    --------
    bool
        True if clash detected, False otherwise
    """
    for i in range(ligand_coords.shape[0]):
        lig_coord = ligand_coords[i]
        for j in range(protein_coords.shape[0]):
            dist = 0.0
            for k in range(3):  # x, y, z dimensions
                dist += (lig_coord[k] - protein_coords[j, k]) ** 2
            if dist < clash_threshold * clash_threshold:
                return True
    return False

@jit(nopython=True, fastmath=True)
def fast_calculate_clash_score(ligand_coords, protein_coords, lig_radii, prot_radii, threshold=0.7):
    """
    Calculates clash score between ligand and protein atoms using JIT compilation.
    
    Parameters:
    -----------
    ligand_coords : numpy.ndarray
        Coordinates of ligand atoms (N x 3)
    protein_coords : numpy.ndarray
        Coordinates of protein atoms (M x 3)
    lig_radii : numpy.ndarray
        VDW radii of ligand atoms (N)
    prot_radii : numpy.ndarray
        VDW radii of protein atoms (M)
    threshold : float
        Scaling factor for VDW radii overlap
        
    Returns:
    --------
    float
        Clash score (higher means more severe clashes)
    """
    clash_score = 0.0
    clashing_atoms = 0
    
    for i in range(ligand_coords.shape[0]):
        for j in range(protein_coords.shape[0]):
            # Calculate distance between atoms
            dx = ligand_coords[i, 0] - protein_coords[j, 0]
            dy = ligand_coords[i, 1] - protein_coords[j, 1]
            dz = ligand_coords[i, 2] - protein_coords[j, 2]
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Calculate minimum allowed distance
            min_allowed = (lig_radii[i] + prot_radii[j]) * threshold
            
            # Check for clash and add to score if found
            if distance < min_allowed:
                overlap = (min_allowed - distance) / min_allowed
                clash_score += overlap * overlap  # Square to penalize severe clashes more
                clashing_atoms += 1
                
                if clashing_atoms >= 3:  # Early exit for multiple clashes
                    return clash_score * 10
    
    return clash_score

# ------------------------------------------------------------------------------
# Common Utility Mixin Classes
# ------------------------------------------------------------------------------

class GridUtilsMixin:
    """Mixin class providing grid-related utility methods for search algorithms."""
    
    def initialize_grid_points(self, center, protein=None, spacing=None, radius=None, force_rebuild=False):
        """
        Initialize grid points for search space sampling.
        
        Parameters:
        -----------
        center : array-like
            Center coordinates
        protein : Protein, optional
            Protein object for pocket detection
        spacing : float, optional
            Grid spacing (overrides self.grid_spacing)
        radius : float, optional
            Grid radius (overrides self.grid_radius)
        force_rebuild : bool
            Whether to force rebuilding the grid even if it already exists
        """
        if spacing is None:
            spacing = getattr(self, 'grid_spacing', 0.375)
        if radius is None:
            radius = getattr(self, 'grid_radius', 10.0)
            
        if force_rebuild or not hasattr(self, 'grid_points') or self.grid_points is None:
            self.grid_points = []

            pocket_centers = []

            if protein is not None and hasattr(protein, 'detect_pockets'):
                try:
                    pockets = protein.detect_pockets()
                    if pockets:
                        if hasattr(self, 'logger'):
                            self.logger.info(f"[BLIND] Detected {len(pockets)} binding pockets")
                        else:
                            print(f"[BLIND] Detected {len(pockets)} binding pockets")
                        pocket_centers = [p['center'] for p in pockets]
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"Error detecting pockets: {e}")
                    else:
                        print(f"Warning: Error detecting pockets: {e}")

            if hasattr(self, 'grid_center') and self.grid_center is not None:
                pocket_centers = [self.grid_center]
            elif not pocket_centers:
                pocket_centers = [center]

            for idx, c in enumerate(pocket_centers):
                local_grid = generate_spherical_grid(
                    center=c,
                    radius=radius,
                    spacing=spacing
                )
                self.grid_points.extend(local_grid)
                message = f"  -> Grid {idx+1}: {len(local_grid)} points at {c}"
                if hasattr(self, 'logger'):
                    self.logger.info(message)
                else:
                    print(message)

            message = (f"Initialized total grid with {len(self.grid_points)} points "
                      f"(spacing: {spacing}, radius: {radius})")
            if hasattr(self, 'logger'):
                self.logger.info(message)
            else:
                print(message)

            # Save Light Sphere PDB (subsample)
            subsample_rate = 20
            if hasattr(self, 'output_dir') and self.output_dir is not None:
                sphere_path = Path(self.output_dir) / "sphere.pdb"
                sphere_path.parent.mkdir(parents=True, exist_ok=True)
                with open(sphere_path, 'w') as f:
                    for idx, point in enumerate(self.grid_points):
                        if idx % subsample_rate == 0:
                            f.write(
                                f"HETATM{idx+1:5d} {'S':<2s}   SPH A   1    "
                                f"{point[0]:8.3f}{point[1]:8.3f}{point[2]:8.3f}  1.00  0.00          S\n"
                            )
                message = f"Sphere grid written to {sphere_path} (subsampled every {subsample_rate} points)"
                if hasattr(self, 'logger'):
                    self.logger.info(message)
                else:
                    print(message)
    
    def initialize_smart_grid(self, protein, center, radius, spacing=0.5, margin=0.7):
        """
        Create a smart grid that avoids protein atoms, especially backbone.
        
        Parameters:
            protein: Protein object
            center: 3D center coordinates
            radius: Grid radius
            spacing: Grid spacing
            margin: Safety margin factor for protein atoms
        
        Returns:
            array of valid grid points
        """
        # Get protein atoms
        if hasattr(protein, 'active_site') and protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Generate dense spherical grid
        all_grid_points = generate_spherical_grid(center, radius, spacing)
        
        # Create KD-tree for efficient proximity search
        protein_coords = np.array([atom['coords'] for atom in protein_atoms])
        protein_kdtree = cKDTree(protein_coords)
        
        # Get VDW radii
        vdw_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }
        
        # Find max protein atom radius
        max_radius = 0.0
        for atom in protein_atoms:
            symbol = atom.get('element', atom.get('name', 'C'))[0]
            max_radius = max(max_radius, vdw_radii.get(symbol, 1.7))
        
        # Safety distance = max protein atom radius + margin
        safety_distance = max_radius * margin
        
        # Filter out grid points too close to protein atoms
        valid_grid_points = []
        for point in all_grid_points:
            # Find protein atoms within safety distance
            indices = protein_kdtree.query_ball_point(point, safety_distance)
            
            # If no close protein atoms, consider point valid
            if not indices:
                valid_grid_points.append(point)
        
        message = f"Generated {len(valid_grid_points)} valid grid points out of {len(all_grid_points)} total"
        if hasattr(self, 'logger'):
            self.logger.info(message)
        else:
            print(message)
        
        # If too few valid points, expand grid or relax constraints
        if len(valid_grid_points) < 100:
            message = "Warning: Few valid grid points. Expanding search..."
            if hasattr(self, 'logger'):
                self.logger.warning(message)
            else:
                print(message)
            return self.initialize_smart_grid(protein, center, radius * 1.2, spacing, margin * 0.9)
        
        return np.array(valid_grid_points)


class ClashDetectionMixin:
    """Mixin class providing clash detection and resolution methods."""
    
    def get_vdw_radii(self, atoms):
        """
        Get van der Waals radii for atoms.
        
        Parameters:
        -----------
        atoms : list
            List of atom dictionaries
            
        Returns:
        --------
        numpy.ndarray
            Array of VDW radii
        """
        # Default VDW radii for common elements
        vdw_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }
        
        radii = np.zeros(len(atoms))
        for i, atom in enumerate(atoms):
            # Get atom symbol (first character of element or name)
            symbol = atom.get('element', atom.get('symbol', atom.get('name', 'C')))
            if isinstance(symbol, str) and len(symbol) > 0:
                symbol = symbol[0]
            else:
                symbol = 'C'  # Default to carbon
                
            # Look up radius, default to carbon if not found
            radii[i] = vdw_radii.get(symbol, 1.7)
            
        return radii
    
    def _enhanced_clash_detection(self, protein, ligand, clash_threshold=0.7, backbone_threshold=0.75):
        """
        More sophisticated clash detection with special handling for backbone atoms.
        
        Parameters:
            protein: Protein object with atoms
            ligand: Ligand object with atoms
            clash_threshold: Factor to determine clash distance for non-backbone atoms
            backbone_threshold: Stricter factor for backbone atoms
            
        Returns:
            (bool, float): (is_clashing, clash_score) tuple
        """
        # Get protein atoms in active site
        if hasattr(protein, 'active_site') and protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Extract coordinates and convert to numpy arrays
        ligand_coords = np.array([atom['coords'] for atom in ligand.atoms])
        protein_coords = np.array([atom['coords'] for atom in protein_atoms])
        
        # Get VDW radii
        ligand_radii = self.get_vdw_radii(ligand.atoms)
        protein_radii = self.get_vdw_radii(protein_atoms)
        
        clash_score = 0.0
        max_overlap = 0.0
        clashing_atoms = 0
        
        # Check each ligand atom against all protein atoms
        for i, lig_atom in enumerate(ligand.atoms):
            l_coords = lig_atom['coords']
            l_radius = ligand_radii[i]
            
            for j, p_atom in enumerate(protein_atoms):
                p_coords = p_atom['coords']
                p_name = p_atom.get('name', '')
                p_radius = protein_radii[j]
                
                # Check if protein atom is backbone (CA, C, N, O)
                is_backbone = p_name in ['CA', 'C', 'N', 'O'] and p_name.strip() == p_name
                
                # Use stricter threshold for backbone atoms
                threshold = backbone_threshold if is_backbone else clash_threshold
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Minimum allowed distance based on vdW radii
                min_allowed = (p_radius + l_radius) * threshold
                
                # Check for clash
                if distance < min_allowed:
                    overlap = (min_allowed - distance) / min_allowed
                    clash_score += overlap ** 2  # Square to emphasize severe clashes
                    max_overlap = max(max_overlap, overlap)
                    clashing_atoms += 1
                    
                    # Early exit for severe clashes, especially with backbone
                    if is_backbone and overlap > 0.4:  # 40% overlap with backbone is severe
                        return True, 999.0
                    
                    if clashing_atoms >= 3:  # Multiple clashes indicate poor pose
                        return True, clash_score * 10
        
        # Return clash status and score
        return (clash_score > 0.5 or max_overlap > 0.5), clash_score
    
    def _check_pose_validity(self, ligand, protein, clash_threshold=1.5, center=None, radius=None):
        """
        Check if ligand pose clashes with protein atoms and is within sphere.
        
        Parameters:
            ligand: Ligand object with .atoms
            protein: Protein object with .atoms or active_site['atoms']
            clash_threshold: Ångström cutoff for hard clash
            center: Center of the search sphere (optional)
            radius: Radius of the search sphere (optional)
            
        Returns:
            bool: True if pose is valid (no severe clash and within sphere), False otherwise
        """
        # Check if ligand is within sphere (if center and radius provided)
        if center is not None and radius is not None:
            if not is_inside_sphere(ligand, center, radius):
                return False
        
        # Continue with existing clash detection
        ligand_coords = np.array([atom['coords'] for atom in ligand.atoms])
        
        # Use active site atoms if defined
        if hasattr(protein, 'active_site') and protein.active_site and 'atoms' in protein.active_site:
            protein_coords = np.array([atom['coords'] for atom in protein.active_site['atoms']])
        else:
            protein_coords = np.array([atom['coords'] for atom in protein.atoms])
        
        # Use fast clash detection if numba is available
        if HAS_NUMBA:
            if fast_clash_detection(ligand_coords, protein_coords, clash_threshold):
                return False
        else:
            # Fallback to non-JIT version
            for lig_coord in ligand_coords:
                distances = np.linalg.norm(protein_coords - lig_coord, axis=1)
                if np.any(distances < clash_threshold):
                    return False  # Clash detected
        
        # Enhanced clash detection for more subtle clashes
        is_clashing, _ = self._enhanced_clash_detection(protein, ligand)
        return not is_clashing
    
    def _gentle_clash_relief(self, protein, pose, reference=None, max_steps=20, max_movement=0.3, radius=None, center=None):
        """
        Gently move atoms to relieve clashes while preserving overall pose.

        Parameters:
            protein: Protein object
            pose: Ligand pose
            reference: Reference pose (optional)
            max_steps: Maximum optimization steps
            max_movement: Maximum allowed movement
            radius: Sphere radius
            center: Sphere center

        Returns:
            Improved ligand pose
        """
        import copy
        from scipy.spatial.transform import Rotation

        # Fallback center and radius from protein
        if center is None and hasattr(protein, 'active_site'):
            center = protein.active_site['center']
        if radius is None and hasattr(protein, 'active_site'):
            radius = protein.active_site['radius']

        # Make work copy
        working_pose = copy.deepcopy(pose)
        
        try:
            current_score = self.scoring_function.score(protein, working_pose)
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"Error scoring pose during clash relief: {e}")
            else:
                print(f"Warning: Error scoring pose during clash relief: {e}")
            current_score = float('inf')
            
        best_pose = copy.deepcopy(working_pose)
        best_score = current_score

        is_clashing, _ = self._enhanced_clash_detection(protein, working_pose)
        if not is_clashing and center is not None and radius is not None:
            return enforce_sphere_boundary(working_pose, center, radius)

        protein_atoms = protein.active_site['atoms'] if hasattr(protein, 'active_site') and 'atoms' in protein.active_site else protein.atoms
        protein_coords = np.array([atom['coords'] for atom in protein_atoms])

        for step in range(max_steps):
            improved = False

            # Try small translations
            for direction in [
                [0.1, 0, 0], [-0.1, 0, 0],
                [0, 0.1, 0], [0, -0.1, 0],
                [0, 0, 0.1], [0, 0, -0.1],
                [0.07, 0.07, 0.07], [-0.07, -0.07, -0.07]
            ]:
                test_pose = copy.deepcopy(working_pose)
                test_pose.translate(np.array(direction))
                
                if center is not None and radius is not None:
                    test_pose = enforce_sphere_boundary(test_pose, center, radius)

                is_clashing, clash_score = self._enhanced_clash_detection(protein, test_pose)
                if not is_clashing:
                    try:
                        test_score = self.scoring_function.score(protein, test_pose)
                        if test_score < best_score:
                            best_pose = copy.deepcopy(test_pose)
                            best_score = test_score
                            improved = True
                            break
                    except Exception as e:
                        # Skip this pose if scoring fails
                        if hasattr(self, 'logger'):
                            self.logger.debug(f"Error scoring test pose: {e}")
                        continue

            if improved:
                working_pose = copy.deepcopy(best_pose)
                continue

            # Try small rotations
            axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            angles = [0.05, -0.05, 0.1, -0.1]

            for axis in axes:
                for angle in angles:
                    test_pose = copy.deepcopy(working_pose)
                    rotation = Rotation.from_rotvec(np.array(axis) * angle)
                    centroid = np.mean(test_pose.xyz, axis=0)
                    test_pose.translate(-centroid)
                    test_pose.rotate(rotation.as_matrix())
                    test_pose.translate(centroid)
                    
                    if center is not None and radius is not None:
                        test_pose = enforce_sphere_boundary(test_pose, center, radius)

                    is_clashing, clash_score = self._enhanced_clash_detection(protein, test_pose)
                    if not is_clashing:
                        try:
                            test_score = self.scoring_function.score(protein, test_pose)
                            if test_score < best_score:
                                best_pose = copy.deepcopy(test_pose)
                                best_score = test_score
                                improved = True
                                break
                        except Exception:
                            continue
                if improved:
                    break

            if not improved and step > max_steps // 2:
                test_pose = self._adjust_clashing_atoms(protein, working_pose)
                if center is not None and radius is not None:
                    test_pose = enforce_sphere_boundary(test_pose, center, radius)
                is_clashing, clash_score = self._enhanced_clash_detection(protein, test_pose)
                if not is_clashing or clash_score < 0.1:
                    try:
                        test_score = self.scoring_function.score(protein, test_pose)
                        if test_score < best_score * 1.5:
                            best_pose = copy.deepcopy(test_pose)
                            best_score = test_score
                            improved = True
                    except Exception:
                        pass

            if improved:
                working_pose = copy.deepcopy(best_pose)
            else:
                test_pose = copy.deepcopy(working_pose)
                protein_center = np.mean(protein_coords, axis=0)
                ligand_center = np.mean(test_pose.xyz, axis=0)
                away_vector = ligand_center - protein_center
                if np.linalg.norm(away_vector) > 0:
                    away_vector = away_vector / np.linalg.norm(away_vector)
                    test_pose.translate(away_vector * 0.2)
                    if center is not None and radius is not None:
                        test_pose = enforce_sphere_boundary(test_pose, center, radius)

                    is_clashing, clash_score = self._enhanced_clash_detection(protein, test_pose)
                    if not is_clashing or clash_score < 0.1:
                        working_pose = copy.deepcopy(test_pose)

        if center is not None and radius is not None:
            best_pose = enforce_sphere_boundary(best_pose, center, radius)
        return best_pose
    
    def _adjust_clashing_atoms(self, protein, pose, max_adjustment=0.3, clash_threshold=1.5):
        """
        Adjust atomic positions in a pose to relieve steric clashes with protein.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        pose : Ligand
            Ligand pose to adjust
        max_adjustment : float
            Maximum distance to move atoms (Angstroms)
        clash_threshold : float
            Threshold for identifying clashes (Angstroms)
        
        Returns:
        --------
        Ligand
            Adjusted pose with reduced clashes
        """
        import copy
        
        # Make a copy of the pose to avoid modifying the original
        adjusted_pose = copy.deepcopy(pose)
        
        # Identify clashing atoms
        clashing_atoms = []
        
        # Get VDW radii
        vdw_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }
        
        # Use active site atoms if defined
        if hasattr(protein, 'active_site') and protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Find clashing atoms and their clash severity
        for lig_idx, lig_atom in enumerate(adjusted_pose.atoms):
            lig_coords = lig_atom['coords']
            lig_symbol = lig_atom.get('symbol', 'C')
            lig_radius = vdw_radii.get(lig_symbol, 1.7)
            
            for prot_atom in protein_atoms:
                prot_coords = prot_atom['coords']
                prot_symbol = prot_atom.get('element', prot_atom.get('name', 'C'))[0]
                prot_radius = vdw_radii.get(prot_symbol, 1.7)
                
                # Calculate distance and minimum allowed distance
                distance = np.linalg.norm(lig_coords - prot_coords)
                min_allowed = (lig_radius + prot_radius) * 0.7  # 70% of sum of vdW radii
                
                if distance < min_allowed:
                    # Calculate clash severity
                    overlap = min_allowed - distance
                    clashing_atoms.append((lig_idx, prot_coords, overlap))
                    break  # Found a clash for this atom, move to the next
        
        if not clashing_atoms:
            return adjusted_pose  # No clashes to fix
        
        # Sort clashing atoms by overlap severity (most severe first)
        clashing_atoms.sort(key=lambda x: x[2], reverse=True)
        
        # Adjust each clashing atom
        for atom_idx, prot_coords, overlap in clashing_atoms:
            atom = adjusted_pose.atoms[atom_idx]
            atom_coords = atom['coords']
            
            # Calculate vector away from the clashing protein atom
            vector = atom_coords - prot_coords
            if np.linalg.norm(vector) < 0.1:  # Avoid division by zero
                # If atoms are almost overlapping, use a random direction
                vector = np.random.randn(3)
            
            # Normalize direction vector
            direction = vector / np.linalg.norm(vector)
            
            # Calculate adjustment - more severe clashes get moved further
            adjustment_magnitude = min(max_adjustment, overlap * 1.2)  # Move a bit more than the overlap
            adjustment = direction * adjustment_magnitude
            
            # Update atom coordinates
            new_coords = atom_coords + adjustment
            adjusted_pose.atoms[atom_idx]['coords'] = new_coords
            adjusted_pose.xyz[atom_idx] = new_coords
        
        return adjusted_pose
    
    def _generate_minimal_clash_pose(self, protein, ligand, center, radius, max_attempts=20):
        """
        Generate a pose with minimal steric clashes within the specified radius.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand template to position
        center : numpy.ndarray
            Center coordinates of the search space
        radius : float
            Radius of the search space
        max_attempts : int
            Maximum number of generation attempts
        
        Returns:
        --------
        Ligand
            Ligand pose with minimal clashes
        """
        import copy
        from scipy.spatial.transform import Rotation
        import random
        
        best_pose = None
        best_clash_score = float('inf')
        
        # Helper function to calculate clash score
        def calculate_clash_score(pose):
            _, score = self._enhanced_clash_detection(protein, pose)
            return score
        
        # Try multiple poses to find one with minimal clashes
        for attempt in range(max_attempts):
            # Create a new pose
            pose = copy.deepcopy(ligand)
            
            # Generate a random point in the sphere
            r = radius * random.random() ** (1/3)  # For uniform sampling within sphere volume
            theta = random.uniform(0, 2 * np.pi)
            phi = random.uniform(0, np.pi)
            
            x = center[0] + r * np.sin(phi) * np.cos(theta)
            y = center[1] + r * np.sin(phi) * np.sin(theta)
            z = center[2] + r * np.cos(phi)
            
            # Move ligand centroid to this point
            centroid = np.mean(pose.xyz, axis=0)
            translation = np.array([x, y, z]) - centroid
            pose.translate(translation)
            
            # Apply random rotation
            rotation = Rotation.random()
            rotation_matrix = rotation.as_matrix()
            centroid = np.mean(pose.xyz, axis=0)
            pose.translate(-centroid)
            pose.rotate(rotation_matrix)
            pose.translate(centroid)
            
            # Calculate clash score
            clash_score = calculate_clash_score(pose)
            
            # Update best pose if this one has fewer clashes
            if clash_score < best_clash_score:
                best_pose = copy.deepcopy(pose)
                best_clash_score = clash_score
                
                # If we found a pose with no clashes, return it immediately
                if clash_score == 0.0:
                    return best_pose
            
            # Progress reporting for long runs
            if attempt > 0 and attempt % 5 == 0:
                if hasattr(self, 'logger'):
                    self.logger.debug(f"  Generated {attempt}/{max_attempts} poses, best clash score: {best_clash_score:.2f}")
                else:
                    print(f"  Generated {attempt}/{max_attempts} poses, best clash score: {best_clash_score:.2f}")
        
        # If we found a pose with acceptable clash score, try to improve it with gentle adjustments
        if best_pose is not None and hasattr(self, '_adjust_clashing_atoms'):
            if hasattr(self, 'logger'):
                self.logger.info(f"  Found pose with clash score {best_clash_score:.2f}, applying adjustments...")
            else:
                print(f"  Found pose with clash score {best_clash_score:.2f}, applying adjustments...")
            best_pose = self._adjust_clashing_atoms(protein, best_pose)
        
        message = f"  Generated pose with final clash score: {calculate_clash_score(best_pose):.2f}"
        if hasattr(self, 'logger'):
            self.logger.info(message)
        else:
            print(message)
        return best_pose


class PoseGenerationMixin:
    """Mixin class providing methods for generating valid poses."""
    
    def _generate_valid_pose(self, protein, ligand, center, radius, max_attempts=50):
        """
        Generate a valid ligand pose within the specified radius that doesn't have severe clashes.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand template to position
        center : numpy.ndarray
            Center coordinates of the search space
        radius : float
            Radius of the search space
        max_attempts : int
            Maximum number of generation attempts
        
        Returns:
        --------
        Ligand
            Valid ligand pose
        """
        import copy
        import numpy as np
        from scipy.spatial.transform import Rotation
        import random
        
        # First try a simple approach by generating random poses
        for attempt in range(max_attempts):
            # Create a new pose
            pose = copy.deepcopy(ligand)
            
            # Sample from the grid points if available
            if hasattr(self, 'grid_points') and self.grid_points is not None and len(self.grid_points) > 0:
                point = random.choice(self.grid_points)
                centroid = np.mean(pose.xyz, axis=0)
                translation = point - centroid
                pose.translate(translation)
            else:
                # Generate a random point in the sphere if no grid
                r = radius * random.random() ** (1/3)
                theta = random.uniform(0, 2 * np.pi)
                phi = random.uniform(0, np.pi)
                
                x = center[0] + r * np.sin(phi) * np.cos(theta)
                y = center[1] + r * np.sin(phi) * np.sin(theta)
                z = center[2] + r * np.cos(phi)
                
                # Move ligand centroid to this point
                centroid = np.mean(pose.xyz, axis=0)
                translation = np.array([x, y, z]) - centroid
                pose.translate(translation)
            
            # Apply random rotation
            rotation = Rotation.random()
            rotation_matrix = rotation.as_matrix()
            centroid = np.mean(pose.xyz, axis=0)
            pose.translate(-centroid)
            pose.rotate(rotation_matrix)
            pose.translate(centroid)
            
            # Check if the pose is valid (no severe clashes)
            if self._check_pose_validity(pose, protein):
                # If we have clash relief capability, apply it to improve the pose
                if hasattr(self, '_gentle_clash_relief'):
                    try:
                        relaxed_pose = self._gentle_clash_relief(protein, pose, max_steps=5, max_movement=0.3)
                        return relaxed_pose
                    except Exception as e:
                        if hasattr(self, 'logger'):
                            self.logger.warning(f"Warning: Error during clash relief: {e}")
                        else:
                            print(f"Warning: Error during clash relief: {e}")
                        return pose
                else:
                    return pose
        
        # If we couldn't generate a valid pose with random sampling, 
        # try using a more specialized method
        message = "  Random pose generation failed, trying specialized minimal clash generation..."
        if hasattr(self, 'logger'):
            self.logger.info(message)
        else:
            print(message)
        return self._generate_minimal_clash_pose(protein, ligand, center, radius)
    
    def _generate_random_pose(self, ligand, center, radius):
        """
        Generate a random ligand pose within the specified radius.
        
        Parameters:
        -----------
        ligand : Ligand
            Ligand template to position
        center : numpy.ndarray
            Center coordinates of the search space
        radius : float
            Radius of the search space
            
        Returns:
        --------
        Ligand
            Random ligand pose
        """
        # More uniform sampling within sphere volume
        r = radius * (0.8 + 0.2 * random.random())  # Bias toward outer region
        theta = random.uniform(0, 2 * np.pi)
        phi = random.uniform(0, np.pi)

        x = center[0] + r * np.sin(phi) * np.cos(theta)
        y = center[1] + r * np.sin(phi) * np.sin(theta)
        z = center[2] + r * np.cos(phi)

        pose = copy.deepcopy(ligand)
        centroid = np.mean(pose.xyz, axis=0)
        
        translation = np.array([x, y, z]) - centroid
        pose.translate(translation)

        # Apply random rotation
        rotation = Rotation.random()
        rotation_matrix = rotation.as_matrix()
        centroid = np.mean(pose.xyz, axis=0)
        pose.translate(-centroid)
        pose.rotate(rotation_matrix)
        pose.translate(centroid)

        return pose

    def _validate_conformation(self, ligand):
        """
        Validate a ligand conformation to ensure no overlapping atoms or invalid bond lengths.
        
        Parameters:
        -----------
        ligand : Ligand
            Ligand to validate
        
        Returns:
        --------
        bool
            True if the conformation is valid, False otherwise
        """
        # Check for overlapping atoms
        for i in range(len(ligand.xyz)):
            for j in range(i + 1, len(ligand.xyz)):
                distance = np.linalg.norm(ligand.xyz[i] - ligand.xyz[j])
                if distance < 1.2:  # Adjusted threshold for atom overlap (in Å)
                    return False
        
        # Check for valid bond lengths
        for bond in ligand.bonds:
            atom1 = ligand.xyz[bond['begin_atom_idx']]
            atom2 = ligand.xyz[bond['end_atom_idx']]
            bond_length = np.linalg.norm(atom1 - atom2)
            if bond_length < 0.9 or bond_length > 2.0:  # Adjusted bond length range (in Å)
                return False
        
        return True
    
    def _repair_conformation(self, ligand, max_attempts=5):
        """
        Attempt to repair an invalid ligand conformation.

        Parameters:
        -----------
        ligand : Ligand
            Ligand to repair
        max_attempts : int
            Maximum number of repair attempts

        Returns:
        --------
        Ligand
            Repaired ligand or a new random pose if repair fails
        """
        
        for attempt in range(max_attempts):
            # Apply small random perturbations to atom positions
            perturbation = np.random.normal(0, 0.2, ligand.xyz.shape)  # 0.2 Å standard deviation
            ligand.xyz += perturbation
            
            # Revalidate after perturbation
            if self._validate_conformation(ligand):
                return ligand
            
            # Attempt to resolve steric clashes by energy minimization
            try:
                ligand = self._minimize_energy(ligand, max_iterations=200)
                if self._validate_conformation(ligand):
                    return ligand
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.warning(f"Energy minimization failed: {e}")
                else:
                    print(f"Energy minimization failed: {e}")
        
        # If repair fails, generate a new random pose
        message = "Repair failed after maximum attempts. Generating a new random pose..."
        if hasattr(self, 'logger'):
            self.logger.warning(message)
        else:
            print(message)
        
        # Generate new random coordinates
        pose = copy.deepcopy(ligand)
        
        # Apply random rotation
        rotation = Rotation.random()
        centroid = np.mean(pose.xyz, axis=0)
        pose.translate(-centroid)
        pose.rotate(rotation.as_matrix())
        pose.translate(centroid)
        
        # Apply small random translation
        pose.translate(np.random.normal(0, 0.5, 3))
        
        return pose
    
    def _minimize_energy(self, ligand, max_iterations=100):
        """
        Perform energy minimization to resolve steric clashes and optimize ligand geometry.

        Parameters:
        -----------
        ligand : Ligand
            Ligand to minimize
        max_iterations : int
            Maximum number of optimization iterations

        Returns:
        --------
        Ligand
            Minimized ligand
        """

        def energy_function(coords):
            # Example energy function: penalize overlapping atoms and bond length deviations
            coords = coords.reshape(ligand.xyz.shape)
            energy = 0.0
            
            # Penalize overlapping atoms
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    distance = np.linalg.norm(coords[i] - coords[j])
                    if distance < 1.2:  # Overlap threshold
                        energy += (1.2 - distance) ** 2
            
            # Penalize invalid bond lengths
            for bond in ligand.bonds:
                atom1 = coords[bond['begin_atom_idx']]
                atom2 = coords[bond['end_atom_idx']]
                bond_length = np.linalg.norm(atom1 - atom2)
                if bond_length < 0.9:
                    energy += (0.9 - bond_length) ** 2
                elif bond_length > 2.0:
                    energy += (bond_length - 2.0) ** 2
            
            return energy

        # Flatten coordinates for optimization
        initial_coords = ligand.xyz.flatten()
        
        # Use scipy's L-BFGS-B optimizer
        try:
            result = minimize(energy_function, initial_coords, method='L-BFGS-B', options={'maxiter': max_iterations})
            # Update ligand coordinates with minimized values
            ligand.xyz = result.x.reshape(ligand.xyz.shape)
            
            # Update atom coordinates in the atoms list as well
            for i, atom in enumerate(ligand.atoms):
                atom['coords'] = ligand.xyz[i]
                
            return ligand
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Minimization failed: {e}")
            else:
                print(f"Minimization failed: {e}")
            # Return original ligand if minimization fails
            return ligand


class OptimizationMixin:
    """Mixin class providing optimization methods for search algorithms."""
    
    def _local_optimization(self, pose, protein, step_size=0.3, angle_step=0.05, max_steps=15):
        """
        Perform local optimization of pose using gradient descent with clash detection.
        
        Parameters:
        -----------
        pose : Ligand
            Ligand pose to optimize
        protein : Protein
            Protein target
        step_size : float
            Maximum translation step size (Å)
        angle_step : float
            Maximum rotation step size (radians)
        max_steps : int
            Maximum number of optimization steps
        
        Returns:
        --------
        tuple
            (optimized_pose, optimized_score)
        """
        # Make working copy of pose
        working_pose = copy.deepcopy(pose)
        
        # Get initial score
        try:
            current_score = self.scoring_function.score(protein, working_pose)
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"Error in initial scoring during local optimization: {e}")
            else:
                print(f"Warning: Error in initial scoring during local optimization: {e}")
            return pose, float('inf')
        
        # Store best pose and score
        best_pose = copy.deepcopy(working_pose)
        best_score = current_score
        
        # Define possible directions for gradient search
        directions = [
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ]
        
        # Define rotation axes
        axes = [
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ]
        
        # Optimization loop
        for step in range(max_steps):
            improved = False
            
            # Try translations
            for direction in directions:
                # Scale direction vector to step size
                move_vector = np.array(direction) * step_size
                
                # Create test pose
                test_pose = copy.deepcopy(working_pose)
                test_pose.translate(move_vector)
                
                # Check if pose is valid (no clashes)
                if not self._check_pose_validity(test_pose, protein):
                    continue
                
                # Score the pose
                try:
                    test_score = self.scoring_function.score(protein, test_pose)
                except Exception:
                    continue
                
                # Update best pose if improved
                if test_score < best_score:
                    best_pose = copy.deepcopy(test_pose)
                    best_score = test_score
                    improved = True
            
            # Try rotations
            for axis in axes:
                # Create rotation vector
                rot_vector = np.array(axis) * angle_step
                
                # Create test pose
                test_pose = copy.deepcopy(working_pose)
                
                # Apply rotation around centroid
                centroid = np.mean(test_pose.xyz, axis=0)
                test_pose.translate(-centroid)
                
                # Create rotation matrix
                rotation = Rotation.from_rotvec(rot_vector)
                test_pose.rotate(rotation.as_matrix())
                
                # Translate back
                test_pose.translate(centroid)
                
                # Check if pose is valid (no clashes)
                if not self._check_pose_validity(test_pose, protein):
                    continue
                
                # Score the pose
                try:
                    test_score = self.scoring_function.score(protein, test_pose)
                except Exception:
                    continue
                
                # Update best pose if improved
                if test_score < best_score:
                    best_pose = copy.deepcopy(test_pose)
                    best_score = test_score
                    improved = True
            
            # Update working pose if improved
            if improved:
                working_pose = copy.deepcopy(best_pose)
            else:
                # Reduce step sizes if no improvement
                step_size *= 0.7
                angle_step *= 0.7
                
                # Early termination if step sizes are too small
                if step_size < 0.01 or angle_step < 0.005:
                    break
        
        # Return best pose and score
        return best_pose, best_score
    
    def _adjust_search_radius(self, initial_radius, iteration, total_iterations, decay_rate=0.5):
        """
        Shrink search radius over iterations.
        
        Parameters:
        -----------
        initial_radius : float
            Initial search radius
        iteration : int
            Current iteration
        total_iterations : int
            Total number of iterations
        decay_rate : float
            Rate of radius decay (0-1)
            
        Returns:
        --------
        float
            Adjusted radius for the current iteration
        """
        factor = 1.0 - (iteration / total_iterations) * decay_rate
        return max(initial_radius * factor, initial_radius * 0.5)


# ------------------------------------------------------------------------------
# Base Parallel Search Class
# ------------------------------------------------------------------------------
class ParallelSearch(DockingSearch, GridUtilsMixin, ClashDetectionMixin, PoseGenerationMixin, OptimizationMixin):
    """
    Base class for parallel search algorithms in molecular docking.
    
    This class provides common functionality for parallel search implementations
    and includes utility methods from the mixin classes.
    """
    
    def __init__(self, scoring_function, max_iterations=100, output_dir=None, 
                 grid_spacing=0.375, grid_radius=10.0, grid_center=None, args=None):
        """
        Initialize parallel search.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function to evaluate poses
        max_iterations : int
            Maximum number of iterations
        output_dir : str or Path
            Output directory for saving results
        grid_spacing : float
            Spacing between grid points
        grid_radius : float
            Radius of the search sphere
        grid_center : array-like
            Center coordinates of the search sphere
        args : argparse.Namespace
            Command-line arguments
        """
        super().__init__(scoring_function, max_iterations)
        self.output_dir = output_dir
        self.grid_spacing = grid_spacing
        self.grid_radius = grid_radius
        self.grid_center = grid_center
        self.grid_points = None
        self.args = args

        # Setup logging
        if output_dir:
            self.logger = setup_logging(output_dir)
        else:
            import logging
            self.logger = logging.getLogger("null_logger")
            self.logger.addHandler(logging.NullHandler())
    
    def search(self, protein, ligand):
        """
        Perform docking search in parallel.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
            
        Returns:
        --------
        list
            List of (pose, score) tuples, sorted by score
        """
        print("\n🔍 Performing docking (parallel mode enabled)...\n")
        return self.improve_rigid_docking(protein, ligand, self.args)

    def improve_rigid_docking(self, protein, ligand, args):
        """
        Improve rigid docking by finding optimal ligand poses.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
        args : argparse.Namespace
            Command-line arguments
            
        Returns:
        --------
        list
            List of (pose, score) tuples, sorted by score
        """
        # Skip redundant pocket detection if already performed
        if hasattr(protein, '_pockets_detected') and protein._pockets_detected:
            print("Using previously detected pockets")
        else:
            if not protein.active_site:
                if hasattr(args, 'site') and args.site:
                    radius = max(getattr(args, 'radius', 15.0), 12.0)
                    protein.define_active_site(args.site, radius)
                elif hasattr(args, 'detect_pockets') and args.detect_pockets:
                    try:
                        pockets = protein.detect_pockets()
                        if pockets:
                            radius = max(pockets[0]['radius'], 15.0)
                            protein.define_active_site(pockets[0]['center'], radius)
                        else:
                            center = np.mean(protein.xyz, axis=0)
                            protein.define_active_site(center, 15.0)
                    except Exception as e:
                        print(f"[WARNING] Error detecting pockets: {e}")
                        center = np.mean(protein.xyz, axis=0)
                        protein.define_active_site(center, 15.0)
                else:
                    center = np.mean(protein.xyz, axis=0)
                    protein.define_active_site(center, 15.0)
            protein._pockets_detected = True  # Mark as detected now

        center = protein.active_site['center']
        radius = protein.active_site['radius']

        # Initialize grid points for sampling
        self.initialize_grid_points(center, protein, radius=radius)

        n_initial_random = min(self.max_iterations // 2, 2000)
        poses = []
        max_attempts = n_initial_random * 5
        attempts = 0

        progress_bar = None
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=n_initial_random, desc="Generating initial poses")
        except ImportError:
            pass

        while len(poses) < n_initial_random and attempts < max_attempts:
            pose = copy.deepcopy(ligand)
            r = radius * (0.8 + 0.2 * random.random())
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            x = center[0] + r * np.sin(phi) * np.cos(theta)
            y = center[1] + r * np.sin(phi) * np.sin(theta)
            z = center[2] + r * np.cos(phi)

            translation = np.array([x, y, z]) - np.mean(pose.xyz, axis=0)
            pose.translate(translation)

            rot = Rotation.random()
            centroid = np.mean(pose.xyz, axis=0)
            pose.translate(-centroid)
            pose.rotate(rot.as_matrix())
            pose.translate(centroid)

            # Optional gentle outward nudge
            pose.translate(np.random.normal(0.2, 0.1, size=3))

            if not is_inside_sphere(pose, center, radius):
                # Calculate centroid-to-center vector
                centroid = np.mean(pose.xyz, axis=0)
                to_center = center - centroid
                # Scale vector to move back to sphere boundary
                dist = np.linalg.norm(to_center)
                if dist > 0:
                    move_vector = to_center * (dist - radius*0.9)/dist
                    pose.translate(move_vector)

            if detect_steric_clash(protein.atoms, pose.atoms):
                attempts += 1
                continue

            if not is_within_grid(pose, center, radius):
                attempts += 1
                continue

            poses.append(pose)
            attempts += 1
            
            if progress_bar:
                progress_bar.update(1)

        if progress_bar:
            progress_bar.close()

        print(f"Generated {len(poses)} poses (from {attempts} attempts). Now scoring...")

        def score_pose(pose):
            if detect_steric_clash(protein.atoms, pose.atoms):
                return (pose, float('inf'))
            try:
                return (pose, self.scoring_function.score(protein, pose))
            except Exception as e:
                print(f"[WARNING] Error scoring pose: {e}")
                return (pose, float('inf'))

        # Use parallel processing for scoring
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(score_pose, poses)

        results = [r for r in results if np.isfinite(r[1])]
        if not results:
            print("[ERROR] All poses clashed or failed scoring.")
            return []

        # Sort results by score (ascending)
        results.sort(key=lambda x: x[1])
        print(f"Best docking score: {results[0][1]:.2f}")

        # Apply local optimization if requested
        if getattr(args, 'local_opt', False):
            print("Applying local optimization to top poses...")
            top = results[:10]
            optimized = []
            
            progress_bar = None
            try:
                from tqdm import tqdm
                progress_bar = tqdm(total=len(top), desc="Optimizing poses")
            except ImportError:
                pass
                
            for pose, _ in top:
                try:
                    opt_pose, opt_score = self._local_optimization(pose, protein)
                    optimized.append((opt_pose, opt_score))
                except Exception as e:
                    print(f"[WARNING] Error during local optimization: {e}")
                    optimized.append((pose, _))
                
                if progress_bar:
                    progress_bar.update(1)
            
            if progress_bar:
                progress_bar.close()
                
            optimized.sort(key=lambda x: x[1])
            return optimized

        return results


# ------------------------------------------------------------------------------
# Parallel Genetic Algorithm
# ------------------------------------------------------------------------------

class ParallelGeneticAlgorithm(GeneticAlgorithm, GridUtilsMixin, ClashDetectionMixin, PoseGenerationMixin, OptimizationMixin):
    """
    Parallel implementation of genetic algorithm for molecular docking.
    
    This class extends the GeneticAlgorithm with parallel processing capabilities
    to improve performance on multi-core systems.
    """
    
    def __init__(self, scoring_function, max_iterations=100, population_size=50, 
                 mutation_rate=0.3, crossover_rate=0.8, tournament_size=3, 
                 n_processes=None, batch_size=None, process_pool=None, 
                 output_dir=None, perform_local_opt=False, grid_spacing=0.375, 
                 grid_radius=10.0, grid_center=None, logger=None):
        """
        Initialize parallel genetic algorithm.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function to evaluate poses
        max_iterations : int
            Maximum number of generations
        population_size : int
            Size of the population
        mutation_rate : float
            Probability of mutation (0.0 to 1.0)
        crossover_rate : float
            Probability of crossover (0.0 to 1.0)
        tournament_size : int
            Number of individuals in tournament selection
        n_processes : int
            Number of parallel processes (None = use all available cores)
        batch_size : int
            Number of poses to evaluate in each batch
        process_pool : multiprocessing.Pool
            Process pool to use (None = create a new pool)
        output_dir : str or Path
            Output directory
        perform_local_opt : bool
            Whether to perform local optimization
        grid_spacing : float
            Spacing between grid points
        grid_radius : float
            Radius of the search sphere
        grid_center : array-like
            Center coordinates of the search sphere
        logger : logging.Logger
            Logger instance
        """
        super().__init__(scoring_function, max_iterations, population_size, mutation_rate)
        self.scoring_function = scoring_function  # Ensure this is set
        self.output_dir = output_dir
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.perform_local_opt = perform_local_opt
        self.grid_spacing = grid_spacing  
        self.grid_radius = grid_radius  
        self.grid_center = np.array(grid_center) if grid_center is not None else np.array([0.0, 0.0, 0.0]) 
        self.logger = logger or logging.getLogger(__name__) 
        self.grid_points = None

        # Setup parallel processing
        if n_processes is None:
            self.n_processes = mp.cpu_count()
        else:
            self.n_processes = n_processes

        if batch_size is None:
            self.batch_size = max(1, self.population_size // (self.n_processes * 2))
        else:
            self.batch_size = batch_size

        self.process_pool = process_pool
        self.own_pool = False

        # Performance tracking
        self.eval_time = 0.0
        self.total_time = 0.0
        self.best_score = float('inf')
        self.best_pose = None
    
    def initialize_population(self, protein, ligand):
        """
        Initialize random population for genetic algorithm within spherical grid.
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand object
        Returns:
        --------
        list
            List of (pose, score) tuples
        """
        population = []
        
        # Determine search space
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            center = np.mean(protein.xyz, axis=0)
            radius = 15.0  # Arbitrary default

        self.initialize_grid_points(center, protein=protein)
        
        print(f"Using {self.n_processes} CPU cores for evaluation")
        print(f"Using {self.batch_size} poses per process for evaluation")
        print(f"Using {self.population_size} poses in total")
        print(f"Using {self.mutation_rate} mutation rate")
        print(f"Using {self.crossover_rate} crossover rate")
        print(f"Using {self.tournament_size} tournament size")
        print(f"Performing local optimization: {self.perform_local_opt}")
        print(f"Grid spacing: {self.grid_spacing}")
        print(f"Grid radius: {self.grid_radius}")
        
        # Add retry logic here
        attempts = 0
        max_attempts = self.population_size * 10
        
        # Initialize progress bar
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=self.population_size, desc="Initializing population")
        except ImportError:
            progress_bar = None
        
        # Generate initial population with retry logic
        while len(population) < self.population_size and attempts < max_attempts:
            attempts += 1
            
            pose = copy.deepcopy(ligand)
            
            # Select a random point from precomputed spherical grid
            random_grid_point = random.choice(self.grid_points)
            
            # Move the ligand centroid to that random point
            centroid = np.mean(pose.xyz, axis=0)
            translation = random_grid_point - centroid
            pose.translate(translation)
            
            # Apply random rotation with bias toward center of pocket
            rotation = Rotation.random()
            centroid = np.mean(pose.xyz, axis=0)
            vector_to_center = center - centroid
            vector_to_center /= np.linalg.norm(vector_to_center)
            
            # Small rotation (~10 degrees) toward pocket center
            bias_rotation = Rotation.from_rotvec(0.2 * vector_to_center)  # 0.2 rad ≈ 11 degrees
            biased_rotation = rotation * bias_rotation
            rotation_matrix = biased_rotation.as_matrix()
            
            # Apply rotation
            pose.translate(-centroid)
            pose.rotate(rotation_matrix)
            pose.translate(centroid)
            
            # Add random translation
            translation_vector = np.random.normal(1.5, 0.5, size=3)
            pose.translate(translation_vector)
            
            # Ensure pose is inside sphere
            if not is_inside_sphere(pose, center, radius):
                centroid = np.mean(pose.xyz, axis=0)
                to_center = center - centroid
                dist = np.linalg.norm(to_center)
                if dist > 0:
                    move_vector = to_center * (dist - radius*0.9)/dist
                    pose.translate(move_vector)
                    
            # Filters for valid poses
            if not is_within_grid(pose, center, radius):
                if attempts % 100 == 0:
                    self.logger.debug(f"Attempt {attempts}: Pose is outside grid, trying again...")
                continue  # Skip this pose if it's outside the grid
                
            # Check clash validity with pose relaxation
            if not self._check_pose_validity(pose, protein):
                try:
                    # Try to relax the pose to avoid clashes
                    relaxed_pose = self._gentle_clash_relief(protein, pose, max_steps=5)
                    if self._check_pose_validity(relaxed_pose, protein):
                        population.append((relaxed_pose, None))
                        if progress_bar:
                            progress_bar.update(1)
                        elif len(population) % 10 == 0:
                            print(f"Generated {len(population)}/{self.population_size} valid poses (after relaxation)")
                    else:
                        if attempts % 100 == 0:
                            self.logger.debug(f"Attempt {attempts}: Clash could not be resolved, trying again...")
                except Exception as e:
                    if attempts % 100 == 0:
                        self.logger.warning(f"Error during clash resolution: {e}")
                continue
            
            # If we reach here, the pose is valid
            population.append((pose, None))
            if progress_bar:
                progress_bar.update(1)
            elif len(population) % 10 == 0:
                print(f"Generated {len(population)}/{self.population_size} valid poses")
        
        if progress_bar:
            progress_bar.close()
            
        if len(population) < self.population_size:
            self.logger.warning(f"Could only generate {len(population)}/{self.population_size} valid poses after {attempts} attempts")
        else:
            self.logger.info(f"Successfully generated {len(population)} valid poses (took {attempts} attempts)")
        
        return population
    
    def _evaluate_population(self, protein, population):
        """
        Evaluate population using batch processing for improved GPU efficiency.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        population : list
            List of (pose, score) tuples
        
        Returns:
        --------
        list
            Evaluated population as (pose, score) tuples
        """
        # Extract poses from population
        poses = [pose for pose, _ in population]
        
        # Batch scoring with the scoring function
        if hasattr(self.scoring_function, 'score_batch'):
            self.logger.info(f"Using batch scoring for {len(poses)} poses...")
            try:
                scores = self.scoring_function.score_batch(protein, poses)
                
                # Combine poses with scores
                results = [(copy.deepcopy(pose), score) for pose, score in zip(poses, scores)]
            except Exception as e:
                self.logger.error(f"Batch scoring failed: {e}, falling back to sequential scoring")
                # Fall back to sequential scoring
                results = []
                for i, (pose, _) in enumerate(population):
                    try:
                        score = self.scoring_function.score(protein, pose)
                        results.append((copy.deepcopy(pose), score))
                    except Exception as e:
                        self.logger.warning(f"Error scoring pose {i}: {e}")
                        results.append((copy.deepcopy(pose), float('inf')))
        else:
            # Fall back to sequential scoring
            self.logger.info("Batch scoring not available, using sequential scoring...")
            
            # Initialize progress bar
            try:
                from tqdm import tqdm
                progress_bar = tqdm(total=len(population), desc="Evaluating poses")
            except ImportError:
                progress_bar = None
                
            results = []
            for i, (pose, _) in enumerate(population):
                try:
                    score = self.scoring_function.score(protein, pose)
                    results.append((copy.deepcopy(pose), score))
                except Exception as e:
                    self.logger.warning(f"Error scoring pose {i}: {e}")
                    results.append((copy.deepcopy(pose), float('inf')))
                
                if progress_bar:
                    progress_bar.update(1)
                elif i % 10 == 0 and i > 0 and len(population) > 50:
                    print(f"  Evaluating pose {i}/{len(population)}...")
            
            if progress_bar:
                progress_bar.close()
        
        return results

    def _mutate(self, individual, original_individual, center, radius):
        """
        Improved mutation with boundary constraints.
        
        Parameters:
        -----------
        individual : Ligand
            Individual to mutate
        original_individual : Ligand
            Original individual (backup)
        center : numpy.ndarray
            Center of the search space
        radius : float
            Radius of the search space
            
        Returns:
        --------
        Ligand
            Mutated individual
        """
        if random.random() >= self.mutation_rate:
            return individual  # No mutation
        
        # Store original in case mutation fails
        backup = copy.deepcopy(individual)
        
        # Determine molecular radius for safety calculations
        atom_coords = individual.xyz
        centroid = np.mean(atom_coords, axis=0)
        molecular_radius = np.max(np.linalg.norm(atom_coords - centroid, axis=1))
        
        # Calculate safe parameters based on distance from boundary
        dist_to_center = np.linalg.norm(centroid - center)
        dist_to_boundary = radius - dist_to_center
        boundary_factor = min(1.0, dist_to_boundary / (molecular_radius + 0.5))
        
        # Scale mutation magnitudes by boundary factor (smaller near boundary)
        max_translation = 0.5 * boundary_factor  # Smaller steps near boundary
        max_rotation = 0.2 * boundary_factor     # Smaller rotations near boundary
        
        # Choose mutation type with bias toward safer operations near boundary
        if dist_to_boundary < molecular_radius * 1.5:
            # Near boundary: prefer rotation or gentle inward movement
            operations = ['inward', 'rotation', 'gentle_translation']
            weights = [0.5, 0.4, 0.1]  # Higher weight for boundary-safe operations
            mutation_type = random.choices(operations, weights=weights, k=1)[0]
        else:
            # Far from boundary: more freedom to move
            operations = ['translation', 'rotation', 'both']
            weights = [0.4, 0.3, 0.3]
            mutation_type = random.choices(operations, weights=weights, k=1)[0]
        
        # Apply the selected mutation
        if mutation_type == 'translation':
            # Random translation with boundary-aware magnitude
            translation = np.random.normal(0, max_translation, 3)
            individual.translate(translation)
            
        elif mutation_type == 'gentle_translation':
            # Very small random translation
            translation = np.random.normal(0, 0.2, 3)
            individual.translate(translation)
            
        elif mutation_type == 'inward':
            # Move toward center
            centroid = np.mean(individual.xyz, axis=0)
            vector_to_center = center - centroid
            dist_to_center = np.linalg.norm(vector_to_center)
            
            if dist_to_center > 0:
                # Random inward movement magnitude
                inward_magnitude = min(0.5, dist_to_center * 0.1)
                move_vector = (vector_to_center / dist_to_center) * np.random.uniform(0.1, inward_magnitude)
                individual.translate(move_vector)
                
        elif mutation_type == 'rotation':
            # Random rotation with boundary-aware angle
            angle = np.random.normal(0, max_rotation)
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            
            rotation = Rotation.from_rotvec(axis * angle)
            centroid = np.mean(individual.xyz, axis=0)
            individual.translate(-centroid)
            individual.rotate(rotation.as_matrix())
            individual.translate(centroid)
            
        elif mutation_type == 'both':
            # Both translation and rotation
            translation = np.random.normal(0, max_translation * 0.7, 3)
            individual.translate(translation)
            
            angle = np.random.normal(0, max_rotation * 0.7)
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            
            rotation = Rotation.from_rotvec(axis * angle)
            centroid = np.mean(individual.xyz, axis=0)
            individual.translate(-centroid)
            individual.rotate(rotation.as_matrix())
            individual.translate(centroid)
        
        # Verify all atoms are inside sphere
        if not is_fully_inside_sphere(individual, center, radius):
            # Try to reposition
            if not reposition_inside_sphere(individual, center, radius):
                # Revert to original if repositioning failed
                individual.xyz = backup.xyz.copy()
                
                # Optionally try a minimal inward movement as last resort
                centroid = np.mean(individual.xyz, axis=0)
                vector_to_center = center - centroid
                dist_to_center = np.linalg.norm(vector_to_center)
                
                if dist_to_center > 0:
                    # Very gentle inward nudge
                    move_vector = (vector_to_center / dist_to_center) * 0.1
                    individual.translate(move_vector)
        
        # Verify no clashes were introduced
        if not self._check_pose_validity(individual, self.protein):
            # Revert to original if mutation caused clashes
            individual.xyz = backup.xyz.copy()
        
        return individual
    
    def _crossover_pair(self, parent1, parent2):
        """
        Perform crossover between two parents using a more sophisticated approach.
        
        Parameters:
        -----------
        parent1 : Ligand
            First parent
        parent2 : Ligand
            Second parent
        
        Returns:
        --------
        tuple
            (child1, child2) as Ligand objects
        """
        # Create deep copies to avoid modifying parents
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        center = self.protein.active_site['center']
        radius = self.protein.active_site['radius']
        
        # Calculate centroids
        centroid1 = np.mean(parent1.xyz, axis=0)
        centroid2 = np.mean(parent2.xyz, axis=0)
        
        # Weighted centroid crossover
        alpha = random.uniform(0.3, 0.7)  # Random weight for variability
        new_centroid1 = alpha * centroid1 + (1 - alpha) * centroid2
        new_centroid2 = (1 - alpha) * centroid1 + alpha * centroid2
        
        # Apply translation to children
        child1.translate(new_centroid1 - centroid1)
        child2.translate(new_centroid2 - centroid2)
        
        # Fragment-based crossover
        fragment_indices = random.sample(range(len(parent1.xyz)), len(parent1.xyz) // 2)
        for idx in fragment_indices:
            child1.xyz[idx], child2.xyz[idx] = child2.xyz[idx], child1.xyz[idx]
        
        # Rotation interpolation
        rotation1 = Rotation.random()
        rotation2 = Rotation.random()
        key_times = [0, 1]
        rotations = Rotation.concatenate([rotation1, rotation2])
        slerp = Slerp(key_times, rotations)
        interpolated_rotation = slerp([alpha])[0]  # Interpolate at alpha
        
        # Apply interpolated rotation to children
        centroid1 = np.mean(child1.xyz, axis=0)
        centroid2 = np.mean(child2.xyz, axis=0)
        
        child1.translate(-centroid1)
        child1.rotate(interpolated_rotation.as_matrix())
        child1.translate(centroid1)
        child1 = enforce_sphere_boundary(child1, center, radius)
        
        child2.translate(-centroid2)
        child2.rotate(interpolated_rotation.as_matrix())
        child2.translate(centroid2)
        child2 = enforce_sphere_boundary(child2, center, radius)
        
        # Validate children
        if not self._validate_conformation(child1):
            child1 = self._repair_conformation(child1)
        
        if not is_inside_sphere(child1, center, radius):
            centroid = np.mean(child1.xyz, axis=0)
            to_center = center - centroid
            dist = np.linalg.norm(to_center)
            if dist > 0:
                move_vector = to_center * (dist - radius*0.9)/dist
                child1.translate(move_vector)
            
        if not self._validate_conformation(child2):
            child2 = self._repair_conformation(child2)
        
        if not is_inside_sphere(child2, center, radius):
            centroid = np.mean(child2.xyz, axis=0)
            to_center = center - centroid
            dist = np.linalg.norm(to_center)
            if dist > 0:
                move_vector = to_center * (dist - radius*0.9)/dist
                child2.translate(move_vector)
        
        return child1, child2
    
    def _selection(self, population):
        """
        Perform selection to choose parents for the next generation.
        
        Parameters:
        -----------
        population : list
            List of (pose, score) tuples
            
        Returns:
        --------
        list
            Selected individuals as (pose, score) tuples
        """
        # Tournament selection with diversity preservation
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(population, min(self.tournament_size, len(population)))
            tournament.sort(key=lambda x: x[1])
            selected.append(tournament[0])
            
        # Ensure some diversity by adding some random individuals
        if len(selected) > 5:
            random_indices = random.sample(range(len(population)), min(5, len(population)))
            for idx in random_indices:
                if population[idx] not in selected:
                    selected.append(population[idx])
                    
        return selected[:self.population_size]
    
    def search(self, protein, ligand):
        """
        Perform genetic algorithm search in parallel.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
            
        Returns:
        --------
        list
            List of (pose, score) tuples, sorted by score
        """
        start_time = time.time()
        
        # Setup search space
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            center = np.mean(protein.xyz, axis=0)
            radius = 10.0

        self.smart_grid_points = self.initialize_smart_grid(protein, center, radius)
        self.protein = protein  # Store protein for access in crossover
        self.center = center
        self.radius = radius

        # Ensure active site is properly defined
        if not hasattr(protein, 'active_site') or protein.active_site is None:
            protein.active_site = {
                'center': center,
                'radius': radius
            }
        if 'atoms' not in protein.active_site or protein.active_site['atoms'] is None:
            protein.active_site['atoms'] = [
                atom for atom in protein.atoms
                if np.linalg.norm(atom['coords'] - center) <= radius
            ]
            self.logger.info(f"Added {len(protein.active_site['atoms'])} atoms into active_site region")

        self.logger.info(f"Searching around center {center} with radius {radius}")
        
        # Initialize population
        population = self.initialize_population(protein, ligand)
        
        # Evaluate initial population
        evaluated_population = self._evaluate_population(protein, population)
        if not evaluated_population:
            self.logger.error("No valid poses found during population initialization.")
            return []
            
        # Filter out infinite scores
        evaluated_population = [p for p in evaluated_population if np.isfinite(p[1])]
        if not evaluated_population:
            self.logger.error("All poses have infinite scores.")
            return []
            
        # Sort population by score
        evaluated_population.sort(key=lambda x: x[1])
        
        # Store best individual
        best_individual = evaluated_population[0]
        self.best_pose = best_individual[0]
        self.best_score = best_individual[1]
        
        self.logger.info(f"Generation 0: Best score = {self.best_score:.4f}")
        
        # Track all individuals for diverse results
        all_individuals = [evaluated_population[0]]
        
        # Create progress bar
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=self.max_iterations, desc="Evolution progress")
        except ImportError:
            progress_bar = None
        
        # Main evolutionary loop
        for generation in range(self.max_iterations):
            current_radius = self._adjust_search_radius(radius, generation, self.max_iterations)

            gen_start = time.time()
            
            # Select parents
            parents = self._selection(evaluated_population)
            
            # Create offspring through crossover and mutation
            offspring = []
            
            # Apply genetic operators
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    parent1 = parents[i][0]
                    parent2 = parents[i+1][0]
                    
                    # Crossover with probability
                    if random.random() < self.crossover_rate:
                        child1, child2 = self._crossover_pair(parent1, parent2)
                    else:
                        child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                    
                    # Mutation
                    self._mutate(child1, copy.deepcopy(parent1), center, current_radius)
                    self._mutate(child2, copy.deepcopy(parent2), center, current_radius)

                    offspring.append((child1, None))
                    offspring.append((child2, None))
            
            # Evaluate offspring
            eval_start = time.time()
            evaluated_offspring = self._evaluate_population(protein, offspring)
            self.eval_time += time.time() - eval_start
            
            # Filter out invalid scores
            evaluated_offspring = [p for p in evaluated_offspring if np.isfinite(p[1])]
            
            # Combine parent and offspring populations (μ + λ)
            combined = evaluated_population + evaluated_offspring
            
            # Keep only the best individuals (elitism)
            combined.sort(key=lambda x: x[1])
            evaluated_population = combined[:self.population_size]
            
            # Update best solution
            if evaluated_population[0][1] < self.best_score:
                self.best_pose = evaluated_population[0][0]
                self.best_score = evaluated_population[0][1]
                all_individuals.append(evaluated_population[0])
            
            # Display progress
            gen_time = time.time() - gen_start
            message = (f"Generation {generation + 1}/{self.max_iterations}: "
                      f"Best score = {self.best_score:.4f}, "
                      f"Current best = {evaluated_population[0][1]:.4f}, "
                      f"Time = {gen_time:.2f}s")
            
            self.logger.info(message)
            
            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix({"Best": f"{self.best_score:.4f}"})
            
            # Apply local search to the best individual occasionally
            if self.perform_local_opt and generation % 5 == 0:
                try:
                    best_pose, best_score = self._local_optimization(
                        evaluated_population[0][0], protein)
                    
                    if best_score < self.best_score:
                        self.best_pose = best_pose
                        self.best_score = best_score
                        
                        # Replace best individual in population
                        evaluated_population[0] = (best_pose, best_score)
                        evaluated_population.sort(key=lambda x: x[1])
                        all_individuals.append((best_pose, best_score))
                        
                        self.logger.info(f"Local optimization improved score to {best_score:.4f}")
                except Exception as e:
                    self.logger.warning(f"Error during local optimization: {e}")
        
        if progress_bar:
            progress_bar.close()
            
        # Return unique solutions, best first
        self.total_time = time.time() - start_time
        self.logger.info(f"\nSearch completed in {self.total_time:.2f} seconds")
        self.logger.info(f"Evaluation time: {self.eval_time:.2f} seconds ({self.eval_time/self.total_time*100:.1f}%)")
        
        # Filter duplicate poses based on RMSD
        unique_individuals = []
        for pose, score in all_individuals:
            # Skip poses with bad scores
            if not np.isfinite(score):
                continue
                
            # Check if this pose is already in our unique set
            is_duplicate = False
            for unique_pose, _ in unique_individuals:
                try:
                    rmsd = calculate_rmsd(pose.xyz, unique_pose.xyz)
                    if rmsd < 2.0:  # RMSD threshold for considering poses as duplicates
                        is_duplicate = True
                        break
                except Exception:
                    # If RMSD calculation fails, assume it's not a duplicate
                    pass
                    
            if not is_duplicate:
                unique_individuals.append((pose, score))
        
        # Sort unique individuals by score
        unique_individuals.sort(key=lambda x: x[1])
        
        # Save top pose to PDB if output directory is specified
        if self.output_dir and len(unique_individuals) > 0:
            top_pose = unique_individuals[0][0]
            top_score = unique_individuals[0][1]
            
            output_path = Path(self.output_dir) / f"top_pose_{top_score:.2f}.pdb"
            try:
                top_pose.write_pdb(output_path)
                self.logger.info(f"Saved top pose to {output_path}")
            except Exception as e:
                self.logger.error(f"Error saving top pose: {e}")
        
        # Return results
        return unique_individuals


# ------------------------------------------------------------------------------
# Parallel Random Search
# ------------------------------------------------------------------------------
class ParallelRandomSearch(RandomSearch, GridUtilsMixin, ClashDetectionMixin, PoseGenerationMixin, OptimizationMixin):
    """
    Parallel implementation of random search for molecular docking.
    
    This class extends the RandomSearch with parallel processing capabilities
    to improve performance on multi-core systems.
    """
    
    def __init__(self, scoring_function, max_iterations=10, n_processes=None, 
                 batch_size=None, process_pool=None, output_dir=None, 
                 grid_spacing=0.375, grid_radius=10.0, grid_center=None):
        """
        Initialize parallel random search.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function to evaluate poses
        max_iterations : int
            Maximum number of iterations
        n_processes : int
            Number of parallel processes (None = use all available cores)
        batch_size : int
            Number of poses to evaluate in each batch
        process_pool : multiprocessing.Pool
            Process pool to use (None = create a new pool)
        output_dir : str or Path
            Output directory
        grid_spacing : float
            Spacing between grid points
        grid_radius : float
            Radius of the search sphere
        grid_center : array-like
            Center coordinates of the search sphere
        """
        super().__init__(scoring_function, max_iterations)
        self.output_dir = output_dir
        self.grid_spacing = grid_spacing
        self.grid_radius = grid_radius
        self.grid_center = grid_center
        self.grid_points = None

        # Setup parallel processing
        if n_processes is None:
            self.n_processes = mp.cpu_count()
        else:
            self.n_processes = n_processes

        if batch_size is None:
            self.batch_size = max(10, self.max_iterations // (self.n_processes * 5))
        else:
            self.batch_size = batch_size

        self.process_pool = process_pool
        self.own_pool = False

        # Performance tracking
        self.eval_time = 0.0
        self.total_time = 0.0
        self.best_score = float('inf')

        # Set up logging
        if output_dir:
            self.logger = setup_logging(output_dir)
        else:
            import logging
            self.logger = logging.getLogger("null_logger")
            self.logger.addHandler(logging.NullHandler())

    def search(self, protein, ligand):
        """
        Perform random search in parallel.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
            
        Returns:
        --------
        list
            List of (pose, score) tuples, sorted by score
        """
        start_time = time.time()

        # Setup search space
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            center = np.mean(protein.xyz, axis=0)
            radius = 10.0
            
        self.smart_grid_points = self.initialize_smart_grid(protein, center, radius)
        
        # Ensure active site is properly defined
        if not hasattr(protein, 'active_site') or protein.active_site is None:
            protein.active_site = {
                'center': center,
                'radius': radius
            }
        if 'atoms' not in protein.active_site or protein.active_site['atoms'] is None:
            protein.active_site['atoms'] = [
                atom for atom in protein.atoms
                if np.linalg.norm(atom['coords'] - center) <= radius
            ]
            self.logger.info(f"Added {len(protein.active_site['atoms'])} atoms into active_site region")

        self.logger.info(f"Searching around center {center} with radius {radius}")
        self.logger.info(f"Using {self.n_processes} CPU cores for evaluation")

        # Save sphere grid
        self.initialize_grid_points(center, protein=protein)

        # Initialize results
        results = []
        
        # New variables to track clash failures
        fail_counter = 0
        max_failures = 30  # After 30 consecutive fails, expand radius
        radius_increment = 1.0  # How much to expand each time
        
        # Try to create a progress bar
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=self.max_iterations, desc="Evaluating poses")
        except ImportError:
            progress_bar = None
        
        for i in range(self.max_iterations):
            if progress_bar:
                progress_bar.update(1)
            elif i % 25 == 0 and i > 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = avg_time * (self.max_iterations - i)
                self.logger.info(f"Progress: {i}/{self.max_iterations} poses evaluated ({i/self.max_iterations*100:.1f}%) - "
                               f"Est. remaining: {remaining:.1f}s")

            # Adjust radius dynamically
            current_radius = self._adjust_search_radius(radius, i, self.max_iterations)

            pose = self._generate_random_pose(ligand, center, current_radius)
            
            # Apply soft outward nudge to avoid deep burial
            nudge = np.random.normal(0.2, 0.1, size=3)
            pose.translate(nudge)

            if not is_inside_sphere(pose, center, current_radius):
                # Calculate centroid-to-center vector
                centroid = np.mean(pose.xyz, axis=0)
                to_center = center - centroid
                # Scale vector to move back to sphere boundary
                dist = np.linalg.norm(to_center)
                if dist > 0:
                    move_vector = to_center * (dist - radius*0.9)/dist
                    pose.translate(move_vector)

            # Initial clash checks
            if not self._check_pose_validity(pose, protein, center=center, radius=current_radius):
                fail_counter += 1
                if fail_counter >= max_failures:
                    radius += radius_increment
                    current_radius += radius_increment
                    fail_counter = 0
                    self.logger.warning(f"Auto-expanding search radius to {radius:.2f} Å due to repeated clashes!")
                continue

            # Score valid pose and add to results
            try:
                score = self.scoring_function.score(protein, pose)
            except Exception as e:
                self.logger.warning(f"Error scoring pose: {e}")
                continue

            # Final validation *before* appending
            if not self._check_pose_validity(pose, protein, center=center, radius=current_radius):
                fail_counter += 1
                if fail_counter >= max_failures:
                    radius += radius_increment
                    current_radius += radius_increment
                    fail_counter = 0
                    self.logger.warning(f"Auto-expanding search radius to {radius:.2f} Å due to repeated clashes!")
                continue

            fail_counter = 0
            results.append((pose, score))
            
            # Update progress bar with best score
            if progress_bar and results:
                current_best = min(r[1] for r in results)
                progress_bar.set_postfix({"Best": f"{current_best:.4f}"})

        if progress_bar:
            progress_bar.close()

        # Optional: Refine top N poses with local optimization
        if results and hasattr(self, '_local_optimization'):
            self.logger.info("Refining top poses with local optimization...")
            
            try:
                from tqdm import tqdm
                opt_progress = tqdm(total=min(5, len(results)), desc="Optimizing top poses")
            except ImportError:
                opt_progress = None
                
            for i, (pose, score) in enumerate(sorted(results, key=lambda x: x[1])[:5]):  # Top 5 poses
                try:
                    optimized_pose, optimized_score = self._local_optimization(pose, protein)
                    results.append((optimized_pose, optimized_score))
                    if opt_progress:
                        opt_progress.update(1)
                        opt_progress.set_postfix({"Score": f"{optimized_score:.4f}"})
                except Exception as e:
                    self.logger.warning(f"Error during local optimization of pose {i}: {e}")
            
            if opt_progress:
                opt_progress.close()

        # Re-sort results after refinement
        results.sort(key=lambda x: x[1])

        self.total_time = time.time() - start_time

        if not results:
            self.logger.error("No valid poses generated! All poses clashed or failed. Returning empty result.")
            return []

        # Filter duplicate poses based on RMSD
        unique_results = []
        for pose, score in results:
            # Skip poses with bad scores
            if not np.isfinite(score):
                continue
                
            # Check if this pose is already in our unique set
            is_duplicate = False
            for unique_pose, _ in unique_results:
                try:
                    rmsd = calculate_rmsd(pose.xyz, unique_pose.xyz)
                    if rmsd < 2.0:  # RMSD threshold for considering poses as duplicates
                        is_duplicate = True
                        break
                except Exception:
                    # If RMSD calculation fails, assume it's not a duplicate
                    pass
                    
            if not is_duplicate:
                unique_results.append((pose, score))
        
        # Print summary info
        self.logger.info(f"Search completed in {self.total_time:.2f} seconds")
        if unique_results:
            self.logger.info(f"Best score: {unique_results[0][1]:.4f}")
            self.logger.info(f"Unique poses found: {len(unique_results)}")
        else:
            self.logger.warning("No valid unique poses found!")

        # Save top pose to PDB if output directory is specified
        if self.output_dir and len(unique_results) > 0:
            top_pose = unique_results[0][0]
            top_score = unique_results[0][1]
            
            output_path = Path(self.output_dir) / f"top_pose_{top_score:.2f}.pdb"
            try:
                top_pose.write_pdb(output_path)
                self.logger.info(f"Saved top pose to {output_path}")
            except Exception as e:
                self.logger.error(f"Error saving top pose: {e}")

        return unique_results
    
    def _generate_random_pose(self, ligand, center, radius):
        """
        Generate a random ligand pose within the specified radius.
        
        Parameters:
        -----------
        ligand : Ligand
            Ligand template to position
        center : numpy.ndarray
            Center coordinates of the search space
        radius : float
            Radius of the search sphere
            
        Returns:
        --------
        Ligand
            Random ligand pose
        """
        # More uniform sampling within sphere volume
        r = radius * (0.8 + 0.2 * random.random())  # Bias toward outer region
        theta = random.uniform(0, 2 * np.pi)
        phi = random.uniform(0, np.pi)

        x = center[0] + r * np.sin(phi) * np.cos(theta)
        y = center[1] + r * np.sin(phi) * np.sin(theta)
        z = center[2] + r * np.cos(phi)

        pose = copy.deepcopy(ligand)
        centroid = np.mean(pose.xyz, axis=0)
        
        translation = np.array([x, y, z]) - centroid
        pose.translate(translation)

        # Apply random rotation
        rotation = Rotation.random()
        rotation_matrix = rotation.as_matrix()
        centroid = np.mean(pose.xyz, axis=0)
        pose.translate(-centroid)
        pose.rotate(rotation_matrix)
        pose.translate(centroid)

        return pose
    
    def evaluate_batch(self, protein, poses):
        """
        Evaluate a batch of poses in parallel.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        poses : list
            List of ligand poses to evaluate
            
        Returns:
        --------
        list
            List of (pose, score) tuples
        """
        # Initialize multiprocessing pool if needed
        if self.process_pool is None:
            self.process_pool = mp.Pool(self.n_processes)
            self.own_pool = True
            
        # Define task for each worker
        def score_task(pose):
            try:
                if not self._check_pose_validity(pose, protein):
                    return pose, float('inf')
                    
                score = self.scoring_function.score(protein, pose)
                return pose, score
            except Exception as e:
                self.logger.warning(f"Error in worker process: {e}")
                return pose, float('inf')
        
        # Execute tasks in parallel
        results = []
        try:
            results = self.process_pool.map(score_task, poses)
        except Exception as e:
            self.logger.error(f"Error in parallel evaluation: {e}")
            # Fall back to sequential evaluation
            for pose in poses:
                try:
                    if not self._check_pose_validity(pose, protein):
                        continue
                        
                    score = self.scoring_function.score(protein, pose)
                    results.append((pose, score))
                except Exception as e:
                    self.logger.warning(f"Error scoring pose: {e}")
                    
        # Filter out invalid scores
        return [r for r in results if np.isfinite(r[1])]
    
    def _execute_search_batch(self, protein, ligand, center, radius, batch_size=None):
        """
        Execute a batch of random searches.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand template
        center : numpy.ndarray
            Center coordinates of the search space
        radius : float
            Radius of the search sphere
        batch_size : int
            Size of the batch
            
        Returns:
        --------
        list
            List of (pose, score) tuples
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Generate batch of poses
        poses = []
        for _ in range(batch_size):
            pose = self._generate_random_pose(ligand, center, radius)
            poses.append(pose)
            
        # Evaluate poses in parallel
        return self.evaluate_batch(protein, poses)
    
    def _adjust_search_radius(self, initial_radius, iteration, total_iterations, decay_rate=0.5):
        """
        Shrink search radius over iterations.
        
        Parameters:
        -----------
        initial_radius : float
            Initial search radius
        iteration : int
            Current iteration
        total_iterations : int
            Total number of iterations
        decay_rate : float
            Rate of radius decay (0-1)
            
        Returns:
        --------
        float
            Adjusted radius for the current iteration
        """
        factor = 1.0 - (iteration / total_iterations) * decay_rate
        return max(initial_radius * factor, initial_radius * 0.5)
    
    def _check_pose_validity(self, ligand, protein, clash_threshold=1.5, center=None, radius=None):
        """
        Check if ligand pose clashes with protein atoms and is within sphere.
        
        Parameters:
            ligand: Ligand object with .atoms
            protein: Protein object with .atoms or active_site['atoms']
            clash_threshold: Ångström cutoff for hard clash
            center: Center of the search sphere (optional)
            radius: Radius of the search sphere (optional)
            
        Returns:
            bool: True if pose is valid (no severe clash and within sphere), False otherwise
        """
        # Check if ligand is within sphere (if center and radius provided)
        if center is not None and radius is not None:
            if not is_inside_sphere(ligand, center, radius):
                return False
        
        # Continue with existing clash detection
        ligand_coords = np.array([atom['coords'] for atom in ligand.atoms])
        
        # Use active site atoms if defined
        if hasattr(protein, 'active_site') and protein.active_site and 'atoms' in protein.active_site:
            protein_coords = np.array([atom['coords'] for atom in protein.active_site['atoms']])
        else:
            protein_coords = np.array([atom['coords'] for atom in protein.atoms])
        
        # Check for clashes
        for lig_coord in ligand_coords:
            distances = np.linalg.norm(protein_coords - lig_coord, axis=1)
            if np.any(distances < clash_threshold):
                return False  # Clash detected
        
        # Enhanced clash detection for more subtle clashes
        if hasattr(self, '_enhanced_clash_detection'):
            is_clashing, _ = self._enhanced_clash_detection(protein, ligand)
            return not is_clashing
        
        return True
    
    def cleanup(self):
        """Clean up resources after search."""
        if self.own_pool and self.process_pool is not None:
            self.process_pool.close()
            self.process_pool.join()
            self.process_pool = None

# ------------------------------------------------------------------------------
# Hybrid Search Algorithm
# ------------------------------------------------------------------------------

class HybridSearch(GridUtilsMixin, ClashDetectionMixin, PoseGenerationMixin, OptimizationMixin):
    """
    Hybrid search algorithm combining Monte Carlo, genetic algorithm, and simulated annealing.
    
    This class implements a sophisticated hybrid approach that combines the strengths
    of different search strategies for improved performance in molecular docking.
    """
    
    def __init__(self, scoring_function, max_iterations=300, population_size=30,
                 n_processes=None, output_dir=None, grid_spacing=0.375,
                 grid_radius=10.0, grid_center=None, temperature_start=5.0,
                 temperature_end=0.1, cooling_factor=0.95, mutation_rate=0.3,
                 crossover_rate=0.7, local_opt_frequency=5):
        """
        Initialize hybrid search algorithm.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function to evaluate poses
        max_iterations : int
            Maximum number of iterations
        population_size : int
            Size of the population
        n_processes : int
            Number of parallel processes (None = use all available cores)
        output_dir : str or Path
            Output directory
        grid_spacing : float
            Spacing between grid points
        grid_radius : float
            Radius of the search sphere
        grid_center : array-like
            Center coordinates of the search sphere
        temperature_start : float
            Starting temperature for simulated annealing
        temperature_end : float
            Ending temperature for simulated annealing
        cooling_factor : float
            Cooling factor for simulated annealing
        mutation_rate : float
            Probability of mutation in genetic algorithm phase
        crossover_rate : float
            Probability of crossover in genetic algorithm phase
        local_opt_frequency : int
            Frequency of local optimization (every N iterations)
        """
        self.scoring_function = scoring_function
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.output_dir = output_dir
        self.grid_spacing = grid_spacing
        self.grid_radius = grid_radius
        self.grid_center = grid_center
        self.grid_points = None
        
        # Simulated annealing parameters
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.cooling_factor = cooling_factor
        
        # Genetic algorithm parameters
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = 3
        
        # Local optimization parameters
        self.local_opt_frequency = local_opt_frequency
        
        # Setup parallel processing
        if n_processes is None:
            self.n_processes = mp.cpu_count()
        else:
            self.n_processes = n_processes
            
        # Performance tracking
        self.eval_time = 0.0
        self.total_time = 0.0
        self.best_score = float('inf')
        self.best_pose = None
        
        # Set up logging
        if output_dir:
            self.logger = setup_logging(output_dir)
        else:
            import logging
            self.logger = logging.getLogger("hybrid_search")
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
    
    def _initialize_population(self, protein, ligand, center, radius):
        """
        Initialize population for hybrid search within spherical grid.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand template
        center : numpy.ndarray
            Center coordinates of the search space
        radius : float
            Radius of the search sphere
            
        Returns:
        --------
        list
            List of (pose, score) tuples
        """
        population = []
        
        self.logger.info(f"Initializing population with {self.population_size} individuals")
        
        # Initialize grid points for sampling
        self.initialize_grid_points(center, protein=protein, radius=radius)
        
        # Add retry logic here
        attempts = 0
        max_attempts = self.population_size * 10
        
        # Initialize progress bar
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=self.population_size, desc="Initializing population")
        except ImportError:
            progress_bar = None
        
        # Generate initial population with retry logic
        while len(population) < self.population_size and attempts < max_attempts:
            attempts += 1
            
            pose = self._generate_valid_pose(protein, ligand, center, radius)
            
            if pose is None:
                continue
                
            # Score the pose
            try:
                score = self.scoring_function.score(protein, pose)
                population.append((pose, score))
                
                if progress_bar:
                    progress_bar.update(1)
                    if len(population) > 0:
                        progress_bar.set_postfix({"Best": f"{min(p[1] for p in population):.4f}"})
                elif len(population) % 5 == 0:
                    self.logger.info(f"Generated {len(population)}/{self.population_size} valid poses")
            except Exception as e:
                self.logger.warning(f"Error scoring pose during initialization: {e}")
        
        if progress_bar:
            progress_bar.close()
            
        if len(population) < self.population_size:
            self.logger.warning(f"Could only generate {len(population)}/{self.population_size} valid poses after {attempts} attempts")
        else:
            self.logger.info(f"Successfully generated {len(population)} valid poses (took {attempts} attempts)")
        
        return population
    
    def _simulated_annealing_step(self, pose, protein, center, radius, temperature):
        """
        Perform one step of simulated annealing.
        
        Parameters:
        -----------
        pose : Ligand
            Current pose
        protein : Protein
            Protein object
        center : numpy.ndarray
            Center coordinates of the search space
        radius : float
            Radius of the search sphere
        temperature : float
            Current temperature
            
        Returns:
        --------
        tuple
            (new_pose, new_score, accepted) - new pose, its score, and whether it was accepted
        """
        current_score = self.scoring_function.score(protein, pose)
        
        # Create new candidate
        new_pose = copy.deepcopy(pose)
        
        # Apply random perturbation (combination of translation and rotation)
        # Scale perturbation size with temperature
        translation_magnitude = 0.5 * temperature / self.temperature_start
        rotation_magnitude = 0.2 * temperature / self.temperature_start
        
        # Random translation
        translation = np.random.normal(0, translation_magnitude, 3)
        new_pose.translate(translation)
        
        # Random rotation
        angle = np.random.normal(0, rotation_magnitude)
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        rotation = Rotation.from_rotvec(axis * angle)
        centroid = np.mean(new_pose.xyz, axis=0)
        new_pose.translate(-centroid)
        new_pose.rotate(rotation.as_matrix())
        new_pose.translate(centroid)
        
        # Ensure pose is inside search sphere
        if not is_inside_sphere(new_pose, center, radius):
            new_pose = enforce_sphere_boundary(new_pose, center, radius)
        
        # Check validity (no severe clashes)
        if not self._check_pose_validity(new_pose, protein):
            # Try to resolve clashes gently
            try:
                new_pose = self._gentle_clash_relief(protein, new_pose, max_steps=5, center=center, radius=radius)
            except Exception as e:
                self.logger.debug(f"Error during clash relief: {e}")
                return pose, current_score, False
            
            # Check again after clash relief
            if not self._check_pose_validity(new_pose, protein):
                return pose, current_score, False
        
        # Score new pose
        try:
            new_score = self.scoring_function.score(protein, new_pose)
        except Exception as e:
            self.logger.debug(f"Error scoring pose: {e}")
            return pose, current_score, False
        
        # Accept or reject according to Metropolis criterion
        if new_score <= current_score:
            # Always accept if score is better
            return new_pose, new_score, True
        else:
            # Accept with probability based on score difference and temperature
            delta = new_score - current_score
            probability = np.exp(-delta / temperature)
            
            if random.random() < probability:
                return new_pose, new_score, True
            else:
                return pose, current_score, False
    
    def search(self, protein, ligand):
        """
        Perform hybrid search for molecular docking.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
            
        Returns:
        --------
        list
            List of (pose, score) tuples, sorted by score
        """
        start_time = time.time()
        
        # Setup search space
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            center = np.mean(protein.xyz, axis=0)
            radius = 10.0
        
        self.logger.info(f"Starting hybrid search around center {center} with radius {radius}")
        
        # Initialize smart grid for guided sampling
        self.smart_grid_points = self.initialize_smart_grid(protein, center, radius)
        
        # Ensure active site is properly defined
        if not hasattr(protein, 'active_site') or protein.active_site is None:
            protein.active_site = {
                'center': center,
                'radius': radius
            }
        if 'atoms' not in protein.active_site or protein.active_site['atoms'] is None:
            protein.active_site['atoms'] = [
                atom for atom in protein.atoms
                if np.linalg.norm(atom['coords'] - center) <= radius
            ]
            self.logger.info(f"Added {len(protein.active_site['atoms'])} atoms into active_site region")
        
        # Initialize population
        population = self._initialize_population(protein, ligand, center, radius)
        
        # Sort by score
        population.sort(key=lambda x: x[1])
        
        # Update best solution
        self.best_pose = population[0][0]
        self.best_score = population[0][1]
        
        self.logger.info(f"Initial best score: {self.best_score:.4f}")
        
        # Track all promising solutions
        all_solutions = [population[0]]
        
        # Initialize temperature for simulated annealing
        temperature = self.temperature_start
        
        # Create progress bar
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=self.max_iterations, desc="Hybrid search progress")
        except ImportError:
            progress_bar = None
        
        # Main search loop
        for iteration in range(self.max_iterations):
            # Update progress bar
            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix({"Best": f"{self.best_score:.4f}", "T": f"{temperature:.2f}"})
            
            # Determine search strategy based on current state
            # - Early iterations: more exploration (SA and diverse moves)
            # - Middle iterations: balanced (GA with occasional SA)
            # - Late iterations: more exploitation (GA and local optimization)
            progress = iteration / self.max_iterations
            
            if progress < 0.3:
                # Early phase: focus on exploration with simulated annealing
                strategy = "SA"
            elif progress < 0.7:
                # Middle phase: genetic algorithm with occasional SA
                strategy = "GA" if random.random() < 0.7 else "SA"
            else:
                # Late phase: focused optimization
                strategy = "GA"
            
            # Execute selected strategy
            if strategy == "SA":
                # Simulated annealing for diverse exploration
                for i, (pose, score) in enumerate(population[:10]):  # Apply SA to top individuals
                    new_pose, new_score, accepted = self._simulated_annealing_step(
                        pose, protein, center, radius, temperature)
                    
                    if accepted:
                        population[i] = (new_pose, new_score)
                        
                        # Update best solution if improved
                        if new_score < self.best_score:
                            self.best_pose = copy.deepcopy(new_pose)
                            self.best_score = new_score
                            all_solutions.append((copy.deepcopy(new_pose), new_score))
                            self.logger.info(f"New best score (SA): {new_score:.4f}")
                
                # Cool temperature
                temperature = max(temperature * self.cooling_factor, self.temperature_end)
                
            elif strategy == "GA":
                # Genetic algorithm for exploitation of good solutions
                # Select parents
                parents = []
                for _ in range(self.population_size):
                    # Tournament selection
                    candidates = random.sample(population, min(self.tournament_size, len(population)))
                    candidates.sort(key=lambda x: x[1])
                    parents.append(candidates[0])
                
                # Create offspring through crossover and mutation
                offspring = []
                
                for i in range(0, len(parents), 2):
                    if i + 1 < len(parents):
                        parent1 = parents[i][0]
                        parent2 = parents[i+1][0]
                        
                        # Crossover with probability
                        if random.random() < self.crossover_rate:
                            child1, child2 = self._crossover_pair(parent1, parent2)
                        else:
                            child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                        
                        # Mutation
                        self._mutate(child1, copy.deepcopy(parent1), center, radius)
                        self._mutate(child2, copy.deepcopy(parent2), center, radius)
                        
                        # Ensure validity and score
                        if self._check_pose_validity(child1, protein):
                            try:
                                child1_score = self.scoring_function.score(protein, child1)
                                offspring.append((child1, child1_score))
                            except Exception as e:
                                self.logger.debug(f"Error scoring child1: {e}")
                                
                        if self._check_pose_validity(child2, protein):
                            try:
                                child2_score = self.scoring_function.score(protein, child2)
                                offspring.append((child2, child2_score))
                            except Exception as e:
                                self.logger.debug(f"Error scoring child2: {e}")
                
                # Combine parent and offspring populations and select best
                combined = population + offspring
                combined.sort(key=lambda x: x[1])
                population = combined[:self.population_size]
                
                # Update best solution if improved
                if population[0][1] < self.best_score:
                    self.best_pose = copy.deepcopy(population[0][0])
                    self.best_score = population[0][1]
                    all_solutions.append((copy.deepcopy(population[0][0]), population[0][1]))
                    self.logger.info(f"New best score (GA): {self.best_score:.4f}")
            
            # Periodically apply local optimization to best solution
            if iteration % self.local_opt_frequency == 0:
                try:
                    opt_pose, opt_score = self._local_optimization(
                        population[0][0], protein, step_size=0.2, angle_step=0.05, max_steps=10)
                    
                    if opt_score < population[0][1]:
                        population[0] = (opt_pose, opt_score)
                        # Re-sort population
                        population.sort(key=lambda x: x[1])
                        
                        # Update best solution if improved
                        if opt_score < self.best_score:
                            self.best_pose = copy.deepcopy(opt_pose)
                            self.best_score = opt_score
                            all_solutions.append((copy.deepcopy(opt_pose), opt_score))
                            self.logger.info(f"New best score (local opt): {opt_score:.4f}")
                except Exception as e:
                    self.logger.warning(f"Error during local optimization: {e}")
        
        if progress_bar:
            progress_bar.close()
            
        # Final cleanup and sorting
        self.total_time = time.time() - start_time
        self.logger.info(f"Search completed in {self.total_time:.2f} seconds")
        self.logger.info(f"Best score: {self.best_score:.4f}")
        
        # Filter duplicate poses based on RMSD
        unique_solutions = []
        for pose, score in sorted(all_solutions, key=lambda x: x[1]):
            # Skip poses with bad scores
            if not np.isfinite(score):
                continue
                
            # Check if this pose is already in our unique set
            is_duplicate = False
            for unique_pose, _ in unique_solutions:
                try:
                    rmsd = calculate_rmsd(pose.xyz, unique_pose.xyz)
                    if rmsd < 2.0:  # RMSD threshold for considering poses as duplicates
                        is_duplicate = True
                        break
                except Exception:
                    # If RMSD calculation fails, assume it's not a duplicate
                    pass
                    
            if not is_duplicate:
                unique_solutions.append((pose, score))
        
        # Save top pose to PDB if output directory is specified
        if self.output_dir and len(unique_solutions) > 0:
            top_pose = unique_solutions[0][0]
            top_score = unique_solutions[0][1]
            
            output_path = Path(self.output_dir) / f"top_pose_{top_score:.2f}.pdb"
            try:
                top_pose.write_pdb(output_path)
                self.logger.info(f"Saved top pose to {output_path}")
            except Exception as e:
                self.logger.error(f"Error saving top pose: {e}")
        
        return unique_solutions
    
    def _crossover_pair(self, parent1, parent2):
        """
        Perform crossover between two parents.
        
        Parameters:
        -----------
        parent1 : Ligand
            First parent
        parent2 : Ligand
            Second parent
        
        Returns:
        --------
        tuple
            (child1, child2) as Ligand objects
        """
        # Create deep copies to avoid modifying parents
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # Calculate centroids
        centroid1 = np.mean(parent1.xyz, axis=0)
        centroid2 = np.mean(parent2.xyz, axis=0)
        
        # Weighted centroid crossover
        alpha = random.uniform(0.3, 0.7)  # Random weight for variability
        new_centroid1 = alpha * centroid1 + (1 - alpha) * centroid2
        new_centroid2 = (1 - alpha) * centroid1 + alpha * centroid2
        
        # Apply translation to children
        child1.translate(new_centroid1 - centroid1)
        child2.translate(new_centroid2 - centroid2)
        
        # Fragment-based crossover
        fragment_indices = random.sample(range(len(parent1.xyz)), len(parent1.xyz) // 2)
        for idx in fragment_indices:
            child1.xyz[idx], child2.xyz[idx] = child2.xyz[idx], child1.xyz[idx]
            
            # Update atom coordinates in the atoms list as well
            if hasattr(child1, 'atoms') and hasattr(child2, 'atoms'):
                child1.atoms[idx]['coords'], child2.atoms[idx]['coords'] = child2.atoms[idx]['coords'], child1.atoms[idx]['coords']
        
        # Rotation interpolation
        rotation1 = Rotation.random()
        rotation2 = Rotation.random()
        key_times = [0, 1]
        rotations = Rotation.concatenate([rotation1, rotation2])
        slerp = Slerp(key_times, rotations)
        interpolated_rotation = slerp([alpha])[0]  # Interpolate at alpha
        
        # Apply interpolated rotation to children
        centroid1 = np.mean(child1.xyz, axis=0)
        centroid2 = np.mean(child2.xyz, axis=0)
        
        child1.translate(-centroid1)
        child1.rotate(interpolated_rotation.as_matrix())
        child1.translate(centroid1)
        
        child2.translate(-centroid2)
        child2.rotate(interpolated_rotation.as_matrix())
        child2.translate(centroid2)
        
        return child1, child2

    def _mutate(self, individual, original_individual, center, radius):
        """
        Mutate an individual by applying random changes.
        
        Parameters:
        -----------
        individual : Ligand
            Individual to mutate
        original_individual : Ligand
            Original individual for backup
        center : numpy.ndarray
            Center coordinates
        radius : float
            Search radius
            
        Returns:
        --------
        Ligand
            Mutated individual
        """
        if random.random() >= self.mutation_rate:
            return individual  # No mutation
        
        # Store original in case mutation fails
        backup = copy.deepcopy(individual)
        
        # Choose mutation type
        mutation_type = random.choice(['translation', 'rotation', 'both'])
        
        if mutation_type == 'translation' or mutation_type == 'both':
            # Random translation
            translation = np.random.normal(0, 0.5, 3)
            individual.translate(translation)
            
        if mutation_type == 'rotation' or mutation_type == 'both':
            # Random rotation
            angle = np.random.normal(0, 0.2)
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            
            rotation = Rotation.from_rotvec(axis * angle)
            centroid = np.mean(individual.xyz, axis=0)
            individual.translate(-centroid)
            individual.rotate(rotation.as_matrix())
            individual.translate(centroid)
        
        # Ensure the individual is still inside the sphere
        if not is_inside_sphere(individual, center, radius):
            # Try to reposition
            if not reposition_inside_sphere(individual, center, radius):
                # Revert to original if repositioning failed
                individual.xyz = backup.xyz.copy()
                
                # Update atom coordinates in atoms list
                if hasattr(individual, 'atoms') and hasattr(backup, 'atoms'):
                    for i in range(len(individual.atoms)):
                        individual.atoms[i]['coords'] = backup.atoms[i]['coords']
        
        return individual


# ------------------------------------------------------------------------------
# Flexible Ligand Docking Support
# ------------------------------------------------------------------------------

class FlexibleLigandSearch(HybridSearch):
    """
    Search algorithm for flexible ligand docking.
    
    This class extends the HybridSearch to support ligand flexibility
    by sampling different conformations during the search process.
    """
    
    def __init__(self, scoring_function, max_iterations=300, population_size=30,
                 n_processes=None, output_dir=None, grid_spacing=0.375,
                 grid_radius=10.0, grid_center=None, temperature_start=5.0,
                 temperature_end=0.1, cooling_factor=0.95, mutation_rate=0.3,
                 crossover_rate=0.7, local_opt_frequency=5, max_torsions=10,
                 torsion_step=15.0):
        """
        Initialize flexible ligand search.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function to evaluate poses
        max_iterations : int
            Maximum number of iterations
        population_size : int
            Size of the population
        n_processes : int
            Number of parallel processes (None = use all available cores)
        output_dir : str or Path
            Output directory
        grid_spacing : float
            Spacing between grid points
        grid_radius : float
            Radius of the search sphere
        grid_center : array-like
            Center coordinates of the search sphere
        temperature_start : float
            Starting temperature for simulated annealing
        temperature_end : float
            Ending temperature for simulated annealing
        cooling_factor : float
            Cooling factor for simulated annealing
        mutation_rate : float
            Probability of mutation in genetic algorithm phase
        crossover_rate : float
            Probability of crossover in genetic algorithm phase
        local_opt_frequency : int
            Frequency of local optimization (every N iterations)
        max_torsions : int
            Maximum number of torsions to sample
        torsion_step : float
            Step size for torsion sampling (degrees)
        """
        super().__init__(
            scoring_function=scoring_function,
            max_iterations=max_iterations,
            population_size=population_size,
            n_processes=n_processes,
            output_dir=output_dir,
            grid_spacing=grid_spacing,
            grid_radius=grid_radius,
            grid_center=grid_center,
            temperature_start=temperature_start,
            temperature_end=temperature_end,
            cooling_factor=cooling_factor,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            local_opt_frequency=local_opt_frequency
        )
        
        # Flexible ligand parameters
        self.max_torsions = max_torsions
        self.torsion_step = torsion_step
    
    def _identify_rotatable_bonds(self, ligand):
        """
        Identify rotatable bonds in the ligand.
        
        Parameters:
        -----------
        ligand : Ligand
            Ligand object
            
        Returns:
        --------
        list
            List of rotatable bond indices
        """
        rotatable_bonds = []
        
        # Check if the ligand has rotatable bonds defined
        if hasattr(ligand, 'rotatable_bonds'):
            return ligand.rotatable_bonds
        
        # Otherwise, identify potentially rotatable bonds
        for i, bond in enumerate(ligand.bonds):
            # Skip bonds in rings
            if bond.get('in_ring', False):
                continue
                
            # Skip terminal bonds (to hydrogens, etc.)
            begin_atom = ligand.atoms[bond['begin_atom_idx']]
            end_atom = ligand.atoms[bond['end_atom_idx']]
            
            begin_symbol = begin_atom.get('symbol', begin_atom.get('element', 'C'))
            end_symbol = end_atom.get('symbol', end_atom.get('element', 'C'))
            
            # Skip bonds to hydrogen
            if begin_symbol == 'H' or end_symbol == 'H':
                continue
                
            # Skip non-single bonds
            if bond.get('bond_order', 1) != 1:
                continue
            
            rotatable_bonds.append(i)
        
        # Limit the number of rotatable bonds to consider
        if len(rotatable_bonds) > self.max_torsions:
            self.logger.warning(f"Limiting rotatable bonds from {len(rotatable_bonds)} to {self.max_torsions}")
            rotatable_bonds = rotatable_bonds[:self.max_torsions]
            
        return rotatable_bonds
    
    def _rotate_bond(self, ligand, bond_idx, angle_degrees):
        """
        Rotate a bond in the ligand by the specified angle.
        
        Parameters:
        -----------
        ligand : Ligand
            Ligand object
        bond_idx : int
            Index of the bond to rotate
        angle_degrees : float
            Rotation angle in degrees
            
        Returns:
        --------
        Ligand
            Ligand with rotated bond
        """
        # Make a copy of the ligand
        new_ligand = copy.deepcopy(ligand)
        
        # Get the bond
        bond = new_ligand.bonds[bond_idx]
        
        # Get atom indices for the bond
        begin_idx = bond['begin_atom_idx']
        end_idx = bond['end_atom_idx']
        
        # Get atom coordinates
        begin_coords = new_ligand.xyz[begin_idx]
        end_coords = new_ligand.xyz[end_idx]
        
        # Calculate bond vector
        bond_vector = end_coords - begin_coords
        bond_vector = bond_vector / np.linalg.norm(bond_vector)
        
        # Convert angle to radians
        angle_radians = np.radians(angle_degrees)
        
        # Create rotation matrix around bond vector
        rotation = Rotation.from_rotvec(bond_vector * angle_radians)
        rotation_matrix = rotation.as_matrix()
        
        # Identify atoms on the end side of the bond
        # This is a simplified approach - a more robust implementation would use
        # a graph traversal algorithm to identify connected components
        
        # Translate ligand so that begin_atom is at origin
        new_ligand.translate(-begin_coords)
        
        # Find atoms on the "end" side of the bond
        # We'll use a simple heuristic: atoms closer to end_atom than begin_atom
        end_side_atoms = []
        for i, atom_coords in enumerate(new_ligand.xyz):
            if i == begin_idx or i == end_idx:
                continue
                
            # Calculate distances to begin and end atoms
            dist_to_begin = np.linalg.norm(atom_coords)
            dist_to_end = np.linalg.norm(atom_coords - (end_coords - begin_coords))
            
            # If closer to end atom, add to list
            if dist_to_end < dist_to_begin:
                end_side_atoms.append(i)
        
        # Rotate atoms on end side
        for i in end_side_atoms:
            # Rotate atom
            new_ligand.xyz[i] = np.dot(rotation_matrix, new_ligand.xyz[i])
            
            # Update atom coordinates in atoms list
            if hasattr(new_ligand, 'atoms'):
                new_ligand.atoms[i]['coords'] = new_ligand.xyz[i]
        
        # Rotate end atom
        end_coords_origin = end_coords - begin_coords
        new_ligand.xyz[end_idx] = np.dot(rotation_matrix, end_coords_origin)
        
        # Update atom coordinates in atoms list
        if hasattr(new_ligand, 'atoms'):
            new_ligand.atoms[end_idx]['coords'] = new_ligand.xyz[end_idx]
        
        # Translate ligand back
        new_ligand.translate(begin_coords)
        
        return new_ligand
    
    def _generate_conformer(self, ligand, rotatable_bonds=None):
        """
        Generate a new conformer by rotating random bonds.
        
        Parameters:
        -----------
        ligand : Ligand
            Ligand object
        rotatable_bonds : list, optional
            List of rotatable bond indices
            
        Returns:
        --------
        Ligand
            New ligand conformer
        """
        # Identify rotatable bonds if not provided
        if rotatable_bonds is None:
            rotatable_bonds = self._identify_rotatable_bonds(ligand)
            
        # If no rotatable bonds, return copy of original ligand
        if not rotatable_bonds:
            return copy.deepcopy(ligand)
            
        # Create new conformer
        conformer = copy.deepcopy(ligand)
        
        # Randomly select bonds to rotate
        n_bonds = random.randint(1, min(3, len(rotatable_bonds)))
        bonds_to_rotate = random.sample(rotatable_bonds, n_bonds)
        
        # Rotate each selected bond
        for bond_idx in bonds_to_rotate:
            # Random rotation angle
            angle = random.uniform(-self.torsion_step, self.torsion_step)
            
            # Rotate bond
            conformer = self._rotate_bond(conformer, bond_idx, angle)
            
            # Validate conformation
            if not self._validate_conformation(conformer):
                # If invalid, try to repair
                conformer = self._repair_conformation(conformer)
        
        return conformer
    
    def _generate_valid_pose(self, protein, ligand, center, radius, max_attempts=50):
        """
        Generate a valid ligand pose with flexibility.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand template
        center : numpy.ndarray
            Center coordinates of the search space
        radius : float
            Radius of the search sphere
        max_attempts : int
            Maximum number of generation attempts
            
        Returns:
        --------
        Ligand
            Valid ligand pose
        """
        # Identify rotatable bonds
        rotatable_bonds = self._identify_rotatable_bonds(ligand)
        
        # Generate a valid rigid pose first
        rigid_pose = super()._generate_valid_pose(protein, ligand, center, radius, max_attempts)
        
        # If no rotatable bonds or rigid pose generation failed, return rigid pose
        if not rotatable_bonds or rigid_pose is None:
            return rigid_pose
            
        # Generate a set of conformers and find the best one
        best_pose = rigid_pose
        best_score = float('inf')
        
        try:
            best_score = self.scoring_function.score(protein, best_pose)
        except Exception as e:
            self.logger.warning(f"Error scoring initial pose: {e}")
            return rigid_pose
            
        # Try different conformers
        for _ in range(5):  # Generate a few conformers to try
            conformer = self._generate_conformer(rigid_pose, rotatable_bonds)
            
            # Check if conformer is valid
            if not self._check_pose_validity(conformer, protein):
                continue
                
            # Score conformer
            try:
                score = self.scoring_function.score(protein, conformer)
                
                # Update best pose if improved
                if score < best_score:
                    best_pose = conformer
                    best_score = score
            except Exception as e:
                self.logger.debug(f"Error scoring conformer: {e}")
        
        return best_pose
    
    def _mutate(self, individual, original_individual, center, radius):
        """
        Enhanced mutation with conformational changes.
        
        Parameters:
        -----------
        individual : Ligand
            Individual to mutate
        original_individual : Ligand
            Original individual (backup)
        center : numpy.ndarray
            Center of the search space
        radius : float
            Radius of the search space
            
        Returns:
        --------
        Ligand
            Mutated individual
        """
        # First apply standard rigid body mutation
        super()._mutate(individual, original_individual, center, radius)
        
        # Then apply conformational mutation with some probability
        if random.random() < self.mutation_rate * 0.5:  # Reduced probability for conformation changes
            rotatable_bonds = self._identify_rotatable_bonds(individual)
            
            if rotatable_bonds:
                # Store original in case mutation fails
                backup = copy.deepcopy(individual)
                
                # Apply conformational changes
                mutated = self._generate_conformer(individual, rotatable_bonds)
                
                # Check validity
                if self._validate_conformation(mutated) and self._check_pose_validity(mutated, self.protein):
                    # Update individual with mutated conformer
                    individual.xyz = mutated.xyz.copy()
                    
                    # Update atom coordinates in atoms list
                    if hasattr(individual, 'atoms') and hasattr(mutated, 'atoms'):
                        for i in range(len(individual.atoms)):
                            individual.atoms[i]['coords'] = mutated.atoms[i]['coords']
                else:
                    # Revert to backup if mutation failed
                    individual.xyz = backup.xyz.copy()
                    
                    # Update atom coordinates in atoms list
                    if hasattr(individual, 'atoms') and hasattr(backup, 'atoms'):
                        for i in range(len(individual.atoms)):
                            individual.atoms[i]['coords'] = backup.atoms[i]['coords']
        
        return individual
    
    def _crossover_pair(self, parent1, parent2):
        """
        Enhanced crossover with conformational exchange.
        
        Parameters:
        -----------
        parent1 : Ligand
            First parent
        parent2 : Ligand
            Second parent
        
        Returns:
        --------
        tuple
            (child1, child2) as Ligand objects
        """
        # Apply standard rigid body crossover
        child1, child2 = super()._crossover_pair(parent1, parent2)
        
        # Then apply conformational crossover with some probability
        if random.random() < self.crossover_rate * 0.5:  # Reduced probability for conformation changes
            rotatable_bonds = self._identify_rotatable_bonds(parent1)
            
            if rotatable_bonds:
                # Choose a subset of rotatable bonds to exchange
                n_bonds = min(len(rotatable_bonds), 3)
                bond_indices = random.sample(rotatable_bonds, n_bonds)
                
                # For each selected bond, exchange torsion angles between children
                for bond_idx in bond_indices:
                    # Get bond
                    bond1 = child1.bonds[bond_idx]
                    bond2 = child2.bonds[bond_idx]
                    
                    # Get atom indices
                    begin_idx1 = bond1['begin_atom_idx']
                    end_idx1 = bond1['end_atom_idx']
                    begin_idx2 = bond2['begin_atom_idx']
                    end_idx2 = bond2['end_atom_idx']
                    
                    # Get atom coordinates
                    begin_coords1 = child1.xyz[begin_idx1]
                    end_coords1 = child1.xyz[end_idx1]
                    begin_coords2 = child2.xyz[begin_idx2]
                    end_coords2 = child2.xyz[end_idx2]
                    
                    # Calculate bond vectors
                    bond_vector1 = end_coords1 - begin_coords1
                    bond_vector1 = bond_vector1 / np.linalg.norm(bond_vector1)
                    bond_vector2 = end_coords2 - begin_coords2
                    bond_vector2 = bond_vector2 / np.linalg.norm(bond_vector2)
                    
                    # Calculate angle between bond vectors
                    dot_product = np.clip(np.dot(bond_vector1, bond_vector2), -1.0, 1.0)
                    angle = np.arccos(dot_product)
                    
                    # Calculate cross product to determine rotation direction
                    cross_product = np.cross(bond_vector1, bond_vector2)
                    if np.linalg.norm(cross_product) > 0:
                        cross_product = cross_product / np.linalg.norm(cross_product)
                    
                    # Determine rotation axis and angle
                    if angle > 0.001:  # Avoid rotation if bonds are already aligned
                        # Rotate child1 to match child2 bond orientation
                        child1 = self._rotate_bond(child1, bond_idx, np.degrees(angle))
                        
                        # Validate child1 after rotation
                        if not self._validate_conformation(child1):
                            child1 = self._repair_conformation(child1)
                
                # Ensure both children are valid after conformational changes
                child1 = enforce_sphere_boundary(child1, self.center, self.radius)
                child2 = enforce_sphere_boundary(child2, self.center, self.radius)
                
                if not self._validate_conformation(child1):
                    child1 = self._repair_conformation(child1)
                    
                if not self._validate_conformation(child2):
                    child2 = self._repair_conformation(child2)
        
        return child1, child2
    
    def _local_optimization(self, pose, protein, step_size=0.3, angle_step=0.05, max_steps=15):
        """
        Enhanced local optimization with torsion angle optimization.
        
        Parameters:
        -----------
        pose : Ligand
            Ligand pose to optimize
        protein : Protein
            Protein target
        step_size : float
            Maximum translation step size (Å)
        angle_step : float
            Maximum rotation step size (radians)
        max_steps : int
            Maximum number of optimization steps
        
        Returns:
        --------
        tuple
            (optimized_pose, optimized_score)
        """
        # First apply standard rigid body optimization
        opt_pose, opt_score = super()._local_optimization(pose, protein, step_size, angle_step, max_steps)
        
        # Then apply torsion angle optimization
        rotatable_bonds = self._identify_rotatable_bonds(opt_pose)
        
        if not rotatable_bonds:
            return opt_pose, opt_score
            
        # Optimize each rotatable bond individually
        for bond_idx in rotatable_bonds:
            # Try different rotation angles
            angles = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]
            
            best_angle = 0.0
            best_bond_score = opt_score
            
            for angle in angles:
                # Skip zero angle (no rotation)
                if angle == 0.0:
                    continue
                    
                # Create test pose
                test_pose = self._rotate_bond(opt_pose, bond_idx, angle)
                
                # Check validity
                if not self._check_pose_validity(test_pose, protein):
                    continue
                    
                # Score test pose
                try:
                    test_score = self.scoring_function.score(protein, test_pose)
                    
                    # Update best angle if improved
                    if test_score < best_bond_score:
                        best_angle = angle
                        best_bond_score = test_score
                except Exception as e:
                    self.logger.debug(f"Error scoring test pose: {e}")
            
            # Apply best rotation if it improved the score
            if best_angle != 0.0 and best_bond_score < opt_score:
                opt_pose = self._rotate_bond(opt_pose, bond_idx, best_angle)
                opt_score = best_bond_score
        
        return opt_pose, opt_score


# ------------------------------------------------------------------------------
# Main Factory Function
# ------------------------------------------------------------------------------

def create_search_algorithm(algorithm_name, scoring_function, **kwargs):
    """
    Factory function to create search algorithm instances.
    
    Parameters:
    -----------
    algorithm_name : str
        Name of the search algorithm
    scoring_function : ScoringFunction
        Scoring function to use for pose evaluation
    **kwargs : dict
        Additional parameters for the search algorithm
        
    Returns:
    --------
    object
        Instance of the requested search algorithm
    """
    algorithms = {
        'parallel': ParallelSearch,
        'genetic': ParallelGeneticAlgorithm,
        'random': ParallelRandomSearch,
        'hybrid': HybridSearch,
        'flexible': FlexibleLigandSearch
    }
    
    if algorithm_name not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Available algorithms: {', '.join(algorithms.keys())}")
    
    return algorithms[algorithm_name](scoring_function, **kwargs)