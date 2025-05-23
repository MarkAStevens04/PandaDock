"""
Metal-Based Docking Framework for PandaDock
This module implements metal coordination constraints and specialized scoring for metal-containing ligands.
"""

import numpy as np
import copy
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import json

# Metal coordination geometry definitions
METAL_COORDINATION_GEOMETRIES = {
    'octahedral': {
        'coordination_number': 6,
        'ideal_angles': [90.0, 180.0],  # cis and trans angles
        'tolerance': 15.0,
        'metals': ['Fe', 'Co', 'Ni', 'Cr', 'Mn', 'Mo', 'W', 'Ru', 'Os', 'Rh', 'Ir', 'Pd', 'Pt']
    },
    'tetrahedral': {
        'coordination_number': 4,
        'ideal_angles': [109.47],  # tetrahedral angle
        'tolerance': 15.0,
        'metals': ['Zn', 'Cd', 'Hg', 'Cu', 'Ag', 'Au', 'Be', 'Mg', 'Al', 'Ga', 'In']
    },
    'square_planar': {
        'coordination_number': 4,
        'ideal_angles': [90.0, 180.0],  # cis and trans angles in plane
        'tolerance': 15.0,
        'metals': ['Pt', 'Pd', 'Ni', 'Cu', 'Au']
    },
    'trigonal_bipyramidal': {
        'coordination_number': 5,
        'ideal_angles': [90.0, 120.0, 180.0],
        'tolerance': 15.0,
        'metals': ['Fe', 'Co', 'Ni', 'Cu', 'Zn']
    },
    'linear': {
        'coordination_number': 2,
        'ideal_angles': [180.0],
        'tolerance': 10.0,
        'metals': ['Ag', 'Au', 'Cu', 'Hg']
    }
}

# Metal-ligand bond lengths (Angstroms)
METAL_BOND_LENGTHS = {
    'Fe': {'N': 2.0, 'O': 1.9, 'S': 2.3, 'C': 2.0, 'P': 2.4},
    'Zn': {'N': 2.1, 'O': 2.0, 'S': 2.3, 'C': 2.0, 'P': 2.4},
    'Cu': {'N': 2.0, 'O': 1.9, 'S': 2.3, 'C': 1.9, 'P': 2.3},
    'Ni': {'N': 2.0, 'O': 2.0, 'S': 2.3, 'C': 1.9, 'P': 2.3},
    'Co': {'N': 2.0, 'O': 2.0, 'S': 2.3, 'C': 1.9, 'P': 2.3},
    'Mn': {'N': 2.2, 'O': 2.1, 'S': 2.4, 'C': 2.1, 'P': 2.4},
    'Mg': {'N': 2.1, 'O': 2.0, 'S': 2.5, 'C': 2.2, 'P': 2.5},
    'Ca': {'N': 2.4, 'O': 2.3, 'S': 2.8, 'C': 2.5, 'P': 2.8},
    'Pt': {'N': 2.0, 'O': 2.0, 'S': 2.3, 'C': 2.0, 'P': 2.3},
    'Pd': {'N': 2.0, 'O': 2.0, 'S': 2.3, 'C': 2.0, 'P': 2.3}
}

class MetalCenter:
    """
    Represents a metal center in a protein or ligand.
    """
    
    def __init__(self, metal_atom, coordination_geometry='auto', max_coordination=6):
        """
        Initialize metal center.
        
        Parameters:
        -----------
        metal_atom : dict
            Metal atom information with 'coords', 'element', etc.
        coordination_geometry : str
            Expected coordination geometry ('auto' for automatic detection)
        max_coordination : int
            Maximum coordination number to consider
        """
        self.metal_atom = metal_atom
        self.element = metal_atom.get('element', metal_atom.get('name', ''))[:2].strip()
        self.coords = np.array(metal_atom['coords'])
        self.coordination_geometry = coordination_geometry
        self.max_coordination = max_coordination
        
        # Coordination environment
        self.coordinating_atoms = []
        self.coordination_bonds = []
        self.coordination_number = 0
        
        # Determine preferred geometry
        if coordination_geometry == 'auto':
            self.preferred_geometry = self._determine_preferred_geometry()
        else:
            self.preferred_geometry = coordination_geometry
            
        self.logger = logging.getLogger(f"MetalCenter_{self.element}")
    
    def _determine_preferred_geometry(self):
        """Determine preferred coordination geometry based on metal type."""
        for geometry, info in METAL_COORDINATION_GEOMETRIES.items():
            if self.element in info['metals']:
                return geometry
        return 'octahedral'  # Default fallback
    
    def add_coordinating_atom(self, atom, bond_type='coordinate'):
        """
        Add a coordinating atom to this metal center.
        
        Parameters:
        -----------
        atom : dict
            Coordinating atom information
        bond_type : str
            Type of coordination bond
        """
        atom_coords = np.array(atom['coords'])
        distance = np.linalg.norm(atom_coords - self.coords)
        
        coordination_info = {
            'atom': atom,
            'coords': atom_coords,
            'distance': distance,
            'bond_type': bond_type,
            'element': atom.get('element', atom.get('name', ''))[:1]
        }
        
        self.coordinating_atoms.append(coordination_info)
        self.coordination_number = len(self.coordinating_atoms)
    
    def get_coordination_score(self):
        """
        Calculate how well the current coordination matches the preferred geometry.
        
        Returns:
        --------
        float
            Coordination score (lower is better, 0 is perfect)
        """
        if self.coordination_number == 0:
            return 1000.0  # Very bad score for no coordination
            
        geometry_info = METAL_COORDINATION_GEOMETRIES.get(self.preferred_geometry, {})
        ideal_coordination = geometry_info.get('coordination_number', 6)
        ideal_angles = geometry_info.get('ideal_angles', [90.0])
        tolerance = geometry_info.get('tolerance', 15.0)
        
        score = 0.0
        
        # Coordination number penalty
        coord_diff = abs(self.coordination_number - ideal_coordination)
        score += coord_diff * 10.0
        
        # Bond length penalties
        for coord_info in self.coordinating_atoms:
            element = coord_info['element']
            distance = coord_info['distance']
            
            if self.element in METAL_BOND_LENGTHS and element in METAL_BOND_LENGTHS[self.element]:
                ideal_distance = METAL_BOND_LENGTHS[self.element][element]
                distance_penalty = abs(distance - ideal_distance) * 5.0
                score += distance_penalty
            else:
                # Generic penalty for unknown metal-ligand combinations
                if distance < 1.5 or distance > 3.0:
                    score += 20.0
        
        # Angular penalties (only if we have enough atoms)
        if self.coordination_number >= 2:
            angles = self._calculate_coordination_angles()
            angle_score = self._score_coordination_angles(angles, ideal_angles, tolerance)
            score += angle_score
            
        return score
    
    def _calculate_coordination_angles(self):
        """Calculate angles between coordinating atoms."""
        if self.coordination_number < 2:
            return []
            
        angles = []
        coords_list = [info['coords'] for info in self.coordinating_atoms]
        
        for i in range(len(coords_list)):
            for j in range(i + 1, len(coords_list)):
                # Vector from metal to each coordinating atom
                vec1 = coords_list[i] - self.coords
                vec2 = coords_list[j] - self.coords
                
                # Calculate angle
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))
                angles.append(angle)
                
        return angles
    
    def _score_coordination_angles(self, actual_angles, ideal_angles, tolerance):
        """Score how well actual angles match ideal geometry."""
        if not actual_angles:
            return 0.0
            
        score = 0.0
        
        for actual_angle in actual_angles:
            # Find closest ideal angle
            min_deviation = float('inf')
            for ideal_angle in ideal_angles:
                deviation = abs(actual_angle - ideal_angle)
                min_deviation = min(min_deviation, deviation)
            
            # Apply penalty based on deviation
            if min_deviation > tolerance:
                score += (min_deviation - tolerance) * 2.0
                
        return score


class MetalConstraint:
    """
    Represents a constraint for metal coordination during docking.
    """
    
    def __init__(self, metal_center, constraint_type='coordination', strength=1.0, 
                 target_atoms=None, required_coordination=None):
        """
        Initialize metal constraint.
        
        Parameters:
        -----------
        metal_center : MetalCenter
            The metal center to constrain
        constraint_type : str
            Type of constraint ('coordination', 'distance', 'angle')
        strength : float
            Constraint strength (higher = more important)
        target_atoms : list
            Specific atoms that should coordinate (optional)
        required_coordination : int
            Required coordination number (optional)
        """
        self.metal_center = metal_center
        self.constraint_type = constraint_type
        self.strength = strength
        self.target_atoms = target_atoms or []
        self.required_coordination = required_coordination
        
    def evaluate(self, ligand_pose):
        """
        Evaluate how well the ligand pose satisfies this constraint.
        
        Parameters:
        -----------
        ligand_pose : Ligand
            Current ligand pose
            
        Returns:
        --------
        float
            Constraint violation penalty (0 = satisfied)
        """
        if self.constraint_type == 'coordination':
            return self._evaluate_coordination_constraint(ligand_pose)
        elif self.constraint_type == 'distance':
            return self._evaluate_distance_constraint(ligand_pose)
        elif self.constraint_type == 'angle':
            return self._evaluate_angle_constraint(ligand_pose)
        else:
            return 0.0
    
    def _evaluate_coordination_constraint(self, ligand_pose):
        """Evaluate coordination constraint."""
        # Find potential coordinating atoms in ligand
        coordinating_atoms = self._find_coordinating_atoms(ligand_pose)
        
        # Update metal center with current ligand atoms
        temp_metal_center = copy.deepcopy(self.metal_center)
        for atom in coordinating_atoms:
            temp_metal_center.add_coordinating_atom(atom)
        
        # Get coordination score
        coord_score = temp_metal_center.get_coordination_score()
        
        return coord_score * self.strength
    
    def _find_coordinating_atoms(self, ligand_pose, max_distance=3.5):
        """Find atoms in ligand that could coordinate to metal."""
        coordinating_elements = ['N', 'O', 'S', 'P', 'C']  # Common coordinating elements
        coordinating_atoms = []
        
        for atom in ligand_pose.atoms:
            element = atom.get('element', atom.get('name', ''))[:1]
            if element in coordinating_elements:
                atom_coords = np.array(atom['coords'])
                distance = np.linalg.norm(atom_coords - self.metal_center.coords)
                
                if distance <= max_distance:
                    coordinating_atoms.append(atom)
        
        return coordinating_atoms
    
    def _evaluate_distance_constraint(self, ligand_pose):
        """Evaluate distance constraint."""
        penalty = 0.0
        
        for target_atom_info in self.target_atoms:
            # Find corresponding atom in ligand
            ligand_atom = self._find_ligand_atom(ligand_pose, target_atom_info)
            if ligand_atom:
                distance = np.linalg.norm(
                    np.array(ligand_atom['coords']) - self.metal_center.coords
                )
                ideal_distance = target_atom_info.get('ideal_distance', 2.0)
                tolerance = target_atom_info.get('tolerance', 0.3)
                
                if abs(distance - ideal_distance) > tolerance:
                    penalty += abs(distance - ideal_distance) * self.strength
        
        return penalty
    
    def _evaluate_angle_constraint(self, ligand_pose):
        """Evaluate angle constraint."""
        # Implementation for specific angle constraints
        return 0.0
    
    def _find_ligand_atom(self, ligand_pose, target_info):
        """Find specific atom in ligand based on target information."""
        # Simple implementation - can be enhanced for more specific matching
        element = target_info.get('element', '')
        atom_name = target_info.get('name', '')
        
        for atom in ligand_pose.atoms:
            if element and atom.get('element', '').startswith(element):
                return atom
            if atom_name and atom.get('name', '') == atom_name:
                return atom
        
        return None


class MetalDockingScorer:
    """
    Specialized scoring function for metal-containing systems.
    """
    
    def __init__(self, base_scoring_function, metal_centers=None, metal_constraints=None, 
                 metal_weight=10.0):
        """
        Initialize metal docking scorer.
        
        Parameters:
        -----------
        base_scoring_function : ScoringFunction
            Base scoring function to use
        metal_centers : list
            List of MetalCenter objects
        metal_constraints : list
            List of MetalConstraint objects
        metal_weight : float
            Weight for metal coordination terms
        """
        self.base_scoring_function = base_scoring_function
        self.metal_centers = metal_centers or []
        self.metal_constraints = metal_constraints or []
        self.metal_weight = metal_weight
        self.logger = logging.getLogger("MetalDockingScorer")
    
    def score(self, protein, ligand):
        """
        Score ligand pose including metal coordination terms.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
            
        Returns:
        --------
        float
            Total score including metal terms
        """
        # Get base score
        base_score = self.base_scoring_function.score(protein, ligand)
        
        # Calculate metal coordination score
        metal_score = self._calculate_metal_score(ligand)
        
        # Combine scores
        total_score = base_score + (metal_score * self.metal_weight)
        
        self.logger.debug(f"Base score: {base_score:.2f}, Metal score: {metal_score:.2f}, Total: {total_score:.2f}")
        
        return total_score
    
    def _calculate_metal_score(self, ligand):
        """Calculate metal coordination contribution to score."""
        total_metal_score = 0.0
        
        # Evaluate each metal constraint
        for constraint in self.metal_constraints:
            constraint_penalty = constraint.evaluate(ligand)
            total_metal_score += constraint_penalty
        
        # Add coordination bonuses for well-coordinated metals
        for metal_center in self.metal_centers:
            # Find coordinating atoms in ligand
            coordinating_atoms = self._find_coordinating_atoms(ligand, metal_center)
            
            # Update metal center temporarily
            temp_center = copy.deepcopy(metal_center)
            for atom in coordinating_atoms:
                temp_center.add_coordinating_atom(atom)
            
            # Get coordination score (penalty)
            coord_penalty = temp_center.get_coordination_score()
            total_metal_score += coord_penalty
        
        return total_metal_score
    
    def _find_coordinating_atoms(self, ligand, metal_center, max_distance=3.5):
        """Find potential coordinating atoms."""
        coordinating_elements = ['N', 'O', 'S', 'P', 'C']
        coordinating_atoms = []
        
        for atom in ligand.atoms:
            element = atom.get('element', atom.get('name', ''))[:1]
            if element in coordinating_elements:
                distance = np.linalg.norm(
                    np.array(atom['coords']) - metal_center.coords
                )
                if distance <= max_distance:
                    coordinating_atoms.append(atom)
        
        return coordinating_atoms

    def score_batch(self, protein, ligands):
        """Batch scoring for multiple ligands."""
        scores = []
        for ligand in ligands:
            score = self.score(protein, ligand)
            scores.append(score)
        return scores


class MetalDockingPreparation:
    """
    Utilities for preparing metal-containing systems for docking.
    """
    
    @staticmethod
    def detect_metal_centers(protein, metal_elements=None):
        """
        Detect metal centers in protein structure.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        metal_elements : list
            List of metal elements to look for
            
        Returns:
        --------
        list
            List of MetalCenter objects
        """
        if metal_elements is None:
            metal_elements = ['FE', 'ZN', 'CU', 'MG', 'CA', 'MN', 'CO', 
                            'NI', 'PT', 'PD', 'MO', 'W', 'CD', 'HG']
        
        metal_centers = []
        
        for atom in protein.atoms:
            element = atom.get('element', atom.get('name', ''))[:2].strip().upper()
            
            if element in metal_elements:
                metal_center = MetalCenter(atom)
                
                # Find existing coordinating atoms in protein
                coordinating_atoms = MetalDockingPreparation._find_protein_coordinating_atoms(
                    protein, metal_center
                )
                
                for coord_atom in coordinating_atoms:
                    metal_center.add_coordinating_atom(coord_atom, 'protein_coordinate')
                
                metal_centers.append(metal_center)
        
        return metal_centers
    
    @staticmethod
    def _find_protein_coordinating_atoms(protein, metal_center, max_distance=3.0):
        """Find atoms in protein already coordinating to metal."""
        coordinating_elements = ['N', 'O', 'S', 'P']
        coordinating_atoms = []
        
        for atom in protein.atoms:
            if atom == metal_center.metal_atom:
                continue
                
            element = atom.get('element', atom.get('name', ''))[:1]
            if element in coordinating_elements:
                distance = np.linalg.norm(
                    np.array(atom['coords']) - metal_center.coords
                )
                if distance <= max_distance:
                    coordinating_atoms.append(atom)
        
        return coordinating_atoms
    
    @staticmethod
    def create_metal_constraints(metal_centers, constraint_types=None, strengths=None):
        """
        Create metal constraints for docking.
        
        Parameters:
        -----------
        metal_centers : list
            List of MetalCenter objects
        constraint_types : list
            Types of constraints to create
        strengths : list
            Constraint strengths
            
        Returns:
        --------
        list
            List of MetalConstraint objects
        """
        if constraint_types is None:
            constraint_types = ['coordination']
        if strengths is None:
            strengths = [1.0] * len(constraint_types)
        
        constraints = []
        
        for metal_center in metal_centers:
            for constraint_type, strength in zip(constraint_types, strengths):
                constraint = MetalConstraint(
                    metal_center=metal_center,
                    constraint_type=constraint_type,
                    strength=strength
                )
                constraints.append(constraint)
        
        return constraints
    
    @staticmethod
    def analyze_ligand_metal_binding_sites(ligand):
        """
        Analyze potential metal binding sites in ligand.
        
        Parameters:
        -----------
        ligand : Ligand
            Ligand object
            
        Returns:
        --------
        dict
            Analysis of potential metal binding sites
        """
        coordinating_elements = ['N', 'O', 'S', 'P']
        binding_sites = []
        
        for i, atom in enumerate(ligand.atoms):
            element = atom.get('element', atom.get('name', ''))[:1]
            if element in coordinating_elements:
                binding_site = {
                    'atom_index': i,
                    'atom': atom,
                    'element': element,
                    'coords': np.array(atom['coords']),
                    'binding_strength': MetalDockingPreparation._estimate_binding_strength(atom)
                }
                binding_sites.append(binding_site)
        
        # Identify potential chelating groups
        chelating_groups = MetalDockingPreparation._identify_chelating_groups(ligand, binding_sites)
        
        return {
            'binding_sites': binding_sites,
            'chelating_groups': chelating_groups,
            'total_coordinating_atoms': len(binding_sites)
        }
    
    @staticmethod
    def _estimate_binding_strength(atom):
        """Estimate relative metal binding strength of an atom."""
        element = atom.get('element', atom.get('name', ''))[:1]
        
        # Simple strength estimates (higher = stronger)
        strength_map = {
            'N': 0.8,  # Nitrogen (amines, imines, etc.)
            'O': 0.6,  # Oxygen (carbonyls, hydroxyls, etc.)
            'S': 0.9,  # Sulfur (thiols, sulfides, etc.)
            'P': 0.7   # Phosphorus
        }
        
        return strength_map.get(element, 0.3)
    
    @staticmethod
    def _identify_chelating_groups(ligand, binding_sites):
        """Identify potential chelating groups (bidentate, tridentate, etc.)."""
        chelating_groups = []
        
        # Simple distance-based chelate identification
        for i, site1 in enumerate(binding_sites):
            for j, site2 in enumerate(binding_sites[i+1:], i+1):
                distance = np.linalg.norm(site1['coords'] - site2['coords'])
                
                # Typical chelate bite distances
                if 2.5 <= distance <= 4.5:
                    chelating_groups.append({
                        'type': 'bidentate',
                        'atoms': [site1['atom_index'], site2['atom_index']],
                        'distance': distance
                    })
        
        return chelating_groups


class MetalDockingSearch:
    """
    Specialized search algorithm for metal-containing ligands.
    """
    
    def __init__(self, base_search_algorithm, metal_centers, metal_constraints):
        """
        Initialize metal docking search.
        
        Parameters:
        -----------
        base_search_algorithm : SearchAlgorithm
            Base search algorithm to use
        metal_centers : list
            List of metal centers
        metal_constraints : list
            List of metal constraints
        """
        self.base_search_algorithm = base_search_algorithm
        self.metal_centers = metal_centers
        self.metal_constraints = metal_constraints
        self.logger = logging.getLogger("MetalDockingSearch")
    
    def search(self, protein, ligand):
        """
        Perform metal-aware docking search.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
            
        Returns:
        --------
        list
            List of (pose, score) tuples optimized for metal coordination
        """
        self.logger.info(f"Starting metal-aware docking with {len(self.metal_centers)} metal centers")
        
        # Perform base search
        base_results = self.base_search_algorithm.search(protein, ligand)
        
        # Apply metal-specific refinement
        refined_results = []
        
        for pose, score in base_results:
            # Refine pose for better metal coordination
            refined_pose = self._refine_metal_coordination(protein, pose)
            
            # Re-score with metal terms
            if hasattr(self.base_search_algorithm, 'scoring_function'):
                if isinstance(self.base_search_algorithm.scoring_function, MetalDockingScorer):
                    refined_score = self.base_search_algorithm.scoring_function.score(protein, refined_pose)
                else:
                    refined_score = score  # Use original score if not metal scorer
            else:
                refined_score = score
            
            refined_results.append((refined_pose, refined_score))
        
        # Sort by score
        refined_results.sort(key=lambda x: x[1])
        
        self.logger.info(f"Metal-aware docking completed. Best score: {refined_results[0][1]:.2f}")
        
        return refined_results
    
    def _refine_metal_coordination(self, protein, pose, max_iterations=20):
        """
        Refine ligand pose to improve metal coordination.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        pose : Ligand
            Ligand pose to refine
        max_iterations : int
            Maximum refinement iterations
            
        Returns:
        --------
        Ligand
            Refined ligand pose
        """
        refined_pose = copy.deepcopy(pose)
        
        for iteration in range(max_iterations):
            improved = False
            
            for metal_center in self.metal_centers:
                # Find coordinating atoms
                coordinating_atoms = self._find_ligand_coordinating_atoms(refined_pose, metal_center)
                
                if coordinating_atoms:
                    # Try to optimize coordination geometry
                    optimized_pose = self._optimize_coordination_geometry(
                        refined_pose, metal_center, coordinating_atoms
                    )
                    
                    # Check if improvement was made
                    if self._is_better_coordination(optimized_pose, refined_pose, metal_center):
                        refined_pose = optimized_pose
                        improved = True
            
            if not improved:
                break
        
        return refined_pose
    
    def _find_ligand_coordinating_atoms(self, ligand, metal_center, max_distance=3.5):
        """Find atoms in ligand that could coordinate to metal."""
        coordinating_elements = ['N', 'O', 'S', 'P', 'C']
        coordinating_atoms = []
        
        for i, atom in enumerate(ligand.atoms):
            element = atom.get('element', atom.get('name', ''))[:1]
            if element in coordinating_elements:
                distance = np.linalg.norm(
                    np.array(atom['coords']) - metal_center.coords
                )
                if distance <= max_distance:
                    coordinating_atoms.append({
                        'index': i,
                        'atom': atom,
                        'distance': distance,
                        'element': element
                    })
        
        return coordinating_atoms
    
    def _optimize_coordination_geometry(self, pose, metal_center, coordinating_atoms):
        """Optimize ligand pose for better metal coordination geometry."""
        if not coordinating_atoms:
            return pose
        
        optimized_pose = copy.deepcopy(pose)
        
        # Simple optimization: adjust distances to ideal values
        for coord_info in coordinating_atoms:
            element = coord_info['element']
            current_distance = coord_info['distance']
            
            if (metal_center.element in METAL_BOND_LENGTHS and 
                element in METAL_BOND_LENGTHS[metal_center.element]):
                
                ideal_distance = METAL_BOND_LENGTHS[metal_center.element][element]
                
                if abs(current_distance - ideal_distance) > 0.2:  # If distance is off
                    # Move ligand to adjust this distance
                    atom_coords = np.array(coord_info['atom']['coords'])
                    metal_coords = metal_center.coords
                    
                    # Vector from metal to atom
                    vector = atom_coords - metal_coords
                    vector_norm = np.linalg.norm(vector)
                    
                    if vector_norm > 0:
                        # Scale vector to ideal distance
                        scaling_factor = ideal_distance / vector_norm
                        adjustment = vector * (scaling_factor - 1.0)
                        
                        # Apply small adjustment to entire ligand
                        optimized_pose.translate(adjustment * 0.1)  # Small step
        
        return optimized_pose
    
    def _is_better_coordination(self, new_pose, old_pose, metal_center):
        """Check if new pose has better metal coordination than old pose."""
        # Calculate coordination scores for both poses
        old_score = self._calculate_coordination_score(old_pose, metal_center)
        new_score = self._calculate_coordination_score(new_pose, metal_center)
        
        return new_score < old_score
    
    def _calculate_coordination_score(self, pose, metal_center):
        """Calculate coordination score for a pose."""
        coordinating_atoms = self._find_ligand_coordinating_atoms(pose, metal_center)
        
        # Create temporary metal center with ligand atoms
        temp_center = copy.deepcopy(metal_center)
        for coord_info in coordinating_atoms:
            temp_center.add_coordinating_atom(coord_info['atom'])
        
        return temp_center.get_coordination_score()
