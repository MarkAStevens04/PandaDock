# unified_scoring.py - Fixed version

import numpy as np
from scipy.spatial.distance import cdist
import time
import warnings
from pathlib import Path
import os
import sys
from typing import List, Dict, Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandadock.protein import Protein
    from pandadock.ligand import Ligand
    from pandadock.utils import get_logger

class ScoringFunction:
    """Base class for all scoring functions with shared parameters and utility methods."""
    
    def __init__(self):
        # Basic atomic parameters
        self.vdw_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }
        
        # Physics-based parameters
        self.atom_charges = {
            'H': 0.0, 'C': 0.0, 'N': -0.4, 'O': -0.4, 'S': -0.15,
            'P': 0.4, 'F': -0.2, 'Cl': -0.08, 'Br': -0.08, 'I': -0.08
        }
        
        # Recalibrated solvation parameters
        self.atom_solvation = {
            'H': 0.0, 'C': 0.4, 'N': -1.5, 'O': -1.5, 'S': 0.4,
            'P': -0.2, 'F': 0.1, 'Cl': 0.3, 'Br': 0.3, 'I': 0.3
        }
        
        # Extended atom type parameters
        self.atom_type_map = {
            # Carbon types
            'C.3': 'C',    # sp3 carbon
            'C.2': 'A',    # sp2 carbon
            'C.ar': 'A',   # aromatic carbon
            'C.1': 'C',    # sp carbon
            
            # Nitrogen types
            'N.3': 'N',    # sp3 nitrogen
            'N.2': 'NA',   # sp2 nitrogen
            'N.ar': 'NA',  # aromatic nitrogen
            'N.1': 'N',    # sp nitrogen
            'N.am': 'NA',  # amide nitrogen
            
            # Oxygen types
            'O.3': 'OA',   # sp3 oxygen (hydroxyl, ether)
            'O.2': 'O',    # sp2 oxygen (carbonyl)
            'O.co2': 'OA', # carboxylate oxygen
            
            # Sulfur types
            'S.3': 'SA',   # sp3 sulfur
            'S.2': 'S',    # sp2 sulfur
            
            # Default element mappings (fallback)
            'C': 'C',    # Non-polar carbon
            'N': 'N',    # Nitrogen
            'O': 'O',    # Oxygen
            'S': 'S',    # Sulfur
            'P': 'P',    # Phosphorus
            'F': 'F',    # Fluorine
            'Cl': 'Cl',  # Chlorine
            'Br': 'Br',  # Bromine
            'I': 'I',    # Iodine
            'H': 'H',    # Hydrogen
        }
        
        # VDW parameters (r_eq and epsilon)
        self.vdw_params = {
            'C': {'r_eq': 4.00, 'epsilon': 0.150},
            'A': {'r_eq': 4.00, 'epsilon': 0.150},  # Aromatic carbon
            'N': {'r_eq': 3.50, 'epsilon': 0.160},
            'NA': {'r_eq': 3.50, 'epsilon': 0.160}, # H-bond acceptor N
            'O': {'r_eq': 3.20, 'epsilon': 0.200},
            'OA': {'r_eq': 3.20, 'epsilon': 0.200}, # H-bond acceptor O
            'S': {'r_eq': 4.00, 'epsilon': 0.200},
            'SA': {'r_eq': 4.00, 'epsilon': 0.200}, # H-bond acceptor S
            'H': {'r_eq': 2.00, 'epsilon': 0.020},
            'F': {'r_eq': 3.09, 'epsilon': 0.080},
            'Cl': {'r_eq': 3.90, 'epsilon': 0.276},
            'Br': {'r_eq': 4.33, 'epsilon': 0.389},
            'I': {'r_eq': 4.72, 'epsilon': 0.550},
            'P': {'r_eq': 4.20, 'epsilon': 0.200},
        }
        
        # Van der Waals well depth parameters
        self.vdw_well_depth = {
            'C': 0.1094,
            'N': 0.0784,
            'O': 0.2100,
            'S': 0.2500,
            'P': 0.2000,
            'F': 0.0610,
            'Cl': 0.2650,
            'Br': 0.3200,
            'I': 0.4000,
            'H': 0.0157
        }
        
        # H-bond parameters
        self.hbond_params = {
            'O-O': {'r_eq': 1.90, 'epsilon': 5.0},
            'O-N': {'r_eq': 1.90, 'epsilon': 5.0},
            'N-O': {'r_eq': 1.90, 'epsilon': 5.0},
            'N-N': {'r_eq': 1.90, 'epsilon': 5.0},
            'O-S': {'r_eq': 2.50, 'epsilon': 1.0},
            'N-S': {'r_eq': 2.50, 'epsilon': 1.0},
        }
        
        # H-bond donor/acceptor types
        self.hbond_donor_types = {'N', 'NA', 'O', 'OA', 'N.3', 'N.am', 'O.3'}
        self.hbond_acceptor_types = {'O', 'OA', 'N', 'NA', 'SA', 'O.2', 'O.3', 'N.2'}
        
        # Hydrophobic atom types
        self.hydrophobic_types = ['C', 'A', 'F', 'Cl', 'Br', 'I', 'C.3', 'C.2', 'C.ar']
        
        # Atomic solvation parameters
        self.atom_solvation_params = {
            'C': 0.4,
            'A': 0.4,
            'N': -1.5,
            'NA': -1.5,
            'O': -1.5,
            'OA': -1.5,
            'S': -0.8,
            'SA': -0.8,
            'H': 0.0,
            'F': -0.5,
            'Cl': -0.1,
            'Br': -0.1,
            'I': -0.1,
            'P': -0.7,
        }
        
        # Atom volume parameters
        self.atom_volume_params = {
            'C': 33.51,
            'A': 33.51,
            'N': 22.45, 
            'NA': 22.45,
            'O': 17.07,
            'OA': 17.07,
            'S': 33.51,
            'SA': 33.51,
            'H': 0.0,
            'F': 15.45,
            'Cl': 35.82,
            'Br': 42.56,
            'I': 55.06,
            'P': 38.80,
        }
        
        # Constants for desolvation
        self.solpar = 0.005
        self.solvation_k = 3.5
        
        # Distance cutoffs - REDUCED for better performance and less accumulation
        self.vdw_cutoff = 6.0      # Reduced from 8.0
        self.hbond_cutoff = 3.5    # Reduced from 4.0
        self.elec_cutoff = 12.0    # Reduced from 20.0
        self.desolv_cutoff = 6.0   # Reduced from 8.0
        self.hydrophobic_cutoff = 4.0  # Reduced from 4.5
        
        # FIXED: Properly balanced weights for composite scoring
        self.weights = {
            'vdw': 0.8,           # Van der Waals
            'hbond': 1.2,         # H-bonds (favorable)
            'elec': 0.6,          # Electrostatics  
            'desolv': 0.4,        # Desolvation penalty
            'hydrophobic': 1.0,   # Hydrophobic interactions (favorable)
            'clash': 5.0,         # Clash penalty (high weight)
            'entropy': 0.3,       # Entropy penalty
        }
        
        # Debug flag for detailed output
        self.verbose = False
    
    def score(self, protein: 'Protein', ligand: 'Ligand') -> float:
        """
        Calculate binding score between protein and ligand.
        
        Parameters:
        -----------
        protein : Protein
        ligand : Ligand
        
        Returns:
        --------
        float
            Binding score (lower is better)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_component_scores(self, protein: 'Protein', ligand: 'Ligand') -> Dict[str, float]:
        """
        Get individual energy component scores.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of individual energy components
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _get_atom_type(self, atom, default='C'):
        """Determine the atom type for an atom based on available information."""
        # Handle case where atom is a list
        if isinstance(atom, list):
            return [self._get_atom_type(a, default) for a in atom]
        
        # Try to get atom type from atom data
        if isinstance(atom, dict):
            atom_type = atom.get('type', None)
        else:
            raise TypeError(f"Expected a dictionary or list for atom, but got {type(atom)}")
        if atom_type and atom_type in self.atom_type_map:
            return self.atom_type_map[atom_type]
        
        # Fall back to element
        element = None
        if 'element' in atom:
            element = atom['element']
        elif 'name' in atom:
            element = atom['name'][0]
        elif 'symbol' in atom:
            element = atom['symbol']
            
        if element:
            # Convert to uppercase for consistency
            element = element.upper()
            if element in self.atom_type_map:
                return self.atom_type_map[element]
        
        # Default fallback
        return default
    
    def _get_protein_atoms(self, protein):
        """Get active site atoms if defined, otherwise all protein atoms."""
        if isinstance(protein, list):
            # Flatten the list of protein atoms
            atoms = []
            for p in protein:
                if hasattr(p, 'active_site') and p.active_site and 'atoms' in p.active_site:
                    atoms.extend(p.active_site['atoms'])
                else:
                    atoms.extend(getattr(p, 'atoms', []))
            return atoms
        if hasattr(protein, 'active_site') and protein.active_site and 'atoms' in protein.active_site:
            return protein.active_site['atoms']
        else:
            return getattr(protein, 'atoms', [])
        
    def _get_ligand_atoms(self, ligand):
        """Get ligand atoms."""
        return ligand.atoms
    
    def _estimate_pose_restriction(self, ligand, protein=None):
        """
        Estimate pose-specific conformational restriction.
        Returns a factor between 0 (fully restricted) and 1 (fully flexible).
        """
        if not protein or not hasattr(protein, 'active_site') or not protein.active_site.get('atoms'):
            return 0.5  # Fallback if no protein info

        ligand_coords = np.array([atom['coords'] for atom in ligand.atoms if 'coords' in atom])
        protein_coords = np.array([atom['coords'] for atom in protein.active_site['atoms'] if 'coords' in atom])

        # Safety checks
        if len(protein_coords) == 0 or len(ligand_coords) == 0:
            return 0.5

        if ligand_coords.ndim != 2 or protein_coords.ndim != 2 or protein_coords.shape[1] != 3 or ligand_coords.shape[1] != 3:
            return 0.5

        # Compute pairwise distances
        from scipy.spatial import cKDTree
        kdtree = cKDTree(protein_coords)
        close_contacts = kdtree.query_ball_point(ligand_coords, r=4.0)  # 4Å cutoff

        buried_atoms = sum(1 for contacts in close_contacts if len(contacts) > 0)
        burial_fraction = buried_atoms / len(ligand.atoms)

        # Heuristic: more burial → more restriction
        flexibility_factor = 1.0 - burial_fraction  # 0 = buried, 1 = exposed

        # Clamp to [0.1, 1.0] for numerical stability
        return max(0.1, min(1.0, flexibility_factor))

class CPUScoringFunction(ScoringFunction):
    """
    Base class for CPU-based scoring functions.
    Implements all energy term calculations using CPU-based methods.
    """
    
    def calculate_vdw(self, protein_atoms, ligand_atoms):
        """
        Calculate van der Waals energy using a modified 12-6 Lennard-Jones potential
        with atom-specific parameters and smoothing function for close contacts.
        """
        vdw_energy = 0.0
        interaction_count = 0
        
        # LIMIT interactions to prevent accumulation
        max_protein_atoms = min(len(protein_atoms), 100)  # Limit protein atoms
        max_ligand_atoms = min(len(ligand_atoms), 50)     # Limit ligand atoms
        
        for i, p_atom in enumerate(protein_atoms[:max_protein_atoms]):
            p_type = self._get_atom_type(p_atom)
            p_coords = p_atom['coords']
            p_params = self.vdw_params.get(p_type, self.vdw_params['C'])
            
            for j, l_atom in enumerate(ligand_atoms[:max_ligand_atoms]):
                l_type = self._get_atom_type(l_atom)
                l_coords = l_atom['coords']
                l_params = self.vdw_params.get(l_type, self.vdw_params['C'])
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Skip if beyond cutoff
                if distance > self.vdw_cutoff:
                    continue
                
                interaction_count += 1
                
                # Calculate combined parameters
                r_eq = (p_params['r_eq'] + l_params['r_eq']) / 2.0  # Arithmetic mean
                epsilon = np.sqrt(p_params['epsilon'] * l_params['epsilon'])  # Geometric mean
                
                # Prevent division by zero
                if distance < 0.1:
                    distance = 0.1
                
                # Calculate ratio for efficiency
                ratio = r_eq / distance
                
                # Use modified potential with smoother transition for close distances
                if distance >= 0.7 * r_eq:
                    # Regular 12-6 Lennard-Jones but SCALED DOWN
                    vdw_term = epsilon * ((ratio**12) - 2.0 * (ratio**6)) * 0.1  # Scale factor
                    vdw_term = max(min(vdw_term, 5.0), -5.0)  # Stricter clipping
                else:
                    # Linear repulsion function for very close distances
                    rep_factor = 5.0 * (0.7 * r_eq - distance) / (0.7 * r_eq)
                    vdw_term = min(rep_factor, 5.0)  # Cap at 5.0
                
                vdw_energy += vdw_term
        
        # NORMALIZE by number of interactions to prevent accumulation
        if interaction_count > 0:
            vdw_energy = vdw_energy / max(1, interaction_count / 10.0)
        
        return vdw_energy
    
    def calculate_hbond(self, protein_atoms, ligand_atoms, protein=None, ligand=None):
        """
        Calculate hydrogen bonding using a Gaussian-like potential.
        """
        hbond_energy = 0.0
        hbond_count = 0
        
        # LIMIT interactions
        max_protein_atoms = min(len(protein_atoms), 80)
        max_ligand_atoms = min(len(ligand_atoms), 40)
        
        # Check for protein donor - ligand acceptor pairs
        for p_atom in protein_atoms[:max_protein_atoms]:
            p_type = self._get_atom_type(p_atom)
            p_coords = p_atom['coords']
            p_element = p_atom.get('element', p_atom.get('name', 'C'))[0].upper()
            
            for l_atom in ligand_atoms[:max_ligand_atoms]:
                l_type = self._get_atom_type(l_atom)
                l_coords = l_atom['coords']
                l_element = l_atom.get('symbol', 'C').upper()
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Skip if beyond cutoff
                if distance > self.hbond_cutoff:
                    continue
                
                # Protein donor - Ligand acceptor
                if p_element in self.hbond_donor_types and l_element in self.hbond_acceptor_types:
                    hb_key = f"{p_element}-{l_element}"
                    
                    if hb_key in self.hbond_params:
                        params = self.hbond_params[hb_key]
                    else:
                        params = {'r_eq': 1.9, 'epsilon': 3.0}
                    
                    r_eq = params['r_eq']
                    epsilon = params['epsilon'] * 0.5  # Scale down H-bond strength
                    
                    if distance < 0.1:
                        distance = 0.1
                    
                    dist_diff = abs(distance - r_eq)
                    
                    if dist_diff <= 0.8:
                        hbond_term = -epsilon * np.exp(-(dist_diff**2) / 0.3)
                        hbond_term = max(hbond_term, -3.0)  # Cap H-bond strength
                        
                        angle_factor = self._calculate_hbond_angle_factor(p_atom, l_atom, protein, ligand)
                        hbond_energy += hbond_term * angle_factor
                        hbond_count += 1
                
                # Ligand donor - Protein acceptor (similar calculation)
                if l_element in self.hbond_donor_types and p_element in self.hbond_acceptor_types:
                    hb_key = f"{l_element}-{p_element}"
                    
                    if hb_key in self.hbond_params:
                        params = self.hbond_params[hb_key]
                    else:
                        params = {'r_eq': 1.9, 'epsilon': 3.0}
                    
                    r_eq = params['r_eq']
                    epsilon = params['epsilon'] * 0.5
                    
                    dist_diff = abs(distance - r_eq)
                    
                    if dist_diff <= 0.8:
                        hbond_term = -epsilon * np.exp(-(dist_diff**2) / 0.3)
                        hbond_term = max(hbond_term, -3.0)
                        
                        angle_factor = self._calculate_hbond_angle_factor(l_atom, p_atom, ligand, protein)
                        hbond_energy += hbond_term * angle_factor
                        hbond_count += 1
        
        # NORMALIZE to prevent over-accumulation
        if hbond_count > 5:  # If too many H-bonds, normalize
            hbond_energy = hbond_energy * (5.0 / hbond_count)
        
        return hbond_energy
    
    def _calculate_hbond_angle_factor(self, donor_atom, acceptor_atom, donor_mol, acceptor_mol):
        """
        Calculate angular dependency factor for hydrogen bond.
        Returns a value between 0 (poor geometry) and 1 (ideal geometry).
        """
        try:
            # Get coordinates
            donor_coords = donor_atom['coords']
            acceptor_coords = acceptor_atom['coords']
            
            # Calculate basic vector
            d_a_vector = acceptor_coords - donor_coords
            d_a_distance = np.linalg.norm(d_a_vector)
            if d_a_distance < 0.1:
                return 0.0  # Atoms are too close
            
            d_a_vector = d_a_vector / d_a_distance
            
            # For simplicity, we'll use a default angle factor
            return 0.7
            
        except Exception as e:
            return 0.3
    
    def calculate_electrostatics(self, protein_atoms, ligand_atoms):
        """
        Calculate electrostatic interactions using Coulomb's law with
        improved distance-dependent dielectric model.
        """
        elec_energy = 0.0
        coulomb_constant = 332.0  # Convert to kcal/mol
        interaction_count = 0
        
        # LIMIT interactions
        max_protein_atoms = min(len(protein_atoms), 80)
        max_ligand_atoms = min(len(ligand_atoms), 40)
        
        for p_atom in protein_atoms[:max_protein_atoms]:
            p_coords = p_atom['coords']
            p_element = p_atom.get('element', p_atom.get('name', 'C'))[0].upper()
            p_charge = self.atom_charges.get(p_element, 0.0)
            
            # Skip atoms with zero charge
            if abs(p_charge) < 1e-6:
                continue
            
            for l_atom in ligand_atoms[:max_ligand_atoms]:
                l_coords = l_atom['coords']
                l_element = l_atom.get('symbol', 'C').upper()
                l_charge = self.atom_charges.get(l_element, 0.0)
                
                # Skip atoms with zero charge
                if abs(l_charge) < 1e-6:
                    continue
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Skip if beyond cutoff
                if distance > self.elec_cutoff:
                    continue
                
                interaction_count += 1
                
                # Prevent division by zero
                if distance < 0.1:
                    distance = 0.1
                
                # Calculate improved distance-dependent dielectric
                dielectric = 8.0 * distance  # Increased dielectric to reduce magnitude
                
                # Calculate Coulomb energy with Debye-Hückel-like screening
                screening_factor = np.exp(-distance / 15.0)  # Longer screening length
                elec_term = coulomb_constant * p_charge * l_charge * screening_factor / (dielectric * distance)
                
                # STRONG scaling down of electrostatic interactions
                elec_term = np.sign(elec_term) * min(abs(elec_term), 2.0) * 0.1
                
                elec_energy += elec_term
        
        # NORMALIZE by interactions
        if interaction_count > 0:
            elec_energy = elec_energy / max(1, interaction_count / 5.0)
        
        return elec_energy
    
    def calculate_desolvation(self, protein_atoms, ligand_atoms):
        """
        Calculate desolvation energy using recalibrated atomic solvation and volume parameters.
        """
        desolv_energy = 0.0
        sigma = self.solvation_k
        sigma_squared_2 = 2.0 * sigma * sigma
        interaction_count = 0
        
        # LIMIT interactions
        max_protein_atoms = min(len(protein_atoms), 60)
        max_ligand_atoms = min(len(ligand_atoms), 30)
        
        for p_atom in protein_atoms[:max_protein_atoms]:
            p_coords = p_atom['coords']
            p_type = self._get_atom_type(p_atom)
            p_solv = self.atom_solvation_params.get(p_type, 0.0)
            p_vol = self.atom_volume_params.get(p_type, 0.0)
            
            for l_atom in ligand_atoms[:max_ligand_atoms]:
                l_coords = l_atom['coords']
                l_type = self._get_atom_type(l_atom)
                l_solv = self.atom_solvation_params.get(l_type, 0.0)
                l_vol = self.atom_volume_params.get(l_type, 0.0)
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Skip if beyond cutoff
                if distance > self.desolv_cutoff:
                    continue
                
                interaction_count += 1
                
                # Calculate exponential term
                exp_term = np.exp(-(distance*distance) / sigma_squared_2)
                
                # Calculate desolvation energy (volume-weighted)
                desolv_term = (self.solpar * p_solv * l_vol + 
                              self.solpar * l_solv * p_vol) * exp_term
                
                # STRONG scaling down and capping
                desolv_term = np.sign(desolv_term) * min(abs(desolv_term), 1.0) * 0.2
                
                desolv_energy += desolv_term
        
        # NORMALIZE
        if interaction_count > 0:
            desolv_energy = desolv_energy / max(1, interaction_count / 8.0)
        
        return desolv_energy
    
    def calculate_hydrophobic(self, protein_atoms, ligand_atoms):
        """
        Calculate hydrophobic interactions using physics-based approach.
        """
        hydrophobic_score = 0.0
        
        # Identify hydrophobic atoms with LIMITS
        p_hydrophobic = [atom for atom in protein_atoms[:60] 
                        if self._get_atom_type(atom) in self.hydrophobic_types]
        
        l_hydrophobic = [atom for atom in ligand_atoms[:30] 
                        if self._get_atom_type(atom) in self.hydrophobic_types]
        
        interaction_count = 0
        
        for p_atom in p_hydrophobic:
            p_coords = p_atom['coords']
            
            for l_atom in l_hydrophobic:
                l_coords = l_atom['coords']
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Skip if beyond cutoff
                if distance > self.hydrophobic_cutoff:
                    continue
                
                interaction_count += 1
                
                # Linear hydrophobic interaction term with smoother transition
                if distance < 0.5:  # Avoid unrealistic close contacts
                    contact_score = 0.0
                else:
                    # Stronger interaction for closer contact with smoothing
                    direct_factor = (self.hydrophobic_cutoff - distance) / self.hydrophobic_cutoff
                    # Apply sigmoidal scaling for smoother transition and SCALE DOWN
                    contact_score = (direct_factor / (1.0 + np.exp(-(direct_factor*10 - 5)))) * 0.5
                
                hydrophobic_score -= contact_score  # Negative since it's favorable
        
        # NORMALIZE hydrophobic interactions
        if interaction_count > 8:
            hydrophobic_score = hydrophobic_score * (8.0 / interaction_count)
        
        return hydrophobic_score
    
    def calculate_clashes(self, protein_atoms, ligand_atoms):
        """
        Calculate steric clashes using Van der Waals overlap and exponential repulsion.
        """
        clash_score = 0.0
        
        # LIMIT atoms to check
        max_protein_atoms = min(len(protein_atoms), 100)
        max_ligand_atoms = min(len(ligand_atoms), 50)

        for p_atom in protein_atoms[:max_protein_atoms]:
            if 'coords' not in p_atom:
                continue
            p_coords = p_atom['coords']
            p_type = self._get_atom_type(p_atom)
            p_radius = self.vdw_params.get(p_type, self.vdw_params['C'])['r_eq'] / 2.0

            for l_atom in ligand_atoms[:max_ligand_atoms]:
                if 'coords' not in l_atom:
                    continue
                l_coords = l_atom['coords']
                l_type = self._get_atom_type(l_atom)
                l_radius = self.vdw_params.get(l_type, self.vdw_params['C'])['r_eq'] / 2.0

                distance = np.linalg.norm(p_coords - l_coords)
                min_allowed = (p_radius + l_radius) * 0.7
                upper_bound = (p_radius + l_radius) * 1.2

                if distance < min_allowed:
                    # SCALED DOWN clash penalty
                    repulsion = (np.exp((min_allowed - distance) / min_allowed) - 1.0) * 0.2
                    clash_score += min(repulsion ** 2, 2.0)  # Cap individual clashes
                elif distance < upper_bound:
                    soft_penalty = (upper_bound - distance) / (upper_bound - min_allowed)
                    clash_score += 0.05 * (soft_penalty ** 2)  # Reduced soft penalty

        return min(clash_score, 10.0)  # Cap total clash score

    def calculate_enhanced_clashes(self, protein_atoms, ligand_atoms, backbone_factor=2.0):
        """
        Calculate clash score with enhanced sensitivity to backbone atoms.
        """
        clash_score = 0.0
        vdw_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }
        clash_distance_factor = 0.7
        backbone_atoms = {'CA', 'C', 'N', 'O'}
        
        # LIMIT atoms
        max_protein_atoms = min(len(protein_atoms), 80)
        max_ligand_atoms = min(len(ligand_atoms), 40)
        
        for p_atom in protein_atoms[:max_protein_atoms]:
            p_coords = p_atom['coords']
            p_element = p_atom.get('element', p_atom.get('name', 'C'))[0]
            p_radius = vdw_radii.get(p_element, 1.7)
            
            # Check if this is a backbone atom
            is_backbone = False
            if 'name' in p_atom:
                atom_name = p_atom['name'].strip()
                is_backbone = atom_name in backbone_atoms
            
            for l_atom in ligand_atoms[:max_ligand_atoms]:
                l_coords = l_atom['coords']
                l_element = l_atom.get('symbol', 'C')
                l_radius = vdw_radii.get(l_element, 1.7)
                
                distance = np.linalg.norm(p_coords - l_coords)
                min_allowed = (p_radius + l_radius) * clash_distance_factor
                
                if distance < min_allowed:
                    clash_severity = min_allowed / max(distance, 0.1)
                    
                    # SCALED DOWN penalties
                    if is_backbone:
                        clash_score += min((clash_severity ** 2) * backbone_factor * 0.1, 1.0)
                    else:
                        clash_score += min(clash_severity ** 2 * 0.1, 0.5)
        
        return min(clash_score, 5.0)  # Cap enhanced clash score

    def calculate_entropy(self, ligand, protein=None):
        """Calculate entropy penalty - SCALED DOWN."""
        n_rotatable = len(getattr(ligand, 'rotatable_bonds', []))
        n_atoms = len(ligand.atoms)  
        flexibility = self._estimate_pose_restriction(ligand, protein)
        # MUCH smaller entropy penalty
        entropy_penalty = 0.1 * n_rotatable * flexibility * (1.0 + 0.01 * n_atoms)
        return min(entropy_penalty, 2.0)  # Cap entropy penalty

class CompositeScoringFunction(CPUScoringFunction):
    """Composite scoring function combining all energy terms."""
    
    def score(self, protein, ligand):
        """
        Calculate composite score with enhanced backbone clash detection.
        """
        protein_atoms = self._get_protein_atoms(protein)
        ligand_atoms = self._get_ligand_atoms(ligand)
        
        # Calculate energy components
        vdw = self.calculate_vdw(protein_atoms, ligand_atoms)
        hbond = self.calculate_hbond(protein_atoms, ligand_atoms)
        elec = self.calculate_electrostatics(protein_atoms, ligand_atoms)
        desolv = self.calculate_desolvation(protein_atoms, ligand_atoms)
        hydrophobic = self.calculate_hydrophobic(protein_atoms, ligand_atoms)
        
        # Regular clash detection
        clash = self.calculate_clashes(protein_atoms, ligand_atoms)
        
        # Enhanced backbone-aware clash detection
        backbone_clash = self.calculate_enhanced_clashes(protein_atoms, ligand_atoms, backbone_factor=3.0)
        
        # Entropy calculation
        entropy = self.calculate_entropy(ligand, protein)
        
        # Combine scores with special handling for backbone clashes
        total = (
            self.weights['vdw'] * vdw +
            self.weights['hbond'] * hbond +  # H-bonds are already negative
            self.weights['elec'] * elec +
            self.weights['desolv'] * desolv +
            self.weights['hydrophobic'] * hydrophobic +  # Already negative
            self.weights['clash'] * clash +  # Positive penalty
            self.weights['clash'] * backbone_clash +  # Additional backbone penalty
            self.weights['entropy'] * entropy  # Positive penalty
        )
        
        # Apply severe penalty for significant backbone clashes
        if backbone_clash > 1.0:  # Lower threshold
            backbone_penalty = backbone_clash * 2.0  # Reduced penalty
            total += backbone_penalty
        
        # Print breakdown if verbose
        if self.verbose:
            print(f"VDW: {vdw:.3f}, H-bond: {hbond:.3f}, Elec: {elec:.3f}, "
                f"Desolv: {desolv:.3f}, Hydrophobic: {hydrophobic:.3f}, "
                f"Clash: {clash:.3f}, Backbone Clash: {backbone_clash:.3f}, "
                f"Entropy: {entropy:.3f}")
            print(f"Total: {total:.3f}")
        
        # FINAL SCORE: More negative = better binding
        return -total  # Negate so that lower (more negative) is better
    
    def get_component_scores(self, protein, ligand):
        """
        Get individual energy component scores for detailed analysis.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of individual energy components
        """
        protein_atoms = self._get_protein_atoms(protein)
        ligand_atoms = self._get_ligand_atoms(ligand)
        
        # Calculate energy components
        vdw = self.calculate_vdw(protein_atoms, ligand_atoms)
        hbond = self.calculate_hbond(protein_atoms, ligand_atoms)
        elec = self.calculate_electrostatics(protein_atoms, ligand_atoms)
        desolv = self.calculate_desolvation(protein_atoms, ligand_atoms)
        hydrophobic = self.calculate_hydrophobic(protein_atoms, ligand_atoms)
        clash = self.calculate_clashes(protein_atoms, ligand_atoms)
        backbone_clash = self.calculate_enhanced_clashes(protein_atoms, ligand_atoms, backbone_factor=3.0)
        entropy = self.calculate_entropy(ligand, protein)
        
        # Calculate weighted components (as they contribute to final score)
        components = {
            'Van der Waals': self.weights['vdw'] * vdw,
            'H-Bond': self.weights['hbond'] * hbond,
            'Electrostatic': self.weights['elec'] * elec,
            'Desolvation': self.weights['desolv'] * desolv,
            'Hydrophobic': self.weights['hydrophobic'] * hydrophobic,
            'Clash': self.weights['clash'] * clash,
            'Backbone_Clash': self.weights['clash'] * backbone_clash,
            'Entropy': self.weights['entropy'] * entropy,
        }
        
        # Calculate total (before final negation)
        subtotal = sum(components.values())
        
        # Add severe backbone penalty if applicable
        if backbone_clash > 1.0:
            backbone_penalty = backbone_clash * 2.0
            components['Backbone_Penalty'] = backbone_penalty
            subtotal += backbone_penalty
        
        # Add the final total (negated for better = more negative)
        components['Total'] = -subtotal
        
        return components

class EnhancedScoringFunction(CompositeScoringFunction):
    """
    EnhancedScoringFunction with recalibrated weights.
    """
    
    def __init__(self):
        super().__init__()
        
        # REBALANCED weights for better scoring
        self.weights = {
            'vdw': 1.0,           # Van der Waals interactions
            'hbond': 2.0,         # H-bonds (more important)
            'elec': 0.8,          # Electrostatics
            'desolv': 0.3,        # Desolvation penalty (reduced)
            'hydrophobic': 1.5,   # Hydrophobic interactions (important)
            'clash': 8.0,         # Clash penalty (high importance)
            'entropy': 0.4        # Entropy penalty
        }

# Add the GPU scoring classes with get_component_scores method
class GPUScoringFunction(ScoringFunction):
    """
    Base class for GPU-accelerated scoring functions.
    """
    
    def __init__(self, device='cuda', precision='float32'):
        super().__init__()
        self.device_name = device
        self.precision = precision
        self.device = None
        self.torch_available = False
        self.cupy_available = False
        self._init_gpu()
    
    def _init_gpu(self):
        """Initialize GPU resources and check available hardware."""
        self.device = 'cpu'  # Force CPU for compatibility
        self.gpu_available = False  # Disable GPU checks
        
        try:
            import torch
            self.torch_available = True
            
            if self.device_name == 'cuda' and torch.cuda.is_available():
                self.device = torch.device('cuda')
                gpu_name = torch.cuda.get_device_name(0)
                print(f"Using GPU: {gpu_name}")
                
                if self.precision == 'float64':
                    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
                else:
                    torch.set_default_tensor_type(torch.cuda.FloatTensor)
            else:
                self.device = torch.device('cpu')
                if self.precision == 'float64':
                    torch.set_default_tensor_type(torch.DoubleTensor)
        except ImportError:
            print("PyTorch not available. Using CPU calculations.")
    
    def score(self, protein, ligand):
        """Calculate composite score using GPU acceleration."""
        # Use CPU implementation with same logic but potentially GPU-accelerated components
        cpu_scorer = CompositeScoringFunction()
        cpu_scorer.weights = self.weights  # Use our weights
        cpu_scorer.verbose = self.verbose
        return cpu_scorer.score(protein, ligand)
    
    def get_component_scores(self, protein, ligand):
        """Get component scores using CPU implementation."""
        cpu_scorer = CompositeScoringFunction()
        cpu_scorer.weights = self.weights
        return cpu_scorer.get_component_scores(protein, ligand)

class EnhancedGPUScoringFunction(GPUScoringFunction):
    """Enhanced GPU-accelerated scoring function with calibrated weights."""
    
    def __init__(self, device='cuda', precision='float32'):
        super().__init__(device, precision)
        
        # Same improved weights as EnhancedScoringFunction
        self.weights = {
            'vdw': 1.0,
            'hbond': 2.0,
            'elec': 0.8,
            'desolv': 0.3,
            'hydrophobic': 1.5,
            'clash': 8.0,
            'entropy': 0.4
        }

class TetheredScoringFunction:
    """A scoring function wrapper that adds a penalty term based on RMSD from a reference position."""
    
    def __init__(self, base_scoring_function, reference_ligand, weight=10.0, max_penalty=100.0):
        self.base_scoring_function = base_scoring_function
        self.reference_coordinates = reference_ligand.xyz.copy()
        self.weight = weight
        self.max_penalty = max_penalty
        self.weights = self.base_scoring_function.weights.copy()
        self.verbose = getattr(self.base_scoring_function, 'verbose', False)
    
    def score(self, protein, ligand):
        """Calculate tethered score."""
        base_score = self.base_scoring_function.score(protein, ligand)
        rmsd = self.calculate_rmsd(ligand.xyz)
        rmsd_penalty = min(self.weight * rmsd, self.max_penalty)
        
        if self.verbose:
            print(f"Base score: {base_score:.2f}, RMSD: {rmsd:.2f}, RMSD penalty: {rmsd_penalty:.2f}")
            print(f"Total tethered score: {base_score + rmsd_penalty:.2f}")

        return base_score + rmsd_penalty
    
    def get_component_scores(self, protein, ligand):
        """Get component scores with RMSD penalty."""
        components = self.base_scoring_function.get_component_scores(protein, ligand)
        rmsd = self.calculate_rmsd(ligand.xyz)
        rmsd_penalty = min(self.weight * rmsd, self.max_penalty)
        
        components['RMSD_Penalty'] = rmsd_penalty
        components['Total'] = components.get('Total', 0) + rmsd_penalty
        
        return components
    
    def calculate_rmsd(self, coordinates):
        """Calculate RMSD between coordinates and reference coordinates."""
        if len(coordinates) != len(self.reference_coordinates):
            raise ValueError(f"Coordinate mismatch: reference has {len(self.reference_coordinates)} atoms, but current pose has {len(coordinates)} atoms")
        
        squared_diff = np.sum((coordinates - self.reference_coordinates) ** 2, axis=1)
        return np.sqrt(np.mean(squared_diff))
    
    def __getattr__(self, name):
        return getattr(self.base_scoring_function, name)