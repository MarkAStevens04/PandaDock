import numpy as np
from copy import deepcopy
from pathlib import Path
import os
import tempfile
import seaborn as sns
import matplotlib.pyplot as plt

class MMFFMinimization:
    """
    MMFF94 Force Field minimization for ligands using RDKit.
    This provides full molecular mechanics energy minimization.
    """
    
    def __init__(self, max_iterations=200, converge_criterion=0.01):
        """
        Initialize MMFF minimization.
        
        Parameters:
        -----------
        max_iterations : int
            Maximum number of minimization steps
        converge_criterion : float
            Convergence criterion for energy change
        """
        self.max_iterations = max_iterations
        self.converge_criterion = converge_criterion
        self._check_rdkit()
    
    def _check_rdkit(self):
        """Check if RDKit is available and raise import error if not."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            self.rdkit_available = True
        except ImportError:
            print("Warning: RDKit not available. MMFF minimization will not work.")
            self.rdkit_available = False
    
    def minimize_ligand(self, ligand):
        """
        Perform MMFF minimization on a ligand.
        
        Parameters:
        -----------
        ligand : Ligand
            Ligand object from PandaDock
        
        Returns:
        --------
        Ligand
            Minimized ligand
        """
        if not self.rdkit_available:
            print("RDKit not available. Skipping minimization.")
            return ligand
        
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            # Create a temporary file for the ligand
            fd, tmp_file = tempfile.mkstemp(suffix='.sdf')
            os.close(fd)
            
            # Write ligand to SDF file
            self._write_ligand_to_sdf(ligand, tmp_file)
            
            # Read with RDKit
            mol = Chem.SDMolSupplier(tmp_file)[0]
            if mol is None:
                print("Error: Could not read ligand with RDKit.")
                return ligand
            
            # Set up MMFF force field
            success = AllChem.MMFFOptimizeMolecule(
                mol, 
                maxIters=self.max_iterations,
                nonBondedThresh=10.0,  # Angstroms
                confId=0
            )
            
            if success == -1:
                print("Warning: MMFF setup failed. Falling back to UFF.")
                AllChem.UFFOptimizeMolecule(mol, maxIters=self.max_iterations)
            
            # Write minimized structure back to file
            writer = Chem.SDWriter(tmp_file)
            writer.write(mol)
            writer.close()
            
            # Read back into a new ligand object
            minimized_ligand = self._read_ligand_from_sdf(tmp_file)
            
            # Clean up
            os.unlink(tmp_file)
            
            return minimized_ligand
            
        except Exception as e:
            print(f"Error during minimization: {e}")
            return ligand
    
    def minimize_pose(self, protein, ligand_pose, distance_cutoff=2.0):
        """
        Perform constrained minimization of a ligand pose in protein environment.
        
        Parameters:
        -----------
        protein : Protein
            Protein object from PandaDock
        ligand_pose : Ligand
            Ligand pose to minimize
        distance_cutoff : float
            Distance cutoff for protein-ligand interactions (Angstroms)
        
        Returns:
        --------
        Ligand
            Minimized ligand pose
        """
        if not self.rdkit_available:
            print("RDKit not available. Skipping pose minimization.")
            return ligand_pose
        
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            # Create temporary files
            fd1, tmp_ligand = tempfile.mkstemp(suffix='.sdf')
            fd2, tmp_protein = tempfile.mkstemp(suffix='.pdb')
            os.close(fd1)
            os.close(fd2)
            
            # Write ligand and protein to files
            self._write_ligand_to_sdf(ligand_pose, tmp_ligand)
            self._write_protein_to_pdb(protein, tmp_protein)
            
            # Read with RDKit
            lig_mol = Chem.SDMolSupplier(tmp_ligand)[0]
            prot_mol = Chem.MolFromPDB(tmp_protein)
            
            if lig_mol is None or prot_mol is None:
                print("Error reading molecules for constrained minimization.")
                return ligand_pose
            
            # Create a combined system for MMFF
            combo = Chem.CombineMols(prot_mol, lig_mol)
            
            # Setup MMFF and minimize
            try:
                # This part is experimental and may not work in all RDKit versions
                mp = AllChem.MMFFGetMoleculeProperties(combo)
                ff = AllChem.MMFFGetMoleculeForceField(
                    combo, mp, nonBondedThresh=distance_cutoff
                )
                
                # Freeze protein atoms
                for i in range(prot_mol.GetNumAtoms()):
                    ff.AddFixedPoint(i)
                
                # Run minimization
                ff.Minimize(maxIts=self.max_iterations, 
                           energyTol=self.converge_criterion)
                
                # Extract ligand part
                minimized_mol = Chem.DeleteSubstructs(combo, prot_mol)
                
                # Write to file
                writer = Chem.SDWriter(tmp_ligand)
                writer.write(minimized_mol)
                writer.close()
                
                # Read minimized pose
                minimized_pose = self._read_ligand_from_sdf(tmp_ligand)
                
            except Exception as e:
                print(f"MMFF constrained minimization failed: {e}")
                print("Falling back to ligand-only minimization.")
                AllChem.MMFFOptimizeMolecule(lig_mol, maxIters=self.max_iterations)
                
                # Write to file
                writer = Chem.SDWriter(tmp_ligand)
                writer.write(lig_mol)
                writer.close()
                
                # Read minimized pose
                minimized_pose = self._read_ligand_from_sdf(tmp_ligand)
            
            # Clean up
            os.unlink(tmp_ligand)
            os.unlink(tmp_protein)
            
            return minimized_pose
            
        except Exception as e:
            print(f"Error during constrained minimization: {e}")
            return ligand_pose
    
    def _write_ligand_to_sdf(self, ligand, filename):
        """Write ligand to SDF file."""
        with open(filename, 'w') as f:
            f.write("Ligand\n")
            f.write("  PandaDock\n\n")
            
            # Number of atoms and bonds
            f.write(f"{len(ligand.atoms):3d}{len(ligand.bonds):3d}  0  0  0  0  0  0  0  0999 V2000\n")
            
            # Atoms
            for atom in ligand.atoms:
                coords = atom['coords']
                symbol = atom.get('symbol', 'C')
                f.write(f"{coords[0]:10.4f}{coords[1]:10.4f}{coords[2]:10.4f} {symbol:<3}  0  0  0  0  0  0  0  0  0  0  0  0\n")
            
            # Bonds
            for bond in ligand.bonds:
                a1 = bond['begin_atom_idx'] + 1  # 1-based indexing in SDF
                a2 = bond['end_atom_idx'] + 1
                type_num = bond.get('bond_type', 1)
                if isinstance(type_num, str):
                    type_num = 1  # Default to single bond
                f.write(f"{a1:3d}{a2:3d}{type_num:3d}  0  0  0  0\n")
            
            # Terminator
            f.write("M  END\n$$$$\n")
    
    def _write_protein_to_pdb(self, protein, filename):
        """Write protein to PDB file."""
        with open(filename, 'w') as f:
            for i, atom in enumerate(protein.atoms):
                name = atom.get('name', '').ljust(4)
                res_name = atom.get('residue_name', 'UNK')
                chain_id = atom.get('chain_id', 'A')
                res_id = atom.get('residue_id', 1)
                coords = atom['coords']
                
                f.write(f"ATOM  {i+1:5d} {name} {res_name:3s} {chain_id:1s}{res_id:4d}    "
                        f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}  1.00  0.00\n")
            f.write("END\n")
    
    def _read_ligand_from_sdf(self, filename):
        """Read ligand from SDF file."""
        from .ligand import Ligand
        return Ligand(filename)


class ImprovedElectrostatics:
    """
    Improved electrostatics calculations with Poisson-Boltzmann inspired model.
    This provides more accurate treatment of charge-charge interactions.
    """
    
    def __init__(self, ionic_strength=0.15, temperature=298.15, 
                 interior_dielectric=4.0, solvent_dielectric=80.0):
        """
        Initialize improved electrostatics model.
        
        Parameters:
        -----------
        ionic_strength : float
            Ionic strength in mol/L
        temperature : float
            Temperature in Kelvin
        interior_dielectric : float
            Dielectric constant inside protein and ligand
        solvent_dielectric : float
            Dielectric constant of solvent
        """
        self.ionic_strength = ionic_strength  # mol/L
        self.temperature = temperature  # K
        self.interior_dielectric = interior_dielectric
        self.solvent_dielectric = solvent_dielectric
        
        # Physical constants
        self.k_boltzmann = 1.380649e-23  # J/K
        self.e_charge = 1.602176634e-19  # C
        self.n_avogadro = 6.02214076e23  # 1/mol
        self.epsilon_0 = 8.8541878128e-12  # F/m
        
        # Compute derived quantities
        self.kappa = self._compute_kappa()  # Debye screening length (Å^-1)
        
        # Atom parameters
        self.atom_charges = {
            'H': 0.0, 'C': 0.0, 'N': -0.5, 'O': -0.5, 'S': -0.2,
            'P': 0.5, 'F': -0.25, 'Cl': -0.1, 'Br': -0.1, 'I': -0.1
        }
        
        # Atomic radii for accessibility calculations
        self.atom_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }
    
    def _compute_kappa(self):
        """
        Compute the Debye screening parameter (kappa) based on ionic strength.
        
        Returns:
        --------
        float
            Debye screening parameter in Å^-1
        """
        # Convert from SI units to more convenient units for docking
        # Factor of 10 is to convert from m^-1 to Å^-1
        kappa_squared = (2 * self.ionic_strength * self.n_avogadro * self.e_charge**2) / \
                        (self.epsilon_0 * self.solvent_dielectric * self.k_boltzmann * self.temperature)
        return np.sqrt(kappa_squared) * 1e-10  # Convert to Å^-1
    
    def calculate_electrostatics(self, protein, ligand):
        """
        Calculate electrostatic interaction energy using a modified Poisson-Boltzmann approach.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
        
        Returns:
        --------
        float
            Electrostatic interaction energy in kcal/mol
        """
        # Get active site atoms if defined
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Calculate solvent accessible surface for protein and ligand atoms
        protein_sasa = self._calc_approximate_sasa(protein_atoms)
        ligand_sasa = self._calc_approximate_sasa(ligand.atoms)
        
        # Initialize energy
        elec_energy = 0.0
        
        # Constants for conversion to kcal/mol (from J)
        conversion = 1.0 / (4.184 * 1000)  # J to kcal/mol
        
        # Calculate pairwise interactions
        for i, p_atom in enumerate(protein_atoms):
            p_coords = p_atom['coords']
            p_symbol = p_atom.get('element', p_atom.get('name', 'C'))[0]
            p_charge = self.atom_charges.get(p_symbol, 0.0)
            p_buried = 1.0 - min(1.0, protein_sasa[i])
            
            for j, l_atom in enumerate(ligand.atoms):
                l_coords = l_atom['coords']
                l_symbol = l_atom.get('symbol', 'C')
                l_charge = self.atom_charges.get(l_symbol, 0.0)
                l_buried = 1.0 - min(1.0, ligand_sasa[j])
                
                # Skip if either charge is zero
                if abs(p_charge * l_charge) < 1e-6:
                    continue
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Skip if too far away
                if distance > 15.0:  # Cutoff for efficiency
                    continue
                
                # Calculate effective dielectric based on burial
                # More buried atoms experience lower dielectric
                burial_factor = (p_buried + l_buried) / 2.0
                effective_dielectric = self.interior_dielectric + \
                                     (self.solvent_dielectric - self.interior_dielectric) * (1.0 - burial_factor)
                
                # Modified Coulomb with Debye-Hückel screening
                if distance < 0.1:  # Avoid division by zero
                    energy = 0.0
                else:
                    # Coulomb term with screening
                    coulomb = 332.0 * p_charge * l_charge / (effective_dielectric * distance)
                    screening = np.exp(-self.kappa * distance)
                    energy = coulomb * screening
                
                elec_energy += energy
        
        return elec_energy
    
    def _calc_approximate_sasa(self, atoms):
        """
        Calculate approximate solvent accessible surface area (SASA) for each atom.
        
        Parameters:
        -----------
        atoms : list
            List of atom dictionaries
        
        Returns:
        --------
        list
            List of SASA values for each atom
        """
        # Convert atoms to numpy array for faster computation
        coords = np.array([atom['coords'] for atom in atoms])
        n_atoms = len(atoms)
        
        # Get radii
        radii = np.zeros(n_atoms)
        for i, atom in enumerate(atoms):
            symbol = atom.get('element', atom.get('name', atom.get('symbol', 'C')))[0]
            radii[i] = self.atom_radii.get(symbol, 1.7)
        
        # Add water probe radius (1.4 Å)
        radii_with_probe = radii + 1.4
        
        # Calculate approximate SASA by check for neighboring atoms
        sasa = np.ones(n_atoms)  # Start with fully exposed
        
        # Calculate all pairwise distances
        dist_matrix = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                dist_matrix[i,j] = dist
                dist_matrix[j,i] = dist
        
        # For each atom, check how many neighbors are within van der Waals contact
        for i in range(n_atoms):
            r_i = radii_with_probe[i]
            
            # Count neighbors that overlap
            for j in range(n_atoms):
                if i == j:
                    continue
                    
                r_j = radii[j]  # Note: no probe for neighbors
                dist = dist_matrix[i,j]
                
                # Check if atoms overlap
                if dist < (r_i + r_j):
                    # Estimate overlap
                    overlap = 1.0 - (dist / (r_i + r_j))
                    # Reduce SASA proportionally to overlap
                    sasa[i] -= overlap * 0.1  # Scale factor to avoid overestimation
        
        # Ensure SASA is non-negative
        sasa = np.maximum(0.0, sasa)
        
        return sasa


class GeneralizedBornSolvation:
    """
    Generalized Born (GB) model for solvation energy.
    This provides an implicit solvent model for calculating solvation effects.
    """
    
    def __init__(self, temperature=298.15, solvent_dielectric=80.0, 
                 interior_dielectric=1.0, surface_tension=0.00542):
        """
        Initialize GB solvation model.
        
        Parameters:
        -----------
        temperature : float
            Temperature in Kelvin
        solvent_dielectric : float
            Dielectric constant of solvent
        interior_dielectric : float
            Dielectric constant inside protein and ligand
        surface_tension : float
            Surface tension parameter for nonpolar contribution (kcal/mol/Å²)
        """
        self.temperature = temperature
        self.solvent_dielectric = solvent_dielectric
        self.interior_dielectric = interior_dielectric
        self.surface_tension = surface_tension
        
        # Atom parameters
        self.atom_charges = {
            'H': 0.0, 'C': 0.0, 'N': -0.5, 'O': -0.5, 'S': -0.2,
            'P': 0.5, 'F': -0.25, 'Cl': -0.1, 'Br': -0.1, 'I': -0.1
        }
        
        # Atom radii for GB calculations
        self.atom_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }
        
        # Scale factor for Born radii calculations
        self.scale_factor = 0.8

        self.atom_solvation = {
            'C': 0.4,
            'N': -1.5,
            'O': -1.5,
            'S': -0.8,
            'P': -0.7,
            'F': -0.5,
            'Cl': -0.1,
            'Br': -0.1,
            'I': -0.1,
            'H': 0.0
        }
        
        # Constants
        self.solpar = 0.005  # Updated from 0.05 to 0.005
        self.solvation_k = 3.5  # Solvation radius in Å
    
    def calculate_solvation_free_energy(self, molecule, molecule_type='ligand'):
        """
        Calculate solvation free energy using GB model.
        
        Parameters:
        -----------
        molecule : Ligand or Protein
            Molecule object
        molecule_type : str
            Type of molecule ('ligand' or 'protein')
        
        Returns:
        --------
        tuple
            (polar_energy, nonpolar_energy, total_energy) in kcal/mol
        """
        if molecule_type == 'ligand':
            atoms = molecule.atoms
            atom_list = []
            for atom in atoms:
                symbol = atom.get('symbol', 'C')
                coords = atom['coords']
                charge = self.atom_charges.get(symbol, 0.0)
                radius = self.atom_radii.get(symbol, 1.7)
                atom_list.append((coords, charge, radius))
        else:  # protein
            atoms = molecule.atoms
            atom_list = []
            for atom in atoms:
                symbol = atom.get('element', atom.get('name', 'C'))[0]
                coords = atom['coords']
                charge = self.atom_charges.get(symbol, 0.0)
                radius = self.atom_radii.get(symbol, 1.7)
                atom_list.append((coords, charge, radius))
        
        # Calculate Born radii
        born_radii = self._calculate_born_radii(atom_list)
        
        # Calculate polar solvation energy (electrostatic)
        polar_energy = self._calculate_polar_energy(atom_list, born_radii)
        
        # Calculate nonpolar solvation energy (cavity formation)
        nonpolar_energy = self._calculate_nonpolar_energy(atom_list)
        
        # Total solvation energy
        total_energy = polar_energy + nonpolar_energy
        
        return (polar_energy, nonpolar_energy, total_energy)
    
    def calculate_binding_solvation(self, protein, ligand):
        """
        Calculate solvation contribution to binding free energy.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
        
        Returns:
        --------
        float
            Solvation contribution to binding in kcal/mol
        """
        # Get active site atoms if defined
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
            p_obj = type('obj', (object,), {'atoms': protein_atoms})
        else:
            p_obj = protein
        
        # Calculate solvation for ligand alone
        ligand_polar, ligand_nonpolar, ligand_total = self.calculate_solvation_free_energy(
            ligand, 'ligand')
        
        # Calculate solvation for protein alone
        protein_polar, protein_nonpolar, protein_total = self.calculate_solvation_free_energy(
            p_obj, 'protein')
        
        # Create a combined molecule for the complex
        complex_atoms = []
        
        # Add protein atoms
        for atom in protein_atoms:
            symbol = atom.get('element', atom.get('name', 'C'))[0]
            coords = atom['coords']
            charge = self.atom_charges.get(symbol, 0.0)
            radius = self.atom_radii.get(symbol, 1.7)
            complex_atoms.append((coords, charge, radius))
        
        # Add ligand atoms
        for atom in ligand.atoms:
            symbol = atom.get('symbol', 'C')
            coords = atom['coords']
            charge = self.atom_charges.get(symbol, 0.0)
            radius = self.atom_radii.get(symbol, 1.7)
            complex_atoms.append((coords, charge, radius))
        
        # Calculate Born radii for complex
        complex_born_radii = self._calculate_born_radii(complex_atoms)
        
        # Calculate polar solvation energy for complex
        complex_polar = self._calculate_polar_energy(complex_atoms, complex_born_radii)
        
        # Calculate nonpolar solvation energy for complex
        complex_nonpolar = self._calculate_nonpolar_energy(complex_atoms)
        
        complex_total = complex_polar + complex_nonpolar
        
        # Solvation contribution to binding = ΔG_solv(complex) - ΔG_solv(protein) - ΔG_solv(ligand)
        solvation_contribution = complex_total - protein_total - ligand_total
        
        return solvation_contribution
    
    def _calculate_born_radii(self, atom_list):
        """
        Calculate effective Born radii for atoms.
        
        Parameters:
        -----------
        atom_list : list
            List of (coords, charge, radius) tuples
        
        Returns:
        --------
        list
            List of Born radii for each atom
        """
        n_atoms = len(atom_list)
        born_radii = np.zeros(n_atoms)
        
        # Extract coordinates and radii
        coords = np.array([atom[0] for atom in atom_list])
        radii = np.array([atom[2] for atom in atom_list])
        
        # Calculate self-volumes and initialize Born radii
        for i in range(n_atoms):
            # Initial Born radius is atom radius
            born_radii[i] = 1.0 / radii[i]
        
        # Adjust Born radii based on atom overlaps (simplified)
        for i in range(n_atoms):
            r_i = radii[i]
            c_i = coords[i]
            
            for j in range(n_atoms):
                if i == j:
                    continue
                
                r_j = radii[j]
                c_j = coords[j]
                
                # Calculate distance
                d_ij = np.linalg.norm(c_i - c_j)
                
                # Skip pairs that are too far apart
                if d_ij > r_i + r_j + 5.0:
                    continue
                
                # Calculate contribution to Born radius
                if d_ij < 0.1:  # Avoid numerical issues
                    continue
                
                # Simplified Still formula
                born_term = r_j / (d_ij * d_ij) * np.exp(-d_ij * d_ij / (4.0 * r_i * r_j))
                born_radii[i] += born_term
        
        # Convert summed terms to actual Born radii
        for i in range(n_atoms):
            if born_radii[i] > 0:
                born_radii[i] = 1.0 / (born_radii[i] * self.scale_factor)
            else:
                born_radii[i] = radii[i]  # Fallback to atom radius
        
        return born_radii
    
    def _calculate_polar_energy(self, atom_list, born_radii):
        """
        Calculate polar solvation energy using GB model.
        
        Parameters:
        -----------
        atom_list : list
            List of (coords, charge, radius) tuples
        born_radii : list
            List of Born radii for each atom
        
        Returns:
        --------
        float
            Polar solvation energy in kcal/mol
        """
        n_atoms = len(atom_list)
        polar_energy = 0.0
        
        # Extract charges and coordinates
        coords = np.array([atom[0] for atom in atom_list])
        charges = np.array([atom[1] for atom in atom_list])
        
        # Calculate energy
        for i in range(n_atoms):
            q_i = charges[i]
            r_i = born_radii[i]
            c_i = coords[i]
            
            # Self-energy term
            self_energy = -166.0 * (q_i * q_i) / (2.0 * r_i) * (1.0 - (1.0 / self.solvent_dielectric))
            polar_energy += self_energy
            
            # Cross-terms
            for j in range(i+1, n_atoms):
                q_j = charges[j]
                r_j = born_radii[j]
                c_j = coords[j]
                
                d_ij = np.linalg.norm(c_i - c_j)
                
                if d_ij < 0.1:  # Avoid numerical issues
                    continue
                
                # "f_GB" term for GB equation
                f_gb = np.sqrt(d_ij * d_ij + r_i * r_j * np.exp(-d_ij * d_ij / (4.0 * r_i * r_j)))
                
                # Cross-term energy
                cross_energy = -166.0 * (q_i * q_j) / f_gb * (1.0 - (1.0 / self.solvent_dielectric))
                polar_energy += cross_energy
        
        return polar_energy
    
    def _calculate_nonpolar_energy(self, atom_list):
        """
        Calculate nonpolar solvation energy (cavity formation and van der Waals).
        
        Parameters:
        -----------
        atom_list : list
            List of (coords, charge, radius) tuples
        
        Returns:
        --------
        float
            Nonpolar solvation energy in kcal/mol
        """
        # Extract coordinates and radii
        coords = np.array([atom[0] for atom in atom_list])
        radii = np.array([atom[2] for atom in atom_list])
        
        # Calculate approximate surface area for each atom
        n_atoms = len(atom_list)
        sasa = np.ones(n_atoms)  # Start with fully exposed
        
        # Calculate all pairwise distances
        dist_matrix = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                dist_matrix[i,j] = dist
                dist_matrix[j,i] = dist
        
        # Add probe radius to atomic radii
        radii_with_probe = radii + 1.4  # Water probe radius
        
        # For each atom, check how many neighbors are within van der Waals contact
        for i in range(n_atoms):
            r_i = radii_with_probe[i]
            
            # Count neighbors that overlap
            for j in range(n_atoms):
                if i == j:
                    continue
                    
                r_j = radii[j]  # No probe for neighbors
                dist = dist_matrix[i,j]
                
                # Check if atoms overlap
                if dist < (r_i + r_j):
                    # Estimate overlap
                    overlap = 1.0 - (dist / (r_i + r_j))
                    # Reduce SASA proportionally to overlap
                    sasa[i] -= overlap * 0.2  # Reduced scale factor
        
        # Ensure SASA is non-negative
        sasa = np.maximum(0.0, sasa)
        
        # Calculate surface area (4πr²) and apply exposure factor
        atom_areas = 4.0 * np.pi * radii_with_probe * radii_with_probe * sasa
        
        # Calculate nonpolar energy using surface area model
        nonpolar_energy = self.surface_tension * np.sum(atom_areas)
        
        return nonpolar_energy


class MonteCarloSampling:
    """
    Enhanced sampling using Monte Carlo with Metropolis criterion.
    This provides better exploration of conformational space.
    """
    
    def __init__(self, scoring_function, temperature=300.0, n_steps=1000, 
                 max_translation=2.0, max_rotation=0.3, cooling_factor=0.95, output_dir=None):
        """
        Initialize Monte Carlo sampling.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function to evaluate poses
        temperature : float
            Simulation temperature in Kelvin
        n_steps : int
            Number of Monte Carlo steps
        max_translation : float
            Maximum translation step size in Angstroms
        max_rotation : float
            Maximum rotation step size in radians
        cooling_factor : float
            Temperature cooling factor for simulated annealing (< 1.0)
        """
        self.scoring_function = scoring_function
        self.temperature = temperature
        self.n_steps = n_steps
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        self.cooling_factor = cooling_factor
        self.output_dir = output_dir 
        
        # Gas constant in kcal/(mol·K)
        self.gas_constant = 1.9872e-3
        
        # Set up simulated annealing schedule if cooling factor < 1.0
        if cooling_factor < 1.0:
            self.use_annealing = True
        else:
            self.use_annealing = False
    
    def run_sampling(self, protein, ligand, start_pose=None):
        """
        Run Monte Carlo sampling to explore ligand conformational space.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object (used as starting pose if start_pose is None)
        start_pose : Ligand, optional
            Starting ligand pose
        
        Returns:
        --------
        list
            List of (pose, score) tuples, sorted by score (best first)
        """
        import copy
        from scipy.spatial.transform import Rotation
        import numpy as np
        
        # Use provided starting pose or ligand
        if start_pose is None:
            current_pose = copy.deepcopy(ligand)
        else:
            current_pose = copy.deepcopy(start_pose)
        
        # Evaluate initial pose
        current_score = self.scoring_function.score(protein, current_pose)
        
        # Initialize tracking for acceptance ratio
        accepted = 0
        
        # Initialize best pose and collection of poses
        best_pose = copy.deepcopy(current_pose)
        best_score = current_score
        collected_poses = [(copy.deepcopy(current_pose), current_score)]
        
        # Set initial temperature
        temperature = self.temperature
        
        # Print header for progress tracking
        print(f"Starting Monte Carlo sampling ({self.n_steps} steps)")
        print(f"Initial score: {current_score:.2f}")
        
        # Main sampling loop
        for step in range(self.n_steps):
            # Create a candidate pose
            candidate_pose = copy.deepcopy(current_pose)
            
            # Apply random translation
            translation = np.random.uniform(-self.max_translation, self.max_translation, 3)
            candidate_pose.translate(translation)
            
            # Apply random rotation
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            angle = np.random.uniform(-self.max_rotation, self.max_rotation)
            rotation = Rotation.from_rotvec(axis * angle)
            
            # Apply rotation around center of mass
            centroid = np.mean(candidate_pose.xyz, axis=0)
            candidate_pose.translate(-centroid)
            candidate_pose.rotate(rotation.as_matrix())
            candidate_pose.translate(centroid)
            
            # Evaluate candidate pose
            candidate_score = self.scoring_function.score(protein, candidate_pose)
            
            # Calculate energy difference (negative score = better)
            delta_score = candidate_score - current_score
            
            # Metropolis criterion
            accept = False
            if delta_score <= 0:  # Better score, always accept
                accept = True
            else:
                # Calculate Boltzmann factor
                boltzmann_factor = np.exp(-delta_score / (self.gas_constant * temperature))
                # Accept with probability exp(-ΔE/kT)
                if np.random.random() < boltzmann_factor:
                    accept = True
            
            # Update current pose if accepted
            if accept:
                current_pose = candidate_pose
                current_score = candidate_score
                accepted += 1
                
                # Update best pose if needed
                # Insert after accepting a pose or updating the best pose:
            if current_score < best_score:
                best_pose = copy.deepcopy(current_pose)
                best_score = current_score
                
                # Add the progress tracking code:
                if self.output_dir:
                    from .utils import save_intermediate_result, update_status
                    
                    save_intermediate_result(
                        best_pose,
                        best_score,
                        step + 1,  # Current step
                        self.output_dir,
                        self.n_steps
                    )
                    
                    update_status(
                        self.output_dir,
                        current_step=step + 1,
                        temperature=temperature,
                        best_score=best_score,
                        acceptance_ratio=accepted/(step+1)
                    )
                
                # Add to collection (limit to 100 poses)
                if len(collected_poses) < 100:
                    collected_poses.append((copy.deepcopy(current_pose), current_score))
            
            # Cool temperature if using simulated annealing
            if self.use_annealing:
                temperature *= self.cooling_factor
            
            # Print progress
            if (step + 1) % (self.n_steps // 10) == 0:
                acceptance_ratio = accepted / (step + 1)
                print(f"Step {step + 1}/{self.n_steps}, "
                      f"Score: {current_score:.2f}, "
                      f"Best: {best_score:.2f}, "
                      f"Acceptance: {acceptance_ratio:.2f}, "
                      f"Temp: {temperature:.1f}K")
        
        # Final stats
        acceptance_ratio = accepted / self.n_steps
        print(f"Sampling completed. Final score: {current_score:.2f}, "
              f"Best score: {best_score:.2f}, "
              f"Acceptance ratio: {acceptance_ratio:.2f}")
        
        # Sort collected poses by score
        collected_poses.sort(key=lambda x: x[1])
        
        return collected_poses


class PhysicsBasedScoring:
    """
    Physics-based scoring function combining molecular mechanics terms.
    Provides accurate binding energy by integrating physical interaction models.
    """

    def __init__(self, use_gpu=True):
        # GPU flag
        self.use_gpu = use_gpu
        if self.use_gpu:
            print("INFO - Physics-based scoring initialized with GPU")
        else:
            print("INFO - Physics-based scoring initialized without GPU")
        
        # Define hydrophobic atom types
        self.hydrophobic_types = {'C', 'F', 'I', 'Br', 'Cl', 'S'}

        # Initialize physical models
        from .physics import ImprovedElectrostatics, GeneralizedBornSolvation
        self.electrostatics = ImprovedElectrostatics()
        self.solvation = GeneralizedBornSolvation()

        # Component weights (modifiable)
        self.weights = {
            'vdw': 0.3,           # Increased from 0.1662
            'hbond': 0.2,         # Increased from 0.1209
            'elec': 0.2,          # Increased from 0.1406
            'desolv': 0.005,       # Decreased from 0.1322 to reduce domination
            'hydrophobic': 0.2,   # Increased from 0.1418  
            'clash': 1.0,         # Kept the same
            'entropy': 0.25       # Slightly decreased from 0.2983
        }

        # Parameters
        self.vdw_radii = {'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
                          'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98}
        self.vdw_well_depth = {'H': 0.02, 'C': 0.10, 'N': 0.16, 'O': 0.20, 'S': 0.25,
                               'P': 0.20, 'F': 0.08, 'Cl': 0.25, 'Br': 0.32, 'I': 0.40}
        self.hbond_donors = {'N', 'O'}
        self.hbond_acceptors = {'N', 'O', 'F', 'Cl'}
        self.hbond_strength = 5.0
        self.hbond_distance = 3.0
        self.entropy_per_rot_bond = 0.4
        self.entropy_per_nonrot_bond = 0.2


    def score(self, protein, ligand):
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms

        vdw = self._calc_vdw_energy(protein_atoms, ligand.atoms)
        hbond = self._calc_hbond_energy(protein_atoms, ligand.atoms)
        elec = self.electrostatics.calculate_electrostatics(protein, ligand)
        desolv = self.solvation.calculate_binding_solvation(protein, ligand)
        entropy = self._calc_entropy_penalty(ligand)
        clash = self._calc_clash_energy(protein_atoms, ligand.atoms)
        hydrophobic = self._calc_hydrophobic_interactions(protein_atoms, ligand.atoms)


        non_rotatable_bonds = len(getattr(ligand, 'non_rotatable_bonds', []))
        entropy_penalty = non_rotatable_bonds * self.entropy_per_nonrot_bond

        total = (
            self.weights['vdw'] * vdw +
            self.weights['hbond'] * hbond +
            self.weights['elec'] * elec +
            self.weights['desolv'] * desolv +
            self.weights['entropy'] * (entropy) +
            self.weights['hydrophobic'] * hydrophobic +
            self.weights['clash'] * clash
        )
        return total

    def _calc_vdw_energy(self, protein_atoms, ligand_atoms):
        energy = 0.0
        for p in protein_atoms:
            p_coords = p['coords']
            p_type = p.get('element', p.get('name', 'C'))[0]
            pr = self.vdw_radii.get(p_type, 1.7)
            pd = self.vdw_well_depth.get(p_type, 0.1)
            for l in ligand_atoms:
                l_coords = l['coords']
                l_type = l.get('symbol', 'C')
                lr = self.vdw_radii.get(l_type, 1.7)
                ld = self.vdw_well_depth.get(l_type, 0.1)
                d = np.linalg.norm(p_coords - l_coords)
                if d > 10.0:
                    continue
                sigma = (pr + lr) * 0.5
                epsilon = np.sqrt(pd * ld)
                if d < 0.1:
                    energy += 1000
                else:
                    r = sigma / d
                    energy += epsilon * (r**12 - 2 * r**6)
        return energy

    def _calc_hbond_energy(self, protein_atoms, ligand_atoms):
        e = 0.0
        p_donors = [a for a in protein_atoms if a.get('element', a.get('name', 'C'))[0] in self.hbond_donors]
        p_acceptors = [a for a in protein_atoms if a.get('element', a.get('name', 'C'))[0] in self.hbond_acceptors]
        l_donors = [a for a in ligand_atoms if a.get('symbol', 'C') in self.hbond_donors]
        l_acceptors = [a for a in ligand_atoms if a.get('symbol', 'C') in self.hbond_acceptors]
        for d in p_donors:
            for a in l_acceptors:
                d_c, a_c = d['coords'], a['coords']
                dist = np.linalg.norm(d_c - a_c)
                if dist <= self.hbond_distance:
                    f = 1.0 - (dist / self.hbond_distance)
                    e += -self.hbond_strength * f**2
        for d in l_donors:
            for a in p_acceptors:
                d_c, a_c = d['coords'], a['coords']
                dist = np.linalg.norm(d_c - a_c)
                if dist <= self.hbond_distance:
                    f = 1.0 - (dist / self.hbond_distance)
                    e += -self.hbond_strength * f**2
        return e

    def _calc_entropy_penalty(self, ligand, protein=None):
        n_rotatable = len(getattr(ligand, 'rotatable_bonds', []))
        n_atoms = len(ligand.atoms)
        flexibility = self._estimate_pose_restriction(ligand, protein)
        entropy_penalty = 0.5 * n_rotatable * flexibility * (1.0 + 0.05 * n_atoms)
        return entropy_penalty

    
    def _estimate_pose_restriction(self, ligand, protein=None):
        """
        Estimate pose-specific conformational restriction.
        Returns a factor between 0 (fully restricted) and 1 (fully flexible).
        Currently based on fraction of ligand atoms buried in protein.
        
        Parameters:
        -----------
        ligand : Ligand
        protein : Protein (optional)
        
        Returns:
        --------
        float : Restriction factor (0.0 to 1.0)
        """
        if not protein or not hasattr(protein, 'active_site'):
            return 0.5  # Fallback if no protein info

        ligand_coords = np.array([atom['coords'] for atom in ligand.atoms])
        protein_coords = np.array([atom['coords'] for atom in protein.active_site['atoms']])

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

    def _calc_clash_energy(self, protein_atoms, ligand_atoms, cutoff=2.0, penalty=5.0):
        count = 0
        for p in protein_atoms:
            for l in ligand_atoms:
                if np.linalg.norm(np.array(p['coords']) - np.array(l['coords'])) < cutoff:
                    count += 1
        energy = min(count * penalty, 100.0)
        return max(energy, 0.0)

    def _calc_hydrophobic_interactions(self, protein_atoms, ligand_atoms, cutoff=4.5):
        """Calculate hydrophobic interactions between hydrophobic atoms."""
        score = 0.0
        
        # Identify hydrophobic atoms
        p_hydrophobic = [a for a in protein_atoms if a.get('element', a.get('name', 'C'))[0] in self.hydrophobic_types]
        l_hydrophobic = [a for a in ligand_atoms if a.get('symbol', 'C') in self.hydrophobic_types]
        
        # Calculate interactions
        for p in p_hydrophobic:
            p_coords = p['coords']
            for l in l_hydrophobic:
                l_coords = l['coords']
                dist = np.linalg.norm(np.array(p_coords) - np.array(l_coords))
                
                if dist <= cutoff:
                    contact_score = (cutoff - dist) / cutoff
                    score += contact_score
        
        # Apply capping
        max_hydrophobic = 200.0
        capped_score = min(score, max_hydrophobic)
        
        return capped_score




# Standardized energy component interfaces for compatibility with reporter
    def _calculate_electrostatics_physics(self, protein_atoms, ligand_atoms):
        from types import SimpleNamespace
        protein = SimpleNamespace(active_site={'atoms': protein_atoms})
        ligand = SimpleNamespace(atoms=ligand_atoms)
        return self.electrostatics.calculate_electrostatics(protein, ligand)

    def _calculate_desolvation_physics(self, protein_atoms, ligand_atoms):
        from types import SimpleNamespace
        protein = SimpleNamespace(active_site={'atoms': protein_atoms})
        ligand = SimpleNamespace(atoms=ligand_atoms)
        return self.solvation.calculate_binding_solvation(protein, ligand)

    def _calculate_vdw_physics(self, protein_atoms, ligand_atoms):
        return self._calc_vdw_energy(protein_atoms, ligand_atoms)

    def _calculate_hbond_physics(self, protein_atoms, ligand_atoms, protein, ligand):
        return self._calc_hbond_energy(protein_atoms, ligand_atoms)

    def _calculate_entropy(self, ligand, protein=None):
        return self._calc_entropy_penalty(ligand, protein)

    def _calculate_entropy_penalty(self, ligand, protein=None):
        return self._calc_entropy_penalty(ligand, protein)

    def _calculate_clashes_physics(self, protein_atoms, ligand_atoms, protein=None, ligand=None):
        return self._calc_clash_energy(protein_atoms, ligand_atoms)

    def _calculate_hydrophobic_physics(self, protein_atoms, ligand_atoms):
        return self._calc_hydrophobic_interactions(protein_atoms, ligand_atoms)


        
class PhysicsBasedScoringFunction(PhysicsBasedScoring):
    """
    Physics-based scoring function using calibrated energy terms and parameters.
    Implements a comprehensive free energy model for protein-ligand binding.
    """
    
    def __init__(self):
        super().__init__()
        
        # Override with physics-based calibrated weights
        self.weights = {
            'vdw': 0.3,           # Increased from 0.1662
            'hbond': 0.2,         # Increased from 0.1209
            'elec': 0.2,          # Increased from 0.1406
            'desolv': 0.005,       # Decreased from 0.1322 to reduce domination
            'hydrophobic': 0.2,   # Increased from 0.1418  
            'clash': 1.0,         # Kept the same
            'entropy': 0.25       # Slightly decreased from 0.2983
        }
        
        # Extended atom type parameters for more precise interactions
        self.atom_type_map = {
            # Carbon types (more specific than just 'C')
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
        
        # H-bond parameters (12-10 potential)
        self.hbond_params = {
            # Donor-Acceptor type pairs (r_eq in Å, epsilon in kcal/mol)
            'O-O': {'r_eq': 1.90, 'epsilon': 5.0},
            'O-N': {'r_eq': 1.90, 'epsilon': 5.0},
            'N-O': {'r_eq': 1.90, 'epsilon': 5.0},
            'N-N': {'r_eq': 1.90, 'epsilon': 5.0},
            'O-S': {'r_eq': 2.50, 'epsilon': 1.0},
            'N-S': {'r_eq': 2.50, 'epsilon': 1.0},
        }
        
        # Define H-bond donors and acceptors more precisely
        self.hbond_donor_types = {'N', 'NA', 'O', 'OA', 'N.3', 'N.am', 'O.3'}
        self.hbond_acceptor_types = {'O', 'OA', 'N', 'NA', 'SA', 'O.2', 'O.3', 'N.2'}
        
        # Update hydrophobic types
        self.hydrophobic_types = ['C', 'A', 'F', 'Cl', 'Br', 'I', 'C.3', 'C.2', 'C.ar']
        
        # Atomic solvation parameters (kcal/mol·Å²)
        self.atom_solvation_params = {
            'C': 0.4,    # Reduced from 12.77
            'A': 0.4,    # Reduced from 12.77
            'N': -1.5,   # Less negative from -17.40
            'NA': -1.5,  # Less negative
            'O': -1.5,   # Less negative from -17.40
            'OA': -1.5,  # Less negative
            'S': -0.8,   # Less negative from -8.31
            'SA': -0.8,  # Less negative
            'H': 0.0,    # Unchanged
            'F': -0.5,   # Less negative from -6.60
            'Cl': -0.1,  # Slightly adjusted from -0.72
            'Br': -0.1,  # Slightly adjusted from -0.85
            'I': -0.1,   # Slightly adjusted from -0.88
            'P': -0.7,   # Less negative from -6.70
        }
        
        # Atom volume parameters (Å³)
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
        self.solpar = 0.005   # kcal/mol per Å²
        self.solvation_k = 3.5  # Å, solvation radius
        
        # Distance cutoffs
        self.vdw_cutoff = 8.0  # Å for van der Waals
        self.hbond_cutoff = 4.0  # Å for hydrogen bonds
        self.elec_cutoff = 20.0  # Å for electrostatics
        self.desolv_cutoff = 8.0  # Å for desolvation
        
    def _get_atom_type(self, atom, default='C'):
        """Determine the atom type for an atom based on available information."""
        # Try to get atom type from atom data
        atom_type = atom.get('type', None)
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
    
    def score(self, protein, ligand):
        """Calculate comprehensive physics-based binding score."""
        # Get protein and ligand atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
            
        ligand_atoms = ligand.atoms
        
        # Calculate all energy terms
        vdw_energy = self._calculate_vdw_physics(protein_atoms, ligand_atoms)
        hbond_energy = self._calculate_hbond_physics(protein_atoms, ligand_atoms, protein, ligand)
        elec_energy = self._calculate_electrostatics_physics(protein_atoms, ligand_atoms)
        desolv_energy = self._calculate_desolvation_physics(protein_atoms, ligand_atoms)
        entropy_energy = self._calculate_entropy(ligand_atoms, protein_atoms)
        hydrophobic = self._calc_hydrophobic_interactions(protein_atoms, ligand.atoms)
        clash = self._calculate_clashes_physics(protein_atoms, ligand_atoms, protein, ligand)
        
        # Combine scores with calibrated weights
        total_score = (
            self.weights['vdw'] * vdw_energy +
            self.weights['hbond'] * hbond_energy +
            self.weights['elec'] * elec_energy +
            self.weights['desolv'] * desolv_energy +
            self.weights['entropy'] * entropy_energy +
            self.weights['clash'] *  clash +
            self.weights['hydrophobic'] * hydrophobic
        )
        
        return total_score
    
    def _calculate_vdw_physics(self, protein_atoms, ligand_atoms):
        """
        Calculate van der Waals energy using a modified 12-6 Lennard-Jones potential
        with atom-specific parameters and smoothing function for close contacts.
        """
        vdw_energy = 0.0
        
        for p_atom in protein_atoms:
            p_type = self._get_atom_type(p_atom)
            p_coords = p_atom['coords']
            p_params = self.vdw_params.get(p_type, self.vdw_params['C'])
            
            for l_atom in ligand_atoms:
                l_type = self._get_atom_type(l_atom)
                l_coords = l_atom['coords']
                l_params = self.vdw_params.get(l_type, self.vdw_params['C'])
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Skip if beyond cutoff
                if distance > self.vdw_cutoff:
                    continue
                
                # Calculate combined parameters
                r_eq = (p_params['r_eq'] + l_params['r_eq']) / 2.0  # Arithmetic mean
                epsilon = np.sqrt(p_params['epsilon'] * l_params['epsilon'])  # Geometric mean
                
                # Prevent division by zero
                if distance < 0.1:
                    distance = 0.1
                
                # Calculate ratio for efficiency
                ratio = r_eq / distance
                
                # Use modified potential with smoothing at close distances
                if distance >= 0.5 * r_eq:
                    # Regular 12-6 Lennard-Jones
                    vdw_term = epsilon * ((ratio**12) - 2.0 * (ratio**6))
                    vdw_term = min(max(vdw_term, -50.0), 50.0)
                else:
                    # Smoothed function for very close distances
                    smoothed_ratio = 0.5 * r_eq / distance
                    vdw_term = epsilon * ((smoothed_ratio**12) - 2.0 * (smoothed_ratio**6))
                    vdw_term = min(max(vdw_term, -50.0), 50.0)
                
                vdw_energy += vdw_term
        
        return vdw_energy
    
    def _calculate_hbond_physics(self, protein_atoms, ligand_atoms, protein, ligand):
        """
        Calculate hydrogen bonding using a 12-10 potential with angular dependence.
        """
        hbond_energy = 0.0
        
        # Check for protein donor - ligand acceptor pairs
        for p_atom in protein_atoms:
            p_type = self._get_atom_type(p_atom)
            p_coords = p_atom['coords']
            p_element = p_atom.get('element', p_atom.get('name', 'C'))[0].upper()
            
            for l_atom in ligand_atoms:
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
                    # Get H-bond parameters
                    hb_key = f"{p_element}-{l_element}"
                    
                    # Look up parameters or use defaults
                    if hb_key in self.hbond_params:
                        params = self.hbond_params[hb_key]
                    else:
                        # Default parameters for this pair
                        params = {'r_eq': 1.9, 'epsilon': 3.0}
                    
                    # Get parameters
                    r_eq = params['r_eq']
                    epsilon = params['epsilon']
                    
                    # 12-10 potential
                    if distance < 0.1:
                        distance = 0.1
                    
                    # Calculate 12-10 energy term (ideal at r_eq)
                    ratio = r_eq / distance
                    hbond_term = epsilon * ((ratio**12) - 2.0 * (ratio**10))
                    
                    # Apply angular factor (simplified)
                    angle_factor = self._calculate_hbond_angle_factor(p_atom, l_atom, protein, ligand)
                    hbond_energy += hbond_term * angle_factor
                
                # Ligand donor - Protein acceptor
                if l_element in self.hbond_donor_types and p_element in self.hbond_acceptor_types:
                    # Similar calculation as above with reversed roles
                    hb_key = f"{l_element}-{p_element}"
                    
                    if hb_key in self.hbond_params:
                        params = self.hbond_params[hb_key]
                    else:
                        params = {'r_eq': 1.9, 'epsilon': 3.0}
                    
                    r_eq = params['r_eq']
                    epsilon = params['epsilon']
                    
                    if distance < 0.1:
                        distance = 0.1
                    
                    ratio = r_eq / distance
                    hbond_term = epsilon * ((ratio**12) - 2.0 * (ratio**10))
                    
                    angle_factor = self._calculate_hbond_angle_factor(l_atom, p_atom, ligand, protein)
                    hbond_energy += hbond_term * angle_factor
        
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
            # In a real implementation, you'd use bonding information to calculate precise angles
            return 0.5  # Default 50% effectiveness
            
        except Exception as e:
            return 0.25  # Fallback if calculation fails
    
    def _calculate_electrostatics_physics(self, protein_atoms, ligand_atoms):
        """
        Calculate electrostatic interactions using Coulomb's law with
        distance-dependent dielectric.
        """
        elec_energy = 0.0
        coulomb_constant = 332.0  # Convert to kcal/mol
        
        for p_atom in protein_atoms:
            p_coords = p_atom['coords']
            p_element = p_atom.get('element', p_atom.get('name', 'C'))[0].upper()
            p_charge = self.atom_charges.get(p_element, 0.0)
            
            # Skip atoms with zero charge
            if abs(p_charge) < 1e-6:
                continue
            
            for l_atom in ligand_atoms:
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
                
                # Prevent division by zero
                if distance < 0.1:
                    distance = 0.1
                
                # Calculate distance-dependent dielectric
                # Uses ε(r) = 4r model
                dielectric = 4.0 * distance
                
                # Calculate Coulomb energy
                elec_term = coulomb_constant * p_charge * l_charge / (dielectric * distance)
                elec_energy += elec_term
        
        return elec_energy
    
    def _calculate_desolvation_physics(self, protein_atoms, ligand_atoms):
        """
        Calculate desolvation energy using atomic solvation and volume parameters.
        """
        desolv_energy = 0.0
        sigma = self.solvation_k  # Solvation radius in Å
        sigma_squared_2 = 2.0 * sigma * sigma  # Pre-calculate
        
        for p_atom in protein_atoms:
            p_coords = p_atom['coords']
            p_type = self._get_atom_type(p_atom)
            p_solv = self.atom_solvation_params.get(p_type, 0.0)
            p_vol = self.atom_volume_params.get(p_type, 0.0)
            
            for l_atom in ligand_atoms:
                l_coords = l_atom['coords']
                l_type = self._get_atom_type(l_atom)
                l_solv = self.atom_solvation_params.get(l_type, 0.0)
                l_vol = self.atom_volume_params.get(l_type, 0.0)
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Skip if beyond cutoff
                if distance > self.desolv_cutoff:
                    continue
                
                # Calculate exponential term
                exp_term = np.exp(-(distance*distance) / sigma_squared_2)
                
                # Calculate desolvation energy (volume-weighted)
                desolv_term = (self.solpar * p_solv * l_vol + 
                              self.solpar * l_solv * p_vol) * exp_term
                desolv_term = np.sign(desolv_term) * min(abs(desolv_term), 5.0)
                
                desolv_energy += desolv_term
        
        return desolv_energy
    def _calculate_hydrophobic_physics(self, protein_atoms, ligand_atoms):
        """Calculate hydrophobic interactions between hydrophobic atoms."""
        score = 0.0
        cutoff = 4.5  # Hydrophobic interaction cutoff
        
        # Identify hydrophobic atoms
        p_hydrophobic = [a for a in protein_atoms if self._get_atom_type(a) in self.hydrophobic_types]
        l_hydrophobic = [a for a in ligand_atoms if self._get_atom_type(a) in self.hydrophobic_types]
        
        # Calculate interactions
        for p in p_hydrophobic:
            p_coords = p['coords']
            for l in l_hydrophobic:
                l_coords = l['coords']
                dist = np.linalg.norm(np.array(p_coords) - np.array(l_coords))
                
                if dist <= cutoff:
                    contact_score = (cutoff - dist) / cutoff
                    score += contact_score
        
        # Apply capping
        max_hydrophobic = 200.0
        capped_score = min(score, max_hydrophobic)
        
        return capped_score
    
    def _calculate_clashes_energy(self, protein_atoms, ligand_atoms):
        """Calculate steric clashes between protein and ligand atoms."""
        clash_energy = 0.0
        cutoff = 2.0  # Clash distance cutoff
        penalty = 5.0  # Clash penalty factor
        
        # Count clashes
        count = 0
        for p in protein_atoms:
            p_coords = p['coords']
            p_type = self._get_atom_type(p)
            p_radius = self.vdw_radii.get(p_type[0], 1.7)
            
            for l in ligand_atoms:
                l_coords = l['coords']
                l_type = self._get_atom_type(l)
                l_radius = self.vdw_radii.get(l_type[0], 1.7)
                
                # Calculate distance
                dist = np.linalg.norm(np.array(p_coords) - np.array(l_coords))
                
                # Calculate minimum allowed distance (70% of sum of vdW radii)
                min_allowed = (p_radius + l_radius) * 0.7
                
                if dist < min_allowed:
                    count += 1
        
        # Apply penalty and capping
        clash_energy = count * penalty
        max_clash = 100.0
        capped_clash = min(clash_energy, max_clash)
        
        return capped_clash
    
    

    
    def _calculate_entropy_penalty(self, ligand, protein=None):
        n_rotatable = len(getattr(ligand, 'rotatable_bonds', []))
        n_atoms = len(ligand)
        flexibility = self._estimate_pose_restriction(ligand, protein)
        entropy_penalty = 0.5 * n_rotatable * flexibility * (1.0 + 0.05 * n_atoms)
        return entropy_penalty

    
    def _estimate_pose_restriction(self, ligand, protein=None):
        """
        Estimate pose-specific conformational restriction.
        Returns a factor between 0 (fully restricted) and 1 (fully flexible).
        Currently based on fraction of ligand atoms buried in protein.
        
        Parameters:
        -----------
        ligand : Ligand
        protein : Protein (optional)
        
        Returns:
        --------
        float : Restriction factor (0.0 to 1.0)
        """
        if not protein or not hasattr(protein, 'active_site'):
            return 0.5  # Fallback if no protein info

        ligand_coords = np.array([atom['coords'] for atom in ligand.atoms])
        protein_coords = np.array([atom['coords'] for atom in protein.active_site['atoms']])

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


       