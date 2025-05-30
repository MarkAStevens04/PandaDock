# reporting.py
"""
Comprehensive reporting module for PandaDock.
This module provides detailed reporting functionality for molecular docking results,
including energy breakdowns, RMSD calculations, and visualization capabilities.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pandas as pd
from tabulate import tabulate
import csv


class EnhancedJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types and other special types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (Path, datetime)):
            return str(obj)
        elif isinstance(obj, bool):
            return bool(obj)
        return super().default(obj)
class DockingReporter:
    """
    Comprehensive reporting class for molecular docking results.
    Generates detailed reports with energy breakdowns, visualizations, and validation metrics.
    """
    
    def __init__(self, output_dir, args, timestamp=None):
        """
        Initialize the docking reporter.
        
        Parameters:
        -----------
        output_dir : str
            Output directory path
        args : argparse.Namespace
            Command-line arguments
        timestamp : str, optional
            Timestamp for report identification
        """
        self.output_dir = Path(output_dir)
        self.args = args
        
        # Create timestamp if not provided
        if timestamp is None:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.timestamp = timestamp
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize report data
        self.run_info = {
            "timestamp": self.timestamp,
            "protein": str(args.protein),
            "ligand": str(args.ligand),
            "algorithm": args.algorithm if hasattr(args, 'algorithm') else "unknown",
            "hardware": self._get_hardware_info(args)
        }
        
        # For storing results
        self.results = []
        self.scoring_breakdown = []
        self.validation_results = None
    
    def _get_hardware_info(self, args):
        """Extract hardware configuration information from arguments."""
        hardware = {}
        
        if hasattr(args, 'use_gpu'):
            hardware["use_gpu"] = args.use_gpu
        if hasattr(args, 'gpu_id'):
            hardware["gpu_id"] = args.gpu_id
        if hasattr(args, 'cpu_workers'):
            hardware["cpu_workers"] = args.cpu_workers
        
        return hardware
    
    def add_results(self, results, detailed_energy=None):
        """
        Add docking results to the report.
        
        Parameters:
        -----------
        results : list
            List of (pose, score) tuples
        detailed_energy : list, optional
            List of detailed energy breakdowns as dictionaries
        """
        self.results = results
        
        # If detailed energy information is provided, store it
        if detailed_energy:
            self.scoring_breakdown = detailed_energy
            print(f"DEBUG - Scoring breakdown stored: {self.scoring_breakdown}")
        
        # Sort results by score if not already sorted
        self.results.sort(key=lambda x: x[1])
    
    def add_validation_results(self, validation_results):
        """
        Add validation results to the report.
        
        Parameters:
        -----------
        validation_results : dict or list
            Validation results from comparing to a reference structure
        """
        self.validation_results = validation_results
        
    def extract_energy_components(self, scoring_function, protein, ligand_poses):
        """
        Extract detailed energy components for each pose using the available scoring function.
        This version works with physics-based scoring functions and applies weights correctly.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function used for docking
        protein : Protein
            Protein object
        ligand_poses : list
            List of ligand poses
        
        Returns:
        --------
        list
            List of dictionaries containing energy components for each pose
        """
        print("Starting physics-based energy component extraction...")
        energy_breakdown = []
        
        # Get the class name of the scoring function for debugging
        scoring_class = scoring_function.__class__.__name__
        print(f"Scoring function class: {scoring_class}")
        
        # Check for weights attribute which is required for proper reporting
        has_weights = hasattr(scoring_function, 'weights')
        if has_weights:
            print(f"Scoring function has weights: {scoring_function.weights}")
        else:
            print("Warning: Scoring function has no weights attribute")
            # Create default weights to avoid errors
            scoring_function.weights = {
            'vdw': 1.0, 'hbond': 1.0, 'elec': 1.0, 'desolv': 1.0, 
            'hydrophobic': 1.0, 'clash': 1.0, 'entropy': 1.0
            } 
        
        # Process each pose
        for i, pose in enumerate(ligand_poses):
            print(f"Processing pose {i+1}/{len(ligand_poses)}")
            components = {}
            weighted_components = {}
            
            # Get total score first
            total_score = scoring_function.score(protein, pose)
            print(f"Total score for pose {i+1}: {total_score:.2f}")
            
            # Get protein atoms properly
            if hasattr(protein, 'active_site') and protein.active_site and 'atoms' in protein.active_site:
                protein_atoms = protein.active_site['atoms']
                print(f"Using {len(protein_atoms)} atoms from active site")
            else:
                protein_atoms = protein.atoms
                print(f"Using all {len(protein_atoms)} protein atoms")
            
            # Calculate VDW energy
            vdw_energy = None
            for method_name in ['_calc_vdw_energy', '_calculate_vdw_physics', '_calculate_vdw_energy']:
                if hasattr(scoring_function, method_name):
                    try:
                        method = getattr(scoring_function, method_name)
                        vdw_energy = method(protein_atoms, pose.atoms)
                        components['Van der Waals'] = vdw_energy
                        weighted_components['Van der Waals'] = vdw_energy * scoring_function.weights.get('vdw', 1.0)
                        print(f"VDW energy: {vdw_energy:.2f} (weighted: {weighted_components['Van der Waals']:.2f})")
                        break
                    except Exception as e:
                        print(f"Error calculating VDW with {method_name}: {str(e)}")
            
            # If no method worked, use placeholder
            if 'Van der Waals' not in components:
                components['Van der Waals'] = 0.0
                weighted_components['Van der Waals'] = 0.0
            
            # Similar pattern for H-Bond energy
            hbond_energy = None
            for method_name in ['_calc_hbond_energy', '_calculate_hbond_physics', '_calculate_hbond_energy']:
                if hasattr(scoring_function, method_name):
                    try:
                        method = getattr(scoring_function, method_name)
                        if method_name == '_calculate_hbond_physics':
                            hbond_energy = method(protein_atoms, pose.atoms, protein, pose)
                        else:
                            hbond_energy = method(protein_atoms, pose.atoms)
                        components['H-Bond'] = hbond_energy
                        weighted_components['H-Bond'] = hbond_energy * scoring_function.weights.get('hbond', 1.0)
                        print(f"H-Bond energy: {hbond_energy:.2f} (weighted: {weighted_components['H-Bond']:.2f})")
                        break
                    except Exception as e:
                        print(f"Error calculating H-Bond with {method_name}: {str(e)}")
            
            if 'H-Bond' not in components:
                components['H-Bond'] = 0.0
                weighted_components['H-Bond'] = 0.0
            
            # Electrostatics 
            elec_energy = None
            if hasattr(scoring_function, 'electrostatics') and hasattr(scoring_function.electrostatics, 'calculate_electrostatics'):
                try:
                    elec_energy = scoring_function.electrostatics.calculate_electrostatics(protein, pose)
                except Exception as e:
                    print(f"Error with electrostatics object: {str(e)}")
            
            if elec_energy is None:
                for method_name in ['_calculate_electrostatics_physics', '_calc_electrostatics']:
                    if hasattr(scoring_function, method_name):
                        try:
                            method = getattr(scoring_function, method_name)
                            elec_energy = method(protein_atoms, pose.atoms)
                            break
                        except Exception as e:
                            print(f"Error calculating Electrostatics with {method_name}: {str(e)}")
            
            if elec_energy is not None:
                components['Electrostatic'] = elec_energy
                weighted_components['Electrostatic'] = elec_energy * scoring_function.weights.get('elec', 1.0)
                print(f"Electrostatic energy: {elec_energy:.2f} (weighted: {weighted_components['Electrostatic']:.2f})")
            else:
                components['Electrostatic'] = 0.0
                weighted_components['Electrostatic'] = 0.0
            
            # Continue with other energy components similarly...
            # Desolvation
            desolv_energy = None
            # Try with solvation object first
            if hasattr(scoring_function, 'solvation') and hasattr(scoring_function.solvation, 'calculate_binding_solvation'):
                try:
                    desolv_energy = scoring_function.solvation.calculate_binding_solvation(protein, pose)
                except Exception as e:
                    print(f"Error with solvation object: {str(e)}")
            
            # Try direct methods if object approach didn't work
            if desolv_energy is None:
                for method_name in ['_calculate_desolvation_physics', '_calc_desolvation']:
                    if hasattr(scoring_function, method_name):
                        try:
                            method = getattr(scoring_function, method_name)
                            desolv_energy = method(protein_atoms, pose.atoms)
                            break
                        except Exception as e:
                            print(f"Error calculating Desolvation with {method_name}: {str(e)}")
            
            if desolv_energy is not None:
                components['Desolvation'] = desolv_energy
                weighted_components['Desolvation'] = desolv_energy * scoring_function.weights.get('desolv', 1.0)
                print(f"Desolvation energy: {desolv_energy:.2f} (weighted: {weighted_components['Desolvation']:.2f})")
            else:
                components['Desolvation'] = 0.0
                weighted_components['Desolvation'] = 0.0
            
            # Hydrophobic
            hydrophobic_energy = None
            for method_name in ['_calculate_hydrophobic_physics', '_calc_hydrophobic_interactions']:
                if hasattr(scoring_function, method_name):
                    try:
                        method = getattr(scoring_function, method_name)
                        hydrophobic_energy = method(protein_atoms, pose.atoms)
                        break
                    except Exception as e:
                        print(f"Error calculating Hydrophobic with {method_name}: {str(e)}")
            
            if hydrophobic_energy is not None:
                components['Hydrophobic'] = hydrophobic_energy
                weighted_components['Hydrophobic'] = hydrophobic_energy * scoring_function.weights.get('hydrophobic', 1.0)
                print(f"Hydrophobic energy: {hydrophobic_energy:.2f} (weighted: {weighted_components['Hydrophobic']:.2f})")
            else:
                components['Hydrophobic'] = 0.0
                weighted_components['Hydrophobic'] = 0.0
            
            # Clash
            clash_energy = None
            for method_name in ['_calculate_clashes_physics', '_calc_clash_energy']:
                if hasattr(scoring_function, method_name):
                    try:
                        method = getattr(scoring_function, method_name)
                        clash_energy = method(protein_atoms, pose.atoms)
                        break
                    except Exception as e:
                        print(f"Error calculating Clash with {method_name}: {str(e)}")
            
            if clash_energy is not None:
                components['Clash'] = clash_energy
                weighted_components['Clash'] = clash_energy * scoring_function.weights.get('clash', 1.0)
                print(f"Clash energy: {clash_energy:.2f} (weighted: {weighted_components['Clash']:.2f})")
            else:
                components['Clash'] = 0.0
                weighted_components['Clash'] = 0.0
            
            # Entropy
            entropy_energy = None
            for method_name in ['_calculate_entropy', '_calc_entropy_penalty']:
                if hasattr(scoring_function, method_name):
                    try:
                        method = getattr(scoring_function, method_name)
                        entropy_energy = method(pose)
                        break
                    except Exception as e:
                        print(f"Error calculating Entropy with {method_name}: {str(e)}")
            
            if entropy_energy is not None:
                components['Entropy'] = entropy_energy
                weighted_components['Entropy'] = entropy_energy * scoring_function.weights.get('entropy', 1.0)
                print(f"Entropy energy: {entropy_energy:.2f} (weighted: {weighted_components['Entropy']:.2f})")
            else:
                components['Entropy'] = 0.0
                weighted_components['Entropy'] = 0.0
            
            # Add total score
            components['Total'] = total_score
            weighted_components['Total'] = total_score
            
            # Store both raw and weighted components
            result = {
                'raw': components,
                'weighted': weighted_components
            }
            energy_breakdown.append(result)
            # For any missing components, leave them out rather than estimate
            missing_components = []
            for component in ['Van der Waals', 'H-Bond', 'Electrostatic', 'Desolvation', 'Hydrophobic', 'Entropy', 'Clash']:
                if component not in components:
                    missing_components.append(component)
            
            if missing_components:
                print(f"Missing components: {missing_components}")
                # Do not estimate missing components as this leads to errors
            
            # Add total score
            components['Total'] = total_score
            weighted_components['Total'] = total_score
            
            print(f"Total score: {total_score:.2f}")
            # Store both raw and weighted components
            result = {
                'raw': components,
                'weighted': weighted_components
            }
            energy_breakdown.append(result)
        # After calling extract_energy_components
            
        return energy_breakdown
        
    
    def generate_basic_report(self):
        """
        Generate a basic text report for docking results.
        
        Returns:
        --------
        str
            Path to the generated report file
        """
        report_path = self.output_dir / "docking_report.txt"
        
        # Extract needed information
        protein_path = self.args.protein
        ligand_path = self.args.ligand
        algorithm = self.args.algorithm if hasattr(self.args, 'algorithm') else "unknown"
        iterations = self.args.iterations if hasattr(self.args, 'iterations') else "unknown"
        
        # Calculate elapsed time if available
        if hasattr(self.args, 'start_time'):
            elapsed_time = datetime.now().timestamp() - self.args.start_time
        else:
            elapsed_time = 0.0
        
        # Sort results by score
        sorted_results = sorted(self.results, key=lambda x: x[1])
        
        with open(report_path, 'w') as f:
            f.write("=====================================================\n")
            f.write("        PandaDock - Python Molecular Docking Results    \n")
            f.write("=====================================================\n\n")
            
            # Write run information
            f.write("RUN INFORMATION\n")
            f.write("--------------\n")
            f.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Protein: {protein_path}\n")
            f.write(f"Ligand: {ligand_path}\n")
            f.write(f"Algorithm: {algorithm}\n")
            f.write(f"Iterations/Generations: {iterations}\n")
            f.write(f"Total Runtime: {elapsed_time:.2f} seconds\n\n")
            
            # Write summary of results
            f.write("RESULTS SUMMARY\n")
            f.write("--------------\n")
            f.write(f"Total Poses Generated: {len(self.results)}\n")
            f.write(f"Best Score: {sorted_results[0][1]:.4f}\n")
            f.write(f"Worst Score: {sorted_results[-1][1]:.4f}\n")
            f.write(f"Average Score: {sum([score for _, score in self.results])/len(self.results):.4f}\n\n")
            
            # Write top 10 poses
            f.write("TOP 10 POSES\n")
            f.write("--------------\n")
            f.write("Rank\tScore\tFile\n")
            for i, (pose, score) in enumerate(sorted_results[:10]):
                f.write(f"{i+1}\t{score:.4f}\tpose_{i+1}_score_{score:.1f}.pdb\n")
            
            f.write("\n\nFull results are available in the output directory.\n")
            f.write("=====================================================\n")
        
        print(f"Basic report written to {report_path}")
        return report_path
    
    # Fix for the formatting issues in energy breakdown reporting

    def generate_detailed_report(self, include_energy_breakdown=True):
        """
        Generate a detailed report for docking results with energy breakdowns.
        
        Parameters:
        -----------
        include_energy_breakdown : bool
            Whether to include detailed energy breakdowns
        
        Returns:
        --------
        str
            Path to the generated report file
        """
        report_path = self.output_dir / "detailed_docking_report.txt"
        
        # Extract needed information
        protein_path = self.args.protein
        ligand_path = self.args.ligand
        algorithm = self.args.algorithm if hasattr(self.args, 'algorithm') else "unknown"
        
        # Get algorithm parameters
        algorithm_params = {}
        if algorithm == "genetic":
            if hasattr(self.args, 'population_size'):
                algorithm_params["Population Size"] = self.args.population_size
            if hasattr(self.args, 'mutation_rate'):
                algorithm_params["Mutation Rate"] = self.args.mutation_rate
        elif algorithm == "monte-carlo":
            if hasattr(self.args, 'mc_steps'):
                algorithm_params["Steps"] = self.args.mc_steps
            if hasattr(self.args, 'temperature'):
                algorithm_params["Temperature"] = f"{self.args.temperature} K"
        
        # Get scoring function details
        scoring_type = "standard"
        if hasattr(self.args, 'enhanced_scoring') and self.args.enhanced_scoring:
            scoring_type = "enhanced"
        if hasattr(self.args, 'physics_based') and self.args.physics_based:
            scoring_type = "physics-based"
        
        # Sort results by score
        sorted_results = sorted(self.results, key=lambda x: x[1])
        
        with open(report_path, 'w') as f:
            
            
            # Write run information
            f.write("RUN INFORMATION\n")
            f.write("--------------\n")
            f.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Report ID: {self.timestamp}\n")
            f.write(f"Protein: {protein_path}\n")
            f.write(f"Ligand: {ligand_path}\n\n")
            
            # Algorithm details
            f.write("ALGORITHM DETAILS\n")
            f.write("-----------------\n")
            f.write(f"Algorithm: {algorithm.capitalize()}\n")
            for param, value in algorithm_params.items():
                f.write(f"{param}: {value}\n")
            f.write(f"Scoring Function: {scoring_type.capitalize()}\n")
            
            if hasattr(self.args, 'use_gpu') and self.args.use_gpu:
                f.write(f"Hardware Acceleration: GPU (ID: {getattr(self.args, 'gpu_id', 0)})\n")
                if hasattr(self.args, 'gpu_precision'):
                    f.write(f"GPU Precision: {self.args.gpu_precision}\n")
            else:
                cpu_workers = getattr(self.args, 'cpu_workers', 'all available')
                f.write(f"Hardware Acceleration: CPU ({cpu_workers} cores)\n")
            
            if hasattr(self.args, 'exhaustiveness'):
                f.write(f"Exhaustiveness: {self.args.exhaustiveness}\n")
            
            if hasattr(self.args, 'local_opt') and self.args.local_opt:
                f.write("Local Optimization: Enabled\n")
            
            f.write("\n")
            
            # Write summary of results
            f.write("RESULTS SUMMARY\n")
            f.write("--------------\n")
            f.write(f"Total Poses Generated: {len(self.results)}\n")
            f.write(f"Best Score: {sorted_results[0][1]:.4f}\n")
            f.write(f"Worst Score: {sorted_results[-1][1]:.4f}\n")
            f.write(f"Average Score: {sum([score for _, score in self.results])/len(self.results):.4f}\n")
            f.write(f"Score Standard Deviation: {np.std([score for _, score in self.results]):.4f}\n\n")
            
            # Write top poses
            f.write("TOP POSES RANKING\n")
            f.write("----------------\n")
            f.write("Rank  Score      Filename\n")
            f.write("----- ---------- ----------------------------------------\n")
            for i, (pose, score) in enumerate(sorted_results[:min(20, len(sorted_results))]):
                f.write(f"{i+1:5d} {score:10.4f} pose_{i+1}_score_{score:.1f}.pdb\n")
            f.write("\n")
            
            # Write energy breakdown if available
            if include_energy_breakdown and self.scoring_breakdown:
                print(f"DEBUG - Writing energy breakdown: {self.scoring_breakdown}")
                f.write("ENERGY COMPONENT BREAKDOWN (TOP 10 POSES)\n")
                f.write("----------------------------------------\n")
                
                # FIX: Check proper structure of scoring_breakdown
                if len(self.scoring_breakdown) > 0:
                    # Create fixed header
                    components = ['Van der Waals', 'Entropy', 'H-Bond', 'Electrostatic', 
                                'Desolvation', 'Hydrophobic', 'Clash', 'Total']
                    
                    # Write header line
                    header = "Pose  " + " ".join([f"{comp[:8]:>10}" for comp in components])
                    f.write(header + "\n")
                    
                    # Write separator line with proper length
                    separator = "-" * len(header)
                    f.write(separator + "\n")
                    
                    # Write each pose data
                    for i, breakdown in enumerate(self.scoring_breakdown[:min(10, len(self.scoring_breakdown))]):
                        # Check for proper structure ('raw' and 'weighted' dictionaries)
                        if isinstance(breakdown, dict) and 'weighted' in breakdown:
                            weighted = breakdown['weighted']
                            
                            # Get component values with fallbacks to 0.0
                            row_values = []
                            for comp in components:
                                # Find value for this component
                                value = 0.0
                                
                                # Try different key permutations
                                if comp in weighted:
                                    value = weighted[comp]
                                elif comp.lower() in weighted:
                                    value = weighted[comp.lower()]
                                elif comp.replace(' ', '') in weighted:
                                    value = weighted[comp.replace(' ', '')]
                                elif comp.replace(' ', '_') in weighted:
                                    value = weighted[comp.replace(' ', '_')]
                                
                                row_values.append(value)
                            
                            # Format the row
                            row = f"{i+1:4d} " + " ".join([f"{val:10.2f}" for val in row_values])
                            f.write(row + "\n")
                        else:
                            # Fallback for unexpected data structure
                            f.write(f"{i+1:4d}  No detailed energy data available in proper format\n")
                    
                    f.write("\n")
                else:
                    f.write("No energy breakdown data available.\n\n")
            
            # Write validation results if available
            if self.validation_results:
                f.write("VALIDATION AGAINST REFERENCE STRUCTURE\n")
                f.write("------------------------------------\n")
                
                if isinstance(self.validation_results, list):
                    # List of validation results for multiple poses
                    f.write("RMSD Results for Top Poses:\n")
                    f.write("Rank  Pose     RMSD (Å)  Status\n")
                    f.write("----- -------- --------- --------------\n")
                    
                    for i, result in enumerate(self.validation_results[:min(10, len(self.validation_results))]):
                        pose_idx = result.get('pose_index', i)
                        rmsd = result.get('rmsd', 0.0)
                        status = "Success" if rmsd < 2.0 else "Failure"
                        f.write(f"{i+1:5d} {pose_idx+1:8d} {rmsd:9.4f} {status}\n")
                    
                    # Best RMSD
                    if len(self.validation_results) > 0:
                        best_rmsd = min(result.get('rmsd', float('inf')) for result in self.validation_results)
                        f.write(f"\nBest RMSD: {best_rmsd:.4f} Å\n")
                        f.write(f"Docking Success: {'Yes' if best_rmsd < 2.0 else 'No'}\n")
                
                elif isinstance(self.validation_results, dict):
                    # Single validation result
                    rmsd = self.validation_results.get('rmsd', 0.0)
                    max_dev = self.validation_results.get('max_deviation', 0.0)
                    min_dev = self.validation_results.get('min_deviation', 0.0)
                    success = self.validation_results.get('success', False)
                    
                    f.write(f"Overall RMSD: {rmsd:.4f} Å\n")
                    f.write(f"Maximum Atomic Deviation: {max_dev:.4f} Å\n")
                    f.write(f"Minimum Atomic Deviation: {min_dev:.4f} Å\n")
                    f.write(f"Docking Success: {'Yes' if success else 'No'}\n")
                    
                    # Report per-atom deviations if available
                    atom_deviations = self.validation_results.get('atom_deviations', None)
                    if atom_deviations is not None:
                        f.write("\nTop 10 Atom Deviations:\n")
                        sorted_indices = np.argsort(atom_deviations)[::-1]
                        for i in range(min(10, len(sorted_indices))):
                            idx = sorted_indices[i]
                            f.write(f"Atom {idx + 1}: {atom_deviations[idx]:.4f} Å\n")
                
                f.write("\n")
            
            f.write("===========================================================\n")
            f.write("Report generated by PandaDock - Python Molecular Docking Tool\n")
        
        print(f"Detailed report written to {report_path}")
        return report_path
    
    def generate_csv_report(self):
        """
        Generate CSV files with docking results and energy breakdowns.
        
        Returns:
        --------
        tuple
            Paths to the generated CSV files (results_csv, energy_csv)
        """
        # Create paths for CSV files
        results_csv = self.output_dir / "docking_results.csv"
        energy_csv = self.output_dir / "energy_breakdown.csv"
        
        # Write docking results to CSV
        with open(results_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Score', 'Filename'])
            
            sorted_results = sorted(self.results, key=lambda x: x[1])
            for i, (pose, score) in enumerate(sorted_results):
                writer.writerow([i+1, score, f"pose_{i+1}_score_{score:.1f}.pdb"])
        
        # Write energy breakdown to CSV if available
        if self.scoring_breakdown:
            with open(energy_csv, 'w', newline='') as f:
                # Get all energy components from first breakdown
                components = list(self.scoring_breakdown[0].keys())
                
                writer = csv.writer(f)
                writer.writerow(['Pose'] + components)
                
                for i, energy in enumerate(self.scoring_breakdown):
                    writer.writerow([i+1] + [energy[comp] for comp in components])
            
            print(f"Energy breakdown CSV written to {energy_csv}")
            return results_csv, energy_csv
        
        print(f"Results CSV written to {results_csv}")
        return results_csv, None

    def generate_json_report(self):
        """
        Generate a JSON report with all docking information.
        
        Returns:
        --------
        str
            Path to the generated JSON file
        """
        json_path = self.output_dir / "docking_report.json"
        
        # Create a dictionary with all report information
        report_data = {
            "run_info": {
                "timestamp": str(self.timestamp),
                "protein": str(self.args.protein) if hasattr(self.args, 'protein') else "unknown",
                "ligand": str(self.args.ligand) if hasattr(self.args, 'ligand') else "unknown",
                "algorithm": self.args.algorithm if hasattr(self.args, 'algorithm') else "unknown"
            },
            "results": {
                "pose_count": len(self.results),
                "poses": []
            }
        }
        
        # Add pose information (converting all values to basic Python types)
        if self.results:
            sorted_results = sorted(self.results, key=lambda x: x[1])
            report_data["results"]["best_score"] = float(sorted_results[0][1])
            
            for i, (_, score) in enumerate(sorted_results):
                report_data["results"]["poses"].append({
                    "rank": i+1,
                    "score": float(score)
                })
        
        # Simplify scoring breakdown to basic Python types
        if self.scoring_breakdown:
            breakdown_list = []
            for energy in self.scoring_breakdown:
                energy_dict = {}
                for key, value in energy.items():
                    # Convert numpy values to Python native types
                    if hasattr(value, "item"):  # numpy scalar
                        energy_dict[key] = value.item()
                    elif isinstance(value, bool):
                        energy_dict[key] = bool(value)
                    else:
                        energy_dict[key] = float(value) if isinstance(value, (int, float)) else str(value)
                breakdown_list.append(energy_dict)
            report_data["energy_breakdown"] = breakdown_list
        
        # Write JSON file with explicit error handling
        try:
            with open(json_path, 'w') as f:
                json.dump(report_data, f, indent=4)
            print(f"JSON report written to {json_path}")
        except TypeError as e:
            print(f"JSON serialization error: {e}")
            print("Writing simplified JSON report without complex data")
            
            # Simplified report as fallback
            basic_report = {
                "run_info": report_data["run_info"],
                "results": {
                    "pose_count": report_data["results"]["pose_count"],
                    "best_score": report_data["results"]["best_score"] if "best_score" in report_data["results"] else None
                }
            }
            
            with open(json_path, 'w') as f:
                json.dump(basic_report, f, indent=4)
            
        return json_path
    
        # Write JSON file
        #with open(json_path, 'w') as f:
        #    json.dump(report_data, f, indent=4, cls=EnhancedJSONEncoder)
        
        #print(f"JSON report written to {json_path}")
        #return json_path
    
    
            
    def generate_plots(self, save_dir=None):
        """
        Generate plots visualizing the docking results.
        
        Parameters:
        -----------
        save_dir : str, optional
            Directory to save plots, defaults to the output directory
        
        Returns:
        --------
        list
            Paths to the generated plot files
        """
        if save_dir is None:
            save_dir = self.output_dir
        else:
            save_dir = Path(save_dir)
            os.makedirs(save_dir, exist_ok=True)
        
        plot_paths = []
        
        # 1. Score distribution plot
        score_plot_path = save_dir / "score_distribution.png"
        plt.figure(figsize=(10, 6))
        
        scores = [score for _, score in self.results]
        plt.hist(scores, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Docking Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Docking Scores')
        plt.grid(alpha=0.3)
        plt.savefig(score_plot_path)
        plt.close()
        plot_paths.append(score_plot_path)
        
        # 2. Score rank plot
        rank_plot_path = save_dir / "score_rank.png"
        plt.figure(figsize=(10, 6))
        
        sorted_scores = sorted(scores)
        plt.plot(range(1, len(sorted_scores) + 1), sorted_scores, marker='o', linestyle='-', 
                 color='darkorange', markersize=5)
        plt.xlabel('Rank')
        plt.ylabel('Docking Score')
        plt.title('Docking Scores by Rank')
        plt.grid(alpha=0.3)
        plt.savefig(rank_plot_path)
        plt.close()
        plot_paths.append(rank_plot_path)
        
        # 3. Energy component breakdown plot (for top 10 poses)
        if self.scoring_breakdown:
            energy_plot_path = save_dir / "energy_breakdown.png"
            plt.figure(figsize=(12, 8))
            
            # Get top 10 energy breakdowns
            top_poses = min(10, len(self.scoring_breakdown))
            
            # Get all energy components and filter out 'total'
            components = [key for key in self.scoring_breakdown[0].keys() if key.lower() != 'total']
            
            # Setup colors for components
            colors = plt.cm.tab10(np.linspace(0, 1, len(components)))
            
            # Setup plot
            bar_width = 0.8 / len(components)
            r = np.arange(top_poses)
            
            # Plot each component
            for i, component in enumerate(components):
                # Extract numeric values, handling cases where values are dictionaries
                values = [
                    energy[component].get('value', 0.0) if isinstance(energy[component], dict) else energy[component]
                    for energy in self.scoring_breakdown[:top_poses]
                ]
                plt.bar(r + i * bar_width, values, width=bar_width, color=colors[i], 
                        edgecolor='gray', alpha=0.7, label=component)
            
            # Formatting
            plt.xlabel('Pose Rank')
            plt.ylabel('Energy Contribution')
            plt.title('Energy Component Breakdown for Top Poses')
            plt.xticks(r + bar_width * (len(components) - 1) / 2, [f'{i+1}' for i in range(top_poses)])
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(energy_plot_path)
            plt.close()
            plot_paths.append(energy_plot_path)
        
        # 4. RMSD plot if validation results are available
        if self.validation_results and isinstance(self.validation_results, list):
            rmsd_plot_path = save_dir / "rmsd_plot.png"
            plt.figure(figsize=(10, 6))
            
            # Extract RMSD values
            rmsd_values = [result.get('rmsd', 0.0) for result in self.validation_results]
            
            plt.bar(range(1, len(rmsd_values) + 1), rmsd_values, color='lightgreen', edgecolor='darkgreen')
            plt.axhline(y=2.0, color='red', linestyle='--', label='Success Threshold (2.0 Å)')
            
            plt.xlabel('Pose Rank')
            plt.ylabel('RMSD (Å)')
            plt.title('RMSD Comparison with Reference Structure')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(rmsd_plot_path)
            plt.close()
            plot_paths.append(rmsd_plot_path)
        
        print(f"Generated {len(plot_paths)} plots in {save_dir}")
        return plot_paths
    
    def generate_html_report(self):
        """
        Generate a comprehensive HTML report with embedded plots and tables.
        
        Returns:
        --------
        str
            Path to the generated HTML report
        """
        html_path = self.output_dir / "docking_report.html"
        
        # Generate plots in a plots subdirectory
        plots_dir = self.output_dir / "plots"
        os.makedirs(plots_dir, exist_ok=True)
        plot_paths = self.generate_plots(save_dir=plots_dir)
        
        # Extract relative paths
        rel_plot_paths = [os.path.relpath(path, self.output_dir) for path in plot_paths]
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>PandaDock Report - {self.timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background-color: #3498db; color: white; padding: 20px; margin-bottom: 20px; }}
                .section {{ background-color: #f9f9f9; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 12px; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .plot-container {{ display: flex; flex-wrap: wrap; justify-content: space-around; }}
                .plot {{ margin: 10px; max-width: 600px; }}
                .success {{ color: green; font-weight: bold; }}
                .failure {{ color: red; font-weight: bold; }}
                .footnote {{ font-size: 0.8em; color: #7f8c8d; text-align: center; margin-top: 40px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>PandaDock Molecular Docking Report</h1>
                    <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>Run Information</h2>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Report ID</td><td>{self.timestamp}</td></tr>
                        <tr><td>Protein</td><td>{self.args.protein}</td></tr>
                        <tr><td>Ligand</td><td>{self.args.ligand}</td></tr>
                        <tr><td>Algorithm</td><td>{self.args.algorithm if hasattr(self.args, 'algorithm') else "unknown"}</td></tr>
        """
        
        # Add algorithm-specific parameters
        if hasattr(self.args, 'algorithm'):
            if self.args.algorithm == "genetic":
                if hasattr(self.args, 'population_size'):
                    html_content += f"<tr><td>Population Size</td><td>{self.args.population_size}</td></tr>\n"
                if hasattr(self.args, 'mutation_rate'):
                    html_content += f"<tr><td>Mutation Rate</td><td>{self.args.mutation_rate}</td></tr>\n"
            elif self.args.algorithm == "monte-carlo":
                if hasattr(self.args, 'mc_steps'):
                    html_content += f"<tr><td>MC Steps</td><td>{self.args.mc_steps}</td></tr>\n"
                if hasattr(self.args, 'temperature'):
                    html_content += f"<tr><td>Temperature</td><td>{self.args.temperature} K</td></tr>\n"
        
        # Add hardware information
        if hasattr(self.args, 'use_gpu'):
            gpu_text = "Yes" if self.args.use_gpu else "No"
            html_content += f"<tr><td>GPU Acceleration</td><td>{gpu_text}</td></tr>\n"
            if self.args.use_gpu and hasattr(self.args, 'gpu_id'):
                html_content += f"<tr><td>GPU ID</td><td>{self.args.gpu_id}</td></tr>\n"
        
        if hasattr(self.args, 'cpu_workers'):
            cpu_workers = str(self.args.cpu_workers) if self.args.cpu_workers else "All Available"
            html_content += f"<tr><td>CPU Workers</td><td>{cpu_workers}</td></tr>\n"
            
        # Add scoring function information
        scoring_type = "Standard"
        if hasattr(self.args, 'enhanced_scoring') and self.args.enhanced_scoring:
            scoring_type = "Enhanced"
        if hasattr(self.args, 'physics_based') and self.args.physics_based:
            scoring_type = "Physics-based"
        html_content += f"<tr><td>Scoring Function</td><td>{scoring_type}</td></tr>\n"
        
        html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Results Summary</h2>
        """
        
        # Add results summary
        if self.results:
            sorted_results = sorted(self.results, key=lambda x: x[1])
            html_content += f"""
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Total Poses Generated</td><td>{len(self.results)}</td></tr>
                        <tr><td>Best Score</td><td>{sorted_results[0][1]:.4f}</td></tr>
                        <tr><td>Worst Score</td><td>{sorted_results[-1][1]:.4f}</td></tr>
                        <tr><td>Average Score</td><td>{sum([score for _, score in self.results])/len(self.results):.4f}</td></tr>
                        <tr><td>Score Standard Deviation</td><td>{np.std([score for _, score in self.results]):.4f}</td></tr>
                    </table>
            """
        else:
            html_content += "<p>No docking results available.</p>\n"
        
        html_content += """
                </div>
                
                <div class="section">
                    <h2>Top Docking Poses</h2>
        """
        
        # Add top poses table
        if self.results:
            sorted_results = sorted(self.results, key=lambda x: x[1])
            html_content += """
                    <table>
                        <tr>
                            <th>Rank</th>
                            <th>Score</th>
                            <th>Filename</th>
                        </tr>
            """
            
            for i, (pose, score) in enumerate(sorted_results[:min(20, len(sorted_results))]):
                filename = f"pose_{i+1}_score_{score:.1f}.pdb"
                html_content += f"""
                        <tr>
                            <td>{i+1}</td>
                            <td>{score:.4f}</td>
                            <td>{filename}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
            """
        else:
            html_content += "<p>No docking poses available.</p>\n"
            
        # Add energy breakdown if available
        html_content += """
                </div>
        """
        
        if self.scoring_breakdown:
            print(f"DEBUG - Writing energy breakdown to HTML: {self.scoring_breakdown}")
            html_content += """
                <div class="section">
                    <h2>Energy Component Breakdown</h2>
            """
            
            # Get all energy components
            components = list(self.scoring_breakdown[0].keys())
            
            html_content += """
                    <table>
                        <tr>
                            <th>Pose</th>
            """
            
            for comp in components:
                html_content += f"<th>{comp}</th>\n"
            
            html_content += "</tr>\n"
            
            # Add energy values for top poses
            # Add energy values for top poses
            for i, energy in enumerate(self.scoring_breakdown[:min(10, len(self.scoring_breakdown))]):
                html_content += f"<tr><td>{i+1}</td>\n"
                for comp in components:
                    # Extract numeric value if energy[comp] is a dictionary, otherwise use the value directly
                    value = energy[comp].get('value', 0.0) if isinstance(energy[comp], dict) else energy[comp]
                    html_content += f"<td>{value:.2f}</td>\n"
                html_content += "</tr>\n"
            
            html_content += """
                    </table>
                </div>
            """
            
        # Add validation results if available
        if self.validation_results:
            html_content += """
                <div class="section">
                    <h2>Validation Against Reference Structure</h2>
            """
            
            if isinstance(self.validation_results, list):
                # Multiple pose validation
                html_content += """
                    <table>
                        <tr>
                            <th>Rank</th>
                            <th>Pose</th>
                            <th>RMSD (Å)</th>
                            <th>Status</th>
                        </tr>
                """
                
                for i, result in enumerate(self.validation_results[:min(10, len(self.validation_results))]):
                    pose_idx = result.get('pose_index', i)
                    rmsd = result.get('rmsd', 0.0)
                    success = rmsd < 2.0
                    status_class = "success" if success else "failure"
                    status_text = "Success" if success else "Failure"
                    
                    html_content += f"""
                        <tr>
                            <td>{i+1}</td>
                            <td>{pose_idx+1}</td>
                            <td>{rmsd:.4f}</td>
                            <td class="{status_class}">{status_text}</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                """
                
                # Best RMSD summary
                if len(self.validation_results) > 0:
                    best_rmsd = min(result.get('rmsd', float('inf')) for result in self.validation_results)
                    success = best_rmsd < 2.0
                    status_class = "success" if success else "failure"
                    
                    html_content += f"""
                    <p>Best RMSD: <strong>{best_rmsd:.4f} Å</strong></p>
                    <p>Docking Success: <span class="{status_class}">{success}</span></p>
                    """
            
            elif isinstance(self.validation_results, dict):
                # Single validation result
                rmsd = self.validation_results.get('rmsd', 0.0)
                max_dev = self.validation_results.get('max_deviation', 0.0)
                min_dev = self.validation_results.get('min_deviation', 0.0)
                success = self.validation_results.get('success', False)
                status_class = "success" if success else "failure"
                
                html_content += f"""
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Overall RMSD</td><td>{rmsd:.4f} Å</td></tr>
                        <tr><td>Maximum Atomic Deviation</td><td>{max_dev:.4f} Å</td></tr>
                        <tr><td>Minimum Atomic Deviation</td><td>{min_dev:.4f} Å</td></tr>
                        <tr><td>Docking Success</td><td class="{status_class}">{success}</td></tr>
                    </table>
                """
                
                # Top atom deviations if available
                atom_deviations = self.validation_results.get('atom_deviations', None)
                if atom_deviations is not None:
                    html_content += """
                    <h3>Top 10 Atom Deviations</h3>
                    <table>
                        <tr><th>Atom</th><th>Deviation (Å)</th></tr>
                    """
                    
                    sorted_indices = np.argsort(atom_deviations)[::-1]
                    for i in range(min(10, len(sorted_indices))):
                        idx = sorted_indices[i]
                        html_content += f"""
                        <tr><td>Atom {idx + 1}</td><td>{atom_deviations[idx]:.4f}</td></tr>
                        """
                    
                    html_content += """
                    </table>
                    """
            
            html_content += """
                </div>
            """
            
        # Add visualization section
        html_content += """
                <div class="section">
                    <h2>Visualizations</h2>
                    <div class="plot-container">
        """
        
        # Add all generated plots
        for rel_path in rel_plot_paths:
            plot_title = os.path.basename(rel_path).replace('.png', '').replace('_', ' ').title()
            html_content += f"""
                        <div class="plot">
                            <h3>{plot_title}</h3>
                            <img src="{rel_path}" alt="{plot_title}" width="100%">
                        </div>
            """
        
        html_content += """
                    </div>
                </div>
                
                <div class="footnote">
                    <p>Report generated by PandaDock - Python Molecular Docking Tool</p>
                    <p>Report ID: """ + self.timestamp + """</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report written to {html_path}")
        return html_path
