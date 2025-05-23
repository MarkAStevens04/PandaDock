"""
Main entry script for PandaDock with GPU/CPU hardware acceleration.

This module provides the primary command-line interface for PandaDock,
a molecular docking tool with support for various algorithms and hardware acceleration.
"""

import argparse
import os
import sys
import time
import copy
import random
import traceback
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Any
import logging
import json
import traceback
import json
from pathlib import Path

# Rich formatting for better CLI experience
try:
    from rich_argparse import RichHelpFormatter
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    RichHelpFormatter = argparse.HelpFormatter

# Core PandaDock imports
from . import __version__
from .protein import Protein
from .ligand import Ligand
from .utils import (
    setup_logging, save_docking_results, create_initial_files,
    update_status, save_intermediate_result, save_complex_to_pdb
)
from .unified_scoring import (
    ScoringFunction,
    CompositeScoringFunction,
    EnhancedScoringFunction,
    GPUScoringFunction,
    EnhancedGPUScoringFunction,
    TetheredScoringFunction,
)
from .search import GeneticAlgorithm, RandomSearch
from .parallel_search import (
    ParallelSearch, ParallelGeneticAlgorithm, ParallelRandomSearch,
    HybridSearch, FlexibleLigandSearch, create_search_algorithm
)
from .pockets import OptimizedCastpDetector
from .pandadock import PANDADOCKAlgorithm
from .virtual_screening import VirtualScreeningManager
from .batch_screening import run_virtual_screening
from .preparation import prepare_protein, prepare_ligand
from .reporting import DockingReporter
from .validation import validate_against_reference
from .main_integration import (
    add_hardware_options,
    configure_hardware,
    setup_hardware_acceleration,
    create_optimized_scoring_function,
    create_optimized_search_algorithm,
    get_scoring_type_from_args,
    get_algorithm_type_from_args,
    get_algorithm_kwargs_from_args
)
from .reporting import EnhancedJSONEncoder, DockingReporter
from .hybrid_manager import HybridDockingManager
from .metal_docking import MetalDockingSearch
from .cli_visualizer import ProteinCLIVisualizer, print_pocket_detection_results
# Physics-based modules (optional)
try:
    from .physics import (
        MMFFMinimization, MonteCarloSampling, PhysicsBasedScoring,
        GeneralizedBornSolvation
    )
    from .physics import PhysicsBasedScoringFunction
    PHYSICS_AVAILABLE = True
except ImportError as e:
    PHYSICS_AVAILABLE = False
    print(f"Warning: Physics-based modules not available. Some features will be disabled. Reason: {e}")

# CUDA-specific modules (optional)
try:
    import torch
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA not available. GPU acceleration will not work.")


# ============================================================================
# Constants and Configuration
# ============================================================================

DEFAULT_POPULATION_SIZE = 10
DEFAULT_ITERATIONS = 10
DEFAULT_RADIUS = 15.0
DEFAULT_GRID_SPACING = 0.500
DEFAULT_PH = 7.4
DEFAULT_TEMPERATURE = 300.0
DEFAULT_MC_STEPS = 1000

# ASCII Art for branding
PANDADOCK_ASCII = """
██████╗  █████╗ ███╗   ██╗██████╗  █████╗ ██████╗  ██████╗  ██████╗██╗  ██╗
██╔══██╗██╔══██╗████╗  ██║██╔══██╗██╔══██╗██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝
██████╔╝███████║██╔██╗ ██║██║  ██║███████║██║  ██║██║   ██║██║     █████╔╝ 
██╔═══╝ ██╔══██║██║╚██╗██║██║  ██║██╔══██║██║  ██║██║   ██║██║     ██╔═██╗ 
██║     ██║  ██║██║ ╚████║██████╔╝██║  ██║██████╔╝╚██████╔╝╚██████╗██║  ██╗
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚═════╝╚═╝  ╚═╝
"""


# ============================================================================
# Utility Functions
# ============================================================================

def print_pandadock_header():
    """Print PandaDock header with version information."""
    # Color codes
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    
    print(f"{CYAN}{PANDADOCK_ASCII}{RESET}")
    print(f"{BOLD}{GREEN}PandaDock Molecular Docking Suite 🚀{RESET}")
    print(f"{YELLOW}Version: {__version__}{RESET}")
    print("-" * 80)


def check_for_updates(logger: Optional[Any] = None) -> None:
    """
    Check for newer versions of PandaDock on PyPI and notify user if available.
    
    Args:
        logger: Logger instance for output
    """
    try:
        import requests
        import pkg_resources
        from packaging import version
        
        # Check cache - only verify once per day
        cache_file = os.path.join(os.path.expanduser("~"), ".pandadock_version_check")
        current_time = time.time()
        
        # If cache exists and is less than 24 hours old, skip check
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                last_check = float(f.read().strip())
            if current_time - last_check < 86400:  # 24 hours
                return
        
        # Update cache timestamp
        with open(cache_file, "w") as f:
            f.write(str(current_time))
        
        # Get current installed version
        current_version = pkg_resources.get_distribution("pandadock").version
        
        # Query PyPI for the latest version
        response = requests.get("https://pypi.org/pypi/pandadock/json", timeout=2)
        latest_version = response.json()["info"]["version"]
        
        # Compare versions
        if version.parse(latest_version) > version.parse(current_version):
            update_message = f"""
{'='*70}
  New version available: PandaDock {latest_version} (you have {current_version})
  Update with: pip install --upgrade pandadock
  See release notes at: https://github.com/pritampanda15/PandaDock/releases
{'='*70}
"""
            if logger:
                logger.info(update_message)
            else:
                print(update_message)
    except Exception:
        # Silently fail if version check doesn't work
        pass


def validate_input_files(protein_path: str, ligand_path: Optional[str] = None) -> bool:
    """
    Validate that input files exist and are readable.
    
    Args:
        protein_path: Path to protein file
        ligand_path: Path to ligand file (optional for virtual screening)
        
    Returns:
        bool: True if files are valid
    """
    if not os.path.exists(protein_path):
        print(f"Error: Protein file not found: {protein_path}")
        return False
    
    if ligand_path and not os.path.exists(ligand_path):
        print(f"Error: Ligand file not found: {ligand_path}")
        return False
    
    return True


def create_output_directory(base_output: str, protein_path: str, 
                          ligand_path: Optional[str], algorithm: str) -> str:
    """
    Create a descriptive output directory name.
    
    Args:
        base_output: Base output directory name
        protein_path: Path to protein file
        ligand_path: Path to ligand file
        algorithm: Algorithm name
        
    Returns:
        str: Full output directory path
    """
    protein_base = Path(protein_path).stem
    ligand_base = Path(ligand_path).stem if ligand_path else "virtual_screening"
    readable_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_name = f"{protein_base}_{ligand_base}_{algorithm}_{readable_date}"
    
    output_dir = f"{base_output}_{output_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


# ============================================================================
# Flexible Residue Detection
# ============================================================================

def detect_flexible_residues(protein: Protein, binding_site_residues: List[str], 
                           max_residues: int = 5, logger: Optional[Any] = None) -> List[str]:
    """
    Automatically detect flexible residues in the binding site.
    
    Args:
        protein: Protein object
        binding_site_residues: List of residue IDs in the binding site
        max_residues: Maximum number of flexible residues to detect
        logger: Logger instance
        
    Returns:
        List of detected flexible residue IDs
    """
    # Define residues with flexible sidechains
    flexible_aa_types = [
        'ARG', 'LYS', 'GLU', 'GLN', 'MET', 'PHE', 'TYR', 'TRP',
        'LEU', 'ILE', 'ASP', 'ASN', 'HIS', 'SER', 'THR', 'CYS', 'VAL'
    ]
    
    if logger:
        logger.info(f"Searching for flexible residues among {len(binding_site_residues)} binding site residues")
    
    candidate_residues = []
    
    for res_id in binding_site_residues:
        if res_id in protein.residues:
            residue_atoms = protein.residues[res_id]
            
            # Get residue type
            if residue_atoms and 'residue_name' in residue_atoms[0]:
                res_type = residue_atoms[0]['residue_name']
                
                # Check if it's a residue type with flexible sidechain
                if res_type in flexible_aa_types:
                    # Calculate distance from binding site center
                    if protein.active_site and 'center' in protein.active_site:
                        center = protein.active_site['center']
                        
                        # Use CA atom for distance calculation
                        ca_atom = next((atom for atom in residue_atoms if atom.get('name', '') == 'CA'), None)
                        
                        if ca_atom:
                            distance = np.linalg.norm(ca_atom['coords'] - center)
                            candidate_residues.append((res_id, distance, res_type))
                            if logger:
                                logger.info(f"  Found candidate flexible residue: {res_id} ({res_type}) - distance: {distance:.2f}Å")
    
    # Sort by distance to center (closest first)
    candidate_residues.sort(key=lambda x: x[1])
    
    if logger:
        logger.info(f"Selected {min(max_residues, len(candidate_residues))} flexible residues:")
        for i, (res_id, distance, res_type) in enumerate(candidate_residues[:max_residues]):
            logger.info(f"  {i+1}. {res_id} ({res_type}) - distance: {distance:.2f}Å")
    
    # Return up to max_residues
    return [res_id for res_id, _, _ in candidate_residues[:max_residues]]


def prepare_protein_configs(protein: Protein, args: argparse.Namespace, 
                          logger: Optional[Any] = None) -> List[dict]:
    """
    Prepare protein configurations for benchmarking.
    
    Args:
        protein: Protein object
        args: Command-line arguments
        logger: Logger instance
        
    Returns:
        List of protein configurations for benchmarking
    """
    configs = []
    
    # Rigid configuration
    configs.append({
        'type': 'rigid',
        'protein': protein,
        'flexible_residues': []
    })
    
    # Flexible configuration if requested
    if (hasattr(args, 'flex_residues') and args.flex_residues) or \
       (hasattr(args, 'auto_flex') and args.auto_flex):
        
        flex_protein = copy.deepcopy(protein)
        flex_residues = []
        
        if hasattr(args, 'flex_residues') and args.flex_residues:
            # Use user-specified flexible residues
            flex_residues = args.flex_residues
            if logger:
                logger.info(f"Using user-specified flexible residues: {', '.join(flex_residues)}")
        elif hasattr(args, 'auto_flex') and args.auto_flex:
            # Auto-detect flexible residues based on binding site
            if protein.active_site and 'residues' in protein.active_site:
                binding_site_residues = protein.active_site['residues']
                flex_residues = detect_flexible_residues(
                    protein, binding_site_residues,
                    max_residues=getattr(args, 'max_flex_residues', 5),
                    logger=logger
                )
                if logger:
                    logger.info(f"Auto-detected flexible residues: {', '.join(flex_residues)}")
            else:
                if logger:
                    logger.info("Warning: No active site defined. Cannot auto-detect flexible residues.")
        
        if flex_residues:
            flex_protein.define_flexible_residues(
                flex_residues,
                max_rotatable_bonds=getattr(args, 'max_flex_bonds', 3)
            )
            configs.append({
                'type': 'flexible',
                'protein': flex_protein,
                'flexible_residues': flex_residues
            })
        else:
            if logger:
                logger.info("No flexible residues defined. Using only rigid configuration.")
    
    return configs


# ============================================================================
# Result Writing and Reporting
# ============================================================================

def write_results_to_txt(results: List[Tuple], output_dir: str, elapsed_time: float,
                        protein_path: str, ligand_path: str, algorithm: str,
                        iterations: int, logger: Optional[Any] = None) -> str:
    """
    Write docking results to a text file.
    
    Args:
        results: List of (pose, score) tuples
        output_dir: Output directory
        elapsed_time: Total elapsed time in seconds
        protein_path: Path to protein file
        ligand_path: Path to ligand file
        algorithm: Docking algorithm used
        iterations: Number of iterations/generations
        logger: Logger instance
        
    Returns:
        str: Path to the results file
    """
    results_path = Path(output_dir) / "docking_scores.txt"
    
    with open(results_path, 'w') as f:
        f.write(f"""
{'='*80}
{PANDADOCK_ASCII}
               PandaDock - Python Molecular Docking Tool                             
               https://github.com/pritampanda15/PandaDock                   
{'='*80}
""")
        
        # Write run information
        f.write("RUN INFORMATION\n")
        f.write("-" * 15 + "\n")
        f.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Protein: {protein_path}\n")
        f.write(f"Ligand: {ligand_path}\n")
        f.write(f"Algorithm: {algorithm}\n")
        f.write(f"Iterations/Generations: {iterations}\n")
        f.write(f"Total Runtime: {elapsed_time:.2f} seconds\n\n")
        
        # Check if results is empty
        if not results:
            f.write("RESULTS SUMMARY\n")
            f.write("-" * 15 + "\n")
            f.write("No valid docking solutions found.\n")
            f.write("This can occur due to incompatible structures, overly strict scoring parameters,\n")
            f.write("or issues with the search space definition.\n\n")
        else:
            # Sort results by score (lowest first)
            sorted_results = sorted(results, key=lambda x: x[1])
            
            # Write summary of results
            f.write("RESULTS SUMMARY\n")
            f.write("-" * 15 + "\n")
            f.write(f"Total Poses Generated: {len(results)}\n")
            f.write(f"Best Score: {sorted_results[0][1]:.4f}\n")
            f.write(f"Worst Score: {sorted_results[-1][1]:.4f}\n")
            f.write(f"Average Score: {sum([score for _, score in results])/len(results):.4f}\n\n")
            
            # Write top 10 poses
            f.write("TOP 10 POSES\n")
            f.write("-" * 12 + "\n")
            f.write("Rank\tScore\tFile\n")
            for i, (pose, score) in enumerate(sorted_results[:10]):
                f.write(f"{i+1}\t{score:.4f}\tpose_{i+1}_score_{score:.1f}.pdb\n")
            
            f.write("\n\nFull results are available in the output directory.\n")
        
        f.write("=" * 53 + "\n")
    
    if logger:
        logger.info(f"Detailed results written to {results_path}")
    
    return str(results_path)


# ============================================================================
# Command Line Interface Setup
# ============================================================================

def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Set up the command-line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    formatter_class = RichHelpFormatter if HAS_RICH else argparse.HelpFormatter
    
    parser = argparse.ArgumentParser(
        description="🐼 PandaDock: Physics-based Molecular Docking 🚀",
        formatter_class=formatter_class,
    )
    
    # Version
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
    # Required arguments
    parser.add_argument('-p', '--protein', required=True, 
                       help='Path to protein PDB file')
    parser.add_argument('-l', '--ligand', required=False, 
                       help='Path to ligand MOL/SDF file')
    
    # Basic options
    basic_group = parser.add_argument_group('Basic Options')
    basic_group.add_argument('-o', '--output', default='docking_results',
                           help='Output directory for docking results')
    basic_group.add_argument('--verbose', action='store_true', 
                           help='Enable verbose output')
    basic_group.add_argument('--log-file', type=str, default='pandadock.log',
                           help='Log file for verbose output')
    basic_group.add_argument('--seed', type=int, default=None,
                           help='Random seed for reproducibility')
    
    # Algorithm options
    algo_group = parser.add_argument_group('Algorithm Options')
    algo_group.add_argument('-a', '--algorithm',
                          choices=['random', 'genetic', 'pandadock', 'hybrid', 'flexible'],
                          default='genetic',
                          help='Docking algorithm to use (default: genetic)')
    algo_group.add_argument('--hybrid', action='store_true',
                          help='Use hybrid search algorithm (combines GA, SA, and MC)')
    algo_group.add_argument('--flexible', action='store_true',
                          help='Enable flexible ligand docking')
    algo_group.add_argument('-i', '--iterations', type=int, default=DEFAULT_ITERATIONS,
                          help=f'Number of iterations/generations (default: {DEFAULT_ITERATIONS})')
    
    # Active site options
    site_group = parser.add_argument_group('Active Site Options')
    site_group.add_argument('-s', '--site', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                          help='Active site center coordinates')
    site_group.add_argument('-r', '--radius', type=float, default=DEFAULT_RADIUS,
                          help=f'Active site radius in Angstroms (default: {DEFAULT_RADIUS})')
    site_group.add_argument('--detect-pockets', action='store_true',
                          help='Automatically detect binding pockets')
    site_group.add_argument('--grid-spacing', type=float, default=DEFAULT_GRID_SPACING,
                          help=f'Grid spacing in Å (default: {DEFAULT_GRID_SPACING})')
    site_group.add_argument('--grid-radius', type=float, default=DEFAULT_RADIUS,
                          help=f'Grid radius in Å (default: {DEFAULT_RADIUS})')
    site_group.add_argument('--optimized-pockets', action='store_true',
                   help='Use optimized CASTp-based pocket detection algorithm')
    site_group.add_argument('--pocket-probe-radius', type=float, default=1.4,
                   help='Probe radius for pocket detection (default: 1.4 Å)')
    site_group.add_argument('--pocket-grid-spacing', type=float, default=0.8,
                   help='Grid spacing for pocket detection (default: 0.8 Å)')
    site_group.add_argument('--show-structure', action='store_true',
                   help='Show ASCII visualization of protein structure and pockets')
    site_group.add_argument('--no-viz', action='store_true',
                   help='Disable all visualizations (text-only output)')
    site_group.add_argument('--viz-width', type=int, default=80,
                   help='Width of ASCII visualization (default: 80)')
    site_group.add_argument('--viz-height', type=int, default=25,
                   help='Height of ASCII visualization (default: 25)')
    site_group.add_argument('--show-manual-site', action='store_true',
                   help='Always show visualization for manually specified active sites')
    
    
    # Quick mode options
    mode_group = parser.add_argument_group('Mode Options')
    mode_group.add_argument('--fast-mode', action='store_true',
                          help='Run with minimal enhancements for quick results')
    mode_group.add_argument('--enhanced', action='store_true',
                          help='Use enhanced algorithms for more accurate (but slower) results')
    mode_group.add_argument('--auto-algorithm', action='store_true',
                          help='Automatically select the best docking algorithm')
    
    # Enhanced docking options
    enhanced_group = parser.add_argument_group('Enhanced Options')
    enhanced_group.add_argument('--enhanced-scoring', action='store_true',
                              help='Use enhanced scoring function with electrostatics')
    enhanced_group.add_argument('--prepare-molecules', action='store_true',
                              help='Prepare protein and ligand before docking')
    enhanced_group.add_argument('--population-size', type=int, default=DEFAULT_POPULATION_SIZE,
                              help=f'Population size for genetic algorithm (default: {DEFAULT_POPULATION_SIZE})')
    enhanced_group.add_argument('--exhaustiveness', type=int, default=1,
                              help='Number of independent docking runs (default: 1)')
    enhanced_group.add_argument('--local-opt', action='store_true',
                              help='Enable local optimization on top poses')
    enhanced_group.add_argument('--ph', type=float, default=DEFAULT_PH,
                              help=f'pH for protein preparation (default: {DEFAULT_PH})')
    
    # Physics-based options
    physics_group = parser.add_argument_group('Physics-Based Options')
    physics_group.add_argument('--physics-based', action='store_true',
                             help='Use full physics-based scoring (slow but most accurate)')
    physics_group.add_argument('--mmff-minimization', action='store_true',
                             help='Use MMFF94 force field minimization (requires RDKit)')
    physics_group.add_argument('--monte-carlo', action='store_true',
                             help='Use Monte Carlo sampling instead of genetic algorithm')
    physics_group.add_argument('--mc-steps', type=int, default=DEFAULT_MC_STEPS,
                             help=f'Number of Monte Carlo steps (default: {DEFAULT_MC_STEPS})')
    physics_group.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                             help=f'Temperature for Monte Carlo simulation in K (default: {DEFAULT_TEMPERATURE})')
    
    # PandaDock algorithm options
    pandadock_group = parser.add_argument_group('PandaDock Algorithm Options')
    pandadock_group.add_argument('--high-temp', type=float, default=1000.0,
                               help='High temperature for pandadock MD simulations (K)')
    pandadock_group.add_argument('--target-temp', type=float, default=300.0,
                               help='Target temperature for pandadock cooling (K)')
    pandadock_group.add_argument('--num-conformers', type=int, default=10,
                               help='Number of ligand conformers to generate')
    pandadock_group.add_argument('--num-orientations', type=int, default=10,
                               help='Number of orientations to try for each conformer')
    pandadock_group.add_argument('--md-steps', type=int, default=1000,
                               help='Number of MD steps for simulated annealing')
    pandadock_group.add_argument('--minimize-steps', type=int, default=200,
                               help='Number of minimization steps for final refinement')
    pandadock_group.add_argument('--use-grid', action='store_true',
                               help='Use grid-based energy calculations')
    pandadock_group.add_argument('--cooling-factor', type=float, default=0.95,
                               help='Cooling factor for simulated annealing')
    
    # Tethered docking options
    tethered_group = parser.add_argument_group('Tethered Docking Options')
    tethered_group.add_argument('--tethered-docking', action='store_true',
                              help='Use tethered scoring with reference structure')
    tethered_group.add_argument('--tether-weight', type=float, default=10.0,
                              help='Weight for tethered scoring')
    tethered_group.add_argument('--reference', 
                              help='Reference ligand structure for validation')
    tethered_group.add_argument('--exact-alignment', action='store_true',
                              help='Align docked pose exactly to reference structure')
    
    # Flexible residue options
    flex_group = parser.add_argument_group('Flexible Residue Options')
    flex_group.add_argument('--flex-residues', nargs='+',
                          help='Specify flexible residue IDs (e.g., A_42 B_57)')
    flex_group.add_argument('--max-flex-bonds', type=int, default=3,
                          help='Maximum rotatable bonds per residue (default: 3)')
    flex_group.add_argument('--auto-flex', action='store_true',
                          help='Automatically detect flexible residues in the binding site')
    flex_group.add_argument('--max-flex-residues', type=int, default=5,
                          help='Maximum number of flexible residues to detect (default: 5)')
    
    # Reporting options
    report_group = parser.add_argument_group('Reporting Options')
    report_group.add_argument('--report-format', 
                            choices=['text', 'csv', 'json', 'html', 'all'],
                            default='all', help='Report format (default: all)')
    report_group.add_argument('--report-name', type=str, default=None,
                            help='Custom name for the report files')
    report_group.add_argument('--detailed-energy', action='store_true',
                            help='Include detailed energy component breakdown')
    report_group.add_argument('--skip-plots', action='store_true',
                            help='Skip generating plots for reports')
    
    # Add hardware, virtual screening, analysis, and advanced search options
    add_hardware_options(parser)
    add_virtual_screening_options(parser)
    add_analysis_options(parser)
    add_advanced_search_options(parser)
    
    return parser


def add_virtual_screening_options(parser: argparse.ArgumentParser) -> None:
    """Add virtual screening command-line options."""
    vs_group = parser.add_argument_group('Virtual Screening')
    vs_group.add_argument('--virtual-screening', action='store_true',
                         help='Run virtual screening of multiple ligands')
    vs_group.add_argument('--ligand-library', type=str,
                         help='Path to directory containing ligand files')
    vs_group.add_argument('--screening-output', type=str, default='screening_results',
                        help='Output directory for screening results')
    vs_group.add_argument('--num-modes', type=int, default=9,
                        help='Number of binding modes to generate per ligand')
    vs_group.add_argument('--vs-exhaustiveness', type=int, default=8,
                        help='Exhaustiveness of search for virtual screening')
    vs_group.add_argument('--parallel-screening', action='store_true',
                        help='Use parallel processing for screening')
    vs_group.add_argument('--screening-processes', type=int, default=None,
                        help='Number of parallel processes for screening')
    vs_group.add_argument('--prepare-vs-ligands', action='store_true',
                        help='Prepare ligands before screening')
    vs_group.add_argument('--top-hits', type=int, default=10,
                        help='Number of top compounds to save detailed results')


def add_analysis_options(parser: argparse.ArgumentParser) -> None:
    """Add pose clustering and analysis command-line options."""
    analysis = parser.add_argument_group('Analysis Options')
    
    # Clustering options
    analysis.add_argument('--cluster-poses', action='store_true',
                         help='Perform clustering of docking poses')
    analysis.add_argument('--clustering-method', choices=['hierarchical', 'dbscan'],
                        default='hierarchical',
                        help='Method for clustering poses')
    analysis.add_argument('--rmsd-cutoff', type=float, default=2.0,
                        help='RMSD cutoff for pose clustering')
    
    # Interaction analysis
    analysis.add_argument('--analyze-interactions', action='store_true',
                         help='Generate interaction fingerprints and analysis')
    analysis.add_argument('--interaction-types', nargs='+',
                        choices=['hbond', 'hydrophobic', 'ionic', 'aromatic', 'halogen'],
                        default=['hbond', 'hydrophobic', 'ionic'],
                        help='Interaction types to include in analysis')
    
    # Binding mode analysis
    analysis.add_argument('--classify-modes', action='store_true',
                         help='Classify binding modes of docking poses')
    analysis.add_argument('--discover-modes', action='store_true',
                         help='Automatically discover binding modes')
    analysis.add_argument('--n-modes', type=int, default=5,
                        help='Number of binding modes to discover')
    
    # Energy analysis
    analysis.add_argument('--energy-decomposition', action='store_true',
                         help='Perform energy decomposition analysis')
    analysis.add_argument('--per-residue-energy', action='store_true',
                         help='Calculate per-residue energy contributions')
    
    # Comprehensive reporting
    analysis.add_argument('--generate-analysis-report', action='store_true',
                         help='Generate comprehensive docking report')
    analysis.add_argument('--analysis-report-format', choices=['html', 'pdf', 'txt'],
                        default='html',
                        help='Format for analysis report')
    analysis.add_argument('--analysis-report-sections', nargs='+',
                        choices=['summary', 'clusters', 'interactions', 'energetics'],
                        default=['summary', 'clusters', 'interactions', 'energetics'],
                        help='Sections to include in the analysis report')


def add_advanced_search_options(parser: argparse.ArgumentParser) -> None:
    """Add command-line options for advanced search algorithms."""
    adv_search = parser.add_argument_group('Advanced Search Algorithms')
    
    # Algorithm selection
    adv_search.add_argument('--advanced-search', 
                           choices=['gradient', 'replica-exchange', 'ml-guided', 
                                   'fragment-based', 'hybrid'],
                           help='Advanced search algorithm to use')
    
    # Gradient-based options
    adv_search.add_argument('--gradient-step', type=float, default=0.1,
                          help='Step size for gradient calculation')
    adv_search.add_argument('--convergence-threshold', type=float, default=0.01,
                          help='Convergence threshold for gradient-based search')
    
    # Replica exchange options
    adv_search.add_argument('--n-replicas', type=int, default=4,
                          help='Number of replicas for replica exchange')
    adv_search.add_argument('--replica-temperatures', type=float, nargs='+',
                          help='Temperatures for replicas (e.g., 300 400 500 600)')
    adv_search.add_argument('--exchange-steps', type=int, default=10,
                          help='Number of exchange attempts in replica exchange')
    
    # ML-guided options
    adv_search.add_argument('--surrogate-model', choices=['rf', 'gp', 'nn'], default='rf',
                          help='Surrogate model type for ML-guided search')
    adv_search.add_argument('--exploitation-factor', type=float, default=0.8,
                          help='Exploitation vs exploration balance (0-1)')
    
    # Fragment-based options
    adv_search.add_argument('--fragment-min-size', type=int, default=5,
                          help='Minimum fragment size for fragment-based docking')
    adv_search.add_argument('--growth-steps', type=int, default=3,
                          help='Number of fragment growth steps')
    
    # Hybrid search options
    adv_search.add_argument('--ga-iterations', type=int, default=50,
                          help='Genetic algorithm iterations in hybrid search')
    adv_search.add_argument('--lbfgs-iterations', type=int, default=50,
                          help='L-BFGS iterations in hybrid search')
    adv_search.add_argument('--top-n-for-local', type=int, default=10,
                          help='Top N poses to optimize with L-BFGS in hybrid search')


# Integration of Metal Docking with PandaDock Main Framework

# =============================================================================
# Command Line Options for Metal Docking
# =============================================================================

def add_metal_docking_options(parser):
    """Add command-line options for metal-based docking."""
    metal_group = parser.add_argument_group('Metal-Based Docking Options')
    
    # Enable metal docking
    metal_group.add_argument('--metal-docking', action='store_true',
                           help='Enable metal-aware docking for metalloproteins')
    
    # Metal detection and specification
    metal_group.add_argument('--detect-metals', action='store_true',
                           help='Automatically detect metal centers in protein')
    metal_group.add_argument('--metal-centers', nargs='+', type=str,
                           help='Specify metal center coordinates as "X,Y,Z:ELEMENT" (e.g., "10.5,20.3,15.2:ZN")')
    metal_group.add_argument('--metal-elements', nargs='+', type=str,
                           default=['FE', 'ZN', 'CU', 'MG', 'CA', 'MN', 'CO', 'NI'],
                           help='Metal elements to detect (default: FE ZN CU MG CA MN CO NI)')
    
    # Coordination constraints
    metal_group.add_argument('--coordination-geometry', choices=['auto', 'octahedral', 'tetrahedral', 
                                                               'square_planar', 'trigonal_bipyramidal', 'linear'],
                           default='auto',
                           help='Expected coordination geometry (default: auto-detect)')
    metal_group.add_argument('--max-coordination', type=int, default=6,
                           help='Maximum coordination number (default: 6)')
    metal_group.add_argument('--metal-constraint-strength', type=float, default=10.0,
                           help='Strength of metal coordination constraints (default: 10.0)')
    
    # Advanced metal options
    metal_group.add_argument('--metal-bond-tolerance', type=float, default=0.3,
                           help='Tolerance for metal-ligand bond lengths in Å (default: 0.3)')
    metal_group.add_argument('--coordination-weight', type=float, default=10.0,
                           help='Weight for coordination geometry scoring (default: 10.0)')
    metal_group.add_argument('--require-coordination', action='store_true',
                           help='Require ligand to coordinate to at least one metal')
    
    # Output options
    metal_group.add_argument('--save-metal-analysis', action='store_true',
                           help='Save detailed metal coordination analysis')
# ============================================================================
# Core Docking Pipeline Functions
# ============================================================================

def setup_protein_and_ligand(args: argparse.Namespace, logger: Any) -> Tuple[Protein, Optional[Any], str, str]:
    """
    Set up protein and ligand objects, with optional preparation.
    
    Args:
        args: Command-line arguments
        logger: Logger instance
        
    Returns:
        Tuple of (protein, ligand, protein_path, ligand_path)
    """
    protein_path = args.protein
    ligand_path = args.ligand
    temp_dir = None
    
    # Create temporary directory for prepared files if needed
    if getattr(args, 'prepare_molecules', False):
        temp_dir = Path('prepared_files_for_pandadock')
        os.makedirs(temp_dir, exist_ok=True)
        
        logger.info("Preparing molecules for docking...")
        
        # Prepare protein
        logger.info(f"Preparing protein {args.protein}...")
        prepared_protein = prepare_protein(
            args.protein,
            output_file=temp_dir / f"prepared_{Path(args.protein).name}",
            ph=args.ph
        )
        
        # Prepare ligand
        if args.ligand:
            logger.info(f"Preparing ligand {args.ligand}...")
            prepared_ligand = prepare_ligand(
                args.ligand,
                output_file=temp_dir / f"prepared_{Path(args.ligand).name}",
                n_conformers=5 if args.algorithm == 'genetic' else 1
            )
            ligand_path = prepared_ligand
        
        protein_path = prepared_protein
    
    # Load protein
    logger.info(f"Loading protein from {protein_path}...")
    protein = Protein(protein_path)
    
    # Load ligand if provided
    ligand = None
    if ligand_path:
        logger.info(f"Loading ligand from {ligand_path}...")
        ligand = Ligand(ligand_path)
    
    return protein, ligand, protein_path, ligand_path

# Add this function to your main.py file (preferably near the top, after the imports)

def detect_pockets_optimized(protein, args, logger):
    """
    Enhanced pocket detection with proper caching and visualization.
    """
    
    # Create a unique cache key based on detection parameters
    cache_key = f"optimized_{getattr(args, 'pocket_probe_radius', 1.4):.1f}_{getattr(args, 'pocket_grid_spacing', 0.8):.1f}"
    
    # Check if pockets have already been detected and cached
    if (hasattr(protein, '_cached_pockets') and 
        protein._cached_pockets is not None and
        hasattr(protein, '_cache_key') and 
        protein._cache_key == cache_key):
        
        if not getattr(args, 'quiet_pockets', False):
            logger.info("✅ Using previously detected pockets (cached)")
        
        pockets = protein._cached_pockets
        
        # Always show visualization if enabled (even for cached results)
        if not getattr(args, 'no_viz', False):
            show_pocket_visualization(protein, pockets, args, logger)
            
        return pockets
    
    # If we reach here, we need to run detection
    try:
        if hasattr(args, 'optimized_pockets') and args.optimized_pockets:
            from .pockets import OptimizedCastpDetector
            
            # Get parameters from args or use defaults
            probe_radius = getattr(args, 'pocket_probe_radius', 1.4)
            grid_spacing = getattr(args, 'pocket_grid_spacing', 0.8)
            verbose = not getattr(args, 'quiet_pockets', False)
            
            detector = OptimizedCastpDetector(
                probe_radius=probe_radius,
                grid_spacing=grid_spacing,
                verbose=verbose
            )
            pockets = detector.detect_pockets(protein)
            
            # Cache the results with the key
            protein._cached_pockets = pockets
            protein._cache_key = cache_key
            protein._detection_method = 'optimized'
            
        else:
            if not getattr(args, 'quiet_pockets', False):
                logger.info("Using default pocket detection...")
            pockets = protein.detect_pockets()
            
            # Cache the results
            protein._cached_pockets = pockets
            protein._cache_key = 'default'
            protein._detection_method = 'default'
        
        # Always show visualization for new results
        if not getattr(args, 'no_viz', False):
            show_pocket_visualization(protein, pockets, args, logger)
        
        # Log summary
        if not getattr(args, 'quiet_pockets', False):
            method_name = "Optimized" if hasattr(args, 'optimized_pockets') and args.optimized_pockets else "Default"
            
            if pockets:
                logger.info(f"✅ {method_name} detector found {len(pockets)} pockets")
                
                # Log details about the best pocket
                best_pocket = pockets[0]
                logger.info(f"🎯 Using pocket 1 for docking:")
                logger.info(f"   Volume: {best_pocket.get('volume', 'N/A'):.1f} Å³")
                logger.info(f"   Center: ({best_pocket['center'][0]:.1f}, {best_pocket['center'][1]:.1f}, {best_pocket['center'][2]:.1f})")
                logger.info(f"   Radius: {best_pocket['radius']:.1f} Å")
                logger.info(f"   Residues: {len(best_pocket.get('residues', []))}")
            else:
                logger.warning("❌ No pockets detected - will use protein center")
            
        return pockets
            
    except ImportError as e:
        logger.error(f"Error importing optimized pocket detector: {e}")
        logger.info("Falling back to default pocket detection...")
        
        pockets = protein.detect_pockets()
        protein._cached_pockets = pockets
        protein._cache_key = 'default_fallback'
        protein._detection_method = 'default_fallback'
        return pockets
    
    except Exception as e:
        logger.error(f"Error in pocket detection: {e}")
        logger.info("Falling back to default pocket detection...")
        try:
            pockets = protein.detect_pockets()
            protein._cached_pockets = pockets
            protein._cache_key = 'default_fallback'
            protein._detection_method = 'default_fallback'
            return pockets
        except Exception as e2:
            logger.error(f"Default pocket detection also failed: {e2}")
            protein._cached_pockets = []
            protein._cache_key = 'failed'
            return []

def show_pocket_visualization(protein, pockets, args, logger):
    """
    Show pocket visualization (enhanced to show protein even without pockets).
    """
    try:
        print("\n" + "="*80)
        print("🧬 PROTEIN STRUCTURE AND BINDING POCKETS")
        print("="*80)
        
        # Get protein name from file path if available
        protein_name = "Protein"
        if hasattr(args, 'protein') and args.protein:
            protein_name = os.path.basename(args.protein).split('.')[0]
        
        from .cli_visualizer import ProteinCLIVisualizer
        visualizer = ProteinCLIVisualizer(
            width=getattr(args, 'viz_width', 80),
            height=getattr(args, 'viz_height', 25)
        )
        
        if pockets:
            # Show pocket summary
            summary = visualizer.create_pocket_summary_diagram(pockets, protein_name)
            print(summary)
            
            # Show detailed structure if requested or terminal is wide enough
            if (getattr(args, 'show_structure', False) or 
                (hasattr(os, 'get_terminal_size') and 
                 os.get_terminal_size().columns >= 85)):
                
                print("\n")
                detailed_viz = visualizer.visualize_protein_with_pockets(
                    protein, pockets, f"{protein_name} - 3D Structure View"
                )
                print(detailed_viz)
        else:
            # Show protein structure even without pockets
            print(f"╔══════════════════════════════════════════════════════════════════╗")
            print(f"║ 🧬 {protein_name} - No Binding Pockets Detected                    ║")
            print(f"╠══════════════════════════════════════════════════════════════════╣")
            print(f"║ ⚠️  No distinct pockets found with current parameters            ║") 
            print(f"║ 🎯 Will use protein center for docking                          ║")
            print(f"║                                                                  ║")
            print(f"║ 💡 Try adjusting parameters:                                    ║")
            print(f"║    --pocket-probe-radius 1.6                                    ║")
            print(f"║    --pocket-grid-spacing 0.6                                    ║")
            print(f"╚══════════════════════════════════════════════════════════════════╝")
            
            # Still show protein structure
            if (getattr(args, 'show_structure', False) or 
                (hasattr(os, 'get_terminal_size') and 
                 os.get_terminal_size().columns >= 85)):
                
                print("\n")
                detailed_viz = visualizer.visualize_protein_with_pockets(
                    protein, [], f"{protein_name} - 3D Structure View (No Pockets)"
                )
                print(detailed_viz)
        
        print("="*80)
        
    except Exception as viz_error:
        logger.warning(f"Could not display visualization: {viz_error}")
        # Fallback to text summary
        if pockets:
            for i, pocket in enumerate(pockets):
                logger.info(f"Pocket {i+1}: Volume={pocket.get('volume', 0):.1f}Å³, "
                           f"Center={pocket['center']}, Radius={pocket['radius']:.1f}Å")
        else:
            logger.info("No pockets detected - using protein center for docking")

# Also, update your setup_active_site function to use the cached results:
def setup_active_site(protein: Protein, output_dir: str, args: argparse.Namespace, logger: Any) -> None:
    """
    Set up the active site for the protein (optimized to prevent duplicate detection).
    
    Args:
        protein: Protein object
        output_dir: Output directory
        args: Command-line arguments
        logger: Logger instance
    """
    if args.site:
        logger.info(f"Using provided active site center: {args.site}")
        protein.define_active_site(args.site, args.radius)
        update_status(
            output_dir,
            active_site_center=args.site,
            active_site_radius=args.radius
        )
        if not getattr(args, 'no_viz', False):
            show_manual_site_visualization(protein, args, logger)
    elif args.detect_pockets or (hasattr(args, 'optimized_pockets') and args.optimized_pockets):
        logger.info("Detecting binding pockets...")
        update_status(output_dir, status="detecting_pockets")
        
        # This will use cached results if available
        pockets = detect_pockets_optimized(protein, args, logger)
        
        if pockets:
            logger.info(f"Found {len(pockets)} potential binding pockets")
            # Use the largest pocket by default
            best_pocket = pockets[0]
            logger.info(f"Using largest pocket as active site")
            protein.define_active_site(best_pocket['center'], best_pocket['radius'])
            update_status(
                output_dir,
                active_site_center=best_pocket['center'].tolist(),
                active_site_radius=best_pocket['radius'],
                detected_pockets=len(pockets),
                pocket_volume=best_pocket.get('volume', 0)
            )
        else:
            # Use a more reasonable default and print a warning
            center = np.mean(protein.xyz, axis=0)
            radius = 8.0  # More reasonable default radius
            protein.define_active_site(center, radius)
            logger.warning(f"No binding pockets found. Using protein center with radius {radius}Å")
    else:
        logger.info("No active site specified, using whole protein")

# Add this function to clear cache if needed (optional)
def clear_pocket_cache(protein):
    """Clear cached pocket detection results."""
    if hasattr(protein, '_cached_pockets'):
        delattr(protein, '_cached_pockets')
    if hasattr(protein, '_detection_method'):
        delattr(protein, '_detection_method')
    if hasattr(protein, '_visualization_shown'):
        delattr(protein, '_visualization_shown')

def show_manual_site_visualization(protein, args, logger):
    """
    Show visualization for manually specified active site.
    
    Parameters:
    -----------
    protein : Protein
        Protein object with defined active site
    args : argparse.Namespace
        Command-line arguments
    logger : logging.Logger
        Logger instance
    """
    try:
        print("\n" + "="*80)
        print("🧬 PROTEIN STRUCTURE WITH MANUAL ACTIVE SITE")
        print("="*80)
        
        # Get protein name from file path if available
        protein_name = "Protein"
        if hasattr(args, 'protein') and args.protein:
            protein_name = os.path.basename(args.protein).split('.')[0]
        
        # Create a "fake" pocket from the manually specified site
        if hasattr(protein, 'active_site') and protein.active_site:
            manual_pocket = {
                'id': 1,
                'center': protein.active_site['center'],
                'radius': protein.active_site['radius'],
                'volume': (4/3) * np.pi * (protein.active_site['radius'] ** 3),  # Sphere volume
                'atoms': protein.active_site.get('atoms', []),
                'residues': protein.active_site.get('residues', []),
                'manual': True  # Flag to indicate this is manually specified
            }
            
            from .cli_visualizer import ProteinCLIVisualizer
            visualizer = ProteinCLIVisualizer(
                width=getattr(args, 'viz_width', 80),
                height=getattr(args, 'viz_height', 25)
            )
            
            # Show manual site summary
            show_manual_site_summary(manual_pocket, protein_name, args)
            
            # Show detailed structure if requested or terminal is wide enough
            if (getattr(args, 'show_structure', False) or 
                (hasattr(os, 'get_terminal_size') and 
                 os.get_terminal_size().columns >= 85)):
                
                print("\n")
                detailed_viz = visualizer.visualize_protein_with_pockets(
                    protein, [manual_pocket], f"{protein_name} - Manual Active Site"
                )
                print(detailed_viz)
            
            print("="*80)
        
    except Exception as viz_error:
        logger.warning(f"Could not display manual site visualization: {viz_error}")

def show_manual_site_summary(pocket, protein_name, args):
    """Show summary for manually specified active site."""
    center = pocket['center']
    radius = pocket['radius']
    volume = pocket['volume']
    
    print(f"╔══════════════════════════════════════════════════════════════════╗")
    print(f"║ 🎯 {protein_name} - Manual Active Site Specified                 ║")
    print(f"╠══════════════════════════════════════════════════════════════════╣")
    print(f"║ 📍 Center: ({center[0]:6.1f}, {center[1]:6.1f}, {center[2]:6.1f})                           ║")
    print(f"║ 📏 Radius: {radius:5.1f} Å                                              ║")
    print(f"║ 📦 Volume: {volume:7.0f} Ų (estimated sphere)                        ║")
    
    if len(pocket.get('atoms', [])) > 0:
        print(f"║ 🧪 Atoms in site: {len(pocket['atoms']):4d}                                    ║")
    
    if len(pocket.get('residues', [])) > 0:
        print(f"║ 🧬 Residues in site: {len(pocket['residues']):3d}                                  ║")
    
    print(f"║                                                                  ║")
    print(f"║ 💡 Command used: -s {center[0]:.0f} {center[1]:.0f} {center[2]:.0f} -r {radius:.0f}                            ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝")


def apply_preprocessing_options(args: argparse.Namespace, logger: Any) -> None:
    """
    Apply preprocessing options based on command-line arguments.
    
    Args:
        args: Command-line arguments
        logger: Logger instance
    """
    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        logger.info(f"Set random seed to {args.seed}")
    
    # Process mode flags
    if args.auto_algorithm:
        if args.physics_based:
            args.algorithm = 'pandadock'
            logger.info("Auto-selecting PANDADOCK algorithm for physics-based scoring")
        elif args.enhanced_scoring and args.local_opt:
            args.monte_carlo = True
            logger.info("Auto-selecting Monte Carlo algorithm for enhanced scoring with local optimization")
        elif args.fast_mode:
            args.algorithm = 'random'
            logger.info("Auto-selecting Random search algorithm for fast mode")
        else:
            args.algorithm = 'genetic'
            logger.info("Auto-selecting Genetic algorithm (default)")
    
    if args.fast_mode:
        logger.info("Running in fast mode with minimal enhancements")
        args.enhanced_scoring = False
        args.physics_based = False
        args.mmff_minimization = False
        args.monte_carlo = False
        args.local_opt = False
        args.exhaustiveness = 1
        args.prepare_molecules = False
        args.population_size = 50
    
    if args.monte_carlo:
        logger.info("Running in Monte Carlo mode")
        args.algorithm = 'monte-carlo'
        args.exhaustiveness = 1
    
    if args.enhanced:
        logger.info("Running with enhanced algorithms (slower but more accurate)")
        args.enhanced_scoring = True
        args.local_opt = True
        if args.population_size < 100:
            args.population_size = 100


def handle_virtual_screening(args: argparse.Namespace, logger: Any) -> int:
    """
    Handle virtual screening workflow.
    
    Args:
        args: Command-line arguments
        logger: Logger instance
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    if not args.protein or not args.ligand_library:
        logger.error("Both --protein and --ligand-library arguments are required for virtual screening")
        return 1
    
    logger.info(f"Running virtual screening on {args.protein} with ligands from {args.ligand_library}")
    
    # Configure hardware if needed
    hw_config = configure_hardware(args)
    
    
    # Create output directory
    screening_output = Path(args.screening_output)
    screening_output.mkdir(parents=True, exist_ok=True)
    
    # Define parameters
    kwargs = {
        'use_gpu': getattr(args, 'use_gpu', False),
        'physics_based': getattr(args, 'physics_based', False),
        'enhanced_scoring': getattr(args, 'enhanced_scoring', True),
        'exhaustiveness': getattr(args, 'vs_exhaustiveness', 8),
        'num_modes': getattr(args, 'num_modes', 9),
        'max_evals': getattr(args, 'max_evals', 10000),
        'rmsd_thresh': getattr(args, 'rmsd_thresh', 2.0),
        'grid_spacing': getattr(args, 'grid_spacing', 0.375),
        'grid_radius': getattr(args, 'grid_radius', 10.0),
        'cpu_workers': getattr(args, 'screening_processes', None),
        'site': getattr(args, 'site', None),
        'radius': getattr(args, 'radius', 10.0),
        'prepare_molecules': getattr(args, 'prepare_vs_ligands', True),
        'parallel': getattr(args, 'parallel_screening', True),
        'hardware_config': hw_config
    }
    
    try:
        run_virtual_screening(
            protein_file=args.protein,
            ligand_dir=args.ligand_library,
            output_dir=args.screening_output,
            **kwargs
        )
        logger.info("Virtual screening completed successfully!")
        logger.info(f"Results saved to: {screening_output}")
        return 0
    except Exception as e:
        logger.error(f"Error during virtual screening: {e}")
        traceback.print_exc()
        return 1


def run_docking_search(protein: Protein, ligand: Any, args: argparse.Namespace, 
                      logger: Any, output_dir: str) -> List[Tuple]:
    """
    Run the main docking search algorithm.
    
    Args:
        protein: Protein object
        ligand: Ligand object
        args: Command-line arguments
        logger: Logger instance
        output_dir: Output directory path
        
    Returns:
        List of (pose, score) tuples
    """
    # Configure hardware settings
    hw_config = configure_hardware(args)
    # Setup hardware acceleration
    hybrid_manager = setup_hardware_acceleration(hw_config)
    
    # Apply MMFF minimization if requested
    if args.mmff_minimization and PHYSICS_AVAILABLE:
        logger.info("Applying MMFF94 force field minimization to ligand")
        update_status(output_dir, status="minimizing_ligand")
        minimizer = MMFFMinimization()
        ligand = minimizer.minimize_ligand(ligand)
        logger.info("Ligand minimization complete")
    elif args.mmff_minimization and not PHYSICS_AVAILABLE:
        logger.warning("MMFF minimization requested but physics module not available. Skipping.")
    
    # Create scoring function
    scoring_type = get_scoring_type_from_args(args)
    
    if scoring_type == 'physics' and PHYSICS_AVAILABLE:
        logger.info("Using physics-based scoring function (MM-GBSA inspired)")
        scoring_function = PhysicsBasedScoring()
        update_status(output_dir, scoring_function="physics-based")
    else:
        scoring_function = create_optimized_scoring_function(scoring_type, hw_config)
        
        if scoring_type == 'enhanced':
            logger.info("Using enhanced scoring function with hardware acceleration")
            update_status(output_dir, scoring_function="enhanced")
        else:
            logger.info("Using standard composite scoring function with hardware acceleration")
            update_status(output_dir, scoring_function="standard")
    
    # Get algorithm type and parameters
    algorithm_type = get_algorithm_type_from_args(args)
    algorithm_kwargs = get_algorithm_kwargs_from_args(args)
    
    # Inject grid parameters
    algorithm_kwargs.update({
        'grid_spacing': args.grid_spacing,
        'grid_radius': args.grid_radius,
        'output_dir': output_dir
    })
    
    if hasattr(args, 'site') and args.site:
        algorithm_kwargs['grid_center'] = args.site
    
    # Handle reference ligand if provided
    reference_ligand = None
    if args.reference:
        logger.info(f"Loading reference ligand from {args.reference}...")
        reference_ligand = Ligand(args.reference)
        update_status(output_dir, reference_ligand=str(args.reference))
    
    # Create search algorithm
    if args.advanced_search:
        from .advanced_search import create_advanced_search_algorithm
        
        adv_search_kwargs = {'output_dir': output_dir}
        if args.advanced_search == 'gradient':
            adv_search_kwargs.update({
                'gradient_step': args.gradient_step,
                'convergence_threshold': args.convergence_threshold
            })
        elif args.advanced_search == 'replica-exchange':
            adv_search_kwargs.update({
                'n_replicas': args.n_replicas,
                'temperatures': args.replica_temperatures,
                'exchange_steps': args.exchange_steps
            })
        
        search_algorithm = create_advanced_search_algorithm(
            args.advanced_search,
            scoring_function,
            **adv_search_kwargs
        )
        logger.info(f"Using advanced search algorithm: {args.advanced_search}")
    else:
        algorithm_kwargs['hw_config'] = hw_config  # Pass hw_config to the algorithm
        search_algorithm = create_optimized_search_algorithm(
            hybrid_manager,
            algorithm_type,
            scoring_function,
            **algorithm_kwargs,
        )
    
    # Update status before starting search
    update_status(
        output_dir,
        status="searching",
        search_algorithm=algorithm_type,
        search_params=algorithm_kwargs
    )
    
    # Run docking search
    logger.info("Performing docking...")
    
    if args.reference and args.tethered_docking:
        logger.info(f"Using tethered reference-based docking with weight {args.tether_weight}...")
        from .unified_scoring import TetheredScoringFunction
        from .scoring_factory import create_scoring_function
        
        base_function = create_scoring_function(
            use_gpu=getattr(args, 'use_gpu', False),
            physics_based=args.physics_based,
            enhanced=args.enhanced_scoring
        )
        
        scoring_function = TetheredScoringFunction(
            base_function,
            reference_ligand,
            weight=args.tether_weight
        )
        all_results = search_algorithm.search(protein, ligand)
        
    elif args.reference and args.exact_alignment:
        logger.info("Using exact alignment with reference structure...")
        all_results = search_algorithm.exact_reference_docking(
            protein,
            ligand,
            reference_ligand,
            skip_optimization=not args.local_opt
        )
    elif args.reference and not args.exact_alignment:
        logger.info("Using reference-guided docking...")
        all_results = search_algorithm.reference_guided_docking(
            protein,
            ligand,
            reference_ligand,
            skip_optimization=not args.local_opt
        )
    elif args.exhaustiveness > 1:
        logger.info(f"Running {args.exhaustiveness} independent docking runs...")
        all_results = hybrid_manager.run_ensemble_docking(
            protein=protein,
            ligand=ligand,
            n_runs=args.exhaustiveness,
            algorithm_type=algorithm_type,
            **algorithm_kwargs
        )
    elif algorithm_type == 'monte-carlo' and PHYSICS_AVAILABLE:
        logger.info(f"Using Monte Carlo sampling with {args.mc_steps} steps at {args.temperature}K")
        mc_algorithm = MonteCarloSampling(
            scoring_function,
            temperature=args.temperature,
            n_steps=args.mc_steps,
            cooling_factor=args.cooling_factor,
            output_dir=output_dir
        )
        all_results = mc_algorithm.run_sampling(protein, ligand)
    elif hasattr(args, 'enhanced') and args.enhanced:
        logger.info("Using enhanced rigid docking algorithm...")
        all_results = search_algorithm.improve_rigid_docking(protein, ligand, args)
    else:
        all_results = search_algorithm.search(protein, ligand)
    
    # Clean up hybrid manager
    if hybrid_manager:
        hybrid_manager.cleanup()
    
    return all_results


def apply_post_processing(all_results: List[Tuple], protein: Protein, args: argparse.Namespace,
                         output_dir: str, logger: Any) -> List[Tuple]:
    """
    Apply post-processing steps to docking results.
    
    Args:
        all_results: List of (pose, score) tuples
        protein: Protein object
        args: Command-line arguments
        output_dir: Output directory path
        logger: Logger instance
        
    Returns:
        List of processed (pose, score) tuples
    """
    if not all_results:
        logger.warning("No valid docking solutions found.")
        return []
    
    # Update status after search completes
    update_status(
        output_dir,
        status="post_processing",
        initial_poses_count=len(all_results)
    )
    
    # Apply local optimization if requested
    optimized_results = []
    if args.local_opt and all_results:
        logger.info("Performing local optimization on top poses (enabled by --local-opt)...")
        update_status(output_dir, status="local_optimization")
        
        poses_to_optimize = min(10, len(all_results))
        sorted_initial_results = sorted(all_results, key=lambda x: x[1])
        
        for i, (pose, score) in enumerate(sorted_initial_results[:poses_to_optimize]):
            logger.info(f"Optimizing pose {i+1} (initial score: {score:.2f})...")
            update_status(
                output_dir,
                optimizing_pose=i+1,
                initial_score=score,
                total_poses_to_optimize=poses_to_optimize
            )
            
            try:
                # Apply local optimization (implementation depends on search algorithm)
                if hasattr(args, 'mmff_minimization') and args.mmff_minimization and PHYSICS_AVAILABLE:
                    logger.info("Using MMFF minimization in protein environment")
                    minimizer = MMFFMinimization()
                    opt_pose = minimizer.minimize_pose(protein, pose)
                    # Re-score optimized pose
                    from .unified_scoring import CompositeScoringFunction
                    scoring_function = CompositeScoringFunction()
                    opt_score = scoring_function.score(protein, opt_pose)
                    optimized_results.append((opt_pose, opt_score))
                else:
                    # Keep original pose if no optimization method available
                    optimized_results.append((pose, score))
                
                # Save intermediate result
                save_intermediate_result(pose, score, i, output_dir)
                
            except Exception as e:
                logger.warning(f"Error during local optimization of pose {i+1}: {e}")
                optimized_results.append((pose, score))
        
        # Combine optimized results with remaining unoptimized results
        optimized_indices = set(range(poses_to_optimize))
        remaining_results = [r for i, r in enumerate(sorted_initial_results) 
                           if i not in optimized_indices]
        
        all_results = optimized_results + remaining_results
        logger.info("Local optimization complete.")
        update_status(
            output_dir,
            status="optimization_complete",
            optimized_poses_count=len(optimized_results)
        )
    
    # Sort final results
    all_results.sort(key=lambda x: x[1])
    
    # Remove duplicates and filter results
    unique_results = []
    seen_scores = set()
    
    for pose, score in all_results:
        # Round score to avoid floating point comparison issues
        rounded_score = round(score, 4)
        if rounded_score not in seen_scores:
            unique_results.append((pose, score))
            seen_scores.add(rounded_score)
            
            # Keep at most 20 poses
            if len(unique_results) >= 20:
                break
    
    update_status(
        output_dir,
        status="analyzing_results",
        final_poses_count=len(unique_results),
        best_score=unique_results[0][1] if unique_results else None
    )
    
    return unique_results


def perform_analysis(all_results: List[Tuple], protein: Protein, args: argparse.Namespace,
                    output_dir: str, logger: Any) -> dict:
    """
    Perform advanced analysis if requested.
    
    Args:
        all_results: List of (pose, score) tuples
        protein: Protein object
        args: Command-line arguments
        output_dir: Output directory path
        logger: Logger instance
        
    Returns:
        dict: Analysis results
    """
    analysis_results = {}
    
    if not (args.cluster_poses or args.analyze_interactions or args.classify_modes or
            args.energy_decomposition or args.generate_analysis_report):
        return analysis_results
    
    if not all_results:
        logger.info("Skipping analysis as no valid docking solutions were found.")
        return analysis_results
    
    try:
        from .analysis import (
            PoseClusterer, InteractionFingerprinter,
            BindingModeClassifier, EnergyDecomposition,
            DockingReportGenerator
        )
    except ImportError as e:
        logger.warning(f"Analysis modules not available: {e}")
        return analysis_results
    
    logger.info("Performing advanced analysis...")
    update_status(output_dir, status="analyzing")
    
    # Extract poses and scores
    poses = [pose for pose, _ in all_results]
    scores = [score for _, score in all_results]
    
    # Clustering
    if args.cluster_poses:
        logger.info("Clustering docking poses...")
        update_status(output_dir, status="clustering")
        
        clusterer = PoseClusterer(
            method=args.clustering_method,
            rmsd_cutoff=args.rmsd_cutoff
        )
        clustering_results = clusterer.cluster_poses(poses)
        analysis_results['clustering'] = clustering_results
        
        logger.info(f"Found {len(clustering_results['clusters'])} clusters")
        for i, cluster in enumerate(clustering_results['clusters']):
            logger.info(f"Cluster {i+1}: {len(cluster['members'])} poses, "
                       f"best score: {cluster['best_score']:.2f}")
        
        update_status(
            output_dir,
            clustering_complete=True,
            clusters_count=len(clustering_results['clusters'])
        )
    
    # Interaction analysis
    if args.analyze_interactions:
        logger.info("Analyzing protein-ligand interactions...")
        update_status(output_dir, status="analyzing_interactions")
        
        interaction_file = os.path.join(output_dir, "interaction_analysis.txt")
        fingerprinter = InteractionFingerprinter(
            interaction_types=args.interaction_types
        )
        
        poses_to_analyze = min(5, len(all_results))
        
        with open(interaction_file, 'w') as f:
            f.write("=" * 40 + "\n")
            f.write(" Protein-Ligand Interaction Analysis\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Analyzing top {poses_to_analyze} poses:\n\n")
            
            for i, (pose, score) in enumerate(all_results[:poses_to_analyze]):
                f.write(f"Pose {i+1} (Score: {score:.2f})\n")
                f.write("-" * 28 + "\n")
                
                key_interactions = fingerprinter.analyze_key_interactions(protein, pose)
                for interaction in key_interactions:
                    f.write(f"  {interaction}\n")
                
                f.write("\n")
                
                logger.info(f"Interactions for pose {i+1} (score: {score:.2f}):")
                for interaction in key_interactions:
                    logger.info(f"  {interaction}")
        
        logger.info(f"Interaction analysis saved to: {interaction_file}")
        analysis_results['interactions'] = interaction_file
    
    # Binding mode analysis
    if args.classify_modes or args.discover_modes:
        logger.info("Analyzing binding modes...")
        update_status(output_dir, status="analyzing_binding_modes")
        
        classifier = BindingModeClassifier()
        
        if args.discover_modes:
            discovered_modes = classifier.discover_modes(
                protein, poses, n_modes=args.n_modes
            )
            logger.info(f"Discovered {len(discovered_modes)} binding modes")
            for i, mode in enumerate(discovered_modes):
                logger.info(f"Mode {i+1}: {mode['count']} poses, "
                           f"best score: {mode['best_score']:.2f}")
            analysis_results['discovered_modes'] = discovered_modes
        
        if args.classify_modes:
            poses_to_classify = min(10, len(all_results))
            classifications = []
            for i, (pose, score) in enumerate(all_results[:poses_to_classify]):
                mode = classifier.classify_pose(protein, pose)
                logger.info(f"Pose {i+1} (score: {score:.2f}): {mode}")
                classifications.append(mode)
            analysis_results['classifications'] = classifications
    
    # Energy decomposition
    if args.energy_decomposition:
        logger.info("Performing energy decomposition analysis...")
        update_status(output_dir, status="energy_decomposition")
        
        from .unified_scoring import CompositeScoringFunction
        scoring_function = CompositeScoringFunction()
        decomposer = EnergyDecomposition(scoring_function)
        
        # Analyze top pose
        top_pose = all_results[0][0]
        energy_decomposition = decomposer.decompose_energy(protein, top_pose)
        analysis_results['energy_decomposition'] = energy_decomposition
        
        logger.info("Energy components for top pose:")
        for component, value in energy_decomposition.items():
            logger.info(f"  {component}: {value:.2f}")
        
        if args.per_residue_energy:
            logger.info("Top residue contributions:")
            res_contributions = decomposer.residue_contributions(protein, top_pose)
            for res, value in res_contributions[:5]:
                logger.info(f"  {res}: {value:.2f}")
            analysis_results['residue_contributions'] = res_contributions
    
    # Generate comprehensive report
    if args.generate_analysis_report:
        logger.info("Generating comprehensive docking report...")
        update_status(output_dir, status="generating_report")
        
        report_generator = DockingReportGenerator(
            report_format=args.analysis_report_format,
            include_sections=args.analysis_report_sections
        )
        
        report_file = os.path.join(output_dir, f"docking_report.{args.analysis_report_format}")
        report_generator.generate_report(
            protein, poses, scores, report_file,
            clustering_results=analysis_results.get('clustering'),
            energy_decomposition=analysis_results.get('energy_decomposition')
        )
        logger.info(f"Report generated: {report_file}")
        analysis_results['report_file'] = report_file
    
    return analysis_results


def save_and_report_results(unique_results: List[Tuple], protein: Protein, 
                          args: argparse.Namespace, output_dir: str, 
                          elapsed_time: float, logger: Any) -> None:
    """
    Save results and generate reports.
    
    Args:
        unique_results: List of unique (pose, score) tuples
        protein: Protein object
        args: Command-line arguments
        output_dir: Output directory path
        elapsed_time: Total elapsed time
        logger: Logger instance
    """
    if not unique_results:
        logger.warning("No valid docking solutions found.")
        update_status(output_dir, status="completed", success=False)
        return
    
    logger.info("Docking completed successfully!")
    logger.info(f"Saving results to {output_dir}...")
    update_status(output_dir, status="saving_results")
    
    # Get flexible residues if they exist
    flexible_residues = None
    if hasattr(protein, 'flexible_residues') and protein.flexible_residues:
        flexible_residues = protein.flexible_residues
        logger.info(f"Found {len(flexible_residues)} flexible residues to include in output")
    
    # Save docking results
    save_docking_results(unique_results, output_dir, flexible_residues=flexible_residues)
    
    # Save protein-ligand complexes
    for i, (pose, score) in enumerate(unique_results[:10]):
        complex_path = Path(output_dir) / f"complex_pose_{i+1}_score_{score:.2f}.pdb"
        save_complex_to_pdb(protein, pose, complex_path)
    
    # Validate against reference if provided
    validation_results = None
    if hasattr(args, 'reference') and args.reference and not getattr(args, 'exact_alignment', False):
        logger.info("Validating against reference structure...")
        update_status(output_dir, status="validating")
        validation_results = validate_against_reference(args, unique_results, output_dir)
    
    # Generate comprehensive reports
    logger.info("Generating comprehensive docking reports...")
    update_status(output_dir, status="generating_reports")
    
    # Initialize reporter
    readable_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    reporter = DockingReporter(output_dir, args, timestamp=readable_date)
    
    # Extract energy components for reporting
    try:
        from .unified_scoring import CompositeScoringFunction
        scoring_function = CompositeScoringFunction()
        
        logger.info("Extracting energy components for detailed reporting...")
        energy_breakdown = reporter.extract_energy_components(
            scoring_function,
            protein,
            [pose for pose, _ in unique_results[:min(20, len(unique_results))]]
        )
        reporter.add_results(unique_results, energy_breakdown=energy_breakdown)
        
        # Generate reports
        reporter.generate_detailed_report(include_energy_breakdown=True)
        reporter.generate_csv_report()
        reporter.generate_json_report()
        html_report = reporter.generate_html_report()
        reporter.generate_binding_affinity_report()
        reporter.plot_energy_breakdown()
        
    except Exception as e:
        logger.warning(f"Could not extract energy components: {e}")
        reporter.add_results(unique_results)
        
        # Generate basic reports
        reporter.generate_basic_report()
        reporter.generate_csv_report()
        reporter.generate_json_report()
    
    # Add validation results if available
    if validation_results:
        reporter.add_validation_results(validation_results)
    
    # Write detailed results to text file
    write_results_to_txt(
        results=unique_results,
        output_dir=output_dir,
        elapsed_time=elapsed_time,
        protein_path=args.protein,
        ligand_path=args.ligand,
        algorithm=get_algorithm_type_from_args(args),
        iterations=args.iterations if get_algorithm_type_from_args(args) != 'monte-carlo' else args.mc_steps,
        logger=logger
    )
    
    # Print summary
    logger.info(f"Docking completed in {elapsed_time:.1f} seconds")
    logger.info(f"Best score: {unique_results[0][1]:.2f}")
    logger.info(f"Results saved to: {output_dir}")
    
    # Final status update
    update_status(
        output_dir,
        status="completed",
        success=True,
        elapsed_time=elapsed_time,
        best_score=unique_results[0][1],
        final_poses_count=len(unique_results)
    )


def print_success_message():
    """Print success message with ASCII art."""
    success_ascii = r"""
╔════════════════════════════════╗
║   🎉 Successful Docking! 🎉    ║
║                                ║
║   🐼  PandaDock Completed!     ║ 
║                                ║
║ Dock Smarter. Discover Faster. ║
╚════════════════════════════════╝
"""
    print(success_ascii)


def print_error_message(args_available: bool = True):
    """Print error message with appropriate ASCII art."""
    if args_available:
        # Docking failed after valid command
        error_ascii = r"""
                                        \ / _
                                    ___,,,
                                    \_[o o]
    Errare humanum est!              C\  _\/
            /                     _____),_/__
        ________                  /     \/   /
    _|       .|                /      o   /
    | |       .|               /          /
    \|       .|              /          /
    |________|             /_        \/
    __|___|__             _//\        \
_____|_________|____       \  \ \        \
                    _|       ///  \        \
                |               \       /
                |               /      /
                |              /      /
________________  |             /__    /_
b'ger        ...|_|.............. /______\.......

            ❌ Error: The docking process encountered an issue! 🐼💥
"""
        print(error_ascii)
    else:
        # Argument parsing failure
        print("🐼❌ PandaDock Error: Invalid command or missing inputs. Use '--help' for guidance!")

def setup_student_researcher_defaults(args):
    """
    Optimize PandaDock for students and researchers using CPU.
    Provides GPU-quality results with CPU-friendly performance.
    """
    print("\n🎓 PandaDock: Optimized for Academic Research")
    print("💻 CPU-Focused • 🚀 Parallel Processing • 📊 High-Quality Results\n")
    
    # Force CPU-optimized settings
    original_settings = {}
    
    # 1. Always use enhanced scoring (better results)
    if not getattr(args, 'enhanced_scoring', False):
        original_settings['enhanced_scoring'] = getattr(args, 'enhanced_scoring', False)
        args.enhanced_scoring = True
        print("✅ Enabled enhanced scoring for better accuracy")
    
    # 2. Disable GPU (ensure compatibility)
    if getattr(args, 'use_gpu', False):
        original_settings['use_gpu'] = args.use_gpu
        args.use_gpu = False
        print("💻 Using CPU mode for maximum compatibility")
    
    # 3. Disable physics-based (too slow for CPU)
    if getattr(args, 'physics_based', False):
        original_settings['physics_based'] = args.physics_based
        args.physics_based = False
        print("⚡ Disabled physics-based scoring for faster execution")
    
    # 4. Enable local optimization (better poses)
    if not getattr(args, 'local_opt', False):
        original_settings['local_opt'] = getattr(args, 'local_opt', False)
        args.local_opt = True
        print("🎯 Enabled local optimization for pose refinement")
    
    # 5. Set optimal population size
    if getattr(args, 'population_size', 50) < 50:
        original_settings['population_size'] = getattr(args, 'population_size', 50)
        args.population_size = 50
        print("👥 Set population size to 50 for quality/speed balance")
    
    # 6. Ensure sufficient iterations
    if getattr(args, 'iterations', 10) < 50:
        original_settings['iterations'] = getattr(args, 'iterations', 10)
        args.iterations = 100
        print("🔄 Increased iterations to 100 for thorough search")
    
    # 7. Use all CPU cores
    import os
    if not getattr(args, 'cpu_workers', None):
        args.cpu_workers = os.cpu_count()
        print(f"🚀 Using all {args.cpu_workers} CPU cores for parallel processing")
    
    # 8. Recommend best algorithms for students
    if getattr(args, 'algorithm', 'genetic') not in ['genetic', 'hybrid', 'random']:
        original_settings['algorithm'] = args.algorithm
        args.algorithm = 'genetic'
        print("🧬 Using genetic algorithm (recommended for students)")
    
    # 9. Auto-enable molecule preparation
    if not getattr(args, 'prepare_molecules', False):
        original_settings['prepare_molecules'] = getattr(args, 'prepare_molecules', False)
        args.prepare_molecules = True
        print("🧪 Enabled molecule preparation for better results")
    
    print(f"\n📊 Final Configuration:")
    print(f"   Algorithm: {args.algorithm}")
    print(f"   Enhanced Scoring: {args.enhanced_scoring}")
    print(f"   Local Optimization: {args.local_opt}")
    print(f"   Population Size: {args.population_size}")
    print(f"   Iterations: {args.iterations}")
    print(f"   CPU Workers: {args.cpu_workers}")
    print(f"   Molecule Prep: {args.prepare_molecules}")
    print("")
    
    return args, original_settings


def print_student_success_message():
    """Print success message optimized for academic users."""
    success_message = f"""
╔══════════════════════════════════════════╗
║  🎉 PandaDock Completed Successfully! 🎉 ║
║                                          ║
║  🎓 Perfect for Academic Research        ║
║  💻 CPU-Optimized Performance            ║
║  📊 Publication-Ready Results            ║
║                                          ║
║  🔬 Ready for Analysis & Visualization   ║
╚══════════════════════════════════════════╝
"""
    print(success_message)


# =============================================================================
#Metal Docking Setup in main() function
# =============================================================================

def setup_metal_docking(protein, args, logger):
    """
    Setup metal docking components based on arguments.
    
    Parameters:
    -----------
    protein : Protein
        Protein object
    args : argparse.Namespace
        Command-line arguments
    logger : logging.Logger
        Logger instance
        
    Returns:
    --------
    tuple
        (metal_centers, metal_constraints, is_metal_docking)
    """
    if not args.metal_docking:
        return [], [], False
    
    logger.info("Setting up metal-aware docking...")
    
    # Import metal docking components
    from .metal_docking import MetalDockingPreparation, MetalCenter
    
    metal_centers = []
    
    # Method 1: Auto-detect metals
    if args.detect_metals:
        logger.info("Auto-detecting metal centers...")
        detected_centers = MetalDockingPreparation.detect_metal_centers(
            protein, metal_elements=args.metal_elements
        )
        metal_centers.extend(detected_centers)
        logger.info(f"Detected {len(detected_centers)} metal centers")
        
        for i, center in enumerate(detected_centers):
            logger.info(f"  Metal {i+1}: {center.element} at {center.coords}")
            logger.info(f"    Coordination: {center.coordination_number} atoms")
    
    # Method 2: User-specified metal centers
    if args.metal_centers:
        logger.info("Using user-specified metal centers...")
        for metal_spec in args.metal_centers:
            try:
                coords_str, element = metal_spec.split(':')
                x, y, z = map(float, coords_str.split(','))
                
                # Create artificial metal atom
                metal_atom = {
                    'coords': np.array([x, y, z]),
                    'element': element.upper(),
                    'name': element.upper(),
                    'residue_name': 'MET',
                    'residue_number': 999
                }
                
                metal_center = MetalCenter(
                    metal_atom,
                    coordination_geometry=args.coordination_geometry,
                    max_coordination=args.max_coordination
                )
                
                metal_centers.append(metal_center)
                logger.info(f"Added metal center: {element} at ({x:.2f}, {y:.2f}, {z:.2f})")
                
            except ValueError as e:
                logger.warning(f"Invalid metal center specification '{metal_spec}': {e}")
    
    if not metal_centers:
        logger.warning("No metal centers found or specified. Metal docking disabled.")
        return [], [], False
    
    # Create metal constraints
    logger.info("Creating metal coordination constraints...")
    metal_constraints = MetalDockingPreparation.create_metal_constraints(
        metal_centers,
        constraint_types=['coordination'],
        strengths=[args.metal_constraint_strength]
    )
    
    logger.info(f"Created {len(metal_constraints)} metal constraints")
    
    return metal_centers, metal_constraints, True


def create_metal_aware_scoring_function(base_scoring_function, metal_centers, metal_constraints, args):
    """
    Create metal-aware scoring function.
    
    Parameters:
    -----------
    base_scoring_function : ScoringFunction
        Base scoring function
    metal_centers : list
        List of metal centers
    metal_constraints : list
        List of metal constraints
    args : argparse.Namespace
        Command-line arguments
        
    Returns:
    --------
    MetalDockingScorer
        Metal-aware scoring function
    """
    from .metal_docking import MetalDockingScorer
    
    metal_scorer = MetalDockingScorer(
        base_scoring_function=base_scoring_function,
        metal_centers=metal_centers,
        metal_constraints=metal_constraints,
        metal_weight=args.coordination_weight
    )
    
    return metal_scorer


def create_metal_aware_search_algorithm(base_algorithm, metal_centers, metal_constraints):
    """
    Create metal-aware search algorithm.
    
    Parameters:
    -----------
    base_algorithm : SearchAlgorithm
        Base search algorithm
    metal_centers : list
        List of metal centers
    metal_constraints : list
        List of metal constraints
        
    Returns:
    --------
    MetalDockingSearch
        Metal-aware search algorithm
    """
    from .metal_docking import MetalDockingSearch
    
    metal_search = MetalDockingSearch(
        base_search_algorithm=base_algorithm,
        metal_centers=metal_centers,
        metal_constraints=metal_constraints
    )
    
    return metal_search

def get_error_help(error_type, error_message):
    """
    Provide specific help based on common error types.
    
    Parameters:
    -----------
    error_type : str
        Type of error
    error_message : str
        Error message
        
    Returns:
    --------
    str or None
        Help message if available
    """
    error_message_lower = error_message.lower()
    
    if error_type == "FileNotFoundError":
        return "Check that all input files exist and paths are correct"
    
    elif error_type == "MemoryError":
        return "Reduce population size, iterations, or use --fast-mode"
    
    elif error_type == "KeyError" and "active_site" in error_message_lower:
        return "Try using --site X,Y,Z to specify binding site coordinates"
    
    elif error_type == "AttributeError" and "score" in error_message_lower:
        return "Check scoring function initialization - try --enhanced-scoring"
    
    elif error_type == "RuntimeError" and ("cuda" in error_message_lower or "gpu" in error_message_lower):
        return "GPU error detected - try removing --use-gpu flag"
    
    elif error_type == "IndexError" or error_type == "ValueError":
        if "xyz" in error_message_lower or "coordinates" in error_message_lower:
            return "Check input structure files for correct format and coordinates"
    
    elif "timeout" in error_message_lower or "time" in error_message_lower:
        return "Process timed out - try reducing --iterations or --population-size"
    
    elif "permission" in error_message_lower:
        return "Check file permissions for output directory"
    
    return None
# ============================================================================
# Main Function
# ============================================================================

def main() -> int:
    """
    Main entry point for PandaDock.
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # Initialize return value and timing
    return_code = 0
    start_time = time.time()
    args = None
    output_dir = None
    logger = None
    
    try:
        # Print header
        print_pandadock_header()
        
        # Parse command line arguments
        parser = setup_argument_parser()
        add_metal_docking_options(parser)
        args = parser.parse_args()
        # Store full command for reproducibility
        full_command = "pandadock " + " ".join(sys.argv[1:])
        args.full_command = full_command
        
        # Validate required arguments
        if not args.protein:
            parser.error("--protein argument is required")
        
        if not args.ligand and not args.virtual_screening:
            parser.error("Either --ligand or --virtual-screening argument is required")
        
        if args.virtual_screening and not args.ligand_library:
            parser.error("--ligand-library is required for virtual screening")
        
        # Validate input files
        if not validate_input_files(args.protein, args.ligand):
            return 1
        
        # Create output directory
        output_dir = create_output_directory(
            args.output, args.protein, args.ligand, args.algorithm
        )
        
        # Create initial files and setup logging
        create_initial_files(output_dir, args)
        logger = setup_logging(output_dir)
        
        logger.info("=" * 60)
        logger.info("PandaDock - Python Molecular Docking")
        logger.info("=" * 60)
        logger.info(f"PandaDock starting - output will be saved to {output_dir}")
        
        # Check for updates
        check_for_updates(logger)
        
        # Apply preprocessing options
        apply_preprocessing_options(args, logger)
        
        # Validate physics-based scoring requirements
        if args.physics_based and not args.enhanced_scoring:
            parser.error("Physics-based scoring requires --enhanced-scoring")
        
        # Handle virtual screening workflow
        if args.virtual_screening:
            return handle_virtual_screening(args, logger)
        
        # Update status
        update_status(
            output_dir,
            algorithm=args.algorithm,
            physics_based=args.physics_based,
            enhanced_scoring=args.enhanced_scoring,
            local_opt=args.local_opt,
            exhaustiveness=args.exhaustiveness,
            status="initializing"
        )
        
        # Setup protein and ligand
        protein, ligand, protein_path, ligand_path = setup_protein_and_ligand(args, logger)
        
        # Setup active site
        setup_active_site(protein, output_dir, args, logger)
        
        # Handle flexible residues if requested
        if args.auto_flex or args.flex_residues:
            logger.info("Preparing flexible protein configurations...")
            update_status(output_dir, status="preparing_flexible_residues")
            configs = prepare_protein_configs(protein, args, logger)
            
            if len(configs) > 1:
                logger.info(f"Using flexible protein configuration with {len(configs[1]['flexible_residues'])} flexible residues")
                protein = configs[1]['protein']
                update_status(
                    output_dir,
                    flexible_residues=configs[1]['flexible_residues'],
                    flexible_residues_count=len(configs[1]['flexible_residues'])
                )
            else:
                logger.info("No flexible configuration available, using rigid protein")
                update_status(output_dir, flexible_residues_count=0)
        
        metal_centers, metal_constraints, is_metal_docking = setup_metal_docking(protein, args, logger)
        if is_metal_docking:
            logger.info("Creating metal-aware scoring function...")
            scoring_type = get_scoring_type_from_args(args)
            base_scoring_function = create_optimized_scoring_function(args, configure_hardware(args))
            scoring_function = create_metal_aware_scoring_function(
                base_scoring_function, metal_centers, metal_constraints, args
            )
            update_status(output_dir, scoring_function="metal-aware")
        else:
            scoring_type = get_scoring_type_from_args(args)
            scoring_function = create_optimized_scoring_function(args, configure_hardware(args))
        # Create enhanced search algorithm
        algorithm_type = get_algorithm_type_from_args(args)
        algorithm_kwargs = get_algorithm_kwargs_from_args(args)
        hybrid_manager =  configure_hardware(args)
        base_search_algorithm = create_optimized_search_algorithm(
            hybrid_manager, algorithm_type, scoring_function, **algorithm_kwargs, hw_config=configure_hardware(args)
        )        
        search_algorithm = create_metal_aware_search_algorithm(
            base_search_algorithm, metal_centers, metal_constraints, args
        ) if is_metal_docking else base_search_algorithm
        
        # Analyze ligand for metal binding potential
        if is_metal_docking:
            logger.info("Analyzing ligand metal binding potential...")
            from .metal_docking import MetalDockingPreparation
            
            ligand_analysis = MetalDockingPreparation.analyze_ligand_metal_binding_sites(ligand)
            logger.info(f"Ligand has {ligand_analysis['total_coordinating_atoms']} potential coordinating atoms")
            logger.info(f"Found {len(ligand_analysis['chelating_groups'])} potential chelating groups")
            
            if args.require_coordination and ligand_analysis['total_coordinating_atoms'] == 0:
                logger.error("ERROR: --require-coordination specified but ligand has no coordinating atoms!")
                return 1
            
            # Save analysis if requested
            if args.save_metal_analysis:
                analysis_file = Path(output_dir) / "metal_analysis.json"
                with open(analysis_file, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    analysis_copy = copy.deepcopy(ligand_analysis)
                    for site in analysis_copy['binding_sites']:
                        site['coords'] = site['coords'].tolist()
                    json.dump(analysis_copy, f, indent=2)
                logger.info(f"Metal analysis saved to {analysis_file}")
        # Run docking search
        all_results = run_docking_search(protein, ligand, args, logger, output_dir)
        
        # Apply post-processing
        unique_results = apply_post_processing(all_results, protein, args, output_dir, logger)
        
        # Perform analysis if requested
        analysis_results = perform_analysis(unique_results, protein, args, output_dir, logger)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Save and report results
        save_and_report_results(unique_results, protein, args, output_dir, elapsed_time, logger)
        
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        if logger:
            logger.info(f"\nDocking interrupted by user after {elapsed_time:.1f} seconds")
        else:
            print(f"\nDocking interrupted by user after {elapsed_time:.1f} seconds")
        return_code = 1
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        
        # Get detailed error information
        error_type = type(e).__name__
        error_message = str(e)
        error_traceback = traceback.format_exc()
        
        # Print comprehensive error to console IMMEDIATELY
        print(f"\n🚨 PANDADOCK ERROR DETECTED 🚨")
        print(f"=" * 60)
        print(f"Error Type: {error_type}")
        print(f"Error Message: {error_message}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed Time: {elapsed_time:.1f} seconds")
        print(f"\n📍 Location in code:")
        
        # Get the last few lines of traceback for quick debugging
        tb_lines = error_traceback.strip().split('\n')
        if len(tb_lines) > 4:
            print("...")
            for line in tb_lines[-4:]:
                print(line)
        else:
            print(error_traceback)
        
        print(f"=" * 60)
        
        # Enhanced status update with error details
        if output_dir:
            try:
                error_info = {
                    'type': error_type,
                    'message': error_message,
                    'traceback': error_traceback,
                    'elapsed_time': elapsed_time,
                    'timestamp': time.time(),
                    'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'command_line': ' '.join(sys.argv)
                }
                
                # Write to status.json
                status_file = Path(output_dir) / "status.json"
                status_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(status_file, 'w') as f:
                    json.dump({
                        'status': 'error',
                        'error': error_info,
                        'timestamp': time.time(),
                        'datetime': time.strftime('%Y-%m-%d %H:%M:%S')
                    }, f, indent=2)
                
                # Write human-readable error log
                error_log = Path(output_dir) / "error_log.txt"
                with open(error_log, 'w') as f:
                    f.write(f"PandaDock Error Log\n")
                    f.write(f"=" * 30 + "\n")
                    f.write(f"Time: {error_info['datetime']}\n")
                    f.write(f"Command: {error_info['command_line']}\n")
                    f.write(f"Error Type: {error_type}\n")
                    f.write(f"Error Message: {error_message}\n")
                    f.write(f"Elapsed Time: {elapsed_time:.1f} seconds\n")
                    f.write(f"\nFull Traceback:\n")
                    f.write(error_traceback)
                
                print(f"\n📋 Error details saved to:")
                print(f"   • {status_file}")
                print(f"   • {error_log}")
                
            except Exception as save_error:
                print(f"[WARNING] Could not save error details: {save_error}")
        
        # Try to provide specific help based on error type
        help_message = get_error_help(error_type, error_message)
        if help_message:
            print(f"\n💡 Possible solution:")
            print(f"   {help_message}")
        
        # Show error ASCII (your existing one)
        error_ascii = r"""
                                        \ / _
                                    ___,,,
                                    \_[o o]
    Errare humanum est!              C\  _\/
            /                     _____),_/__
        ________                  /     \/   /
    _|       .|                /      o   /
    | |       .|               /          /
    \|       .|              /          /
    |________|             /_        \/
    __|___|__             _//\        \
_____|_________|____       \  \ \        \
                    _|       ///  \        \
                |               \       /
                |               /      /
                |              /      /
________________  |             /__    /_
b'ger        ...|_|.............. /______\.......

            ❌ Error: The docking process encountered an issue! 🐼💥
        """
        print(error_ascii)
        
        print(f"\nDocking failed after {elapsed_time:.1f} seconds")
        return_code = 1
    
    finally:
        # Clean up temporary files
        temp_dir = Path('prepared_files_for_pandadock')
        if temp_dir.exists() and hasattr(args, 'prepare_molecules') and args.prepare_molecules:
            import shutil
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                if logger:
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
                else:
                    print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to clean up temporary directory: {e}")
                else:
                    print(f"Failed to clean up temporary directory: {e}")
        
        # Print final message
        if return_code == 0:
            print_success_message()
        else:
            print_error_message(args is not None)
        
        if logger:
            logger.info("=" * 60)
        else:
            print("=" * 60)
    
    return return_code


if __name__ == "__main__":
    sys.exit(main())