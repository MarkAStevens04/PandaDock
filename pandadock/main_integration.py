"""
CLI integration module for PandaDock hardware acceleration.

This module extends the main.py command-line interface with GPU/CPU options
and provides utilities for hardware configuration and algorithm optimization.
"""

import argparse
import os
import logging
from typing import Dict, Any, Optional, Union
import torch
import numpy as np
import logging

# Core utilities
import psutil

# Factory and algorithm imports
from .unified_scoring import (
    ScoringFunction,
    CompositeScoringFunction,
    EnhancedScoringFunction,
    GPUScoringFunction,
    EnhancedGPUScoringFunction,
    TetheredScoringFunction,
)
from .scoring_factory import create_scoring_function
from .search import GeneticAlgorithm, RandomSearch
from .parallel_search import (
    ParallelSearch, ParallelGeneticAlgorithm, ParallelRandomSearch,
    HybridSearch, FlexibleLigandSearch, create_search_algorithm
)
from .pandadock import PANDADOCKAlgorithm

# Physics-based models (optional)
try:
    from .physics import (
        MMFFMinimization, MonteCarloSampling, PhysicsBasedScoring,
        GeneralizedBornSolvation
    )
    PHYSICS_AVAILABLE = True
except ImportError:
    PHYSICS_AVAILABLE = False

# Preparation and validation utilities
from .preparation import prepare_protein, prepare_ligand
from .validation import validate_docking, calculate_ensemble_rmsd
from .utils import save_docking_results, save_complex_to_pdb, calculate_rmsd


# ============================================================================
# Hardware Configuration Functions
# ============================================================================

def add_hardware_options(parser: argparse.ArgumentParser) -> None:
    """
    Add hardware acceleration options to the argument parser.
    
    Args:
        parser: Argument parser to modify
    """
    hw_group = parser.add_argument_group('Hardware Acceleration')
    
    # GPU options
    hw_group.add_argument('--use-gpu', action='store_true',
                         help='Use GPU acceleration if available')
    hw_group.add_argument('--gpu-id', type=int, default=0,
                         help='GPU device ID to use (default: 0)')
    hw_group.add_argument('--gpu-precision', choices=['float32', 'float64'],
                         default='float32',
                         help='Numerical precision for GPU calculations (default: float32)')
    
    # CPU options
    hw_group.add_argument('--cpu-workers', type=int, default=None,
                         help='Number of CPU workers for parallel processing (default: all cores)')
    hw_group.add_argument('--cpu-affinity', action='store_true',
                         help='Set CPU affinity for better performance')
    
    # Hybrid options
    hw_group.add_argument('--workload-balance', type=float, default=None,
                         help='GPU/CPU workload balance (0.0-1.0, higher = more GPU work)')
    hw_group.add_argument('--auto-tune', action='store_true',
                         help='Automatically tune hardware parameters for best performance')
    
    # Hybrid algorithm specific options
    hw_group.add_argument('--hybrid-temperature-start', type=float, default=5.0,
                         help='Starting temperature for hybrid search algorithm (default: 5.0)')
    hw_group.add_argument('--hybrid-temperature-end', type=float, default=0.1,
                         help='Ending temperature for hybrid search algorithm (default: 0.1)')
    hw_group.add_argument('--hybrid-cooling-factor', type=float, default=0.95,
                         help='Cooling factor for hybrid search algorithm (default: 0.95)')
    
    # Flexible ligand docking options
    hw_group.add_argument('--max-torsions', type=int, default=10,
                         help='Maximum number of torsions to sample in flexible ligand docking (default: 10)')

def detect_and_configure_gpu(args, logger=None):
    """
    Enhanced GPU detection and configuration with proper fallback handling.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    logger : logging.Logger, optional
        Logger instance
        
    Returns:
    --------
    dict
        Enhanced hardware configuration with GPU details
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    hw_config = {
        'use_gpu': False,
        'gpu_available': False,
        'gpu_provider': None,
        'gpu_device': None,
        'gpu_memory': 0,
        'cuda_version': None,
        'gpu_count': 0
    }
    
    # Only proceed if user explicitly requested GPU
    if not getattr(args, 'use_gpu', False):
        logger.info("[GPU] GPU acceleration not requested")
        return hw_config
    
    logger.info("[GPU] GPU acceleration requested - checking availability...")
    
    # Check PyTorch CUDA availability
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_id = getattr(args, 'gpu_id', 0)
            
            # Validate GPU ID
            if gpu_id >= gpu_count:
                logger.warning(f"[GPU] Requested GPU ID {gpu_id} not available. Using GPU 0 instead.")
                gpu_id = 0
                args.gpu_id = 0
            
            # Set device
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device)
            
            # Get GPU information
            gpu_props = torch.cuda.get_device_properties(gpu_id)
            gpu_name = gpu_props.name
            gpu_memory = gpu_props.total_memory / (1024**3)  # Convert to GB
            cuda_version = torch.version.cuda
            
            # Test GPU functionality
            try:
                # Simple tensor operation to test GPU
                test_tensor = torch.randn(100, 100, device=device)
                test_result = torch.matmul(test_tensor, test_tensor)
                del test_tensor, test_result
                torch.cuda.empty_cache()
                
                # GPU is working
                hw_config.update({
                    'use_gpu': True,
                    'gpu_available': True,
                    'gpu_provider': 'pytorch',
                    'gpu_device': device,
                    'gpu_id': gpu_id,
                    'gpu_name': gpu_name,
                    'gpu_memory': gpu_memory,
                    'cuda_version': cuda_version,
                    'gpu_count': gpu_count
                })
                
                logger.info(f"[GPU] ✅ PyTorch CUDA initialized successfully")
                logger.info(f"[GPU] Device: {gpu_name}")
                logger.info(f"[GPU] Memory: {gpu_memory:.2f} GB")
                logger.info(f"[GPU] CUDA Version: {cuda_version}")
                logger.info(f"[GPU] Using GPU {gpu_id} of {gpu_count} available")
                
                return hw_config
                
            except Exception as e:
                logger.error(f"[GPU] GPU functionality test failed: {e}")
                
        else:
            logger.warning("[GPU] PyTorch reports CUDA is not available")
            
    except ImportError:
        logger.warning("[GPU] PyTorch not installed")
    except Exception as e:
        logger.error(f"[GPU] Error checking PyTorch CUDA: {e}")
    
    # Try CuPy as fallback
    logger.info("[GPU] Trying CuPy as fallback...")
    try:
        import cupy as cp
        
        # Test CuPy functionality
        gpu_id = getattr(args, 'gpu_id', 0)
        with cp.cuda.Device(gpu_id):
            # Test basic operation
            test_array = cp.random.randn(100, 100)
            test_result = cp.dot(test_array, test_array)
            del test_array, test_result
            
            # Get GPU information
            gpu_name = cp.cuda.runtime.getDeviceProperties(gpu_id)['name'].decode()
            gpu_memory = cp.cuda.Device(gpu_id).mem_info[1] / (1024**3)  # Total memory in GB
            cuda_version = cp.cuda.runtime.driverGetVersion()
            
            hw_config.update({
                'use_gpu': True,
                'gpu_available': True,
                'gpu_provider': 'cupy',
                'gpu_device': f"cuda:{gpu_id}",
                'gpu_id': gpu_id,
                'gpu_name': gpu_name,
                'gpu_memory': gpu_memory,
                'cuda_version': cuda_version
            })
            
            logger.info(f"[GPU] ✅ CuPy CUDA initialized successfully")
            logger.info(f"[GPU] Device: {gpu_name}")
            logger.info(f"[GPU] Memory: {gpu_memory:.2f} GB")
            
            return hw_config
            
    except ImportError:
        logger.warning("[GPU] CuPy not installed")
    except Exception as e:
        logger.error(f"[GPU] CuPy initialization failed: {e}")
    
    # GPU not available - force CPU mode
    logger.error("[GPU] ❌ No working GPU acceleration found")
    logger.info("[GPU] Falling back to CPU-only mode")
    
    # Important: Override user setting to prevent errors
    args.use_gpu = False
    hw_config['use_gpu'] = False
    
    return hw_config
def configure_hardware(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Enhanced hardware configuration that properly handles GPU initialization.
    Replace the existing configure_hardware function with this.
    """
    logger = logging.getLogger(__name__)
    
    # GPU Detection and Configuration
    gpu_config = detect_and_configure_gpu(args, logger)
    
    # CPU Configuration
    cpu_workers = getattr(args, 'cpu_workers', None)
    if cpu_workers is None:
        import os
        cpu_workers = os.cpu_count()
        logger.info(f"[CPU] Auto-detected {cpu_workers} CPU cores")
    else:
        logger.info(f"[CPU] Using {cpu_workers} CPU worker threads")
    
    # Combine configurations
    hw_config = {
        **gpu_config,
        'cpu_workers': cpu_workers,
        'workload_balance': getattr(args, 'workload_balance', None)
    }
    hw_config['gpu_precision'] = getattr(args, 'gpu_precision', None)
    
    # Set workload balance if GPU is available
    if hw_config['use_gpu'] and hw_config['workload_balance'] is None:
        hw_config['workload_balance'] = 0.8  # Favor GPU
        logger.info(f"[GPU] Auto-set workload balance to {hw_config['workload_balance']} (GPU-favored)")
    
    # CPU Affinity
    if getattr(args, 'cpu_affinity', False):
        try:
            import psutil
            process = psutil.Process()
            cpu_count = psutil.cpu_count(logical=True)
            process.cpu_affinity(list(range(cpu_count)))
            logger.info(f"[CPU] CPU affinity set to {cpu_count} cores")
            hw_config['cpu_affinity'] = list(range(cpu_count))
        except Exception as e:
            logger.warning(f"[CPU] Failed to set CPU affinity: {e}")
    
    # Print final configuration
    logger.info("\n" + "="*60)
    logger.info(" HARDWARE CONFIGURATION SUMMARY")
    logger.info("="*60)
    logger.info(f" GPU Acceleration: {'✅ ENABLED' if hw_config['use_gpu'] else '❌ DISABLED'}")
    
    if hw_config['use_gpu']:
        logger.info(f" GPU Provider: {hw_config['gpu_provider'].upper()}")
        logger.info(f" GPU Device: {hw_config.get('gpu_name', 'Unknown')}")
        logger.info(f" GPU Memory: {hw_config.get('gpu_memory', 0):.2f} GB")
        logger.info(f" CUDA Version: {hw_config.get('cuda_version', 'Unknown')}")
        if 'workload_balance' in hw_config:
            logger.info(f" GPU/CPU Balance: {hw_config['workload_balance']:.1f} (higher = more GPU)")
    
    logger.info(f" CPU Workers: {hw_config['cpu_workers']}")
    logger.info(f" CPU Affinity: {'Enabled' if getattr(args, 'cpu_affinity', False) else 'Disabled'}")
    logger.info("="*60 + "\n")
    #print_hardware_summary(hw_config)
    return hw_config


def setup_hardware_acceleration(hw_config: Dict[str, Any]):
    """
    Set up hardware acceleration based on configuration.
    
    Args:
        hw_config: Hardware configuration dictionary
        
    Returns:
        HybridDockingManager instance
    """
    try:
        from .hybrid_manager import HybridDockingManager
        
        # Create hybrid manager with specified configuration
        manager = HybridDockingManager(
            use_gpu=hw_config.get('use_gpu', False),
            n_cpu_workers=hw_config.get('cpu_workers', None),
            gpu_device_id=hw_config.get('gpu_id', 0),
            workload_balance=hw_config.get('workload_balance', 0.8)
        )
        
        return manager
    except ImportError as e:
        print(f"[WARNING] HybridDockingManager not available: {e}")
        print("[INFO] Using fallback CPU-only manager")
        
        # Create a simple fallback manager
        class FallbackManager:
            def __init__(self):
                self.use_gpu = False
                self.n_cpu_workers = hw_config.get('cpu_workers', os.cpu_count())
            
            def cleanup(self):
                pass
            
            def run_ensemble_docking(self, **kwargs):
                # Simple ensemble implementation
                results = []
                protein = kwargs.get('protein')
                ligand = kwargs.get('ligand')
                n_runs = kwargs.get('n_runs', 1)
                algorithm_type = kwargs.get('algorithm_type', 'genetic')
                
                for i in range(n_runs):
                    # Create algorithm instance
                    algorithm = create_optimized_search_algorithm(
                        self, algorithm_type, None, **kwargs
                    )
                    # Run search
                    run_results = algorithm.search(protein, ligand)
                    results.extend(run_results)
                
                return results
        
        return FallbackManager()

def create_hardware_manager_from_args(args):
    """Create HybridDockingManager from command line arguments."""
    from .hybrid_manager import HybridDockingManager
    
    return HybridDockingManager(
        use_gpu=getattr(args, 'use_gpu', False),
        n_cpu_workers=getattr(args, 'cpu_workers', None),
        gpu_device_id=getattr(args, 'gpu_id', 0),
        workload_balance=getattr(args, 'workload_balance', 0.8)
    )

# ============================================================================
# Scoring Function Creation
# ============================================================================
def create_optimized_scoring_function(args, hw_config):
    """
    Enhanced scoring function creation that respects GPU configuration.
    """
    logger = logging.getLogger(__name__)
    
    # Determine if GPU should be used based on hw_config, not just args
    if isinstance(hw_config, dict):
        use_gpu = hw_config.get('use_gpu', False) and hw_config.get('gpu_available', False)
    else:
        use_gpu = getattr(hw_config, 'use_gpu', False) and getattr(hw_config, 'gpu_available', False)
    
    # Override args.use_gpu if hardware config says no GPU
    if hasattr(args, 'use_gpu') and args.use_gpu and not use_gpu:
        logger.warning("[SCORING] GPU requested but not available - using CPU scoring function")
        args.use_gpu = False
    
    try:
        from .scoring_factory import create_scoring_function
        
        scoring_function = create_scoring_function(
            use_gpu=use_gpu,
            physics_based=getattr(args, 'physics_based', False),
            enhanced=getattr(args, 'enhanced_scoring', True),
            tethered=getattr(args, 'tethered_scoring', False),
            reference_ligand=getattr(args, 'reference_ligand', None),
            weights=getattr(args, 'scoring_weights', None),
            # device=hw_config.get('gpu_device', 'cpu'),
            # precision=getattr(args, 'gpu_precision', 'float32'),
            verbose=getattr(args, 'verbose', False)
        )
        
        scoring_type = "GPU-accelerated" if use_gpu else "CPU-based"
        logger.info(f"[SCORING] Created {scoring_type} scoring function: {type(scoring_function).__name__}")
        
        return scoring_function
        
    except Exception as e:
        logger.error(f"[SCORING] Error creating scoring function: {e}")
        logger.info("[SCORING] Falling back to default CPU scoring function")
        
        # Fallback to basic scoring function
        from .scoring_factory import create_scoring_function
        return create_scoring_function(use_gpu=False)

# ============================================================================
# Search Algorithm Creation
# ============================================================================

def create_optimized_search_algorithm(manager, algorithm_type, scoring_function, hw_config, **kwargs):
    """
    Enhanced search algorithm creation that properly uses GPU configuration.
    """
    logger = logging.getLogger(__name__)
    
    # Extract parameters
    hw_config = kwargs.pop('hw_config', {})
    grid_spacing = kwargs.pop('grid_spacing', 0.375)
    grid_radius = kwargs.pop('grid_radius', 10.0)
    grid_center = kwargs.pop('grid_center', None)
    output_dir = kwargs.pop('output_dir', None)

    ## Get GPU usage from hardware config
    use_gpu = hw_config.get('use_gpu', False) and hw_config.get('gpu_available', False)
    gpu_device = hw_config.get('gpu_device', None)
    gpu_provider = hw_config.get('gpu_provider', None)
    
    # Get CPU workers count
    cpu_workers = hw_config.get('cpu_workers', None)
    if cpu_workers is None:
        import os
        cpu_workers = os.cpu_count() or 1
    
    # Log hardware usage info
    if use_gpu:
        gpu_name = hw_config.get('gpu_name', 'Unknown GPU')
        logger.info(f"[SEARCH] Using GPU acceleration: {gpu_name}")
        logger.info(f"[SEARCH] Algorithm: {algorithm_type}")
    else:
        logger.info(f"[SEARCH] Using CPU acceleration with {cpu_workers} workers")
        logger.info(f"[SEARCH] Algorithm: {algorithm_type}")
    
        # Create algorithm with proper parameters
    if algorithm_type == 'genetic':
        return ParallelGeneticAlgorithm(
            scoring_function=scoring_function,
            grid_spacing=grid_spacing,
            grid_radius=grid_radius,
            grid_center=grid_center,
            output_dir=output_dir,
            max_iterations=kwargs.get('max_iterations', 10),
            population_size=kwargs.get('population_size', 50),
            mutation_rate=kwargs.get('mutation_rate', 0.3),
            n_processes=cpu_workers,
            # GPU parameters
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            gpu_provider=gpu_provider
        )

    elif algorithm_type == 'random':
        return ParallelRandomSearch(
            scoring_function=scoring_function,
            grid_spacing=grid_spacing,
            grid_radius=grid_radius,
            grid_center=grid_center,
            output_dir=output_dir,
            max_iterations=kwargs.get('max_iterations', 100),
            n_processes=cpu_workers,
            # GPU parameters
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            gpu_provider=gpu_provider
        )

    elif algorithm_type in ['hybrid', 'flexible']:
        # Configure parameters for new algorithms
        search_params = {
            'scoring_function': scoring_function,
            'max_iterations': kwargs.get('max_iterations', 100),
            'output_dir': output_dir,
            'grid_spacing': grid_spacing,
            'grid_radius': grid_radius,
            'grid_center': grid_center,
            'n_processes': cpu_workers,
            # GPU parameters
            'use_gpu': use_gpu,
            'gpu_device': gpu_device,
            'gpu_provider': gpu_provider
        }
        
        # Add algorithm-specific parameters
        if algorithm_type in ['hybrid', 'flexible', 'genetic']:
            search_params.update({
                'population_size': kwargs.get('population_size', 50),
                'mutation_rate': kwargs.get('mutation_rate', 0.3),
                'crossover_rate': kwargs.get('crossover_rate', 0.8),
            })
            
        if algorithm_type in ['hybrid', 'flexible']:
            search_params.update({
                'temperature_start': kwargs.get('high_temp', 5.0),
                'temperature_end': kwargs.get('target_temp', 0.1),
                'cooling_factor': kwargs.get('cooling_factor', 0.95),
                'local_opt_frequency': 5,
            })
            
        if algorithm_type == 'flexible':
            search_params.update({
                'max_torsions': kwargs.get('max_torsions', 10),
                'torsion_step': 15.0,
            })
            
        # Create the algorithm using the factory function
        if algorithm_type == 'hybrid':
            return HybridSearch(**search_params)
        elif algorithm_type == 'flexible':
            return FlexibleLigandSearch(**search_params)
    
    else:
        logger.warning(f"[SEARCH] Unknown algorithm type: {algorithm_type}")
        logger.info("[SEARCH] Falling back to genetic algorithm")
        return create_optimized_search_algorithm(
            manager, 'genetic', scoring_function, hw_config,
            grid_spacing=grid_spacing, grid_radius=grid_radius,
            grid_center=grid_center, output_dir=output_dir, **kwargs
        )
def create_consistent_scoring_function(args):
    """
    Create a scoring function that provides consistent, high-quality results
    regardless of hardware. Focus on CPU implementations with GPU-quality accuracy.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Consistent, high-quality scoring function
    """
    try:
        # Always prefer Enhanced scoring for best results
        if getattr(args, 'enhanced_scoring', False) or getattr(args, 'physics_based', False):
            try:
                from .unified_scoring import EnhancedScoringFunction
                print("[INFO] Using EnhancedScoringFunction for high-quality scoring")
                return EnhancedScoringFunction()
            except ImportError:
                print("[WARNING] EnhancedScoringFunction not available")
        
        # Try GPUScoringFunction but with CPU implementations
        try:
            from .unified_scoring import GPUScoringFunction
            
            # Configure for CPU use with GPU-quality algorithms
            scoring_function = GPUScoringFunction(device='cpu', precision='float32')
            print("[INFO] Using GPUScoringFunction in CPU mode for consistent results")
            return scoring_function
            
        except ImportError:
            print("[WARNING] GPUScoringFunction not available")
        
        # Fallback to CompositeScoringFunction
        try:
            from .unified_scoring import CompositeScoringFunction
            print("[INFO] Using CompositeScoringFunction as fallback")
            return CompositeScoringFunction()
        except ImportError:
            print("[WARNING] CompositeScoringFunction not available")
        
        # Ultimate fallback - basic scoring
        try:
            from .unified_scoring import ScoringFunction
            print("[INFO] Using basic ScoringFunction as ultimate fallback")
            return ScoringFunction()
        except ImportError:
            raise RuntimeError("No scoring function available!")
    
    except Exception as e:
        print(f"[ERROR] Scoring function creation failed: {e}")
        raise


def optimize_for_cpu_users():
    """
    Optimization strategy for CPU-only users (students/researchers).
    This ensures they get the best possible experience and results.
    """
    optimization_tips = {
        'hardware': {
            'cpu_workers': 'auto',  # Use all available cores
            'memory_efficient': True,
            'batch_processing': True,
        },
        'algorithms': {
            'preferred': ['genetic', 'hybrid', 'random'],
            'avoid': ['pandadock'],  # Too resource intensive
            'default_params': {
                'population_size': 50,  # Good balance of quality vs speed
                'max_iterations': 100,  # Sufficient for most cases
                'local_opt': True,  # Always enable for better results
            }
        },
        'scoring': {
            'enhanced_scoring': True,  # Always use enhanced scoring
            'physics_based': False,    # Too slow for CPU-only users
            'batch_scoring': True,     # More efficient
        }
    }
    
    return optimization_tips


def create_user_friendly_defaults(args):
    """
    Set user-friendly defaults for students and researchers.
    Optimize for the best results with reasonable computational cost.
    """
    # Auto-optimize for CPU users
    if not getattr(args, 'use_gpu', False):
        print("[INFO] 🎓 Optimizing for CPU-based research workflow")
        
        # Enable enhanced scoring by default
        if not hasattr(args, 'enhanced_scoring') or not args.enhanced_scoring:
            args.enhanced_scoring = True
            print("[INFO] ✅ Enabled enhanced scoring for better results")
        
        # Disable physics-based scoring (too slow for CPU)
        if getattr(args, 'physics_based', False):
            args.physics_based = False
            print("[INFO] ⚡ Disabled physics-based scoring for faster CPU execution")
        
        # Enable local optimization by default
        if not getattr(args, 'local_opt', False):
            args.local_opt = True
            print("[INFO] 🎯 Enabled local optimization for improved poses")
        
        # Set reasonable defaults
        if not hasattr(args, 'population_size') or args.population_size < 50:
            args.population_size = 50
            print("[INFO] 👥 Set population size to 50 for good quality/speed balance")
        
        # Ensure sufficient iterations
        if not hasattr(args, 'iterations') or args.iterations < 50:
            args.iterations = 100
            print("[INFO] 🔄 Set iterations to 100 for thorough search")
        
        # Use all available CPU cores
        if not hasattr(args, 'cpu_workers') or args.cpu_workers is None:
            import os
            args.cpu_workers = os.cpu_count()
            print(f"[INFO] 💻 Using all {args.cpu_workers} CPU cores for parallel processing")
    
    return args


# Integration strategy for main.py
def setup_cpu_optimized_docking():
    """
    Setup function to optimize PandaDock for CPU-based users.
    This should be called early in main.py
    """
    print("\n" + "="*60)
    print("🎓 PandaDock: Optimized for Students & Researchers")
    print("="*60)
    print("💻 CPU-Focused | 🚀 Parallel Processing | 📊 GPU-Quality Results")
    print("="*60 + "\n")
    
    # Performance tips for users
    tips = [
        "💡 Use --enhanced-scoring for best results",
        "⚡ Use --local-opt for pose refinement", 
        "🔄 Increase --iterations for thorough search",
        "👥 Larger --population-size improves diversity",
        "🎯 Use --auto-algorithm for smart selection"
    ]
    
    print("📋 Tips for Better Results:")
    for tip in tips:
        print(f"   {tip}")
    print("")


# Modified algorithm mapping for CPU focus
ALGORITHM_MAPPING = {
    'genetic': 'ParallelGeneticAlgorithm',      # Most robust, well-tested
    'random': 'ParallelRandomSearch',           # Fast, good for initial exploration  
    'hybrid': 'HybridSearch',                   # Best of both worlds
    'flexible': 'FlexibleLigandSearch',         # For flexible ligand docking
    'monte-carlo': 'HybridSearch',              # MC behavior via HybridSearch
    'pandadock': 'HybridSearch',                # PANDADOCK-like via HybridSearch
}

RECOMMENDED_ALGORITHMS = ['genetic', 'hybrid', 'random']
ADVANCED_ALGORITHMS = ['flexible', 'monte-carlo']
LEGACY_ALGORITHMS = ['pandadock']  # Redirect to modern equivalents

# ============================================================================
# Algorithm Configuration Helpers
# ============================================================================

def get_scoring_type_from_args(args: argparse.Namespace) -> str:
    """
    Determine scoring type based on command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Scoring type string
    """
    if getattr(args, 'physics_based', False):
        return 'physics'
    elif getattr(args, 'enhanced_scoring', False):
        return 'enhanced'
    else:
        return 'standard'


def get_algorithm_type_from_args(args: argparse.Namespace) -> str:
    """
    Determine algorithm type based on command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Algorithm type string
    """
    if hasattr(args, 'flexible') and args.flexible:
        return 'flexible'
    elif hasattr(args, 'hybrid') and args.hybrid:
        return 'hybrid'
    elif hasattr(args, 'monte_carlo') and args.monte_carlo:
        return 'monte-carlo'
    else:
        return getattr(args, 'algorithm', 'genetic')


def get_algorithm_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Get algorithm keyword arguments based on command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary of algorithm-specific parameters
    """
    algorithm_type = get_algorithm_type_from_args(args)
    algorithm_kwargs = {}
    
    # Common parameters
    if hasattr(args, 'iterations'):
        algorithm_kwargs['max_iterations'] = args.iterations
    
    # Algorithm-specific parameters
    if algorithm_type == 'genetic':
        algorithm_kwargs.update({
            'population_size': getattr(args, 'population_size', 50),
            'mutation_rate': getattr(args, 'mutation_rate', 0.2),
            'perform_local_opt': getattr(args, 'local_opt', False),
        })
    
    elif algorithm_type == 'monte-carlo':
        algorithm_kwargs.update({
            'n_steps': getattr(args, 'mc_steps', 1000),
            'temperature': getattr(args, 'temperature', 300.0),
            'cooling_factor': getattr(args, 'cooling_factor', 0.95),
        })
    
    elif algorithm_type == 'pandadock':
        algorithm_kwargs.update({
            'high_temp': getattr(args, 'high_temp', 1000.0),
            'target_temp': getattr(args, 'target_temp', 300.0),
            'num_conformers': getattr(args, 'num_conformers', 10),
            'num_orientations': getattr(args, 'num_orientations', 10),
            'md_steps': getattr(args, 'md_steps', 1000),
            'minimize_steps': getattr(args, 'minimize_steps', 200),
            'use_grid': getattr(args, 'use_grid', False),
        })
    
    elif algorithm_type in ['hybrid', 'flexible']:
        algorithm_kwargs.update({
            'population_size': getattr(args, 'population_size', 30),
            'mutation_rate': getattr(args, 'mutation_rate', 0.3),
            'crossover_rate': getattr(args, 'crossover_rate', 0.7),
            'temperature_start': getattr(args, 'hybrid_temperature_start', 5.0),
            'temperature_end': getattr(args, 'hybrid_temperature_end', 0.1),
            'cooling_factor': getattr(args, 'hybrid_cooling_factor', 0.95),
            'local_opt_frequency': 5,
        })
        
        if algorithm_type == 'flexible':
            algorithm_kwargs.update({
                'max_torsions': getattr(args, 'max_torsions', 10),
                'torsion_step': 15.0,
            })
    
    # Add hardware-specific parameters
    algorithm_kwargs.update({
        'use_gpu': getattr(args, 'use_gpu', False),
        'cpu_workers': getattr(args, 'cpu_workers', None),
    })
    
    return algorithm_kwargs


# ============================================================================
# Validation and Testing Utilities
# ============================================================================

def validate_hardware_configuration(hw_config: Dict[str, Any]) -> bool:
    """
    Validate that the hardware configuration is valid and usable.
    
    Args:
        hw_config: Hardware configuration dictionary
        
    Returns:
        True if configuration is valid
    """
    try:
        # Check GPU configuration if requested
        if hw_config.get('use_gpu', False):
            if not hw_config.get('gpu_capabilities', {}).get('available', False):
                print("[ERROR] GPU requested but not available")
                return False
            
            gpu_id = hw_config.get('gpu_id', 0)
            device_count = hw_config.get('gpu_capabilities', {}).get('device_count', 0)
            if gpu_id >= device_count:
                print(f"[ERROR] GPU ID {gpu_id} not available (only {device_count} devices found)")
                return False
        
        # Check CPU configuration
        cpu_workers = hw_config.get('cpu_workers', 1)
        max_workers = hw_config.get('cpu_info', {}).get('logical_cores', os.cpu_count())
        if cpu_workers > max_workers:
            print(f"[WARNING] Requested {cpu_workers} CPU workers but only {max_workers} cores available")
            hw_config['cpu_workers'] = max_workers
        
        # Check workload balance
        workload_balance = hw_config.get('workload_balance')
        if workload_balance is not None:
            if not (0.0 <= workload_balance <= 1.0):
                print(f"[ERROR] Workload balance must be between 0.0 and 1.0, got {workload_balance}")
                return False
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Hardware configuration validation failed: {e}")
        return False


def test_hardware_performance(hw_config: Dict[str, Any]) -> Dict[str, float]:
    """
    Run basic performance tests on the configured hardware.
    
    Args:
        hw_config: Hardware configuration dictionary
        
    Returns:
        Dictionary containing performance metrics
    """
    performance_metrics = {}
    
    try:
        import time
        import numpy as np
        
        # CPU performance test
        print("[INFO] Running CPU performance test...")
        start_time = time.time()
        
        # Simple matrix multiplication test
        size = 1000
        a = np.random.rand(size, size).astype(np.float32)
        b = np.random.rand(size, size).astype(np.float32)
        c = np.dot(a, b)
        
        cpu_time = time.time() - start_time
        performance_metrics['cpu_matmul_time'] = cpu_time
        performance_metrics['cpu_gflops'] = (2.0 * size**3) / (cpu_time * 1e9)
        
        print(f"[INFO] CPU test completed in {cpu_time:.3f}s ({performance_metrics['cpu_gflops']:.2f} GFLOPS)")
        
        # GPU performance test if available
        if hw_config.get('use_gpu', False):
            try:
                if hw_config.get('gpu_provider') == 'pytorch':
                    import torch
                    device = f"cuda:{hw_config.get('gpu_id', 0)}"
                    
                    print(f"[INFO] Running GPU performance test on {device}...")
                    start_time = time.time()
                    
                    # GPU matrix multiplication test
                    a_gpu = torch.rand(size, size, device=device, dtype=torch.float32)
                    b_gpu = torch.rand(size, size, device=device, dtype=torch.float32)
                    c_gpu = torch.matmul(a_gpu, b_gpu)
                    torch.cuda.synchronize()
                    
                    gpu_time = time.time() - start_time
                    performance_metrics['gpu_matmul_time'] = gpu_time
                    performance_metrics['gpu_gflops'] = (2.0 * size**3) / (gpu_time * 1e9)
                    performance_metrics['gpu_speedup'] = cpu_time / gpu_time
                    
                    print(f"[INFO] GPU test completed in {gpu_time:.3f}s ({performance_metrics['gpu_gflops']:.2f} GFLOPS)")
                    print(f"[INFO] GPU speedup: {performance_metrics['gpu_speedup']:.2f}x")
                
                elif hw_config.get('gpu_provider') == 'cupy':
                    import cupy as cp
                    
                    print("[INFO] Running GPU performance test with CuPy...")
                    start_time = time.time()
                    
                    # CuPy matrix multiplication test
                    a_gpu = cp.random.rand(size, size, dtype=cp.float32)
                    b_gpu = cp.random.rand(size, size, dtype=cp.float32)
                    c_gpu = cp.dot(a_gpu, b_gpu)
                    cp.cuda.Stream.null.synchronize()
                    
                    gpu_time = time.time() - start_time
                    performance_metrics['gpu_matmul_time'] = gpu_time
                    performance_metrics['gpu_gflops'] = (2.0 * size**3) / (gpu_time * 1e9)
                    performance_metrics['gpu_speedup'] = cpu_time / gpu_time
                    
                    print(f"[INFO] GPU test completed in {gpu_time:.3f}s ({performance_metrics['gpu_gflops']:.2f} GFLOPS)")
                    print(f"[INFO] GPU speedup: {performance_metrics['gpu_speedup']:.2f}x")
                    
            except Exception as e:
                print(f"[WARNING] GPU performance test failed: {e}")
        
    except Exception as e:
        print(f"[WARNING] Performance testing failed: {e}")
    
    return performance_metrics


# ============================================================================
# Memory Management Utilities
# ============================================================================

def estimate_memory_requirements(protein_atoms: int, ligand_atoms: int, 
                                population_size: int, use_gpu: bool = False) -> Dict[str, float]:
    """
    Estimate memory requirements for docking calculations.
    
    Args:
        protein_atoms: Number of protein atoms
        ligand_atoms: Number of ligand atoms
        population_size: Population size for genetic algorithm
        use_gpu: Whether GPU acceleration is used
        
    Returns:
        Dictionary containing memory estimates in MB
    """
    # Basic memory estimates (rough approximations)
    bytes_per_atom = 32  # Approximate memory per atom for coordinates and properties
    bytes_per_pose = ligand_atoms * bytes_per_atom
    bytes_per_population = population_size * bytes_per_pose
    
    # Protein memory (loaded once)
    protein_memory = protein_atoms * bytes_per_atom / 1024**2  # MB
    
    # Ligand population memory
    ligand_memory = bytes_per_population / 1024**2  # MB
    
    # Scoring matrix memory (protein-ligand interactions)
    interaction_memory = (protein_atoms * ligand_atoms * 8) / 1024**2  # MB (assuming float64)
    
    # Total CPU memory estimate
    cpu_memory = protein_memory + ligand_memory + interaction_memory
    
    # GPU memory estimate (if applicable)
    gpu_memory = 0.0
    if use_gpu:
        # GPU typically requires additional memory for CUDA context and libraries
        gpu_overhead = 500.0  # MB
        gpu_memory = cpu_memory + gpu_overhead
    
    return {
        'protein_memory_mb': protein_memory,
        'ligand_memory_mb': ligand_memory,
        'interaction_memory_mb': interaction_memory,
        'total_cpu_memory_mb': cpu_memory,
        'total_gpu_memory_mb': gpu_memory,
        'recommended_ram_gb': max(4.0, cpu_memory * 2 / 1024)  # 2x safety factor, minimum 4GB
    }


def check_memory_availability(memory_requirements: Dict[str, float], 
                            hw_config: Dict[str, Any]) -> bool:
    """
    Check if sufficient memory is available for the calculations.
    
    Args:
        memory_requirements: Memory requirements from estimate_memory_requirements
        hw_config: Hardware configuration
        
    Returns:
        True if sufficient memory is available
    """
    try:
        # Check system RAM
        available_ram_gb = hw_config.get('cpu_info', {}).get('memory_available', 0) / 1024**3
        required_ram_gb = memory_requirements['recommended_ram_gb']
        
        if available_ram_gb < required_ram_gb:
            print(f"[WARNING] Insufficient RAM: {available_ram_gb:.1f} GB available, "
                  f"{required_ram_gb:.1f} GB recommended")
            return False
        
        # Check GPU memory if applicable
        if hw_config.get('use_gpu', False):
            gpu_info = hw_config.get('gpu_info', {})
            available_gpu_gb = gpu_info.get('memory_total', 0) / 1024**3
            required_gpu_gb = memory_requirements['total_gpu_memory_mb'] / 1024
            
            if available_gpu_gb < required_gpu_gb:
                print(f"[WARNING] Insufficient GPU memory: {available_gpu_gb:.1f} GB available, "
                      f"{required_gpu_gb:.1f} GB required")
                return False
        
        print(f"[INFO] Memory check passed: {available_ram_gb:.1f} GB RAM available, "
              f"{required_ram_gb:.1f} GB required")
        
        return True
        
    except Exception as e:
        print(f"[WARNING] Memory check failed: {e}")
        return True  # Assume it's okay if we can't check


# ============================================================================
# Logging and Monitoring
# ============================================================================

def setup_performance_monitoring(hw_config: Dict[str, Any]) -> Optional[Any]:
    """
    Set up performance monitoring for hardware utilization.
    
    Args:
        hw_config: Hardware configuration
        
    Returns:
        Performance monitor object or None
    """
    try:
        import threading
        import time
        
        class PerformanceMonitor:
            def __init__(self, hw_config):
                self.hw_config = hw_config
                self.monitoring = False
                self.monitor_thread = None
                self.stats = {
                    'cpu_usage': [],
                    'memory_usage': [],
                    'gpu_usage': [],
                    'gpu_memory_usage': []
                }
            
            def start_monitoring(self):
                """Start performance monitoring in a separate thread."""
                self.monitoring = True
                self.monitor_thread = threading.Thread(target=self._monitor_loop)
                self.monitor_thread.daemon = True
                self.monitor_thread.start()
            
            def stop_monitoring(self):
                """Stop performance monitoring."""
                self.monitoring = False
                if self.monitor_thread:
                    self.monitor_thread.join()
            
            def _monitor_loop(self):
                """Main monitoring loop."""
                while self.monitoring:
                    try:
                        # Monitor CPU and memory
                        cpu_percent = psutil.cpu_percent(interval=None)
                        memory_percent = psutil.virtual_memory().percent
                        
                        self.stats['cpu_usage'].append(cpu_percent)
                        self.stats['memory_usage'].append(memory_percent)
                        
                        # Monitor GPU if available
                        if self.hw_config.get('use_gpu', False):
                            try:
                                if self.hw_config.get('gpu_provider') == 'pytorch':
                                    import torch
                                    gpu_id = self.hw_config.get('gpu_id', 0)
                                    gpu_memory_allocated = torch.cuda.memory_allocated(gpu_id)
                                    gpu_memory_cached = torch.cuda.memory_reserved(gpu_id)
                                    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
                                    
                                    gpu_usage = (gpu_memory_allocated / total_memory) * 100
                                    self.stats['gpu_usage'].append(gpu_usage)
                                    self.stats['gpu_memory_usage'].append(gpu_memory_allocated)
                                    
                            except Exception:
                                pass
                        
                        time.sleep(1.0)  # Monitor every second
                        
                    except Exception:
                        pass
            
            def get_summary(self):
                """Get monitoring summary."""
                summary = {}
                
                if self.stats['cpu_usage']:
                    summary['avg_cpu_usage'] = sum(self.stats['cpu_usage']) / len(self.stats['cpu_usage'])
                    summary['max_cpu_usage'] = max(self.stats['cpu_usage'])
                
                if self.stats['memory_usage']:
                    summary['avg_memory_usage'] = sum(self.stats['memory_usage']) / len(self.stats['memory_usage'])
                    summary['max_memory_usage'] = max(self.stats['memory_usage'])
                
                if self.stats['gpu_usage']:
                    summary['avg_gpu_usage'] = sum(self.stats['gpu_usage']) / len(self.stats['gpu_usage'])
                    summary['max_gpu_usage'] = max(self.stats['gpu_usage'])
                
                return summary
        
        return PerformanceMonitor(hw_config)
        
    except Exception as e:
        print(f"[WARNING] Could not set up performance monitoring: {e}")
        return None


# ============================================================================
# Export Functions
# ============================================================================

__all__ = [
    'add_hardware_options',
    'setup_hardware_acceleration',
    'create_optimized_scoring_function',
    'create_optimized_search_algorithm',
    'get_scoring_type_from_args',
    'get_algorithm_type_from_args',
    'get_algorithm_kwargs_from_args',
    'validate_hardware_configuration',
    'test_hardware_performance',
    'estimate_memory_requirements',
    'check_memory_availability',
    'setup_performance_monitoring',
    'configure_hardware'
]