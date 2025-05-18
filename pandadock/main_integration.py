"""
CLI integration module for PandaDock hardware acceleration.
This module extends the main.py command-line interface with GPU/CPU options.
"""

# Core CLI utilities
import argparse
import os

# Factory and algorithms
from .scoring_factory import create_scoring_function
from .search import GeneticAlgorithm, RandomSearch
from .parallel_search import ParallelGeneticAlgorithm, ParallelRandomSearch
from .pandadock import PANDADOCKAlgorithm

# Physics-based models
from .physics import (
    MMFFMinimization,
    MonteCarloSampling,
    PhysicsBasedScoring,
    GeneralizedBornSolvation
)

# Preparation and validation
from .preparation import prepare_protein, prepare_ligand
from .validation import validate_docking, calculate_ensemble_rmsd

# Batch and utilities
from .utils import save_docking_results, save_complex_to_pdb
from .utils import calculate_rmsd
import psutil

def add_hardware_options(parser):
    """
    Add hardware acceleration options to the argument parser.
    
    Parameters:
    -----------
    parser : argparse.ArgumentParser
        Argument parser to modify
    """
    # Create a hardware acceleration group
    hw_group = parser.add_argument_group('Hardware Acceleration')
    
    # GPU options
    hw_group.add_argument('--use-gpu', action='store_true',
                         help='Use GPU acceleration if available')
    hw_group.add_argument('--gpu-id', type=int, default=0,
                         help='GPU device ID to use (default: 0)')
    hw_group.add_argument('--gpu-precision', choices=['float32', 'float64'], default='float32',
                         help='Numerical precision for GPU calculations (default: float32)')
    
    # CPU options
    hw_group.add_argument('--cpu-workers', type=int, default=None,
                         help='Number of CPU workers for parallel processing (default: all cores)')
    hw_group.add_argument('--cpu-affinity', action='store_true',
                         help='Set CPU affinity for better performance')
    
    # Hybrid options
    hw_group.add_argument('--workload-balance', type=float, default=None,
                         help='GPU/CPU workload balance (0.0-1.0, higher values assign more work to GPU)')
    hw_group.add_argument('--auto-tune', action='store_true',
                         help='Automatically tune hardware parameters for best performance')


# def configure_hardware(args):
#     """
#     Configure hardware settings based on command-line arguments.
    
#     Parameters:
#     -----------
#     args : argparse.Namespace
#         Command-line arguments
    
#     Returns:
#     --------
#     dict
#         Hardware configuration dictionary
#     """
#     # Basic hardware configuration
#     hw_config = {
#         'use_gpu': args.use_gpu,
#         'gpu_id': args.gpu_id,
#         'gpu_precision': args.gpu_precision,
#         'cpu_workers': args.cpu_workers,
#     }
    
#     # Add optional workload balance if specified
#     if args.workload_balance is not None:
#         hw_config['workload_balance'] = max(0.0, min(1.0, args.workload_balance))
    
#     # Set CPU affinity if requested
#     if args.cpu_affinity:
#         try:
#             import psutil
#             process = psutil.Process()
#             # Get available CPU cores
#             cpu_count = psutil.cpu_count(logical=True)
#             # Set affinity to all cores
#             process.cpu_affinity(list(range(cpu_count)))
#             print(f"CPU affinity set to {cpu_count} cores")
#         except ImportError:
#             print("Warning: psutil module not available, CPU affinity not set")
#         except Exception as e:
#             print(f"Warning: Failed to set CPU affinity: {e}")
    
#     return hw_config

def configure_hardware(args):
    """
    Configure hardware settings based on command-line arguments.
    Detects and sets up GPU and CPU resources.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments with hardware options
    
    Returns:
    --------
    dict
        Hardware configuration dictionary
    """
    # Basic hardware configuration
    hw_config = {
        'use_gpu': args.use_gpu,
        'gpu_id': args.gpu_id,
        'gpu_precision': args.gpu_precision,
        'cpu_workers': args.cpu_workers,
    }
    
    # Add optional workload balance if specified
    if args.workload_balance is not None:
        hw_config['workload_balance'] = max(0.0, min(1.0, args.workload_balance))
    
    # Verify GPU availability when requested
    if args.use_gpu:
        gpu_available = False
        
        # Try PyTorch first
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = torch.cuda.get_device_name(args.gpu_id)
                print(f"[INFO] Using GPU (PyTorch): {gpu_info}")
                hw_config['gpu_info'] = gpu_info
                hw_config['gpu_provider'] = 'pytorch'
                gpu_available = True
                
                # Set device
                hw_config['device'] = f"cuda:{args.gpu_id}"
                
                # Additional GPU details
                print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(args.gpu_id).total_memory / 1024**3:.2f} GB")
                print(f"[INFO] CUDA Version: {torch.version.cuda}")
            else:
                print("[WARNING] PyTorch CUDA not available, checking CuPy...")
        except ImportError:
            print("[INFO] PyTorch not available, checking CuPy...")
        
        # Try CuPy if PyTorch fails
        if not gpu_available:
            try:
                import cupy as cp
                try:
                    # Set device
                    cp.cuda.Device(args.gpu_id).use()
                    gpu_info = cp.cuda.runtime.getDeviceProperties(args.gpu_id)['name'].decode()
                    print(f"[INFO] Using GPU (CuPy): {gpu_info}")
                    hw_config['gpu_info'] = gpu_info
                    hw_config['gpu_provider'] = 'cupy'
                    gpu_available = True
                    
                    # Additional GPU details
                    print(f"[INFO] GPU Memory: {cp.cuda.Device(args.gpu_id).mem_info[1] / 1024**3:.2f} GB")
                    print(f"[INFO] CUDA Version: {cp.cuda.runtime.driverGetVersion()}")
                except Exception as e:
                    print(f"[WARNING] CuPy CUDA error: {e}")
            except ImportError:
                print("[WARNING] Neither PyTorch nor CuPy available for GPU acceleration")
        
        # If GPU is not available, fallback to CPU
        if not gpu_available:
            print("[WARNING] GPU acceleration requested but no compatible GPU found")
            print("[INFO] Falling back to CPU-only mode")
            hw_config['use_gpu'] = False
    
    # Configure CPU settings
    if hw_config['cpu_workers'] is None:
        # Auto-detect number of CPU cores
        import os
        hw_config['cpu_workers'] = os.cpu_count()
        print(f"[INFO] Auto-detected {hw_config['cpu_workers']} CPU cores")
    else:
        print(f"[INFO] Using {hw_config['cpu_workers']} CPU worker threads")
    
    # Set CPU affinity if requested
    if args.cpu_affinity:
        try:
            import psutil
            process = psutil.Process()
            # Get available CPU cores
            cpu_count = psutil.cpu_count(logical=True)
            # Set affinity to all cores
            process.cpu_affinity(list(range(cpu_count)))
            print(f"[INFO] CPU affinity set to {cpu_count} cores")
            hw_config['cpu_affinity'] = list(range(cpu_count))
        except ImportError:
            print("[WARNING] psutil module not available, CPU affinity not set")
        except Exception as e:
            print(f"[WARNING] Failed to set CPU affinity: {e}")
    
    # Auto-tune hardware parameters if requested
    if args.auto_tune:
        print("[INFO] Auto-tuning hardware parameters...")
        
        if hw_config['use_gpu']:
            # For GPU, allocate more work to GPU
            hw_config['workload_balance'] = 0.8
            print(f"[INFO] Auto-tuned workload balance: {hw_config['workload_balance']} (GPU-focused)")
        else:
            # For CPU-only, optimize thread count based on available cores
            import os
            cores = os.cpu_count()
            if cores > 8:
                # Leave some cores for system processes
                hw_config['cpu_workers'] = max(1, cores - 2)
                print(f"[INFO] Auto-tuned CPU workers: {hw_config['cpu_workers']} (from {cores} cores)")
            else:
                # Use all cores on smaller systems
                hw_config['cpu_workers'] = cores
                print(f"[INFO] Auto-tuned CPU workers: {hw_config['cpu_workers']} (all cores)")
    
    # Print hardware configuration summary
    print("\n[INFO] Hardware Configuration Summary:")
    print(f"  - GPU Acceleration: {'Enabled' if hw_config['use_gpu'] else 'Disabled'}")
    if hw_config['use_gpu']:
        print(f"  - GPU Device ID: {hw_config['gpu_id']}")
        print(f"  - GPU Precision: {hw_config['gpu_precision']}")
        if 'gpu_info' in hw_config:
            print(f"  - GPU Info: {hw_config['gpu_info']}")
        if 'gpu_provider' in hw_config:
            print(f"  - GPU Provider: {hw_config['gpu_provider']}")
    print(f"  - CPU Workers: {hw_config['cpu_workers']}")
    if 'workload_balance' in hw_config:
        print(f"  - Workload Balance: {hw_config['workload_balance']} (higher = more GPU work)")
    print(f"  - CPU Affinity: {'Enabled' if args.cpu_affinity else 'Disabled'}")
    print("") 
    
    return hw_config


def setup_hardware_acceleration(hw_config):
    """
    Set up hardware acceleration based on configuration.
    
    Parameters:
    -----------
    hw_config : dict
        Hardware configuration dictionary
    
    Returns:
    --------
    object
        HybridDockingManager instance
    """
    from .hybrid_manager import HybridDockingManager
    
    # Create hybrid manager with specified configuration
    manager = HybridDockingManager(
        use_gpu=hw_config.get('use_gpu', False),
        n_cpu_workers=hw_config.get('cpu_workers', None),
        gpu_device_id=hw_config.get('gpu_id', 0),
        workload_balance=hw_config.get('workload_balance', 0.8)
    )
    
    return manager


# def create_optimized_scoring_function(args):
#     """
#     Create an optimized scoring function based on user arguments.
#     Automatically selects CPU/GPU/physics-based implementation.
#     """
#     try:
#         scoring_function = create_scoring_function(
#             use_gpu=getattr(args, 'use_gpu', False),
#             physics_based=getattr(args, 'physics_based', False),
#             enhanced=getattr(args, 'enhanced_scoring', True),
#             tethered=getattr(args, 'tethered_scoring', False),
#             reference_ligand=getattr(args, 'reference_ligand', None),
#             weights=getattr(args, 'scoring_weights', None),
#             device=getattr(args, 'gpu_device', 'cuda'),
#             precision=getattr(args, 'gpu_precision', 'float32'),
#             verbose=getattr(args, 'verbose', False)
#         )
#         print(f"[DEBUG] Created scoring function: {type(scoring_function).__name__}")
#         return scoring_function

#     except Exception as e:
#         print("[ERROR] Could not initialize physics-based scoring:", str(e))
#         print("Warning: Physics-based modules not available. Some features will be disabled.")
#         return create_scoring_function()
    
def create_optimized_scoring_function(args):
    """
    Create an optimized scoring function based on user arguments.
    Automatically selects CPU/GPU/physics-based implementation.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments with scoring options
        
    Returns:
    --------
    ScoringFunction
        Optimized scoring function instance
    """
    try:
        # Check if GPU is available when --use-gpu is specified
        if getattr(args, 'use_gpu', False):
            # Try to import PyTorch and check CUDA availability
            try:
                import torch
                if not torch.cuda.is_available():
                    print("[WARNING] GPU requested but CUDA is not available through PyTorch")
                    print("[INFO] Checking for CuPy...")
                    
                    # Try CuPy as fallback
                    try:
                        import cupy as cp
                        try:
                            # Test CuPy CUDA
                            gpu_info = cp.cuda.runtime.getDeviceProperties(0)
                            print(f"[INFO] Using GPU via CuPy: {gpu_info['name'].decode()}")
                        except:
                            print("[WARNING] GPU requested but CUDA is not available through CuPy")
                            print("[INFO] Falling back to CPU implementation")
                            args.use_gpu = False
                    except ImportError:
                        print("[WARNING] Neither PyTorch nor CuPy is available for GPU acceleration")
                        print("[INFO] Falling back to CPU implementation")
                        args.use_gpu = False
                else:
                    # PyTorch with CUDA is available
                    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
            except ImportError:
                print("[WARNING] PyTorch not available for GPU detection")
                print("[INFO] Checking for CuPy...")
                
                # Try CuPy for GPU detection
                try:
                    import cupy as cp
                    try:
                        # Test CuPy CUDA
                        gpu_info = cp.cuda.runtime.getDeviceProperties(0)
                        print(f"[INFO] Using GPU via CuPy: {gpu_info['name'].decode()}")
                    except:
                        print("[WARNING] GPU requested but CUDA is not available through CuPy")
                        print("[INFO] Falling back to CPU implementation")
                        args.use_gpu = False
                except ImportError:
                    print("[WARNING] Neither PyTorch nor CuPy is available for GPU acceleration")
                    print("[INFO] Falling back to CPU implementation")
                    args.use_gpu = False
        
        # Create the scoring function
        from .scoring_factory import create_scoring_function
        scoring_function = create_scoring_function(
            use_gpu=getattr(args, 'use_gpu', False),
            physics_based=getattr(args, 'physics_based', False),
            enhanced=getattr(args, 'enhanced_scoring', True),
            tethered=getattr(args, 'tethered_scoring', False),
            reference_ligand=getattr(args, 'reference_ligand', None),
            weights=getattr(args, 'scoring_weights', None),
            device=getattr(args, 'gpu_device', 'cuda'),
            precision=getattr(args, 'gpu_precision', 'float32'),
            verbose=getattr(args, 'verbose', False)
        )
        
        print(f"[INFO] Created scoring function: {type(scoring_function).__name__}")
        return scoring_function
    except Exception as e:
        print(f"[ERROR] Could not initialize scoring function: {str(e)}")
        print("[WARNING] Falling back to default scoring function")
        from .scoring_factory import create_scoring_function
        return create_scoring_function()


def create_optimized_search_algorithm(manager, algorithm_type, scoring_function, **kwargs):
    """
    Create an optimized search algorithm based on available hardware.
    
    Parameters:
    -----------
    manager : HybridDockingManager
        Manager for hybrid CPU/GPU acceleration
    algorithm_type : str
        Type of search algorithm ('genetic', 'random', 'monte-carlo', 'pandadock')
    scoring_function : ScoringFunction
        Scoring function for pose evaluation
    **kwargs : dict
        Additional algorithm-specific parameters
        
    Returns:
    --------
    object
        Search algorithm instance
    """
    # Extract grid-related settings from kwargs
    grid_spacing = kwargs.pop('grid_spacing', 0.375)
    grid_radius = kwargs.pop('grid_radius', 10.0)
    grid_center = kwargs.pop('grid_center', None)
    output_dir = kwargs.pop('output_dir', None)

    # EXPLICITLY GET use_gpu from kwargs or manager
    use_gpu = kwargs.get('use_gpu', getattr(manager, 'use_gpu', False))

    # CRITICAL: Add this print statement for debugging
    print(f"DEBUG: create_optimized_search_algorithm received use_gpu={use_gpu}")

    # AUTO-SELECT PARALLEL IMPLEMENTATIONS
    cpu_workers = kwargs.get('cpu_workers', None)
    # use_gpu = kwargs.get('use_gpu', False)
    
    # Determine parallelization strategy
    use_parallel = (cpu_workers is not None and cpu_workers > 1) or use_gpu
    
    if cpu_workers is None:
        cpu_workers = manager.n_cpu_workers
    if cpu_workers is None:
        cpu_workers = os.cpu_count() or 1
    if cpu_workers > 1:
        use_parallel = True
    else:
        use_parallel = False
        
    # Log hardware usage info
    if use_gpu:
        print(f"[INFO] Using GPU acceleration for '{algorithm_type}'")
    else:
        print(f"[INFO] Using CPU acceleration for '{algorithm_type}'")
        
    if use_parallel:
        print(f"[INFO] Using parallel implementation for '{algorithm_type}' with {cpu_workers} CPU workers")
    else:
        print(f"[INFO] Using serial implementation for '{algorithm_type}'")

    # Select appropriate algorithm implementation based on type and hardware
    if algorithm_type == 'genetic':
        if use_gpu:  # GPU-accelerated genetic algorithm
            try:
                from .gpu_search import ParallelGeneticAlgorithm
                return ParallelGeneticAlgorithm(
                    scoring_function=scoring_function,
                    grid_spacing=grid_spacing,
                    grid_radius=grid_radius,
                    grid_center=grid_center,
                    output_dir=output_dir,
                    **kwargs
                )
            except ImportError as e:
                print(f"[WARNING] GPU-accelerated genetic algorithm not available: {e}")
                print("[INFO] Falling back to CPU parallel implementation")
                use_gpu = False
                use_parallel = True
                
        if use_parallel:  # CPU parallel genetic algorithm
            from .parallel_search import ParallelGeneticAlgorithm
            # Remove irrelevant kwargs
            for key in ['n_steps', 'temperature', 'cooling_factor', 'high_temp', 'target_temp',
                        'num_conformers', 'num_orientations', 'md_steps', 'minimize_steps',
                        'use_grid', 'use_monte_carlo']:
                kwargs.pop(key, None)
            return ParallelGeneticAlgorithm(
                scoring_function=scoring_function,
                grid_spacing=grid_spacing,
                grid_radius=grid_radius,
                grid_center=grid_center,
                output_dir=output_dir,
                **kwargs
            )
        else:  # Serial genetic algorithm
            from .search import GeneticAlgorithm
            return GeneticAlgorithm(
                scoring_function=scoring_function,
                grid_spacing=grid_spacing,
                grid_radius=grid_radius,
                grid_center=grid_center,
                output_dir=output_dir,
                **kwargs
            )

    elif algorithm_type == 'random':
        if use_gpu:  # GPU-accelerated random search (not implemented yet)
            print("[WARNING] GPU acceleration not implemented for random search algorithm")
            print("[INFO] Falling back to CPU parallel implementation")
            use_gpu = False
            use_parallel = True
            
        if use_parallel:  # CPU parallel random search
            from .parallel_search import ParallelRandomSearch
            return ParallelRandomSearch(
                scoring_function=scoring_function,
                grid_spacing=grid_spacing,
                grid_radius=grid_radius,
                grid_center=grid_center,
                output_dir=output_dir,
                **kwargs
            )
        else:  # Serial random search
            from .search import RandomSearch
            return RandomSearch(
                scoring_function=scoring_function,
                grid_spacing=grid_spacing,
                grid_radius=grid_radius,
                grid_center=grid_center,
                output_dir=output_dir,
                **kwargs
            )
            
    # Monte Carlo sampling
    elif algorithm_type == 'monte-carlo':
        try:
            from .physics import MonteCarloSampling
            # Remove unsupported kwargs
            kwargs.pop('grid_radius', None)
            kwargs.pop('grid_spacing', None)
            kwargs.pop('grid_center', None)
            kwargs.pop('max_iterations', None)
            kwargs.pop('workload_balance', None)            
            kwargs.pop('num_processes', None)
            kwargs.pop('output_dir', None)
            kwargs.pop('mc_steps', None)
            kwargs.pop('temperature', None)
            
            return MonteCarloSampling(
                scoring_function=scoring_function,
                grid_spacing=grid_spacing,
                grid_radius=grid_radius,
                grid_center=grid_center,
                output_dir=output_dir,
                **kwargs
            )
        except ImportError as e:
            print(f"[WARNING] Monte Carlo sampling not available: {e}")
            print("[INFO] Falling back to genetic algorithm")
            return create_optimized_search_algorithm(
                manager,
                'genetic',
                scoring_function,
                grid_spacing=grid_spacing,
                grid_radius=grid_radius,
                grid_center=grid_center,
                output_dir=output_dir,
                **kwargs
            )
            
    # PANDADOCK algorithm
    elif algorithm_type == 'pandadock':
        try:
            from .pandadock import PANDADOCKAlgorithm
            
            # Remove unsupported kwargs
            kwargs.pop('grid_radius', None)
            kwargs.pop('grid_spacing', None)
            kwargs.pop('grid_center', None)
            kwargs.pop('output_dir', None)

            return PANDADOCKAlgorithm(
                scoring_function=scoring_function,
                output_dir=output_dir,
                **kwargs
            )
        except ImportError as e:
            print(f"[WARNING] PANDADOCK algorithm not available: {e}")
            print("[INFO] Falling back to genetic algorithm")
            return create_optimized_search_algorithm(
                manager,
                'genetic',
                scoring_function,
                grid_spacing=grid_spacing,
                grid_radius=grid_radius,
                grid_center=grid_center,
                output_dir=output_dir,
                **kwargs
            )
    
    # Unknown algorithm type
    else:
        print(f"[WARNING] Unknown algorithm type: {algorithm_type}")
        print("[INFO] Falling back to genetic algorithm")
        return create_optimized_search_algorithm(
            manager,
            'genetic',
            scoring_function,
            grid_spacing=grid_spacing,
            grid_radius=grid_radius,
            grid_center=grid_center,
            output_dir=output_dir,
            **kwargs
        )

# def create_optimized_search_algorithm(manager, algorithm_type, scoring_function, **kwargs):
#     """
#     Create an optimized search algorithm based on available hardware.
#     """
#     # Extract grid-related settings from kwargs
#     grid_spacing = kwargs.pop('grid_spacing', 0.375)
#     grid_radius = kwargs.pop('grid_radius', 10.0)
#     grid_center = kwargs.pop('grid_center', None)
#     output_dir = kwargs.pop('output_dir', None)

#     # AUTO-SELECT PARALLEL IMPLEMENTATIONS
#     cpu_workers = kwargs.get('cpu_workers', None)
#     use_gpu = kwargs.get('use_gpu', False)
#     use_parallel = (cpu_workers is not None and cpu_workers > 1) or use_gpu
#     if cpu_workers is None:
#         cpu_workers = manager.n_cpu_workers
#     if cpu_workers is None:
#         cpu_workers = os.cpu_count() or 1
#     if cpu_workers > 1:
#         use_parallel = True
#     else:
#         use_parallel = False
#     # Check if GPU is available
#     if use_gpu:
#         print(f"[INFO] Using GPU acceleration for '{algorithm_type}'")
#     else:
#         print(f"[INFO] Using CPU acceleration for '{algorithm_type}'")
#     # Check if parallel implementation is available
#     if use_parallel:
#         print(f"[INFO] Using parallel implementation for '{algorithm_type}' with {cpu_workers} CPU workers")
#     else:
#         print(f"[INFO] Using serial implementation for '{algorithm_type}'")

#     # ----------------------------------------------------
#     if algorithm_type == 'genetic':
#         if use_gpu:  # Check if GPU is enabled
#             # Import GPU-specific genetic algorithm implementation
#             from .gpu_search import ParallelGeneticAlgorithm
#             return ParallelGeneticAlgorithm(
#                 scoring_function=scoring_function,
#                 grid_spacing=grid_spacing,
#                 grid_radius=grid_radius,
#                 grid_center=grid_center,
#                 output_dir=output_dir,
#                 **kwargs            
#             )
#         elif use_parallel:  # Check if parallel implementation is enabled and GPU is not used. if use_parallel:
#             from .parallel_search import ParallelGeneticAlgorithm
#             # Remove irrelevant kwargs
#             for key in ['n_steps', 'temperature', 'cooling_factor', 'high_temp', 'target_temp',
#                         'num_conformers', 'num_orientations', 'md_steps', 'minimize_steps',
#                         'use_grid', 'use_monte_carlo']:
#                 kwargs.pop(key, None)
#             return ParallelGeneticAlgorithm(
#                 scoring_function=scoring_function,
#                 grid_spacing=grid_spacing,
#                 grid_radius=grid_radius,
#                 grid_center=grid_center,
#                 output_dir=output_dir,
#                 **kwargs
#             )
#         else:
#             from .search import GeneticAlgorithm
#             return GeneticAlgorithm(
#                 scoring_function=scoring_function,
#                 grid_spacing=grid_spacing,
#                 grid_radius=grid_radius,
#                 grid_center=grid_center,
#                 output_dir=output_dir,
#                 **kwargs
#             )

#     elif algorithm_type == 'random':
#         if use_parallel:
#             from .parallel_search import ParallelRandomSearch
#             return ParallelRandomSearch(
#                 scoring_function=scoring_function,
#                 grid_spacing=grid_spacing,
#                 grid_radius=grid_radius,
#                 grid_center=grid_center,
#                 output_dir=output_dir,
#                 **kwargs
#             )
#         else:
#             from .search import RandomSearch
#             return RandomSearch(
#                 scoring_function=scoring_function,
#                 grid_spacing=grid_spacing,
#                 grid_radius=grid_radius,
#                 grid_center=grid_center,
#                 output_dir=output_dir,
#                 **kwargs
#             )
#     # ----------------------------------------------------
#     # Add support for Monte Carlo sampling
#     # ----------------------------------------------------
#     elif algorithm_type == 'monte-carlo':
#         try:
#             from .unified_scoring import MonteCarloSampling
#              # Remove unsupported kwargs for this algorithm
#             kwargs.pop('grid_radius', None)
#             kwargs.pop('grid_spacing', None)
#             kwargs.pop('grid_center', None)
#             kwargs.pop('max_iterations', None)
#             kwargs.pop('workload_balance', None)            
#             kwargs.pop('num_processes', None)
#             kwargs.pop('output_dir', None)
#             kwargs.pop('mc_steps', None)
#             kwargs.pop('temperature', None)
#             return MonteCarloSampling(
#                 scoring_function=scoring_function,
#                 grid_spacing=grid_spacing,  # Pass grid_spacing explicitly
#                 grid_radius=grid_radius,    # 💥 PASS properl
#                 grid_center=grid_center,     # 💥 PASS proper
#                 output_dir=kwargs.pop('output_dir', None),
#                 **kwargs
#             )
#         except ImportError:
#             print("Monte Carlo sampling not available. Using genetic algorithm instead.")
#             return create_optimized_search_algorithm(
#                 manager,
#                 'genetic',
#                 scoring_function,
#                 grid_spacing=grid_spacing,  # Pass grid_spacing explicitly
#                 grid_radius=grid_radius,    # 💥 PASS properly
#                 grid_center=grid_center,     # 💥 PASS properly
#                 output_dir=kwargs.pop('output_dir', None),
#                 **kwargs
#             )
            
#     elif algorithm_type == 'pandadock':
#         try:
#             from .pandadock import PANDADOCKAlgorithm
            
#             # Remove unsupported kwargs for this algorithm
#             kwargs.pop('grid_radius', None)
#             kwargs.pop('grid_spacing', None)
#             kwargs.pop('grid_center', None)
#             kwargs.pop('output_dir', None)

#             return PANDADOCKAlgorithm(
#                 scoring_function=scoring_function,
#                 output_dir=kwargs.pop('output_dir', None),
#                 **kwargs
#             )
#         ##############
#         # Add exception handling for PANDADOCK
#         ##############

#         except ImportError:
#             print("PANDADOCK algorithm not available. Using genetic algorithm instead.")
#             return create_optimized_search_algorithm(
#                 manager,
#                 'genetic',
#                 scoring_function,
#                 grid_spacing=grid_spacing,  # Pass grid_spacing explicitly
#                 grid_radius=grid_radius,    # 💥 PASS properly
#                 grid_center=grid_center,     # 💥 PASS properly
#                 output_dir=kwargs.pop('output_dir', None),
#                 **kwargs
#             )
    
#     # Unknown algorithm type
#     else:
#         print(f"Unknown algorithm type: {algorithm_type}. Using genetic algorithm.")
#         return create_optimized_search_algorithm(
#             manager,
#             'genetic',
#             scoring_function,
#             grid_spacing=grid_spacing,  # Pass grid_spacing explicitly
#             grid_radius=grid_radius,    # 💥 PASS properl
#             grid_center=grid_center,     # 💥 PASS property
#             output_dir=kwargs.pop('output_dir', None),
#             **kwargs
#         )
def get_scoring_type_from_args(args):
    """
    Determine scoring type based on command-line arguments.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    
    Returns:
    --------
    str
        Scoring type ('standard', 'enhanced', 'pandadock' or 'physics')
    """
    if args.physics_based:
        return 'physics'
    elif args.enhanced_scoring:
        return 'enhanced'
    else:
        return 'standard'
def get_algorithm_type_from_args(args):
    """
    Determine algorithm type based on command-line arguments.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    
    Returns:
    --------
    str
        Algorithm type ('genetic', 'random', 'monte-carlo', or 'pandadock')
    """
    if args.monte_carlo:
        return 'monte-carlo'
    else:
        return args.algorithm  # Now can include 'pandadock' as an option
    


def get_algorithm_kwargs_from_args(args):
    """
    Get algorithm keyword arguments based on command-line arguments and algorithm type.
    """
    algorithm_type = get_algorithm_type_from_args(args)
    algorithm_kwargs = {}
    
    # Common parameters for most algorithms
    if hasattr(args, 'iterations'):
        algorithm_kwargs['max_iterations'] = args.iterations
    
    # Algorithm-specific parameters
    if algorithm_type == 'genetic':
        if hasattr(args, 'population_size'):
            algorithm_kwargs['population_size'] = args.population_size
        
        if hasattr(args, 'mutation_rate'):
            algorithm_kwargs['mutation_rate'] = getattr(args, 'mutation_rate', 0.2)

        if hasattr(args, 'local_opt'):
            algorithm_kwargs['perform_local_opt'] = args.local_opt
        
    elif algorithm_type == 'monte-carlo':
        # Monte Carlo specific parameters
        if hasattr(args, 'mc_steps'):
            algorithm_kwargs['n_steps'] = args.mc_steps
        
        if hasattr(args, 'temperature'):
            algorithm_kwargs['temperature'] = args.temperature
            
        if hasattr(args, 'cooling_factor'):
            algorithm_kwargs['cooling_factor'] = args.cooling_factor
    
    elif algorithm_type == 'pandadock':
        # pandadock specific parameters
        if hasattr(args, 'high_temp'):
            algorithm_kwargs['high_temp'] = args.high_temp
            
        if hasattr(args, 'target_temp'):
            algorithm_kwargs['target_temp'] = args.target_temp
            
        if hasattr(args, 'num_conformers'):
            algorithm_kwargs['num_conformers'] = args.num_conformers
            
        if hasattr(args, 'num_orientations'):
            algorithm_kwargs['num_orientations'] = args.num_orientations
            
        if hasattr(args, 'md_steps'):
            algorithm_kwargs['md_steps'] = args.md_steps
            
        if hasattr(args, 'minimize_steps'):
            algorithm_kwargs['minimize_steps'] = args.minimize_steps
            
        if hasattr(args, 'use_grid'):
            algorithm_kwargs['use_grid'] = args.use_grid
    
    return algorithm_kwargs