"""
GPU-accelerated search algorithms for PandaDock.
This module provides GPU-accelerated implementations of search algorithms for molecular docking.
"""

import numpy as np
import copy
import random
import time
from pathlib import Path
from scipy.spatial.transform import Rotation
import os
import logging
import torch

from .search import DockingSearch, GeneticAlgorithm
from .utils import (
    calculate_rmsd, is_within_grid, detect_steric_clash, 
    generate_spherical_grid, is_inside_sphere, random_point_in_sphere, 
    save_intermediate_result, update_status, enforce_sphere_boundary
)

from .parallel_search import ParallelGeneticAlgorithm
from .utils import is_inside_sphere, detect_steric_clash

class GPUDockingSearch(DockingSearch):
    """
    Base class for GPU-accelerated docking search algorithms.
    
    This class extends the DockingSearch base class with GPU-specific functionality.
    GPU acceleration is implemented by offloading computationally intensive parts
    of the search algorithm to the GPU using either PyTorch or CuPy, depending on availability.
    """
    
    def __init__(self, scoring_function, max_iterations=100, output_dir=None, 
                 grid_spacing=0.375, grid_radius=10.0, grid_center=None,
                 device='cuda', precision='float32'):
        """
        Initialize the GPU-accelerated search algorithm.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function for pose evaluation
        max_iterations : int
            Maximum number of iterations/generations
        output_dir : str or Path
            Directory for output files
        grid_spacing : float
            Spacing between grid points
        grid_radius : float
            Radius of the search sphere
        grid_center : array-like
            Center coordinates of the search sphere
        device : str
            PyTorch device ('cuda' or 'cuda:n')
        precision : str
            Numerical precision ('float32' or 'float64')
        """
        super().__init__(scoring_function, max_iterations, output_dir, 
                         grid_spacing, grid_radius, grid_center)
        
        self.device_name = device
        self.precision = precision
        self.device = None
        self.torch_available = False
        self.cupy_available = False

        # GPU-specific attributes
        self.gpu_initialized = False
        self.gpu_constants = {}
        
        # Initialize GPU resources
        self._init_gpu()
        
        self.logger = logging.getLogger(__name__)
    
    def _init_gpu(self):
        """Initialize GPU resources and check availability."""
        # Try PyTorch first
        try:
            import torch
            self.torch_available = True
            
            # Check if CUDA is available
            if torch.cuda.is_available():
                self.device = torch.device(self.device_name)
                gpu_name = torch.cuda.get_device_name(0)
                print(f"Using GPU: {gpu_name}")
                
                # Set default tensor type
                if self.precision == 'float64':
                    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
                else:
                    torch.set_default_tensor_type(torch.cuda.FloatTensor)
            else:
                print("Warning: CUDA not available. Falling back to CPU.")
                self.device = torch.device('cpu')
                
            # Test GPU with a small calculation
            a = torch.rand(1000, 1000, device=self.device)
            b = torch.rand(1000, 1000, device=self.device)
            c = torch.matmul(a, b)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        
        except ImportError:
            print("PyTorch not available. Trying CuPy...")
            self.device = torch.device('cpu')
            #print("GPU not available or not requested. Using CPU via PyTorch.")
            if self.precision == 'float64':
                torch.set_default_tensor_type(torch.DoubleTensor)
            
           

    
            
            # Try CuPy as fallback
            try:
                import cupy as cp
                self.cupy_available = True
                
                # Check if CUDA is available
                try:
                    gpu_info = cp.cuda.runtime.getDeviceProperties(0)
                    print(f"Using GPU via CuPy: {gpu_info['name'].decode()}")
                except:
                    print("Warning: CUDA not available for CuPy. Falling back to CPU.")
                
                # Set precision
                self.cp = cp
                if self.precision == 'float64':
                    self.cp_dtype = cp.float64
                else:
                    self.cp_dtype = cp.float32
                
                # Test GPU with a small calculation
                a = cp.random.rand(1000, 1000).astype(self.cp_dtype)
                b = cp.random.rand(1000, 1000).astype(self.cp_dtype)
                c = cp.matmul(a, b)
                cp.cuda.stream.get_current_stream().synchronize()
                
            except ImportError:
                print("Neither PyTorch nor CuPy available. Falling back to CPU calculations.")
                print("For GPU acceleration, install PyTorch or CuPy with CUDA support.")
                self.torch_available = False
                self.cupy_available = False
    
    def _set_gpu_constants(self, center=None, radius=None):
        """Set GPU constant values for search parameters."""
        if not self.gpu_initialized:
            return
        
        try:
            import torch
            constants = {}
            
            if center is not None:
                constants['center'] = torch.tensor(center, device=self.gpu_device)
            
            if radius is not None:
                constants['radius'] = torch.tensor(radius, device=self.gpu_device)
                
            self.gpu_constants = constants
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error setting GPU constants: {e}")
            else:
                print(f"Error setting GPU constants: {e}")
    
    def search(self, protein, ligand):
        """
        Perform docking search using GPU acceleration.
        
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
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement this method")

    def _filter_poses_gpu(self, poses, protein):
        """
        Filter poses for steric clashes using GPU acceleration.
        
        Parameters:
        -----------
        poses : list
            List of ligand poses
        protein : Protein
            Protein object
        
        Returns:
        --------
        list
            Filtered list of valid poses
        """
        if not poses:
            return []
        
        if self.torch_available:
            return self._filter_poses_torch(poses, protein)
        elif self.cupy_available:
            return self._filter_poses_cupy(poses, protein)
        else:
            # Fall back to CPU implementation
            return [pose for pose in poses if not detect_steric_clash(protein.atoms, pose.atoms)]
    
    def _filter_poses_torch(self, poses, protein):
        """
        Filter poses using PyTorch for GPU acceleration.
        
        Parameters:
        -----------
        poses : list
            List of ligand poses
        protein : Protein
            Protein object
        
        Returns:
        --------
        list
            Filtered list of valid poses
        """
        import torch
        
        # Get protein atoms
        if hasattr(protein, 'active_site') and protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Extract protein coordinates
        # protein_coords = torch.tensor([atom['coords'] for atom in protein_atoms], 
        #                              device=self.device)
        # Much faster tensor creation
        protein_coords_array = np.array([atom['coords'] for atom in protein_atoms])
        protein_coords = torch.tensor(protein_coords_array, device=self.device)
        
        valid_poses = []
        for pose in poses:
            # Extract ligand coordinates
            # ligand_coords = torch.tensor([atom['coords'] for atom in pose.atoms], 
            #                             device=self.device)
            ligand_coords_array = np.array([atom['coords'] for atom in pose.atoms])
            ligand_coords = torch.tensor(ligand_coords_array, device=self.device)  # ligand_coords_array, device=self.device)
            
            # Calculate all pairwise distances efficiently
            dists = torch.cdist(ligand_coords, protein_coords)
            
            # Check if any distances are below clash threshold
            if not torch.any(dists < 1.5):  # 1.5Å threshold for clashes
                valid_poses.append(pose)
        
        return valid_poses
    
    def _filter_poses_cupy(self, poses, protein):
        """
        Filter poses using CuPy for GPU acceleration.
        
        Parameters:
        -----------
        poses : list
            List of ligand poses
        protein : Protein
            Protein object
        
        Returns:
        --------
        list
            Filtered list of valid poses
        """
        import cupy as cp
        
        # Get protein atoms
        if hasattr(protein, 'active_site') and protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Extract protein coordinates
        protein_coords = cp.array([atom['coords'] for atom in protein_atoms], 
                                 dtype=self.cp_dtype)
        
        valid_poses = []
        for pose in poses:
            # Extract ligand coordinates
            ligand_coords = cp.array([atom['coords'] for atom in pose.atoms], 
                                    dtype=self.cp_dtype)
            
            # Calculate all pairwise distances efficiently
            dists = cp.sqrt(((ligand_coords[:, None, :] - protein_coords[None, :, :]) ** 2).sum(axis=2))
            
            # Check if any distances are below clash threshold
            if not cp.any(dists < 1.5):  # 1.5Å threshold for clashes
                valid_poses.append(pose)
        
        return valid_poses
    
    def _score_batch_gpu(self, protein, poses):
        """
        Score a batch of poses using GPU acceleration.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        poses : list
            List of ligand poses
        
        Returns:
        --------
        list
            List of (pose, score) tuples
        """
        # This is a basic implementation that delegates to the scoring function
        # Subclasses may override with more optimized GPU batching
        results = []
        for pose in poses:
            score = self.scoring_function.score(protein, pose)
            results.append((pose, score))
        
        return results


class ParallelGeneticAlgorithm(GPUDockingSearch):
    """
    GPU-accelerated genetic algorithm for molecular docking.
    
    This class implements a genetic algorithm that leverages GPU acceleration
    for fitness evaluation, selection, crossover, and mutation.
    """
    
    def __init__(self, scoring_function, max_iterations=100, population_size=150, 
                 mutation_rate=0.2, crossover_rate=0.8, tournament_size=3, 
                 output_dir=None, perform_local_opt=False, grid_spacing=0.375, 
                 grid_radius=10.0, grid_center=None, device='cuda', precision='float32'):
        """
        Initialize GPU-accelerated genetic algorithm.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function for pose evaluation
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
        output_dir : str or Path
            Directory for output files
        perform_local_opt : bool
            Whether to perform local optimization on top poses
        grid_spacing : float
            Spacing between grid points
        grid_radius : float
            Radius of the search sphere
        grid_center : array-like
            Center coordinates of the search sphere
        device : str
            PyTorch device ('cuda' or 'cuda:n')
        precision : str
            Numerical precision ('float32' or 'float64')
        """
        super().__init__(scoring_function, max_iterations, output_dir, 
                         grid_spacing, grid_radius, grid_center, device, precision)
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.perform_local_opt = perform_local_opt
        
        # Performance tracking
        self.eval_time = 0.0
        self.total_time = 0.0
        self.best_score = float('inf')
        self.best_pose = None
    
    def search(self, protein, ligand):
        """
        Perform genetic algorithm search with GPU acceleration.
        
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
        
        # Transfer center and radius to GPU constants
        self._set_gpu_constants(center=center, radius=radius)
        
        # Generate poses on GPU
        gpu_poses = self._generate_gpu_poses(ligand, center, radius, self.population_size)
        
        # CRITICAL: Apply sphere constraint check
        valid_poses = []
        for pose in gpu_poses:
            if is_inside_sphere(pose, center, radius):
                valid_poses.append(pose)
        # Calculate minimum required poses - use a fraction of population_size instead
        min_required_poses = max(10, self.population_size // 4)  # At least 25% of population
    
        # If too few valid poses, regenerate with stricter constraints
        if len(valid_poses) < min_required_poses:
            print(f"Warning: Only {len(valid_poses)}/{len(gpu_poses)} poses within sphere. Regenerating...")
            gpu_poses = self._generate_gpu_poses(ligand, center, radius, self.population_size)
            valid_poses = [pose for pose in gpu_poses if is_inside_sphere(pose, center, radius)]
        # Implement regeneration logic here
        while len(valid_poses) < min_required_poses:
            # Generate additional poses
            additional_poses = self._generate_gpu_poses(
                ligand, center, radius * 0.9, min_required_poses - len(valid_poses))
            
            # Filter for valid poses
            for pose in additional_poses:
                if is_inside_sphere(pose, center, radius):
                    valid_poses.append(pose)
                    
                # Break if we have enough
                if len(valid_poses) >= min_required_poses:
                    break
        
        # Ensure active site atoms are defined for faster scoring
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
        
        # Initialize grid points
        self.initialize_grid_points(center, protein=protein)
        
        print(f"Using GPU acceleration for genetic algorithm")
        print(f"Population size: {self.population_size}")
        print(f"Maximum generations: {self.max_iterations}")
        print(f"Mutation rate: {self.mutation_rate}")
        print(f"Crossover rate: {self.crossover_rate}")
        print(f"Local optimization: {self.perform_local_opt}")
        
        # Initialize population using the new method
        print("Generating initial population...")
        population = self._generate_initial_population(protein, ligand, center, radius, max_attempts=200)
        
        if len(population) < self.population_size:
            print(f"WARNING: Could only generate {len(population)}/{self.population_size} valid poses")
            if len(population) == 0:
                print("ERROR: No valid poses generated during population initialization")
                print("Falling back to CPU implementation...")
                
                # Fall back to CPU implementation
                from .search import GeneticAlgorithm
                cpu_algorithm = GeneticAlgorithm(
                    self.scoring_function, 
                    self.max_iterations, 
                    population_size=self.population_size,
                    mutation_rate=self.mutation_rate
                )
                # Copy over grid parameters
                cpu_algorithm.grid_spacing = self.grid_spacing
                cpu_algorithm.grid_radius = self.grid_radius
                cpu_algorithm.grid_center = self.grid_center
                cpu_algorithm.output_dir = self.output_dir
                
                # Run CPU search and return results
                return cpu_algorithm.search(protein, ligand)
        
        print("Evaluating initial population...")
        
        # Use batch scoring for initial population
        poses = [pose for pose, _ in population]
        scores = self._batch_score_poses(protein, poses)
        
        # Combine poses and scores
        evaluated_population = [(poses[i], scores[i]) for i in range(len(poses))]
        print(f"Debug: Initial population contains {len(evaluated_population)} poses")
        
        # Filter out any invalid scores
        evaluated_population = [(pose, score) for pose, score in evaluated_population if np.isfinite(score)]
        
        # Check if we have any valid evaluated poses
        if not evaluated_population:
            print("ERROR: All pose evaluations failed with GPU implementation")
            print("Falling back to CPU implementation...")
            
            # Fall back to CPU implementation
            from .search import GeneticAlgorithm
            cpu_algorithm = GeneticAlgorithm(
                self.scoring_function, 
                self.max_iterations, 
                population_size=self.population_size,
                mutation_rate=self.mutation_rate
            )
            # Copy over grid parameters
            cpu_algorithm.grid_spacing = self.grid_spacing
            cpu_algorithm.grid_radius = self.grid_radius
            cpu_algorithm.grid_center = self.grid_center
            cpu_algorithm.output_dir = self.output_dir
            
            # Run CPU search and return results
            return cpu_algorithm.search(protein, ligand)
        
        # Sort by score
        evaluated_population.sort(key=lambda x: x[1])
        
        # Store best individual
        best_individual = evaluated_population[0]
        self.best_pose = best_individual[0]
        self.best_score = best_individual[1]
        
        print(f"Generation 0: Best score = {self.best_score:.4f}")
        
        # Track all individuals if population is diverse
        all_individuals = [evaluated_population[0]]
        
        # Main evolutionary loop
        for generation in range(self.max_iterations):
            self.current_generation = generation  # Store for adaptive mutation
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
                        child1, child2 = self._crossover_pair(parent1, parent2, center, radius)
                    else:
                        child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                    
                    # Apply enhanced mutation
                    self._mutate(child1, center, radius)
                    self._mutate(child2, center, radius)
                    
                    offspring.append((child1, None))
                    offspring.append((child2, None))
            
            # Filter offspring using improved validity check
            filtered_offspring = []
            for pose, _ in offspring:
                if self._check_pose_validity(pose, protein):
                    filtered_offspring.append((pose, None))
            
            offspring = filtered_offspring
            
            # Ensure we have enough offspring
            attempts = 0
            while len(offspring) < self.population_size and attempts < 50:
                # Add random individuals if needed
                pose = self._generate_random_pose(ligand, center, radius)
                if self._check_pose_validity(pose, protein):
                    offspring.append((pose, None))
                attempts += 1
            
            # Evaluate offspring using batch scoring
            eval_start = time.time()
            offspring_poses = [pose for pose, _ in offspring]
            offspring_scores = self._batch_score_poses(protein, offspring_poses)
            evaluated_offspring = [(offspring_poses[i], offspring_scores[i]) for i in range(len(offspring_poses))]
            # Filter out any invalid scores
            evaluated_offspring = [(pose, score) for pose, score in evaluated_offspring if np.isfinite(score)]
            self.eval_time += time.time() - eval_start
            
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
                
                # Save intermediate result
                if self.output_dir:
                    save_intermediate_result(
                        self.best_pose, self.best_score, generation + 1, 
                        self.output_dir, self.max_iterations
                    )
                    
                    # Update status
                    update_status(
                        self.output_dir,
                        current_generation=generation + 1,
                        best_score=self.best_score,
                        total_generations=self.max_iterations,
                        progress=(generation + 1) / self.max_iterations
                    )
            
            # Display progress
            gen_time = time.time() - gen_start
            print(f"Generation {generation + 1}/{self.max_iterations}: "
                f"Best score = {self.best_score:.4f}, "
                f"Current best = {evaluated_population[0][1]:.4f}, "
                f"Time = {gen_time:.2f}s")
            
            # Apply local search to the best individual occasionally
            if self.perform_local_opt and generation % 5 == 0:
                from .search import DockingSearch
                best_pose, best_score = DockingSearch._local_optimization(
                    self, evaluated_population[0][0], protein
                )
                
                if best_score < self.best_score:
                    self.best_pose = best_pose
                    self.best_score = best_score
                    
                    # Replace best individual in population
                    evaluated_population[0] = (best_pose, best_score)
                    evaluated_population.sort(key=lambda x: x[1])
                    all_individuals.append((best_pose, best_score))
        
        # Final local optimization for top poses
        if self.perform_local_opt:
            print("\nPerforming final local optimization on top poses...")
            optimized_results = []
            
            # Optimize top 5 poses
            poses_to_optimize = min(5, len(evaluated_population))
            for i, (pose, score) in enumerate(evaluated_population[:poses_to_optimize]):
                from .search import DockingSearch
                opt_pose, opt_score = DockingSearch._local_optimization(
                    self, pose, protein
                )
                optimized_results.append((opt_pose, opt_score))
                print(f"  Pose {i+1}: Score improved from {score:.4f} to {opt_score:.4f}")
            
            # Combine with remaining poses
            optimized_results.extend(evaluated_population[poses_to_optimize:])
            optimized_results.sort(key=lambda x: x[1])
            
            self.total_time = time.time() - start_time
            print(f"\nSearch completed in {self.total_time:.2f} seconds")
            print(f"Best score: {optimized_results[0][1]:.4f}")
            
            return optimized_results
        
        # Return unique solutions, best first
        self.total_time = time.time() - start_time
        print(f"\nSearch completed in {self.total_time:.2f} seconds")
        print(f"Evaluation time: {self.eval_time:.2f} seconds ({self.eval_time/self.total_time*100:.1f}%)")
        print(f"Best score: {evaluated_population[0][1]:.4f}")
        
        # Sort all individuals by score and ensure uniqueness
        all_individuals.extend(evaluated_population)
        all_individuals.sort(key=lambda x: x[1])
        
        # Remove duplicates
        unique_results = []
        seen_scores = set()
        for pose, score in all_individuals:
            rounded_score = round(score, 4)
            if rounded_score not in seen_scores:
                unique_results.append((pose, score))
                seen_scores.add(rounded_score)
            
            if len(unique_results) >= 20:  # Limit to top 20
                break
        
        # Free GPU memory after search
        if self.torch_available:
            import torch
            torch.cuda.empty_cache()
        
        return unique_results
    def _selection(self, population):
        """
        Tournament selection of parents.
        
        Parameters:
        -----------
        population : list
            List of (pose, score) tuples
        
        Returns:
        --------
        list
            Selected parents as (pose, score) tuples
        """
        selected = []
        
        for _ in range(self.population_size):
            # Select random individuals for tournament
            tournament = random.sample(population, min(self.tournament_size, len(population)))
            
            # Select the best from tournament
            tournament.sort(key=lambda x: x[1])
            selected.append(tournament[0])
        
        return selected
    
    def _crossover_pair(self, parent1, parent2, center, radius):
        """
        Perform crossover between two parents.
        
        Parameters:
        -----------
        parent1 : Ligand
            First parent
        parent2 : Ligand
            Second parent
        center : array-like
            Center coordinates of the search sphere
        radius : float
            Radius of the search sphere
        
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
        
        # Rotation interpolation
        rotation1 = Rotation.random()
        rotation2 = Rotation.random()
        
        # Apply rotations to children
        c1_centroid = np.mean(child1.xyz, axis=0)
        child1.translate(-c1_centroid)
        child1.rotate(rotation1.as_matrix())
        child1.translate(c1_centroid)
        
        c2_centroid = np.mean(child2.xyz, axis=0)
        child2.translate(-c2_centroid)
        child2.rotate(rotation2.as_matrix())
        child2.translate(c2_centroid)
        
        # Ensure children are within the search sphere
        if not is_inside_sphere(child1, center, radius):
            child1 = copy.deepcopy(parent1)  # Revert to parent
            
        if not is_inside_sphere(child2, center, radius):
            child2 = copy.deepcopy(parent2)  # Revert to parent
        
        return child1, child2
    

    def _save_sphere_pdb(self, center, radius, filename="sphere.pdb"):
        """
        Save a PDB file with a sphere of dummy atoms centered at `center` with radius `radius`.
        
        Parameters:
        -----------
        center : np.ndarray
            3D center of the sphere
        radius : float
            Radius of the sphere
        filename : str
            Output PDB filename
        """
        with open(filename, 'w') as f:
            for i in range(100):  # Sample 100 points on the sphere
                theta = np.random.uniform(0, 2*np.pi)
                phi = np.random.uniform(0, np.pi)
                
                x = center[0] + radius * np.sin(phi) * np.cos(theta)
                y = center[1] + radius * np.sin(phi) * np.sin(theta)
                z = center[2] + radius * np.cos(phi)
                
                f.write(
                    f"ATOM  {i+1:5d}  X   SPH A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          X\n"
                )
            f.write("END\n")
    def _mutate(self, individual, center, radius):
        """
        Mutate an individual with probability mutation_rate.
        
        Parameters:
        -----------
        individual : Ligand or tuple
            Individual to mutate, either a Ligand object or a (pose, score) tuple
        center : array-like
            Center coordinates of the search sphere
        radius : float
            Radius of the search sphere
        """
        # Check if we're dealing with a tuple (pose, score)
        if isinstance(individual, (list, tuple)) and len(individual) >= 1:
            pose = individual[0]  # Extract the pose from the tuple
        else:
            pose = individual  # Assume it's already a Ligand object
        
        # Safety check for Ligand object
        if not hasattr(pose, 'xyz'):
            print(f"Warning: Expected Ligand object but got {type(pose)}")
            return individual  # Return unchanged
        
        # Save original coordinates for potential reversion
        original_xyz = pose.xyz.copy()
        
        # Only mutate with probability self.mutation_rate
        if random.random() >= self.mutation_rate:
            return individual  # No mutation
        
        # Apply random mutation
        mutation_type = random.choice(['translation', 'rotation', 'both'])
        
        if mutation_type in ['translation', 'both']:
            # Random translation
            translation = np.random.normal(0, 1.0, 3)  # Standard deviation reduced to 1.0
            pose.translate(translation)
        
        if mutation_type in ['rotation', 'both']:
            # Random rotation
            angle = np.random.normal(0, 0.3)  # Reduce from 0.5 to 0.3 radians (~17 degrees)
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            
            rotation = Rotation.from_rotvec(angle * axis)
            centroid = np.mean(pose.xyz, axis=0)
            pose.translate(-centroid)
            pose.rotate(rotation.as_matrix())
            pose.translate(centroid)
            #pose = enforce_sphere_boundary(pose, center, radius)
        
        # Check if pose is still within sphere
        if not is_inside_sphere(pose, center, radius):
            # Move pose back to sphere boundary
            centroid = np.mean(pose.xyz, axis=0)
            to_center = center - centroid
            dist = np.linalg.norm(to_center)
            
            if dist > 0:
                # Calculate how far to move back toward center
                move_vector = to_center * (dist - radius*0.9)/dist
                pose.translate(move_vector)
                
                # Verify the move worked
                if not is_inside_sphere(pose, center, radius):
                    # If still outside, revert to original
                    pose.xyz = original_xyz
        
        # Check if pose is valid (no severe clashes)
        # This would depend on your implementation of _check_pose_validity
        
        # Return the original tuple structure if input was a tuple
        if isinstance(individual, (list, tuple)) and len(individual) > 1:
            return (pose, individual[1])  # (pose, score)
        else:
            return pose  # Just return the pose
    
    def _generate_random_pose(self, ligand, center, radius):
        """
        Generate a random ligand pose within the search sphere.
        
        Parameters:
        -----------
        ligand : Ligand
            Ligand to position
        center : array-like
            Center coordinates of the search sphere
        radius : float
            Radius of the search sphere
        
        Returns:
        --------
        Ligand
            Randomly positioned ligand
        """
        # Make sure center is a numpy array
        center = np.array(center)
        if center.shape != (3,):
            print(f"Warning: Invalid center shape: {center.shape}, using [0,0,0]")
            center = np.array([0.0, 0.0, 0.0])
        
        pose = copy.deepcopy(ligand)
        # CRITICAL: Add this check before returning
        if not is_inside_sphere(pose, center, radius):
            # Either reject this pose or move it back inside the sphere
            centroid = np.mean(pose.xyz, axis=0)
            direction = centroid - center
            distance = np.linalg.norm(direction)
            if distance > 0:
                # Normalize and scale to keep within radius
                direction = direction / distance
                new_position = center + direction * (radius * 0.8)  # Use 80% of radius to be safe
                # Translate the pose
                translation = new_position - centroid
                pose.translate(translation)
        try:
            # Choose a random grid point if available
            if self.grid_points is not None and len(self.grid_points) > 0:
                grid_point = random.choice(self.grid_points)
                centroid = np.mean(pose.xyz, axis=0)
                translation = grid_point - centroid
                pose.translate(translation)
            else:
                # Generate a random point in the sphere
                r = radius * random.random() ** (1/3)  # Uniform distribution in sphere volume
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
            centroid = np.mean(pose.xyz, axis=0)
            pose.translate(-centroid)
            pose.rotate(rotation.as_matrix())
            pose.translate(centroid)
            
            return pose
        except Exception as e:
            print(f"Error in _generate_random_pose: {e}")
            # Return a simple default position as fallback
            pose = copy.deepcopy(ligand)
            centroid = np.mean(pose.xyz, axis=0)
            translation = center - centroid
            pose.translate(translation)
            return pose
        
    def _check_pose_validity(self, ligand, protein, clash_threshold=2.0):
        # Safety checks
        if not ligand.atoms:
            return False  # No ligand atoms
        
        # Get ligand coordinates
        ligand_coords = np.array([atom['coords'] for atom in ligand.atoms])
        
        # Get protein coordinates
        if hasattr(protein, 'active_site') and protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        if not protein_atoms:
            return True  # No protein atoms to clash with
        
        protein_coords = np.array([atom['coords'] for atom in protein_atoms])
        
        # Ensure coordinates have proper shape for torch.cdist (batch, dims)
        if len(ligand_coords) == 0:
            return False
        
        if len(protein_coords) == 0:
            return True  # No protein atoms to clash with

        # Convert to torch tensors with proper reshaping
        ligand_coords = torch.tensor(ligand_coords, device=self.device)
        protein_coords = torch.tensor(protein_coords, device=self.device)
        
        # Reshape if needed (ensure 2D tensors)
        if ligand_coords.dim() == 1:
            ligand_coords = ligand_coords.unsqueeze(0)  # Add batch dimension
        
        if protein_coords.dim() == 1:
            protein_coords = protein_coords.unsqueeze(0)  # Add batch dimension
        
        # Calculate distances using try-except to catch errors
        try:
            distances = torch.cdist(ligand_coords, protein_coords)
            
            # Check for clashes
            if torch.any(distances < clash_threshold):
                return False  # Clash detected
            
            return True
        except Exception as e:
            print(f"Warning: Error in distance calculation: {e}")
            print(f"Ligand shape: {ligand_coords.shape}, Protein shape: {protein_coords.shape}")
            
            # Fall back to CPU calculation
            lig_numpy = ligand_coords.cpu().numpy()
            prot_numpy = protein_coords.cpu().numpy()
            
            for l_coord in lig_numpy:
                for p_coord in prot_numpy:
                    distance = np.linalg.norm(l_coord - p_coord)
                    if distance < clash_threshold:
                        return False  # Clash detected
            
            return True
    
    def _generate_initial_population(self, protein, ligand, center, radius, max_attempts=200):
        """
        Generate valid initial population with more robust approach.
        """
        population = []
        print(f"Attempting to generate {self.population_size} valid poses...")
        
        attempts = 0
        while len(population) < self.population_size and attempts < max_attempts:
            # Create a fresh copy of the ligand
            pose = copy.deepcopy(ligand)
            
            # Try different sampling strategies based on attempt count
            if attempts < max_attempts // 3:
                # Strategy 1: Uniform random sampling within sphere
                r = radius * random.random() ** (1/3)
                theta = random.uniform(0, 2 * np.pi)
                phi = random.uniform(0, np.pi)
                
                x = center[0] + r * np.sin(phi) * np.cos(theta)
                y = center[1] + r * np.sin(phi) * np.sin(theta)
                z = center[2] + r * np.sin(phi) * np.cos(phi)
                
            elif attempts < 2 * max_attempts // 3:
                # Strategy 2: Sample further from protein surface
                r = radius * (0.5 + 0.5 * random.random())  # Bias toward outer half of sphere
                theta = random.uniform(0, 2 * np.pi)
                phi = random.uniform(0, np.pi)
                
                x = center[0] + r * np.sin(phi) * np.cos(theta)
                y = center[1] + r * np.sin(phi) * np.sin(theta)
                z = center[2] + r * np.sin(phi) * np.cos(phi)
                
            else:
                # Strategy 3: Try grid-based sampling
                if hasattr(self, 'grid_points') and self.grid_points is not None and len(self.grid_points) > 0:
                    point = self.grid_points[random.randint(0, len(self.grid_points)-1)]
                    x, y, z = point
                else:
                    # Fallback to random with larger radius
                    r = radius * 1.2 * random.random() ** (1/3)
                    theta = random.uniform(0, 2 * np.pi)
                    phi = random.uniform(0, np.pi)
                    
                    x = center[0] + r * np.sin(phi) * np.cos(theta)
                    y = center[1] + r * np.sin(phi) * np.sin(theta)
                    z = center[2] + r * np.sin(phi) * np.cos(phi)
            
            # Move ligand to this point
            centroid = np.mean(pose.xyz, axis=0)
            translation = np.array([x, y, z]) - centroid
            pose.translate(translation)
            
            # Apply random rotation
            rotation = Rotation.random()
            rotation_matrix = rotation.as_matrix()
            pose.translate(-centroid)
            pose.rotate(rotation_matrix)
            pose.translate(centroid)
            
            # Move slightly away from protein center to avoid deep clashes
            away_vector = np.array([x, y, z]) - center
            if np.linalg.norm(away_vector) > 0.1:
                away_vector = away_vector / np.linalg.norm(away_vector) * 2.0  # Push 2Å out
                pose.translate(away_vector)
            
            # Check validity with higher threshold during initialization
            if self._check_pose_validity(pose, protein, clash_threshold=2.0):
                population.append((pose, None))  # Score will be computed later
                if len(population) % 5 == 0 or len(population) == self.population_size:
                    print(f"  Generated {len(population)}/{self.population_size} valid poses")
            
            attempts += 1
            if attempts % 20 == 0:
                print(f"  Made {attempts}/{max_attempts} attempts...")
        
        return population
    
    def _evaluate_population(self, protein, population):
        """
        Evaluate population using the same scoring function as the CPU version.
        """
        results = []
        
        # Process all poses
        for i, (pose, _) in enumerate(population):
            # Show progress for large populations
            if i % 10 == 0 and i > 0 and len(population) > 50:
                print(f"  Evaluating pose {i}/{len(population)}...")
                    
            # Use the scoring_function that was passed to the constructor
            # This ensures consistent scoring between CPU and GPU versions
            score = self.scoring_function.score(protein, pose)
            results.append((copy.deepcopy(pose), score))
                
        return results
    
    def _batch_score_poses(self, protein, poses, batch_size=5):
        """
        Score multiple poses in batches for better GPU utilization.
        """
        if not self.torch_available:
            # Fall back to CPU implementation if PyTorch is not available
            return [self.scoring_function.score(protein, pose) for pose in poses]
        
        import torch
        with torch.no_grad():  # Disable gradient tracking for inference
            scores = []
            
            # Process in batches
            for i in range(0, len(poses), batch_size):
                batch = poses[i:i+batch_size]
                batch_scores = []
                
                # Use CPU fallback for now since we don't know the structure of scoring_function
                # In a real implementation, you would modify this to leverage GPU acceleration
                for pose in batch:
                    try:
                        score = self.scoring_function.score(protein, pose)
                        batch_scores.append(score)
                    except Exception as e:
                        print(f"Warning: Failed to score pose: {str(e)}")
                        # Use a high score to indicate failure
                        batch_scores.append(float('inf'))
                
                scores.extend(batch_scores)
            
            return scores
    
    def is_inside_sphere_gpu(self, pose, center, radius):
        """Check if pose centroid is within the sphere on GPU"""
        # Get pose centroid - implement GPU version if needed
        centroid = np.mean(pose.xyz, axis=0)
        # Calculate distance to center
        distance = np.linalg.norm(centroid - center)
        print(f"Ligand at distance {distance} from center, radius is {radius}")
        # Return True if inside sphere
        return distance <= radius
    
    # def _enforce_sphere_constraint(self, poses, center, radius):
    #     """Ensure all poses are within the sphere boundary"""
    #     constrained_poses = []
    #     for pose in poses:
    #         if is_inside_sphere(pose, center, radius):
    #             constrained_poses.append(pose)
    #         else:
    #             # Option 1: Skip poses outside boundary
    #             continue
                
    #             # Option 2: Or adjust poses to fit within boundary
    #             # adjusted_pose = self._adjust_pose_to_sphere(pose, center, radius)
    #             # constrained_poses.append(adjusted_pose)
        
    #     return constrained_poses

    def _enforce_sphere_constraint(self, poses, center, radius):
        """
        Ensure all poses are within the sphere boundary
        
        Parameters:
        -----------
        poses : Ligand or list of Ligand
            Single ligand or list of ligands to check
        center : array-like
            Center coordinates of sphere
        radius : float
            Radius of sphere
            
        Returns:
        --------
        list
            List of poses within sphere boundary
        """
        # Handle case where 'poses' is a single Ligand object
        if not isinstance(poses, (list, tuple)) and hasattr(poses, 'atoms'):
            # Single ligand case
            if is_inside_sphere(poses, center, radius):
                return [poses]  # Return as a list containing the single pose
            else:
                # Option 1: Return empty list if the pose is outside boundary
                return []
                
                # Option 2: Adjust pose to fit within boundary and return
                # adjusted_pose = self._adjust_pose_to_sphere(poses, center, radius)
                # return [adjusted_pose]
        
        # Handle case where 'poses' is a collection
        constrained_poses = []
        for pose in poses:
            if is_inside_sphere(pose, center, radius):
                constrained_poses.append(pose)
            else:
                # Option 1: Skip poses outside boundary
                continue
                
                # Option 2: Adjust poses to fit within boundary
                # adjusted_pose = self._adjust_pose_to_sphere(pose, center, radius)
                # constrained_poses.append(adjusted_pose)
        
        return constrained_poses
    
    def _adjust_pose_to_sphere(self, pose, center, radius):
        """
        Adjust a pose to fit within a sphere boundary
        
        Parameters:
        -----------
        pose : Ligand
            Ligand pose to adjust
        center : array-like
            Center coordinates of sphere
        radius : float
            Radius of sphere
            
        Returns:
        --------
        Ligand
            Adjusted ligand pose
        """
        # Create a deep copy to avoid modifying the original
        import copy
        adjusted_pose = copy.deepcopy(pose)
        
        # Calculate centroid of the pose
        centroid = np.mean([atom['coords'] for atom in adjusted_pose.atoms], axis=0)
        
        # Calculate vector from center to centroid
        vector = centroid - center
        
        # Calculate distance from center to centroid
        distance = np.linalg.norm(vector)
        
        # If pose is outside the sphere, move it to just inside
        if distance > radius:
            # Create unit vector
            unit_vector = vector / distance
            
            # Calculate new position (0.9 * radius to give some margin)
            new_position = center + unit_vector * (0.9 * radius)
            
            # Calculate translation vector
            translation = new_position - centroid
            
            # Apply translation
            adjusted_pose.translate(translation)
        
        return adjusted_pose
    
    def _generate_gpu_poses(self, ligand, center, radius, num_poses=None):
        """
        Generate multiple ligand poses using GPU acceleration.
        
        Parameters:
        -----------
        ligand : Ligand
            Template ligand
        center : np.ndarray
            Center coordinates
        radius : float
            Radius of search sphere
        num_poses : int, optional
            Number of poses to generate. If None, uses population_size
        
        Returns:
        --------
        list
            List of generated poses
        """
        # Use population_size if num_poses not provided
        if num_poses is None:
            num_poses = self.population_size
        
        poses = []
        for _ in range(num_poses):
            # Generate a single pose
            pose = copy.deepcopy(ligand)
            
            # Generate a random point inside the sphere
            r = radius * random.random() ** (1/3)  # For uniform distribution in sphere
            theta = random.uniform(0, 2 * np.pi)
            phi = random.uniform(0, np.pi)
            
            x = center[0] + r * np.sin(phi) * np.cos(theta)
            y = center[1] + r * np.sin(phi) * np.sin(theta)
            z = center[2] + r * np.cos(phi)
            
            # Move ligand centroid to this point
            centroid = np.mean(pose.xyz, axis=0)
            translation = np.array([x, y, z]) - centroid
            pose.translate(translation)
            #pose = enforce_sphere_boundary(pose, center, radius)
            
            # Apply random rotation
            rotation = Rotation.random()
            centroid = np.mean(pose.xyz, axis=0)
            pose.translate(-centroid)
            pose.rotate(rotation.as_matrix())
            pose.translate(centroid)
            #pose = enforce_sphere_boundary(pose, center, radius)
            
            # Ensure the pose is inside the sphere
            if not is_inside_sphere(pose, center, radius):
                # Move it back inside if needed
                centroid = np.mean(pose.xyz, axis=0)
                vector = centroid - center
                distance = np.linalg.norm(vector)
                if distance > 0:
                    vector = vector / distance  # Normalize
                    new_centroid = center + vector * (radius * 0.8)  # 80% of radius for safety
                    translation = new_centroid - centroid
                    pose.translate(translation)
            
            poses.append(pose)
        return poses