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

from .search import DockingSearch
from .utils import (
    calculate_rmsd, is_within_grid, detect_steric_clash, 
    generate_spherical_grid, is_inside_sphere, random_point_in_sphere, 
    save_intermediate_result, update_status
)

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
        protein_coords = torch.tensor([atom['coords'] for atom in protein_atoms], 
                                     device=self.device)
        
        valid_poses = []
        for pose in poses:
            # Extract ligand coordinates
            ligand_coords = torch.tensor([atom['coords'] for atom in pose.atoms], 
                                        device=self.device)
            
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
        
        # Initialize population
        population = []
        print("Generating initial population...")
        
        # Generate clash-free initial poses using GPU acceleration
        for i in range(self.population_size * 2):  # Generate more than needed to ensure enough valid poses
            if len(population) >= self.population_size:
                break
                
            pose = self._generate_random_pose(ligand, center, radius)
            
            if not detect_steric_clash(protein.atoms, pose.atoms):
                population.append(pose)
            
            if i % 10 == 0:
                print(f"  Generated {len(population)}/{self.population_size} valid poses")
        
        # Evaluate initial population
        print("Evaluating initial population...")
        evaluated_population = []
        
        # Safety check: make sure we have valid poses
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
        
        # Add clear debug output
        print(f"Debug: Initial population contains {len(population)} poses")
        
        # Try to evaluate each pose with improved error handling
        success_count = 0
        failure_count = 0
        for i, pose in enumerate(population[:self.population_size]):
            try:
                score = self.scoring_function.score(protein, pose)
                evaluated_population.append((pose, score))
                success_count += 1
                
                # Periodic progress update for large populations
                if (i+1) % 10 == 0:
                    print(f"  Evaluated {i+1}/{min(len(population), self.population_size)} poses")
                
            except Exception as e:
                print(f"  Warning: Failed to evaluate pose {i}: {str(e)}")
                failure_count += 1
                
                # If too many failures, break and fall back
                if failure_count > 5 and success_count == 0:
                    print("ERROR: Multiple scoring failures with GPU implementation")
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
                    
                    # Mutation
                    if random.random() < self.mutation_rate:
                        self._mutate(child1, center, radius)
                    if random.random() < self.mutation_rate:
                        self._mutate(child2, center, radius)
                    
                    offspring.append((child1, None))
                    offspring.append((child2, None))
            
            # Filter offspring for clash-free poses using GPU acceleration
            offspring = [(pose, None) for pose in self._filter_poses_gpu([pose for pose, _ in offspring], protein)]
            
            # Ensure we have enough offspring
            while len(offspring) < self.population_size:
                # Add random individuals if needed
                pose = self._generate_random_pose(ligand, center, radius)
                if not detect_steric_clash(protein.atoms, pose.atoms):
                    offspring.append((pose, None))
            
            # Evaluate offspring
            eval_start = time.time()
            evaluated_offspring = []
            for pose, _ in offspring:
                score = self.scoring_function.score(protein, pose)
                evaluated_offspring.append((pose, score))
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
    
    def _mutate(self, individual, center, radius):
        """
        Mutate an individual.
        
        Parameters:
        -----------
        individual : Ligand
            Ligand to mutate
        center : array-like
            Center coordinates of the search sphere
        radius : float
            Radius of the search sphere
        """
        # Save original state to revert if needed
        original_xyz = individual.xyz.copy()
        
        # Apply random translation
        translation = np.random.normal(0, 0.5, 3)  # Standard deviation of 0.5 Å
        individual.translate(translation)
        
        # Apply random rotation
        angle = np.random.normal(0, 0.2)  # Standard deviation ~11 degrees
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        rotation = Rotation.from_rotvec(axis * angle)
        ind_centroid = np.mean(individual.xyz, axis=0)
        individual.translate(-ind_centroid)
        individual.rotate(rotation.as_matrix())
        individual.translate(ind_centroid)
        
        # If mutation moved the ligand outside the sphere, revert
        if not is_inside_sphere(individual, center, radius):
            individual.xyz = original_xyz
    
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
