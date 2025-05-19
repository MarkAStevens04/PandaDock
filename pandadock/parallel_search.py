"""
Parallel search algorithms for PandaDock.
This module provides parallel implementations of search algorithms for molecular docking
that leverage multi-core CPUs for improved performance.
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

from .search import DockingSearch
from .search import GeneticAlgorithm, RandomSearch
from .utils import (
    calculate_rmsd, is_within_grid, detect_steric_clash, 
    generate_spherical_grid, generate_cartesian_grid, 
    is_inside_sphere, random_point_in_sphere, local_optimize_pose
)

# ------------------------------------------------------------------------------
# Base Parallel Search Class
# ------------------------------------------------------------------------------
class ParallelSearch(DockingSearch):
    def search(self, protein, ligand):
        print("\n🔍 Performing docking (parallel mode enabled)...\n")
        return self.improve_rigid_docking(protein, ligand, self.args)

    def improve_rigid_docking(self, protein, ligand, args):
        # Skip redundant pocket detection if already performed
        if hasattr(protein, '_pockets_detected') and protein._pockets_detected:
            print("Using previously detected pockets")
        else:
            if not protein.active_site:
                if hasattr(args, 'site') and args.site:
                    radius = max(getattr(args, 'radius', 15.0), 12.0)
                    protein.define_active_site(args.site, radius)
                elif hasattr(args, 'detect_pockets') and args.detect_pockets:
                    pockets = protein.detect_pockets()
                    if pockets:
                        radius = max(pockets[0]['radius'], 15.0)
                        protein.define_active_site(pockets[0]['center'], radius)
                    else:
                        center = np.mean(protein.xyz, axis=0)
                        protein.define_active_site(center, 15.0)
                else:
                    center = np.mean(protein.xyz, axis=0)
                    protein.define_active_site(center, 15.0)
            protein._pockets_detected = True  # Mark as detected now

        center = protein.active_site['center']
        radius = protein.active_site['radius']

        n_initial_random = min(self.max_iterations // 2, 2000)
        poses = []
        max_attempts = n_initial_random * 5
        attempts = 0

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

        print(f"Generated {len(poses)} poses (from {attempts} attempts). Now scoring...")

        def score_pose(pose):
            if detect_steric_clash(protein.atoms, pose.atoms):
                return (pose, float('inf'))
            return (pose, self.scoring_function.score(protein, pose))

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(score_pose, poses)

        results = [r for r in results if np.isfinite(r[1])]
        if not results:
            print("[ERROR] All poses clashed or failed scoring.")
            return []

        results.sort(key=lambda x: x[1])
        print(f"Best docking score: {results[0][1]:.2f}")

        if getattr(args, 'local_opt', False):
            print("Applying local optimization to top poses...")
            top = results[:10]
            optimized = []
            for pose, _ in top:
                opt_pose, opt_score = self._enhanced_local_optimization(
                    protein, pose, step_size=0.3, angle_step=0.05, max_steps=15)
                optimized.append((opt_pose, opt_score))
            optimized.sort(key=lambda x: x[1])
            return optimized

        return results
    


# ------------------------------------------------------------------------------
# Parallel Genetic Algorithm
# ------------------------------------------------------------------------------

class ParallelGeneticAlgorithm(GeneticAlgorithm):
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
                    print(f"Attempt {attempts}: Pose is outside grid, trying again...")
                continue  # Skip this pose if it's outside the grid
                
            # Check clash validity with pose relaxation
            if not self._check_pose_validity(pose, protein):
                try:
                    # Try to relax the pose to avoid clashes
                    relaxed_pose = self._gentle_clash_relief(protein, pose, max_steps=5)
                    if self._check_pose_validity(relaxed_pose, protein):
                        population.append((relaxed_pose, None))
                        if len(population) % 10 == 0:
                            print(f"Generated {len(population)}/{self.population_size} valid poses (after relaxation)")
                    else:
                        if attempts % 100 == 0:
                            print(f"Attempt {attempts}: Clash could not be resolved, trying again...")
                except Exception as e:
                    if attempts % 100 == 0:
                        print(f"Error during clash resolution: {e}")
                continue
            
            # If we reach here, the pose is valid
            population.append((pose, None))
            if len(population) % 10 == 0:
                print(f"Generated {len(population)}/{self.population_size} valid poses")
        
        if len(population) < self.population_size:
            print(f"Warning: Could only generate {len(population)}/{self.population_size} valid poses after {attempts} attempts")
        else:
            print(f"Successfully generated {len(population)} valid poses (took {attempts} attempts)")
        
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
            print(f"Using batch scoring for {len(poses)} poses...")
            scores = self.scoring_function.score_batch(protein, poses)
            
            # Combine poses with scores
            results = [(copy.deepcopy(pose), score) for pose, score in zip(poses, scores)]
        else:
            # Fall back to sequential scoring
            print("Batch scoring not available, using sequential scoring...")
            results = []
            for i, (pose, _) in enumerate(population):
                score = self.scoring_function.score(protein, pose)
                results.append((copy.deepcopy(pose), score))
                
                # Show progress for large populations
                if i % 10 == 0 and i > 0 and len(population) > 50:
                    print(f"  Evaluating pose {i}/{len(population)}...")
        
        return results
    # def _evaluate_population_batched(self, protein, population, batch_size=8):
    #     """
    #     Evaluate population in batches for improved GPU efficiency.
        
    #     Parameters:
    #     -----------
    #     protein : Protein
    #         Protein object
    #     population : list
    #         List of (pose, score) tuples
    #     batch_size : int
    #         Batch size for parallel evaluation
        
    #     Returns:
    #     --------
    #     list
    #         Evaluated population as (pose, score) tuples
    #     """
    #     results = []
        
    #     # Create batches
    #     batches = [population[i:i + batch_size] for i in range(0, len(population), batch_size)]
        
    #     print(f"Evaluating {len(population)} poses in {len(batches)} batches of size {batch_size}")
        
    #     # Process each batch
    #     for batch_idx, batch in enumerate(batches):
    #         batch_results = []
            
    #         # Check if we can use GPU batch processing
    #         if (hasattr(self.scoring_function, 'torch_available') and 
    #             self.scoring_function.torch_available and 
    #             hasattr(self.scoring_function, 'score_batch')):
    #             # Process the batch in parallel on GPU using a score_batch method
    #             # Note: You would need to implement score_batch in your scoring function
    #             batch_poses = [pose for pose, _ in batch]
    #             batch_scores = self.scoring_function.score_batch(protein, batch_poses)
    #             batch_results = [(copy.deepcopy(pose), score) for (pose, _), score in zip(batch, batch_scores)]
                    
    #         else:
    #             # If no GPU batch method available, process with CPU parallelism
    #             if self.n_processes > 1:
    #                 # Use process pool for CPU parallelism
    #                 def score_pose(pose_tuple):
    #                     pose, _ = pose_tuple
    #                     score = self.scoring_function.score(protein, pose)
    #                     return (copy.deepcopy(pose), score)
                    
    #                 with mp.Pool(processes=min(self.n_processes, batch_size)) as pool:
    #                     batch_results = pool.map(score_pose, batch)
    #             else:
    #                 # Sequential processing
    #                 for i, (pose, _) in enumerate(batch):
    #                     score = self.scoring_function.score(protein, pose)
    #                     batch_results.append((copy.deepcopy(pose), score))
            
    #         # Add results from this batch
    #         results.extend(batch_results)
            
    #         # Print progress
    #         if batch_idx % max(1, len(batches)//10) == 0:
    #             print(f"  Evaluated batch {batch_idx+1}/{len(batches)} ({len(results)}/{len(population)} poses)")
        
    #     return results

    # def initialize_population(self, protein, ligand):
    #     """
    #     Initialize random population for genetic algorithm within spherical grid.

    #     Parameters:
    #     -----------
    #     protein : Protein
    #         Protein object
    #     ligand : Ligand object

    #     Returns:
    #     --------
    #     list
    #         List of (pose, score) tuples
    #     """
    #     population = []

    #     for _ in range(self.population_size):
    #         pose = self._generate_valid_pose(protein, ligand, 
    #                                         protein.active_site['center'], 
    #                                         protein.active_site['radius'])
    #     # Determine search space
    #     if protein.active_site:
    #         center = protein.active_site['center']
    #         radius = protein.active_site['radius']
    #     else:
    #         center = np.mean(protein.xyz, axis=0)
    #         radius = 15.0  # Arbitrary default

    #     self.initialize_grid_points(center, protein=protein)

    #     print(f"Using {self.n_processes} CPU cores for evaluation")
    #     print(f"Using {self.batch_size} poses per process for evaluation")
    #     print(f"Using {self.population_size} poses in total")
    #     print(f"Using {self.mutation_rate} mutation rate")
    #     print(f"Using {self.crossover_rate} crossover rate")
    #     print(f"Using {self.tournament_size} tournament size")
    #     print(f"Performing local optimization: {self.perform_local_opt}")
    #     print(f"Grid spacing: {self.grid_spacing}")
    #     print(f"Grid radius: {self.grid_radius}")

    #     # # Generate initial population
    #     # for _ in range(self.population_size):
    #     #     pose = copy.deepcopy(ligand)

    #         # Add retry logic here
    #     attempts = 0
    #     max_attempts = self.population_size * 10
        
    #     # Generate initial population with retry logic
    #     while len(population) < self.population_size and attempts < max_attempts:
    #         attempts += 1
            
    #         pose = copy.deepcopy(ligand)

    #         # Select a random point from precomputed spherical grid
    #         random_grid_point = random.choice(self.grid_points)

    #         # Move the ligand centroid to that random point
    #         centroid = np.mean(pose.xyz, axis=0)
    #         translation = random_grid_point - centroid
    #         pose.translate(translation)

    #         # Apply random rotation with bias toward center of pocket
    #         rotation = Rotation.random()
    #         centroid = np.mean(pose.xyz, axis=0)
    #         vector_to_center = center - centroid
    #         vector_to_center /= np.linalg.norm(vector_to_center)

    #         # Small rotation (~10 degrees) toward pocket center
    #         bias_rotation = Rotation.from_rotvec(0.2 * vector_to_center)  # 0.2 rad ≈ 11 degrees
    #         biased_rotation = rotation * bias_rotation
    #         rotation_matrix = biased_rotation.as_matrix()

    #         # Apply rotation
    #         pose.translate(-centroid)
    #         pose.rotate(rotation_matrix)
    #         pose.translate(centroid)

    #         # Add random translation
    #         translation_vector = np.random.normal(1.5, 0.5, size=3)
    #         pose.translate(translation_vector)
            
    #         # Filters for valid poses
    #         if not self._check_pose_validity(pose, protein):
    #             relaxed_pose = self._gentle_clash_relief(protein, pose, max_steps=5)
    #             if self._check_pose_validity(relaxed_pose, protein):
    #                 population.append((relaxed_pose, None))
    #             continue

    #         if not is_within_grid(pose, center, radius):
    #             continue  # Skip this pose if it's outside the grid
                
    #         # If the pose is valid, add it to the population
    #         population.append((pose, None))

    #     return population
    
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
        from .utils import generate_spherical_grid
        all_grid_points = generate_spherical_grid(center, radius, spacing)
        
        # Create KD-tree for efficient proximity search
        from scipy.spatial import cKDTree
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
        
        print(f"Generated {len(valid_grid_points)} valid grid points out of {len(all_grid_points)} total")
        
        # If too few valid points, expand grid or relax constraints
        if len(valid_grid_points) < 100:
            print("Warning: Few valid grid points. Expanding search...")
            return self.initialize_smart_grid(protein, center, radius * 1.2, spacing, margin * 0.9)
        
        return np.array(valid_grid_points)
    def initialize_grid_points(self, center, protein=None):
        """
        Initialize grid points for search space sampling.
        
        Parameters:
        -----------
        center : array-like
            Center coordinates
        protein : Protein, optional
            Protein object for pocket detection
        """
        if self.grid_points is None:
            self.grid_points = []

            pocket_centers = []

            if protein is not None and hasattr(protein, 'detect_pockets'):
                pockets = protein.detect_pockets()
                if pockets:
                    self.logger.info(f"[BLIND] Detected {len(pockets)} binding pockets")
                    pocket_centers = [p['center'] for p in pockets]

            if pocket_centers:
                for idx, c in enumerate(pocket_centers):
                    local_grid = generate_spherical_grid(
                        center=c,
                        radius=self.grid_radius,
                        spacing=self.grid_spacing
                    )
                    self.grid_points.extend(local_grid)
                    self.logger.info(f"  -> Grid {idx+1}: {len(local_grid)} points at {c}")
            else:
                # Fallback to full-protein blind grid
                self.logger.warning("[BLIND] No pockets detected, generating full-protein grid")

                coords = np.array([atom['coords'] for atom in protein.atoms])
                min_corner = np.min(coords, axis=0) - 2.0  # small padding
                max_corner = np.max(coords, axis=0) + 2.0

                self.grid_points = generate_cartesian_grid(min_corner, max_corner, spacing=self.grid_spacing)
                self.logger.info(f"[BLIND] Generated {len(self.grid_points)} grid points covering entire protein")

            self.logger.info(f"Initialized total grid with {len(self.grid_points)} points "
                            f"(spacing: {self.grid_spacing}, radius: {self.grid_radius})")

            # # Save Light Sphere PDB (subsample)
            # subsample_rate = 20
            # if self.output_dir is not None:
            #     sphere_path = Path(self.output_dir) / "sphere.pdb"
            #     sphere_path.parent.mkdir(parents=True, exist_ok=True)
            #     with open(sphere_path, 'w') as f:
            #         for idx, point in enumerate(self.grid_points):
            #             if idx % subsample_rate == 0:
            #                 f.write(
            #                     f"HETATM{idx+1:5d} {'S':<2s}   SPH A   1    "
            #                     f"{point[0]:8.3f}{point[1]:8.3f}{point[2]:8.3f}  1.00  0.00          S\n"
            #                 )
            #     self.logger.info(f"Sphere grid written to {sphere_path} (subsampled every {subsample_rate} points)")

            # Save grid visualization if output directory exists
        subsample_rate = 20
        if self.output_dir is not None:
            sphere_path = Path(self.output_dir) / "sphere.pdb"
            sphere_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(sphere_path, 'w') as f:
                for idx, point in enumerate(self.grid_points):
                    if idx % subsample_rate == 0:
                        f.write(
                            f"HETATM{idx+1:5d} {'S':<2s}   SPH A   1    "
                            f"{point[0]:8.3f}{point[1]:8.3f}{point[2]:8.3f}  1.00  0.00          S\n"
                        )
            
            # Add check before logging
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.info(f"Sphere grid written to {sphere_path} (subsampled every {subsample_rate} points)")
            else:
                print(f"Sphere grid written to {sphere_path} (subsampled every {subsample_rate} points)")

    
    def _adjust_search_radius(self, initial_radius, generation, total_generations):
        """
        Shrink the search radius over generations (parallel version).
        
        Parameters:
        -----------
        initial_radius : float
            Initial search radius
        generation : int
            Current generation
        total_generations : int
            Total number of generations
            
        Returns:
        --------
        float
            Adjusted radius for the current generation
        """
        decay_rate = 0.5  # You can tune it (0.3 or 0.7)
        factor = 1.0 - (generation / total_generations) * decay_rate
        return max(initial_radius * factor, initial_radius * 0.5)

    def _generate_orientations(self, ligand, protein):
        orientations = []

        # Get active site center and radius
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            center = np.mean(protein.xyz, axis=0)
            radius = 15.0

        bad_orientations = 0
        max_bad_orientations = self.num_orientations * 10

        while len(orientations) < self.num_orientations and bad_orientations < max_bad_orientations:
            pose = copy.deepcopy(ligand)
            pose.random_rotate()

            sampled_point = random_point_in_sphere(center, radius)
            pose.translate(sampled_point)

            if is_inside_sphere(pose, center, radius):
                if self._check_pose_validity(pose, protein):
                    orientations.append(pose)
                else:
                    bad_orientations += 1
            else:
                bad_orientations += 1

        return orientations


    def _check_pose_validity(self, ligand, protein, clash_threshold=1.5):
        """
        Check if ligand pose clashes with protein atoms.
        
        Parameters:
        -----------
        ligand : Ligand
            Ligand to check
        protein : Protein
            Protein object
        clash_threshold : float
            Distance threshold for clash detection (Å)
            
        Returns:
        --------
        bool
            True if pose is valid (no severe clash), False otherwise
        """
        # ligand_coords = np.array([atom['coords'] for atom in ligand.atoms])
        
        # # Use active site atoms if defined
        # if hasattr(protein, 'active_site') and protein.active_site and 'atoms' in protein.active_site:
        #     protein_coords = np.array([atom['coords'] for atom in protein.active_site['atoms']])
        # else:
        #     protein_coords = np.array([atom['coords'] for atom in protein.atoms])
        
        # for lig_coord in ligand_coords:
        #     distances = np.linalg.norm(protein_coords - lig_coord, axis=1)
        #     if np.any(distances < clash_threshold):
        #         return False  # Clash detected
        
        # return True
        is_clashing, _ = self._enhanced_clash_detection(protein, ligand)
        return not is_clashing

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


        # Initialize population with clash-free poses
        population = []
        print("Generating initial clash-free population...")
        for i in range(self.population_size):
            if i % 10 == 0:
                print(f"  {i}/{self.population_size} poses generated")
            
            pose = self._generate_valid_pose(protein, ligand, center, radius)
            score = self.scoring_function.score(protein, pose)
            population.append((pose, score))
    
        # Sort population by score
        population.sort(key=lambda x: x[1])


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
            print(f"[INFO] Added {len(protein.active_site['atoms'])} atoms into active_site region")

        print(f"Searching around center {center} with radius {radius}")
        
        # Initialize population
        population = self.initialize_population(protein, ligand)
        
        # Evaluate initial population
        evaluated_population = self._evaluate_population(protein, population)
        if not evaluated_population:
            print("[Error] No valid poses found during population initialization.")
            return []
        # Sort population by score
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
                   # self._mutate(child1, copy.deepcopy(parent1), center, current_radius)
                    self._mutate(individual=child1, original_individual=copy.deepcopy(parent1), center=center, radius=current_radius)
                   # self._mutate(child2, copy.deepcopy(parent2), center, current_radius)
                    self._mutate(individual=child2, original_individual=copy.deepcopy(parent2), center=center, radius=current_radius)

                    offspring.append((child1, None))
                    offspring.append((child2, None))
            
            # Evaluate offspring
            eval_start = time.time()
            evaluated_offspring = self._evaluate_population(protein, offspring)
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
            
            # Display progress
            gen_time = time.time() - gen_start
            print(f"Generation {generation + 1}/{self.max_iterations}: "
                  f"Best score = {self.best_score:.4f}, "
                  f"Current best = {evaluated_population[0][1]:.4f}, "
                  f"Time = {gen_time:.2f}s")
            
            # Apply local search to the best individual occasionally
            if hasattr(self, '_local_optimization') and generation % 5 == 0:
                best_pose, best_score = self._local_optimization(evaluated_population[0][0], protein)
                
                if best_score < self.best_score:
                    self.best_pose = best_pose
                    self.best_score = best_score
                    
                    # Replace best individual in population
                    evaluated_population[0] = (best_pose, best_score)
                    evaluated_population.sort(key=lambda x: x[1])
                    all_individuals.append((best_pose, best_score))
        
        # Return unique solutions, best first
        self.total_time = time.time() - start_time
        print(f"\nSearch completed in {self.total_time:.2f} seconds")
        print(f"Evaluation time: {self.eval_time:.2f} seconds ({self.eval_time/self.total_time*100:.1f}%)")
        
        # Sort all_individuals by score and ensure uniqueness
        all_individuals.sort(key=lambda x: x[1])
        
        # Return results
        return all_individuals
    

    # def _evaluate_population(self, protein, population):
    #     """
    #     Evaluate population using batch processing for improved GPU efficiency.
        
    #     Parameters:
    #     -----------
    #     protein : Protein
    #         Protein object
    #     population : list
    #         List of (pose, score) tuples
        
    #     Returns:
    #     --------
    #     list
    #         Evaluated population as (pose, score) tuples
    #     """
    #     # Extract poses from population
    #     poses = [pose for pose, _ in population]
        
    #     # Batch scoring with the scoring function
    #     if hasattr(self.scoring_function, 'score_batch'):
    #         print(f"Using batch scoring for {len(poses)} poses...")
    #         scores = self.scoring_function.score_batch(protein, poses)
            
    #         # Combine poses with scores
    #         results = [(copy.deepcopy(pose), score) for pose, score in zip(poses, scores)]
    #     else:
    #         # Fall back to sequential scoring
    #         print("Batch scoring not available, using sequential scoring...")
    #         results = []
    #         for i, (pose, _) in enumerate(population):
    #             score = self.scoring_function.score(protein, pose)
    #             results.append((copy.deepcopy(pose), score))
                
    #             # Show progress for large populations
    #             if i % 10 == 0 and i > 0 and len(population) > 50:
    #                 print(f"  Evaluating pose {i}/{len(population)}...")
        
    #     return results
    
    # def _selection(self, population):
    #     """
    #     Tournament selection of parents.
        
    #     Parameters:
    #     -----------
    #     population : list
    #         List of (pose, score) tuples
        
    #     Returns:
    #     --------
    #     list
    #         Selected parents as (pose, score) tuples
    #     """
    #     selected = []
        
    #     for _ in range(self.population_size):
    #         # Select random individuals for tournament
    #         tournament = random.sample(population, min(self.tournament_size, len(population)))
            
    #         # Select the best from tournament
    #         tournament.sort(key=lambda x: x[1])
    #         selected.append(tournament[0])
        
    #     return selected
    # Increase population diversity

    def _selection(self, population):
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
        
        child2.translate(-centroid2)
        child2.rotate(interpolated_rotation.as_matrix())
        child2.translate(centroid2)
        
        # Validate children
        if not self._validate_conformation(child1):
            #print("Child1 failed validation. Attempting repair...")
            child1 = self._repair_conformation(child1)
        
        if not is_inside_sphere(child1, center, radius):
            centroid = np.mean(child1.xyz, axis=0)
            to_center = center - centroid
            dist = np.linalg.norm(to_center)
            if dist > 0:
                move_vector = to_center * (dist - radius*0.9)/dist
                child1.translate(move_vector)
            
        if not self._validate_conformation(child2):
            #print("Child2 failed validation. Attempting repair...")
            child2 = self._repair_conformation(child2)
        
        if not is_inside_sphere(child2, center, radius):
            centroid = np.mean(child2.xyz, axis=0)
            to_center = center - centroid
            dist = np.linalg.norm(to_center)
            if dist > 0:
                move_vector = to_center * (dist - radius*0.9)/dist
                child2.translate(move_vector)
        
        return child1, child2

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
                    #print(f"Validation failed: Overlapping atoms at indices {i} and {j} (distance: {distance:.2f} Å)")
                    return False
        
        # Check for valid bond lengths
        for bond in ligand.bonds:
            atom1 = ligand.xyz[bond['begin_atom_idx']]
            atom2 = ligand.xyz[bond['end_atom_idx']]
            bond_length = np.linalg.norm(atom1 - atom2)
            if bond_length < 0.9 or bond_length > 2.0:  # Adjusted bond length range (in Å)
                #print(f"Validation failed: Invalid bond length ({bond_length:.2f} Å) between atoms {bond['begin_atom_idx']} and {bond['end_atom_idx']}")
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
        #print("Attempting to repair ligand conformation...")
        
        for attempt in range(max_attempts):
            #print(f"Repair attempt {attempt + 1}/{max_attempts}...")
            # Apply small random perturbations to atom positions
            perturbation = np.random.normal(0, 0.2, ligand.xyz.shape)  # 0.2 Å standard deviation
            ligand.xyz += perturbation
            
            # Revalidate after perturbation
            if self._validate_conformation(ligand):
                #print("Repair successful after random perturbation.")
                return ligand
            
            # Attempt to resolve steric clashes by energy minimization
            try:
                #print("Applying energy minimization to repair ligand...")
                ligand = self._minimize_energy(ligand, max_iterations=200)
                if self._validate_conformation(ligand):
                    #print("Repair successful after energy minimization.")
                    return ligand
            except Exception as e:
                print(f"Energy minimization failed: {e}")
        
        # If repair fails, generate a new random pose
        print("Repair failed after maximum attempts. Generating a new random pose...")
        return self._generate_random_pose(ligand, np.mean(ligand.xyz, axis=0), 15.0)  # Example radius
    
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
        result = minimize(energy_function, initial_coords, method='L-BFGS-B', options={'maxiter': max_iterations})
        
        # Update ligand coordinates with minimized values
        ligand.xyz = result.x.reshape(ligand.xyz.shape)
        return ligand
    

    def _generate_random_pose(self, ligand, center, radius):
        """
        Generate a random ligand pose within a sphere.
        
        Parameters:
        -----------
        ligand : Ligand
            Ligand to position
        center : array-like
            Center coordinates
        radius : float
            Sphere radius
            
        Returns:
        --------
        Ligand
            Ligand with random position and orientation
        """
        while True:
            r = radius * random.random() ** (1.0 / 3.0)
            theta = random.uniform(0, 2 * np.pi)
            phi = random.uniform(0, np.pi)

            x = center[0] + r * np.sin(phi) * np.cos(theta)
            y = center[1] + r * np.sin(phi) * np.sin(theta)
            z = center[2] + r * np.cos(phi)

            pose = copy.deepcopy(ligand)
            centroid = np.mean(pose.xyz, axis=0)
            translation = np.array([x, y, z]) - centroid
            pose.translate(translation)

            if is_within_grid(pose, center, radius):
                return pose

    def _generate_valid_pose(self, protein, ligand, center, radius, max_attempts=50):
        """
        Generate a valid, clash-free pose within the grid.
        
        Parameters:
            protein: Protein object
            ligand: Ligand object
            center: Search center
            radius: Search radius
            max_attempts: Maximum attempts before giving up
            
        Returns:
            Ligand pose
        """
        for attempt in range(max_attempts):
            # Create a fresh copy
            pose = copy.deepcopy(ligand)
            
            # Choose a random valid grid point
            if not hasattr(self, 'smart_grid_points') or self.smart_grid_points is None:
                self.smart_grid_points = self.initialize_smart_grid(protein, center, radius)
            
            random_point = random.choice(self.smart_grid_points)
            
            # Move ligand centroid to this point
            centroid = np.mean(pose.xyz, axis=0)
            translation = random_point - centroid
            pose.translate(translation)
            
            # Apply random rotation with bias toward pocket
            rotation = Rotation.random()
            
            # Apply rotation around centroid
            centroid = np.mean(pose.xyz, axis=0)
            pose.translate(-centroid)
            pose.rotate(rotation.as_matrix())
            pose.translate(centroid)
            
            # Check for clashes
            is_clashing, clash_score = self._enhanced_clash_detection(protein, pose)
            
            if not is_clashing:
                return pose
            
            # Apply gentle clash relief if close to valid
            if clash_score < 0.5 and attempt > max_attempts // 2:
                relaxed_pose = self._gentle_clash_relief(protein, pose, max_steps=5, max_movement=0.3)
                is_clashing, clash_score = self._enhanced_clash_detection(protein, relaxed_pose)
                if not is_clashing:
                    return relaxed_pose
        
        # If we reach here, couldn't find a valid pose in max_attempts
        # Fall back to least-clashing pose with extra relaxation
        print("Warning: Couldn't generate clash-free pose. Using relaxed pose with minimal clashes.")
        return self._generate_minimal_clash_pose(protein, ligand, center, radius)
    ##############
    # Mutation
    ##############

    def _mutate(self, individual, original_individual, protein, center, radius):
        """
        Mutate an individual with probability mutation_rate and respect current radius.
        """

        if random.random() >= self.mutation_rate:
            return  # No mutation

        # Perform either translation, rotation, or both
        mutation_type = random.choice(['translation', 'rotation', 'both'])

        if mutation_type in ['translation', 'both']:
            translation = np.random.normal(0, 2.0, 3)  # 2.0 Å standard deviation
            individual.translate(translation)

        if mutation_type in ['rotation', 'both']:
            angle = np.random.normal(0, 0.5)  # ~30 degrees std dev
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)

            rotation = Rotation.from_rotvec(angle * axis)
            centroid = np.mean(individual.xyz, axis=0)
            individual.translate(-centroid)
            individual.rotate(rotation.as_matrix())
            individual.translate(centroid)
        
        if not is_inside_sphere(individual, center, radius):
            # Calculate centroid-to-center vector
            centroid = np.mean(individual.xyz, axis=0)
            to_center = center - centroid
            # Scale vector to move back to sphere boundary
            dist = np.linalg.norm(to_center)
            if dist > 0:
                move_vector = to_center * (dist - radius*0.9)/dist
                individual.translate(move_vector)

        if not is_within_grid(individual, center, radius):
            # If out of bounds, revert to original
            individual.xyz = original_individual.xyz.copy()
            original_xyz = individual.xyz.copy()

        # Verify no clashes were introduced
        if not self._check_pose_validity(individual, protein):
            # Revert to original if mutation caused clashes
            individual.xyz = original_xyz

        return individual


    def _local_optimization(self, pose, protein):
        """
        Perform local optimization of pose using gradient descent with clash detection.
        
        Parameters:
        -----------
        pose : Ligand
            Ligand pose to optimize
        protein : Protein
            Protein target
        
        Returns:
        --------
        tuple
            (optimized_pose, optimized_score)
        """
        return super()._local_optimization(pose, protein)   


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
        import numpy as np
        
        # Make a copy of the pose to avoid modifying the original
        adjusted_pose = copy.deepcopy(pose)
        
        # Identify clashing atoms
        clashing_atoms = []
        
        # Get VDW radii (can use the ones from scoring function if available)
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
        import numpy as np
        from scipy.spatial.transform import Rotation
        import random
        
        best_pose = None
        best_clash_score = float('inf')
        
        # Get VDW radii for clash detection
        vdw_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }
        
        # Use active site atoms if defined
        if hasattr(protein, 'active_site') and protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Helper function to calculate clash score
        def calculate_clash_score(pose):
            clash_score = 0.0
            for lig_atom in pose.atoms:
                lig_coords = lig_atom['coords']
                lig_symbol = lig_atom.get('symbol', 'C')
                lig_radius = vdw_radii.get(lig_symbol, 1.7)
                
                for prot_atom in protein_atoms:
                    prot_coords = prot_atom['coords']
                    prot_symbol = prot_atom.get('element', prot_atom.get('name', 'C'))[0]
                    prot_radius = vdw_radii.get(prot_symbol, 1.7)
                    
                    distance = np.linalg.norm(lig_coords - prot_coords)
                    min_allowed = (lig_radius + prot_radius) * 0.7  # 70% of sum of vdW radii
                    
                    if distance < min_allowed:
                        overlap = min_allowed - distance
                        clash_score += overlap * overlap  # Square to penalize severe clashes more
            
            return clash_score
        
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
                print(f"  Generated {attempt}/{max_attempts} poses, best clash score: {best_clash_score:.2f}")
        
        # If we found a pose with acceptable clash score, try to improve it with gentle adjustments
        if best_pose is not None and hasattr(self, '_adjust_clashing_atoms'):
            print(f"  Found pose with clash score {best_clash_score:.2f}, applying adjustments...")
            best_pose = self._adjust_clashing_atoms(protein, best_pose)
        
        print(f"  Generated pose with final clash score: {calculate_clash_score(best_pose):.2f}")
        return best_pose
    
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
            if self.grid_points is not None and len(self.grid_points) > 0:
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
                        print(f"Warning: Error during clash relief: {e}")
                        return pose
                else:
                    return pose
        
        # If we couldn't generate a valid pose with random sampling, 
        # try using a more specialized method
        print("  Random pose generation failed, trying specialized minimal clash generation...")
        return self._generate_minimal_clash_pose(protein, ligand, center, radius)
    def _gentle_clash_relief(self, protein, pose, reference=None, max_steps=20, max_movement=0.3):
        """
        Gently move atoms to relieve clashes while preserving overall pose.
        
        Parameters:
            protein: Protein object
            pose: Ligand pose
            reference: Reference pose (optional)
            max_steps: Maximum optimization steps
            max_movement: Maximum allowed movement
            
        Returns:
            Improved ligand pose
        """
        import copy
        
        # Make work copy
        working_pose = copy.deepcopy(pose)
        current_score = self.scoring_function.score(protein, working_pose)
        best_pose = copy.deepcopy(working_pose)
        best_score = current_score
        
        # Find clashing atoms
        clashing_atoms = []
        is_clashing, _ = self._enhanced_clash_detection(protein, working_pose)
        
        if not is_clashing:
            return working_pose
        
        # Identify protein atoms in the vicinity of each ligand atom
        protein_atoms = protein.active_site['atoms'] if hasattr(protein, 'active_site') and 'atoms' in protein.active_site else protein.atoms
        protein_coords = np.array([atom['coords'] for atom in protein_atoms])
        
        # For each step
        for step in range(max_steps):
            improved = False
            
            # Try small translations in all directions
            for direction in [
                [0.1, 0, 0], [-0.1, 0, 0],
                [0, 0.1, 0], [0, -0.1, 0],
                [0, 0, 0.1], [0, 0, -0.1],
                [0.07, 0.07, 0.07], [-0.07, -0.07, -0.07]
            ]:
                # Create test pose
                test_pose = copy.deepcopy(working_pose)
                test_pose.translate(np.array(direction))
                
                # Check if translated pose is valid
                is_clashing, clash_score = self._enhanced_clash_detection(protein, test_pose)
                
                if not is_clashing:
                    # Score the clash-free pose
                    test_score = self.scoring_function.score(protein, test_pose)
                    
                    if test_score < best_score:
                        best_pose = copy.deepcopy(test_pose)
                        best_score = test_score
                        improved = True
                        break
            
            # If translations didn't help, try small rotations
            if not improved:
                axes = [[1,0,0], [0,1,0], [0,0,1]]
                angles = [0.05, -0.05, 0.1, -0.1]
                
                for axis in axes:
                    for angle in angles:
                        # Create test pose
                        test_pose = copy.deepcopy(working_pose)
                        
                        # Apply rotation
                        rotation = Rotation.from_rotvec(np.array(axis) * angle)
                        centroid = np.mean(test_pose.xyz, axis=0)
                        test_pose.translate(-centroid)
                        test_pose.rotate(rotation.as_matrix())
                        test_pose.translate(centroid)
                        
                        # Check validity
                        is_clashing, clash_score = self._enhanced_clash_detection(protein, test_pose)
                        
                        if not is_clashing:
                            test_score = self.scoring_function.score(protein, test_pose)
                            
                            if test_score < best_score:
                                best_pose = copy.deepcopy(test_pose)
                                best_score = test_score
                                improved = True
                                break
                    
                    if improved:
                        break
            
            # If still not improved, try per-atom movements
            if not improved and step > max_steps // 2:
                test_pose = self._adjust_clashing_atoms(protein, working_pose)
                is_clashing, clash_score = self._enhanced_clash_detection(protein, test_pose)
                
                if not is_clashing or clash_score < 0.1:
                    test_score = self.scoring_function.score(protein, test_pose)
                    
                    if test_score < best_score * 1.5:  # Allow for some score degradation to fix clashes
                        best_pose = copy.deepcopy(test_pose)
                        best_score = test_score
                        improved = True
            
            # Update current pose if improved
            if improved:
                working_pose = copy.deepcopy(best_pose)
            else:
                # If no improvement and clashes persist, move ligand away from protein slightly
                test_pose = copy.deepcopy(working_pose)
                
                # Calculate vector from protein center to ligand center
                protein_center = np.mean(protein_coords, axis=0)
                ligand_center = np.mean(test_pose.xyz, axis=0)
                
                # Create unit vector pointing away from protein
                away_vector = ligand_center - protein_center
                if np.linalg.norm(away_vector) > 0:
                    away_vector = away_vector / np.linalg.norm(away_vector)
                    
                    # Move slightly away
                    test_pose.translate(away_vector * 0.2)
                    
                    # Check if this helped
                    is_clashing, clash_score = self._enhanced_clash_detection(protein, test_pose)
                    
                    if not is_clashing or clash_score < 0.1:
                        working_pose = copy.deepcopy(test_pose)
                        improved = True
        
        # Final check
        is_clashing, clash_score = self._enhanced_clash_detection(protein, best_pose)
        
        return best_pose
    
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
        # Get VDW radii
        vdw_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }
        
        clash_score = 0.0
        max_overlap = 0.0
        clashing_atoms = 0
        
        # Get protein atoms in active site
        if hasattr(protein, 'active_site') and protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Check each ligand atom against all protein atoms
        for lig_atom in ligand.atoms:
            l_coords = lig_atom['coords']
            l_symbol = lig_atom.get('symbol', 'C')
            l_radius = vdw_radii.get(l_symbol, 1.7)
            
            for p_atom in protein_atoms:
                p_coords = p_atom['coords']
                p_name = p_atom.get('name', '')
                p_symbol = p_atom.get('element', p_atom.get('name', 'C'))[0]
                p_radius = vdw_radii.get(p_symbol, 1.7)
                
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
# ------------------------------------------------------------------------------
# Parallel Random Search
# ------------------------------------------------------------------------------


class ParallelRandomSearch(RandomSearch):
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

        # Grid parameters
        self.grid_points = None
        self.grid_radius = grid_radius
        self.grid_spacing = grid_spacing
        self.grid_center = grid_center

        # Set up logging - always create a logger, but it might be a null logger
        if output_dir:
            self.logger = setup_logging(output_dir)
        else:
            import logging
            self.logger = logging.getLogger("null_logger")
            self.logger.addHandler(logging.NullHandler())
        
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
        from .utils import generate_spherical_grid
        all_grid_points = generate_spherical_grid(center, radius, spacing)
        
        # Create KD-tree for efficient proximity search
        from scipy.spatial import cKDTree
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
        
        print(f"Generated {len(valid_grid_points)} valid grid points out of {len(all_grid_points)} total")
        
        # If too few valid points, expand grid or relax constraints
        if len(valid_grid_points) < 100:
            print("Warning: Few valid grid points. Expanding search...")
            return self.initialize_smart_grid(protein, center, radius * 1.2, spacing, margin * 0.9)
        
        return np.array(valid_grid_points)
    def initialize_grid_points(self, center, protein=None):
        from .utils import generate_spherical_grid

        if self.grid_points is None:
            self.grid_points = []

            pocket_centers = []

            if protein is not None and hasattr(protein, 'detect_pockets'):
                pockets = protein.detect_pockets()
                if pockets:
                    print(f"[BLIND] Detected {len(pockets)} binding pockets")
                    pocket_centers = [p['center'] for p in pockets]

            if self.grid_center is not None:
                pocket_centers = [self.grid_center]
            elif not pocket_centers:
                pocket_centers = [center]


            for idx, c in enumerate(pocket_centers):
                local_grid = generate_spherical_grid(
                    center=c,
                    radius=self.grid_radius,
                    spacing=self.grid_spacing
                )
                self.grid_points.extend(local_grid)
                print(f"  -> Grid {idx+1}: {len(local_grid)} points at {c}")

            print(f"Initialized total grid with {len(self.grid_points)} points "
                  f"(spacing: {self.grid_spacing}, radius: {self.grid_radius})")

            # Save Light Sphere PDB (subsample)
            subsample_rate = 20
            if self.output_dir is not None:
                sphere_path = Path(self.output_dir) / "sphere.pdb"
                sphere_path.parent.mkdir(parents=True, exist_ok=True)
                with open(sphere_path, 'w') as f:
                    for idx, point in enumerate(self.grid_points):
                        if idx % subsample_rate == 0:
                            f.write(
                                f"HETATM{idx+1:5d} {'S':<2s}   SPH A   1    "
                                f"{point[0]:8.3f}{point[1]:8.3f}{point[2]:8.3f}  1.00  0.00          S\n"
                            )
                self.logger.info(f"Sphere grid written to {sphere_path} (subsampled every {subsample_rate} points)")


    def _adjust_search_radius(self, initial_radius, iteration, total_iterations):
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
            
        Returns:
        --------
        float
            Adjusted radius for the current iteration
        """
        decay_rate = 0.5
        factor = 1.0 - (iteration / total_iterations) * decay_rate
        return max(initial_radius * factor, initial_radius * 0.5)

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
        # self.center = center
        # self.radius = radius
        # Initialize population with clash-free poses
        # population = []
        # print("Generating initial clash-free population...")
        # for i in range(self.population_size):
        #     if i % 10 == 0:
        #         print(f"  {i}/{self.population_size} poses generated")
            
        #     pose = self._generate_valid_pose(protein, ligand, center, radius)
        #     score = self.scoring_function.score(protein, pose)
        #     population.append((pose, score))
        
        # # Sort population by score
        # population.sort(key=lambda x: x[1])
        
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
            print(f"[INFO] Added {len(protein.active_site['atoms'])} atoms into active_site region")

        print(f"Searching around center {center} with radius {radius}")
        print(f"Using {self.n_processes} CPU cores for evaluation")

        # Save sphere grid
        self.initialize_grid_points(center, protein=protein)

        results = []
        # New variables to track clash failures
        fail_counter = 0
        max_failures = 30  # After 30 consecutive fails, expand radius
        radius_increment = 1.0  # How much to expand each time
        
        for i in range(self.max_iterations):
            if i % 25 == 0 and i > 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = avg_time * (self.max_iterations - i)
                print(f"Progress: {i}/{self.max_iterations} poses evaluated ({i/self.max_iterations*100:.1f}%) - "
                      f"Est. remaining: {remaining:.1f}s")

            # Adjust radius dynamically
            current_radius = self._adjust_search_radius(radius, i, self.max_iterations)

            pose = self._generate_random_pose(ligand, center, current_radius)
            # Apply soft outward nudge to avoid deep burial
            nudge = np.random.normal(0.2, 0.1, size=3)
            pose.translate(nudge)

            if not is_inside_sphere(ligand, center, current_radius):
                # Calculate centroid-to-center vector
                centroid = np.mean(pose.xyz, axis=0)
                to_center = center - centroid
                # Scale vector to move back to sphere boundary
                dist = np.linalg.norm(to_center)
                if dist > 0:
                    move_vector = to_center * (dist - radius*0.9)/dist
                    pose.translate(move_vector)

            # Initial clash checks
            if detect_steric_clash(protein.atoms, pose.atoms) or not self._check_pose_validity(pose, protein, center=center, radius=radius):
                fail_counter += 1
                if fail_counter >= max_failures:
                    radius += radius_increment
                    fail_counter = 0
                    print(f"⚡ Auto-expanding search radius to {radius:.2f} Å due to repeated clashes!")
                continue

            # Score valid pose and add to results
            score = self.scoring_function.score(protein, pose)

            # Final validation *before* appending
            if detect_steric_clash(protein.atoms, pose.atoms) or not self._check_pose_validity(pose, protein, center=center, radius=radius):
                fail_counter += 1
                if fail_counter >= max_failures:
                    radius += radius_increment
                    fail_counter = 0
                    print(f"⚡ Auto-expanding search radius to {radius:.2f} Å due to repeated clashes!")
                continue

            fail_counter = 0
            results.append((pose, score))

            # Double-check if the score is valid
            if not self._check_pose_validity(pose, protein):
                fail_counter += 1
                if fail_counter >= max_failures:
                    radius += radius_increment
                    fail_counter = 0
                    print(f"⚡ Auto-expanding search radius to {radius:.2f} Å due to repeated clashes!")
                continue  # Skip pose, try again

        # Optional: Refine top N poses with local optimization
        for i, (pose, score) in enumerate(results[:5]):  # Top 5 poses
            optimized_pose, optimized_score = self._local_optimization(pose, protein)
            results[i] = (optimized_pose, optimized_score)

        # Re-sort results after refinement
        results.sort(key=lambda x: x[1])

        self.total_time = time.time() - start_time

        if not results:
            print("⚠️ No valid poses generated! All poses clashed or failed. Returning empty result.")
            return []

        # Otherwise, continue
        print(f"Search completed in {self.total_time:.2f} seconds")
        print(f"Best score: {results[0][1]:.4f}")

        return results

    def _local_optimization(self, pose, protein):
        """
        Perform local optimization on a pose.
        
        Parameters:
        -----------
        pose : Ligand
            Ligand pose to optimize
        protein : Protein
            Protein object
            
        Returns:
        --------
        tuple
            (optimized_pose, optimized_score)
        """
        return local_optimize_pose(pose, protein, self.scoring_function)

    def _generate_orientations(self, ligand, protein):
        orientations = []

        # Get active site center and radius
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            center = np.mean(protein.xyz, axis=0)
            radius = 15.0

        bad_orientations = 0
        max_bad_orientations = self.num_orientations * 10

        while len(orientations) < self.num_orientations and bad_orientations < max_bad_orientations:
            pose = copy.deepcopy(ligand)
            pose.random_rotate()

            sampled_point = random_point_in_sphere(center, radius)
            pose.translate(sampled_point)

            if is_inside_sphere(pose, center, radius):
                if self._check_pose_validity(pose, protein):
                    orientations.append(pose)
                else:
                    bad_orientations += 1
            else:
                bad_orientations += 1

        return orientations


    # def _check_pose_validity(self, ligand, protein, clash_threshold=1.5):
    #     """
    #     Check if ligand pose clashes with protein atoms.
        
    #     Parameters:
    #         ligand: Ligand object with .atoms
    #         protein: Protein object with .atoms or active_site['atoms']
    #         clash_threshold: Ångström cutoff for hard clash
            
    #     Returns:
    #         bool: True if pose is valid (no severe clash), False otherwise
    #     """
    #     ligand_coords = np.array([atom['coords'] for atom in ligand.atoms])
    #     if not is_inside_sphere(ligand, self.center, self.radius):
    #         return False  # Skip this pose and generate a new one
    #     # Use active site atoms if defined
    #     if hasattr(protein, 'active_site') and protein.active_site and 'atoms' in protein.active_site:
    #         protein_coords = np.array([atom['coords'] for atom in protein.active_site['atoms']])
    #     else:
    #         protein_coords = np.array([atom['coords'] for atom in protein.atoms])
        
    #     for lig_coord in ligand_coords:
    #         distances = np.linalg.norm(protein_coords - lig_coord, axis=1)
    #         if np.any(distances < clash_threshold):
    #             return False  # Clash detected
    #     # score = self.scoring_function.score_for_pose_generation(protein, ligand)
    #     # return score < 900.0  # Valid if score is reasonable
    #     # return True
    #     is_clashing, _ = self._enhanced_clash_detection(protein, ligand)
    #     return not is_clashing

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
        
        for lig_coord in ligand_coords:
            distances = np.linalg.norm(protein_coords - lig_coord, axis=1)
            if np.any(distances < clash_threshold):
                return False  # Clash detected
        
        # Return True if passes all checks
        is_clashing, _ = self._enhanced_clash_detection(protein, ligand)
        return not is_clashing


        

    def _generate_random_pose(self, ligand, center, radius):
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
        # Get VDW radii
        vdw_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }
        
        clash_score = 0.0
        max_overlap = 0.0
        clashing_atoms = 0
        
        # Get protein atoms in active site
        if hasattr(protein, 'active_site') and protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Check each ligand atom against all protein atoms
        for lig_atom in ligand.atoms:
            l_coords = lig_atom['coords']
            l_symbol = lig_atom.get('symbol', 'C')
            l_radius = vdw_radii.get(l_symbol, 1.7)
            
            for p_atom in protein_atoms:
                p_coords = p_atom['coords']
                p_name = p_atom.get('name', '')
                p_symbol = p_atom.get('element', p_atom.get('name', 'C'))[0]
                p_radius = vdw_radii.get(p_symbol, 1.7)
                
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
    
    def _generate_valid_pose(self, protein, ligand, center, radius, max_attempts=50):
        """
        Generate a valid, clash-free pose within the grid.
        
        Parameters:
            protein: Protein object
            ligand: Ligand object
            center: Search center
            radius: Search radius
            max_attempts: Maximum attempts before giving up
            
        Returns:
            Ligand pose
        """
        for attempt in range(max_attempts):
            # Create a fresh copy
            pose = copy.deepcopy(ligand)
            
            # Choose a random valid grid point
            if not hasattr(self, 'smart_grid_points') or self.smart_grid_points is None:
                self.smart_grid_points = self.initialize_smart_grid(protein, center, radius)
            
            random_point = random.choice(self.smart_grid_points)
            
            # Move ligand centroid to this point
            centroid = np.mean(pose.xyz, axis=0)
            translation = random_point - centroid
            pose.translate(translation)
            
            # Apply random rotation with bias toward pocket
            rotation = Rotation.random()
            
            # Apply rotation around centroid
            centroid = np.mean(pose.xyz, axis=0)
            pose.translate(-centroid)
            pose.rotate(rotation.as_matrix())
            pose.translate(centroid)
            
            # Check for clashes
            is_clashing, clash_score = self._enhanced_clash_detection(protein, pose)
            
            if not is_clashing:
                return pose
            
            # Apply gentle clash relief if close to valid
            if clash_score < 0.5 and attempt > max_attempts // 2:
                relaxed_pose = self._gentle_clash_relief(protein, pose, max_steps=5, max_movement=0.3)
                is_clashing, clash_score = self._enhanced_clash_detection(protein, relaxed_pose)
                if not is_clashing:
                    return relaxed_pose
        
        # If we reach here, couldn't find a valid pose in max_attempts
        # Fall back to least-clashing pose with extra relaxation
        print("Warning: Couldn't generate clash-free pose. Using relaxed pose with minimal clashes.")
        return self._generate_minimal_clash_pose(protein, ligand, center, radius)
    

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
        import numpy as np
        
        # Make a copy of the pose to avoid modifying the original
        adjusted_pose = copy.deepcopy(pose)
        
        # Identify clashing atoms
        clashing_atoms = []
        
        # Get VDW radii (can use the ones from scoring function if available)
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
    def _gentle_clash_relief(self, protein, pose, reference=None, max_steps=20, max_movement=0.3):
        """
        Gently move atoms to relieve clashes while preserving overall pose.
        
        Parameters:
            protein: Protein object
            pose: Ligand pose
            reference: Reference pose (optional)
            max_steps: Maximum optimization steps
            max_movement: Maximum allowed movement
            
        Returns:
            Improved ligand pose
        """
        import copy
        
        # Make work copy
        working_pose = copy.deepcopy(pose)
        current_score = self.scoring_function.score(protein, working_pose)
        best_pose = copy.deepcopy(working_pose)
        best_score = current_score
        
        # Find clashing atoms
        clashing_atoms = []
        is_clashing, _ = self._enhanced_clash_detection(protein, working_pose)
        
        if not is_clashing:
            return working_pose
        
        # Identify protein atoms in the vicinity of each ligand atom
        protein_atoms = protein.active_site['atoms'] if hasattr(protein, 'active_site') and 'atoms' in protein.active_site else protein.atoms
        protein_coords = np.array([atom['coords'] for atom in protein_atoms])
        
        # For each step
        for step in range(max_steps):
            improved = False
            
            # Try small translations in all directions
            for direction in [
                [0.1, 0, 0], [-0.1, 0, 0],
                [0, 0.1, 0], [0, -0.1, 0],
                [0, 0, 0.1], [0, 0, -0.1],
                [0.07, 0.07, 0.07], [-0.07, -0.07, -0.07]
            ]:
                # Create test pose
                test_pose = copy.deepcopy(working_pose)
                test_pose.translate(np.array(direction))
                
                # Check if translated pose is valid
                is_clashing, clash_score = self._enhanced_clash_detection(protein, test_pose)
                
                if not is_clashing:
                    # Score the clash-free pose
                    test_score = self.scoring_function.score(protein, test_pose)
                    
                    if test_score < best_score:
                        best_pose = copy.deepcopy(test_pose)
                        best_score = test_score
                        improved = True
                        break
            
            # If translations didn't help, try small rotations
            if not improved:
                axes = [[1,0,0], [0,1,0], [0,0,1]]
                angles = [0.05, -0.05, 0.1, -0.1]
                
                for axis in axes:
                    for angle in angles:
                        # Create test pose
                        test_pose = copy.deepcopy(working_pose)
                        
                        # Apply rotation
                        rotation = Rotation.from_rotvec(np.array(axis) * angle)
                        centroid = np.mean(test_pose.xyz, axis=0)
                        test_pose.translate(-centroid)
                        test_pose.rotate(rotation.as_matrix())
                        test_pose.translate(centroid)
                        
                        # Check validity
                        is_clashing, clash_score = self._enhanced_clash_detection(protein, test_pose)
                        
                        if not is_clashing:
                            test_score = self.scoring_function.score(protein, test_pose)
                            
                            if test_score < best_score:
                                best_pose = copy.deepcopy(test_pose)
                                best_score = test_score
                                improved = True
                                break
                    
                    if improved:
                        break
            
            # If still not improved, try per-atom movements
            if not improved and step > max_steps // 2:
                test_pose = self._adjust_clashing_atoms(protein, working_pose)
                is_clashing, clash_score = self._enhanced_clash_detection(protein, test_pose)
                
                if not is_clashing or clash_score < 0.1:
                    test_score = self.scoring_function.score(protein, test_pose)
                    
                    if test_score < best_score * 1.5:  # Allow for some score degradation to fix clashes
                        best_pose = copy.deepcopy(test_pose)
                        best_score = test_score
                        improved = True
            
            # Update current pose if improved
            if improved:
                working_pose = copy.deepcopy(best_pose)
            else:
                # If no improvement and clashes persist, move ligand away from protein slightly
                test_pose = copy.deepcopy(working_pose)
                
                # Calculate vector from protein center to ligand center
                protein_center = np.mean(protein_coords, axis=0)
                ligand_center = np.mean(test_pose.xyz, axis=0)
                
                # Create unit vector pointing away from protein
                away_vector = ligand_center - protein_center
                if np.linalg.norm(away_vector) > 0:
                    away_vector = away_vector / np.linalg.norm(away_vector)
                    
                    # Move slightly away
                    test_pose.translate(away_vector * 0.2)
                    
                    # Check if this helped
                    is_clashing, clash_score = self._enhanced_clash_detection(protein, test_pose)
                    
                    if not is_clashing or clash_score < 0.1:
                        working_pose = copy.deepcopy(test_pose)
                        improved = True
        
        # Final check
        is_clashing, clash_score = self._enhanced_clash_detection(protein, best_pose)
        
        return best_pose