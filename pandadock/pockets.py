# Replace your OptimizedCastpDetector class in pocket_detector.py with this faster version:

import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import time

class OptimizedCastpDetector:
    """
    An optimized CASTp-inspired algorithm for protein pocket detection.
    """
    def __init__(self, probe_radius=1.4, grid_spacing=0.8, verbose=True):
        self.probe_radius = probe_radius
        self.grid_spacing = grid_spacing
        self.pockets = []
        self.verbose = verbose
        
    def detect_pockets(self, protein):
        """Detect pockets using an optimized grid-based approach with progress tracking."""
        if self.verbose:
            print("🔍 Starting optimized pocket detection...")
        
        start_time = time.time()
        
        # Step 1: Identify surface atoms to focus our search
        if self.verbose:
            print("📍 Step 1/5: Identifying surface atoms...")
        surface_atoms, surface_indices = self._get_surface_atoms(protein)
        if self.verbose:
            print(f"   ✓ Identified {len(surface_atoms)} surface atoms ({time.time()-start_time:.1f}s)")
        
        # Step 2: Create grid focused around surface atoms only
        step_start = time.time()
        if self.verbose:
            print("🌐 Step 2/5: Creating optimized grid...")
        grid_points = self._generate_focused_grid(protein, surface_atoms)
        if self.verbose:
            print(f"   ✓ Created grid with {len(grid_points)} points ({time.time()-step_start:.1f}s)")
        
        # Step 3: Identify pocket points using vectorized operations
        step_start = time.time()
        if self.verbose:
            print("🎯 Step 3/5: Identifying pocket points...")
        pocket_points, buriedness = self._identify_pocket_points_vectorized(grid_points, protein)
        if self.verbose:
            print(f"   ✓ Found {len(pocket_points)} potential pocket points ({time.time()-step_start:.1f}s)")
        
        # Step 4: Cluster pocket points using density-based clustering
        step_start = time.time()
        if self.verbose:
            print("🧩 Step 4/5: Clustering pocket points...")
        pocket_clusters = self._cluster_pocket_points(pocket_points, buriedness)
        if self.verbose:
            print(f"   ✓ Clustered into {len(pocket_clusters)} distinct pockets ({time.time()-step_start:.1f}s)")
        
        # Step 5: Calculate pocket properties
        step_start = time.time()
        if self.verbose:
            print("📊 Step 5/5: Calculating pocket properties...")
        pockets = self._calculate_pocket_properties(pocket_clusters, protein)
        if self.verbose:
            total_time = time.time() - start_time
            print(f"   ✓ Calculated properties for {len(pockets)} pockets ({time.time()-step_start:.1f}s)")
            print(f"🎉 Pocket detection completed in {total_time:.1f} seconds!")
        
        self.pockets = self._convert_to_pandadock_format(pockets)
        return self.pockets
    
    def _convert_to_pandadock_format(self, pockets):
        """Convert pocket format to match PandaDock expectations."""
        pandadock_pockets = []
        
        for pocket in pockets:
            # Convert to PandaDock expected format
            pandadock_pocket = {
                'center': pocket['center'],
                'radius': self._estimate_radius(pocket),
                'volume': pocket['volume'],
                'atoms': pocket['atoms'],
                'residues': pocket['residues'],
                'score': pocket.get('volume', 0),  # Use volume as score
                'id': pocket['id']
            }
            pandadock_pockets.append(pandadock_pocket)
        
        # Sort by volume (larger pockets first)
        pandadock_pockets.sort(key=lambda x: x['volume'], reverse=True)
        return pandadock_pockets
    
    def _estimate_radius(self, pocket):
        """Estimate pocket radius from volume."""
        volume = pocket['volume']
        # Assume spherical approximation: V = (4/3)πr³
        radius = (3 * volume / (4 * np.pi)) ** (1/3)
        return max(radius, 5.0)  # Minimum radius of 5Å
    
    def _get_surface_atoms(self, protein):
        """Identify surface atoms using a simple accessibility criterion (optimized)."""
        coords = protein.xyz
        
        # Use a more efficient method for large proteins
        if len(coords) > 2000:
            # Subsample for initial surface detection
            step = max(1, len(coords) // 2000)
            coords_sub = coords[::step]
            indices_sub = list(range(0, len(coords), step))
        else:
            coords_sub = coords
            indices_sub = list(range(len(coords)))
        
        # Build KD-tree for fast distance queries
        tree = KDTree(coords_sub)
        
        surface_indices = []
        surface_coords = []
        
        # For each atom, check if it's accessible (vectorized approach)
        for i, coord in enumerate(coords_sub):
            # Find neighbors within a certain radius
            neighbors = tree.query_ball_point(coord, 4.0)  # Fixed radius for speed
            
            # If atom has fewer neighbors than a threshold, it's likely on the surface
            if len(neighbors) < 15:  # Reduced threshold for speed
                actual_idx = indices_sub[i]
                surface_indices.append(actual_idx)
                surface_coords.append(coord)
        
        return np.array(surface_coords), surface_indices
    
    def _generate_focused_grid(self, protein, surface_atoms):
        """Generate grid points focused around surface atoms only (optimized)."""
        if len(surface_atoms) == 0:
            return np.array([])
        
        # Use a more efficient grid generation for large proteins
        coords = protein.xyz
        min_coords = np.min(coords, axis=0) - 3.0  # Smaller margin
        max_coords = np.max(coords, axis=0) + 3.0
        
        # Adaptive grid spacing based on protein size
        protein_size = np.max(max_coords - min_coords)
        if protein_size > 50:
            effective_spacing = self.grid_spacing * 1.5  # Coarser for large proteins
        else:
            effective_spacing = self.grid_spacing
        
        # Create coarse grid
        x = np.arange(min_coords[0], max_coords[0], effective_spacing * 2)
        y = np.arange(min_coords[1], max_coords[1], effective_spacing * 2)
        z = np.arange(min_coords[2], max_coords[2], effective_spacing * 2)
        
        X, Y, Z = np.meshgrid(x, y, z)
        grid_points_coarse = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        
        # Filter grid points more efficiently
        if len(surface_atoms) > 0:
            surface_tree = KDTree(surface_atoms)
            distances, _ = surface_tree.query(grid_points_coarse)
            potential_pocket_area = grid_points_coarse[distances < 6.0]
            
            # Limit grid size for performance
            if len(potential_pocket_area) > 50000:
                # Subsample if too many points
                step = len(potential_pocket_area) // 50000
                potential_pocket_area = potential_pocket_area[::step]
                
            return potential_pocket_area
        else:
            return np.array([])
    
    def _identify_pocket_points_vectorized(self, grid_points, protein):
        """Identify pocket points using vectorized operations (optimized)."""
        if len(grid_points) == 0:
            return np.array([]), np.array([])
            
        coords = protein.xyz
        
        # Build KD-tree for protein atoms
        atom_tree = KDTree(coords)
        
        # Process grid points in larger batches for efficiency
        batch_size = 20000  # Increased batch size
        pocket_points = []
        buriedness_scores = []
        
        # Get reasonable VDW radii (simplified)
        avg_radius = 1.7  # Use average radius for speed
        
        for i in range(0, len(grid_points), batch_size):
            batch = grid_points[i:i+batch_size]
            
            # Find nearest atoms and distances for each grid point
            k_neighbors = min(10, len(coords))  # Reduce neighbors for speed
            distances, indices = atom_tree.query(batch, k=k_neighbors)
            
            # Simplified checks for speed
            nearest_atom_dist = distances[:, 0]
            outside_atoms = nearest_atom_dist > avg_radius
            near_surface = nearest_atom_dist < avg_radius + 2*self.probe_radius
            not_far = nearest_atom_dist < 4.0
            
            # Simplified buriedness calculation
            atoms_within_6A = (distances < 6.0).sum(axis=1)  # Reduced radius for speed
            
            # Select points that meet all criteria
            mask = outside_atoms & near_surface & not_far
            selected_points = batch[mask]
            selected_buriedness = atoms_within_6A[mask]
            
            if len(selected_points) > 0:
                pocket_points.append(selected_points)
                buriedness_scores.append(selected_buriedness)
        
        # Combine results from all batches
        if pocket_points and len(pocket_points) > 0:
            pocket_points = np.vstack(pocket_points)
            buriedness_scores = np.concatenate(buriedness_scores)
            return pocket_points, buriedness_scores
        else:
            return np.array([]), np.array([])
    
    def _cluster_pocket_points(self, pocket_points, buriedness):
        """Cluster pocket points into distinct pockets (fixed clustering)."""
        if len(pocket_points) == 0:
            return []
        
        if self.verbose:
            print(f"   🔍 Clustering {len(pocket_points)} pocket points...")
        
        # Use more permissive clustering parameters to actually find pockets
        try:
            from sklearn.cluster import DBSCAN
            import numpy as np
            
            # Adaptive clustering based on point density
            if len(pocket_points) > 10000:
                # For large datasets
                eps = 2.5
                min_samples = 8
            elif len(pocket_points) > 5000:
                # For medium datasets  
                eps = 2.0
                min_samples = 6
            else:
                # For smaller datasets - more permissive
                eps = 1.8
                min_samples = 4
            
            if self.verbose:
                print(f"   📊 Using DBSCAN with eps={eps}, min_samples={min_samples}")
            
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pocket_points)
            labels = clustering.labels_
            
            # Group points by cluster label
            clusters = {}
            for i, label in enumerate(labels):
                if label == -1:  # Skip noise points
                    continue
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(pocket_points[i])
            
            # Convert to arrays and filter by size
            refined_clusters = []
            for label, points in clusters.items():
                points_array = np.array(points)
                # Keep clusters with at least 10 points
                if len(points_array) >= 10:
                    refined_clusters.append(points_array)
            
            # If still no clusters, try even more permissive parameters
            if len(refined_clusters) == 0 and len(pocket_points) > 20:
                if self.verbose:
                    print("   🔄 No clusters found, trying more permissive parameters...")
                
                # Very permissive clustering
                clustering = DBSCAN(eps=3.5, min_samples=3).fit(pocket_points)
                labels = clustering.labels_
                
                clusters = {}
                for i, label in enumerate(labels):
                    if label == -1:
                        continue
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(pocket_points[i])
                
                for label, points in clusters.items():
                    points_array = np.array(points)
                    if len(points_array) >= 5:  # Even more permissive
                        refined_clusters.append(points_array)
            
            # Sort clusters by size (largest first)
            refined_clusters.sort(key=len, reverse=True)
            
            if self.verbose:
                print(f"   ✓ Found {len(refined_clusters)} pocket clusters")
                for i, cluster in enumerate(refined_clusters[:3]):
                    print(f"     Cluster {i+1}: {len(cluster)} points")
            
            return refined_clusters[:10]  # Return top 10 clusters
            
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ Clustering failed: {e}")
            return []
    
    def _calculate_pocket_properties(self, pocket_clusters, protein):
        """Calculate properties for each pocket (enhanced to handle edge cases)."""
        pockets = []
        coords = protein.xyz
        
        if len(pocket_clusters) == 0:
            if self.verbose:
                print("   ⚠️ No pocket clusters to analyze")
            return pockets
        
        # Build KD-tree for protein atoms once
        atom_tree = KDTree(coords)
        
        for i, cluster in enumerate(pocket_clusters):
            # Skip very small clusters (more permissive)
            if len(cluster) < 4:
                continue
                
            # Calculate center as center of mass
            center = np.mean(cluster, axis=0)
            
            # Calculate volume - more robust approach
            try:
                from scipy.spatial import ConvexHull
                if len(cluster) >= 4:  # Need at least 4 points for ConvexHull
                    hull = ConvexHull(cluster)
                    volume = hull.volume
                    # Ensure reasonable volume
                    if volume < 10:
                        volume = len(cluster) * 2.0  # Fallback based on point count
                else:
                    # Fallback for small clusters
                    volume = len(cluster) * 2.0
            except Exception as e:
                # Fallback to point-based volume estimation
                volume = len(cluster) * 2.0
            
            # Ensure minimum volume for detection
            volume = max(volume, 20.0)  # Minimum 20 Ų volume
            
            # Find atoms that form the pocket
            try:
                pocket_atom_indices = atom_tree.query_ball_point(center, 7.0)
            except:
                pocket_atom_indices = []
            
            # Identify residues forming the pocket
            pocket_residues = set()
            for atom_idx in pocket_atom_indices[:200]:  # Reasonable limit
                if atom_idx < len(protein.atoms):
                    atom = protein.atoms[atom_idx]
                    if 'residue_id' in atom:
                        pocket_residues.add(atom['residue_id'])
                    elif 'res_id' in atom:
                        pocket_residues.add(atom['res_id'])
                    elif 'chain_res' in atom:
                        pocket_residues.add(atom['chain_res'])
            
            # Store pocket information
            pocket_info = {
                'id': i + 1,
                'center': center,
                'volume': volume,
                'n_points': len(cluster),
                'atoms': pocket_atom_indices,
                'residues': list(pocket_residues),
                'n_mouths': 1,
                'solvent_exposure': True
            }
            
            pockets.append(pocket_info)
            
            if self.verbose:
                print(f"   ✓ Pocket {i+1}: {volume:.0f}Ų, {len(pocket_residues)} residues")
        
        return pockets