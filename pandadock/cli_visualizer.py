"""
CLI Protein and Active Site Visualizer for PandaDock
"""
import numpy as np
import os
import sys
from typing import List, Dict, Any, Tuple, Optional

class ProteinCLIVisualizer:
    """
    Create ASCII-based visualizations of protein structures and active sites in the terminal.
    """
    
    def __init__(self, width=80, height=25):
        """
        Initialize the CLI visualizer.
        
        Parameters:
        -----------
        width : int
            Width of the visualization canvas
        height : int
            Height of the visualization canvas
        """
        self.width = width
        self.height = height
        self.canvas = None
        self.colors_enabled = self._check_color_support()
        
    def _check_color_support(self):
        """Check if terminal supports colors."""
        return (
            hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and
            (os.environ.get('TERM', '').find('color') != -1 or 
             os.environ.get('COLORTERM') is not None)
        )
    
    def _colorize(self, text, color_code):
        """Add color to text if terminal supports it."""
        if self.colors_enabled:
            return f"\033[{color_code}m{text}\033[0m"
        return text
    
    def _project_3d_to_2d(self, coords, center=None, scale=None):
        """
        Project 3D coordinates to 2D for visualization.
        
        Parameters:
        -----------
        coords : np.ndarray
            3D coordinates (N x 3)
        center : np.ndarray, optional
            Center point for projection
        scale : float, optional
            Scaling factor
            
        Returns:
        --------
        tuple
            (x_2d, y_2d) projected coordinates
        """
        if center is None:
            center = np.mean(coords, axis=0)
        
        # Center the coordinates
        centered = coords - center
        
        # Simple orthographic projection (ignore Z for now, can be enhanced)
        x_proj = centered[:, 0]
        y_proj = centered[:, 1]
        
        # Auto-scale to fit the canvas
        if scale is None:
            x_range = np.max(x_proj) - np.min(x_proj)
            y_range = np.max(y_proj) - np.min(y_proj)
            scale = min((self.width - 4) / max(x_range, 1), (self.height - 4) / max(y_range, 1)) * 0.8
        
        # Scale and translate to canvas coordinates
        x_2d = ((x_proj * scale) + self.width // 2).astype(int)
        y_2d = ((y_proj * scale) + self.height // 2).astype(int)
        
        # Clip to canvas bounds
        x_2d = np.clip(x_2d, 0, self.width - 1)
        y_2d = np.clip(y_2d, 0, self.height - 1)
        
        return x_2d, y_2d, scale
    
    def _create_canvas(self):
        """Create empty canvas."""
        self.canvas = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
    def _draw_point(self, x, y, char='·', color_code=None):
        """Draw a point on the canvas."""
        if 0 <= x < self.width and 0 <= y < self.height:
            if color_code and self.colors_enabled:
                self.canvas[y][x] = f"\033[{color_code}m{char}\033[0m"
            else:
                self.canvas[y][x] = char
    
    def _draw_circle(self, center_x, center_y, radius, char='○', color_code=None):
        """Draw a circle on the canvas."""
        for angle in np.linspace(0, 2*np.pi, max(8, int(radius * 4))):
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            self._draw_point(x, y, char, color_code)
    
    def _draw_sphere_outline(self, center_x, center_y, radius, char='◯', color_code=None):
        """Draw a 3D sphere outline representation."""
        # Draw multiple concentric circles to simulate 3D sphere
        for r in [radius * 0.3, radius * 0.6, radius]:
            for angle in np.linspace(0, 2*np.pi, max(6, int(r * 3))):
                x = int(center_x + r * np.cos(angle))
                y = int(center_y + r * np.sin(angle) * 0.6)  # Flatten Y to simulate perspective
                if r == radius:
                    self._draw_point(x, y, char, color_code)
                else:
                    self._draw_point(x, y, '·', color_code)
    
    def _draw_arrow(self, start_x, start_y, end_x, end_y, color_code=None):
        """Draw an arrow from start to end point."""
        # Draw line
        dx = end_x - start_x
        dy = end_y - start_y
        length = max(abs(dx), abs(dy))
        
        if length > 0:
            for i in range(length + 1):
                x = int(start_x + (dx * i) / length)
                y = int(start_y + (dy * i) / length)
                char = '-' if abs(dx) > abs(dy) else '|'
                if i == length:
                    char = '→' if dx > 0 else '←' if dx < 0 else '↓' if dy > 0 else '↑'
                self._draw_point(x, y, char, color_code)
    
    def _render_canvas(self):
        """Convert canvas to string for display."""
        lines = []
        for row in self.canvas:
            lines.append(''.join(row))
        return '\n'.join(lines)
    
    def visualize_protein_with_pockets(self, protein, pockets=None, title="Protein Structure with Binding Pockets"):
        """
        Create a visualization of protein structure with detected pockets.
        
        Parameters:
        -----------
        protein : Protein
            Protein object with coordinates
        pockets : list, optional
            List of detected pockets
        title : str
            Title for the visualization
            
        Returns:
        --------
        str
            ASCII art visualization
        """
        self._create_canvas()
        
        # Get protein coordinates (subsample for clarity)
        coords = protein.xyz
        subsample_step = max(1, len(coords) // 200)  # Show max 200 points
        coords_sub = coords[::subsample_step]
        
        # Project coordinates to 2D
        x_2d, y_2d, scale = self._project_3d_to_2d(coords_sub)
        
        # Draw protein atoms
        for x, y in zip(x_2d, y_2d):
            self._draw_point(x, y, '·', '37')  # Cyan dots for protein
        
        # Draw pockets if provided
        if pockets:
            pocket_colors = ['91', '92', '93', '94', '95', '96']  # Different colors for pockets
            
            for i, pocket in enumerate(pockets[:6]):  # Show max 6 pockets
                center = pocket['center']
                radius = pocket.get('radius', 5.0)
                
                # Project pocket center to 2D
                center_x, center_y, _ = self._project_3d_to_2d(
                    center.reshape(1, -1), 
                    center=np.mean(coords, axis=0), 
                    scale=scale
                )
                
                # Calculate 2D radius
                radius_2d = max(2, int(radius * scale / 10))
                
                color = pocket_colors[i % len(pocket_colors)]
                
                # Draw pocket outline
                self._draw_sphere_outline(center_x[0], center_y[0], radius_2d, '◯', color)
                
                # Draw arrow pointing to pocket
                arrow_start_x = 2 + i * 10
                arrow_start_y = 1
                self._draw_arrow(arrow_start_x, arrow_start_y, center_x[0], center_y[0], color)
                
                # Add pocket label
                if center_x[0] + 2 < self.width and center_y[0] < self.height:
                    label = f"P{i+1}"
                    for j, char in enumerate(label):
                        if center_x[0] + j < self.width:
                            self._draw_point(center_x[0] + j, center_y[0], char, color)
        
        # Create header and footer
        header = f"┌{'─' * (self.width - 2)}┐"
        footer = f"└{'─' * (self.width - 2)}┘"
        title_line = f"│ {title:<{self.width - 3}}│"
        
        # Create legend
        legend_lines = []
        legend_lines.append(f"│ {'Legend:':<{self.width - 3}}│")
        legend_lines.append(f"│ {self._colorize('·', '37')} Protein atoms{'':<{self.width - 16}}│")
        
        if pockets:
            legend_lines.append(f"│ {'Binding Pockets:':<{self.width - 3}}│")
            for i, pocket in enumerate(pockets[:6]):
                color = pocket_colors[i % len(pocket_colors)]
                volume = pocket.get('volume', 0)
                line = f"│ {self._colorize('◯', color)} P{i+1}: Vol={volume:.0f}Å³"
                padding = self.width - len(f" P{i+1}: Vol={volume:.0f}Å³") - 3
                legend_lines.append(f"{line}{' ' * padding}│")
        
        # Combine everything
        canvas_str = self._render_canvas()
        
        result = []
        result.append(header)
        result.append(title_line)
        result.append(f"│{'─' * (self.width - 2)}│")
        
        # Add canvas lines
        for line in canvas_str.split('\n'):
            result.append(f"│{line}│")
        
        result.append(f"│{'─' * (self.width - 2)}│")
        
        # Add legend
        for legend_line in legend_lines:
            result.append(legend_line)
        
        result.append(footer)
        
        return '\n'.join(result)
    
    def create_pocket_summary_diagram(self, pockets, protein_name="Protein"):
        """
        Create a compact summary diagram of detected pockets.
        
        Parameters:
        -----------
        pockets : list
            List of detected pockets
        protein_name : str
            Name of the protein
            
        Returns:
        --------
        str
            ASCII summary diagram
        """
        if not pockets:
            return self._colorize("No pockets detected", '91')
        
        lines = []
        lines.append("╔═══════════════════════════════════════════════════════════════════╗")
        lines.append(f"║ 🧬 {protein_name} - Detected Binding Pockets{' ' * (30 - len(protein_name))}║")
        lines.append("╠═══════════════════════════════════════════════════════════════════╣")
        
        pocket_colors = ['91', '92', '93', '94', '95', '96']
        
        for i, pocket in enumerate(pockets):
            color = pocket_colors[i % len(pocket_colors)]
            volume = pocket.get('volume', 0)
            center = pocket['center']
            radius = pocket.get('radius', 0)
            n_residues = len(pocket.get('residues', []))
            
            # Create visual representation
            sphere_chars = ['◯', '○', '●']
            size_char = sphere_chars[min(2, int(volume / 100))]  # Size based on volume
            
            pocket_visual = self._colorize(f"{size_char}", color)
            
            # Format pocket information
            pocket_info = (f"║ {pocket_visual} Pocket {i+1:2d}: "
                          f"Vol={volume:6.0f}Å³  "
                          f"R={radius:4.1f}Å  "
                          f"Residues={n_residues:3d}  "
                          f"Center=({center[0]:5.1f},{center[1]:5.1f},{center[2]:5.1f}) ║")
            
            lines.append(pocket_info)
        
        lines.append("╠═══════════════════════════════════════════════════════════════════╣")
        lines.append(f"║ 🎯 Recommendation: Use Pocket 1 (largest) for docking{' ' * 18}║")
        lines.append("╚═══════════════════════════════════════════════════════════════════╝")
        
        return '\n'.join(lines)
    
    def create_docking_progress_diagram(self, protein, ligand, active_site_center, active_site_radius, 
                                       current_best_score=None, iteration=None, total_iterations=None):
        """
        Create a diagram showing docking progress.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand  
            Ligand object
        active_site_center : np.ndarray
            Center of active site
        active_site_radius : float
            Radius of active site
        current_best_score : float, optional
            Current best docking score
        iteration : int, optional
            Current iteration
        total_iterations : int, optional
            Total iterations
            
        Returns:
        --------
        str
            ASCII docking progress diagram
        """
        self._create_canvas()
        
        # Subsample protein coordinates
        coords = protein.xyz
        subsample_step = max(1, len(coords) // 100)
        coords_sub = coords[::subsample_step]
        
        # Project to 2D
        x_2d, y_2d, scale = self._project_3d_to_2d(coords_sub)
        
        # Draw protein
        for x, y in zip(x_2d, y_2d):
            self._draw_point(x, y, '·', '37')
        
        # Draw active site
        center_x, center_y, _ = self._project_3d_to_2d(
            active_site_center.reshape(1, -1),
            center=np.mean(coords, axis=0),
            scale=scale
        )
        radius_2d = max(3, int(active_site_radius * scale / 8))
        
        # Draw active site sphere
        self._draw_sphere_outline(center_x[0], center_y[0], radius_2d, '◯', '91')
        
        # Draw ligand position (if available)
        if hasattr(ligand, 'xyz') and len(ligand.xyz) > 0:
            ligand_center = np.mean(ligand.xyz, axis=0)
            lig_x, lig_y, _ = self._project_3d_to_2d(
                ligand_center.reshape(1, -1),
                center=np.mean(coords, axis=0),
                scale=scale
            )
            self._draw_point(lig_x[0], lig_y[0], '💊', '93')
        
        # Create progress info
        canvas_str = self._render_canvas()
        
        lines = []
        lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
        lines.append("│ 🔬 PandaDock - Molecular Docking in Progress                                 │")
        lines.append("├─────────────────────────────────────────────────────────────────────────────┤")
        
        # Add canvas
        for line in canvas_str.split('\n'):
            lines.append(f"│ {line:<77} │")
        
        lines.append("├─────────────────────────────────────────────────────────────────────────────┤")
        lines.append(f"│ Legend: {self._colorize('·', '37')} Protein  {self._colorize('◯', '91')} Active Site  💊 Ligand{' ' * 35} │")
        
        if current_best_score is not None:
            lines.append(f"│ 🎯 Current Best Score: {current_best_score:8.2f}{' ' * 45} │")
        
        if iteration is not None and total_iterations is not None:
            progress = (iteration / total_iterations) * 100
            progress_bar = '█' * int(progress / 2) + '░' * (50 - int(progress / 2))
            lines.append(f"│ 📊 Progress: [{progress_bar}] {progress:5.1f}%{' ' * 11} │")
            lines.append(f"│ 🔄 Iteration: {iteration:4d} / {total_iterations:4d}{' ' * 50} │")
        
        lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
        
        return '\n'.join(lines)

def create_quick_protein_visualization(protein, pockets=None, width=60, height=20):
    """
    Quick function to create a protein visualization.
    
    Parameters:
    -----------
    protein : Protein
        Protein object
    pockets : list, optional
        List of detected pockets
    width : int
        Visualization width
    height : int
        Visualization height
        
    Returns:
    --------
    str
        ASCII visualization
    """
    visualizer = ProteinCLIVisualizer(width=width, height=height)
    return visualizer.visualize_protein_with_pockets(protein, pockets)

def print_pocket_detection_results(protein, pockets, protein_name="Unknown"):
    """
    Print formatted pocket detection results with visualization.
    
    Parameters:
    -----------
    protein : Protein
        Protein object
    pockets : list
        List of detected pockets
    protein_name : str
        Name of the protein
    """
    visualizer = ProteinCLIVisualizer()
    
    # Print summary diagram
    summary = visualizer.create_pocket_summary_diagram(pockets, protein_name)
    print(summary)
    
    # Print detailed visualization if terminal is wide enough
    if os.get_terminal_size().columns >= 85:
        print("\n")
        detailed_viz = visualizer.visualize_protein_with_pockets(protein, pockets)
        print(detailed_viz)
    else:
        print(f"\n💡 Terminal too narrow for detailed visualization (need 85+ columns, have {os.get_terminal_size().columns})")
        print("   Try expanding your terminal or use: pandadock ... --no-viz for text-only output")
