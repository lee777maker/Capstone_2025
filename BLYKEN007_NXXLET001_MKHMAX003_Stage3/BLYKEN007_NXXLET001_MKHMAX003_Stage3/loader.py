import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import json
from pathlib import Path

# Design Pattern: Strategy for different rendering backends
class FPVRendererStrategy(ABC):
    """Strategy interface for FPV rendering backends"""
    
    @abstractmethod
    def render_fpv(self, vertices: np.ndarray, triangles: np.ndarray, 
                  colors: np.ndarray, position: np.ndarray, 
                  orientation: Tuple[float, float]) -> Optional[np.ndarray]:
        """Render first-person view from given position and orientation"""
        pass

# Design Pattern: Singleton for configuration management
class FPVConfig:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FPVConfig, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.resolution = (640, 480)
        self.fov = 60  # degrees
        self.near_plane = 0.1
        self.far_plane = 100.0
        self.max_workers = 4
        self.cache_size = 10  # Number of FPV views to cache

# Design Pattern: Factory for creating renderers
class FPVRendererFactory:
    @staticmethod
    def create_renderer(renderer_type: str = "default") -> FPVRendererStrategy:
        if renderer_type == "opengl":
            return OpenGLRenderer()
        elif renderer_type == "software":
            return SoftwareRenderer()
        else:
            return DefaultRenderer()

# Design Pattern: Observer for FPV updates
class FPVCameraObserver(ABC):
    @abstractmethod
    def on_camera_update(self, position: np.ndarray, orientation: Tuple[float, float]):
        pass

# Concrete implementations
class DefaultRenderer(FPVRendererStrategy):
    """Default FPV renderer using efficient numpy operations"""
    
    def __init__(self):
        self.config = FPVConfig()
        self.vertex_cache = {}  # Simple cache for recently processed vertices
        
    def render_fpv(self, vertices: np.ndarray, triangles: np.ndarray, 
                  colors: np.ndarray, position: np.ndarray, 
                  orientation: Tuple[float, float]) -> Optional[np.ndarray]:
        """Render FPV using efficient matrix operations"""
        try:
            # Apply view transformation
            transformed_vertices = self._apply_view_transform(vertices, position, orientation)
            
            # Apply projection
            projected_vertices = self._apply_projection(transformed_vertices)
            
            # Clip vertices outside view frustum
            visible_vertices, visible_triangles, visible_colors = self._frustum_culling(
                projected_vertices, triangles, colors
            )
            
            # Render to image buffer
            image_buffer = self._rasterize(
                visible_vertices, visible_triangles, visible_colors
            )
            
            return image_buffer
            
        except Exception as e:
            print(f"Error in FPV rendering: {e}")
            return None
    
    def _apply_view_transform(self, vertices: np.ndarray, 
                            position: np.ndarray, 
                            orientation: Tuple[float, float]) -> np.ndarray:
        """Apply view transformation matrix efficiently"""
        yaw, pitch = orientation
        
        # Create rotation matrices
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
        
        # Combined rotation matrix
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw * sin_pitch, sin_yaw * cos_pitch, 0],
            [0, cos_pitch, sin_pitch, 0],
            [-sin_yaw, -cos_yaw * sin_pitch, cos_yaw * cos_pitch, 0],
            [0, 0, 0, 1]
        ])
        
        # Translation matrix
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = -position
        
        # Combine transformations
        view_matrix = rotation_matrix @ translation_matrix
        
        # Apply to vertices (using homogeneous coordinates)
        homogeneous_vertices = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
        transformed_vertices = (view_matrix @ homogeneous_vertices.T).T
        
        return transformed_vertices[:, :3]
    
    def _apply_projection(self, vertices: np.ndarray) -> np.ndarray:
        """Apply perspective projection"""
        config = FPVConfig()
        fov_rad = np.radians(config.fov)
        aspect_ratio = config.resolution[0] / config.resolution[1]
        
        # Perspective projection matrix
        f = 1.0 / np.tan(fov_rad / 2)
        near, far = config.near_plane, config.far_plane
        
        projection_matrix = np.array([
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ])
        
        # Apply projection
        homogeneous_vertices = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
        projected_vertices = (projection_matrix @ homogeneous_vertices.T).T
        
        # Perspective divide
        w = projected_vertices[:, 3:]
        projected_vertices = projected_vertices[:, :3] / w
        
        return projected_vertices
    
    def _frustum_culling(self, vertices: np.ndarray, triangles: np.ndarray, 
                        colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Efficiently remove geometry outside view frustum"""
        # Simple bounding box culling
        in_frustum = np.all(
            (vertices >= -1) & (vertices <= 1), 
            axis=1
        )
        
        # Get indices of visible vertices
        visible_vertex_indices = np.where(in_frustum)[0]
        
        if len(visible_vertex_indices) == 0:
            return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))
        
        # Create mapping from original indices to new indices
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(visible_vertex_indices)}
        
        # Filter triangles where all vertices are visible
        visible_triangles = []
        visible_triangle_indices = []
        
        for i, triangle in enumerate(triangles):
            if all(vertex_idx in index_mapping for vertex_idx in triangle):
                visible_triangles.append([index_mapping[vertex_idx] for vertex_idx in triangle])
                visible_triangle_indices.append(i)
        
        if not visible_triangles:
            return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))
        
        visible_vertices = vertices[visible_vertex_indices]
        visible_triangles = np.array(visible_triangles)
        visible_colors = colors[visible_vertex_indices] if colors is not None else None
        
        return visible_vertices, visible_triangles, visible_colors
    
    def _rasterize(self, vertices: np.ndarray, triangles: np.ndarray, 
                  colors: np.ndarray) -> np.ndarray:
        """Simple rasterization to image buffer"""
        config = FPVConfig()
        width, height = config.resolution
        
        # Initialize image buffer
        image_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        
        if len(vertices) == 0:
            return image_buffer
        
        # Convert normalized device coordinates to screen coordinates
        screen_coords = np.zeros_like(vertices)
        screen_coords[:, 0] = (vertices[:, 0] + 1) * 0.5 * width
        screen_coords[:, 1] = (1 - vertices[:, 1]) * 0.5 * height
        
        # Simple triangle rasterization (would be optimized in real implementation)
        for triangle in triangles:
            # Get triangle vertices and colors
            tri_verts = screen_coords[triangle]
            tri_colors = colors[triangle] if colors is not None else np.array([128, 128, 128])
            
            # Simple filling (in a real implementation, use proper rasterization)
            min_x = max(0, int(np.min(tri_verts[:, 0])))
            max_x = min(width, int(np.max(tri_verts[:, 0])) + 1)
            min_y = max(0, int(np.min(tri_verts[:, 1])))
            max_y = min(height, int(np.max(tri_verts[:, 1])) + 1)
            
            # Fill bounding box with average color (simplified)
            avg_color = np.mean(tri_colors, axis=0).astype(np.uint8)
            image_buffer[min_y:max_y, min_x:max_x] = avg_color
        
        return image_buffer

# Design Pattern: Proxy for efficient FPV management
class FPVManager:
    """Manages FPV rendering with caching and optimization"""
    
    def __init__(self, renderer_type: str = "default"):
        self.renderer = FPVRendererFactory.create_renderer(renderer_type)
        self.config = FPVConfig()
        self.cache = {}  # Cache for rendered FPV views
        self.observers = []  # Observer pattern for camera updates
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
    
    def add_observer(self, observer: FPVCameraObserver):
        """Add an observer for camera updates"""
        self.observers.append(observer)
    
    def remove_observer(self, observer: FPVCameraObserver):
        """Remove an observer"""
        self.observers.remove(observer)
    
    def notify_observers(self, position: np.ndarray, orientation: Tuple[float, float]):
        """Notify all observers of camera updates"""
        for observer in self.observers:
            observer.on_camera_update(position, orientation)
    
    def get_fpv(self, mesh_data: Dict[str, Any], position: np.ndarray, 
                orientation: Tuple[float, float]) -> Optional[np.ndarray]:
        """Get FPV image with caching"""
        # Create cache key
        cache_key = self._create_cache_key(position, orientation)
        
        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Render FPV asynchronously
        future = self.executor.submit(
            self._render_fpv_async, mesh_data, position, orientation, cache_key
        )
        
        # Return placeholder while rendering
        return self._create_placeholder()
    
    def _render_fpv_async(self, mesh_data: Dict[str, Any], position: np.ndarray, 
                         orientation: Tuple[float, float], cache_key: str):
        """Render FPV asynchronously and cache result"""
        vertices = mesh_data.get('vertices')
        triangles = mesh_data.get('triangles')
        colors = mesh_data.get('colors')
        
        if vertices is None or triangles is None:
            return None
        
        # Render FPV
        image = self.renderer.render_fpv(vertices, triangles, colors, position, orientation)
        
        # Cache result
        if image is not None:
            self.cache[cache_key] = image
            
            # Manage cache size
            if len(self.cache) > self.config.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
        
        return image
    
    def _create_cache_key(self, position: np.ndarray, orientation: Tuple[float, float]) -> str:
        """Create a unique cache key for position and orientation"""
        pos_key = ",".join(f"{x:.2f}" for x in position)
        orient_key = ",".join(f"{x:.2f}" for x in orientation)
        return f"{pos_key}|{orient_key}"
    
    def _create_placeholder(self) -> np.ndarray:
        """Create a placeholder image while rendering"""
        config = FPVConfig()
        placeholder = np.zeros((config.resolution[1], config.resolution[0], 3), dtype=np.uint8)
        placeholder[:] = [50, 50, 50]  # Dark gray
        
        # Add "Rendering..." text
        center_x, center_y = config.resolution[0] // 2, config.resolution[1] // 2
        text_size = 20
        text_color = [200, 200, 200]
        
        # Simple text rendering (would use proper text rendering in real implementation)
        placeholder[
            center_y - text_size:center_y + text_size,
            center_x - 60:center_x + 60
        ] = text_color
        
        return placeholder

# Design Pattern: Facade for simplified FPV access
class FPVFacade:
    """Simplified interface for FPV functionality"""
    
    def __init__(self, mesh_data: Dict[str, Any]):
        self.mesh_data = mesh_data
        self.fpv_manager = FPVManager()
        self.current_position = np.array([0, 0, 0])
        self.current_orientation = (0, 0)  # yaw, pitch
    
    def update_camera(self, position: np.ndarray, orientation: Tuple[float, float]):
        """Update camera position and orientation"""
        self.current_position = position
        self.current_orientation = orientation
        self.fpv_manager.notify_observers(position, orientation)
    
    def get_current_fpv(self) -> Optional[np.ndarray]:
        """Get FPV from current camera position"""
        return self.fpv_manager.get_fpv(
            self.mesh_data, self.current_position, self.current_orientation
        )
    
    def get_fpv_from_waypoint(self, waypoint_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Get FPV from waypoint data"""
        position = waypoint_data.get('position', np.array([0, 0, 0]))
        orientation = (
            waypoint_data.get('yaw', 0),
            waypoint_data.get('pitch', 0)
        )
        return self.fpv_manager.get_fpv(self.mesh_data, position, orientation)

# Example usage and integration with existing code
if __name__ == "__main__":
    # Example mesh data (would come from your existing loader)
    mesh_data = {
        'vertices': np.random.rand(1000, 3) * 10,  # Example data
        'triangles': np.random.randint(0, 1000, (500, 3)),  # Example data
        'colors': np.random.rand(1000, 3) * 255  # Example data
    }
    
    # Create FPV facade
    fpv_facade = FPVFacade(mesh_data)
    
    # Update camera to a specific position and orientation
    position = np.array([5.0, 5.0, 5.0])
    orientation = (0.5, 0.2)  # yaw, pitch in radians
    fpv_facade.update_camera(position, orientation)
    
    # Get FPV image
    fpv_image = fpv_facade.get_current_fpv()
    
    # Example waypoint data
    waypoint = {
        'position': np.array([3.0, 3.0, 3.0]),
        'yaw': 0.3,
        'pitch': 0.1
    }
    
    # Get FPV from waypoint
    waypoint_fpv = fpv_facade.get_fpv_from_waypoint(waypoint)
    
    print("FPV system initialized successfully")