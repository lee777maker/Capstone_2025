#!/usr/bin/env python3
"""
Production UAV Visualizer - Fixed Architecture
Addresses renderer recreation issues while maintaining original UI design.
"""

import sys
import os
import time
import json
import queue
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Set, Callable
from enum import Enum, auto 
from collections import deque
import logging
import weakref

import numpy as np
import trimesh
from scipy.interpolate import CubicSpline, interp1d
import cv2

# GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Open3D
import open3d as o3d

# Pillow for converting numpy images to Tkinter PhotoImage
from PIL import Image, ImageTk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------------
# Event System
# -------------------------------
class EventBus:
    """Thread-safe event bus for decoupled communication"""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
    
    def subscribe(self, event_type: str, callback: Callable):
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable):
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                except ValueError:
                    pass
    
    def publish(self, event_type: str, data: Any = None):
        with self._lock:
            callbacks = self._subscribers.get(event_type, []).copy()
        
        for callback in callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Event callback error for {event_type}: {e}")

# Global event bus
event_bus = EventBus()

# -------------------------------
# Data Models
# -------------------------------
class InterpolationMethod(Enum):
    LINEAR = "linear"
    SPLINE = "spline"

@dataclass
class Waypoint:
    position: np.ndarray
    yaw: float
    pitch: float
    index: int

    def to_dict(self) -> Dict:
        return {
            'position': self.position.tolist(),
            'yaw': self.yaw,
            'pitch': self.pitch,
            'index': self.index
        }

    @classmethod
    def from_dict(cls, data: Dict, index: int) -> 'Waypoint':
        if isinstance(data, (list, tuple)) and len(data) >= 3:
            pos = np.array(data[:3], dtype=float)
            return cls(position=pos, yaw=0.0, pitch=0.0, index=index)
        
        if 'position' in data and isinstance(data['position'], (list, tuple)):
            pos = np.array(data['position'][:3], dtype=float)
            return cls(position=pos, yaw=data.get('yaw', 0.0), pitch=data.get('pitch', 0.0), index=index)
        
        # Handle various coordinate formats
        x = data.get('x') or data.get('lon') or data.get('longitude')
        y = data.get('y') or data.get('lat') or data.get('latitude')
        z = data.get('z') or data.get('alt') or data.get('altitude')
        
        if x is not None and y is not None and z is not None:
            pos = np.array([float(x), float(y), float(z)], dtype=float)
            return cls(position=pos, yaw=data.get('yaw', 0.0), pitch=data.get('pitch', 0.0), index=index)
        
        raise ValueError("Unrecognized waypoint format")

@dataclass
class Trajectory:
    name: str
    waypoints: List[Waypoint]
    color: Tuple[float, float, float]
    interpolation_method: InterpolationMethod = InterpolationMethod.SPLINE
    _cached_positions: Optional[np.ndarray] = field(default=None, init=False)
    _cached_interpolated: Optional[np.ndarray] = field(default=None, init=False)
    _cache_dirty: bool = field(default=True, init=False)

    def get_positions(self) -> np.ndarray:
        """Get waypoint positions with caching"""
        if self._cache_dirty or self._cached_positions is None:
            self._cached_positions = np.array([wp.position for wp in self.waypoints])
            self._cache_dirty = False
        return self._cached_positions

    def invalidate_cache(self):
        """Mark cache as dirty"""
        self._cache_dirty = True
        self._cached_interpolated = None

    def get_interpolated_path(self, num_points: int = 200) -> np.ndarray:
        """Get interpolated path with caching"""
        if self._cached_interpolated is None or self._cache_dirty:
            if self.interpolation_method == InterpolationMethod.LINEAR:
                self._cached_interpolated = TrajectoryInterpolator.interpolate_linear(self.waypoints, num_points)
            else:
                self._cached_interpolated = TrajectoryInterpolator.interpolate_spline(self.waypoints, num_points)
        return self._cached_interpolated

    def calculate_metrics(self, cruising_speed: float = 5.0, hover_time: float = 2.5) -> Dict:
        positions = self.get_positions()
        if len(positions) < 2:
            return {
                'total_length': 0.0,
                'total_vertical': 0.0,
                'total_duration': 0.0,
                'sharp_corners': 0,
                'num_waypoints': len(self.waypoints)
            }

        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_length = np.sum(distances)
        
        vertical_changes = np.abs(np.diff(positions[:, 2])) if positions.shape[1] > 2 else np.array([0.0])
        total_vertical = np.sum(vertical_changes)
        
        flight_time = total_length / cruising_speed
        total_hover = hover_time * len(self.waypoints)
        total_duration = flight_time + total_hover

        # Calculate sharp corners
        sharp_corners = 0
        for i in range(1, len(positions) - 1):
            v1 = positions[i] - positions[i - 1]
            v2 = positions[i + 1] - positions[i]
            
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                continue
                
            angle = np.arccos(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1))
            if np.degrees(angle) > 90:
                sharp_corners += 1

        return {
            'total_length': total_length,
            'total_vertical': total_vertical,
            'total_duration': total_duration,
            'sharp_corners': sharp_corners,
            'num_waypoints': len(self.waypoints)
        }

# -------------------------------
# Geometry Cache and Management
# -------------------------------
class GeometryCache:
    """Cache for Open3D geometry objects"""
    
    def __init__(self):
        self._cache: Dict[str, o3d.geometry.Geometry] = {}
        self._dirty_flags: Dict[str, bool] = {}
    
    def get(self, key: str) -> Optional[o3d.geometry.Geometry]:
        return self._cache.get(key)
    
    def set(self, key: str, geometry: o3d.geometry.Geometry):
        self._cache[key] = geometry
        self._dirty_flags[key] = False
    
    def mark_dirty(self, key: str):
        self._dirty_flags[key] = True
    
    def is_dirty(self, key: str) -> bool:
        return self._dirty_flags.get(key, True)
    
    def clear(self):
        self._cache.clear()
        self._dirty_flags.clear()

# -------------------------------
# Persistent Renderer
# -------------------------------
class PersistentRenderer:
    """Persistent Open3D renderer that avoids recreation"""
    
    def __init__(self, name: str, width: int = 960, height: int = 720):
        self.name = name
        self.width = width
        self.height = height
        self.vis: Optional[o3d.visualization.Visualizer] = None
        self.geometry_cache = GeometryCache()
        self._is_initialized = False
        self._view_control = None
    
    def initialize(self) -> bool:
        """Initialize the renderer once"""
        if self._is_initialized:
            return True
            
        try:
            self.vis = o3d.visualization.Visualizer()
            # Use visible=False for offscreen rendering
            self.vis.create_window(
                window_name=self.name,
                width=self.width,
                height=self.height,
                visible=False
            )
            
         
            render_option = self.vis.get_render_option()
            render_option.background_color = np.asarray([0.1, 0.1, 0.1])
            render_option.point_size = 5.0
            render_option.line_width = 2.0
            
            self._view_control = self.vis.get_view_control()
            self._is_initialized = True

            # Force initial geometry to ensure rendering pipeline is active
            dummy_geom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self.vis.add_geometry(dummy_geom)
            self.vis.poll_events()
            self.vis.update_renderer()
            logger.info(f"{self.name}: Renderer initialized successfully")
            return True
        except Exception as e:
            logger.error(f"{self.name}: Failed to initialize: {e}")
            return False
    
    def update_viewport(self, width: int, height: int):
        """Update viewport size without recreating renderer"""
        if not self._is_initialized or not self.vis:
            return
            
        self.width = width
        self.height = height
        self.vis.poll_events()
        self.vis.update_renderer()
        # Open3D automatically handles viewport changes
        logger.debug(f"{self.name}: Viewport updated to {width}x{height}")
    
    def update_geometries(self, geometries: Dict[str, o3d.geometry.Geometry]):
        """Update geometries efficiently using cache"""
        if not self._is_initialized or not self.vis:
            logger.warning(f"{self.name}: Renderer not initialized")
            return
        logger.debug(f"{self.name}: Updating {len(geometries)} geometries")    
        try:
            # Remove geometries not in the new set
            current_keys = set(self.geometry_cache._cache.keys())
            new_keys = set(geometries.keys())
            
            for key in current_keys - new_keys:
                geom = self.geometry_cache.get(key)
                if geom:
                    self.vis.remove_geometry(geom, reset_bounding_box=False)
                    
            # Add or update geometries
            for key, geometry in geometries.items():
                cached_geom = self.geometry_cache.get(key)
                
                if cached_geom is None:
                    # Add new geometry
                    self.vis.add_geometry(geometry, reset_bounding_box=False)
                    self.geometry_cache.set(key, geometry)
                elif self.geometry_cache.is_dirty(key):
                    # Update existing geometry
                    self.vis.remove_geometry(cached_geom, reset_bounding_box=False)
                    self.vis.add_geometry(geometry, reset_bounding_box=False)
                    self.geometry_cache.set(key, geometry)
            # After the geometry update loop, add:
            if geometries and self._view_control:
                # Calculate scene bounds
                all_bounds = []
                for geom in geometries.values():
                    if hasattr(geom, 'get_axis_aligned_bounding_box'):
                        bbox = geom.get_axis_aligned_bounding_box()
                        if np.all(np.isfinite(bbox.min_bound)) and np.all(np.isfinite(bbox.max_bound)):
                            all_bounds.extend([bbox.min_bound, bbox.max_bound])
                
                if all_bounds:
                    # Set camera to view the scene
                    bounds_array = np.array(all_bounds)
                    scene_min = np.min(bounds_array, axis=0)
                    scene_max = np.max(bounds_array, axis=0)
                    center = (scene_min + scene_max) / 2.0
                    size = np.linalg.norm(scene_max - scene_min)
                    
                    self._view_control.set_lookat(center)
                    self._view_control.set_front([1, 1, -1])  # Look from front-right
                    self._view_control.set_up([0, 0, 1])      # Z is up
                    self._view_control.set_zoom(0.5)
            self.vis.poll_events()
            self.vis.update_renderer()
            logger.debug(f"{self.name}: Geometries updated")                    
        except Exception as e:
            logger.error(f"{self.name}: Error updating geometries: {e}")
    
    def set_camera_parameters(self, params: Dict[str, Any]):
        """Set camera parameters"""
        if not self._is_initialized or not self._view_control:
            return
            
        try:
            if 'lookat' in params:
                self._view_control.set_lookat(params['lookat'])
            if 'front' in params:
                self._view_control.set_front(params['front'])
            if 'up' in params:
                self._view_control.set_up(params['up'])
            if 'zoom' in params:
                self._view_control.set_zoom(float(params['zoom']))
        except Exception as e:
            logger.error(f"{self.name}: Error setting camera: {e}")
    
    def render_frame(self) -> Optional[np.ndarray]:
        """Render a frame and return as numpy array"""
        if not self._is_initialized or not self.vis:
            return None
            
        try:
            import sys
            import threading
            if threading.current_thread() is not threading.main_thread():
                logger.warning(f"{self.name}: render_frame called from non-main thread")
                return None
            if not hasattr(sys, 'getrefcount'):
                return None
            try:
                self.vis.poll_events()
                self.vis.update_renderer()
                img = self.vis.capture_screen_float_buffer(do_render=True)
            finally:
                pass    
            if img is not None:
                arr = (np.asarray(img) * 255).astype(np.uint8)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                return arr
        except Exception as e:
            logger.error(f"{self.name}: Error rendering frame: {e}")
        
        return None
    def reset_view_to_fit_scene(self):
        """Reset camera to fit all visible geometry"""
        if not self._view_control:
            return
        
        try:
            # Get all geometry bounds
            all_bounds = []
            for geom in self.geometry_cache._cache.values():
                if hasattr(geom, 'get_axis_aligned_bounding_box'):
                    bbox = geom.get_axis_aligned_bounding_box()
                    all_bounds.extend([bbox.min_bound, bbox.max_bound])
            
            if all_bounds:
                bounds_array = np.array(all_bounds)
                scene_center = np.mean(bounds_array, axis=0)
                self._view_control.set_lookat(scene_center)
                self._view_control.set_zoom(0.7)
                
        except Exception as e:
            logger.error(f"Error resetting view: {e}")
    def debug_renderer_state(self):
        """Debug helper to check renderer state"""
        print("\n=== RENDERER DEBUG ===")
        if self.main_renderer and self.main_renderer.vis:
            view_ctrl = self.main_renderer.vis.get_view_control()
            if view_ctrl:
                params = view_ctrl.convert_to_pinhole_camera_parameters()
                print(f"Camera extrinsic:\n{params.extrinsic}")
                print(f"Camera intrinsic:\n{params.intrinsic}")
        
        # Check geometry cache
        if self.main_renderer:
            print(f"Cached geometries: {list(self.main_renderer.geometry_cache._cache.keys())}")
            for key, geom in self.main_renderer.geometry_cache._cache.items():
                if hasattr(geom, 'get_axis_aligned_bounding_box'):
                    bbox = geom.get_axis_aligned_bounding_box()
                    print(f"  {key}: bounds {bbox.min_bound} to {bbox.max_bound}")
        print("=====================\n")
        
    def cleanup(self):
        """Clean up resources"""
        if self.vis is not None:
            try:
                self.vis.destroy_window()
            except Exception as e:
                logger.error(f"{self.name}: Error during cleanup: {e}")
            finally:
                self.vis = None
                self._is_initialized = False

# -------------------------------
# Scene Management
# -------------------------------
class Scene3D:
    """3D scene with model loading and geometry management"""
    
    def __init__(self):
        self.model_trimesh: Optional[trimesh.Trimesh] = None
        self.bounds: Optional[np.ndarray] = None
        self.o3d_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.show_ground_plane = True
        self._geometries: Dict[str, o3d.geometry.Geometry] = {}
        self._dirty = True

    def load_model(self, filepath: str) -> bool:
        """Load 3D model from file"""
        try:
            # Load with trimesh for robustness
            print(f"DEBUG: Attempting to load model from {filepath}")
            try:
                loaded = trimesh.load(filepath)
                if isinstance(loaded, trimesh.Scene):
                    loaded = loaded.to_geometry()
                self.model_trimesh = loaded
                self.bounds = self.model_trimesh.bounds
                logger.info(f"Trimesh model loaded: {filepath}")
            except Exception as e:
                logger.warning(f"Trimesh failed: {e}. Trying Open3D...")

            # Load with Open3D
            try:
                mesh_o3d = o3d.io.read_triangle_mesh(filepath, enable_post_processing=True)
                if mesh_o3d is None or len(mesh_o3d.vertices) == 0:
                    raise RuntimeError("Open3D returned empty mesh")
                
                if not mesh_o3d.has_vertex_normals():
                    mesh_o3d.compute_vertex_normals()
                self.o3d_mesh = mesh_o3d
                if self.o3d_mesh:
                    print(f"DEBUG: Meshed loaded- vertices: {len(np.asarray(self.o3d_mesh.vertices))}, faces: {len(np.asarray(self.o3d_mesh.triangles))}")
                    print(f"DEBUG: Mesh bounds: {self.o3d_mesh.get_axis_aligned_bounding_box()}")
                self._geometries['model'] = mesh_o3d
                self._setup_scene_elements()
                self._dirty = True
                print(f"DEBUG: Scene geometries after loading model: {list(self._geometries.keys())}")
                event_bus.publish('scene_updated')
                logger.info("Open3D mesh loaded successfully")
            except Exception as e:
                logger.error(f"Open3D failed to load mesh: {e}")
                return False

            self._setup_scene_elements()
            self._dirty = True
            event_bus.publish('scene_updated')
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _setup_scene_elements(self):
        """Setup coordinate frame and ground plane"""
        if self.bounds is not None:
            bmin, bmax = self.bounds
            center = (bmin + bmax) / 2.0
            size = float(np.max(bmax - bmin) * 2.0 + 1.0)
        else:
            if self.o3d_mesh and len(np.asarray(self.o3d_mesh.vertices)) > 0:
                verts = np.asarray(self.o3d_mesh.vertices)
                center = verts.mean(axis=0)
                size = float(np.ptp(verts, axis=0).max() * 2.0 + 1.0)
            else:
                center = np.array([0.0, 0.0, 0.0])
                size = 10.0

        # Create coordinate frame
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(size * 0.1, 1.0))
        self._geometries['axes'] = axes

        # Create ground plane
        ground_plane = o3d.geometry.TriangleMesh.create_box(width=size, height=0.01, depth=size)
        ground_plane.translate([center[0] - size/2.0, center[1] - 0.005, center[2] - size/2.0])
        ground_plane.compute_vertex_normals()
        ground_plane.paint_uniform_color([0.5, 0.5, 0.5])
        
        if self.show_ground_plane:
            self._geometries['ground'] = ground_plane

    def toggle_ground_plane(self):
        """Toggle ground plane visibility"""
        self.show_ground_plane = not self.show_ground_plane
        
        if self.show_ground_plane and 'ground' not in self._geometries:
            # Need to recreate ground plane
            self._setup_scene_elements()
        elif not self.show_ground_plane and 'ground' in self._geometries:
            del self._geometries['ground']
        
        self._dirty = True
        event_bus.publish('scene_updated')
        return self.show_ground_plane

    def get_geometries(self) -> Dict[str, o3d.geometry.Geometry]:
        """Get all scene geometries"""
        return self._geometries.copy()

    def is_dirty(self) -> bool:
        """Check if scene needs updating"""
        return self._dirty

    def mark_clean(self):
        """Mark scene as clean"""
        self._dirty = False

# -------------------------------
# Trajectory Interpolator
# -------------------------------
class TrajectoryInterpolator:
    """Optimized trajectory interpolation with caching"""
    
    @staticmethod
    def interpolate_linear(waypoints: List[Waypoint], num_points: int = 100) -> np.ndarray:
        """Linear interpolation between waypoints"""
        positions = np.array([wp.position for wp in waypoints])
        if len(positions) < 2:
            return positions.copy()

        # Calculate cumulative distances
        distances = np.zeros(len(positions))
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[i - 1])
            distances[i] = distances[i - 1] + max(dist, 1e-6)

        if distances[-1] == 0:
            return np.tile(positions[0], (num_points, 1))

        # Interpolate each dimension
        interp_x = interp1d(distances, positions[:, 0], kind='linear')
        interp_y = interp1d(distances, positions[:, 1], kind='linear')
        interp_z = interp1d(distances, positions[:, 2], kind='linear')

        t = np.linspace(0, distances[-1], num_points)
        return np.column_stack([interp_x(t), interp_y(t), interp_z(t)])

    @staticmethod
    def interpolate_spline(waypoints: List[Waypoint], num_points: int = 100) -> np.ndarray:
        """Cubic spline interpolation between waypoints"""
        positions = np.array([wp.position for wp in waypoints])
        if len(positions) < 2:
            return positions.copy()

        # Fall back to linear for insufficient points
        if len(positions) < 4:
            return TrajectoryInterpolator.interpolate_linear(waypoints, num_points)

        # Calculate parameterization
        distances = np.zeros(len(positions))
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[i - 1])
            distances[i] = distances[i - 1] + max(dist, 1e-6)

        if distances[-1] == 0:
            return np.tile(positions[0], (num_points, 1))

        # Check for monotonic parameterization
        if not np.all(np.diff(distances) > 0):
            return TrajectoryInterpolator.interpolate_linear(waypoints, num_points)

        try:
            cs_x = CubicSpline(distances, positions[:, 0])
            cs_y = CubicSpline(distances, positions[:, 1])
            cs_z = CubicSpline(distances, positions[:, 2])

            t = np.linspace(0, distances[-1], num_points)
            return np.column_stack([cs_x(t), cs_y(t), cs_z(t)])
        except Exception:
            return TrajectoryInterpolator.interpolate_linear(waypoints, num_points)

# -------------------------------
# Trajectory Parser
# -------------------------------
class TrajectoryParser:
    """Parse trajectory files with robust error handling"""
    
    @staticmethod
    def parse_json(filepath: str) -> List[Waypoint]:
        """Parse JSON trajectory file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            waypoints = []
            
            # Handle different JSON formats
            if 'positions' in data and 'rotations' in data:
                positions = np.array(data['positions'])
                rotations = np.array(data['rotations'])
                num_cameras = len(positions)
                order = data.get('waypoint_order', list(range(num_cameras)))
                
                for i, idx in enumerate(order):
                    if idx < num_cameras:
                        pos = positions[idx]
                        yaw = rotations[idx, 0] if rotations.shape[1] > 0 else 0.0
                        pitch = rotations[idx, 1] if rotations.shape[1] > 1 else 0.0
                        wp = Waypoint(position=pos, yaw=yaw, pitch=pitch, index=i)
                        waypoints.append(wp)
                        
            elif 'cameras' in data and isinstance(data['cameras'], list):
                cameras = data['cameras']
                order = data.get('waypoint_order', list(range(len(cameras))))
                
                for i, idx in enumerate(order):
                    if idx < len(cameras):
                        cam = cameras[idx]
                        wp = Waypoint.from_dict(cam, i)
                        waypoints.append(wp)
                        
            elif 'waypoints' in data and isinstance(data['waypoints'], list):
                for i, w in enumerate(data['waypoints']):
                    try:
                        wp = Waypoint.from_dict(w, i)
                        waypoints.append(wp)
                    except Exception:
                        if isinstance(w, (list, tuple)) and len(w) >= 3:
                            pos = np.array(w[:3], dtype=float)
                            wp = Waypoint(position=pos, yaw=0.0, pitch=0.0, index=i)
                            waypoints.append(wp)
                            
            elif isinstance(data, list):
                for i, w in enumerate(data):
                    try:
                        wp = Waypoint.from_dict(w, i)
                        waypoints.append(wp)
                    except Exception:
                        if isinstance(w, (list, tuple)) and len(w) >= 3:
                            pos = np.array(w[:3], dtype=float)
                            wp = Waypoint(position=pos, yaw=0.0, pitch=0.0, index=i)
                            waypoints.append(wp)

            logger.info(f"Parsed {len(waypoints)} waypoints from {filepath}")
            return waypoints
            
        except Exception as e:
            logger.error(f"Failed to parse trajectory file: {e}")
            return []

    @staticmethod
    def save_json(filepath: str, waypoints: List[Waypoint]):
        """Save waypoints to JSON file"""
        try:
            positions = [wp.position.tolist() for wp in waypoints]
            rotations = [[wp.yaw, wp.pitch] for wp in waypoints]
            
            data = {
                'positions': positions,
                'rotations': rotations,
                'num_cameras': len(waypoints),
                'has_route': True,
                'waypoint_order': list(range(len(waypoints))),
                'metadata': {
                    'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'num_waypoints': len(waypoints)
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved {len(waypoints)} waypoints to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save trajectory file: {e}")

# -------------------------------
# Main Application
# -------------------------------
class UAVVisualizationApp:
    """Main application with persistent renderers and optimized updates"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("UAV Flight Trajectory Visualization")
        self.root.geometry("1400x900")

        # Core components
        self.scene = Scene3D()
        self.trajectories: List[Trajectory] = []
        self.selected_trajectory: Optional[Trajectory] = None
        self.selected_waypoint: Optional[int] = None

        # Persistent renderers
        self.main_renderer: Optional[PersistentRenderer] = None
        self.fpv_renderer: Optional[PersistentRenderer] = None

        # UI state
        self._main_photo = None
        self._fpv_photo = None
        self._last_render_time = 0
        self._render_interval = 1.0 / 30.0  # 30 FPS cap

        # Event subscriptions
        event_bus.subscribe('scene_updated', self._on_scene_updated)
        event_bus.subscribe('trajectory_updated', self._on_trajectory_updated)

        self.setup_gui()
        
        # Initialize renderers after GUI is set up
        self.root.after(100, self._initialize_renderers)
        
        # Start update loop
        self.root.after(33, self._update_loop)

    def setup_gui(self):
        """Set up the user interface"""
        # Create main layout
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # Left panel - controls
        left_frame = ttk.Frame(main_paned, width=300)
        main_paned.add(left_frame, weight=0)

        # Middle panel - 3D view
        middle_frame = ttk.Frame(main_paned)
        main_paned.add(middle_frame, weight=3)

        # Right panel - info
        right_frame = ttk.Frame(main_paned, width=300)
        main_paned.add(right_frame, weight=0)

        self._setup_control_panel(left_frame)
        self._setup_3d_view(middle_frame)
        self._setup_info_panel(right_frame)

    def _setup_control_panel(self, parent):
        """Set up control panel"""
        # File operations
        file_frame = ttk.LabelFrame(parent, text="File Operations")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_frame, text="Load 3D Model", command=self.load_model).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(file_frame, text="Load Trajectory", command=self.load_trajectory).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(file_frame, text="Save Trajectory", command=self.save_trajectory).pack(fill=tk.X, padx=5, pady=2)

        # Trajectory controls
        traj_frame = ttk.LabelFrame(parent, text="Trajectory Controls")
        traj_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.interp_method = tk.StringVar(value="spline")
        ttk.Radiobutton(traj_frame, text="Linear Interpolation", 
                       variable=self.interp_method, value="linear", 
                       command=self._on_interpolation_changed).pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(traj_frame, text="Spline Interpolation", 
                       variable=self.interp_method, value="spline", 
                       command=self._on_interpolation_changed).pack(anchor=tk.W, padx=5)

        ttk.Button(traj_frame, text="Compare Trajectories", command=self.compare_trajectories).pack(fill=tk.X, padx=5, pady=2)

        # Scene controls
        scene_frame = ttk.LabelFrame(parent, text="Scene Controls")
        scene_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(scene_frame, text="Toggle Ground Plane", command=self.toggle_ground_plane).pack(fill=tk.X, padx=5, pady=2)

    def _setup_3d_view(self, parent):
        """Set up 3D view canvas"""
        self.canvas_frame = ttk.Frame(parent)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Set a minimum size and background
        self.main_canvas = tk.Canvas(
            self.canvas_frame, 
            bg='#1a1a1a',  # Dark background so it's not pure white
            width=640,
            height=480
        )
        self.main_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Add loading text initially
        self.main_canvas.create_text(
            320, 240, 
            text="Initializing 3D View...", 
            fill="white", 
            font=("Arial", 16),
            tags="loading"
        )
        
        # Bind resize event
        self.main_canvas.bind("<Configure>", self._on_canvas_resize)

    def _setup_info_panel(self, parent):
        """Set up information panel"""
        # Trajectory info
        info_frame = ttk.LabelFrame(parent, text="Trajectory Information")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.info_text = tk.Text(info_frame, height=8, width=25)
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # FPV preview
        fpv_frame = ttk.LabelFrame(parent, text="First Person View")
        fpv_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fpv_label = ttk.Label(fpv_frame, text="No waypoint selected", background='black', foreground='white')
        self.fpv_label.pack(fill=tk.BOTH, expand=True)
        
        # Waypoint list
        waypoint_frame = ttk.LabelFrame(parent, text="Waypoints")
        waypoint_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.waypoint_listbox = tk.Listbox(waypoint_frame, height=6)
        waypoint_scrollbar = ttk.Scrollbar(waypoint_frame, orient=tk.VERTICAL, command=self.waypoint_listbox.yview)
        self.waypoint_listbox.configure(yscrollcommand=waypoint_scrollbar.set)
        self.waypoint_listbox.bind('<<ListboxSelect>>', self._on_waypoint_select)
        
        self.waypoint_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        waypoint_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.root.after(100, self._intialize_fpv_default)

    def _intialize_fpv_default(self):
        if self.selected_trajectory and len(self.selected_trajectory.waypoints) > 0:
            self.selected_waypoint = 0
            self.waypoint_listbox.selection_set(0)
            self._update_fpv_camera()

    def _initialize_renderers(self):
        """Initialize persistent renderers"""
        try:
            # Get canvas dimensions
            self.root.update_idletasks()
            main_width = max(self.main_canvas.winfo_width(), 640)
            main_height = max(self.main_canvas.winfo_height(), 480)
            
            # Create persistent renderers with offscreen rendering
            self.main_renderer = PersistentRenderer("Main3D", main_width, main_height)
            self.fpv_renderer = PersistentRenderer("FPV3D", 320, 240)
            
            if not self.main_renderer.initialize():
                logger.error("Failed to initialize main renderer")
                return
                
            if not self.fpv_renderer.initialize():
                logger.error("Failed to initialize FPV renderer")
                return
                
            logger.info("Renderers initialized successfully")
            
            # CRITICAL: Set up initial scene with default geometry
            self._setup_default_scene()
            
            # Force initial render to show something
            self._force_initial_render()
            
        except Exception as e:
            logger.error(f"Failed to initialize renderers: {e}")

    def _setup_default_scene(self):
        """Setup a default scene so user sees something initially"""
        # Create a simple default geometry
        default_geometries = {}
        
        # Add coordinate frame
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        default_geometries['axes'] = axes
        
        # Add a ground plane
        ground = o3d.geometry.TriangleMesh.create_box(width=20, height=0.01, depth=20)
        ground.translate([-10, -0.05, -10])
        ground.paint_uniform_color([0.3, 0.3, 0.3])
        ground.compute_vertex_normals()
        default_geometries['ground'] = ground
        
        # Update both renderers
        self.main_renderer.update_geometries(default_geometries)
        if self.main_renderer._view_control:
            self.main_renderer._view_control.set_lookat([0, 0, 0])
            self.main_renderer._view_control.set_front([0, -1, -1])  # Look from top-front
            self.main_renderer._view_control.set_up([0, 0, 1])       # Z is up
            self.main_renderer._view_control.set_zoom(0.7)
        self.fpv_renderer.update_geometries(default_geometries)



    def _force_initial_render(self):
        """Force an initial render to populate the canvas"""
        if self.main_renderer:
            frame = self.main_renderer.render_frame()
            if frame is not None:
                self._display_main_frame(frame)
    def _update_loop(self):
        """Main update loop with frame rate limiting"""
        current_time = time.time()
        if hasattr(self, '_processing'):
            self.root.after(33, self._update_loop)
            return
        self._processing = True
        # Only render if enough time has passed
        if current_time - self._last_render_time >= self._render_interval:
            try:
                # Render main view
                if self.main_renderer:
                    frame = self.main_renderer.render_frame()
                    if frame is not None:
                        self._display_main_frame(frame)
                
                # Render FPV
                if self.fpv_renderer and self.selected_trajectory and self.selected_waypoint is not None:
                    self._update_fpv_camera()
                    fpv_frame = self.fpv_renderer.render_frame()
                    if fpv_frame is not None:
                        self._display_fpv_frame(fpv_frame)
                
                self._last_render_time = current_time
                
            finally:
                self._processing = False
        
        # Schedule next update
        self.root.after(33, self._update_loop)

    def _on_scene_updated(self, _):
        """Handle scene update events"""
        self._update_scene_rendering()

    def _on_trajectory_updated(self, _):
        """Handle trajectory update events"""
        self._update_trajectory_rendering()
        self._update_info_display()
        self._update_waypoint_list()

    def _update_scene_rendering(self):
        """Update scene geometries in renderers"""
        if not self.main_renderer:
            return
            
        try:
            geometries = self.scene.get_geometries()
            
            # If no geometries from scene, use defaults
            if not geometries:
                self._setup_default_scene()
                return
                
            self.main_renderer.update_geometries(geometries)
            
            if 'model' in geometries:
                model_geom = geometries['model']
                bbox = model_geom.get_axis_aligned_bounding_box()
                center = bbox.get_center()
                extent = bbox.get_extent()
                if np.all(np.isfinite(center)) and np.all(np.isfinite(extent)):
                    view_control = self.main_renderer._view_control
                    if view_control:
                        distance = np.max(extent) * 2.5
                        view_control.set_lookat(center)
                        view_control.set_front([0.5, -0.5, -0.7])  # Look from top-front
                        view_control.set_up([0, 0, 1])       # Z is up
                        #zoom = 0.7 * (20.0/ max(extent))
                        view_control.set_zoom(0.8)
            
            # Reset camera to view the scene
            self.main_renderer.reset_view_to_fit_scene()
            
            # Also update FPV renderer with scene
            if self.fpv_renderer:
                self.fpv_renderer.update_geometries(geometries)
            
            # Force immediate render after scene update
            frame = self.main_renderer.render_frame()
            if frame is not None:
                self._display_main_frame(frame)
                
        except Exception as e:
            logger.error(f"Error updating scene rendering: {e}")

    def _update_trajectory_rendering(self):
        """Update trajectory visualization"""
        if not self.main_renderer:
            return
            
        try:
            trajectory_geometries = {}
            
            for i, traj in enumerate(self.trajectories):
                # Create trajectory path
                path_points = traj.get_interpolated_path(num_points=200)
                if len(path_points) >= 2:
                    lines = [[j, j+1] for j in range(len(path_points)-1)]
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(path_points)
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_set.paint_uniform_color(traj.color)
                    trajectory_geometries[f"trajectory_{i}"] = line_set
                
                # Create waypoint markers
                waypoint_positions = traj.get_positions()
                if len(waypoint_positions) > 0:
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(waypoint_positions)
                    point_cloud.paint_uniform_color(traj.color)
                    trajectory_geometries[f"waypoints_{i}"] = point_cloud
                    
                    # Create camera visualizations for waypoints
                    for j, wp in enumerate(traj.waypoints):
                        camera_geom = self._create_camera_visualization(
                            wp.position, wp.yaw, wp.pitch, scale=0.3, color=traj.color
                        )
                        trajectory_geometries[f"camera_{i}_{j}"] = camera_geom
            
            # Combine scene and trajectory geometries
            all_geometries = self.scene.get_geometries()
            all_geometries.update(trajectory_geometries)
            
            self.main_renderer.update_geometries(all_geometries)
            
        except Exception as e:
            logger.error(f"Error updating trajectory rendering: {e}")

    def _create_camera_visualization(self, position, yaw, pitch, scale=0.5, color=None):
        """Create camera visualization geometry"""
        if color is None:
            color = [1.0, 0.0, 0.0]
            
        # Convert angles to radians
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        
        # Define camera frustum points
        points = [
            [0, 0, 0],  # Camera center
            [-scale, -scale, scale*2],  # Frustum corners
            [scale, -scale, scale*2],
            [scale, scale, scale*2],
            [-scale, scale, scale*2]
        ]
        
        # Rotation matrices
        rot_yaw = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        
        rot_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])
        
        # Transform points
        transformed_points = []
        for point in points:
            rotated = np.dot(rot_pitch, np.dot(rot_yaw, point))
            transformed_points.append(rotated + position)
        
        # Create line set
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # From center to corners
            [1, 2], [2, 3], [3, 4], [4, 1]   # Around frustum
        ]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(transformed_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color(color)
        
        return line_set

    def _update_fpv_camera(self):
        """Update FPV camera position and orientation"""
        if not self.fpv_renderer or not self.selected_trajectory or self.selected_waypoint is None:
            return
            
        try:
            wp = self.selected_trajectory.waypoints[self.selected_waypoint]
            
            # Calculate view direction from yaw and pitch
            yaw_rad = np.radians(wp.yaw)
            pitch_rad = np.radians(wp.pitch)
            
            forward = np.array([
                np.cos(pitch_rad) * np.sin(yaw_rad),
                np.sin(pitch_rad),
                np.cos(pitch_rad) * np.cos(yaw_rad)
            ])
            
            forward = forward / np.linalg.norm(forward)
            
            # Set camera parameters
            params = {
                'front': forward.tolist(),
                'up': [0.0, 1.0, 0.0],
                'lookat': (wp.position + forward * 2.0).tolist(),
                'zoom': 0.5
            }
            
            self.fpv_renderer.set_camera_parameters(params)
            
        except Exception as e:
            logger.error(f"Error updating FPV camera: {e}")

    def _display_main_frame(self, frame: np.ndarray):
        """Display frame in main canvas"""
        try:
            img = Image.fromarray(frame)
            canvas_width = self.main_canvas.winfo_width()
            canvas_height = self.main_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                img = img.resize((canvas_width, canvas_height), Image.LANCZOS)
                self._main_photo = ImageTk.PhotoImage(img)
                
                self.main_canvas.delete("all")
                self.main_canvas.create_image(0, 0, anchor=tk.NW, image=self._main_photo)
                
        except Exception as e:
            logger.error(f"Error displaying main frame: {e}")

    def _display_fpv_frame(self, frame: np.ndarray):
        """Display frame in FPV view"""
        try:
            img = Image.fromarray(frame)
            label_width = self.fpv_label.winfo_width()
            label_height = self.fpv_label.winfo_height()
            
            if label_width > 1 and label_height > 1:
                img = img.resize((label_width, label_height), Image.LANCZOS)
                self._fpv_photo = ImageTk.PhotoImage(img)
                self.fpv_label.configure(image=self._fpv_photo, text="")
                
        except Exception as e:
            logger.error(f"Error displaying FPV frame: {e}")

    def _on_canvas_resize(self, event):
        """Handle canvas resize - update viewport only"""
        if self.main_renderer:
            self.main_renderer.update_viewport(event.width, event.height)

    def _on_interpolation_changed(self):
        """Handle interpolation method change"""
        if not self.selected_trajectory:
            return
            
        method = InterpolationMethod.LINEAR if self.interp_method.get() == "linear" else InterpolationMethod.SPLINE
        
        if self.selected_trajectory.interpolation_method != method:
            self.selected_trajectory.interpolation_method = method
            self.selected_trajectory.invalidate_cache()
            event_bus.publish('trajectory_updated')

    def _on_waypoint_select(self, event):
        """Handle waypoint selection"""
        selection = event.widget.curselection()
        if not selection or not self.selected_trajectory:
            return
            
        self.selected_waypoint = selection[0]
        self._update_fpv_camera()

    def _update_info_display(self):
        """Update trajectory information display"""
        self.info_text.delete(1.0, tk.END)
        
        if not self.selected_trajectory:
            self.info_text.insert(tk.END, "No trajectory selected\n")
            return
            
        try:
            metrics = self.selected_trajectory.calculate_metrics()
            
            self.info_text.insert(tk.END, f"TRAJECTORY: {self.selected_trajectory.name}\n")
            self.info_text.insert(tk.END, "=" * 30 + "\n\n")
            self.info_text.insert(tk.END, f"Waypoints: {metrics['num_waypoints']}\n")
            self.info_text.insert(tk.END, f"Total length: {metrics['total_length']:.2f} m\n")
            self.info_text.insert(tk.END, f"Total vertical: {metrics['total_vertical']:.2f} m\n")
            self.info_text.insert(tk.END, f"Est. duration: {metrics['total_duration']:.1f} s\n")
            self.info_text.insert(tk.END, f"Sharp corners: {metrics['sharp_corners']}\n")
            self.info_text.insert(tk.END, f"Interpolation: {self.selected_trajectory.interpolation_method.value}\n")
            
        except Exception as e:
            logger.error(f"Error updating info display: {e}")
            self.info_text.insert(tk.END, "Error calculating metrics\n")

    def _update_waypoint_list(self):
        """Update waypoint list"""
        self.waypoint_listbox.delete(0, tk.END)
        
        if not self.selected_trajectory:
            return
            
        for i, wp in enumerate(self.selected_trajectory.waypoints):
            pos_str = f"[{wp.position[0]:.1f}, {wp.position[1]:.1f}, {wp.position[2]:.1f}]"
            angle_str = f"(yaw: {wp.yaw:.1f}Ã‚Â°, pitch: {wp.pitch:.1f}Ã‚Â°)"
            self.waypoint_listbox.insert(tk.END, f"{i}: {pos_str} {angle_str}")
            
    def debug_renderer_state(self):
        """Debug helper to check renderer state"""
        print("\n=== RENDERER DEBUG ===")
        if self.main_renderer and self.main_renderer.vis:
            view_ctrl = self.main_renderer.vis.get_view_control()
            if view_ctrl:
                params = view_ctrl.convert_to_pinhole_camera_parameters()
                print(f"Camera extrinsic:\n{params.extrinsic}")
                print(f"Camera intrinsic:\n{params.intrinsic}")
        
        # Check geometry cache
        if self.main_renderer:
            print(f"Cached geometries: {list(self.main_renderer.geometry_cache._cache.keys())}")
            for key, geom in self.main_renderer.geometry_cache._cache.items():
                if hasattr(geom, 'get_axis_aligned_bounding_box'):
                    bbox = geom.get_axis_aligned_bounding_box()
                    print(f"  {key}: bounds {bbox.min_bound} to {bbox.max_bound}")
        print("=====================\n")

# Call this after loading a model:
# self.debug_renderer_state()
    # Core functionality methods
    def load_model(self):
        """Load 3D model file"""
        filetypes = [
            ("3D Models", "*.obj *.ply *.stl"),
            ("OBJ files", "*.obj"),
            ("PLY files", "*.ply"),
            ("STL files", "*.stl"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select 3D Model",
            filetypes=filetypes
        )
        
        if not filepath:
            return
            
        try:
            success = self.scene.load_model(filepath)
            if success:
                messagebox.showinfo("Success", "Model loaded successfully!")
                self.debug_scene_state()
                self.debug_renderer_state()
                self._update_scene_rendering()
                if self.main_renderer:
                    frame = self.main_renderer.render_frame()
                    if frame is not None:
                        self._display_main_frame(frame)
            else:
                messagebox.showerror("Error", "Failed to load model")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def load_trajectory(self):
        """Load trajectory file"""
        filepath = filedialog.askopenfilename(
            title="Select Trajectory File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
            
        try:
            waypoints = TrajectoryParser.parse_json(filepath)
            if not waypoints:
                messagebox.showerror("Error", "Failed to parse trajectory file")
                return
            
            # Create trajectory
            name = Path(filepath).stem
            color = (np.random.rand(), np.random.rand(), np.random.rand())
            trajectory = Trajectory(name, waypoints, color)
            
            self.trajectories.append(trajectory)
            self.selected_trajectory = trajectory
            self.selected_waypoint = None
            
            event_bus.publish('trajectory_updated')
            
            messagebox.showinfo("Success", f"Loaded trajectory with {len(waypoints)} waypoints")
            
        except Exception as e:
            logger.error(f"Error loading trajectory: {e}")
            messagebox.showerror("Error", f"Failed to load trajectory: {e}")

    def save_trajectory(self):
        """Save current trajectory"""
        if not self.selected_trajectory:
            messagebox.showwarning("Warning", "No trajectory selected")
            return
            
        filepath = filedialog.asksaveasfilename(
            title="Save Trajectory",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
            
        try:
            TrajectoryParser.save_json(filepath, self.selected_trajectory.waypoints)
            messagebox.showinfo("Success", "Trajectory saved successfully!")
        except Exception as e:
            logger.error(f"Error saving trajectory: {e}")
            messagebox.showerror("Error", f"Failed to save trajectory: {e}")

    def toggle_ground_plane(self):
        """Toggle ground plane visibility"""
        try:
            is_visible = self.scene.toggle_ground_plane()
            self._update_scene_rendering()
            
            status = "visible" if is_visible else "hidden"
            messagebox.showinfo("Ground Plane", f"Ground plane is now {status}")
        except Exception as e:
            logger.error(f"Error toggling ground plane: {e}")

    def compare_trajectories(self):
        """Compare multiple trajectories"""
        if len(self.trajectories) < 2:
            messagebox.showwarning("Warning", "Need at least 2 trajectories to compare")
            return
            
        try:
            # Create comparison window
            compare_window = tk.Toplevel(self.root)
            compare_window.title("Trajectory Comparison")
            compare_window.geometry("800x600")
            
            # Create text widget for comparison results
            text_widget = tk.Text(compare_window, wrap=tk.WORD)
            scrollbar = ttk.Scrollbar(compare_window, orient=tk.VERTICAL, command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Generate comparison data
            comparison_data = []
            for traj in self.trajectories:
                metrics = traj.calculate_metrics()
                comparison_data.append({
                    'name': traj.name,
                    'metrics': metrics
                })
            
            # Find most efficient
            most_efficient = min(comparison_data, key=lambda x: x['metrics']['total_duration'])
            
            # Display comparison
            text_widget.insert(tk.END, "TRAJECTORY COMPARISON\n")
            text_widget.insert(tk.END, "=" * 50 + "\n\n")
            
            # Header
            header = f"{'Name':<20} {'Length(m)':<12} {'Vertical(m)':<12} {'Duration(s)':<12} {'Waypoints':<10}\n"
            text_widget.insert(tk.END, header)
            text_widget.insert(tk.END, "-" * 70 + "\n")
            
            # Data rows
            for data in comparison_data:
                metrics = data['metrics']
                is_best = " (BEST)" if data['name'] == most_efficient['name'] else ""
                
                row = (f"{data['name']:<20} {metrics['total_length']:<12.2f} "
                      f"{metrics['total_vertical']:<12.2f} {metrics['total_duration']:<12.2f} "
                      f"{metrics['num_waypoints']:<10} {is_best}\n")
                text_widget.insert(tk.END, row)
            
            text_widget.insert(tk.END, f"\nMost efficient: {most_efficient['name']}")
            
        except Exception as e:
            logger.error(f"Error comparing trajectories: {e}")
            messagebox.showerror("Error", f"Failed to compare trajectories: {e}")
    def debug_scene_state(self):
        """Debug method to check current scene state"""
        print("=== DEBUG SCENE STATE ===")
        print(f"Main renderer initialized: {self.main_renderer._is_initialized if self.main_renderer else False}")
        print(f"FPV renderer initialized: {self.fpv_renderer._is_initialized if self.fpv_renderer else False}")
        
        if self.scene.o3d_mesh:
            bbox = self.scene.o3d_mesh.get_axis_aligned_bounding_box()
            print(f"Model bounds: {bbox.min_bound} to {bbox.max_bound}")
            print(f"Model center: {bbox.get_center()}")
        
        print(f"Scene geometries: {list(self.scene._geometries.keys())}")
        print("=========================")
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.main_renderer:
                self.main_renderer.cleanup()
            if self.fpv_renderer:
                self.fpv_renderer.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """Main entry point"""
    root = tk.Tk()
    app = UAVVisualizationApp(root)
    
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            app.cleanup()
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        app.cleanup()

if __name__ == '__main__':
    main()