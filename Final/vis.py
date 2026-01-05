"""
UAV Flight Trajectory Visualization System

This application provides a comprehensive 3D visualization environment for 
Unmanned Aerial Vehicle (UAV) flight trajectories. It features:
- 3D model loading and rendering using Open3D
- Trajectory planning and waypoint management
- Real-time flight simulation with FPV (First Person View)
- Video recording capabilities
- Interactive camera controls and trajectory analysis

Main Components:
1. GUI Framework (Tkinter-based with collapsible panels)
2. 3D Rendering Engine (Open3D with threaded rendering)
3. Trajectory Management System
4. Flight Simulation Engine
5. Video Recording System
6. Camera Control System
7. Utility Functions and Data Structures
"""

import sys
import os
import time
import json
import queue
from threading import Lock, Event, Thread
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Set, Callable
from enum import Enum, auto 
from collections import deque
import logging
from concurrent.futures import ThreadPoolExecutor
import glfw  # For direct window resize
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
# Collapsible Frame Widget
# -------------------------------
class CollapsibleFrame(ttk.Frame):
    """
    A custom collapsible/expandable frame widget for organizing UI controls.
    
    This provides a space-efficient way to group related controls that can be
    shown or hidden based on user preference.
    
    Features:
    - Toggle button with arrow indicators (▼ for expanded, ► for collapsed)
    - Smooth show/hide transitions
    - Customizable title text
    """
    
    def __init__(self, parent, text="", **kwargs):
        """
        Initialize collapsible frame.
        
        Args:
            parent: Parent widget
            text: Display text for the frame header
            **kwargs: Additional Frame configuration options
        """
        super().__init__(parent, **kwargs)
        # Track expansion state
        self.show = tk.BooleanVar(value=True)
        self.text = text
        # Create header frame with toggle button
        self.title_frame = ttk.Frame(self)
        self.title_frame.pack(fill=tk.X, pady=(0, 2))
        # Toggle button with dynamic text
        self.toggle_button = ttk.Button(
            self.title_frame, 
            text=f"▼ {self.text}", 
            command=self.toggle
        )
        self.toggle_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # Content area that can be shown/hidden
        self.content_frame = ttk.Frame(self, padding=8)
        self.content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
    
    def toggle(self):
        """Toggle visibility of the content frame."""
        if self.show.get():
            # Hide content
            self.content_frame.pack_forget()
            self.toggle_button.config(text=f"► {self.text}")
            self.show.set(False)
        else:
            # Show content
            self.content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
            self.toggle_button.config(text=f"▼ {self.text}")
            self.show.set(True)
    
    def get_content_frame(self):
        """Get the content frame for adding child widgets."""
        return self.content_frame

# -------------------------------
# Data structures and utilities
# -------------------------------
class InterpolationMethod(Enum):
    """
    Enumeration of available trajectory interpolation methods.
    
    LINEAR: Straight line segments between waypoints
    SPLINE: Smooth cubic spline interpolation for curved paths
    """
    LINEAR = "linear"
    SPLINE = "spline"

@dataclass
class Waypoint:
    """
    Represents a single waypoint in a UAV flight trajectory.
    
    A waypoint defines a specific position and orientation in 3D space
    that the UAV will pass through during its flight.
    
    Attributes:
        position: 3D coordinates [x, y, z] as numpy array
        yaw: Horizontal rotation angle in degrees (-180 to 180)
        pitch: Vertical rotation angle in degrees (-90 to 90)
        index: Sequential position in the trajectory
    """
    position: np.ndarray
    yaw: float
    pitch: float
    index: int

    def to_dict(self) -> Dict:
        """Convert waypoint to serializable dictionary for JSON export."""
        return {
            'position': self.position.tolist(),
            'yaw': self.yaw,
            'pitch': self.pitch,
            'index': self.index
        }

    @classmethod
    def from_dict(cls, data: Dict, index: int) -> 'Waypoint':
        """
        Create Waypoint from dictionary with flexible format support.
        
        Supports multiple input formats for backward compatibility:
        - Direct position arrays
        - Structured camera data
        - Geographic coordinates (lon, lat, alt)
        """
        # Handle list/tuple format for positions
        if isinstance(data, (list, tuple)) and len(data) >= 3:
            pos = np.array(data[:3], dtype=float)
            return cls(position=pos, yaw=0.0, pitch=0.0, index=index)
        
        # Handle structured camera data
        if 'position' in data and isinstance(data['position'], (list, tuple)):
            pos = np.array(data['position'][:3], dtype=float)
            return cls(position=pos, yaw=data.get('yaw', 0.0), pitch=data.get('pitch', 0.0), index=index)
        
        # Handle geographic coordinate formats
        x = data.get('x') if 'x' in data else data.get('lon') if 'lon' in data else data.get('longitude')
        y = data.get('y') if 'y' in data else data.get('lat') if 'lat' in data else data.get('latitude')
        z = data.get('z') if 'z' in data else data.get('alt') if 'alt' in data else data.get('altitude')
        if x is not None and y is not None and z is not None:
            pos = np.array([float(x), float(y), float(z)], dtype=float)
            return cls(position=pos, yaw=data.get('yaw', 0.0), pitch=data.get('pitch', 0.0), index=index)
        raise ValueError("Unrecognized waypoint format")

@dataclass
class Trajectory:

    """
    Represents a complete UAV flight trajectory composed of multiple waypoints.
    
    A trajectory defines the complete flight path including waypoint positions,
    camera orientations, and interpolation method for path generation.
    
    Attributes:
        name: Descriptive name for the trajectory
        waypoints: Ordered list of waypoints defining the path
        color: RGB color for visualization (0-1 range)
        interpolation_method: Method for path interpolation between waypoints
        _cached_paths: Cache for computed paths to avoid recalculation
        _cache_key: Hash key for cache validation
    """
    name: str
    waypoints: List[Waypoint]
    color: Tuple[float, float, float]
    interpolation_method: InterpolationMethod = InterpolationMethod.SPLINE
    _cached_paths: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _cache_key: Optional[int] = field(default=None, init=False, repr=False)

    def get_positions(self) -> np.ndarray:
        """Extract all waypoint positions as a numpy array."""
        return np.array([wp.position for wp in self.waypoints])
    
    def _make_cache_key(self) -> int:
        """Generate cache key based on waypoint positions and interpolation method."""
        pos = self.get_positions().ravel().astype(np.float32)
        return hash((pos.tobytes(), self.interpolation_method.value))


    def calculate_metrics(self, cruising_speed: float = 5.0, hover_time: float = 2.5) -> Dict:
        """
        Calculate basic flight metrics for the trajectory.
        
        Args:
            cruising_speed: UAV speed in m/s between waypoints
            hover_time: Time in seconds to hover at each waypoint
            
        Returns:
            Dictionary containing:
            - total_length: Total path length in meters
            - total_vertical: Cumulative vertical movement
            - total_duration: Estimated total flight time
            - sharp_corners: Count of sharp turns (>90 degrees)
            - num_waypoints: Number of waypoints
        """
        positions = self.get_positions()
        if len(positions) < 2:
            return {
                'total_length': 0.0,
                'total_vertical': 0.0,
                'total_duration': 0.0,
                'sharp_corners': 0,
                'num_waypoints': len(self.waypoints)
            }
        # Calculate distances between consecutive waypoints
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_length = np.sum(distances)
        # Calculate vertical changes
        vertical_changes = np.abs(np.diff(positions[:, 2])) if positions.shape[1] > 2 else np.array([0.0])
        total_vertical = np.sum(vertical_changes)
        # Calculate flight time components
        flight_time = total_length / cruising_speed
        total_hover = hover_time * len(self.waypoints)
        total_duration = flight_time + total_hover
        # Detect sharp corners (angles > 90 degrees)
        sharp_corners = 0
        for i in range(1, len(positions) - 1):
            v1 = positions[i] - positions[i - 1]
            v2 = positions[i + 1] - positions[i]
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                continue
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
            if np.degrees(angle) > 90:
                sharp_corners += 1
        return {
            'total_length': total_length,
            'total_vertical': total_vertical,
            'total_duration': total_duration,
            'sharp_corners': sharp_corners,
            'num_waypoints': len(self.waypoints)
        }

    @staticmethod
    def _sum_turning_angles(positions: np.ndarray) -> float:
        if positions.shape[0] < 3:
            return 0.0
        v1 = positions[1:-1] - positions[:-2]
        v2 = positions[2:] - positions[1:-1]
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        valid = (n1 > 0) & (n2 > 0)
        if not np.any(valid):
            return 0.0
        dot = np.einsum('ij,ij->i', v1[valid], v2[valid])
        cos_angle = np.clip(dot / (n1[valid] * n2[valid]), -1.0, 1.0)
        return float(np.sum(np.arccos(cos_angle)))

    
    def calculate_detailed_metrics(self, cruising_speed: float = 10.0) -> Dict:
        """Advanced metrics for trajectory comparison"""
        positions = self.get_positions()
        if len(positions) < 2:
            return {
                'total_length': 0.0,
                'cumulative_vertical_displacement': 0.0,
                'cumulative_turning_angle': 0.0,
                'estimated_duration': 0.0,
                'num_waypoints': len(self.waypoints),
            }
        
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_length = np.sum(distances)
        
        if positions.shape[1] > 2:
            vertical_differences = np.abs(np.diff(positions[:, 2]))
            cumulative_vertical_displacement = np.sum(vertical_differences)
        else:
            cumulative_vertical_displacement = 0.0
        
 
        cumulative_turning_angle = Trajectory._sum_turning_angles(positions)

        estimated_duration = total_length / cruising_speed
        
        return {
            'total_length': total_length,
            'cumulative_vertical_displacement': cumulative_vertical_displacement,
            'cumulative_turning_angle': cumulative_turning_angle,
            'estimated_duration': estimated_duration,
            'num_waypoints': len(self.waypoints),
        }
    
    @staticmethod
    def calculate_efficiency_rating(metrics):
        """Score trajectory efficiency 0-10 based on multiple factors"""
        if metrics['total_length'] == 0:
            return 0
        length_factor = min(1.0, 1000 / metrics['total_length'])
        waypoint_factor = min(1.0, 20 / metrics['num_waypoints'])
        turning_factor = 1.0 - (min(metrics['cumulative_turning_angle'] / np.pi, 1.0))
        vertical_factor = 1.0 - (min(metrics['cumulative_vertical_displacement'] / metrics['total_length'], 0.5))
        rating = (length_factor * 0.3 + waypoint_factor * 0.2 + 
                 turning_factor * 0.3 + vertical_factor * 0.2) * 10
        return min(rating, 10.0)

# -------------------------------
# Trajectory parser
# -------------------------------
class TrajectoryParser:
    @staticmethod
    def parse_json(filepath: str) -> List[Waypoint]:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            waypoints = []
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
                    cam = cameras[idx]
                    wp = Waypoint.from_dict(cam, i)
                    waypoints.append(wp)
            elif 'waypoints' in data and isinstance(data['waypoints'], list):
                for i, w in enumerate(data['waypoints']):
                    try:
                        wp = Waypoint.from_dict(w, i)
                    except Exception:
                        if isinstance(w, (list, tuple)):
                            wp = Waypoint(position=np.array(w[:3], dtype=float), yaw=0.0, pitch=0.0, index=i)
                        else:
                            raise
                    waypoints.append(wp)
            else:
                if isinstance(data, list):
                    for i, w in enumerate(data):
                        try:
                            wp = Waypoint.from_dict(w, i)
                        except Exception:
                            if isinstance(w, (list, tuple)):
                                wp = Waypoint(position=np.array(w[:3], dtype=float), yaw=0.0, pitch=0.0, index=i)
                            else:
                                raise
                        waypoints.append(wp)
                else:
                    raise ValueError("Unrecognized trajectory JSON format")
            logger.info(f"Parsed {len(waypoints)} waypoints from {filepath}")
            return waypoints
        except Exception as e:
            logger.error(f"Failed to parse trajectory file: {e}")
            return []

    @staticmethod
    def save_json(filepath: str, waypoints: List[Waypoint]):
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
# Interpolation helpers
# -------------------------------
class TrajectoryInterpolator:
    @staticmethod
    def interpolate_linear(waypoints: List[Waypoint], num_points: int = 100) -> np.ndarray:
        """Straight line interpolation between waypoints"""
        positions = np.array([wp.position for wp in waypoints])
        if len(positions) < 2:
            return positions.copy()
        distances = np.zeros(len(positions))
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[i - 1])
            distances[i] = distances[i - 1] + max(dist, 1e-6)
        if distances[-1] == 0:
            return np.tile(positions[0], (num_points, 1))
        interp_x = interp1d(distances, positions[:, 0], kind='linear')
        interp_y = interp1d(distances, positions[:, 1], kind='linear')
        interp_z = interp1d(distances, positions[:, 2], kind='linear')
        t = np.linspace(0, distances[-1], num_points)
        return np.column_stack([interp_x(t), interp_y(t), interp_z(t)])

    @staticmethod
    def interpolate_spline(waypoints: List[Waypoint], num_points: int = 100) -> np.ndarray:
        """Smooth cubic spline interpolation (falls back to linear if fails)"""
        positions = np.array([wp.position for wp in waypoints])
        if len(positions) < 2:
            return positions.copy()
        distances = np.zeros(len(positions))
        distances = TrajectoryInterpolator._cumdist_np(positions)

        if distances[-1] == 0:
            return np.tile(positions[0], (num_points, 1))
        if len(positions) < 4:
            return TrajectoryInterpolator.interpolate_linear(waypoints, num_points)
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

    @staticmethod
    def _cumdist_np(positions: np.ndarray) -> np.ndarray:
        diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        return np.concatenate(([0.0], np.cumsum(np.maximum(diffs, 1e-12))))


# -------------------------------
# Camera class
# -------------------------------
class Camera3D:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 10.0], dtype=float)
        self.yaw = 0.0
        self.pitch = 0.0
        self.fov = 60.0

    def set_from_waypoint(self, waypoint: Waypoint):
        self.position = waypoint.position.copy()
        self.yaw = waypoint.yaw
        self.pitch = waypoint.pitch

# -------------------------------
# Open3D rendering thread
# -------------------------------
class Open3DRenderThread(threading.Thread):
    """Optimized asynchronous Open3D renderer with thread safety and non-blocking operations"""
    
    def __init__(self, width: int, height: int, name: str = "Open3DRender"):
        super().__init__(daemon=True)
        self.width = width
        self.height = height
        self.name = name
        
        self._cmd_queue = queue.Queue(maxsize=100)  # Bounded queue
        self._frame_callbacks = []
        self._callback_lock = threading.Lock()
        self._should_stop = threading.Event()
        self._geometry_dirty = threading.Event()
        self._camera_dirty = threading.Event()
        self._resize_dirty = threading.Event()
        self._resize_lock = threading.Lock()
        self._geometries = []
        self._camera_params = None
        self._new_size = None
        self._is_resizing = False
        self.vis = None
        self._last_render_time = 0
        self._min_frame_interval = 1.0 / 30.0  # 30 FPS limit
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._last_geometry_update = 0
        self._geometry_update_interval = 0.1  # Update geometries at most every 100ms
        self._callback_executor = ThreadPoolExecutor(max_workers=4)  # reuse threads instead of spawning thousands
        self._initialization_complete = threading.Event()



    def add_frame_callback(self, callback: Callable[[np.ndarray], None]):
        with self._callback_lock:
            self._frame_callbacks.append(callback)
    
    def remove_frame_callback(self, callback: Callable[[np.ndarray], None]):
        with self._callback_lock:
            if callback in self._frame_callbacks:
                self._frame_callbacks.remove(callback)
    
    def set_geometries(self, geometries: list):
        self._cmd_queue.put(('set_geometries', geometries))
        self._geometry_dirty.set()
    
    def set_camera(self, params: dict):
        self._cmd_queue.put(('set_camera', params))
        self._camera_dirty.set()
    
    def resize(self, width: int, height: int):
        """FIXED: Improved resize with debouncing and error handling"""
        # Ignore invalid sizes
        if width <= 0 or height <= 0:
            logger.warning(f"{self.name}: Invalid resize dimensions: {width}x{height}")
            return
            
        # Debounce rapid resize events
        with self._resize_lock:
            if self._is_resizing:
                logger.debug(f"{self.name}: Resize already in progress, updating target size")
                self._new_size = (width, height)
                return
                
            # Check if size actually changed
            if width == self.width and height == self.height:
                return
                
            self._new_size = (width, height)
            self._is_resizing = True
            
        # Queue resize command with higher priority
        try:
            # Clear any pending resize commands
            temp_queue = queue.Queue(maxsize=100)
            while True:
                try:
                    cmd, data = self._cmd_queue.get_nowait()
                    if cmd != 'resize':  # Keep non-resize commands
                        temp_queue.put((cmd, data))
                except queue.Empty:
                    break
            
            # Restore non-resize commands
            while not temp_queue.empty():
                try:
                    self._cmd_queue.put(temp_queue.get_nowait())
                except queue.Full:
                    break
                    
            # Add new resize command
            self._cmd_queue.put(('resize', (width, height)), block=False)
            self._resize_dirty.set()
            
        except queue.Full:
            logger.warning(f"{self.name}: Command queue full during resize")
    
    def stop(self):
        self._should_stop.set()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.join(timeout=1.0)
    
    def run(self):
        """Main render loop with FPS control and command processing"""
        try:
            self._initialize_renderer()
            self._render_loop()
        except Exception as e:
            logger.error(f"{self.name}: Render thread error: {e}")
        finally:
            self._cleanup_renderer()
    
    def _initialize_renderer(self):
        try:
            import open3d as o3d
            backends = ['OpenGL', 'GLFW', 'headless']  # Keep original fallback loop
            for backend in backends:
                try:
                    logger.debug(f"{self.name}: Attempting {backend}")
                    self.vis = o3d.visualization.Visualizer()
                    success = self.vis.create_window(
                        window_name=self.name, 
                        width=self.width, 
                        height=self.height, 
                        visible=False  # Headless
                    )
                    if success:
                        logger.info(f"{self.name}: Visualizer initialized with {backend}")
                        self._initialization_complete.set()
                        return
                    else:
                        logger.warning(f"Window creation failed with {backend}")
                        if self.vis:
                            self.vis.destroy_window()
                            self.vis = None
                except Exception as e:
                    logger.warning(f"Backend {backend} failed: {e}")
                    if self.vis:
                        self.vis.destroy_window()
                        self.vis = None
                    continue
            logger.error(f"{self.name}: All backends failed")
            self.vis = None
        except Exception as e:
            logger.error(f"{self.name}: Init error: {e}")
            self.vis = None
        finally:
            self._initialization_complete.set()
    
    def _cleanup_renderer(self):
        try:
            if self.vis is not None:
                self.vis.destroy_window()
                self.vis = None
        except Exception:
            pass
        logger.info(f"{self.name}: Renderer cleaned up")
    
    def _render_loop(self):
        if self.vis is None:
            return
        while not self._should_stop.is_set():
            try:
                self._process_commands(timeout=0.01)
                current_time = time.time()
                should_render = (
                    current_time - self._last_render_time >= self._min_frame_interval and
                    (self._geometry_dirty.is_set() or self._camera_dirty.is_set() or 
                     current_time - self._last_render_time > 0.1)
                )
                if should_render:
                    self._render_frame()
                    self._last_render_time = current_time
                else:
                    time.sleep(0.005)
            except Exception as e:
                logger.error(f"{self.name}: Error in render loop: {e}")
                time.sleep(0.01)
    
    def _process_commands(self, timeout: float = 0.01):
        try:
            while True:
                try:
                    cmd, data = self._cmd_queue.get(timeout=timeout)
                    self._handle_command(cmd, data)
                    timeout = 0
                except queue.Empty:
                    break
        except Exception as e:
            logger.error(f"{self.name}: Error processing commands: {e}")
    
    def _handle_command(self, cmd: str, data):
        try:
            if cmd == 'set_geometries':
                current_time = time.time()
                if current_time - self._last_geometry_update >= self._geometry_update_interval:
                    self._geometries = data
                    self._update_geometries()
                    self._last_geometry_update = current_time
            elif cmd == 'set_camera':
                self._camera_params = data
                self._update_camera()
            elif cmd == 'resize':
                self._handle_resize(data)  #  Pass data (the size_tuple)
        except Exception as e:
            logger.error(f"{self.name}: Error handling command {cmd}: {e}")
    
    def _update_geometries(self):
        if self.vis is None: return
        try:
            self.vis.clear_geometries()
            for geom in self._geometries:
                if geom is not None:
                    self.vis.add_geometry(geom)
            self._geometry_dirty.clear()
        except Exception as e:
            logger.error(f"{self.name}: Geometry update error: {e}")
    
    def _update_camera(self):
        if self.vis is None or self._camera_params is None: return
        try:
            ctr = self.vis.get_view_control()
            params = self._camera_params
            if 'lookat' in params: ctr.set_lookat(params['lookat'])
            if 'front' in params: ctr.set_front(params['front'])
            if 'up' in params: ctr.set_up(params['up'])
            if 'zoom' in params: ctr.set_zoom(float(params['zoom']))
            self._camera_dirty.clear()
        except Exception as e:
            logger.error(f"{self.name}: Camera update error: {e}")
    
    def _handle_resize(self, size_tuple):
        """GLFW-based window resize to avoid visualizer recreation"""
        if self.vis is None or size_tuple is None: return
        try:
            width, height = size_tuple
            if width == self.width and height == self.height: return
            
            # Try direct GLFW resize (avoids class registration)
            if hasattr(self.vis, 'window_id') and self.vis.window_id:
                glfw.glfwSetWindowSize(self.vis.window_id, width, height)
                self.vis.poll_events()
                self.vis.update_renderer()
                self.width, self.height = width, height
                logger.info(f"{self.name}: Resized via glfwSetWindowSize to {width}x{height}")
                return  # Success, no recreate
            
            # Fallback: Original recreate (only if no window_id)
            logger.warning(f"{self.name}: Falling back to recreate")
            old_geoms = self._geometries.copy()
            old_cam = self._camera_params.copy() if self._camera_params else None
            try:
                self.vis.destroy_window()
            except: pass
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name=self.name, width=width, height=height, visible=False)
            if old_geoms: 
                self._geometries = old_geoms
                self._update_geometries()
            if old_cam: 
                self._camera_params = old_cam
                self._update_camera()
            self.width, self.height = width, height
        except Exception as e:
            logger.error(f"{self.name}: Resize error: {e}")
        finally:
            self._resize_dirty.clear()
            self._new_size = None
            self._is_resizing = False
    
    def _render_frame(self):
        """Capture frame and send to callbacks for GUI display"""
        if self.vis is None: return
        if self._is_resizing: return
        try:
            self.vis.poll_events()
            self.vis.update_renderer()
            img = self.vis.capture_screen_float_buffer(do_render=True)
            if img is not None:
                arr = (np.asarray(img) * 255).astype(np.uint8)
                if arr.ndim == 2: arr = np.stack([arr] * 3, axis=-1)
                if arr.shape[0] > 0 and arr.shape[1] > 0:
                    self._send_frame_to_callbacks(arr)
                    self._update_performance_stats()
                else:
                    logger.warning(f"{self.name}: Invalid frame: {arr.shape}")
        except Exception as e:
            logger.error(f"{self.name}: Render error: {e}")
    
    def _send_frame_to_callbacks(self, frame: np.ndarray):
        with self._callback_lock:
            callbacks = list(self._frame_callbacks)
        if not callbacks:
            return
        for cb in callbacks:
            try:
                self._callback_executor.submit(cb, frame)  # reuse worker threads
            except Exception as e:
                logger.error(f"{self.name}: Error submitting frame callback: {e}")

    
    def _update_performance_stats(self):
        self._frame_count += 1
        current_time = time.time()
        if current_time - self._last_fps_time >= 5.0:
            fps = self._frame_count / (current_time - self._last_fps_time)
            logger.debug(f"{self.name}: Rendering at {fps:.1f} FPS")
            self._frame_count = 0
            self._last_fps_time = current_time

# -------------------------------
# Render Manager
# -------------------------------
class RenderManager:
    """Manages multiple renderers and frame distribution"""
    def __init__(self):
        self.renderers = {}
        self._frame_queue = queue.Queue(maxsize=10)
        self._processing_frames = False
    
    def create_renderer(self, name: str, width: int, height: int) -> Open3DRenderThread:
        if name in self.renderers:
            self.stop_renderer(name)
        renderer = Open3DRenderThread(width, height, name)
        renderer.add_frame_callback(lambda frame, n=name: self._queue_frame(n, frame))
        self.renderers[name] = renderer
        renderer.start()
        return renderer
    
    def stop_renderer(self, name: str):
        if name in self.renderers:
            self.renderers[name].stop()
            self.renderers[name].join(timeout=1.0)
            del self.renderers[name]
    
    def stop_all(self):
        for name in list(self.renderers.keys()):
            self.stop_renderer(name)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_all()
    
    def _queue_frame(self, renderer_name: str, frame: np.ndarray):
        try:
            self._frame_queue.put((renderer_name, frame), block=False)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put((renderer_name, frame), block=False)
            except queue.Empty:
                pass
    
    def get_frame(self, timeout: float = 0.01) -> Optional[tuple]:
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

# -------------------------------
# Improved Update Loop
# -------------------------------
class ImprovedUpdateLoop:
    def __init__(self, app_instance):
        self.app = app_instance
        self.render_manager = RenderManager()
        self._update_running = False
        self._main_frame_queue = queue.Queue(maxsize=2)
        self._fpv_frame_queue = queue.Queue(maxsize=2)
    
    def start(self):
        main_width = self.app.main_canvas.winfo_width() or 960
        main_height = self.app.main_canvas.winfo_height() or 720
        fpv_width = self.app.fpv_label.winfo_width() or 400
        fpv_height = self.app.fpv_label.winfo_height() or 300
        
        self.main_renderer = self.render_manager.create_renderer("MainOpen3D", main_width, main_height)
        self.fpv_renderer = self.render_manager.create_renderer("FPVOpen3D", fpv_width, fpv_height)
        
        self.main_renderer.add_frame_callback(self._queue_main_frame)
        self.fpv_renderer.add_frame_callback(self._queue_fpv_frame)
        
        self._update_running = True
        self._schedule_update()
    
    def stop(self):
        self._update_running = False
        self.render_manager.stop_all()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def _queue_main_frame(self, frame: np.ndarray):
        try:
            self._main_frame_queue.put(frame, block=False)
        except queue.Full:
            try:
                self._main_frame_queue.get_nowait()
                self._main_frame_queue.put(frame, block=False)
            except queue.Empty:
                pass
    
    def _queue_fpv_frame(self, frame: np.ndarray):
        try:
            self._fpv_frame_queue.put(frame, block=False)
        except queue.Full:
            try:
                self._fpv_frame_queue.get_nowait()
                self._fpv_frame_queue.put(frame, block=False)
            except queue.Empty:
                pass
    
    def _schedule_update(self):
        if self._update_running:
            self.app.root.after(16, self._update)
    
    def _update(self):
        try:
            # Handle main canvas updates
            try:
                frame = self._main_frame_queue.get_nowait()
                self.app._display_main_frame(frame)
            except queue.Empty:
                pass
                
            # Handle FPV updates and recording
            try:
                frame = self._fpv_frame_queue.get_nowait()
                self.app._display_fpv_frame(frame)
                
                # Record FPV frames if recording is active (regardless of simulation state)
                if (self.app.video_recorder and 
                    self.app.video_recorder.is_recording):
                    self.app.video_recorder.add_frame(frame)
                    
            except queue.Empty:
                pass
                
        except Exception as e:
            logger.error(f"Error in update loop: {e}")
        self._schedule_update()
    
    def update_main_geometries(self, geometries):
        if hasattr(self, 'main_renderer'):
            self.main_renderer.set_geometries(geometries)
    
    def update_fpv_geometries(self, geometries):
        if hasattr(self, 'fpv_renderer'):
            self.fpv_renderer.set_geometries(geometries)
    
    def update_main_camera(self, params):
        if hasattr(self, 'main_renderer'):
            self.main_renderer.set_camera(params)
    
    def update_fpv_camera(self, params):
        if hasattr(self, 'fpv_renderer'):
            self.fpv_renderer.set_camera(params)
    
    def resize_main(self, width, height):
        if hasattr(self, 'main_renderer'):
            self.main_renderer.resize(width, height)
    
    def resize_fpv(self, width, height):
        if hasattr(self, 'fpv_renderer'):
            self.fpv_renderer.resize(width, height)

# -------------------------------
# Scene3D
# -------------------------------
class Scene3D:
    """3D model loader and scene manager"""
    def __init__(self, app=None):
        self.app = app  # Reference to main application
        self.model_trimesh: Optional[trimesh.Trimesh] = None
        self.bounds: Optional[np.ndarray] = None
        self.o3d_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.o3d_geometries: List[o3d.geometry.Geometry] = []
        self.ground_plane: Optional[o3d.geometry.TriangleMesh] = None
        self.show_ground_plane = False  # Changed to False - no ground plane by default

    def load_model(self, filepath: str) -> bool:
        try:
            try:
                loaded = trimesh.load(filepath)
                if isinstance(loaded, trimesh.Scene):
                    loaded = loaded.to_geometry()
                self.model_trimesh = loaded
                self.bounds = self.model_trimesh.bounds
                logger.info(f"Trimesh model loaded successfully: {filepath}")
            except Exception as e:
                logger.warning(f"Trimesh failed: {e}. Continuing with Open3D only.")

            try:
                mesh_o3d = o3d.io.read_triangle_mesh(filepath, enable_post_processing=True)
                if mesh_o3d is None or len(mesh_o3d.vertices) == 0:
                    raise RuntimeError("Open3D returned empty mesh")
                if not mesh_o3d.has_vertex_normals():
                    mesh_o3d.compute_vertex_normals()
                self.o3d_mesh = mesh_o3d
                
                logger.info("Open3D mesh loaded and normals computed.")
            except Exception as e:
                logger.error(f"Open3D failed to load mesh: {e}")
                self.o3d_mesh = None
                return False

            if self.bounds is not None:
                bmin, bmax = self.bounds
                center = (bmin + bmax) / 2.0
                size = float(np.max(bmax - bmin) * 2.0 + 1.0)
            else:
                verts = np.asarray(self.o3d_mesh.vertices)
                center = verts.mean(axis=0) if len(verts) else np.array([0.0, 0.0, 0.0])
                
                size = float(np.ptp(verts, axis=0).max() * 2.0 + 1.0) if len(verts) else 10.0

            #  Auto-center camera on model
            app_center = center + np.array([0.0, 0.0, np.linalg.norm(bmax - bmin) * 1.5])  # Eye above center
            self.app.camera.position = app_center
            self.app._update_main_camera()  # Or queue via update_loop_manager

            
            self.ground_plane = o3d.geometry.TriangleMesh.create_box(width=size, height=0.01, depth=size)
            self.ground_plane.translate([center[0] - size/2.0, center[1] - 0.005, center[2] - size/2.0])
            self.ground_plane.compute_vertex_normals()
            self.ground_plane.paint_uniform_color([0.5, 0.5, 0.5])

            self.o3d_geometries = [self.o3d_mesh]
            # Only add ground plane if enabled (now False by default)
            if self.show_ground_plane:
                self.o3d_geometries.append(self.ground_plane)
                
            logger.info("Scene3D configured (Open3D geometries ready).")
            return True
        except Exception as e:
            logger.error(f"Scene3D.load_model failed: {e}")
            return False

    def get_trimesh_components(self) -> List[trimesh.Trimesh]:
        comps = []
        if self.model_trimesh is not None:
            comps.append(self.model_trimesh)
        return comps

# -------------------------------
# Enhanced FlightManager with UAV Physics
# -------------------------------
class FlightManager:
    """Handles flight simulation with realistic UAV physics and smooth transitions"""
    
    def __init__(self, app_instance):
        self.app = app_instance
        self.flight_simulation_active = False
        self.flight_animation_id = None
        self.current_flight_position = 0.0
        self.interpolated_path = None
        self.interpolated_yaw = None
        self.interpolated_pitch = None
        self.flight_speed_factor = tk.DoubleVar(value=1.0)
        self.flight_paused = False
        self.selected_waypoint = None
        
        # UAV Physics parameters
        self.max_speed = 15.0  # m/s
        self.max_acceleration = 2.0  # m/s²
        self.max_angular_velocity = 30.0  # degrees/s
        self.hover_time = 2.0  # seconds
        
        # ADD MISSING ATTRIBUTES FROM ORIGINAL CODE
        self.cruising_speed = tk.DoubleVar(value=10.0)
        self.hover_time_var = tk.DoubleVar(value=2.5)  # Renamed to avoid conflict
        
        # Current state
        self.current_velocity = 0.0
        self.current_angular_velocity_yaw = 0.0
        self.current_angular_velocity_pitch = 0.0
        
        # Performance optimization
        self._last_update_time = time.time()
        self._cached_cumdist = None
        self._cached_cumdist_traj = None

    def start_flight_simulation(self):
        if not self.app.selected_trajectory or len(self.app.selected_trajectory.waypoints) < 2:
            messagebox.showwarning("Warning", "Need a trajectory with at least 2 waypoints")
            return
        
        self.flight_simulation_active = True
        self.flight_paused = False
        self.current_flight_position = 0.0
        self.current_velocity = 0.0
        self.current_angular_velocity_yaw = 0.0
        self.current_angular_velocity_pitch = 0.0
        self._last_update_time = time.time()
        
        # Generate smooth interpolated path
        self._generate_smooth_path()
        
        # Setup progress tracking
        if hasattr(self.app, 'flight_progress'):
            self.app.flight_progress['maximum'] = len(self.interpolated_path) - 1
        
        # Update status
        if hasattr(self.app, 'flight_status_label'):
            self.app.flight_status_label.config(
                text=f"Starting flight simulation ({len(self.app.selected_trajectory.waypoints)} waypoints)"
            )
        
        self.animate_flight()

    def _generate_smooth_path(self):
        """Generate smooth path that follows sequential waypoint order"""
        traj = self.app.selected_trajectory
        waypoints = traj.waypoints
        
        if len(waypoints) < 2:
            return
        
        # Ensure waypoints are in sequential order
        waypoints = sorted(waypoints, key=lambda wp: wp.index)
        
        # Calculate total distance for proper resolution
        positions = traj.get_positions()
        segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_distance = np.sum(segment_lengths)
        
        # Use higher resolution for smoother flight (1 point per 0.05m)
        num_points = max(200, int(total_distance * 20))
        
        # Generate path based on interpolation method
        if traj.interpolation_method == InterpolationMethod.SPLINE:
            path = TrajectoryInterpolator.interpolate_spline(waypoints, num_points=num_points)
        else:
            path = TrajectoryInterpolator.interpolate_linear(waypoints, num_points=num_points)
        
        # Generate sequential orientation interpolation
        self.interpolated_path = path
        self.interpolated_yaw = np.zeros(num_points)
        self.interpolated_pitch = np.zeros(num_points)
        
        # Calculate which waypoint segment each path point belongs to
        self.path_segment_indices = np.zeros(num_points, dtype=int)
        
        # Calculate cumulative distance along the interpolated path
        path_distances = np.zeros(num_points)
        for i in range(1, num_points):
            path_distances[i] = path_distances[i-1] + np.linalg.norm(path[i] - path[i-1])
        
        # Calculate cumulative distance for original waypoints
        waypoint_distances = np.zeros(len(waypoints))
        for i in range(1, len(waypoints)):
            waypoint_distances[i] = waypoint_distances[i-1] + np.linalg.norm(waypoints[i].position - waypoints[i-1].position)
        
        # Assign each path point to its corresponding waypoint segment
        for i in range(num_points):
            current_dist = path_distances[i]
            
            # Find which segment this point belongs to
            segment_idx = 0
            for j in range(len(waypoint_distances)-1):
                if waypoint_distances[j] <= current_dist < waypoint_distances[j+1]:
                    segment_idx = j
                    break
                elif current_dist >= waypoint_distances[-1]:
                    segment_idx = len(waypoint_distances) - 2  # Last segment
            
            self.path_segment_indices[i] = segment_idx
            
            # Interpolate orientation within the segment
            if segment_idx < len(waypoints) - 1:
                segment_start_dist = waypoint_distances[segment_idx]
                segment_end_dist = waypoint_distances[segment_idx + 1]
                segment_length = segment_end_dist - segment_start_dist
                
                if segment_length > 1e-6:
                    alpha = (current_dist - segment_start_dist) / segment_length
                    alpha = np.clip(alpha, 0, 1)
                    
                    wp1 = waypoints[segment_idx]
                    wp2 = waypoints[segment_idx + 1]
                    
                    # Smooth yaw interpolation with shortest path
                    yaw_diff = ((wp2.yaw - wp1.yaw + 180) % 360) - 180
                    self.interpolated_yaw[i] = wp1.yaw + yaw_diff * alpha
                    
                    # Pitch interpolation
                    self.interpolated_pitch[i] = wp1.pitch + (wp2.pitch - wp1.pitch) * alpha
                else:
                    # Zero-length segment, use the waypoint's orientation
                    self.interpolated_yaw[i] = waypoints[segment_idx].yaw
                    self.interpolated_pitch[i] = waypoints[segment_idx].pitch
            else:
                # Beyond last waypoint, use last waypoint's orientation
                self.interpolated_yaw[i] = waypoints[-1].yaw
                self.interpolated_pitch[i] = waypoints[-1].pitch
        
        # Store waypoint positions for visualization
        self.waypoint_positions = positions
        self.total_path_distance = path_distances[-1]

    def _get_waypoint_at_fraction(self, trajectory, t_param: float):
        """Get interpolated waypoint at fraction along trajectory with smooth orientation"""
        try:
            if not trajectory or not trajectory.waypoints:
                return None

            positions = trajectory.get_positions()
            num_points = len(positions)
            if num_points < 2:
                return trajectory.waypoints[0]

            t_param = max(0.0, min(1.0, t_param))

            # Use cached cumulative distances
            if not hasattr(self, '_cached_cumdist') or self._cached_cumdist_traj != trajectory:
                distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                self._cached_cumdist = np.concatenate(([0.0], np.cumsum(distances)))
                self._cached_cumdist_traj = trajectory

            cumdist = self._cached_cumdist
            total_len = cumdist[-1]
            
            if total_len == 0:
                return trajectory.waypoints[0]

            # Find segment
            target_dist = t_param * total_len
            seg_idx = np.searchsorted(cumdist, target_dist) - 1
            seg_idx = max(0, min(seg_idx, len(trajectory.waypoints) - 2))

            # Smooth interpolation
            seg_len = cumdist[seg_idx + 1] - cumdist[seg_idx]
            if seg_len < 1e-6:
                return trajectory.waypoints[seg_idx]
                
            alpha = (target_dist - cumdist[seg_idx]) / seg_len
            
            # Apply easing function for smoother transitions
            alpha_eased = self._ease_in_out_cubic(alpha)
            
            # Position interpolation
            pos = (1 - alpha_eased) * positions[seg_idx] + alpha_eased * positions[seg_idx + 1]

            # Smooth orientation interpolation with proper wrap-around
            wp1 = trajectory.waypoints[seg_idx]
            wp2 = trajectory.waypoints[seg_idx + 1]
            
            # Yaw interpolation with shortest path
            yaw_diff = ((wp2.yaw - wp1.yaw + 180) % 360) - 180
            yaw = wp1.yaw + yaw_diff * alpha_eased
            
            # Pitch interpolation
            pitch = wp1.pitch + (wp2.pitch - wp1.pitch) * alpha_eased

            return Waypoint(position=pos, yaw=yaw, pitch=pitch, index=seg_idx)

        except Exception as e:
            logger.error(f"_get_waypoint_at_fraction error: {e}")
            return None

    def _ease_in_out_cubic(self, x: float) -> float:
        """Easing function for smooth acceleration and deceleration"""
        if x < 0.5:
            return 4 * x * x * x
        else:
            return 1 - pow(-2 * x + 2, 3) / 2

    def animate_flight(self):
        if not self.flight_simulation_active or self.interpolated_path is None:
            return
        
        if self.flight_paused:
            self.flight_animation_id = self.app.root.after(16, self.animate_flight)
            return
        
        current_time = time.time()
        delta_time = current_time - self._last_update_time
        self._last_update_time = current_time
        
        # Apply realistic physics
        self._update_physics(delta_time)
        
        if self.current_flight_position >= len(self.interpolated_path) - 1:
            self.stop_flight_simulation()
            return
        
        # Get current position with smooth interpolation
        pos_idx = int(self.current_flight_position)
        next_idx = min(pos_idx + 1, len(self.interpolated_path) - 1)
        alpha = self.current_flight_position - pos_idx
        
        # Smooth position interpolation
        position = (1 - alpha) * self.interpolated_path[pos_idx] + alpha * self.interpolated_path[next_idx]
        
        # Smooth orientation interpolation
        current_yaw = (1 - alpha) * self.interpolated_yaw[pos_idx] + alpha * self.interpolated_yaw[next_idx]
        current_pitch = (1 - alpha) * self.interpolated_pitch[pos_idx] + alpha * self.interpolated_pitch[next_idx]
        
        # Get current waypoint segment for visualization
        current_segment = self.path_segment_indices[pos_idx]
        
        # Update visualization to show current waypoint
        self.selected_waypoint = current_segment
        if pos_idx % 3 == 0:  # Update every 3 frames for performance
            self.app.update_trajectory_visualization_lightweight()
        
        # Update FPV with smooth camera movement
        self._update_fpv_smooth(position, current_yaw, current_pitch, delta_time)
        
        # Update progress bar
        if hasattr(self.app, 'flight_progress'):
            self.app.flight_progress['value'] = self.current_flight_position
        
        # Update status with waypoint information
        if hasattr(self.app, 'flight_status_label') and pos_idx % 30 == 0:
            total_waypoints = len(self.app.selected_trajectory.waypoints)
            current_waypoint = min(current_segment + 1, total_waypoints)
            progress_percent = (self.current_flight_position / (len(self.interpolated_path) - 1)) * 100
            speed_kmh = self.current_velocity * 3.6
            
            self.app.flight_status_label.config(
                text=f"Waypoint {current_waypoint}/{total_waypoints} - {progress_percent:.1f}% - {speed_kmh:.1f} km/h"
            )
        
        # Schedule next frame
        self.flight_animation_id = self.app.root.after(16, self.animate_flight)

    def _update_physics(self, delta_time: float):
        """Update UAV physics for sequential waypoint following"""
        if self.interpolated_path is None or len(self.interpolated_path) < 2:
            return
        
        pos_idx = int(self.current_flight_position)
        
        # Use the cruising speed from the GUI control
        base_speed = self.cruising_speed.get()
        
        # Adaptive speed control based on path characteristics
        if pos_idx < len(self.interpolated_path) - 10:
            # Look ahead to detect turns and adjust speed
            look_ahead_dist = min(5.0, self.total_path_distance * 0.1)  # Look 5m ahead or 10% of path
            
            # Find look-ahead point
            current_dist = pos_idx / len(self.interpolated_path) * self.total_path_distance
            target_dist = current_dist + look_ahead_dist
            
            if target_dist < self.total_path_distance:
                # Find corresponding path index for look-ahead point
                look_ahead_idx = min(int(target_dist / self.total_path_distance * len(self.interpolated_path)), 
                                len(self.interpolated_path) - 1)
                
                # Calculate curvature between current and look-ahead point
                current_dir = self.interpolated_path[min(pos_idx + 1, len(self.interpolated_path)-1)] - self.interpolated_path[pos_idx]
                future_dir = self.interpolated_path[look_ahead_idx] - self.interpolated_path[pos_idx]
                
                if np.linalg.norm(current_dir) > 0 and np.linalg.norm(future_dir) > 0:
                    current_dir = current_dir / np.linalg.norm(current_dir)
                    future_dir = future_dir / np.linalg.norm(future_dir)
                    turn_angle = np.arccos(np.clip(np.dot(current_dir, future_dir), -1, 1))
                    
                    # Reduce speed for sharp turns
                    turn_factor = 1.0 - min(turn_angle / (np.pi * 0.5), 0.7)  # Reduce up to 70% for 90° turns
                    target_speed = base_speed * turn_factor
                else:
                    target_speed = base_speed
            else:
                # Approaching end of path, slow down
                remaining_dist = self.total_path_distance - current_dist
                target_speed = base_speed * min(1.0, remaining_dist / 10.0)  # Slow down in last 10m
        else:
            # Final approach, slow down gradually
            remaining_points = len(self.interpolated_path) - 1 - self.current_flight_position
            target_speed = base_speed * min(1.0, remaining_points / 50.0)
        
        # Smooth acceleration/deceleration
        speed_diff = target_speed - self.current_velocity
        max_accel = self.max_acceleration * delta_time
        acceleration = np.clip(speed_diff, -max_accel, max_accel)
        self.current_velocity += acceleration
        self.current_velocity = max(0.1, min(base_speed, self.current_velocity))  # Minimum speed of 0.1 m/s
        
        # Advance position based on velocity and path geometry
        if len(self.interpolated_path) > 1 and pos_idx < len(self.interpolated_path) - 1:
            # Calculate distance to next point
            segment_vector = self.interpolated_path[pos_idx + 1] - self.interpolated_path[pos_idx]
            segment_length = np.linalg.norm(segment_vector)
            
            if segment_length > 0:
                # Calculate how many points to advance based on speed
                points_per_second = len(self.interpolated_path) / (self.total_path_distance / self.current_velocity)
                advance_amount = points_per_second * delta_time * self.flight_speed_factor.get()
                
                self.current_flight_position += advance_amount
                
                # Ensure we don't overshoot the path
                self.current_flight_position = min(self.current_flight_position, len(self.interpolated_path) - 1)

    def _update_fpv_smooth(self, position: np.ndarray, target_yaw: float, target_pitch: float, delta_time: float):
        """Update FPV camera with smooth, realistic camera movement"""
        # Smooth camera rotation using angular velocity
        if hasattr(self, '_current_camera_yaw'):
            # Smooth yaw transition
            yaw_diff = ((target_yaw - self._current_camera_yaw + 180) % 360) - 180
            yaw_velocity = np.clip(yaw_diff / 0.5, -self.max_angular_velocity, self.max_angular_velocity)
            self._current_camera_yaw += yaw_velocity * delta_time
            
            # Smooth pitch transition
            pitch_diff = target_pitch - self._current_camera_pitch
            pitch_velocity = np.clip(pitch_diff / 0.5, -self.max_angular_velocity, self.max_angular_velocity)
            self._current_camera_pitch += pitch_velocity * delta_time
        else:
            self._current_camera_yaw = target_yaw
            self._current_camera_pitch = target_pitch
        
        # Calculate forward direction from smoothed orientation
        yaw_rad = np.radians(self._current_camera_yaw)
        pitch_rad = np.radians(self._current_camera_pitch)
        
        forward = np.array([
            np.cos(pitch_rad) * np.sin(yaw_rad),
            np.sin(pitch_rad),
            np.cos(pitch_rad) * np.cos(yaw_rad)
        ])
        
        if np.linalg.norm(forward) == 0:
            forward = np.array([0.0, 0.0, -1.0])
        forward /= np.linalg.norm(forward)
        
        # Get FPV settings
        look_distance = self.app.fpv_distance.get()
        zoom = self.app.fpv_zoom.get()
        
        # Calculate look-at point
        camera_pos = position
        lookat = camera_pos + forward * look_distance
        
        # Camera parameters with smooth transitions
        params = {
            'lookat': lookat.tolist(),
            'front': forward.tolist(),
            'up': [0.0, 1.0, 0.0],
            'zoom': zoom
        }
        
        self.app.update_loop_manager.update_fpv_camera(params)

    def pause_flight_simulation(self):
        self.flight_paused = True

    def resume_flight_simulation(self):
        self.flight_paused = False
        self._last_update_time = time.time()
        if self.flight_simulation_active:
            self.animate_flight()

    def stop_flight_simulation(self):
        self.flight_simulation_active = False
        self.flight_paused = False
        if self.flight_animation_id:
            self.app.root.after_cancel(self.flight_animation_id)
            self.flight_animation_id = None
        
        if hasattr(self.app, 'flight_progress'):
            self.app.flight_progress['value'] = 0
        
        if hasattr(self.app, 'flight_status_label'):
            self.app.flight_status_label.config(text="Flight simulation complete")

# -------------------------------
# Video Recorder 
# -------------------------------
class VideoRecorder:
    """Records FPV view to video file"""
    def __init__(self, output_path: str, fps: int = 30, resolution: Tuple[int, int] = (1920, 1080)):
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.writer = None
        self.is_recording = False
        self.frame_source = "fpv"

    def start_recording(self):
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.resolution)
            self.is_recording = True
            logger.info(f"VideoRecorder: started {self.output_path}")
        except Exception as e:
            logger.error(f"VideoRecorder failed to start: {e}")
            self.is_recording = False

    def add_frame(self, frame: np.ndarray):
        if not self.is_recording or self.writer is None:
            return
        h, w = frame.shape[:2]
        if (w, h) != self.resolution:
            frame = cv2.resize(frame, self.resolution)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame_bgr)

    def stop_recording(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.is_recording = False
        logger.info(f"VideoRecorder: saved {self.output_path}")

# -------------------------------
# Main Application
# -------------------------------
class UAVVisualizationApp:
    """Main application coordinating all components"""
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("UAV Flight Trajectory Visualization (Open3D Embedded)")
        self.root.geometry("1400x900")
        self.fpv_manual_override = {'zoom': False, 'distance': False}  # Track manual adjustments
        self.scene = Scene3D(self)  
        self.camera = Camera3D()
        self.trajectories: List[Trajectory] = []
        self.selected_trajectory: Optional[Trajectory] = None
        self.selected_waypoint: Optional[int] = None
        self.fractional_waypoint = 0.0
        self.video_recorder: Optional[VideoRecorder] = None

        self.camera_x = tk.DoubleVar(value=float(self.camera.position[0]))
        self.camera_y = tk.DoubleVar(value=float(self.camera.position[1]))
        self.camera_z = tk.DoubleVar(value=float(self.camera.position[2]))
        self.camera_yaw = tk.DoubleVar(value=float(self.camera.yaw))
        self.camera_pitch = tk.DoubleVar(value=float(self.camera.pitch))
        
        # FPV camera controls (NEW)
        self.fpv_distance = tk.DoubleVar(value=5.0)
        self.fpv_zoom = tk.DoubleVar(value=0.08)
        
        # Show all waypoints toggle
        self.show_all_waypoints = tk.BooleanVar(value=False)
        
        # Animation/simulation controls
        self.is_simulating = False
        self.simulation_thread = None
        self.current_simulation_waypoint = 0
        self.simulation_progress = 0.0
        self.simulation_speed = tk.DoubleVar(value=1.0)  # Seconds between waypoints
        self.simulation_paused = False

        self.flight_manager = FlightManager(self)

        self._mouse_last = None
        self._is_dragging = False

        self.setup_gui()
        self.update_loop_manager = ImprovedUpdateLoop(self)
        self._main_photo = None
        self._fpv_photo = None
        self.root.after(100, self.update_loop_manager.start)

    def setup_gui(self):
        """Create main UI layout with control panels and 3D views"""
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_paned, width=300)
        main_paned.add(left_frame, weight=0)

        middle_frame = ttk.Frame(main_paned)
        main_paned.add(middle_frame, weight=3)

        right_frame = ttk.Frame(main_paned, width=300)
        main_paned.add(right_frame, weight=0)

        self.setup_control_panel(left_frame)
        self.setup_3d_view(middle_frame)
        self.setup_info_panel(right_frame)

    def setup_control_panel(self, parent):
        file_frame = CollapsibleFrame(parent, text="File Operations")
        file_frame.pack(fill=tk.X, padx=5, pady=2)
        file_content = file_frame.get_content_frame()
        
        ttk.Button(file_content, text="Load 3D Model", command=self.load_model).pack(fill=tk.X, pady=2)
        ttk.Button(file_content, text="Load Trajectory", command=self.load_trajectory).pack(fill=tk.X, pady=2)
        ttk.Button(file_content, text="Save Trajectory", command=self.save_trajectory).pack(fill=tk.X, pady=2)
        ttk.Button(file_content, text="Clear All Trajectories", command=self.clear_all_trajectories).pack(fill=tk.X, pady=2)  # NEW

        camera_frame = CollapsibleFrame(parent, text="Camera Controls")
        camera_frame.pack(fill=tk.X, padx=5, pady=2)
        camera_content = camera_frame.get_content_frame()
        
        self.camera_button = ttk.Button(camera_content, text="Open Camera Controls", command=self.open_camera_window)
        self.camera_button.pack(fill=tk.X, pady=2)
        
        # FPV camera distance control (NEW)
        ttk.Label(camera_content, text="FPV Camera Distance:").pack(pady=(5, 0))
        self.fpv_distance_scale = ttk.Scale(
            camera_content, 
            from_=0.5, 
            to=50.0, 
            variable=self.fpv_distance,
            orient=tk.HORIZONTAL,
            command=self.on_fpv_distance_change
        )
        self.fpv_distance_scale.pack(fill=tk.X, padx=10, pady=2)
        self.fpv_distance_label = ttk.Label(camera_content, text="5.0 m")
        self.fpv_distance_label.pack()
        
        # FPV zoom control (NEW)
        ttk.Label(camera_content, text="FPV Zoom:").pack(pady=(5, 0))
        self.fpv_zoom_scale = ttk.Scale(
            camera_content,
            from_=0.01,
            to=1.0,
            variable=self.fpv_zoom,
            orient=tk.HORIZONTAL,
            command=self.on_fpv_zoom_change
        )
        self.fpv_zoom_scale.pack(fill=tk.X, padx=10, pady=2)
        self.fpv_zoom_label = ttk.Label(camera_content, text="0.08")
        self.fpv_zoom_label.pack()

        traj_frame = CollapsibleFrame(parent, text="Trajectory Controls")
        traj_frame.pack(fill=tk.X, padx=5, pady=2)
        traj_content = traj_frame.get_content_frame()
        
        ttk.Button(traj_content, text="Compare Trajectories", command=self.compare_trajectories).pack(fill=tk.X, pady=2)
        ttk.Button(traj_content, text="Compare Interpolations", command=self.compare_interpolations).pack(fill=tk.X, pady=2)
        ttk.Label(traj_content, text="Interpolation Method:").pack()
        self.interp_method = tk.StringVar(value="spline")
        ttk.Radiobutton(traj_content, text="Linear", variable=self.interp_method, value="linear", command=self.update_interpolation).pack()
        ttk.Radiobutton(traj_content, text="Spline", variable=self.interp_method, value="spline", command=self.update_interpolation).pack()

        # Removed Ground Plane section

        record_frame = CollapsibleFrame(parent, text="Flight Simulation & Recording")
        record_frame.pack(fill=tk.X, padx=5, pady=2)
        record_content = record_frame.get_content_frame()
        
        # Flight simulation controls (separated from recording)
        sim_section = ttk.LabelFrame(record_content, text="Flight Simulation", padding=5)
        sim_section.pack(fill=tk.X, pady=(0, 10))
        
        # Animation speed multiplier
        ttk.Label(sim_section, text="Animation Speed Multiplier:").pack(pady=(0, 0))
        self.animation_speed_scale = ttk.Scale(
            sim_section,
            from_=0.1,
            to=5.0,
            variable=self.flight_manager.flight_speed_factor,
            orient=tk.HORIZONTAL
        )
        self.animation_speed_scale.pack(fill=tk.X, padx=10, pady=2)
        
        # Hover time
        ttk.Label(sim_section, text="Hover Time (s):").pack(pady=(5, 0))
        self.hover_time_scale = ttk.Scale(
            sim_section,
            from_=0.0,
            to=10.0,
            variable=self.flight_manager.hover_time,
            orient=tk.HORIZONTAL
        )
        self.hover_time_scale.pack(fill=tk.X, padx=10, pady=2)
        
        # Cruising speed
        ttk.Label(sim_section, text="Cruising Speed (m/s):").pack(pady=(5, 0))
        self.cruising_speed_scale = ttk.Scale(
            sim_section,
            from_=1.0,
            to=50.0,
            variable=self.flight_manager.cruising_speed,
            orient=tk.HORIZONTAL
        )
        self.cruising_speed_scale.pack(fill=tk.X, padx=10, pady=2)
        
        # Simulation control buttons
        sim_buttons_frame = ttk.Frame(sim_section)
        sim_buttons_frame.pack(fill=tk.X, pady=5)
        
        self.simulate_button = ttk.Button(
            sim_buttons_frame, 
            text="Start Flight Simulation", 
            command=self.toggle_simulation
        )
        self.simulate_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        self.sim_pause_button = ttk.Button(
            sim_buttons_frame, 
            text="Pause", 
            command=self.toggle_simulation_pause, 
            state='disabled'
        )
        self.sim_pause_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        # Flight progress bar
        self.flight_progress = ttk.Progressbar(sim_section, mode='determinate')
        self.flight_progress.pack(fill=tk.X, pady=2)
        
        self.flight_status_label = ttk.Label(sim_section, text="Ready for flight simulation")
        self.flight_status_label.pack(fill=tk.X, pady=2)
        
        # Recording controls (separate section)
        record_section = ttk.LabelFrame(record_content, text="Video Recording", padding=5)
        record_section.pack(fill=tk.X, pady=(0, 5))
        
        # Recording settings
        settings_frame = ttk.Frame(record_section)
        settings_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(settings_frame, text="FPS:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.record_fps = tk.IntVar(value=30)
        fps_scale = ttk.Scale(settings_frame, from_=10, to=60, variable=self.record_fps, 
                            orient=tk.HORIZONTAL, length=100)
        fps_scale.grid(row=0, column=1, sticky=tk.EW, padx=(0, 10))
        self.fps_label = ttk.Label(settings_frame, text="30")
        self.fps_label.grid(row=0, column=2, sticky=tk.W)
        
        ttk.Label(settings_frame, text="Resolution:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.record_resolution = tk.StringVar(value="1280x720")
        res_combo = ttk.Combobox(settings_frame, textvariable=self.record_resolution,
                            values=["640x480", "800x600", "1280x720", "1920x1080"],
                            width=12, state="readonly")
        res_combo.grid(row=1, column=1, columnspan=2, sticky=tk.EW)
        
        settings_frame.columnconfigure(1, weight=1)
        
        self.record_button = ttk.Button(record_content, text="Start Recording & Simulation", command=self.toggle_recording)
        self.record_button.pack(fill=tk.X, pady=2)
        
        # Pause button (disabled initially)
        self.pause_button = ttk.Button(record_content, text="Pause", command=self.toggle_pause, state='disabled')
        self.pause_button.pack(fill=tk.X, pady=2)
        
        # Progress label
        self.progress_label = ttk.Label(record_content, text="Ready to simulate")
        self.progress_label.pack(pady=5)

        # Flight progress bar (from first)
        self.flight_progress = ttk.Progressbar(record_content, mode='determinate')
        self.flight_progress.pack(fill=tk.X, pady=2)

        self.flight_status_label = ttk.Label(record_content, text="Ready for flight simulation")
        self.flight_status_label.pack(fill=tk.X, pady=2)

        edit_frame = CollapsibleFrame(parent, text="Waypoint Editing")
        edit_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        edit_content = edit_frame.get_content_frame()
        
        ttk.Button(edit_content, text="Add Waypoint", command=self.add_waypoint).pack(fill=tk.X, pady=2)
        ttk.Button(edit_content, text="Edit Waypoint", command=self.edit_waypoint).pack(fill=tk.X, pady=2)
        ttk.Button(edit_content, text="Delete Waypoint", command=self.delete_waypoint).pack(fill=tk.X, pady=2)
        
        # Checkbox to toggle showing all waypoints
        self.show_all_checkbox = ttk.Checkbutton(
            edit_content, 
            text="Show All Waypoint Cameras",
            variable=self.show_all_waypoints,
            command=self.on_show_all_toggle
        )
        self.show_all_checkbox.pack(fill=tk.X, pady=5)
        
        ttk.Label(edit_content, text="Click waypoint to isolate its camera:").pack(pady=(5, 2))
        
        self.waypoint_listbox = tk.Listbox(edit_content, height=10)
        self.waypoint_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.waypoint_listbox.bind('<<ListboxSelect>>', self.on_waypoint_select)

    def setup_3d_view(self, parent):
        self.canvas_frame = ttk.Frame(parent)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.main_canvas = tk.Canvas(self.canvas_frame, bg='black')
        self.main_canvas.pack(fill=tk.BOTH, expand=True)
        self.main_canvas.bind("<Configure>", self.on_canvas_resize)
        self.main_canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.main_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.main_canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.main_canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        
        # Keyboard bindings for camera controls
        self.main_canvas.bind("<KeyPress>", self.on_key_press)
        self.main_canvas.bind("<KeyRelease>", self.on_key_release)
        
        # Set focus to canvas to receive keyboard events
        self.main_canvas.focus_set()
        
        # Click to focus canvas for keyboard input
        self.main_canvas.bind("<Button-1>", lambda e: self.main_canvas.focus_set())

    def setup_info_panel(self, parent):
        info_frame = CollapsibleFrame(parent, text="Trajectory Information")
        info_frame.pack(fill=tk.X, padx=5, pady=2)
        info_content = info_frame.get_content_frame()
        
        self.info_text = tk.Text(info_content, height=10, width=30)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        fpv_frame = CollapsibleFrame(parent, text="First Person View")
        fpv_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        fpv_content = fpv_frame.get_content_frame()
        
        self.fpv_label = ttk.Label(fpv_content)
        self.fpv_label.pack(fill=tk.BOTH, expand=True)
        self.fpv_label.bind("<Configure>", self.on_fpv_resize)

        comp_frame = CollapsibleFrame(parent, text="Comparison Results")
        comp_frame.pack(fill=tk.X, padx=5, pady=2)
        comp_content = comp_frame.get_content_frame()
        
        self.comp_text = tk.Text(comp_content, height=8, width=30)
        self.comp_text.pack(fill=tk.BOTH, expand=True)
        comp_frame.toggle()

    # Clear trajectories methods
    def clear_trajectories(self):
        """Clear all trajectories from the scene"""
        self.trajectories = []
        self.selected_trajectory = None
        self.selected_waypoint = None
        self.update_waypoint_list()
        self.update_info()
        self.update_trajectory_visualization()
    
    def clear_all_trajectories(self):
        """Clear all trajectories with user confirmation"""
        if not self.trajectories:
            messagebox.showinfo("No Trajectories", "No trajectories to clear.")
            return
            
        if messagebox.askyesno("Clear All Trajectories", 
                               f"Are you sure you want to clear all {len(self.trajectories)} trajectory(ies)?"):
            self.clear_trajectories()
            messagebox.showinfo("Cleared", "All trajectories have been cleared.")

    def load_model(self):
        # Added prompt to clear trajectories
        filepath = filedialog.askopenfilename(title="Select 3D Model",
                                              filetypes=[("3D Models", "*.obj *.ply *.stl"), ("All Files", "*.*")])
        if not filepath:
            return
        
        # Ask user if they want to clear existing trajectories
        if self.trajectories:
            result = messagebox.askyesnocancel(
                "Clear Trajectories?",
                "Do you want to clear existing trajectories?\n\n"
                "Yes - Clear trajectories and load new model\n"
                "No - Keep trajectories and load new model\n"
                "Cancel - Don't load the new model"
            )
            if result is None:  # Cancel
                return
            elif result:  # Yes - clear trajectories
                self.clear_trajectories()
                
        ok = self.scene.load_model(filepath)
        if not ok:
            messagebox.showerror("Error", f"Failed to load model: Could not process {filepath}")
            return
        self.update_trajectory_visualization()
        messagebox.showinfo("Success", "Model loaded successfully.")

    def load_trajectory(self):
        filepath = filedialog.askopenfilename(title="Select Trajectory File",
                                            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if not filepath:
            return
        waypoints = TrajectoryParser.parse_json(filepath)
        if not waypoints:
            messagebox.showerror("Error", "Failed to parse trajectory")
            return
        name = Path(filepath).stem
        color = (float(np.random.rand()), float(np.random.rand()), float(np.random.rand()))
        traj = Trajectory(name, waypoints, color)
        cleaned_traj = self.remove_duplicate_waypoints(traj)
        self.trajectories.append(cleaned_traj)
        pcd = o3d.geometry.PointCloud()
        pcd.paint_uniform_color(cleaned_traj.color)
        lineset = o3d.geometry.LineSet()
        cleaned_traj._o3d_pcd = pcd
        cleaned_traj._o3d_lineset = lineset
        self.selected_trajectory = cleaned_traj
        self.selected_waypoint = None
        self.update_waypoint_list()
        self.update_info()
        self.update_trajectory_visualization()
        messagebox.showinfo("Success", f"Loaded {len(waypoints)} waypoints (cleaned to {len(cleaned_traj.waypoints)})")

    def save_trajectory(self):
        if not self.selected_trajectory:
            messagebox.showwarning("Warning", "No trajectory selected")
            return
        filepath = filedialog.asksaveasfilename(title="Save Trajectory", defaultextension=".json",
                                                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if not filepath:
            return
        TrajectoryParser.save_json(filepath, self.selected_trajectory.waypoints)
        messagebox.showinfo("Success", "Trajectory saved successfully!")

    def remove_duplicate_waypoints(self, trajectory: Trajectory, tolerance: float = 0.01) -> Trajectory:
        if not trajectory or len(trajectory.waypoints) < 2:
            return trajectory
        cleaned_waypoints = [trajectory.waypoints[0]]
        for i in range(1, len(trajectory.waypoints)):
            current_pos = trajectory.waypoints[i].position
            prev_pos = cleaned_waypoints[-1].position
            if np.linalg.norm(current_pos - prev_pos) > tolerance:
                wp = trajectory.waypoints[i]
                wp.index = len(cleaned_waypoints)
                cleaned_waypoints.append(wp)
        return Trajectory(
            name=trajectory.name + " (cleaned)",
            waypoints=cleaned_waypoints,
            color=trajectory.color,
            interpolation_method=trajectory.interpolation_method
        )
    
    def on_fps_change(self, *args):
        """Update FPS label when slider changes"""
        fps = int(float(self.record_fps.get()))
        self.fps_label.config(text=str(fps))

    def on_show_all_toggle(self):
        """Toggle between showing all waypoint cameras or just the selected one"""
        self.update_trajectory_visualization()
        # Update FPS label when scale changes

    def _get_waypoint_at_fraction(self, traj, frac_index):
        """Get interpolated waypoint at fractional index (e.g., 2.5 = midway between wp 2 and 3)."""
        if not traj or len(traj.waypoints) < 2:
            return None
        base_idx = int(frac_index)
        t = frac_index - base_idx
        if base_idx >= len(traj.waypoints) - 1:
            return traj.waypoints[-1]  # Clamp to last
        wp1 = traj.waypoints[base_idx]
        wp2 = traj.waypoints[base_idx + 1]
        interp_pos = wp1.position * (1 - t) + wp2.position * t
        interp_yaw = wp1.yaw * (1 - t) + wp2.yaw * t
        interp_pitch = wp1.pitch * (1 - t) + wp2.pitch * t
        return Waypoint(position=interp_pos, yaw=interp_yaw, pitch=interp_pitch, index=base_idx)
    
    def update_trajectory_visualization(self):
        """Rebuild 3D scene with trajectories, waypoints, and cameras"""
        main_geoms = self.scene.o3d_geometries.copy()
        fpv_geoms = self.scene.o3d_geometries.copy()
        
        for traj in self.trajectories:
            points = traj.get_positions()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color(traj.color)
            
            #  Improved camera visualization logic
            if self.show_all_waypoints.get():
                # Show all cameras for this trajectory
                for i, wp in enumerate(traj.waypoints):
                    # Check if this specific waypoint is selected
                    is_selected = (traj == self.selected_trajectory and i == self.selected_waypoint)
                    camera_geom = self.generate_camera_visualization(
                        wp.position, wp.yaw, wp.pitch, 
                        scale=0.4 if is_selected else 0.2,  # Larger scale for selected
                        color=[1.0, 0.0, 0.0] if is_selected else list(traj.color)  # Red for selected
                    )
                    main_geoms.append(camera_geom)
            else:
                # Show cameras based on selection state
                if self.selected_trajectory and self.selected_waypoint is not None and traj == self.selected_trajectory:
                    # Only show the selected waypoint's camera for the selected trajectory
                    wp = traj.waypoints[self.selected_waypoint]
                    camera_geom = self.generate_camera_visualization(
                        wp.position, wp.yaw, wp.pitch, 
                        scale=0.4,  # Larger for visibility
                        color=[1.0, 0.0, 0.0]  # Bright red for selected camera
                    )
                    main_geoms.append(camera_geom)
                elif not self.selected_trajectory or traj != self.selected_trajectory:
                    # Show all cameras for non-selected trajectories (smaller, in trajectory color)
                    for i, wp in enumerate(traj.waypoints):
                        camera_geom = self.generate_camera_visualization(
                            wp.position, wp.yaw, wp.pitch, 
                            scale=0.15,  # Small for non-selected trajectories
                            color=list(traj.color)
                        )
                        main_geoms.append(camera_geom)
            
            #  Simplified path rendering logic
            try:
                # Generate interpolated path
                if traj.interpolation_method == InterpolationMethod.LINEAR:
                    path = TrajectoryInterpolator.interpolate_linear(traj.waypoints, num_points=200)
                else:
                    path = TrajectoryInterpolator.interpolate_spline(traj.waypoints, num_points=200)
                
                # Create path line set
                if len(path) > 1:
                    lines = [[i, i+1] for i in range(len(path)-1)]
                    lineset = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(path),
                        lines=o3d.utility.Vector2iVector(lines)
                    )
                    
                    # Highlight path for selected trajectory
                    if traj == self.selected_trajectory:
                        # Thicker/brighter line for selected trajectory
                        lineset.paint_uniform_color([min(1.0, c * 1.3) for c in traj.color])
                    else:
                        lineset.paint_uniform_color(traj.color)
                    
                    main_geoms.extend([pcd, lineset])
                else:
                    main_geoms.append(pcd)
                    
            except Exception as e:
                logger.warning(f"Failed to generate path for trajectory {traj.name}: {e}")
                # Fallback: just show waypoint cloud
                main_geoms.append(pcd)
        
        # Update both renderers
        self.update_loop_manager.update_main_geometries(main_geoms)
        self.update_loop_manager.update_fpv_geometries(fpv_geoms)

    def generate_camera_visualization(self, position, yaw, pitch, scale=0.5, color=None):
        """ Enhanced camera visualization with better geometry"""
        if color is None:
            color = [1.0, 0.0, 0.0]
        
        # Ensure color is a list for Open3D compatibility
        if isinstance(color, tuple):
            color = list(color)
        
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        
        # Create a more visible camera frustum
        points = [
            [0, 0, 0],                    # Camera center
            [-scale, -scale, scale*2],    # Bottom left
            [scale, -scale, scale*2],     # Bottom right  
            [scale, scale, scale*2],      # Top right
            [-scale, scale, scale*2],     # Top left
            [0, 0, scale*3]               # Forward direction indicator
        ]
        
        # Apply rotations
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
        rotated_points = []
        for point in points:
            # Apply pitch then yaw rotation
            rotated = np.dot(rot_yaw, np.dot(rot_pitch, point))
            rotated_points.append(rotated + position)
        
        # Define camera frustum lines
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # Camera center to corners
            [1, 2], [2, 3], [3, 4], [4, 1],  # Frame rectangle
            [0, 5]                            # Forward direction line
        ]
        
        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(rotated_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color(color)
        
        return line_set

    # FPV control callbacks
    def on_fpv_distance_change(self, *args):
        """Update FPV camera when distance slider changes"""
        distance = self.fpv_distance.get()
        self.fpv_distance_label.config(text=f"{distance:.1f} m")
        if self.selected_trajectory and self.selected_waypoint is not None:
            self._update_fpv_camera()
    
    def on_fpv_zoom_change(self, *args):
        """Update FPV camera when zoom slider changes"""
        zoom = self.fpv_zoom.get()
        self.fpv_zoom_label.config(text=f"{zoom:.3f}")
        if self.selected_trajectory and self.selected_waypoint is not None:
            self.fpv_manual_override['zoom'] = True  # Flag manual change
            self._update_fpv_camera()  # Apply immediately

    def open_camera_window(self):
        if hasattr(self, 'camera_window') and self.camera_window is not None and self.camera_window.winfo_exists():
            self.camera_window.lift()
            return
        self.camera_window = tk.Toplevel(self.root)
        self.camera_window.title("Camera Controls")
        self.camera_window.geometry("320x300")
        self.camera_window.protocol("WM_DELETE_WINDOW", lambda: self.camera_window.destroy())

        camera_frame = ttk.Frame(self.camera_window, padding=8)
        camera_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(camera_frame, text="Position:").pack()
        pos_frame = ttk.Frame(camera_frame)
        pos_frame.pack(fill=tk.X, pady=4)
        ttk.Label(pos_frame, text="X:").grid(row=0, column=0, padx=4)
        ttk.Scale(pos_frame, from_=-100, to=100, variable=self.camera_x, command=self.on_camera_slider).grid(row=0, column=1, sticky=tk.EW)
        ttk.Label(pos_frame, text="Y:").grid(row=1, column=0, padx=4)
        ttk.Scale(pos_frame, from_=-100, to=100, variable=self.camera_y, command=self.on_camera_slider).grid(row=1, column=1, sticky=tk.EW)
        ttk.Label(pos_frame, text="Z:").grid(row=2, column=0, padx=4)
        ttk.Scale(pos_frame, from_=-200, to=200, variable=self.camera_z, command=self.on_camera_slider).grid(row=2, column=1, sticky=tk.EW)
        pos_frame.columnconfigure(1, weight=1)

        ttk.Label(camera_frame, text="Rotation:").pack(pady=(8, 0))
        rot_frame = ttk.Frame(camera_frame)
        rot_frame.pack(fill=tk.X)
        ttk.Label(rot_frame, text="Yaw:").grid(row=0, column=0, padx=4)
        ttk.Scale(rot_frame, from_=-180, to=180, variable=self.camera_yaw, command=self.on_camera_slider).grid(row=0, column=1, sticky=tk.EW)
        ttk.Label(rot_frame, text="Pitch:").grid(row=1, column=0, padx=4)
        ttk.Scale(rot_frame, from_=-89, to=89, variable=self.camera_pitch, command=self.on_camera_slider).grid(row=1, column=1, sticky=tk.EW)
        rot_frame.columnconfigure(1, weight=1)

    def on_camera_slider(self, *args):
        self.camera.position = np.array([self.camera_x.get(), self.camera_y.get(), self.camera_z.get()], dtype=float)
        self.camera.yaw = self.camera_yaw.get()
        self.camera.pitch = self.camera_pitch.get()
        self._update_main_camera()
        self._update_fpv_to_main_camera()  # Update FPV when sliders change

    def toggle_recording(self):
        """Toggle recording without simulation (records current FPV view)"""
        if self.video_recorder and self.video_recorder.is_recording:
            self.stop_recording_only()
        else:
            self.start_recording_only()

    def toggle_simulation_pause(self):
        """Toggle pause state for simulation"""
        if self.flight_manager.flight_paused:
            self.flight_manager.resume_flight_simulation()
            self.sim_pause_button.config(text="Pause")
        else:
            self.flight_manager.pause_flight_simulation()
            self.sim_pause_button.config(text="Resume")

    def start_recording_with_simulation(self):
        """Start both recording and simulation together (original behavior)"""
        if not self.selected_trajectory or len(self.selected_trajectory.waypoints) < 2:
            if not self.trajectories or all(len(t.waypoints) < 2 for t in self.trajectories):
                messagebox.showwarning("No Waypoints", "Please add at least 2 waypoints to create a flight path.")
                return
            # Select first trajectory with waypoints if none selected
            for traj in self.trajectories:
                if len(traj.waypoints) >= 2:
                    self.selected_trajectory = traj
                    break
        
        # Parse resolution
        res_str = self.record_resolution.get()
        width, height = map(int, res_str.split('x'))
        fps = self.record_fps.get()
        
        # Get save filepath
        filepath = filedialog.asksaveasfilename(
            title="Save Recording", 
            defaultextension=".mp4", 
            filetypes=[("MP4", "*.mp4")]
        )
        if not filepath:
            return
        
        # Start video recording
        self.video_recorder = VideoRecorder(filepath, fps=fps, resolution=(width, height))
        self.video_recorder.start_recording()
        
        # Start simulation
        self.is_simulating = True
        self.flight_manager.start_flight_simulation()
        
        # Update UI
        self.simulate_button.config(text="Stop Simulation")
        self.sim_pause_button.config(state='normal')
        self.record_sim_button.config(text="Stop Recording + Simulation")
        self.flight_status_label.config(text="Recording and simulating...")
        self.recording_status_label.config(text=f"Recording to {Path(filepath).name}")

    def _update_main_camera(self):
        yaw = np.radians(self.camera.yaw)
        pitch = np.radians(self.camera.pitch)
        forward = np.array([np.cos(pitch) * np.sin(yaw),
                            np.sin(pitch),
                            np.cos(pitch) * np.cos(yaw)], dtype=float)
        if np.linalg.norm(forward) == 0:
            forward = np.array([0.0, 0.0, -1.0])
        forward /= np.linalg.norm(forward)
        lookat = (self.camera.position + forward).tolist()
        params = {
            'front': forward.tolist(),
            'up': [0.0, 1.0, 0.0],
            'lookat': lookat,
            'zoom': 0.5
        }
        self.update_loop_manager.update_main_camera(params)

    def _update_fpv_camera(self):
        # Use adjustable FPV controls
        if self.selected_trajectory and self.selected_waypoint is not None:
            wp = self.selected_trajectory.waypoints[self.selected_waypoint]
            yaw, pitch = np.radians(wp.yaw), np.radians(wp.pitch)
            
            # Calculate forward direction
            forward = np.array([np.cos(pitch) * np.sin(yaw), 
                              np.sin(pitch), 
                              np.cos(pitch) * np.cos(yaw)])
            if np.linalg.norm(forward) == 0:
                forward = np.array([0.0, 0.0, -1.0])
            forward /= np.linalg.norm(forward)
            
            # Get adjustable parameters
            look_distance = self.fpv_distance.get()  # How far ahead to look
            zoom = self.fpv_zoom.get()  # Zoom level
            
            # Calculate camera position (at waypoint position)
            camera_pos = wp.position
            
            # Look ahead from the waypoint position
            lookat = camera_pos + forward * look_distance
            
            # Set up vector
            up = np.array([0.0, 1.0, 0.0])
            
            params = {
                'lookat': lookat.tolist(),
                'front': forward.tolist(),
                'up': up.tolist(),
                'zoom': zoom
            }
            
            self.update_loop_manager.update_fpv_camera(params)

    def add_waypoint(self):
        pos = self.camera.position.copy()
        if not self.selected_trajectory:
            name = f"trajectory_{len(self.trajectories)+1}"
            wp = Waypoint(position=pos, yaw=self.camera.yaw, pitch=self.camera.pitch, index=0)
            traj = Trajectory(name=name, waypoints=[wp], color=(float(np.random.rand()), float(np.random.rand()), float(np.random.rand())))
            self.trajectories.append(traj)
            self.selected_trajectory = traj
            self.selected_waypoint = 0
        else:
            traj = self.selected_trajectory
            idx = len(traj.waypoints)
            wp = Waypoint(position=pos, yaw=self.camera.yaw, pitch=self.camera.pitch, index=idx)
            traj.waypoints.append(wp)
            self.selected_waypoint = idx
        self.update_waypoint_list()
        self.update_info()
        self.update_trajectory_visualization()
        self.selected_trajectory._cached_paths.clear()
        self.selected_trajectory._cache_key = None


    def edit_waypoint(self):
        if not self.selected_trajectory or self.selected_waypoint is None:
            messagebox.showwarning("Warning", "No trajectory/waypoint selected")
            return
        wp = self.selected_trajectory.waypoints[self.selected_waypoint]
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Waypoint")
        dialog.geometry("300x200")
        ttk.Label(dialog, text="X:").grid(row=0, column=0, padx=5, pady=2)
        x_var = tk.DoubleVar(value=wp.position[0])
        ttk.Entry(dialog, textvariable=x_var).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(dialog, text="Y:").grid(row=1, column=0, padx=5, pady=2)
        y_var = tk.DoubleVar(value=wp.position[1])
        ttk.Entry(dialog, textvariable=y_var).grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(dialog, text="Z:").grid(row=2, column=0, padx=5, pady=2)
        z_var = tk.DoubleVar(value=wp.position[2])
        ttk.Entry(dialog, textvariable=z_var).grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(dialog, text="Yaw:").grid(row=3, column=0, padx=5, pady=2)
        yaw_var = tk.DoubleVar(value=wp.yaw)
        ttk.Entry(dialog, textvariable=yaw_var).grid(row=3, column=1, padx=5, pady=2)
        ttk.Label(dialog, text="Pitch:").grid(row=4, column=0, padx=5, pady=2)
        pitch_var = tk.DoubleVar(value=wp.pitch)
        ttk.Entry(dialog, textvariable=pitch_var).grid(row=4, column=1, padx=5, pady=2)
        def save():
            try:
                wp.position = np.array([x_var.get(), y_var.get(), z_var.get()])
                wp.yaw, wp.pitch = yaw_var.get(), pitch_var.get()
                self.update_waypoint_list()
                self.update_info()
                self.update_trajectory_visualization()
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Invalid input: {e}")
        ttk.Button(dialog, text="Save", command=save).grid(row=5, column=0, columnspan=2, pady=10)
        self.selected_trajectory._cached_paths.clear()
        self.selected_trajectory._cache_key = None


    def delete_waypoint(self):
        if not self.selected_trajectory or self.selected_waypoint is None:
            messagebox.showwarning("Warning", "No trajectory/waypoint selected")
            return
        traj = self.selected_trajectory
        idx = self.selected_waypoint
        if 0 <= idx < len(traj.waypoints):
            traj.waypoints.pop(idx)
            for i, wp in enumerate(traj.waypoints):
                wp.index = i
            self.selected_waypoint = None
            self.update_waypoint_list()
            self.update_info()
            self.update_trajectory_visualization()
        self.selected_trajectory._cached_paths.clear()
        self.selected_trajectory._cache_key = None


    def on_waypoint_select(self, event):
        sel = event.widget.curselection()
        if not sel:
            return
        idx = sel[0]
        if not self.selected_trajectory:
            return
        if 0 <= idx < len(self.selected_trajectory.waypoints):
            self.selected_waypoint = idx
            # Update visualization to show only selected camera
            self.update_trajectory_visualization()
            # Update FPV to show selected waypoint's view
            self._update_fpv_camera()

    #  Ensure waypoint list updates properly
    def update_waypoint_list(self):
        """FIXED: Better waypoint list management with selection preservation"""
        # Remember current selection
        current_selection = None
        if self.waypoint_listbox.curselection():
            current_selection = self.waypoint_listbox.curselection()[0]
        
        # Clear and repopulate
        self.waypoint_listbox.delete(0, tk.END)
        
        if not self.selected_trajectory:
            return
        
        # Add waypoints with clear naming
        for i, wp in enumerate(self.selected_trajectory.waypoints):
            camera_name = f"Camera {i + 1}"
            pos_str = f"({wp.position[0]:.1f}, {wp.position[1]:.1f}, {wp.position[2]:.1f})"
            angle_str = f"(yaw:{wp.yaw:.0f}° pitch:{wp.pitch:.0f}°)"
            
            # Show position and angles for better identification
            display_text = f"{camera_name}: {pos_str} {angle_str}"
            self.waypoint_listbox.insert(tk.END, display_text)
        
        # Restore selection if valid
        if (current_selection is not None and 
            current_selection < self.waypoint_listbox.size() and
            current_selection == self.selected_waypoint):
            self.waypoint_listbox.selection_set(current_selection)
            self.waypoint_listbox.see(current_selection)

    def compare_trajectories(self):
        if len(self.trajectories) < 2:
            messagebox.showwarning("Warning", "Need at least 2 trajectories to compare")
            return
        output = []
        for traj in self.trajectories:
            metrics = traj.calculate_detailed_metrics(cruising_speed=self.flight_manager.cruising_speed.get())
            output.append({'name': traj.name, 'metrics': metrics})
        most_efficient = min(output, key=lambda x: x['metrics']['estimated_duration'])
        self.comp_text.delete(1.0, tk.END)
        self.comp_text.insert(tk.END, "TRAJECTORY COMPARISON\n" + "="*50 + "\n\n")
        header = (f"{'Trajectory':<20} {'Length (m)':<12} {'Vert Disp (m)':<12} "
                f"{'Turn Angle (°)':<14} {'Duration (s)':<12} {'Waypoints':<10}\n")
        separator = "-" * 85 + "\n"
        self.comp_text.insert(tk.END, header)
        self.comp_text.insert(tk.END, separator)
        for c in output:
            m = c['metrics']
            is_most_efficient = "(Most Efficient)" if c['name'] == most_efficient['name'] else ""
            row = (f"{c['name']:<20} {m['total_length']:<12.2f} {m['cumulative_vertical_displacement']:<12.2f} "
                f"{np.degrees(m['cumulative_turning_angle']):<14.2f} {m['estimated_duration']:<12.2f} "
                f"{m['num_waypoints']:<10} {is_most_efficient}\n")
            self.comp_text.insert(tk.END, row)
        self.comp_text.insert(tk.END, "\nSUMMARY:\n")
        self.comp_text.insert(tk.END, f"Most efficient: {most_efficient['name']} "
                                    f"({most_efficient['metrics']['estimated_duration']:.2f}s)\n")
        if len(output) > 1:
            self.comp_text.insert(tk.END, "\nEfficiency Comparison:\n")
            for c in output:
                if c['name'] != most_efficient['name']:
                    efficiency_ratio = c['metrics']['estimated_duration'] / most_efficient['metrics']['estimated_duration']
                    percent_slower = (efficiency_ratio - 1) * 100
                    self.comp_text.insert(tk.END, f"{c['name']} is {percent_slower:.1f}% slower than {most_efficient['name']}\n")

    def compare_interpolations(self):
        if not self.selected_trajectory:
            messagebox.showwarning("Warning", "No trajectory selected")
            return
        traj = self.selected_trajectory
        if len(traj.waypoints) < 2:
            messagebox.showwarning("Warning", "Not enough waypoints for interpolation")
            return
        linear_path = TrajectoryInterpolator.interpolate_linear(traj.waypoints, num_points=200)
        spline_path = TrajectoryInterpolator.interpolate_spline(traj.waypoints, num_points=200)
        
        def calculate_path_metrics(path):
            if len(path) < 2:
                return {
                    'length': 0.0,
                    'vertical_displacement': 0.0,
                    'turning_angle': 0.0
                }
            distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
            length = np.sum(distances)
            if path.shape[1] > 2:
                vertical_differences = np.abs(np.diff(path[:, 2]))
                vertical_displacement = np.sum(vertical_differences)
            else:
                vertical_displacement = 0.0
            turning_angle = 0.0
            if len(path) >= 3:
                for i in range(1, len(path) - 1):
                    vec1 = path[i] - path[i-1]
                    vec2 = path[i+1] - path[i]
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    if norm1 > 0 and norm2 > 0:
                        cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle = np.arccos(cos_angle)
                        turning_angle += angle
            return {
                'length': length,
                'vertical_displacement': vertical_displacement,
                'turning_angle': turning_angle
            }
        
        linear_metrics = calculate_path_metrics(linear_path)
        spline_metrics = calculate_path_metrics(spline_path)
        
        self.comp_text.delete(1.0, tk.END)
        self.comp_text.insert(tk.END, f"INTERPOLATION COMPARISON: {traj.name}\n")
        self.comp_text.insert(tk.END, "="*50 + "\n\n")
        self.comp_text.insert(tk.END, "LINEAR INTERPOLATION:\n")
        self.comp_text.insert(tk.END, f"  • Total length: {linear_metrics['length']:.2f} m\n")
        self.comp_text.insert(tk.END, f"  • Cumulative vertical displacement: {linear_metrics['vertical_displacement']:.2f} m\n")
        self.comp_text.insert(tk.END, f"  • Cumulative turning angle: {np.degrees(linear_metrics['turning_angle']):.2f}°\n")
        self.comp_text.insert(tk.END, f"  • Direct waypoint connections\n\n")
        self.comp_text.insert(tk.END, "SPLINE INTERPOLATION:\n")
        self.comp_text.insert(tk.END, f"  • Total length: {spline_metrics['length']:.2f} m\n")
        self.comp_text.insert(tk.END, f"  • Cumulative vertical displacement: {spline_metrics['vertical_displacement']:.2f} m\n")
        self.comp_text.insert(tk.END, f"  • Cumulative turning angle: {np.degrees(spline_metrics['turning_angle']):.2f}°\n")
        self.comp_text.insert(tk.END, f"  • Smooth continuous path\n\n")
        length_diff = spline_metrics['length'] - linear_metrics['length']
        length_diff_percent = (length_diff / linear_metrics['length']) * 100 if linear_metrics['length'] > 0 else 0
        turning_diff = spline_metrics['turning_angle'] - linear_metrics['turning_angle']
        turning_diff_percent = (turning_diff / linear_metrics['turning_angle']) * 100 if linear_metrics['turning_angle'] > 0 else 0
        self.comp_text.insert(tk.END, "COMPARISON:\n")
        self.comp_text.insert(tk.END, f"  • Length difference: {length_diff:+.2f} m ({length_diff_percent:+.1f}%)\n")
        self.comp_text.insert(tk.END, f"  • Turning angle difference: {np.degrees(turning_diff):+.2f}° ({turning_diff_percent:+.1f}%)\n")
        if length_diff > 0:
            self.comp_text.insert(tk.END, "  • Spline path is longer but smoother\n")
        else:
            self.comp_text.insert(tk.END, "  • Spline path is shorter and smoother\n")
        self.show_both_interpolations = True
        self.update_trajectory_visualization()
        messagebox.showinfo("Interpolation Comparison",
                            f"Linear: {linear_metrics['length']:.2f}m, {np.degrees(linear_metrics['turning_angle']):.2f}° turning\n"
                            f"Spline: {spline_metrics['length']:.2f}m, {np.degrees(spline_metrics['turning_angle']):.2f}° turning")

    def update_interpolation(self):
        if not self.selected_trajectory:
            return
        self.selected_trajectory.interpolation_method = InterpolationMethod.LINEAR if self.interp_method.get() == 'linear' else InterpolationMethod.SPLINE
        self.update_trajectory_visualization()

    def on_speed_change(self, *args):
        """Update speed label when slider changes"""
        speed = self.simulation_speed.get()
        self.speed_label.config(text=f"{speed:.1f} sec")
    
    def toggle_pause(self):
        """Toggle pause state during simulation"""
        self.simulation_paused = not self.simulation_paused
        self.pause_button.config(text="Resume" if self.simulation_paused else "Pause")
        self.flight_manager.flight_paused = self.simulation_paused
    

    
    def start_simulation(self, filepath, fps, resolution):
        """Start flight simulation without recording"""
        if not self.selected_trajectory or len(self.selected_trajectory.waypoints) < 2:
            if not self.trajectories or all(len(t.waypoints) < 2 for t in self.trajectories):
                messagebox.showwarning("No Waypoints", "Please add at least 2 waypoints to create a flight path.")
                return
            # Select first trajectory with waypoints if none selected
            for traj in self.trajectories:
                if len(traj.waypoints) >= 2:
                    self.selected_trajectory = traj
                    break
        
        # Start simulation
        self.is_simulating = True
        self.flight_manager.start_flight_simulation()
        
        # Update UI
        self.simulate_button.config(text="Stop Simulation")
        self.sim_pause_button.config(state='normal')
        self.flight_status_label.config(text="Simulation started")

    def start_simulation_only(self):
        """Start flight simulation without recording"""
        if not self.selected_trajectory or len(self.selected_trajectory.waypoints) < 2:
            if not self.trajectories or all(len(t.waypoints) < 2 for t in self.trajectories):
                messagebox.showwarning("No Waypoints", "Please add at least 2 waypoints to create a flight path.")
                return
            # Select first trajectory with waypoints if none selected
            for traj in self.trajectories:
                if len(traj.waypoints) >= 2:
                    self.selected_trajectory = traj
                    break
        
        # Start simulation
        self.is_simulating = True
        self.flight_manager.start_flight_simulation()
        
        # Update UI
        self.simulate_button.config(text="Stop Simulation")
        self.sim_pause_button.config(state='normal')
        self.flight_status_label.config(text="Simulation started (no recording)")

    def stop_simulation_only(self):
        """Stop flight simulation"""
        self.is_simulating = False
        self.flight_manager.stop_flight_simulation()
        
        # Update UI
        self.simulate_button.config(text="Start Flight Simulation")
        self.sim_pause_button.config(state='disabled', text="Pause")
        self.flight_status_label.config(text="Simulation stopped")
    
    def _run_simulation(self):
        """Run the simulation in a separate thread"""
        waypoints = self.selected_trajectory.waypoints
        num_waypoints = len(waypoints)
        
        # Hover at start
        time.sleep(self.flight_manager.hover_time.get())
        
        while self.is_simulating and self.current_simulation_waypoint < num_waypoints:
            if not self.simulation_paused:
                # Update selected waypoint
                self.selected_waypoint = self.current_simulation_waypoint
                
                # Update UI in main thread
                self.root.after(0, self._update_simulation_ui)
                
                # Get current and next waypoint for interpolation
                current_wp = waypoints[self.current_simulation_waypoint]
                
                if self.current_simulation_waypoint < num_waypoints - 1:
                    next_wp = waypoints[self.current_simulation_waypoint + 1]
                    
                    # Calculate steps based on distance and speed
                    dist = np.linalg.norm(next_wp.position - current_wp.position)
                    steps = int((dist / self.flight_manager.cruising_speed.get()) * self.record_fps.get())
                    steps = max(steps, 1)
                    
                    for step in range(steps):
                        if not self.is_simulating or self.simulation_paused:
                            if self.simulation_paused:
                                while self.simulation_paused and self.is_simulating:
                                    time.sleep(0.1)
                            if not self.is_simulating:
                                break
                        
                        # Calculate interpolation factor
                        t = step / float(steps)
                        
                        # Interpolate position
                        interp_pos = current_wp.position * (1 - t) + next_wp.position * t
                        
                        # Interpolate rotation (simple linear interpolation)
                        interp_yaw = current_wp.yaw * (1 - t) + next_wp.yaw * t
                        interp_pitch = current_wp.pitch * (1 - t) + next_wp.pitch * t
                        
                        # Create temporary waypoint for interpolated position
                        temp_wp = Waypoint(
                            position=interp_pos,
                            yaw=interp_yaw,
                            pitch=interp_pitch,
                            index=self.current_simulation_waypoint
                        )
                        
                        # Update FPV camera to interpolated position
                        self.root.after(0, lambda wp=temp_wp: self._update_fpv_for_waypoint(wp))
                        
                        # Wait for next frame
                        time.sleep(1.0 / self.record_fps.get())
                    
                    # Hover at waypoint
                    time.sleep(self.flight_manager.hover_time.get())
                    
                    # Move to next waypoint
                    self.current_simulation_waypoint += 1
                else:
                    # Last waypoint - hold for the duration
                    time.sleep(self.flight_manager.hover_time.get())
                    self.current_simulation_waypoint += 1
            else:
                # Paused
                time.sleep(0.1)
        
        # Simulation complete
        if self.is_simulating:
            self.root.after(0, lambda: setattr(self, 'selected_waypoint', self.current_simulation_waypoint % len(self.selected_trajectory.waypoints) if self.selected_trajectory else 0))

    def toggle_simulation(self):
        """Toggle flight simulation without recording"""
        if self.flight_manager.flight_simulation_active:
            self.stop_simulation_only()
        else:
            self.start_simulation_only()

    def _update_simulation_ui(self):
        """Update UI during simulation (called in main thread)"""
        # Update waypoint list selection
        self.waypoint_listbox.selection_clear(0, tk.END)
        if self.selected_waypoint < self.waypoint_listbox.size():
            self.waypoint_listbox.selection_set(self.selected_waypoint)
            self.waypoint_listbox.see(self.selected_waypoint)
        
        # Update progress label with camera names
        total = len(self.selected_trajectory.waypoints) if self.selected_trajectory else 0
        current = self.current_simulation_waypoint + 1
        camera_name = f"Camera {current}"
        self.progress_label.config(text=f"Recording: {camera_name} ({current}/{total})")
        
        # Update visualization
        self.update_trajectory_visualization_lightweight()
        self._update_fpv_camera()
    
    def _update_fpv_for_waypoint(self, wp):
        """Update FPV camera for a specific waypoint"""
        yaw, pitch = np.radians(wp.yaw), np.radians(wp.pitch)
        
        # Calculate forward direction
        forward = np.array([np.cos(pitch) * np.sin(yaw), 
                          np.sin(pitch), 
                          np.cos(pitch) * np.cos(yaw)])
        if np.linalg.norm(forward) == 0:
            forward = np.array([0.0, 0.0, -1.0])
        forward /= np.linalg.norm(forward)
        
        # Get adjustable parameters
        look_distance = self.fpv_distance.get()
        zoom = self.fpv_zoom.get()
        
        # Calculate camera position
        camera_pos = wp.position
        lookat = camera_pos + forward * look_distance
        
        up = np.array([0.0, 1.0, 0.0])
        
        params = {
            'lookat': lookat.tolist(),
            'front': forward.tolist(),
            'up': up.tolist(),
            'zoom': zoom
        }
        
        self.update_loop_manager.update_fpv_camera(params)
    
    def stop_simulation(self):
        """Stop both simulation and recording (if active)"""
        # Stop simulation
        self.is_simulating = False
        self.flight_manager.stop_flight_simulation()
        
        # Stop recording if active
        if self.video_recorder and self.video_recorder.is_recording:
            self.video_recorder.stop_recording()
            self.video_recorder = None
        
        # Update UI
        self.simulate_button.config(text="Start Flight Simulation")
        self.sim_pause_button.config(state='disabled', text="Pause")
        self.record_button.config(text="Start Recording")
        self.record_sim_button.config(text="Record + Simulate")
        self.flight_status_label.config(text="Simulation complete")
        self.recording_status_label.config(text="Not recording")
        
        if self.video_recorder:  # Was recording
            messagebox.showinfo("Complete", "Simulation and recording have been saved.")
        else:
            messagebox.showinfo("Complete", "Simulation complete.")

    def on_canvas_resize(self, event):
        """FIXED: Debounced canvas resize with minimum size check"""
        width = max(event.width, 100)  # Minimum width
        height = max(event.height, 100)  # Minimum height
        
        # Debounce rapid resize events
        if hasattr(self, '_resize_timer'):
            self.root.after_cancel(self._resize_timer)
        
        self._resize_timer = self.root.after(100, lambda: self._do_main_resize(width, height))

    def _do_main_resize(self, width, height):
        """Perform the actual main canvas resize"""
        try:
            if hasattr(self, 'update_loop_manager') and self.update_loop_manager:
                self.update_loop_manager.resize_main(width, height)
        except Exception as e:
            logger.error(f"Main canvas resize error: {e}")

    def _do_fpv_resize(self, width, height):
        """Perform the actual FPV resize"""
        try:
            if hasattr(self, 'update_loop_manager') and self.update_loop_manager:
                self.update_loop_manager.resize_fpv(width, height)
        except Exception as e:
            logger.error(f"FPV resize error: {e}")

    def on_fpv_resize(self, event):
        """FIXED: Debounced FPV resize with minimum size check"""
        width = max(event.width, 100)  # Minimum width  
        height = max(event.height, 100)  # Minimum height
        
        # Debounce rapid resize events
        if hasattr(self, '_fpv_resize_timer'):
            self.root.after_cancel(self._fpv_resize_timer)
        
        self._fpv_resize_timer = self.root.after(100, lambda: self._do_fpv_resize(width, height))

    def setup_open3d_environment():
        """Initialize Open3D with better error handling"""
        try:
            import open3d as o3d
            
            # Try to suppress some Open3D warnings
            import logging as o3d_logging
            o3d_logging.getLogger("Open3D").setLevel(o3d_logging.ERROR)
            
            # Set Open3D to use software rendering if hardware fails
            try:
                # This might help with context issues
                import os
                os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
                os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
            except:
                pass
                
            logger.info("Open3D environment configured")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Open3D environment: {e}")
            return False

    def on_mouse_press(self, event):
        self._mouse_last = (event.x, event.y)
        self._is_dragging = True

    def on_mouse_drag(self, event):
        if not self._is_dragging or self._mouse_last is None:
            self._mouse_last = (event.x, event.y)
            return
        dx = event.x - self._mouse_last[0]
        dy = event.y - self._mouse_last[1]
        self.camera.yaw += dx * 0.15
        self.camera.pitch = np.clip(self.camera.pitch - dy * 0.15, -89, 89)
        self.camera_x.set(self.camera.position[0])
        self.camera_y.set(self.camera.position[1])
        self.camera_z.set(self.camera.position[2])
        self.camera_yaw.set(self.camera.yaw)
        self.camera_pitch.set(self.camera.pitch)
        self._update_main_camera()
        self._update_fpv_to_main_camera()  # Update FPV when mouse drags
        self._mouse_last = (event.x, event.y)

    def on_mouse_release(self, event):
        self._is_dragging = False
        self._mouse_last = None

    def on_mouse_wheel(self, event):
        delta = event.delta / 120
        self.camera.position[2] += -delta * 0.5
        self.camera_z.set(self.camera.position[2])
        self._update_main_camera()
        self._update_fpv_to_main_camera()  # Update FPV when mouse wheel zooms

    def on_key_press(self, event):
        """Handle keyboard input for selected waypoint camera control"""
        key = event.keysym.lower()
        
        # Check if we have a selected waypoint to control
        if not self.selected_trajectory or self.selected_waypoint is None:
            if key == 'h':  # Always allow help
                self.show_keyboard_help()
            else:
                # Show message that no camera is selected
                self.progress_label.config(text="Select a camera from the list first")
            return
        
        # Get the selected waypoint
        wp = self.selected_trajectory.waypoints[self.selected_waypoint]
        camera_name = f"Camera {self.selected_waypoint + 1}"
        
        # Flag to track if we need to update visualizations
        needs_update = False
        only_fpv_update = False
        
        # Movement controls - modify waypoint position
        if key == 'w':  # Move forward in view direction
            forward = self._get_forward_vector(wp.yaw, wp.pitch)
            wp.position += forward * 0.5
            needs_update = True
        elif key == 's':  # Move backward
            forward = self._get_forward_vector(wp.yaw, wp.pitch)
            wp.position -= forward * 0.5
            needs_update = True
        elif key == 'a':  # Strafe left
            right = self._get_right_vector(wp.yaw)
            wp.position -= right * 0.5
            needs_update = True
        elif key == 'd':  # Strafe right
            right = self._get_right_vector(wp.yaw)
            wp.position += right * 0.5
            needs_update = True
        elif key == 'q':  # Move up
            wp.position[2] += 0.5
            needs_update = True
        elif key == 'e':  # Move down
            wp.position[2] -= 0.5
            needs_update = True
        
        # Rotation controls - modify waypoint orientation
        elif key == 'left':  # Yaw left
            wp.yaw -= 2.0
            if wp.yaw < -180:
                wp.yaw += 360
            only_fpv_update = True  # Rotation only affects FPV, not main geometry
        elif key == 'right':  # Yaw right
            wp.yaw += 2.0
            if wp.yaw > 180:
                wp.yaw -= 360
            only_fpv_update = True
        elif key == 'up':  # Pitch up
            wp.pitch = np.clip(wp.pitch + 2.0, -89, 89)
            only_fpv_update = True
        elif key == 'down':  # Pitch down
            wp.pitch = np.clip(wp.pitch - 2.0, -89, 89)
            only_fpv_update = True
        
        # Zoom controls for FPV view (only update FPV)
        elif key in ['z', 'plus', 'equal']:  # Zoom in FPV
            current_zoom = self.fpv_zoom.get()
            self.fpv_zoom.set(max(0.01, current_zoom - 0.01))
            self.on_fpv_zoom_change()
            return  # No other updates needed
        elif key in ['x', 'minus']:  # Zoom out FPV
            current_zoom = self.fpv_zoom.get()
            self.fpv_zoom.set(min(1.0, current_zoom + 0.01))
            self.on_fpv_zoom_change()
            return  # No other updates needed
        
        # FPV distance controls (only update FPV)
        elif key == 'f':  # Increase FPV look-ahead distance
            current = self.fpv_distance.get()
            self.fpv_distance.set(min(50.0, current + 1.0))
            self.on_fpv_distance_change()
            return  # No other updates needed
        elif key == 'v':  # Decrease FPV look-ahead distance
            current = self.fpv_distance.get()
            self.fpv_distance.set(max(0.5, current - 1.0))
            self.on_fpv_distance_change()
            return  # No other updates needed
        
        # Save current position (update the waypoint)
        elif key == 'space':  # Update waypoint with current changes
            self.progress_label.config(text=f"{camera_name} position updated")
            return
        
        # Help
        elif key == 'h':
            self.show_keyboard_help()
            return
        else:
            return  # Key not handled
        
        # Batch updates for better performance
        if only_fpv_update:
            # Only update FPV for rotation changes
            self._update_fpv_camera()
            self.progress_label.config(text=f"Rotating {camera_name}")
        elif needs_update:
            # Schedule updates with a small delay to batch rapid keypresses
            if hasattr(self, '_update_timer'):
                self.root.after_cancel(self._update_timer)
            self._update_timer = self.root.after(50, self._perform_batch_update)
            self.progress_label.config(text=f"Moving {camera_name}")
    
    def _perform_batch_update(self):
        """Perform batched updates for better performance"""
        # Update waypoint list (lightweight)
        self.update_waypoint_list()
        
        # Update FPV camera first (most important for feedback)
        self._update_fpv_camera()
        
        # Defer heavy visualization update
        if hasattr(self, '_viz_update_timer'):
            self.root.after_cancel(self._viz_update_timer)
        self._viz_update_timer = self.root.after(100, self.update_trajectory_visualization_lightweight)
    
    def update_trajectory_visualization_lightweight(self):
        """Ultra-lightweight update for flight simulation"""
        if not self.selected_trajectory:
            return
        
        main_geoms = self.scene.o3d_geometries.copy()
        
        # Only update the essential elements
        for traj in self.trajectories:
            if traj == self.selected_trajectory:
                # For selected trajectory, just show path and current waypoint
                points = traj.get_positions()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.paint_uniform_color(traj.color)
                
                # Simple path (fewer points for performance)
                if len(points) > 1:
                    path = TrajectoryInterpolator.interpolate_linear(traj.waypoints, num_points=50)  # Reduced points
                    lines = [[i, i+1] for i in range(len(path)-1)]
                    lineset = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(path),
                        lines=o3d.utility.Vector2iVector(lines)
                    )
                    lineset.paint_uniform_color(traj.color)
                    main_geoms.extend([pcd, lineset])
                else:
                    main_geoms.append(pcd)
                
                # Show current waypoint camera only
                if hasattr(self.flight_manager, 'selected_waypoint') and self.flight_manager.selected_waypoint is not None:
                    wp = traj.waypoints[self.flight_manager.selected_waypoint]
                    camera_geom = self.generate_camera_visualization(
                        wp.position, wp.yaw, wp.pitch, 
                        scale=0.3, color=[1.0, 0.0, 0.0]
                    )
                    main_geoms.append(camera_geom)
        
        self.update_loop_manager.update_main_geometries(main_geoms)
    
    def _get_forward_vector(self, yaw, pitch):
        """Calculate forward direction vector from yaw and pitch"""
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        forward = np.array([
            np.cos(pitch_rad) * np.sin(yaw_rad),
            np.sin(pitch_rad),
            np.cos(pitch_rad) * np.cos(yaw_rad)
        ])
        return forward / np.linalg.norm(forward) if np.linalg.norm(forward) > 0 else np.array([0, 0, -1])
    
    def start_recording_only(self):
        """Start recording FPV view without simulation"""
        # Parse resolution
        res_str = self.record_resolution.get()
        width, height = map(int, res_str.split('x'))
        fps = self.record_fps.get()
        
        # Get save filepath
        filepath = filedialog.asksaveasfilename(
            title="Save Recording", 
            defaultextension=".mp4", 
            filetypes=[("MP4", "*.mp4")]
        )
        if not filepath:
            return
        
        # Start video recording
        self.video_recorder = VideoRecorder(filepath, fps=fps, resolution=(width, height))
        self.video_recorder.start_recording()
        
        # Update UI
        self.record_button.config(text="Stop Recording")
        self.recording_status_label.config(text=f"Recording to {Path(filepath).name}")
        
        messagebox.showinfo("Recording Started", 
                        f"Recording FPV view to {Path(filepath).name}\n"
                        f"Move camera or run simulation to capture footage.")
    
    def stop_recording_only(self):
        """Stop recording"""
        if self.video_recorder and self.video_recorder.is_recording:
            self.video_recorder.stop_recording()
            self.video_recorder = None
        
        # Update UI
        self.record_button.config(text="Start Recording")
        self.recording_status_label.config(text="Recording stopped")
    
    def _get_right_vector(self, yaw):
        """Calculate right direction vector from yaw"""
        yaw_rad = np.radians(yaw)
        right = np.array([
            np.cos(yaw_rad),
            0,
            -np.sin(yaw_rad)
        ])
        return right / np.linalg.norm(right) if np.linalg.norm(right) > 0 else np.array([1, 0, 0])
    
    def show_keyboard_help(self):
        """Display keyboard shortcuts help dialog"""
        help_text = """
KEYBOARD SHORTCUTS FOR SELECTED CAMERA CONTROL:

IMPORTANT: First select a camera from the waypoint list!

CAMERA MOVEMENT (moves selected camera):
  W : Move Forward (in camera's view direction)
  S : Move Backward
  A : Strafe Left
  D : Strafe Right
  Q : Move Up (increase altitude)
  E : Move Down (decrease altitude)

CAMERA ROTATION (rotates selected camera):
  Arrow Left  : Yaw Left
  Arrow Right : Yaw Right
  Arrow Up    : Pitch Up (look up)
  Arrow Down  : Pitch Down (look down)

FPV VIEW CONTROLS:
  Z / + : Zoom In (narrower FOV)
  X / - : Zoom Out (wider FOV)
  F : Increase Look-Ahead Distance
  V : Decrease Look-Ahead Distance

OTHER:
  SPACE : Confirm position (feedback message)
  H : Show This Help

HOW IT WORKS:
1. Select a camera from the waypoint list
2. Use keyboard to adjust that camera's position/angle
3. FPV shows the view from that camera
4. Main view shows the camera's position (red icon)
5. All changes are automatically saved to the waypoint

Note: Changes modify the actual waypoint data.
        """
        messagebox.showinfo("Camera Control Shortcuts", help_text)
    
    def on_key_release(self, event):
        """Handle key release events if needed"""
        pass
    
    def _update_fpv_to_main_camera(self):
        """Update FPV to show the main camera's current view"""
        yaw = np.radians(self.camera.yaw)
        pitch = np.radians(self.camera.pitch)
        
        # Calculate forward direction
        forward = np.array([np.cos(pitch) * np.sin(yaw), 
                          np.sin(pitch), 
                          np.cos(pitch) * np.cos(yaw)])
        if np.linalg.norm(forward) == 0:
            forward = np.array([0.0, 0.0, -1.0])
        forward /= np.linalg.norm(forward)
        
        # Use current FPV settings for distance and zoom
        look_distance = self.fpv_distance.get()
        zoom = self.fpv_zoom.get()
        
        # Calculate where the camera is looking
        camera_pos = self.camera.position
        lookat = camera_pos + forward * look_distance
        
        # Set up vector
        up = np.array([0.0, 1.0, 0.0])
        
        params = {
            'lookat': lookat.tolist(),
            'front': forward.tolist(),
            'up': up.tolist(),
            'zoom': zoom
        }
        
        self.update_loop_manager.update_fpv_camera(params)
    
    def add_waypoint_from_current_camera(self):
        """Add a waypoint at the current main camera position and orientation"""
        pos = self.camera.position.copy()
        yaw = self.camera.yaw
        pitch = self.camera.pitch
        
        if not self.selected_trajectory:
            # Create new trajectory if none exists
            name = f"trajectory_{len(self.trajectories)+1}"
            wp = Waypoint(position=pos, yaw=yaw, pitch=pitch, index=0)
            traj = Trajectory(name=name, waypoints=[wp], 
                            color=(float(np.random.rand()), float(np.random.rand()), float(np.random.rand())))
            self.trajectories.append(traj)
            self.selected_trajectory = traj
            self.selected_waypoint = 0
            camera_num = 1
        else:
            # Add to existing trajectory
            traj = self.selected_trajectory
            idx = len(traj.waypoints)
            wp = Waypoint(position=pos, yaw=yaw, pitch=pitch, index=idx)
            traj.waypoints.append(wp)
            self.selected_waypoint = idx
            camera_num = idx + 1
        
        self.update_waypoint_list()
        self.update_info()
        self.update_trajectory_visualization()
        messagebox.showinfo("Camera Added", 
                          f"Camera {camera_num} added\n"
                          f"Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})\n"
                          f"Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}°")
    
    def show_keyboard_help(self):
        """Display keyboard shortcuts help dialog"""
        help_text = """
KEYBOARD SHORTCUTS FOR CAMERA CONTROL:

The FPV window shows what the main camera sees as you move it.

CAMERA MOVEMENT:
  W : Move Forward
  S : Move Backward
  A : Move Left
  D : Move Right
  Q : Move Up
  E : Move Down

CAMERA ROTATION:
  Arrow Left  : Rotate Left (Yaw)
  Arrow Right : Rotate Right (Yaw)
  Arrow Up    : Rotate Up (Pitch)
  Arrow Down  : Rotate Down (Pitch)

ZOOM:
  Z or +     : Zoom In
  X or -     : Zoom Out
  Mouse Wheel: Zoom In/Out

WAYPOINT:
  SPACE : Save current camera view as waypoint

OTHER:
  R : Reset Camera to Default Position
  H : Show This Help

MOUSE CONTROLS:
  Drag: Rotate camera
  Wheel: Zoom

Note: Click on the 3D view to enable keyboard controls.
The FPV window shows a live preview of what the camera sees.
        """
        messagebox.showinfo("Keyboard Shortcuts", help_text)

    def _display_main_frame(self, arr: np.ndarray):
        try:
            img = Image.fromarray(arr)
            cw = self.main_canvas.winfo_width() or 960
            ch = self.main_canvas.winfo_height() or 720
            img = img.resize((cw, ch), Image.LANCZOS)
            self._main_photo = ImageTk.PhotoImage(img)
            self.main_canvas.delete("all")
            self.main_canvas.create_image(0, 0, anchor=tk.NW, image=self._main_photo)
        except Exception as e:
            logger.error(f"_display_main_frame error: {e}")

    def _display_fpv_frame(self, arr: np.ndarray):
        try:
            img = Image.fromarray(arr)
            w = self.fpv_label.winfo_width() or 400
            h = self.fpv_label.winfo_height() or 300
            img = img.resize((w, h), Image.LANCZOS)
            self._fpv_photo = ImageTk.PhotoImage(img)
            self.fpv_label.configure(image=self._fpv_photo)
        except Exception as e:
            logger.error(f"_display_fpv_frame error: {e}")

    def update_info(self):
        self.info_text.delete(1.0, tk.END)
        if not self.selected_trajectory:
            self.info_text.insert(tk.END, "No trajectory selected\n")
            return
        m = self.selected_trajectory.calculate_detailed_metrics(cruising_speed=self.flight_manager.cruising_speed.get())
        self.info_text.insert(tk.END, f"TRAJECTORY: {self.selected_trajectory.name}\n")
        self.info_text.insert(tk.END, "="*40 + "\n\n")
        self.info_text.insert(tk.END, "METRICS:\n")
        self.info_text.insert(tk.END, f"• Waypoints: {m['num_waypoints']}\n")
        self.info_text.insert(tk.END, f"• Total path length: {m['total_length']:.2f} m\n")
        self.info_text.insert(tk.END, f"• Cumulative vertical displacement: {m['cumulative_vertical_displacement']:.2f} m\n")
        self.info_text.insert(tk.END, f"• Cumulative turning angle: {np.degrees(m['cumulative_turning_angle']):.2f}° "
                                    f"({m['cumulative_turning_angle']:.2f} rad)\n")
        self.info_text.insert(tk.END, f"• Estimated flight duration: {m['estimated_duration']:.2f} s\n")
        efficiency = Trajectory.calculate_efficiency_rating(m)
        self.info_text.insert(tk.END, f"• Efficiency rating: {efficiency:.1f}/10\n")

    def close(self):
        try:
            self.update_loop_manager.stop()
        except Exception as e:
            logger.error(f"Error stopping update loop: {e}")

# -------------------------------
# Entry point
# -------------------------------
def main():

    root = tk.Tk()
    app = UAVVisualizationApp(root)
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            app.close()
            root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == '__main__':
    main()