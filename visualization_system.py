#!/usr/bin/env python3
"""
Production UAV Visualizer - Fixed Architecture
Addresses renderer recreation issues, Open3D API mismatches, and Trimesh conversion.
Includes dynamic view fitting, improved FPV setup, lighting, and mouse controls.
Adds oriented waypoint markers (arrows) and directional arrows along trajectory.
Implements debounced resize and background loading.
"""

import sys
import os
import time
import json
import queue
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from enum import Enum
import logging
import math

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
# Interpolation Utilities
# -------------------------------
class TrajectoryInterpolator:
    @staticmethod
    def interpolate_linear(waypoints: List[Waypoint], num_points: int) -> np.ndarray:
        positions = np.array([wp.position for wp in waypoints])
        if len(positions) < 2:
            return positions
        t = np.linspace(0, 1, len(positions))
        interpolator = interp1d(t, positions, axis=0)
        t_new = np.linspace(0, 1, num_points)
        return interpolator(t_new)

    @staticmethod
    def interpolate_spline(waypoints: List[Waypoint], num_points: int) -> np.ndarray:
        positions = np.array([wp.position for wp in waypoints])
        if len(positions) < 3:
            return TrajectoryInterpolator.interpolate_linear(waypoints, num_points)
        t = np.arange(len(positions))
        cs = CubicSpline(t, positions, axis=0, bc_type='natural')
        t_new = np.linspace(0, len(positions) - 1, num_points)
        return cs(t_new)

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
# Persistent Renderer with Debounced Resize
# -------------------------------
class PersistentRenderer:
    """Persistent Open3D renderer with debounced resize support"""
    
    def __init__(self, name: str, width: int = 960, height: int = 720):
        self.name = name
        self.width = width
        self.height = height
        self.vis: Optional[o3d.visualization.Visualizer] = None
        self.geometry_cache = GeometryCache()
        self._is_initialized = False
        self._view_control = None
        self._camera_params = {
            'lookat': np.array([0.0, 0.0, 0.0]),
            'front': np.array([0.0, 0.0, -1.0]),
            'up': np.array([0.0, 1.0, 0.0]),
            'zoom': 0.7
        }
        self._geometries: List[o3d.geometry.Geometry] = []  # Track geometries manually
    
    def initialize(self) -> bool:
        """Initialize the renderer"""
        if self._is_initialized:
            return True
            
        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name=self.name,
                width=self.width,
                height=self.height,
                visible=False
            )
            self._view_control = self.vis.get_view_control()
            self._setup_render_options()
            self._apply_camera_params()
            self._is_initialized = True
            logger.info(f"{self.name}: Renderer initialized successfully")
            return True
        except Exception as e:
            logger.error(f"{self.name}: Failed to initialize: {e}")
            return False
    
    def _setup_render_options(self):
        """Setup rendering options"""
        if self.vis:
            opt = self.vis.get_render_option()
            opt.background_color = np.array([0.15, 0.15, 0.15])
            opt.light_on = True
            opt.point_size = 3.0
            opt.line_width = 2.0
    
    def update_viewport(self, width: int, height: int):
        """Update viewport size with recreation"""
        if width < 1 or height < 1:
            return
        if self.width == width and self.height == height:
            return
            
        logger.debug(f"{self.name}: Resizing to {width}x{height}")
        self.cleanup()
        self.width = width
        self.height = height
        self.initialize()
        # Re-add geometries
        self.update_geometries(self._geometries)
        self._apply_camera_params()
    
    def update_geometries(self, geometries: List[o3d.geometry.Geometry]):
        """Update geometries efficiently using cache"""
        if not self._is_initialized or not self.vis:
            return
            
        try:
            self.vis.clear_geometries()
            for geom in geometries:
                if geom is not None:
                    self.vis.add_geometry(geom, reset_bounding_box=False)
            self._geometries = geometries.copy()  # Update tracked geometries
            self.fit_view()
            self.vis.update_renderer()
        except Exception as e:
            logger.error(f"{self.name}: Error updating geometries: {e}")
    
    def fit_view(self):
        """Fit camera to scene bounding box"""
        if not self.vis or not self._view_control or not self._geometries:
            return
            
        try:
            bbox = o3d.geometry.AxisAlignedBoundingBox()
            for geom in self._geometries:
                if geom is not None:
                    geom_bbox = geom.get_axis_aligned_bounding_box()
                    if not geom_bbox.is_empty():
                        bbox = bbox.volume_union(geom_bbox)
            
            if not bbox.is_empty():
                center = bbox.get_center()
                extent = bbox.get_max_extent()
                self._camera_params['lookat'] = center
                self._camera_params['zoom'] = max(0.02, min(2.0, 0.02 + (extent / 1000.0)))  # Adjust zoom dynamically
                self._camera_params['front'] = np.array([0.0, 0.0, -1.0])
                self._camera_params['up'] = np.array([0.0, 1.0, 0.0])
                self._apply_camera_params()
                logger.debug(f"{self.name}: View fitted to bbox center {center}, extent {extent}")
        except Exception as e:
            logger.error(f"{self.name}: Error fitting view: {e}")
    
    def _apply_camera_params(self):
        """Apply current camera parameters"""
        if self._view_control:
            vc = self._view_control
            vc.set_lookat(self._camera_params['lookat'])
            vc.set_front(self._camera_params['front'])
            vc.set_up(self._camera_params['up'])
            vc.set_zoom(self._camera_params['zoom'])
    
    def set_camera_params(self, params: Dict[str, Any]):
        """Set camera parameters"""
        self._camera_params.update(params)
        self._apply_camera_params()
    
    def render_frame(self) -> Optional[np.ndarray]:
        """Render current frame"""
        if not self._is_initialized or not self.vis:
            return None
            
        try:
            self.vis.poll_events()
            self.vis.update_renderer()
            img = self.vis.capture_screen_float_buffer(do_render=True)
            arr = (np.asarray(img) * 255).astype(np.uint8)
            return arr
        except Exception as e:
            logger.error(f"{self.name}: Render error: {e}")
            return None
    
    def cleanup(self):
        """Cleanup resources"""
        if self.vis:
            try:
                self.vis.destroy_window()
            except Exception as e:
                logger.error(f"{self.name}: Error destroying window: {e}")
            self.vis = None
        self._is_initialized = False

# -------------------------------
# Scene Management
# -------------------------------
class SceneManager:
    """Manages 3D scene elements"""
    
    def __init__(self):
        self.model: Optional[o3d.geometry.TriangleMesh] = None
        self.ground_plane: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_box(width=100.0, height=100.0, depth=0.01)
        self.ground_plane.paint_uniform_color([0.3, 0.3, 0.3])
        self.ground_plane.translate([-50, -50, -0.01])
        self.show_ground = True
        self.coordinate_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
        event_bus.subscribe('trajectory_updated', self._on_trajectory_updated)
        event_bus.subscribe('selection_updated', self._on_selection_updated)
    
    def _on_trajectory_updated(self, data):
        event_bus.publish('scene_updated')
    
    def _on_selection_updated(self, data):
        event_bus.publish('scene_updated')
    
    def load_model(self, filepath: str) -> bool:
        """Load 3D model"""
        try:
            tm = trimesh.load(filepath)
            self.model = trimesh.exchange.open3d.export_mesh(tm, include_texture=False)
            self.model.compute_vertex_normals()
            if not self.model.has_triangle_normals():
                self.model.compute_triangle_normals()
            if not self.model.has_vertex_colors():
                self.model.paint_uniform_color([0.8, 0.8, 0.8])
            logger.info(f"Open3D mesh loaded successfully from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {filepath}: {e}")
            return False
    
    def toggle_ground_plane(self) -> bool:
        """Toggle ground plane"""
        self.show_ground = not self.show_ground
        return self.show_ground
    
    def get_geometries(self, trajectories: List[Trajectory], selected_trajectory: Optional[Trajectory], selected_waypoint: Optional[int]) -> List[o3d.geometry.Geometry]:
        """Get all scene geometries"""
        geometries = []
        
        if self.model:
            geometries.append(self.model)
        
        if self.show_ground:
            geometries.append(self.ground_plane)
        
        geometries.append(self.coordinate_axes)
        
        for traj in trajectories:
            # Trajectory path
            path = traj.get_interpolated_path()
            if len(path) >= 2:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(path)
                lines = [[i, i+1] for i in range(len(path)-1)]
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.paint_uniform_color(traj.color)
                geometries.append(line_set)
            
            # Directional arrows along path
            if len(path) > 10:
                step = max(1, len(path) // 10)
                for i in range(0, len(path)-1, step):
                    pos = path[i]
                    dir_vec = path[i+1] - path[i]
                    if np.linalg.norm(dir_vec) > 1e-6:
                        dir_vec /= np.linalg.norm(dir_vec)
                        arrow = o3d.geometry.TriangleMesh.create_arrow(
                            cylinder_radius=0.05, cone_radius=0.1, 
                            cylinder_height=0.5, cone_height=0.2
                        )
                        arrow.paint_uniform_color(traj.color)
                        # Rotate arrow to direction
                        z_axis = np.array([0,0,1])
                        rotation_axis = np.cross(z_axis, dir_vec)
                        if np.linalg.norm(rotation_axis) > 1e-6:
                            rotation_axis /= np.linalg.norm(rotation_axis)
                            angle = np.arccos(np.dot(z_axis, dir_vec))
                            rot_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
                            arrow.rotate(rot_mat, center=(0,0,0))
                        arrow.translate(pos)
                        geometries.append(arrow)
            
            # Waypoint markers with orientation
            for wp in traj.waypoints:
                color = [1,0,0] if selected_trajectory == traj and selected_waypoint == wp.index else traj.color
                marker = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=0.1, cone_radius=0.2, 
                    cylinder_height=1.0, cone_height=0.5
                )
                marker.paint_uniform_color(color)
                # Apply yaw and pitch rotation
                yaw_rad = np.deg2rad(wp.yaw)
                pitch_rad = np.deg2rad(wp.pitch)
                rot_yaw = o3d.geometry.get_rotation_matrix_from_axis_angle([0,0,1] * yaw_rad)
                rot_pitch = o3d.geometry.get_rotation_matrix_from_axis_angle([0,1,0] * pitch_rad)
                rot = np.matmul(rot_pitch, rot_yaw)
                marker.rotate(rot, center=(0,0,0))
                marker.translate(wp.position)
                geometries.append(marker)
        
        return geometries

# -------------------------------
# UAV Visualization Application
# -------------------------------
class UAVVisualizationApp:
    """Main UAV Visualization Application"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("UAV Trajectory Visualizer")
        self.root.geometry("1280x800")
        self.root.minsize(800, 600)
        
        self.trajectories: List[Trajectory] = []
        self.selected_trajectory: Optional[Trajectory] = None
        self.selected_waypoint: Optional[int] = None
        self.scene = SceneManager()
        
        self.main_renderer = PersistentRenderer("Main 3D View", 960, 720)
        self.fpv_renderer = PersistentRenderer("FPV View", 400, 300)
        
        self._main_photo: Optional[ImageTk.PhotoImage] = None
        self._fpv_photo: Optional[ImageTk.PhotoImage] = None
        self._mouse_last: Optional[Tuple[int, int]] = None
        self._is_dragging = False
        self.resize_after_id: Optional[str] = None
        
        self.setup_ui()
        self._render_loop()
        
        event_bus.subscribe('scene_updated', self._update_scene_rendering)
        event_bus.subscribe('trajectory_updated', self._update_trajectory_ui)
        
        self.root.after(100, self._initialize_renderers)
    
    def setup_ui(self):
        """Setup user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left: 3D canvas
        self.main_canvas = tk.Canvas(main_frame, bg='black')
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.main_canvas.bind("<Configure>", self._on_canvas_resize)
        self.main_canvas.bind("<Button-1>", self._on_mouse_press)
        self.main_canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.main_canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.main_canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        
        # Right: Controls
        control_pane = ttk.Notebook(main_frame, width=400)
        control_pane.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Trajectory tab
        traj_tab = ttk.Frame(control_pane)
        control_pane.add(traj_tab, text="Trajectories")
        
        ttk.Button(traj_tab, text="Load Model", command=self.load_model).pack(pady=5, fill=tk.X)
        ttk.Button(traj_tab, text="Load Trajectory", command=self.load_trajectory).pack(pady=5, fill=tk.X)
        ttk.Button(traj_tab, text="Save Trajectory", command=self.save_trajectory).pack(pady=5, fill=tk.X)
        ttk.Button(traj_tab, text="Toggle Ground", command=self.toggle_ground_plane).pack(pady=5, fill=tk.X)
        ttk.Button(traj_tab, text="Compare Trajectories", command=self.compare_trajectories).pack(pady=5, fill=tk.X)
        ttk.Button(traj_tab, text="Reset View", command=self._reset_main_view).pack(pady=5, fill=tk.X)
        
        self.interp_method = tk.StringVar(value="spline")
        interp_frame = ttk.Frame(traj_tab)
        interp_frame.pack(pady=5, fill=tk.X)
        ttk.Radiobutton(interp_frame, text="Linear", variable=self.interp_method, value="linear", command=self._on_interpolation_changed).pack(side=tk.LEFT)
        ttk.Radiobutton(interp_frame, text="Spline", variable=self.interp_method, value="spline", command=self._on_interpolation_changed).pack(side=tk.LEFT)
        
        self.waypoint_listbox = tk.Listbox(traj_tab, height=10)
        self.waypoint_listbox.pack(pady=5, fill=tk.X)
        self.waypoint_listbox.bind('<<ListboxSelect>>', self._on_waypoint_select)
        
        self.info_text = tk.Text(traj_tab, height=8, wrap=tk.WORD)
        self.info_text.pack(pady=5, fill=tk.X)
        
        # FPV tab
        fpv_tab = ttk.Frame(control_pane)
        control_pane.add(fpv_tab, text="FPV Preview")
        
        self.fpv_label = tk.Label(fpv_tab, bg='black', text="Select waypoint for FPV")
        self.fpv_label.pack(pady=10, fill=tk.BOTH, expand=True)
    
    def _initialize_renderers(self):
        """Initialize renderers"""
        if self.main_renderer.initialize() and self.fpv_renderer.initialize():
            logger.info("Renderers initialized successfully")
            event_bus.publish('scene_updated')
        else:
            messagebox.showerror("Error", "Failed to initialize renderers")
    
    def _render_loop(self):
        """Main rendering loop (~30 FPS)"""
        main_frame = self.main_renderer.render_frame()
        if main_frame is not None:
            self._display_main_frame(main_frame)
        
        fpv_frame = self.fpv_renderer.render_frame()
        if fpv_frame is not None:
            self._display_fpv_frame(fpv_frame)
        
        self.root.after(33, self._render_loop)
    
    def _update_scene_rendering(self, data=None):
        """Update scene rendering"""
        geometries = self.scene.get_geometries(self.trajectories, self.selected_trajectory, self.selected_waypoint)
        self.main_renderer.update_geometries(geometries)
        self.fpv_renderer.update_geometries(geometries)
    
    def _update_fpv_camera(self):
        """Update FPV camera based on selected waypoint"""
        if not self.selected_trajectory or self.selected_waypoint is None:
            return
            
        wp = self.selected_trajectory.waypoints[self.selected_waypoint]
        pos = wp.position
        yaw_rad = np.deg2rad(wp.yaw)
        pitch_rad = np.deg2rad(wp.pitch)
        front = np.array([
            math.cos(pitch_rad) * math.cos(yaw_rad),
            math.cos(pitch_rad) * math.sin(yaw_rad),
            math.sin(pitch_rad)
        ])
        up = np.array([0, 0, 1])  # Assuming Z up
        params = {
            'lookat': pos + front * 10.0,  # Larger offset
            'front': -front,
            'up': up,
            'zoom': 0.1
        }
        self.fpv_renderer.set_camera_params(params)
    
    def _update_trajectory_ui(self, data=None):
        """Update trajectory UI"""
        self._update_info_display()
        self._update_waypoint_list()
    
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
        """Handle canvas resize with debounce"""
        if self.resize_after_id:
            self.root.after_cancel(self.resize_after_id)
        self.resize_after_id = self.root.after(200, lambda: self.main_renderer.update_viewport(event.width, event.height))
    
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
        event_bus.publish('selection_updated')
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
            angle_str = f"(yaw: {wp.yaw:.1f}°, pitch: {wp.pitch:.1f}°)"
            self.waypoint_listbox.insert(tk.END, f"{i}: {pos_str} {angle_str}")
    
    def _on_mouse_press(self, event):
        self._mouse_last = (event.x, event.y)
        self._is_dragging = True
    
    def _on_mouse_drag(self, event):
        if not self._is_dragging or self._mouse_last is None:
            return
        dx = event.x - self._mouse_last[0]
        dy = event.y - self._mouse_last[1]
        params = self.main_renderer._camera_params
        yaw = math.atan2(params['front'][1], params['front'][0])
        pitch = math.asin(params['front'][2])
        yaw += np.deg2rad(dx * 0.2)
        pitch -= np.deg2rad(dy * 0.2)
        pitch = np.clip(pitch, -math.pi/2 + 1e-6, math.pi/2 - 1e-6)
        front = np.array([
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch)
        ])
        params['front'] = front
        self.main_renderer.set_camera_params(params)
        self._mouse_last = (event.x, event.y)
    
    def _on_mouse_release(self, event):
        self._is_dragging = False
        self._mouse_last = None
    
    def _on_mouse_wheel(self, event):
        delta = -event.delta / 120.0  # Positive forward
        params = self.main_renderer._camera_params
        params['zoom'] = np.clip(params['zoom'] + delta * 0.05, 0.01, 2.0)
        self.main_renderer.set_camera_params(params)
    
    def _reset_main_view(self):
        self.main_renderer.fit_view()
    
    def load_model(self):
        """Load 3D model in background"""
        filepath = filedialog.askopenfilename(
            title="Select 3D Model",
            filetypes=[("3D Models", "*.obj *.ply *.stl"), ("All files", "*.*")]
        )
        if not filepath:
            return
        
        def load_task():
            success = self.scene.load_model(filepath)
            def update_ui():
                if success:
                    messagebox.showinfo("Success", "Model loaded successfully!")
                    event_bus.publish('scene_updated')
                else:
                    messagebox.showerror("Error", "Failed to load model")
            self.root.after(0, update_ui)
        
        threading.Thread(target=load_task, daemon=True).start()
    
    def load_trajectory(self):
        """Load trajectory in background"""
        filepath = filedialog.askopenfilename(
            title="Select Trajectory File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return
        
        def load_task():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                waypoints = []
                for i, wp_data in enumerate(data.get('waypoints', [])):
                    wp = Waypoint.from_dict(wp_data, i)
                    waypoints.append(wp)
                logger.info(f"Parsed {len(waypoints)} waypoints from {filepath}")
                
                def update_ui():
                    if not waypoints:
                        messagebox.showerror("Error", "Failed to parse trajectory file")
                        return
                    name = Path(filepath).stem
                    color = (np.random.rand(), np.random.rand(), np.random.rand())
                    trajectory = Trajectory(name, waypoints, color)
                    self.trajectories.append(trajectory)
                    self.selected_trajectory = trajectory
                    self.selected_waypoint = None
                    event_bus.publish('trajectory_updated')
                    messagebox.showinfo("Success", f"Loaded trajectory with {len(waypoints)} waypoints")
                
                self.root.after(0, update_ui)
            except Exception as e:
                logger.error(f"Error loading trajectory: {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load trajectory: {e}"))
        
        threading.Thread(target=load_task, daemon=True).start()
    
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
            data = {'waypoints': [wp.to_dict() for wp in self.selected_trajectory.waypoints]}
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            messagebox.showinfo("Success", "Trajectory saved successfully!")
        except Exception as e:
            logger.error(f"Error saving trajectory: {e}")
            messagebox.showerror("Error", f"Failed to save trajectory: {e}")
    
    def toggle_ground_plane(self):
        """Toggle ground plane visibility"""
        try:
            is_visible = self.scene.toggle_ground_plane()
            event_bus.publish('scene_updated')
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
            compare_window = tk.Toplevel(self.root)
            compare_window.title("Trajectory Comparison")
            compare_window.geometry("800x600")
            
            text_widget = tk.Text(compare_window, wrap=tk.WORD)
            scrollbar = ttk.Scrollbar(compare_window, orient=tk.VERTICAL, command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            comparison_data = []
            for traj in self.trajectories:
                metrics = traj.calculate_metrics()
                comparison_data.append({
                    'name': traj.name,
                    'metrics': metrics
                })
            
            most_efficient = min(comparison_data, key=lambda x: x['metrics']['total_duration'])
            
            text_widget.insert(tk.END, "TRAJECTORY COMPARISON\n")
            text_widget.insert(tk.END, "=" * 50 + "\n\n")
            
            header = f"{'Name':<20} {'Length(m)':<12} {'Vertical(m)':<12} {'Duration(s)':<12} {'Waypoints':<10}\n"
            text_widget.insert(tk.END, header)
            text_widget.insert(tk.END, "-" * 70 + "\n")
            
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
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.main_renderer.cleanup()
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