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
    """A collapsible frame widget that can show/hide its content"""
    
    def __init__(self, parent, text="", **kwargs):
        super().__init__(parent, **kwargs)
        
        self.show = tk.BooleanVar(value=True)
        self.text = text
        
        self.title_frame = ttk.Frame(self)
        self.title_frame.pack(fill=tk.X, pady=(0, 2))
        
        self.toggle_button = ttk.Button(
            self.title_frame, 
            text=f"▼ {self.text}", 
            command=self.toggle
        )
        self.toggle_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.content_frame = ttk.Frame(self, padding=8)
        self.content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
    
    def toggle(self):
        if self.show.get():
            self.content_frame.pack_forget()
            self.toggle_button.config(text=f"► {self.text}")
            self.show.set(False)
        else:
            self.content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
            self.toggle_button.config(text=f"▼ {self.text}")
            self.show.set(True)
    
    def get_content_frame(self):
        return self.content_frame

# -------------------------------
# Data structures and utilities
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
        x = data.get('x') if 'x' in data else data.get('lon') if 'lon' in data else data.get('longitude')
        y = data.get('y') if 'y' in data else data.get('lat') if 'lat' in data else data.get('latitude')
        z = data.get('z') if 'z' in data else data.get('alt') if 'alt' in data else data.get('altitude')
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

    def get_positions(self) -> np.ndarray:
        return np.array([wp.position for wp in self.waypoints])

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
    
    def calculate_detailed_metrics(self, cruising_speed: float = 10.0) -> Dict:
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
        
        cumulative_turning_angle = 0.0
        if len(positions) >= 3:
            for i in range(1, len(positions) - 1):
                vec1 = positions[i] - positions[i-1]
                vec2 = positions[i+1] - positions[i]
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                if norm1 > 0 and norm2 > 0:
                    cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    cumulative_turning_angle += angle
        
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
        positions = np.array([wp.position for wp in waypoints])
        if len(positions) < 2:
            return positions.copy()
        distances = np.zeros(len(positions))
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[i - 1])
            distances[i] = distances[i - 1] + max(dist, 1e-6)
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
class Open3DRenderThread:
    def __init__(self, width: int, height: int, name: str = "Open3DRender"):
        self.width = width
        self.height = height
        self.name = name
        self.vis = None
        self._geometries = []
        self._camera_params = None
        self._frame_callbacks = []
        
    def initialize(self):
        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name=self.name,
                width=self.width,
                height=self.height,
                visible=False
            )
            logger.info(f"{self.name}: Renderer initialized on main thread")
            return True
        except Exception as e:
            logger.error(f"{self.name}: Failed to initialize renderer: {e}")
            return False
    
    def add_frame_callback(self, callback):
        self._frame_callbacks.append(callback)
    
    def set_geometries(self, geometries):
        self._geometries = geometries
        self._update_geometries()
    
    def set_camera(self, params):
        self._camera_params = params
        self._update_camera()
    
    def _update_geometries(self):
        if self.vis is None:
            return
        try:
            self.vis.clear_geometries()
            for geom in self._geometries:
                if geom is not None:
                    self.vis.add_geometry(geom)
        except Exception as e:
            logger.error(f"{self.name}: Error updating geometries: {e}")
    
    def _update_camera(self):
        if self.vis is None or self._camera_params is None:
            return
        try:
            ctr = self.vis.get_view_control()
            params = self._camera_params
            if 'lookat' in params:
                ctr.set_lookat(params['lookat'])
            if 'front' in params:
                ctr.set_front(params['front'])
            if 'up' in params:
                ctr.set_up(params['up'])
            if 'zoom' in params:
                ctr.set_zoom(float(params['zoom']))
        except Exception as e:
            logger.error(f"{self.name}: Error updating camera: {e}")
    
    def render_frame(self):
        if self.vis is None:
            return None
        try:
            self.vis.poll_events()
            self.vis.update_renderer()
            img = self.vis.capture_screen_float_buffer(do_render=True)
            if img is not None:
                arr = (np.asarray(img) * 255).astype(np.uint8)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                for callback in self._frame_callbacks:
                    try:
                        callback(arr)
                    except Exception as e:
                        logger.error(f"Frame callback error: {e}")
                return arr
        except Exception as e:
            logger.error(f"{self.name}: Error rendering frame: {e}")
        return None
    
    def cleanup(self):
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis = None
# -------------------------------
# Render Manager
# -------------------------------
class RenderManager:
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
        self._update_running = False
        self.main_renderer = None
        self.fpv_renderer = None
    
    def start(self):
        main_width = self.app.main_canvas.winfo_width() or 960
        main_height = self.app.main_canvas.winfo_height() or 720
        fpv_width = self.app.fpv_label.winfo_width() or 400
        fpv_height = self.app.fpv_label.winfo_height() or 300
        
        self.main_renderer = Open3DRenderThread(main_width, main_height, "MainOpen3D")
        self.fpv_renderer = Open3DRenderThread(fpv_width, fpv_height, "FPVOpen3D")
        
        if not self.main_renderer.initialize():
            logger.error("Failed to initialize main renderer")
            return
        if not self.fpv_renderer.initialize():
            logger.error("Failed to initialize FPV renderer")
            return
            
        self.main_renderer.add_frame_callback(self.app._display_main_frame)
        self.fpv_renderer.add_frame_callback(self.app._display_fpv_frame)
        
        self._update_running = True
        self._schedule_update()
    
    def _schedule_update(self):
        if self._update_running:
            self.app.root.after(33, self._update)  # ~30 FPS
    
    def _update(self):
        try:
            if self.main_renderer:
                frame = self.main_renderer.render_frame()
                if frame is not None and self.app.video_recorder and self.app.video_recorder.is_recording:
                    self.app.video_recorder.add_frame(frame)
            
            if self.fpv_renderer:
                self.fpv_renderer.render_frame()
                
        except Exception as e:
            logger.error(f"Error in update loop: {e}")
        
        self._schedule_update()
    
    def update_main_geometries(self, geometries):
        if self.main_renderer:
            self.main_renderer.set_geometries(geometries)
    
    def update_fpv_geometries(self, geometries):
        if self.fpv_renderer:
            self.fpv_renderer.set_geometries(geometries)
    
    def update_main_camera(self, params):
        if self.main_renderer:
            self.main_renderer.set_camera(params)
    
    def update_fpv_camera(self, params):
        if self.fpv_renderer:
            self.fpv_renderer.set_camera(params)

    def resize_main(self, width, height):
        if self.main_renderer:
            # For the main thread renderer, we need to recreate the window
            try:
                self.main_renderer.cleanup()
                self.main_renderer = Open3DRenderThread(width, height, "MainOpen3D")
                if self.main_renderer.initialize():
                    self.main_renderer.add_frame_callback(self.app._display_main_frame)
                    # Restore geometries and camera if available
                    if hasattr(self.app, 'scene') and self.app.scene.o3d_geometries:
                        self.main_renderer.set_geometries(self.app.scene.o3d_geometries)
            except Exception as e:
                logger.error(f"Error resizing main renderer: {e}")

    def resize_fpv(self, width, height):
        if self.fpv_renderer:
            # For the FPV renderer, recreate the window
            try:
                self.fpv_renderer.cleanup()
                self.fpv_renderer = Open3DRenderThread(width, height, "FPVOpen3D")
                if self.fpv_renderer.initialize():
                    self.fpv_renderer.add_frame_callback(self.app._display_fpv_frame)
                    # Restore geometries if available
                    if hasattr(self.app, 'scene') and self.app.scene.o3d_geometries:
                        self.fpv_renderer.set_geometries(self.app.scene.o3d_geometries)
            except Exception as e:
                logger.error(f"Error resizing FPV renderer: {e}")
    
    def stop(self):
        self._update_running = False
        if self.main_renderer:
            self.main_renderer.cleanup()
        if self.fpv_renderer:
            self.fpv_renderer.cleanup()
# -------------------------------
# Scene3D
# -------------------------------
class Scene3D:
    def __init__(self):
        self.model_trimesh: Optional[trimesh.Trimesh] = None
        self.bounds: Optional[np.ndarray] = None
        self.o3d_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.o3d_geometries: List[o3d.geometry.Geometry] = []
        self.ground_plane: Optional[o3d.geometry.TriangleMesh] = None
        self.show_ground_plane = True

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

            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(size * 0.1, 1.0))
            self.ground_plane = o3d.geometry.TriangleMesh.create_box(width=size, height=0.01, depth=size)
            self.ground_plane.translate([center[0] - size/2.0, center[1] - 0.005, center[2] - size/2.0])
            self.ground_plane.compute_vertex_normals()
            self.ground_plane.paint_uniform_color([0.5, 0.5, 0.5])

            self.o3d_geometries = [self.o3d_mesh, axes]
            if self.show_ground_plane:
                self.o3d_geometries.append(self.ground_plane)
                
            logger.info("Scene3D configured (Open3D geometries ready).")
            return True
        except Exception as e:
            logger.error(f"Scene3D.load_model failed: {e}")
            return False

    def toggle_ground_plane(self):
        self.show_ground_plane = not self.show_ground_plane
        if self.show_ground_plane and self.ground_plane is not None:
            if self.ground_plane not in self.o3d_geometries:
                self.o3d_geometries.append(self.ground_plane)
        else:
            if self.ground_plane in self.o3d_geometries:
                self.o3d_geometries.remove(self.ground_plane)
        return self.show_ground_plane

    def get_trimesh_components(self) -> List[trimesh.Trimesh]:
        comps = []
        if self.model_trimesh is not None:
            comps.append(self.model_trimesh)
        return comps

# -------------------------------
# Video Recorder
# -------------------------------
class VideoRecorder:
    def __init__(self, output_path: str, fps: int = 30, resolution: Tuple[int, int] = (1920, 1080)):
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.writer = None
        self.is_recording = False

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
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("UAV Flight Trajectory Visualization (Open3D Embedded)")
        self.root.geometry("1400x900")

        self.scene = Scene3D()
        self.camera = Camera3D()
        self.trajectories: List[Trajectory] = []
        self.selected_trajectory: Optional[Trajectory] = None
        self.selected_waypoint: Optional[int] = None

        self.video_recorder: Optional[VideoRecorder] = None

        self.camera_x = tk.DoubleVar(value=float(self.camera.position[0]))
        self.camera_y = tk.DoubleVar(value=float(self.camera.position[1]))
        self.camera_z = tk.DoubleVar(value=float(self.camera.position[2]))
        self.camera_yaw = tk.DoubleVar(value=float(self.camera.yaw))
        self.camera_pitch = tk.DoubleVar(value=float(self.camera.pitch))

        self._mouse_last = None
        self._is_dragging = False

        self.setup_gui()
        self.update_loop_manager = ImprovedUpdateLoop(self)
        self._main_photo = None
        self._fpv_photo = None
        self.root.after(100, self.update_loop_manager.start)

    def setup_gui(self):
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

        camera_frame = CollapsibleFrame(parent, text="Camera Controls")
        camera_frame.pack(fill=tk.X, padx=5, pady=2)
        camera_content = camera_frame.get_content_frame()
        
        self.camera_button = ttk.Button(camera_content, text="Open Camera Controls", command=self.open_camera_window)
        self.camera_button.pack(fill=tk.X, pady=2)

        traj_frame = CollapsibleFrame(parent, text="Trajectory Controls")
        traj_frame.pack(fill=tk.X, padx=5, pady=2)
        traj_content = traj_frame.get_content_frame()
        
        ttk.Button(traj_content, text="Compare Trajectories", command=self.compare_trajectories).pack(fill=tk.X, pady=2)
        ttk.Button(traj_content, text="Compare Interpolations", command=self.compare_interpolations).pack(fill=tk.X, pady=2)
        ttk.Label(traj_content, text="Interpolation Method:").pack()
        self.interp_method = tk.StringVar(value="spline")
        ttk.Radiobutton(traj_content, text="Linear", variable=self.interp_method, value="linear", command=self.update_interpolation).pack()
        ttk.Radiobutton(traj_content, text="Spline", variable=self.interp_method, value="spline", command=self.update_interpolation).pack()

        ground_frame = CollapsibleFrame(parent, text="Ground Plane")
        ground_frame.pack(fill=tk.X, padx=5, pady=2)
        ground_content = ground_frame.get_content_frame()
        
        ttk.Button(ground_content, text="Toggle Ground Plane", command=self.toggle_ground_plane).pack(fill=tk.X, pady=2)

        record_frame = CollapsibleFrame(parent, text="Recording")
        record_frame.pack(fill=tk.X, padx=5, pady=2)
        record_content = record_frame.get_content_frame()
        
        self.record_button = ttk.Button(record_content, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(fill=tk.X, pady=2)

        edit_frame = CollapsibleFrame(parent, text="Waypoint Editing")
        edit_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        edit_content = edit_frame.get_content_frame()
        
        ttk.Button(edit_content, text="Add Waypoint", command=self.add_waypoint).pack(fill=tk.X, pady=2)
        ttk.Button(edit_content, text="Edit Waypoint", command=self.edit_waypoint).pack(fill=tk.X, pady=2)
        ttk.Button(edit_content, text="Delete Waypoint", command=self.delete_waypoint).pack(fill=tk.X, pady=2)
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

    def load_model(self):
        filepath = filedialog.askopenfilename(title="Select 3D Model",
                                              filetypes=[("3D Models", "*.obj *.ply *.stl"), ("All Files", "*.*")])
        if not filepath:
            return
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

    def update_trajectory_visualization(self):
        main_geoms = self.scene.o3d_geometries.copy()
        fpv_geoms = self.scene.o3d_geometries.copy()
        
        for traj in self.trajectories:
            points = traj.get_positions()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color(traj.color)
            
            for i, wp in enumerate(traj.waypoints):
                camera_geom = self.generate_camera_visualization(
                    wp.position, wp.yaw, wp.pitch, 
                    scale=0.2, color=traj.color
                )
                main_geoms.append(camera_geom)
            
            if traj.interpolation_method == InterpolationMethod.LINEAR:
                path = TrajectoryInterpolator.interpolate_linear(traj.waypoints, num_points=200)
                lines = [[i, i+1] for i in range(len(path)-1)]
                lineset = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(path),
                    lines=o3d.utility.Vector2iVector(lines)
                )
                lineset.paint_uniform_color(traj.color)
                main_geoms.extend([pcd, lineset])
            else:
                linear_path = TrajectoryInterpolator.interpolate_linear(traj.waypoints, num_points=200)
                spline_path = TrajectoryInterpolator.interpolate_spline(traj.waypoints, num_points=200)
                linear_lines = [[i, i+1] for i in range(len(linear_path)-1)]
                linear_lineset = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(linear_path),
                    lines=o3d.utility.Vector2iVector(linear_lines)
                )
                linear_lineset.paint_uniform_color([0.7, 0.7, 0.7])
                spline_lines = [[i, i+1] for i in range(len(spline_path)-1)]
                spline_lineset = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(spline_path),
                    lines=o3d.utility.Vector2iVector(spline_lines)
                )
                spline_lineset.paint_uniform_color(traj.color)
                main_geoms.extend([pcd, linear_lineset, spline_lineset])
        
        self.update_loop_manager.update_main_geometries(main_geoms)
        self.update_loop_manager.update_fpv_geometries(fpv_geoms)

    def generate_camera_visualization(self, position, yaw, pitch, scale=0.5, color=None):
        if color is None:
            color = [1.0, 0.0, 0.0]
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        points = [
            [0, 0, 0],
            [-scale, -scale, scale*2],
            [scale, -scale, scale*2],
            [scale, scale, scale*2],
            [-scale, scale, scale*2]
        ]
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
        rotated_points = []
        for point in points:
            rotated = np.dot(rot_pitch, np.dot(rot_yaw, point))
            rotated_points.append(rotated + position)
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [2, 3], [3, 4], [4, 1]
        ]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(rotated_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color(color)
        return line_set

    def toggle_ground_plane(self):
        is_visible = self.scene.toggle_ground_plane()
        self.update_trajectory_visualization()
        messagebox.showinfo("Ground Plane", f"Ground plane is now {'visible' if is_visible else 'hidden'}")

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
        if self.selected_trajectory and self.selected_waypoint is not None:
            wp = self.selected_trajectory.waypoints[self.selected_waypoint]
            yaw, pitch = np.radians(wp.yaw), np.radians(wp.pitch)
            forward = np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
            if np.linalg.norm(forward) == 0:
                forward = np.array([0.0, 0.0, -1.0])
            forward /= np.linalg.norm(forward)
            params = {
                'front': forward.tolist(),
                'up': [0.0, 1.0, 0.0],
                'lookat': (wp.position + forward).tolist(),
                'zoom': 0.5
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

    def on_waypoint_select(self, event):
        sel = event.widget.curselection()
        if not sel:
            return
        idx = sel[0]
        if not self.selected_trajectory:
            return
        if 0 <= idx < len(self.selected_trajectory.waypoints):
            self.selected_waypoint = idx
            self._update_fpv_camera()

    def update_waypoint_list(self):
        self.waypoint_listbox.delete(0, tk.END)
        if not self.selected_trajectory:
            return
        for wp in self.selected_trajectory.waypoints:
            self.waypoint_listbox.insert(tk.END, f"{wp.index}: {wp.position.tolist()} (yaw={wp.yaw:.1f})")

    def compare_trajectories(self):
        if len(self.trajectories) < 2:
            messagebox.showwarning("Warning", "Need at least 2 trajectories to compare")
            return
        output = []
        for traj in self.trajectories:
            metrics = traj.calculate_detailed_metrics()
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

    def toggle_recording(self):
        if self.video_recorder and self.video_recorder.is_recording:
            self.video_recorder.stop_recording()
            self.video_recorder = None
            self.record_button.config(text="Start Recording")
            messagebox.showinfo("Recording", "Stopped and saved recording.")
        else:
            filepath = filedialog.asksaveasfilename(title="Save Recording", defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
            if not filepath:
                return
            res = (self.main_canvas.winfo_width() or 960, self.main_canvas.winfo_height() or 720)
            self.video_recorder = VideoRecorder(filepath, fps=20, resolution=res)
            self.video_recorder.start_recording()
            self.record_button.config(text="Stop Recording")
            messagebox.showinfo("Recording", "Recording started...")

    def on_canvas_resize(self, event):
        self.update_loop_manager.resize_main(event.width, event.height)

    def on_fpv_resize(self, event):
        self.update_loop_manager.resize_fpv(event.width, event.height)

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
        self._mouse_last = (event.x, event.y)

    def on_mouse_release(self, event):
        self._is_dragging = False
        self._mouse_last = None

    def on_mouse_wheel(self, event):
        delta = event.delta / 120
        self.camera.position[2] += -delta * 0.5
        self.camera_z.set(self.camera.position[2])
        self._update_main_camera()

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
        m = self.selected_trajectory.calculate_detailed_metrics()
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