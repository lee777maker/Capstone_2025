# main.py

import sys
import os
import time
import json
import queue
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from enum import Enum
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
                # This is the format with separate positions and rotations arrays
                positions = np.array(data['positions'])
                rotations = np.array(data['rotations'])
                num_cameras = len(positions)
                
                # Check if there's a specific order for waypoints
                if 'waypoint_order' in data:
                    order = data['waypoint_order']
                else:
                    order = list(range(num_cameras))
                
                for i, idx in enumerate(order):
                    if idx < num_cameras:
                        pos = positions[idx]
                        yaw = rotations[idx, 0] if rotations.shape[1] > 0 else 0.0
                        pitch = rotations[idx, 1] if rotations.shape[1] > 1 else 0.0
                        wp = Waypoint(position=pos, yaw=yaw, pitch=pitch, index=i)
                        waypoints.append(wp)
            
            # Check for the camera objects format
            elif 'cameras' in data and isinstance(data['cameras'], list):
                cameras = data['cameras']
                order = data.get('waypoint_order', list(range(len(cameras))))
                for i, idx in enumerate(order):
                    cam = cameras[idx]
                    wp = Waypoint.from_dict(cam, i)
                    waypoints.append(wp)
            
            # Check for the waypoints format
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
            
            # Check if the file is just an array of waypoints
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
            # Save in the format with positions and rotations arrays
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
        
        # Calculate cumulative distances
        distances = np.zeros(len(positions))
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[i - 1])
            distances[i] = distances[i - 1] + max(dist, 1e-6)  # Ensure no zero distances
        
        # Handle case where all points are the same
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
        
        # Calculate cumulative distances
        distances = np.zeros(len(positions))
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[i - 1])
            distances[i] = distances[i - 1] + max(dist, 1e-6)  # Ensure no zero distances
        
        # Handle case where all points are the same
        if distances[-1] == 0:
            return np.tile(positions[0], (num_points, 1))
        
        # For small numbers of points, use linear interpolation
        if len(positions) < 4:
            return TrajectoryInterpolator.interpolate_linear(waypoints, num_points)
        
        # Ensure distances are strictly increasing
        if not np.all(np.diff(distances) > 0):
            # If distances are not strictly increasing, use linear interpolation
            return TrajectoryInterpolator.interpolate_linear(waypoints, num_points)
        
        try:
            cs_x = CubicSpline(distances, positions[:, 0])
            cs_y = CubicSpline(distances, positions[:, 1])
            cs_z = CubicSpline(distances, positions[:, 2])
            t = np.linspace(0, distances[-1], num_points)
            return np.column_stack([cs_x(t), cs_y(t), cs_z(t)])
        except Exception:
            # Fall back to linear interpolation if spline fails
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
class Open3DRenderThread(threading.Thread):
    def __init__(self, cmd_queue: queue.Queue, result_queue: queue.Queue, width: int, height: int, name: str = "Open3DRender"):
        super().__init__(daemon=True)
        self.cmd_queue = cmd_queue
        self.result_queue = result_queue
        self.width = width
        self.height = height
        self._should_stop = False
        self.vis = None
        self.added = False
        self.name = name
        self._pending_geometries = []
        self._camera_params = None

    def run(self):
        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name=self.name, width=self.width, height=self.height, visible=False)
            # Add advanced lighting
            render_option = self.vis.get_render_option()
            render_option.light_on = True
            render_option.point_size = 5.0
            render_option.line_width = 2.0
            self.added = False
            logger.info(f"{self.name}: Open3D visualizer created with advanced lighting")
        except Exception as e:
            logger.error(f"{self.name}: Failed to create Open3D visualizer: {e}")
            self.vis = None

        while not self._should_stop:
            try:
                cmd = self.cmd_queue.get(timeout=0.05)
            except queue.Empty:
                cmd = None

            if cmd:
                try:
                    c = cmd.get('cmd')
                    if c == 'stop':
                        self._should_stop = True
                    elif c == 'clear_geometries' and self.vis is not None:
                        self.vis.clear_geometries()
                        self.added = False
                    elif c == 'add_geometry' and self.vis is not None:
                        geom = cmd.get('geometry')
                        if geom is not None:
                            self.vis.add_geometry(geom)
                            self.added = True
                    elif c == 'set_geometries' and self.vis is not None:
                        self.vis.clear_geometries()
                        for g in cmd.get('geometries', []):
                            self.vis.add_geometry(g)
                        self.added = True
                    elif c == 'set_camera' and self.vis is not None:
                        params = cmd.get('params', {})
                        self._camera_params = params
                        ctr = self.vis.get_view_control()
                        if 'lookat' in params:
                            ctr.set_lookat(params['lookat'])
                        if 'front' in params:
                            ctr.set_front(params['front'])
                        if 'up' in params:
                            ctr.set_up(params['up'])
                        if 'zoom' in params:
                            try:
                                ctr.set_zoom(float(params['zoom']))
                            except Exception:
                                pass
                    elif c == 'resize' and self.vis is not None:
                        self.width = cmd.get('width', self.width)
                        self.height = cmd.get('height', self.height)
                        self.vis.destroy_window()
                        self.vis = o3d.visualization.Visualizer()
                        self.vis.create_window(window_name=self.name, width=self.width, height=self.height, visible=False)
                        # Re-add geometries
                        for g in self._pending_geometries:
                            self.vis.add_geometry(g)
                        self.added = len(self._pending_geometries) > 0
                        # Re-apply camera
                        if self._camera_params:
                            ctr = self.vis.get_view_control()
                            if 'lookat' in self._camera_params:
                                ctr.set_lookat(self._camera_params['lookat'])
                            if 'front' in self._camera_params:
                                ctr.set_front(self._camera_params['front'])
                            if 'up' in self._camera_params:
                                ctr.set_up(self._camera_params['up'])
                            if 'zoom' in self._camera_params:
                                try:
                                    ctr.set_zoom(float(self._camera_params['zoom']))
                                except Exception:
                                    pass
                    elif c == 'render' and self.vis is not None:
                        self.vis.poll_events()
                        self.vis.update_renderer()
                        img = self.vis.capture_screen_float_buffer(do_render=True)
                        arr = (np.asarray(img) * 255).astype(np.uint8)
                        if arr.ndim == 2:
                            arr = np.stack([arr]*3, axis=-1)
                        self.result_queue.put({'type': 'frame', 'image': arr})
                except Exception as e:
                    logger.error(f"{self.name}: Error processing command {cmd}: {e}")

        try:
            if self.vis is not None:
                self.vis.destroy_window()
        except Exception:
            pass
        logger.info(f"{self.name}: stopped")

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
            # Try loading with trimesh for bounds
            try:
                loaded = trimesh.load(filepath)
                if isinstance(loaded, trimesh.Scene):
                    loaded = loaded.dump(concatenate=True)
                self.model_trimesh = loaded
                self.bounds = self.model_trimesh.bounds
                logger.info(f"Trimesh model loaded successfully: {filepath}")
            except Exception as e:
                logger.warning(f"Trimesh failed: {e}. Continuing with Open3D only.")

            # Load with Open3D, enabling texture support
            try:
                mesh_o3d = o3d.io.read_triangle_mesh(filepath, enable_post_processing=True, print_progress=True)
                if mesh_o3d is None or len(mesh_o3d.vertices) == 0:
                    raise RuntimeError("Open3D returned empty mesh")
                if not mesh_o3d.has_vertex_normals():
                    mesh_o3d.compute_vertex_normals()
                # Ensure texture is loaded if available
                if mesh_o3d.has_triangle_uvs() and mesh_o3d.has_textures():
                    logger.info("Texture data loaded successfully")
                else:
                    mesh_o3d.paint_uniform_color([0.7, 0.7, 0.7])  # Fallback color
                self.o3d_mesh = mesh_o3d
                logger.info("Open3D mesh loaded with texture support.")
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
                
            logger.info("Scene3D configured (Open3D geometries with textures ready).")
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
        self.trajectory_info = None  # Store trajectory data for overlay
        self.current_position = None  # Current position for overlay

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
        
        # Add overlays
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self.trajectory_info:
            text = f"Trajectory: {self.trajectory_info['name']}\n"
            text += f"Length: {self.trajectory_info['metrics']['total_length']:.2f} m"
            y0, dy = 30, 30
            for i, line in enumerate(text.split('\n')):
                cv2.putText(frame_bgr, line, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if self.current_position is not None:
            pos_text = f"Pos: [{self.current_position[0]:.2f}, {self.current_position[1]:.2f}, {self.current_position[2]:.2f}]"
            cv2.putText(frame_bgr, pos_text, (10, frame_bgr.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        self.writer.write(frame_bgr)

    def stop_recording(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.is_recording = False
        logger.info(f"VideoRecorder: saved {self.output_path}")

    def set_trajectory_info(self, traj_name: str, metrics: Dict):
        self.trajectory_info = {'name': traj_name, 'metrics': metrics}
    
    def set_current_position(self, position: np.ndarray):
        self.current_position = position

# -------------------------------
# Main Application
# -------------------------------
class UAVVisualizationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("UAV Flight Trajectory Visualization (Open3D Embedded)")
        self.root.geometry("1400x900")
        
        # Core data
        self.scene = Scene3D()
        self.camera = Camera3D()
        self.trajectories: List[Trajectory] = []
        self.selected_trajectory: Optional[Trajectory] = None
        self.selected_waypoint: Optional[int] = None
        self.marker_pool = []
        self.frustum_pool = []
        self.command_history = []
        self.undo_history = []
        self.edit_window = None
        self.layout_config = None
        self.paned_positions = {'left': 300, 'right': 300}
        self.progress_bar = None
        
        # Rendering threads and queues
        self.main_cmd_q = queue.Queue()
        self.main_res_q = queue.Queue()
        self.fpv_cmd_q = queue.Queue()
        self.fpv_res_q = queue.Queue()
        self.main_renderer = None
        self.fpv_renderer = None
        
        # Recording
        self.video_recorder: Optional[VideoRecorder] = None
        
        # Camera control variables
        self.camera_x = tk.DoubleVar(value=float(self.camera.position[0]))
        self.camera_y = tk.DoubleVar(value=float(self.camera.position[1]))
        self.camera_z = tk.DoubleVar(value=float(self.camera.position[2]))
        self.camera_yaw = tk.DoubleVar(value=float(self.camera.yaw))
        self.camera_pitch = tk.DoubleVar(value=float(self.camera.pitch))
        
        # Mouse control state
        self._mouse_last = None
        self._is_dragging = False
        
        # UI components
        self.setup_gui()
        
        # Initialize pools for waypoint optimization
        self.initialize_pools(max_markers=1000)
        
        # Initialize renderers
        self.main_renderer = Open3DRenderThread(self.main_cmd_q, self.main_res_q,
                                               width=self.main_canvas.winfo_width() or 960,
                                               height=self.main_canvas.winfo_height() or 720,
                                               name="MainOpen3D")
        self.main_renderer.start()
        self.fpv_renderer = Open3DRenderThread(self.fpv_cmd_q, self.fpv_res_q,
                                              width=self.fpv_label.winfo_width() or 400,
                                              height=self.fpv_label.winfo_height() or 300,
                                              name="FPVOpen3D")
        self.fpv_renderer.start()
        
        # Start update loop
        self._update_loop_running = True
        self._last_main_frame = None
        self._last_fpv_frame = None
        self.update_loop()


    def initialize_pools(self, max_markers: int = 1000):
        for _ in range(max_markers):
            pcd = o3d.geometry.PointCloud()
            self.marker_pool.append(pcd)
            frustum = o3d.geometry.LineSet()
            self.frustum_pool.append(frustum)

    def add_command(self, command: Dict):
        self.command_history.append(command)
        self.undo_history.clear()

    def undo(self):
        if not self.command_history:
            return
        cmd = self.command_history.pop()
        self.undo_history.append(cmd)
        if cmd['type'] == 'add':
            self.selected_trajectory.waypoints.pop(cmd['index'])
            for i, wp in enumerate(self.selected_trajectory.waypoints):
                wp.index = i
        elif cmd['type'] == 'edit':
            wp = self.selected_trajectory.waypoints[cmd['index']]
            wp.position = cmd['old_position']
            wp.yaw, wp.pitch = cmd['old_yaw'], cmd['old_pitch']
        elif cmd['type'] == 'delete':
            wp = Waypoint(position=cmd['position'], yaw=cmd['yaw'], pitch=cmd['pitch'], index=cmd['index'])
            self.selected_trajectory.waypoints.insert(cmd['index'], wp)
            for i, wp in enumerate(self.selected_trajectory.waypoints):
                wp.index = i
        self.update_waypoint_list()
        self.update_info()
        self.update_trajectory_visualization()
        self._update_fpv_camera()

    def redo(self):
        if not self.undo_history:
            return
        cmd = self.undo_history.pop()
        self.command_history.append(cmd)
        if cmd['type'] == 'add':
            wp = Waypoint(position=cmd['position'], yaw=cmd['yaw'], pitch=cmd['pitch'], index=cmd['index'])
            self.selected_trajectory.waypoints.append(wp)
        elif cmd['type'] == 'edit':
            wp = self.selected_trajectory.waypoints[cmd['index']]
            wp.position = cmd['new_position']
            wp.yaw, wp.pitch = cmd['new_yaw'], cmd['new_pitch']
        elif cmd['type'] == 'delete':
            self.selected_trajectory.waypoints.pop(cmd['index'])
            for i, wp in enumerate(self.selected_trajectory.waypoints):
                wp.index = i
        self.update_waypoint_list()
        self.update_info()
        self.update_trajectory_visualization()
        self._update_fpv_camera()

    def setup_gui(self):
        # Add preset buttons frame
        preset_frame = ttk.Frame(self.root)
        preset_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(preset_frame, text="Editing Mode", command=lambda: self.set_layout('editing')).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="Comparison Mode", command=lambda: self.set_layout('comparison')).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="Presentation Mode", command=lambda: self.set_layout('presentation')).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="Save Layout", command=self.save_layout).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="Load Layout", command=self.load_layout).pack(side=tk.LEFT, padx=5)
        
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)
        
        left_frame = ttk.Frame(self.main_paned, width=self.paned_positions['left'])
        self.main_paned.add(left_frame, weight=0)
        middle_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(middle_frame, weight=3)
        right_frame = ttk.Frame(self.main_paned, width=self.paned_positions['right'])
        self.main_paned.add(right_frame, weight=0)
        
        self.progress_frame = ttk.Frame(self.root)
        self.progress_frame.pack(fill=tk.X, padx=5, pady=5)
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=2, side=tk.LEFT)
        self.progress_bar.pack_forget()
        
        self.setup_control_panel(left_frame)
        self.setup_3d_view(middle_frame)
        self.setup_info_panel(right_frame)

    def setup_control_panel(self, parent):
        file_frame = ttk.LabelFrame(parent, text="File Operations", padding=8)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(file_frame, text="Load 3D Model", command=self.load_model).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Load Trajectory", command=self.load_trajectory).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Save Trajectory", command=self.save_trajectory).pack(fill=tk.X, pady=2)

        camera_button_frame = ttk.LabelFrame(parent, text="Camera", padding=8)
        camera_button_frame.pack(fill=tk.X, padx=5, pady=5)
        self.camera_button = ttk.Button(camera_button_frame, text="Open Camera Controls", command=self.open_camera_window)
        self.camera_button.pack(fill=tk.X, pady=2)

        traj_frame = ttk.LabelFrame(parent, text="Trajectory Controls", padding=8)
        traj_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(traj_frame, text="Compare Trajectories", command=self.compare_trajectories).pack(fill=tk.X, pady=2)
        ttk.Button(traj_frame, text="Compare Interpolations", command=self.compare_interpolations).pack(fill=tk.X, pady=2)
        ttk.Label(traj_frame, text="Interpolation Method:").pack()
        self.interp_method = tk.StringVar(value="spline")
        ttk.Radiobutton(traj_frame, text="Linear", variable=self.interp_method, value="linear", command=self.update_interpolation).pack()
        ttk.Radiobutton(traj_frame, text="Spline", variable=self.interp_method, value="spline", command=self.update_interpolation).pack()
        ttk.Button(traj_frame, text="Toggle Ground Plane", command=self.toggle_ground_plane).pack(fill=tk.X, pady=2)

        record_frame = ttk.LabelFrame(parent, text="Recording", padding=8)
        record_frame.pack(fill=tk.X, padx=5, pady=5)
        self.record_button = ttk.Button(record_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(fill=tk.X, pady=2)

        edit_frame = ttk.LabelFrame(parent, text="Waypoint Editing", padding=8)
        edit_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        ttk.Button(edit_frame, text="Add Waypoint", command=self.add_waypoint).pack(fill=tk.X, pady=2)
        ttk.Button(edit_frame, text="Edit Waypoint", command=self.edit_waypoint).pack(fill=tk.X, pady=2)
        ttk.Button(edit_frame, text="Delete Waypoint", command=self.delete_waypoint).pack(fill=tk.X, pady=2)
        ttk.Button(edit_frame, text="Undo", command=self.undo).pack(fill=tk.X, pady=2)
        ttk.Button(edit_frame, text="Redo", command=self.redo).pack(fill=tk.X, pady=2)
        self.waypoint_listbox = tk.Listbox(edit_frame, height=10)
        self.waypoint_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.waypoint_listbox.bind('<<ListboxSelect>>', self.on_waypoint_select)

    def update_waypoint_list(self):
        """Update the waypoint listbox with the current trajectory's waypoints."""
        self.waypoint_listbox.delete(0, tk.END)
        if not self.selected_trajectory:
            return
        for wp in self.selected_trajectory.waypoints:
            self.waypoint_listbox.insert(tk.END, f"{wp.index}: {wp.position.tolist()} (yaw={wp.yaw:.1f})")
        if self.selected_waypoint is not None and 0 <= self.selected_waypoint < len(self.selected_trajectory.waypoints):
            self.waypoint_listbox.selection_clear(0, tk.END)
            self.waypoint_listbox.selection_set(self.selected_waypoint)
            self.waypoint_listbox.see(self.selected_waypoint)

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
        self.main_canvas.bind("<Button-3>", self.on_right_click)

    def setup_info_panel(self, parent):
        info_frame = ttk.LabelFrame(parent, text="Trajectory Information", padding=8)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        self.info_text = tk.Text(info_frame, height=10, width=30)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        fpv_frame = ttk.LabelFrame(parent, text="First Person View", padding=8)
        fpv_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.fpv_label = ttk.Label(fpv_frame)
        self.fpv_label.pack(fill=tk.BOTH, expand=True)
        self.fpv_label.bind("<Configure>", self.on_fpv_resize)
        self._fpv_photo = None
        comp_frame = ttk.LabelFrame(parent, text="Comparison Results", padding=8)
        comp_frame.pack(fill=tk.X, padx=5, pady=5)
        self.comp_text = tk.Text(comp_frame, height=8, width=30)
        self.comp_text.pack(fill=tk.BOTH, expand=True)

    def set_layout(self, mode: str):
        if mode == 'editing':
            self.paned_positions = {'left': 400, 'right': 200}
        elif mode == 'comparison':
            self.paned_positions = {'left': 200, 'right': 400}
        elif mode == 'presentation':
            self.paned_positions = {'left': 100, 'right': 100}
        self.main_paned.sashpos(0, self.paned_positions['left'])
        self.main_paned.sashpos(1, self.root.winfo_width() - self.paned_positions['right'])

    def save_layout(self):
        filepath = filedialog.asksaveasfilename(title="Save Layout", defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if not filepath:
            return
        with open(filepath, 'w') as f:
            json.dump(self.paned_positions, f)
        messagebox.showinfo("Success", "Layout saved successfully!")

    def load_layout(self):
        filepath = filedialog.askopenfilename(title="Load Layout", filetypes=[("JSON Files", "*.json")])
        if not filepath:
            return
        with open(filepath, 'r') as f:
            self.paned_positions = json.load(f)
        self.main_paned.sashpos(0, self.paned_positions['left'])
        self.main_paned.sashpos(1, self.root.winfo_width() - self.paned_positions['right'])
        messagebox.showinfo("Success", "Layout loaded successfully!")

    def load_model(self):
        filepath = filedialog.askopenfilename(title="Select 3D Model",
                                              filetypes=[("3D Models", "*.obj *.ply *.stl"), ("All Files", "*.*")])
        if not filepath:
            return
        self.progress_bar.pack()
        self.progress_bar.start()
        self.root.update()

        def load_task():
            success = self.scene.load_model(filepath)
            self.root.after(0, lambda: self._finish_load_model(success, filepath))

        threading.Thread(target=load_task, daemon=True).start()

    def _finish_load_model(self, success: bool, filepath: str):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        if not success:
            messagebox.showerror("Error", f"Failed to load model: Could not process {filepath}")
            return
        self.update_trajectory_visualization()
        messagebox.showinfo("Success", "Model loaded successfully.")

    def load_trajectory(self):
        filepath = filedialog.askopenfilename(title="Select Trajectory File",
                                              filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if not filepath:
            return
        self.progress_bar.pack()
        self.progress_bar.start()
        self.root.update()

        def parse_task():
            waypoints = TrajectoryParser.parse_json(filepath)
            self.root.after(0, lambda: self._finish_load_trajectory(waypoints, filepath))

        threading.Thread(target=parse_task, daemon=True).start()

    def _finish_load_trajectory(self, waypoints: List[Waypoint], filepath: str):
        """Handle completion of trajectory loading with UI updates."""
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
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
        self.add_command({
            'type': 'add_trajectory',
            'trajectory': cleaned_traj,
            'index': len(self.trajectories) - 1
        })
        self.update_waypoint_list()
        self.update_info()
        self.update_trajectory_visualization()
        self._update_fpv_camera()
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
        if not self.marker_pool:
            self.initialize_pools()
        geoms = self.scene.o3d_geometries.copy()
        marker_index = 0
        frustum_index = 0

        for traj in self.trajectories:
            points = traj.get_positions()
            if marker_index >= len(self.marker_pool):
                self.marker_pool.append(o3d.geometry.PointCloud())
            pcd = self.marker_pool[marker_index]
            pcd.points = o3d.utility.Vector3dVector(points)
            colors = np.tile(traj.color, (len(points), 1))
            if self.selected_trajectory == traj and self.selected_waypoint is not None:
                if 0 <= self.selected_waypoint < len(points):
                    colors[self.selected_waypoint] = [1.0, 1.0, 0.0]
            pcd.colors = o3d.utility.Vector3dVector(colors)
            geoms.append(pcd)
            marker_index += 1

            positions = np.array([wp.position for wp in traj.waypoints])
            yaws = np.array([np.radians(wp.yaw) for wp in traj.waypoints])
            pitches = np.array([np.radians(wp.pitch) for wp in traj.waypoints])
            scales = np.array([0.3 if (self.selected_trajectory == traj and i == self.selected_waypoint) else 0.2 for i in range(len(traj.waypoints))])

            for i in range(len(traj.waypoints)):
                if frustum_index >= len(self.frustum_pool):
                    self.frustum_pool.append(o3d.geometry.LineSet())
                frustum = self.frustum_pool[frustum_index]
                frustum = self.generate_camera_visualization(
                    positions[i], np.degrees(yaws[i]), np.degrees(pitches[i]), scale=scales[i], color=traj.color
                )
                geoms.append(frustum)
                frustum_index += 1

            # Render both interpolation methods
            linear_path = TrajectoryInterpolator.interpolate_linear(traj.waypoints, num_points=200)
            spline_path = TrajectoryInterpolator.interpolate_spline(traj.waypoints, num_points=200)
            lines = [[i, i+1] for i in range(len(linear_path)-1)]
            linear_lineset = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(linear_path),
                lines=o3d.utility.Vector2iVector(lines)
            )
            linear_lineset.paint_uniform_color([traj.color[0] * 0.5, traj.color[1] * 0.5, traj.color[2] * 0.5])
            geoms.append(linear_lineset)
            lines = [[i, i+1] for i in range(len(spline_path)-1)]
            spline_lineset = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(spline_path),
                lines=o3d.utility.Vector2iVector(lines)
            )
            spline_lineset.paint_uniform_color(traj.color)
            geoms.append(spline_lineset)

            for i in range(0, len(spline_path)-1, 10):
                pos = spline_path[i]
                next_pos = spline_path[i+1] if i+1 < len(spline_path) else spline_path[i]
                direction = next_pos - pos
                if np.linalg.norm(direction) > 1e-6:
                    direction /= np.linalg.norm(direction)
                    yaw = np.arctan2(direction[0], direction[2])
                    pitch = np.arcsin(direction[1])
                    arrow = self.generate_camera_visualization(
                        pos, np.degrees(yaw), np.degrees(pitch), scale=0.1, color=traj.color
                    )
                    geoms.append(arrow)

        for i in range(marker_index, len(self.marker_pool)):
            self.marker_pool[i].points = o3d.utility.Vector3dVector([])
        for i in range(frustum_index, len(self.frustum_pool)):
            self.frustum_pool[i].points = o3d.utility.Vector3dVector([])

        self.main_cmd_q.put({'cmd': 'set_geometries', 'geometries': geoms})
        self.fpv_cmd_q.put({'cmd': 'set_geometries', 'geometries': geoms})

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
        rotated_points = [np.dot(rot_pitch, np.dot(rot_yaw, point)) + position for point in points]
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
        self.main_cmd_q.put({'cmd': 'set_camera', 'params': params})

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
            self.fpv_cmd_q.put({'cmd': 'set_camera', 'params': params})

    def add_waypoint(self):
        pos = self.camera.position.copy()
        if not self.selected_trajectory:
            name = f"trajectory_{len(self.trajectories)+1}"
            wp = Waypoint(position=pos, yaw=self.camera.yaw, pitch=self.camera.pitch, index=0)
            traj = Trajectory(name=name, waypoints=[wp], color=(float(np.random.rand()), float(np.random.rand()), float(np.random.rand())))
            self.trajectories.append(traj)
            self.selected_trajectory = traj
            self.selected_waypoint = 0
            self.add_command({'type': 'add', 'position': pos, 'yaw': self.camera.yaw, 'pitch': self.camera.pitch, 'index': 0})
        else:
            traj = self.selected_trajectory
            idx = len(traj.waypoints)
            wp = Waypoint(position=pos, yaw=self.camera.yaw, pitch=self.camera.pitch, index=idx)
            traj.waypoints.append(wp)
            self.selected_waypoint = idx
            self.add_command({'type': 'add', 'position': pos, 'yaw': self.camera.yaw, 'pitch': self.camera.pitch, 'index': idx})
        self.update_waypoint_list()
        self.update_info()
        self.update_trajectory_visualization()
        self._update_fpv_camera()

    def edit_waypoint(self):
        if not self.selected_trajectory or self.selected_waypoint is None:
            messagebox.showwarning("Warning", "No trajectory/waypoint selected")
            return
        wp = self.selected_trajectory.waypoints[self.selected_waypoint]
        if self.edit_window and self.edit_window.winfo_exists():
            self.edit_window.lift()
            return
        self.edit_window = tk.Toplevel(self.root)
        self.edit_window.title("Edit Waypoint")
        self.edit_window.geometry("300x300")
        x_var = tk.DoubleVar(value=wp.position[0])
        y_var = tk.DoubleVar(value=wp.position[1])
        z_var = tk.DoubleVar(value=wp.position[2])
        yaw_var = tk.DoubleVar(value=wp.yaw)
        pitch_var = tk.DoubleVar(value=wp.pitch)

        def update_waypoint(*args):
            wp.position = np.array([x_var.get(), y_var.get(), z_var.get()])
            wp.yaw, wp.pitch = yaw_var.get(), pitch_var.get()
            self.update_trajectory_visualization()
            self._update_fpv_camera()

        ttk.Label(self.edit_window, text="X:").grid(row=0, column=0, padx=5, pady=2)
        ttk.Entry(self.edit_window, textvariable=x_var).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(self.edit_window, text="Y:").grid(row=1, column=0, padx=5, pady=2)
        ttk.Entry(self.edit_window, textvariable=y_var).grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(self.edit_window, text="Z:").grid(row=2, column=0, padx=5, pady=2)
        ttk.Entry(self.edit_window, textvariable=z_var).grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(self.edit_window, text="Yaw:").grid(row=3, column=0, padx=5, pady=2)
        ttk.Entry(self.edit_window, textvariable=yaw_var).grid(row=3, column=1, padx=5, pady=2)
        ttk.Label(self.edit_window, text="Pitch:").grid(row=4, column=0, padx=5, pady=2)
        ttk.Entry(self.edit_window, textvariable=pitch_var).grid(row=4, column=1, padx=5, pady=2)

        x_var.trace("w", update_waypoint)
        y_var.trace("w", update_waypoint)
        z_var.trace("w", update_waypoint)
        yaw_var.trace("w", update_waypoint)
        pitch_var.trace("w", update_waypoint)

        def save():
            self.add_command({
                'type': 'edit',
                'index': self.selected_waypoint,
                'old_position': wp.position.copy(),
                'old_yaw': wp.yaw,
                'old_pitch': wp.pitch,
                'new_position': np.array([x_var.get(), y_var.get(), z_var.get()]),
                'new_yaw': yaw_var.get(),
                'new_pitch': pitch_var.get()
            })
            self.edit_window.destroy()

        ttk.Button(self.edit_window, text="Save", command=save).grid(row=5, column=0, columnspan=2, pady=10)
        ttk.Button(self.edit_window, text="Undo", command=self.undo).grid(row=6, column=0, pady=5)
        ttk.Button(self.edit_window, text="Redo", command=self.redo).grid(row=6, column=1, pady=5)

        def on_key(event):
            if event.keysym == 'Up':
                z_var.set(z_var.get() + 0.1)
            elif event.keysym == 'Down':
                z_var.set(z_var.get() - 0.1)
            elif event.keysym == 'Left':
                x_var.set(x_var.get() - 0.1)
            elif event.keysym == 'Right':
                x_var.set(x_var.get() + 0.1)
            elif event.keysym == 'w':
                y_var.set(y_var.get() + 0.1)
            elif event.keysym == 's':
                y_var.set(y_var.get() - 0.1)
            elif event.keysym == 'q':
                yaw_var.set(yaw_var.get() - 1.0)
            elif event.keysym == 'e':
                yaw_var.set(yaw_var.get() + 1.0)
        self.edit_window.bind('<Key>', on_key)

    def delete_waypoint(self):
        if not self.selected_trajectory or self.selected_waypoint is None:
            messagebox.showwarning("Warning", "No trajectory/waypoint selected")
            return
        traj = self.selected_trajectory
        idx = self.selected_waypoint
        if 0 <= idx < len(traj.waypoints):
            wp = traj.waypoints[idx]
            self.add_command({
                'type': 'delete',
                'position': wp.position.copy(),
                'yaw': wp.yaw,
                'pitch': wp.pitch,
                'index': idx
            })
            traj.waypoints.pop(idx)
            for i, wp in enumerate(traj.waypoints):
                wp.index = i
            self.selected_waypoint = None
            self.update_waypoint_list()
            self.update_info()
            self.update_trajectory_visualization()
            self._update_fpv_camera()

    def on_waypoint_select(self, event):
        """Handle waypoint selection from the listbox."""
        sel = event.widget.curselection()
        if not sel:
            return
        idx = sel[0]
        if not self.selected_trajectory:
            return
        if 0 <= idx < len(self.selected_trajectory.waypoints):
            self.selected_waypoint = idx
            self._update_fpv_camera()
            self.update_trajectory_visualization()
            self.update_waypoint_list()  # Ensure listbox reflects selection

    def on_right_click(self, event):
        """Select waypoint by right-clicking near it in the 3D view."""
        if not self.selected_trajectory:
            return
        points = self.selected_trajectory.get_positions()
        if len(points) == 0:
            return
        camera_pos = self.camera.position
        distances = np.linalg.norm(points - camera_pos, axis=1)
        idx = np.argmin(distances)
        self.selected_waypoint = idx
        self.update_waypoint_list()
        self.update_trajectory_visualization()
        self._update_fpv_camera()

    def compare_trajectories(self):
        if len(self.trajectories) < 2:
            messagebox.showwarning("Warning", "Need at least 2 trajectories to compare")
            return
        output = []
        for traj in self.trajectories:
            metrics = traj.calculate_metrics()
            output.append({'name': traj.name, 'metrics': metrics})
        self.comp_text.delete(1.0, tk.END)
        self.comp_text.insert(tk.END, "TRAJECTORY COMPARISON\n" + "="*30 + "\n\n")
        for c in output:
            m = c['metrics']
            self.comp_text.insert(tk.END, f"{c['name']}\n  Length: {m['total_length']:.2f} m\n  Vertical: {m['total_vertical']:.2f} m\n  Duration: {m['total_duration']:.2f} s\n  Waypoints: {m['num_waypoints']}\n  Sharp Corners: {m['sharp_corners']}\n\n")

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
        linear_distances = np.linalg.norm(np.diff(linear_path, axis=0), axis=1)
        spline_distances = np.linalg.norm(np.diff(spline_path, axis=0), axis=1)
        linear_length = np.sum(linear_distances)
        spline_length = np.sum(spline_distances)
        linear_corners = 0
        for i in range(1, len(linear_path) - 1):
            v1 = linear_path[i] - linear_path[i - 1]
            v2 = linear_path[i + 1] - linear_path[i]
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                continue
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
            if np.degrees(angle) > 90:
                linear_corners += 1
        spline_corners = 0
        for i in range(1, len(spline_path) - 1):
            v1 = spline_path[i] - spline_path[i - 1]
            v2 = spline_path[i + 1] - spline_path[i]
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                continue
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
            if np.degrees(angle) > 90:
                spline_corners += 1
        min_len = min(len(linear_path), len(spline_path))
        diff = np.linalg.norm(linear_path[:min_len] - spline_path[:min_len], axis=1)
        avg_diff = np.mean(diff)
        max_diff = np.max(diff)
        self.comp_text.delete(1.0, tk.END)
        self.comp_text.insert(tk.END, f"Interpolation Comparison for '{traj.name}'\n" + "="*30 + "\n\n")
        self.comp_text.insert(tk.END, f"Linear Path:\n  Length: {linear_length:.3f} m\n  Sharp Corners: {linear_corners}\n")
        self.comp_text.insert(tk.END, f"Spline Path:\n  Length: {spline_length:.3f} m\n  Sharp Corners: {spline_corners}\n")
        self.comp_text.insert(tk.END, f"\nPointwise Differences:\n  Average: {avg_diff:.3f} m\n  Maximum: {max_diff:.3f} m\n")
        messagebox.showinfo("Interpolation", f"Linear: {linear_length:.3f} m, {linear_corners} corners\nSpline: {spline_length:.3f} m, {spline_corners} corners")

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
            dialog = tk.Toplevel(self.root)
            dialog.title("Recording Settings")
            dialog.geometry("300x150")
            ttk.Label(dialog, text="Framerate (FPS):").pack(pady=5)
            fps_var = tk.IntVar(value=30)
            ttk.Entry(dialog, textvariable=fps_var).pack(pady=5)
            def start():
                filepath = filedialog.asksaveasfilename(title="Save Recording", defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
                if not filepath:
                    dialog.destroy()
                    return
                res = (self.main_canvas.winfo_width() or 960, self.main_canvas.winfo_height() or 720)
                self.video_recorder = VideoRecorder(filepath, fps=fps_var.get(), resolution=res)
                if self.selected_trajectory:
                    self.video_recorder.set_trajectory_info(self.selected_trajectory.name, self.selected_trajectory.calculate_metrics())
                self.video_recorder.start_recording()
                self.record_button.config(text="Stop Recording")
                messagebox.showinfo("Recording", "Recording started...")
                dialog.destroy()
            ttk.Button(dialog, text="Start", command=start).pack(pady=10)

    def on_canvas_resize(self, event):
        if self.main_renderer:
            self.main_cmd_q.put({'cmd': 'resize', 'width': event.width, 'height': event.height})

    def on_fpv_resize(self, event):
        if self.fpv_renderer:
            self.fpv_cmd_q.put({'cmd': 'resize', 'width': event.width, 'height': event.height})

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

    def update_loop(self):
        self.main_cmd_q.put({'cmd': 'render'})
        self.fpv_cmd_q.put({'cmd': 'render'})
        try:
            while True:
                item = self.main_res_q.get_nowait()
                if item['type'] == 'frame':
                    img = item['image']
                    self._last_main_frame = img
                    self._display_main_frame(img)
        except queue.Empty:
            pass
        try:
            while True:
                item = self.fpv_res_q.get_nowait()
                if item['type'] == 'frame':
                    img = item['image']
                    self._last_fpv_frame = img
                    self._display_fpv_frame(img)
        except queue.Empty:
            pass
        if self._update_loop_running:
            self.root.after(33, self.update_loop)

    def _display_main_frame(self, arr: np.ndarray):
        try:
            img = Image.fromarray(arr)
            cw = self.main_canvas.winfo_width() or 960
            ch = self.main_canvas.winfo_height() or 720
            img = img.resize((cw, ch), Image.LANCZOS)
            self._main_photo = ImageTk.PhotoImage(img)
            self.main_canvas.delete("all")
            self.main_canvas.create_image(0, 0, anchor=tk.NW, image=self._main_photo)
            if self.video_recorder and self.video_recorder.is_recording:
                self.video_recorder.set_current_position(self.camera.position)
                self.video_recorder.add_frame(arr)
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
        m = self.selected_trajectory.calculate_metrics()
        self.info_text.insert(tk.END, f"Trajectory: {self.selected_trajectory.name}\n")
        self.info_text.insert(tk.END, f"Waypoints: {m['num_waypoints']}\n")
        self.info_text.insert(tk.END, f"Total length: {m['total_length']:.2f} m\n")
        self.info_text.insert(tk.END, f"Total vertical: {m['total_vertical']:.2f} m\n")
        self.info_text.insert(tk.END, f"Estimated duration: {m['total_duration']:.2f} s\n")
        self.info_text.insert(tk.END, f"Sharp corners: {m['sharp_corners']}\n")

    def close(self):
        self._update_loop_running = False
        try:
            self.main_cmd_q.put({'cmd': 'stop'})
            self.fpv_cmd_q.put({'cmd': 'stop'})
        except Exception:
            pass
        time.sleep(0.2)

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