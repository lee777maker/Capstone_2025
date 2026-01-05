#Imports for data handling and manipulation
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import yaml
import os
import shutil
import tempfile

#visualization libraries
import plotly.graph_objects as go
import plotly.io as pio
import open3d as o3d
import pyrender

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QCoreApplication, QSettings
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QSlider, QLabel, QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox
)
from PyQt5.QtWebEngineWidgets import QWebEngineView  # For embedded webview
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from helpers import (
    init, initialize_logging, load_camera_parameters, generate_camera_3d_thickness, render_scene, create_route_trace, init_renderer, shared
)
#import for flight path visulation
from flight_path_object import FlightPathObjectsBuilder  # Import for the builder


# -----------------------------
# Default configuration for the application
# appears in beginning , when no config.yml
# -----------------------------
DEFAULT_CONFIG = {
    'scene': {
        'id': 2,
        'name': 'Kiri_Vehera_SriLanka',
        'dataset_path': '.',
        'voxel_size': 0.2,
        'render_scale': 0.5,
        'ground_plane': {'enabled': True, 'size': 100, 'z': 0},
        'axes': {'enabled': True, 'length': 10},
    },
    'ui': {
        'window_title': 'UAV Flight Planning',
        'window_geometry': [100, 100, 1200, 800],
        'use_plotly_webview': True,
        'camera_slider_ranges': {
            'X': [-100, 100, 0],
            'Y': [-100, 100, 0],
            'Z': [-100, 100, 10],
            'Yaw': [-180, 180, 0],
            'Pitch': [-90, 90, -20],
        },
    },
    'defaults': {
        'load_mesh': False,
        'load_trajectory': False,
        'default_cameras': [
            {'origin': [0.0, -10.0, 3.0], 'yaw': 90.0, 'pitch': -20},
            {'origin': [3.8, -1.24, 2.0], 'yaw': 162.0, 'pitch': -20},
            {'origin': [5.88, 8.09, 3.0], 'yaw': -126.0, 'pitch': -20},
            {'origin': [-2.35, 3.24, 2.0], 'yaw': -54.0, 'pitch': -20},
            {'origin': [-9.51, -3.09, 3.0], 'yaw': 18.0, 'pitch': -20},
        ],
    },
}

# Loading file 
def load_app_config(config_path: str = "config.yml") -> dict:
    cfg_file = Path(config_path)
    #use default if not exists
    if not cfg_file.exists():
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_file, 'w') as f:
            yaml.safe_dump(DEFAULT_CONFIG, f)
        return DEFAULT_CONFIG
    #load the user config
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    # Merge shallowly with defaults to avoid missing keys
    def merge(d, default):
        out = dict(default) #use dictionary because of nested structures
        for k, v in (d or {}).items():
            if isinstance(v, dict) and isinstance(default.get(k), dict):
                out[k] = merge(v, default[k])
            else:
                out[k] = v
        return out
    return merge(cfg, DEFAULT_CONFIG)

# Thread workers
class RenderThread(QThread):
    render_finished = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    #intialiser with camera params
    def __init__(self, origin: np.ndarray, yaw: float, pitch: float):
        super().__init__()
        self.origin = origin
        self.yaw = float(yaw)
        self.pitch = float(pitch)

    #run thread
    def run(self):
        try:
            colormap, _ = render_scene(self.origin, self.yaw, self.pitch)
            self.render_finished.emit(colormap)
        except Exception as e:
            self.error_signal.emit(f"Render error: {e}")

# -----------------------------
# Core visualization pieces
# -----------------------------
class CameraController:

    #initialiser with default position and orientation
    def __init__(self, start_pos=(0.0, 0.0, 10.0), yaw=0.0, pitch=-20.0):
        self.position = np.array(start_pos, dtype=float)
        self.yaw = float(yaw)
        self.pitch = float(pitch)

    def update_position(self, dx: float, dy: float, dz: float):
        self.position += np.array([dx, dy, dz], dtype=float)

    def set_position(self, x: float, y: float, z: float):
        self.position[:] = [x, y, z]

    def update_rotation(self, dyaw: float, dpitch: float):
        self.yaw = (self.yaw + dyaw) % 360
        self.pitch = float(np.clip(self.pitch + dpitch, -90, 90))

    def set_rotation(self, yaw: float, pitch: float):
        self.yaw = float(yaw) % 360
        self.pitch = float(np.clip(pitch, -90, 90))

    def get_plotly_camera(self):
        return dict(
            eye=dict(x=self.position[0], y=self.position[1], z=self.position[2]),
            up=dict(x=0, y=0, z=1)
        )
    
class PlotlyRenderer:
    def __init__(self):
        self.fig = go.Figure()
        self.camera_controller = CameraController()

    def reset(self):
        self.fig = go.Figure()

    from typing import Any

    def add_objects(self, plotly_objs: List[Any]):
        self.fig.data = []
        for obj in plotly_objs:
            if obj is not None:
                self.fig.add_trace(obj)

    def update_layout(self, title: Optional[str] = None):
        layout_kwargs = dict(
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                camera=self.camera_controller.get_plotly_camera()
            ),
            showlegend=True
        )
        if title:
            layout_kwargs['title'] = title
        self.fig.update_layout(**layout_kwargs)

    def get_html(self) -> str:
        return pio.to_html(self.fig, full_html=False, include_plotlyjs='cdn')

    # Fallback extraction for Matplotlib view
    def get_mesh_data(self) -> Optional[dict]:
        for tr in self.fig.data:
            if isinstance(tr, go.Mesh3d):
                verts = np.column_stack([tr.x, tr.y, tr.z]).astype(float)
                faces = np.column_stack([tr.i, tr.j, tr.k]).astype(int)
                return {'vertices': verts, 'faces': faces}
        return None

    def get_route_data(self) -> Optional[dict]:
        for tr in self.fig.data:
            if isinstance(tr, go.Scatter3d) and tr.mode and 'lines' in tr.mode:
                return {'x': np.array(tr.x, float), 'y': np.array(tr.y, float), 'z': np.array(tr.z, float)}
        return None

class SceneManager:
    def __init__(self, config: dict):
        self.config = config
        self.mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.mesh_path: Optional[str] = None  # Store the mesh path
        self.plotly_objs: List[go.BaseTraceType] = []
        try:
            initialize_logging("config.yml", is_for_photo=False)
            init("config.yml", is_for_photo=False)
        except Exception:
            pass  # safe fallback if helpers handle their own init

    def load_mesh(self, mesh_path: str):
        # Store the mesh path for later use
        self.mesh_path = mesh_path
        # First try to load the mesh file directly with Open3D
        mp = Path(mesh_path)
        if not mp.exists():
            raise FileNotFoundError(f"Mesh file not found: {mp}")
        if mp.suffix.lower() not in ['.ply', '.obj', '.stl', '.glb', '.gltf']:
            raise ValueError("Unsupported mesh format. Use .ply, .obj, .stl, .glb, or .gltf")
        # Load the mesh using Open3D
        self.mesh = o3d.io.read_triangle_mesh(str(mp.absolute()))  # Use absolute path
        if not self.mesh.has_vertices():
            raise ValueError(f"Mesh is empty: {mp}")
        # Compute vertex normals if not present
        if not self.mesh.has_vertex_normals():
            self.mesh.compute_vertex_normals()
        # Try to also load it via pyrender if available
        try:
            # Reinitialize the renderer with the correct mesh path
            init_renderer(scale=self.config['scene'].get('render_scale', 0.5), load_mesh=True)
            # Try to get mesh from pyrender scene if available
            if 'scene' in shared and shared['scene']:
                for node in shared['scene'].nodes:
                    if isinstance(node, pyrender.Node) and node.mesh:
                        vertices = node.mesh.primitives[0].positions
                        faces = node.mesh.primitives[0].indices
                        # Update our mesh with pyrender data if available
                        if vertices is not None and faces is not None:
                            self.mesh = o3d.geometry.TriangleMesh()
                            self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
                            self.mesh.triangles = o3d.utility.Vector3iVector(faces)
                            self.mesh.compute_vertex_normals()
                            break
        except Exception as e:
            print(f"Warning: Could not load mesh via pyrender: {e}")
        # Continue with Open3D mesh
        # Simplify if needed
        vx = float(self.config['scene'].get('voxel_size', 0.2))
        if vx and vx > 0:
            try:
                self.mesh = self.mesh.simplify_vertex_clustering(
                    voxel_size=vx,
                    contraction=o3d.geometry.SimplificationContraction.Average
                )
            except Exception:
                pass

    def add_ground_plane(self):
        gp_cfg = self.config['scene'].get('ground_plane', {})
        if not gp_cfg.get('enabled', True):
            return
        size = float(gp_cfg.get('size', 100))
        z = float(gp_cfg.get('z', 0))
        x = np.linspace(-size, size, 10)
        y = np.linspace(-size, size, 10)
        xg, yg = np.meshgrid(x, y)
        zg = np.full_like(xg, z, dtype=float)
        ground_trace = go.Surface(x=xg, y=yg, z=zg, colorscale='Greys', opacity=0.5, showscale=False, name='Ground')
        self.plotly_objs.append(ground_trace)

    def add_coordinate_axes(self):
        ax_cfg = self.config['scene'].get('axes', {})
        if not ax_cfg.get('enabled', True):
            return
        length = float(ax_cfg.get('length', 10))
        axes = [
            go.Scatter3d(x=[0, length], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=4), name='X'),
            go.Scatter3d(x=[0, 0], y=[0, length], z=[0, 0], mode='lines', line=dict(color='green', width=4), name='Y'),
            go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, length], mode='lines', line=dict(color='blue', width=4), name='Z'),
        ]
        self.plotly_objs.extend(axes)

    def get_scene_objects(self) -> List[go.BaseTraceType]:
        objs = list(self.plotly_objs)
        if self.mesh and self.mesh.has_vertices():
            vertices = np.asarray(self.mesh.vertices)
            triangles = np.asarray(self.mesh.triangles)
            if self.mesh.has_vertex_colors():
                vcols = (np.asarray(self.mesh.vertex_colors) * 255).astype(np.uint8)
                colors = [f'rgb({c[0]},{c[1]},{c[2]})' for c in vcols]
                mesh_trace = go.Mesh3d(
                    x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                    i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                    vertexcolor=colors, opacity=1.0, name='Mesh'
                )
            else:
                mesh_trace = go.Mesh3d(
                    x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                    i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                    color='lightblue', opacity=1.0, name='Mesh'
                )
            objs = [mesh_trace] + objs
        return objs

class WaypointParser:
    def __init__(self):
        self.waypoints: List[dict] = []
        self.order: List[int] = []
        self.metadata: dict = {}
        self.raw_params: dict = {}  # Store raw parameters

    def parse_json(self, json_path: str):
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Waypoint JSON not found: {json_path}")
        try:
            # Try using the helpers function first
            params = load_camera_parameters(str(json_path))
            self.raw_params = params
        except Exception:
            # Fallback to direct JSON loading
            with open(json_path, 'r') as f:
                params = json.load(f)
            self.raw_params = params
        self.metadata = {
            'num_cameras': params.get('num_cameras', len(params.get('positions', []))),
            'scene_name': params.get('scene_name', ''),
            'has_route': params.get('has_route', params.get('waypoint_order') is not None)
        }
        # Get positions - handle different formats
        positions = params.get('positions')
        if positions is None:
            # Try alternate format
            if 'cameras' in params:
                positions = [wp.get('position') for wp in params['cameras']]
            else:
                positions = []
        # Get rotations - handle different formats
        rotations = params.get('rotations')
        if rotations is None:
            # Try alternate format
            if 'cameras' in params:
                rotations = [wp.get('rotation') for wp in params['cameras']]
            else:
                rotations = [[0, 0] for _ in range(len(positions))]  # Default rotations
        # Build waypoints
        self.waypoints = []
        for i, pos in enumerate(positions):
            rot = rotations[i] if i < len(rotations) else [0, 0]
            self.waypoints.append({
                'position': np.array(pos, dtype=float),
                'rotation': np.array(rot, dtype=float)
            })
        # Get order
        self.order = params.get('waypoint_order', list(range(len(self.waypoints))))

class PathVisualizer:
    def __init__(self):
        self.plotly_objs: List[go.BaseTraceType] = []

    def visualize_trajectory(self, waypoints: List[dict], order: List[int], raw_params: dict = None, name: str = "Path", json_path: str = None):
        if not waypoints:
            print("No waypoints to visualize")
            return
        # Use FlightPathObjectsBuilder for primary visualization
        if json_path is None:
            print("No JSON path provided for builder; falling back to simple visualization")
            # Fallback: Simple Scatter3d visualization
            positions = np.array([wp['position'] for wp in waypoints])
            # Ensure order is a valid list of integers
            valid_order = [i for i in order if isinstance(i, int) and 0 <= i < len(waypoints)] if order else list(range(len(waypoints)))
            ordered_positions = positions[valid_order] if valid_order else positions
            n = len(ordered_positions)
            if n > 0:
                color_idx = np.linspace(0, 1, n)
                path_trace = go.Scatter3d(
                    x=ordered_positions[:, 0].tolist(),
                    y=ordered_positions[:, 1].tolist(),
                    z=ordered_positions[:, 2].tolist(),
                    mode='lines+markers',
                    line=dict(
                        color=color_idx.tolist(),
                        colorscale='Viridis',
                        width=6,
                        showscale=True,
                        colorbar=dict(title="Progress")
                    ),
                    marker=dict(
                        size=8,
                        color=color_idx.tolist(),
                        colorscale='Viridis',
                        showscale=False,
                        line=dict(width=2, color='white')
                    ),
                    name=name,
                    hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}'
                )
                self.plotly_objs.append(path_trace)
        else:
            try:
                builder = FlightPathObjectsBuilder(camera_scale=0.5, label_cameras=True)
                plotly_objs, positions, params = builder.build_from_json(
                    json_path,
                    use_route_if_available=True,
                    draw_sequential_path_if_no_route=True,
                    path_colorscale="Viridis",
                    path_line_width=6,
                    path_marker_size=1,
                )
                self.plotly_objs.extend([obj for obj in plotly_objs if obj is not None])  # Filter None objects
                print(f"Successfully used FlightPathObjectsBuilder with {len(plotly_objs)} objects")
                return  # Skip additional camera visualizations since builder includes them
            
            except Exception as e:
                print(f"Error using FlightPathObjectsBuilder: {e}")
                # Fallback to simple visualization
                positions = np.array([wp['position'] for wp in waypoints])
                valid_order = [i for i in order if isinstance(i, int) and 0 <= i < len(waypoints)] if order else list(range(len(waypoints)))
                ordered_positions = positions[valid_order] if valid_order else positions
                n = len(ordered_positions)
                if n > 0:
                    color_idx = np.linspace(0, 1, n)
                    path_trace = go.Scatter3d(
                        x=ordered_positions[:, 0].tolist(),
                        y=ordered_positions[:, 1].tolist(),
                        z=ordered_positions[:, 2].tolist(),
                        mode='lines+markers',
                        line=dict(
                            color=color_idx.tolist(),
                            colorscale='Viridis',
                            width=6,
                            showscale=True,
                            colorbar=dict(title="Progress")
                        ),
                        marker=dict(
                            size=8,
                            color=color_idx.tolist(),
                            colorscale='Viridis',
                            showscale=False,
                            line=dict(width=2, color='white')
                        ),
                        name=name,
                        hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}'
                    )
                    self.plotly_objs.append(path_trace)

        # Camera visualizations (only if builder failed)
        camera_traces = []
        used_indices = valid_order if valid_order else list(range(len(waypoints)))
        for i, wp_idx in enumerate(used_indices):
            if wp_idx < len(waypoints):
                wp = waypoints[wp_idx]
                try:
                    rotation = wp.get('rotation', np.array([0.0, 0.0]))
                    yaw = float(rotation[0]) if rotation.size > 0 else 0.0
                    pitch = float(rotation[1]) if rotation.size > 1 else 0.0
                    cam_objs = generate_camera_3d_thickness(
                        wp['position'], yaw, pitch, text=f"WP{i+1}", scale=0.5
                    )
                    if cam_objs:
                        camera_traces.extend([obj for obj in cam_objs if obj is not None])
                        print(f"Added camera visualization for WP{i+1}")
                except Exception as e:
                    print(f"Warning: Could not generate camera {i+1}: {e}")
                    p = wp['position']
                    camera_traces.append(
                        go.Scatter3d(
                            x=[float(p[0])], y=[float(p[1])], z=[float(p[2])],
                            mode='markers+text',
                            marker=dict(size=10, color='red', symbol='diamond'),
                            text=[f"WP{i+1}"],
                            textposition="top center",
                            showlegend=False,
                            name=f"WP{i+1}"
                        )
                    )
                    print(f"Added fallback marker for WP{i+1}")
        self.plotly_objs.extend(camera_traces)
        print(f"Total plotly objects: {len(self.plotly_objs)}")
        print("*******************************************")
        print("Done visualising trajectory")
        print("************************************************")

    def visualize_default_cameras(self, camera_params: List[dict], name: str = "Default Path"):
        if not camera_params:
            return
        positions = np.array([np.array(p['origin'], float) for p in camera_params])
        n = len(positions)
        cidx = np.linspace(0, 1, n)
        path_trace = go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            mode='lines+markers',
            line=dict(color=cidx, colorscale='Viridis', width=6, showscale=True),
            marker=dict(size=5, color=cidx, colorscale='Viridis', line=dict(width=1, color='white')),
            customdata=np.arange(n),
            name=name
        )
        self.plotly_objs.append(path_trace)
        # Add camera markers
        camera_traces = []
        for i, params in enumerate(camera_params):
            try:
                objs = generate_camera_3d_thickness(
                    params['origin'], params['yaw'], params['pitch'], text=f"Camera {i+1}", scale=0.5
                )
                if objs:
                    camera_traces.extend(objs)
            except Exception:
                # Fallback simple marker
                p = np.array(params['origin'], float)
                camera_traces.append(
                    go.Scatter3d(
                        x=[p[0]], y=[p[1]], z=[p[2]],
                        mode='markers+text',
                        marker=dict(size=8, color='blue'),
                        text=[f"Camera {i+1}"],
                        textposition="top center",
                        showlegend=False
                    )
                )
        self.plotly_objs.extend(camera_traces)

    def get_plotly_objects(self) -> List[go.BaseTraceType]:
        return [obj for obj in self.plotly_objs if obj is not None]

class DataExporter:
    @staticmethod
    def _to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: DataExporter._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [DataExporter._to_serializable(x) for x in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    def export_mission(self, waypoints: List[dict], order: List[int], output_path: str):
        mission = {
            'metadata': {'total_cameras': len(waypoints)},
            'cameras': [
                {'position': self._to_serializable(wp['position']), 'rotation': self._to_serializable(wp['rotation'])}
                for wp in waypoints
            ],
            'waypoint_order': self._to_serializable(order)
        }
        with open(output_path, 'w') as f:
            json.dump(mission, f, indent=2)

class VisualizationSystem:

    def __init__(self, config: dict):
        self.config = config
        self.scene_manager = SceneManager(config)
        self.renderer = PlotlyRenderer()
        self.exporter = DataExporter()
        self.waypoints: List[dict] = []
        self.order: List[int] = []
        self.trajectory_manager: Optional[Tuple[WaypointParser, Optional[PathVisualizer]]] = None
        self.render_thread: Optional[RenderThread] = None
        self.json_path: Optional[str] = None  #Store the loaded JSON path for builder use

    # API
    def load_scene(self, mesh_path: str):
        # Clear any existing trajectory visualization
        self.trajectory_manager = None
        self.waypoints = []
        self.order = []
        # Clear the renderer
        self.renderer.reset()
        # Load new mesh
        self.scene_manager.load_mesh(mesh_path)
        self.scene_manager.plotly_objs.clear()
        self.scene_manager.add_ground_plane()
        self.scene_manager.add_coordinate_axes()
        self.render()

    def load_trajectory(self, json_path: str):
        self.json_path = json_path  #Store json_path for use in visualization
        parser = WaypointParser()
        parser.parse_json(json_path)
        self.waypoints = parser.waypoints
        self.order = parser.order
        self.trajectory_manager = (parser, None)

    def visualize_trajectory(self):
        if not self.trajectory_manager:
            raise ValueError("No trajectory loaded")
        parser, visualizer = self.trajectory_manager
        if visualizer is None:
            visualizer = PathVisualizer()
            # CHANGED: Pass json_path to the visualizer
            visualizer.visualize_trajectory(
                parser.waypoints, parser.order, raw_params=parser.raw_params, json_path=self.json_path
            )
            self.trajectory_manager = (parser, visualizer)
        self.render()

    def load_default_cameras(self, camera_params: List[dict]):
        visualizer = PathVisualizer()
        visualizer.visualize_default_cameras(camera_params)
        self.trajectory_manager = (None, visualizer)
        self.render()

    def compare_trajectories(self, json_paths: List[str]):
        # Accumulate multiple path visualizations
        for i, json_path in enumerate(json_paths):
            parser = WaypointParser()
            parser.parse_json(json_path)
            visualizer = PathVisualizer()
            visualizer.visualize_trajectory(
                parser.waypoints, parser.order, raw_params=parser.raw_params, name=f"Path {i+1}", json_path=json_path  # CHANGED: Pass json_path
            )
            # Add to renderer without clearing
            current = list(self.renderer.fig.data)
            self.renderer.add_objects(list(current) + visualizer.get_plotly_objects())
        self.renderer.update_layout(title="Trajectory Comparison")

    def render(self):
        scene_objects = self.scene_manager.get_scene_objects()
        trajectory_objects = self.trajectory_manager[1].get_plotly_objects() if self.trajectory_manager and self.trajectory_manager[1] else []
        self.renderer.add_objects(scene_objects + trajectory_objects)
        self.renderer.update_layout(title=self.config['ui'].get('window_title', 'UAV Flight Planning'))

    def render_waypoint_view(self, waypoint_index: int, callback):
        if not self.trajectory_manager or not self.trajectory_manager[0]:
            # if No JSON trajectory loaded -> placeholder FPV
            width, height = 640, 480
            colormap = np.full((height, width, 3), 128, dtype=np.uint8)
            callback(colormap)
            return None
        parser, _ = self.trajectory_manager
        if waypoint_index < 0 or waypoint_index >= len(parser.waypoints):
            width, height = 640, 480
            colormap = np.full((height, width, 3), 128, dtype=np.uint8)
            callback(colormap)
            return None
        wp = parser.waypoints[waypoint_index]
        try:
            colormap, _ = render_scene(wp['position'], wp['rotation'][0], wp['rotation'][1])
        except Exception as e:
            print(f"Render error: {e}")
            colormap = np.full((480, 640, 3), 128, dtype=np.uint8)
        callback(colormap)

    def export_mission(self, output_path: str):
        if self.trajectory_manager and self.trajectory_manager[0]:
            parser, _ = self.trajectory_manager
            self.exporter.export_mission(parser.waypoints, parser.order, output_path)

# -----------------------------
# Qt application
# -----------------------------
class ViewerApp(QMainWindow):

    def __init__(self, vis_system: VisualizationSystem, config: dict):
        super().__init__()
        self.vis_system = vis_system
        self.config = config
        self.settings = QSettings("YourOrg", "UAVViewer")
        self._web_window = None  #Store webview window
        self._init_ui()
        self._load_initial_data()

    # UI construction
    def _init_ui(self):
        title = self.config['ui'].get('window_title', 'UAV Flight Planning')
        x, y, w, h = self.config['ui'].get('window_geometry', [100, 100, 1200, 800])
        self.setWindowTitle(title)
        self.setGeometry(x, y, w, h)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        main_layout.addWidget(left_widget, stretch=1)
        
        # Controls
        control_widget = QWidget()
        control_layout = QVBoxLayout()
        control_widget.setLayout(control_layout)
        left_layout.addWidget(control_widget)
        control_layout.addWidget(QLabel("Camera Controls"))

        # sliders from config
        sliders_cfg = self.config['ui'].get('camera_slider_ranges', {})
        self._sliders = {}
        for name, values in sliders_cfg.items():
            min_val, max_val, default = values
            if name not in ['X', 'Y', 'Z', 'Yaw', 'Pitch']:
                continue
            slider = self._add_slider(control_layout, name, min_val, max_val, default, 
                                      getattr(self, f"_update_camera_{name.lower()}"))
            self._sliders[name] = slider

        control_layout.addWidget(QLabel("Scene and Trajectory"))
        self.load_mesh_btn = QPushButton("Load Mesh")
        self.load_mesh_btn.clicked.connect(self._load_mesh_dialog)
        control_layout.addWidget(self.load_mesh_btn)
        self.load_traj_btn = QPushButton("Load Trajectory")
        self.load_traj_btn.clicked.connect(self._load_trajectory_dialog)
        control_layout.addWidget(self.load_traj_btn)
        self.visualize_traj_btn = QPushButton("Show Flight Path")
        self.visualize_traj_btn.clicked.connect(self._visualize_trajectory)
        control_layout.addWidget(self.visualize_traj_btn)

        # Center pane: 3D view with QWebEngineView
        center_widget = QWidget()
        center_layout = QVBoxLayout()
        center_widget.setLayout(center_layout)
        main_layout.addWidget(center_widget, stretch=3)
        self._web_view = QWebEngineView()
        center_layout.addWidget(self._web_view)

        # Right pane: table + FPV + export
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(right_widget, stretch=1)
        self.waypoint_table = QTableWidget()
        self.waypoint_table.setColumnCount(5)
        self.waypoint_table.setHorizontalHeaderLabels(["Index", "X", "Y", "Z", "Yaw"])
        self.waypoint_table.cellClicked.connect(self._waypoint_selected)
        right_layout.addWidget(self.waypoint_table)
        self.export_btn = QPushButton("Export Mission")
        self.export_btn.clicked.connect(self._export_mission_dialog)
        right_layout.addWidget(self.export_btn)
        # FPV canvas
        self.fpv_canvas = FigureCanvas(plt.Figure(figsize=(4, 3)))
        right_layout.addWidget(self.fpv_canvas)
        self.fpv_ax = self.fpv_canvas.figure.add_subplot(111)
        self.fpv_ax.axis('off')

    def _add_slider(self, layout, name, min_val, max_val, default, callback):
        slider_layout = QHBoxLayout()
        label = QLabel(f"{name}: {default}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(min_val), int(max_val))
        slider.setValue(int(default))
        slider.valueChanged.connect(lambda val: (callback(val), label.setText(f"{name}: {val}")))
        slider_layout.addWidget(QLabel(name))
        slider_layout.addWidget(slider)
        slider_layout.addWidget(label)
        layout.addLayout(slider_layout)
        return slider

    # Camera slider callbacks
    def _update_camera_x(self, val):
        cur = self.vis_system.renderer.camera_controller.position[0]
        self.vis_system.renderer.camera_controller.set_position(
            val, 
            self.vis_system.renderer.camera_controller.position[1], 
            self.vis_system.renderer.camera_controller.position[2])
        self._update_3d_view()

    def _update_camera_y(self, val):
        self.vis_system.renderer.camera_controller.set_position(
            self.vis_system.renderer.camera_controller.position[0], 
            val, 
            self.vis_system.renderer.camera_controller.position[2])
        self._update_3d_view()

    def _update_camera_z(self, val):
        self.vis_system.renderer.camera_controller.set_position(
            self.vis_system.renderer.camera_controller.position[0], 
            self.vis_system.renderer.camera_controller.position[1], 
            val)
        self._update_3d_view()

    def _update_camera_yaw(self, val):
        self.vis_system.renderer.camera_controller.set_rotation(
            val, 
            self.vis_system.renderer.camera_controller.pitch)
        self._update_3d_view()

    def _update_camera_pitch(self, val):
        self.vis_system.renderer.camera_controller.set_rotation(
            self.vis_system.renderer.camera_controller.yaw, 
            val)
        self._update_3d_view()


    def _update_3d_view(self):
        print("*******************************************")
        print("now updating 3d view in *update3d*view")
        print("************************************************")
        self.vis_system.render()
        # Render Plotly figure to HTML and load in QWebEngineView
        html = pio.to_html(self.vis_system.renderer.fig, include_plotlyjs='cdn')
        self._web_view.setHtml(html)

    # Dialogs and actions
    def _load_mesh_dialog(self):

        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Mesh File", str(Path.cwd() / "models"), "3D Models (*.obj *.ply *.stl)"
        )
        if not fname:
            return
        try:
            source_path = Path(fname)
            base_name = source_path.stem
            dest_dir = Path("models") / base_name / "Mesh"
            dest_path = dest_dir / source_path.name
            # Avoid copying if source and destination are the same
            if source_path.resolve() != dest_path.resolve():
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(source_path, dest_path)
                print(f"Copied mesh to: {dest_path}")
                # Copy .mtl file if it exists
                mtl_source = source_path.with_suffix('.mtl')
                mtl_dest = dest_path.with_suffix('.mtl')
                if mtl_source.exists() and mtl_source.resolve() != mtl_dest.resolve():
                    shutil.copy(mtl_source, mtl_dest)
                    print(f"Copied MTL to: {mtl_dest}")
                    #alter
                else:
                    print(f"Warning: Could not copy textures: {mtl_source} does not exist or is the same file")
            else:
                print(f"Mesh already at destination: {dest_path}")
            # Update shared config
            config = dict(self.config)
            config['scene_name'] = base_name
            config['dataset_path'] = str(dest_path.parent)
            shared['config'] = config
            # Initialize renderer
            try:
                init_renderer()
            except Exception as e:
                print(f"Warning: Could not init renderer: {e}")
            try:
                mesh = o3d.io.read_triangle_mesh(str(dest_path))
                if not mesh.has_vertices():
                    raise ValueError("Mesh has no vertices")
                vertices = np.asarray(mesh.vertices)
                triangles = np.asarray(mesh.triangles)
            
                mesh_trace = go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=triangles[:, 0],
                    j=triangles[:, 1],
                    k=triangles[:, 2],
                    color='lightblue',
                    opacity=0.5,
                    name='Mesh'
                )
                self.vis_system.renderer.fig.data = [trace for trace in self.vis_system.renderer.fig.data if not isinstance(trace, go.Mesh3d)]  # Clear existing meshes
                self.vis_system.renderer.fig.add_trace(mesh_trace)
                self._update_3d_view()
            except Exception as e:
                print(f"Error loading mesh into Plotly: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load mesh: {e}")

        except Exception as e:
            print(f"Error loading mesh: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load mesh: {e}")


    def _copy_textures(self, mtl_path, dest_dir):
        """Copy texture files referenced in MTL"""
        mtl_dir = mtl_path.parent
        try:
            with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    # Look for texture references
                    for prefix in ['map_Kd', 'map_Ka', 'map_Ks', 'map_d', 'map_Bump']:
                        if line.startswith(prefix):
                            parts = line.split(maxsplit=1)
                            if len(parts) > 1:
                                texture_name = parts[1]
                                texture_path = mtl_dir / texture_name
                                if texture_path.exists():
                                    dest_texture = dest_dir / texture_path.name
                                    if not dest_texture.exists():
                                        shutil.copy2(texture_path, dest_texture)
                                        print(f"Copied texture: {texture_path.name}")
        except Exception as e:
            print(f"Warning: Could not copy textures: {e}")

    def _load_trajectory_dialog(self):
        last_dir = self.settings.value("last_dir_traj", str(Path.home()))
        json_path, _ = QFileDialog.getOpenFileName(self, "Load Trajectory JSON", last_dir, "JSON Files (*.json)")
        if json_path:
            try:
                self.vis_system.load_trajectory(json_path)
                self.settings.setValue("last_dir_traj", str(Path(json_path).parent))
                self._update_waypoint_table()
                if self.vis_system.waypoints:
                    self._update_fpv(0)
                QMessageBox.information(self, "Success", f"Trajectory loaded: {len(self.vis_system.waypoints)} waypoints")
            except Exception as e:
                import traceback
                traceback.print_exc()  # Prints full traceback to console
                QMessageBox.warning(self, "Load Trajectory Error", str(e))

    def _visualize_trajectory(self):
        try:
            self.vis_system.visualize_trajectory()
            self._update_3d_view()
        except ValueError as e:
            QMessageBox.warning(self, "Visualization Error", str(e))

    def _export_mission_dialog(self):
        last_dir = self.settings.value("last_dir_export", str(Path.cwd()))
        output_path, _ = QFileDialog.getSaveFileName(self, "Export Mission", str(Path(last_dir) / "mission_output.json"), "JSON Files (*.json)")
        if output_path:
            try:
                self.vis_system.export_mission(output_path)
                self.settings.setValue("last_dir_export", str(Path(output_path).parent))
                QMessageBox.information(self, "Success", f"Mission exported to {output_path}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", str(e))

    #view updates
    def _update_waypoint_table(self):

        self.waypoint_table.setRowCount(len(self.vis_system.waypoints))
        for i, wp in enumerate(self.vis_system.waypoints):
            pos = wp['position']
            yaw = float(wp['rotation'][0]) if 'rotation' in wp and len(wp['rotation']) > 0 else 0.0
            self.waypoint_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.waypoint_table.setItem(i, 1, QTableWidgetItem(f"{pos[0]:.2f}"))
            self.waypoint_table.setItem(i, 2, QTableWidgetItem(f"{pos[1]:.2f}"))
            self.waypoint_table.setItem(i, 3, QTableWidgetItem(f"{pos[2]:.2f}"))
            self.waypoint_table.setItem(i, 4, QTableWidgetItem(f"{yaw:.2f}"))

    def _waypoint_selected(self, row, _):
        self._update_fpv(row)

    def _update_fpv(self, index: int):
        def update_plot(colormap: np.ndarray):
            self.fpv_ax.clear()
            self.fpv_ax.imshow(colormap)
            self.fpv_ax.axis('off')
            title = "FPV Preview (Placeholder)" if np.all(colormap == 128) else f"FPV Preview - WP{index+1}"
            self.fpv_ax.set_title(title)
            self.fpv_canvas.draw()
        self.vis_system.render_waypoint_view(index, update_plot)


    def _load_initial_data(self):
        # load defaults from config
        defaults = self.config.get('defaults', {})
        if defaults.get('default_cameras'):
            self.vis_system.load_default_cameras(defaults['default_cameras'])
            self._update_3d_view()

if __name__ == "__main__":
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    config = load_app_config("config.yml")
    vis = VisualizationSystem(config)
    window = ViewerApp(vis, config)
    window.show()
    sys.exit(app.exec_())