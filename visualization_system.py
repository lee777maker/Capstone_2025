from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

# Third-party visualization/rendering
import plotly.graph_objects as go
import plotly.io as pio
import open3d as o3d
import pyrender

# Qt
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QCoreApplication, QSettings
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel, QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox
)

# Optional: Plotly in Qt via WebEngine (preferred). Fallback to Matplotlib if unavailable.
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except Exception:
    HAS_WEBENGINE = False
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for canvas rendering
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Your helper module (assumed available in your project)
# If any helper is missing at runtime, guarded calls below will safely continue.
from helpers import (
    init, initialize_logging, load_camera_parameters, generate_camera_3d_thickness,
    render_scene, create_route_trace, init_renderer, shared
)


# -----------------------------
# Config management
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


def load_app_config(config_path: str = "config.yml") -> dict:
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_file, 'w') as f:
            yaml.safe_dump(DEFAULT_CONFIG, f)
        return DEFAULT_CONFIG
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    # Merge shallowly with defaults to avoid missing keys
    def merge(d, default):
        out = dict(default)
        for k, v in (d or {}).items():
            if isinstance(v, dict) and isinstance(default.get(k), dict):
                out[k] = merge(v, default[k])
            else:
                out[k] = v
        return out
    return merge(cfg, DEFAULT_CONFIG)


# -----------------------------
# Thread workers
# -----------------------------
class RenderThread(QThread):
    render_finished = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self, origin: np.ndarray, yaw: float, pitch: float):
        super().__init__()
        self.origin = origin
        self.yaw = float(yaw)
        self.pitch = float(pitch)

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
        return dict(eye=dict(x=self.position[0], y=self.position[1], z=self.position[2]), up=dict(x=0, y=0, z=1))


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
                xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                camera=self.camera_controller.get_plotly_camera()
            ),
            showlegend=True
        )
        if title:
            layout_kwargs['title'] = title
        self.fig.update_layout(**layout_kwargs)

    def get_html(self) -> str:
        return pio.to_html(self.fig, full_html=False, include_plotlyjs='cdn')

    # Fallback extraction for Matplotlib view (if WebEngine not available)
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
        self.plotly_objs: List[go.BaseTraceType] = []
        try:
            initialize_logging("config.yml", is_for_photo=False)
            init("config.yml", is_for_photo=False)
        except Exception:
            pass  # safe fallback if helpers handle their own init

    def load_mesh(self, mesh_path: str):
        # Try to pull mesh from a preloaded pyrender scene if available
        try:
            init_renderer(scale=self.config['scene'].get('render_scale', 0.5), load_mesh=False)
            if 'scene' in shared and shared['scene']:
                for node in shared['scene'].nodes:
                    if isinstance(node, pyrender.Node) and node.mesh:
                        vertices = node.mesh.primitives[0].positions
                        faces = node.mesh.primitives[0].indices
                        self.mesh = o3d.geometry.TriangleMesh()
                        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
                        self.mesh.triangles = o3d.utility.Vector3iVector(faces)
                        break
        except Exception:
            pass

        if self.mesh is None:
            mp = Path(mesh_path)
            if not mp.exists():
                raise FileNotFoundError(f"Mesh file not found: {mp}")
            if mp.suffix.lower() not in ['.ply', '.obj', '.stl', '.glb', '.gltf']:
                raise ValueError("Unsupported mesh format. Use .ply, .obj, .stl, .glb, or .gltf")
            self.mesh = o3d.io.read_triangle_mesh(str(mp))
            if not self.mesh.has_vertices():
                raise ValueError(f"Mesh is empty: {mp}")

        # Simplify/cluster if requested
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
            go.Scatter3d(x=[0, length], y=[0, 0], z=[0, 0], mode='lines',
                         line=dict(color='red', width=4), name='X'),
            go.Scatter3d(x=[0, 0], y=[0, length], z=[0, 0], mode='lines',
                         line=dict(color='green', width=4), name='Y'),
            go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, length], mode='lines',
                         line=dict(color='blue', width=4), name='Z'),
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

    def parse_json(self, json_path: str):
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Waypoint JSON not found: {json_path}")

        # Prefer project helper if available
        try:
            params = load_camera_parameters(str(json_path))
        except Exception:
            with open(json_path, 'r') as f:
                params = json.load(f)

        self.metadata = {
            'num_cameras': params.get('num_cameras', len(params.get('positions', []))),
            'scene_name': params.get('scene_name', ''),
            'has_route': params.get('has_route', True)
        }
        positions = params.get('positions')
        if positions is None:
            positions = [wp.get('position') for wp in params.get('cameras', [])]

        rotations = params.get('rotations')
        if rotations is None:
            rotations = [wp.get('rotation') for wp in params.get('cameras', [])]
        self.waypoints = [
            {'position': np.array(pos, dtype=float), 'rotation': np.array(rot, dtype=float)}
            for pos, rot in zip(positions, rotations)
        ]
        self.order = params.get('waypoint_order', list(range(len(self.waypoints))))


class PathVisualizer:
    def __init__(self):
        self.plotly_objs: List[go.BaseTraceType] = []

    def visualize_trajectory(self, waypoints: List[dict], order: List[int], name: str = "Path"):
        params = {
            'positions': np.array([wp['position'] for wp in waypoints]),
            'rotations': np.array([wp['rotation'] for wp in waypoints]),
            'waypoint_order': order,
            'path_color_scale': 'Plotly3',
            'name': name
        }
        try:
            path_trace = create_route_trace(params)
        except Exception:
            # Fallback: simple polyline
            pts = params['positions'][order]
            path_trace = go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='lines+markers',
                line=dict(color='royalblue', width=6),
                marker=dict(size=4, color='royalblue'),
                name=name
            )
        camera_traces = []
        for i, wp in enumerate(waypoints):
            try:
                cam_objs = generate_camera_3d_thickness(
                    wp['position'], wp['rotation'][0], wp['rotation'][1], text=f"WP{i+1}", scale=0.5
                )
                camera_traces.extend(cam_objs)
            except Exception:
                # Fallback: small axis tripod per camera
                p = wp['position']
                l = 0.3
                camera_traces.extend([
                    go.Scatter3d(x=[p[0], p[0]+l], y=[p[1], p[1]], z=[p[2], p[2]], mode='lines',
                                 line=dict(color='red', width=3), showlegend=False),
                    go.Scatter3d(x=[p[0], p[0]], y=[p[1], p[1]+l], z=[p[2], p[2]], mode='lines',
                                 line=dict(color='green', width=3), showlegend=False),
                    go.Scatter3d(x=[p[0], p[0]], y=[p[1], p[1]], z=[p[2], p[2]+l], mode='lines',
                                 line=dict(color='blue', width=3), showlegend=False),
                ])
        self.plotly_objs.extend([path_trace] + camera_traces)

    def visualize_default_cameras(self, camera_params: List[dict], name: str = "Default Path"):
        positions = np.array([np.array(p['origin'], float) for p in camera_params])
        n = len(positions)
        cidx = np.linspace(0, 1, n)
        path_trace = go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            mode='lines+markers', line=dict(color=cidx, colorscale='Plotly3', width=6),
            marker=dict(size=5, color=cidx, colorscale='Plotly3', line=dict(width=1, color='white')),
            customdata=np.arange(n), name=name
        )
        camera_traces = []
        for i, params in enumerate(camera_params):
            try:
                objs = generate_camera_3d_thickness(params['origin'], params['yaw'], params['pitch'],
                                                    text=f"Camera {i+1}", scale=0.5)
            except Exception:
                p = np.array(params['origin'], float)
                l = 0.3
                objs = [
                    go.Scatter3d(x=[p[0], p[0]+l], y=[p[1], p[1]], z=[p[2], p[2]], mode='lines',
                                 line=dict(color='red', width=3), showlegend=False),
                    go.Scatter3d(x=[p[0], p[0]], y=[p[1], p[1]+l], z=[p[2], p[2]], mode='lines',
                                 line=dict(color='green', width=3), showlegend=False),
                    go.Scatter3d(x=[p[0], p[0]], y=[p[1], p[1]], z=[p[2], p[2]+l], mode='lines',
                                 line=dict(color='blue', width=3), showlegend=False),
                ]
            camera_traces.extend(objs)
        self.plotly_objs.extend([path_trace] + camera_traces)

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
                {'position': self._to_serializable(wp['position']),
                 'rotation': self._to_serializable(wp['rotation'])}
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
        self.trajectory_manager: Optional[Tuple[WaypointParser, PathVisualizer]] = None
        self.render_thread: Optional[RenderThread] = None

    # API used by UI
    def load_scene(self, mesh_path: str):
        self.scene_manager.load_mesh(mesh_path)
        self.scene_manager.plotly_objs.clear()
        self.scene_manager.add_ground_plane()
        self.scene_manager.add_coordinate_axes()
        self.render()

    def load_trajectory(self, json_path: str):
        parser = WaypointParser()
        parser.parse_json(json_path)
        self.waypoints = parser.waypoints
        self.order = parser.order
        visualizer = PathVisualizer()
        visualizer.visualize_trajectory(parser.waypoints, parser.order)
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
            visualizer.visualize_trajectory(parser.waypoints, parser.order, name=f"Path {i+1}")
            # Add incrementally to renderer without clearing
            current = list(self.renderer.fig.data)
            self.renderer.add_objects(list(current) + visualizer.get_plotly_objects())
        self.renderer.update_layout(title="Trajectory Comparison")

    def render(self):
        scene_objects = self.scene_manager.get_scene_objects()
        trajectory_objects = self.trajectory_manager[1].get_plotly_objects() if self.trajectory_manager else []
        self.renderer.add_objects(scene_objects + trajectory_objects)
        self.renderer.update_layout(title=self.config['ui'].get('window_title', 'UAV Flight Planning'))

    def render_waypoint_view(self, waypoint_index: int, callback):
        if not self.trajectory_manager or not self.trajectory_manager[0]:
            # No JSON trajectory loaded -> placeholder FPV
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
        self.render_thread = RenderThread(wp['position'], wp['rotation'][0], wp['rotation'][1])
        self.render_thread.render_finished.connect(callback)
        self.render_thread.error_signal.connect(lambda msg: callback(np.full((480, 640, 3), 128, dtype=np.uint8)))
        self.render_thread.finished.connect(self.render_thread.deleteLater)
        self.render_thread.start()
        return self.render_thread

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
        self._web_view: Optional[QWebEngineView] = None
        self._mpl_canvas: Optional[FigureCanvas] = None
        self._mpl_ax = None

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

        # Left pane: 3D view + controls
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        main_layout.addWidget(left_widget, stretch=2)

        # 3D view
        use_webview = self.config['ui'].get('use_plotly_webview', True) and HAS_WEBENGINE
        if use_webview:
            self._web_view = QWebEngineView()
            left_layout.addWidget(self._web_view)
        else:
            self._mpl_canvas = FigureCanvas(plt.Figure(figsize=(6, 4)))
            left_layout.addWidget(self._mpl_canvas)
            self._mpl_ax = self._mpl_canvas.figure.add_subplot(111, projection='3d')
            self._mpl_ax.set_xlabel('X')
            self._mpl_ax.set_ylabel('Y')
            self._mpl_ax.set_zlabel('Z')

        # Controls
        control_widget = QWidget()
        control_layout = QVBoxLayout()
        control_widget.setLayout(control_layout)
        left_layout.addWidget(control_widget)

        control_layout.addWidget(QLabel("Camera Controls"))
        # Create sliders from config (no hardcoding)
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
        self.vis_system.renderer.camera_controller.set_position(val, self.vis_system.renderer.camera_controller.position[1], self.vis_system.renderer.camera_controller.position[2])
        self._update_3d_view()

    def _update_camera_y(self, val):
        self.vis_system.renderer.camera_controller.set_position(self.vis_system.renderer.camera_controller.position[0], val, self.vis_system.renderer.camera_controller.position[2])
        self._update_3d_view()

    def _update_camera_z(self, val):
        self.vis_system.renderer.camera_controller.set_position(self.vis_system.renderer.camera_controller.position[0], self.vis_system.renderer.camera_controller.position[1], val)
        self._update_3d_view()

    def _update_camera_yaw(self, val):
        self.vis_system.renderer.camera_controller.set_rotation(val, self.vis_system.renderer.camera_controller.pitch)
        self._update_3d_view()

    def _update_camera_pitch(self, val):
        self.vis_system.renderer.camera_controller.set_rotation(self.vis_system.renderer.camera_controller.yaw, val)
        self._update_3d_view()

    # Dialogs and actions
    def _load_mesh_dialog(self):
        last_dir = self.settings.value("last_dir_mesh", str(Path.home()))
        mesh_path, _ = QFileDialog.getOpenFileName(self, "Load Mesh File", last_dir, "Mesh Files (*.ply *.obj *.stl *.glb *.gltf)")
        if mesh_path:
            try:
                self.vis_system.load_scene(mesh_path)
                self.settings.setValue("last_dir_mesh", str(Path(mesh_path).parent))
                self._update_waypoint_table()  #  shows waypoints when loaded
                self._update_3d_view()
            except Exception as e:
                QMessageBox.warning(self, "Load Mesh Error", str(e))

    def _load_trajectory_dialog(self):
        last_dir = self.settings.value("last_dir_traj", str(Path.home()))
        json_path, _ = QFileDialog.getOpenFileName(self, "Load Trajectory JSON", last_dir, "JSON Files (*.json)")
        if json_path:
            try:
                self.vis_system.load_trajectory(json_path)
                self.settings.setValue("last_dir_traj", str(Path(json_path).parent))
                self._update_waypoint_table()
                self._update_3d_view()
                if self.vis_system.waypoints:
                    self._update_fpv(0)
            except Exception as e:
                import traceback
                traceback.print_exc()  # Prints full traceback to console
                QMessageBox.warning(self, "Load Trajectory Error", str(e))
                raise  # stops execution and shows traceback in console


    def _export_mission_dialog(self):
        last_dir = self.settings.value("last_dir_export", str(Path.cwd()))
        output_path, _ = QFileDialog.getSaveFileName(self, "Export Mission", str(Path(last_dir) / "mission_output.json"), "JSON Files (*.json)")
        if output_path:
            try:
                self.vis_system.export_mission(output_path)
                self.settings.setValue("last_dir_export", str(Path(output_path).parent))
            except Exception as e:
                QMessageBox.warning(self, "Export Error", str(e))

    # View updates
    def _update_waypoint_table(self):
        self.waypoint_table.setRowCount(len(self.vis_system.waypoints))
        for i, wp in enumerate(self.vis_system.waypoints):
            pos = wp['position']
            yaw = float(wp['rotation'][0])
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
            title = "FPV Preview (Placeholder)" if np.all(colormap == 128) else "FPV Preview"
            self.fpv_ax.set_title(title)
            self.fpv_canvas.draw()
        self.vis_system.render_waypoint_view(index, update_plot)

    def _update_3d_view(self):
        self.vis_system.render()
        if self._web_view is not None:
            html = self.vis_system.renderer.get_html()
            self._web_view.setHtml(html)
        else:
            # Fallback static Matplotlib view
            ax = self._mpl_ax
            ax.clear()
            mesh_data = self.vis_system.renderer.get_mesh_data()
            if mesh_data:
                ax.plot_trisurf(
                    mesh_data['vertices'][:, 0], mesh_data['vertices'][:, 1], mesh_data['vertices'][:, 2],
                    triangles=mesh_data['faces'], color='lightblue', linewidth=0.2, antialiased=True, alpha=0.9
                )
            route = self.vis_system.renderer.get_route_data()
            if route:
                ax.plot(route['x'], route['y'], route['z'], 'r-', label='Route')
                ax.legend(loc='upper right')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            self._mpl_canvas.draw()

    def _load_initial_data(self):
        #  load defaults from config 
        defaults = self.config.get('defaults', {})
        if defaults.get('default_cameras'):
            self.vis_system.load_default_cameras(defaults['default_cameras'])
        self._update_3d_view()


# -----------------------------
# Main application entry point
# -----------------------------
if __name__ == "__main__":
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)

    config = load_app_config("config.yml")
    vis = VisualizationSystem(config)
    window = ViewerApp(vis, config)
    window.show()
    sys.exit(app.exec_())
