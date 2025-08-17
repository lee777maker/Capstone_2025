
import pyrender
# File: visualization_system.py
import json
import numpy as np
import plotly.graph_objects as go
import open3d as o3d
from pathlib import Path
import yaml
import sys
from typing import List
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel, QTableWidget, QTableWidgetItem, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QCoreApplication
import plotly.io as pio
from helpers import (
    init, initialize_logging, load_camera_parameters, generate_camera_3d_thickness,
    show3d_plotly, render_scene, show2d, create_route_trace, init_renderer, shared
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class RenderThread(QThread):
    render_finished = pyqtSignal(np.ndarray)

    def __init__(self, origin, yaw, pitch):
        super().__init__()
        self.origin = origin
        self.yaw = yaw
        self.pitch = pitch

    def run(self):
        colormap, _ = render_scene(self.origin, self.yaw, self.pitch)
        self.render_finished.emit(colormap)

class SceneManager:
    def __init__(self, config_path: str = "config.yml"):
        initialize_logging(config_path, is_for_photo=False)
        init(config_path, is_for_photo=False)
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}. Using default config.")
            self.config = {'render_scale': 0.5, 'voxel_size': 0.2}
        self.mesh = None
        self.plotly_objs = []

    def load_mesh(self, mesh_path: str):
        init_renderer(scale=self.config.get('render_scale', 0.5), load_mesh=False)
        if 'scene' in shared and shared['scene']:
            for node in shared['scene'].nodes:
                if isinstance(node, pyrender.Node) and node.mesh:
                    vertices = node.mesh.primitives[0].positions
                    faces = node.mesh.primitives[0].indices
                    self.mesh = o3d.geometry.TriangleMesh()
                    self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
                    self.mesh.triangles = o3d.utility.Vector3iVector(faces)
                    print(f"Loaded mesh via init_renderer: {len(self.mesh.triangles)} triangles")
                    return
        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            print(f"[Warning] Mesh file not found: {mesh_path}. Continuing without mesh.")
            self.mesh = None
            return
        if mesh_path.suffix not in ['.ply', '.obj']:
            raise ValueError("Unsupported format. Use .ply or .obj")
        self.mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if not self.mesh.has_vertices():
            print(f"[Warning] Mesh is empty: {mesh_path}")
            self.mesh = None
            return
        voxel_size = self.config.get('voxel_size', 0.2)
        self.mesh = self.mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average
        )
        print(f"Loaded mesh: {len(self.mesh.triangles)} triangles")

    def add_ground_plane(self, size: float = 100, z: float = 0):
        x = np.linspace(-size, size, 10)
        y = np.linspace(-size, size, 10)
        x, y = np.meshgrid(x, y)
        z = np.full_like(x, z)
        ground_trace = go.Surface(x=x, y=y, z=z, colorscale='Greys', opacity=0.5, showscale=False)
        self.plotly_objs.append(ground_trace)

    def add_coordinate_axes(self, length: float = 10):
        axes = [
            go.Scatter3d(x=[0, length], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=4), name='X-axis'),
            go.Scatter3d(x=[0, 0], y=[0, length], z=[0, 0], mode='lines', line=dict(color='green', width=4), name='Y-axis'),
            go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, length], mode='lines', line=dict(color='blue', width=4), name='Z-axis')
        ]
        self.plotly_objs.extend(axes)

    def get_scene_objects(self):
        if not self.mesh or not self.mesh.has_vertices():
            return self.plotly_objs
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        colors = ['rgb(128,128,128)'] * len(vertices) if not self.mesh.has_vertex_colors() else [
            f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in np.asarray(self.mesh.vertex_colors)
        ]
        mesh_trace = go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
            vertexcolor=colors, opacity=1.0
        )
        return [mesh_trace] + self.plotly_objs

class WaypointParser:
    def __init__(self):
        self.waypoints = []
        self.order = []
        self.metadata = {}

    def parse_json(self, json_path: str):
        try:
            params = load_camera_parameters(json_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Waypoint JSON not found: {json_path}")
        self.metadata = {
            'num_cameras': params.get('num_cameras', 0),
            'scene_name': params.get('scene_name', ''),
            'has_route': params.get('has_route', False)
        }
        self.waypoints = [
            {'position': np.array(pos), 'rotation': np.array(rot)}
            for pos, rot in zip(params['positions'], params['rotations'])
        ]
        self.order = params.get('waypoint_order', list(range(len(self.waypoints))))
        print(f"Parsed {len(self.waypoints)} waypoints")

class PathVisualizer:
    def __init__(self):
        self.plotly_objs = []

    def visualize_trajectory(self, waypoints: List[dict], order: List[int], name: str = "Path"):
        params = {
            'positions': np.array([wp['position'] for wp in waypoints]),
            'rotations': np.array([wp['rotation'] for wp in waypoints]),
            'waypoint_order': order,
            'path_color_scale': 'Plotly3'
        }
        path_trace = create_route_trace(params)
        camera_traces = []
        for i, wp in enumerate(waypoints):
            cam_objs = generate_camera_3d_thickness(
                wp['position'], wp['rotation'][0], wp['rotation'][1], text=f"WP{i+1}", scale=0.5
            )
            camera_traces.extend(cam_objs)
        self.plotly_objs.extend([path_trace] + camera_traces)

    def visualize_default_cameras(self, camera_params: List[dict], name: str = "Default Path"):
        positions = np.array([params['origin'] for params in camera_params])
        n = len(positions)
        color_idx = np.linspace(0, 1, n)
        path_trace = go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            mode='lines+markers', line=dict(color=color_idx, colorscale='Plotly3', width=6),
            marker=dict(size=5, color=color_idx, colorscale='Plotly3', line=dict(width=1, color='white')),
            customdata=np.arange(n), name=name
        )
        camera_traces = [
            generate_camera_3d_thickness(params['origin'], params['yaw'], params['pitch'], text=f"Camera {i+1}", scale=0.5)
            for i, params in enumerate(camera_params)
        ]
        self.plotly_objs.extend([path_trace] + sum(camera_traces, []))

    def get_plotly_objects(self):
        return [obj for obj in self.plotly_objs if obj is not None]

class CameraController:
    def __init__(self):
        self.position = np.array([0, 0, 10], dtype=float)
        self.yaw = 0.0
        self.pitch = -20.0

    def update_position(self, dx: float, dy: float, dz: float):
        self.position += np.array([dx, dy, dz], dtype=float)

    def update_rotation(self, dyaw: float, dpitch: float):
        self.yaw = (self.yaw + dyaw) % 360
        self.pitch = np.clip(self.pitch + dpitch, -90, 90)

    def get_camera(self):
        return dict(eye=dict(x=self.position[0], y=self.position[1], z=self.position[2]), up=dict(x=0, y=0, z=1))

class DataExporter:
    def export_mission(self, waypoints: List[dict], order: List[int], output_path: str):
        def to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_serializable(item) for item in obj]
            elif isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            return obj

        mission = {
            'metadata': {'total_cameras': len(waypoints)},
            'cameras': [
                {'position': to_serializable(wp['position']), 'rotation': to_serializable(wp['rotation'])}
                for wp in waypoints
            ],
            'waypoint_order': to_serializable(order)
        }
        with open(output_path, 'w') as f:
            json.dump(mission, f, indent=2)
        print(f"Mission exported to {output_path}")

class PlotlyRenderer:
    def __init__(self):
        self.fig = go.Figure()
        self.camera_controller = CameraController()

    def add_objects(self, plotly_objs: List[go.Trace]):
        self.fig.data = []  # Clear existing traces
        for obj in plotly_objs:
            if obj is not None:
                self.fig.add_trace(obj)

    def update_layout(self):
        self.fig.update_layout(
            scene=dict(
                xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                camera=self.camera_controller.get_camera()
            ),
            showlegend=True
        )

    def get_html(self):
        return pio.to_html(self.fig, full_html=False)

class VisualizationSystem:
    def __init__(self, config_path: str = "config.yml"):
        self.scene_manager = SceneManager(config_path)
        self.trajectory_manager = None
        self.renderer = PlotlyRenderer()
        self.exporter = DataExporter()
        self.waypoints = []
        self.order = []
        self.render_thread = None

    def load_scene(self, mesh_path: str):
        self.scene_manager.load_mesh(mesh_path)
        self.scene_manager.add_ground_plane()
        self.scene_manager.add_coordinate_axes()

    def load_trajectory(self, json_path: str):
        parser = WaypointParser()
        parser.parse_json(json_path)
        self.waypoints = parser.waypoints
        self.order = parser.order
        visualizer = PathVisualizer()
        visualizer.visualize_trajectory(parser.waypoints, parser.order)
        self.trajectory_manager = (parser, visualizer)

    def load_default_cameras(self, camera_params: List[dict]):
        visualizer = PathVisualizer()
        visualizer.visualize_default_cameras(camera_params)
        self.trajectory_manager = (None, visualizer)

    def compare_trajectories(self, json_paths: List[str]):
        colorscales = ['Plotly3', 'Viridis', 'Plasma']
        for i, json_path in enumerate(json_paths):
            parser = WaypointParser()
            parser.parse_json(json_path)
            visualizer = PathVisualizer()
            visualizer.visualize_trajectory(parser.waypoints, parser.order, name=f"Path {i+1}")
            self.renderer.add_objects(visualizer.get_plotly_objects())

    def render(self):
        scene_objects = self.scene_manager.get_scene_objects()
        trajectory_objects = self.trajectory_manager[1].get_plotly_objects() if self.trajectory_manager else []
        self.renderer.add_objects(scene_objects + trajectory_objects)
        self.renderer.update_layout()

    def render_waypoint_view(self, waypoint_index: int, callback):
        if not self.trajectory_manager or not self.trajectory_manager[0]:
            print("No JSON trajectory loaded. Showing placeholder FPV.")
            width, height = 640, 480
            colormap = np.full((height, width, 3), 128, dtype=np.uint8)
            callback(colormap)
            return None
        parser, _ = self.trajectory_manager
        if waypoint_index >= len(parser.waypoints):
            print(f"Waypoint index {waypoint_index} out of range. Showing placeholder FPV.")
            width, height = 640, 480
            colormap = np.full((height, width, 3), 128, dtype=np.uint8)
            callback(colormap)
            return None
        wp = parser.waypoints[waypoint_index]
        self.render_thread = RenderThread(wp['position'], wp['rotation'][0], wp['rotation'][1])
        self.render_thread.render_finished.connect(callback)
        self.render_thread.finished.connect(self.render_thread.deleteLater)
        self.render_thread.start()
        return self.render_thread

    def export_mission(self, output_path: str):
        if self.trajectory_manager and self.trajectory_manager[0]:
            parser, _ = self.trajectory_manager
            self.exporter.export_mission(parser.waypoints, parser.order, output_path)

class ViewerApp(QMainWindow):
    def __init__(self, vis_system):
        super().__init__()
        self.vis_system = vis_system
        self.init_ui()
        self.load_initial_data()

    def init_ui(self):
        self.setWindowTitle("UAV Flight Planning")
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        main_layout.addWidget(left_widget, stretch=2)

        from PyQt5.QtWebEngineWidgets import QWebEngineView
        self.plotly_view = QWebEngineView()
        left_layout.addWidget(self.plotly_view)

        control_widget = QWidget()
        control_layout = QVBoxLayout()
        control_widget.setLayout(control_layout)
        left_layout.addWidget(control_widget)

        control_layout.addWidget(QLabel("Camera Controls"))
        self.add_slider(control_layout, "X", -100, 100, 0, self.update_camera_x)
        self.add_slider(control_layout, "Y", -100, 100, 0, self.update_camera_y)
        self.add_slider(control_layout, "Z", -100, 100, 10, self.update_camera_z)
        self.add_slider(control_layout, "Yaw", -180, 180, 0, self.update_camera_yaw)
        self.add_slider(control_layout, "Pitch", -90, 90, -20, self.update_camera_pitch)

        control_layout.addWidget(QLabel("Scene and Trajectory"))
        self.load_mesh_btn = QPushButton("Load Mesh")
        self.load_mesh_btn.clicked.connect(self.load_mesh)
        control_layout.addWidget(self.load_mesh_btn)

        self.load_traj_btn = QPushButton("Load Trajectory")
        self.load_traj_btn.clicked.connect(self.load_trajectory)
        control_layout.addWidget(self.load_traj_btn)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(right_widget, stretch=1)

        self.waypoint_table = QTableWidget()
        self.waypoint_table.setColumnCount(5)
        self.waypoint_table.setHorizontalHeaderLabels(["Index", "X", "Y", "Z", "Yaw"])
        self.waypoint_table.cellClicked.connect(self.waypoint_selected)
        right_layout.addWidget(self.waypoint_table)

        self.export_btn = QPushButton("Export Mission")
        self.export_btn.clicked.connect(self.export_mission)
        right_layout.addWidget(self.export_btn)

        self.fpv_canvas = FigureCanvas(plt.Figure(figsize=(4, 3)))
        right_layout.addWidget(self.fpv_canvas)
        self.fpv_ax = self.fpv_canvas.figure.add_subplot(111)
        self.fpv_ax.axis('off')

    def add_slider(self, layout, name, min_val, max_val, default, callback):
        slider_layout = QHBoxLayout()
        label = QLabel(f"{name}: {default}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.valueChanged.connect(lambda val: callback(val, label))
        slider_layout.addWidget(QLabel(name))
        slider_layout.addWidget(slider)
        slider_layout.addWidget(label)
        layout.addLayout(slider_layout)
        return slider

    def update_camera_x(self, val, label):
        self.vis_system.renderer.camera_controller.update_position(val - self.vis_system.renderer.camera_controller.position[0], 0, 0)
        label.setText(f"X: {val}")
        self.update_3d_view()

    def update_camera_y(self, val, label):
        self.vis_system.renderer.camera_controller.update_position(0, val - self.vis_system.renderer.camera_controller.position[1], 0)
        label.setText(f"Y: {val}")
        self.update_3d_view()

    def update_camera_z(self, val, label):
        self.vis_system.renderer.camera_controller.update_position(0, 0, val - self.vis_system.renderer.camera_controller.position[2])
        label.setText(f"Z: {val}")
        self.update_3d_view()

    def update_camera_yaw(self, val, label):
        self.vis_system.renderer.camera_controller.update_rotation(val - self.vis_system.renderer.camera_controller.yaw, 0)
        label.setText(f"Yaw: {val}")
        self.update_3d_view()

    def update_camera_pitch(self, val, label):
        self.vis_system.renderer.camera_controller.update_rotation(0, val - self.vis_system.renderer.camera_controller.pitch)
        label.setText(f"Pitch: {val}")
        self.update_3d_view()

    def load_mesh(self):
        mesh_path, _ = QFileDialog.getOpenFileName(self, "Load Mesh File", "", "Mesh Files (*.ply *.obj)")
        if mesh_path:
            self.vis_system.load_scene(mesh_path)
            self.update_3d_view()

    def load_trajectory(self):
        json_path, _ = QFileDialog.getOpenFileName(self, "Load Trajectory JSON", "", "JSON Files (*.json)")
        if json_path:
            self.vis_system.load_trajectory(json_path)
            self.update_waypoint_table()
            self.update_3d_view()
            if self.vis_system.waypoints:
                self.update_fpv(0)

    def update_waypoint_table(self):
        self.waypoint_table.setRowCount(len(self.vis_system.waypoints))
        for i, wp in enumerate(self.vis_system.waypoints):
            pos = wp['position']
            yaw = wp['rotation'][0]
            self.waypoint_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.waypoint_table.setItem(i, 1, QTableWidgetItem(f"{pos[0]:.2f}"))
            self.waypoint_table.setItem(i, 2, QTableWidgetItem(f"{pos[1]:.2f}"))
            self.waypoint_table.setItem(i, 3, QTableWidgetItem(f"{pos[2]:.2f}"))
            self.waypoint_table.setItem(i, 4, QTableWidgetItem(f"{yaw:.2f}"))

    def waypoint_selected(self, row, _):
        self.update_fpv(row)

    def update_fpv(self, index):
        def update_plot(colormap):
            self.fpv_ax.clear()
            self.fpv_ax.imshow(colormap)
            self.fpv_ax.axis('off')
            self.fpv_ax.set_title("FPV Preview (Placeholder)" if np.all(colormap == 128) else "FPV Preview")
            self.fpv_canvas.draw()
        self.vis_system.render_waypoint_view(index, update_plot)

    def export_mission(self):
        output_path, _ = QFileDialog.getSaveFileName(self, "Export Mission", "mission_output.json", "JSON Files (*.json)")
        if output_path:
            self.vis_system.export_mission(output_path)

    def update_3d_view(self):
        self.vis_system.render()
        self.plotly_view.setHtml(self.vis_system.renderer.get_html())

    def load_initial_data(self):
        config_path = "config.yml"
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}. Creating default config.")
            config = {
                'scene_id': 2,
                'scene_name': 'Kiri_Vehera_SriLanka',
                'dataset_path': '.',
                'voxel_size': 0.2,
                'render_scale': 0.5,
                'load_mesh': False,
                'default_cameras': [
                    {'origin': [0.0, -10.0, 3.0], 'yaw': 90.0, 'pitch': -20},
                    {'origin': [3.8, -1.24, 2.0], 'yaw': 162.0, 'pitch': -20},
                    {'origin': [5.88, 8.09, 3.0], 'yaw': -126.0, 'pitch': -20},
                    {'origin': [-2.35, 3.24, 2.0], 'yaw': -54.0, 'pitch': -20},
                    {'origin': [-9.51, -3.09, 3.0], 'yaw': 18.0, 'pitch': -20}
                ]
            }
            try:
                Path(config_path).parent.mkdir(parents=True, exist_ok=True)
                with open(config_path, 'w') as f:
                    yaml.dump(config, f)
            except Exception as e:
                print(f"Failed to create config file: {e}")
                return

        # Load default cameras but don't load mesh or trajectory automatically
        self.vis_system.load_default_cameras(config['default_cameras'])
        self.update_3d_view()

if __name__ == "__main__":
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    vis = VisualizationSystem()
    window = ViewerApp(vis)
    window.show()
    sys.exit(app.exec_())