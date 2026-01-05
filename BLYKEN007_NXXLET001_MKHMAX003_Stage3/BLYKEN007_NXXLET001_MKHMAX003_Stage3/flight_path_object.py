# flight_path_objects.py
from typing import List, Tuple
import numpy as np
import plotly.graph_objects as go

# use helpers as a black box
from helpers import (
    load_camera_parameters,
    generate_camera_3d_thickness,
    create_route_trace,
)

class FlightPathObjectsBuilder:
    """
    Compose helpers.py to create Plotly traces for camera markers and path/route
    without editing helpers.py.
    """

    def __init__(self, camera_scale: float = 0.5, label_cameras: bool = True):
        self.camera_scale = camera_scale
        self.label_cameras = label_cameras

    def build_from_json(
        self,
        json_path: str,
        use_route_if_available: bool = True,
        draw_sequential_path_if_no_route: bool = True,
        path_colorscale: str = "Viridis",
        path_line_width: int = 6,
        path_marker_size: int = 1,
    ) -> Tuple[List[go.Trace], np.ndarray, dict]:
        """
        Returns (plotly_objs, positions, params)
        """
        params = load_camera_parameters(json_path)
        plotly_objs, positions = self.build_from_params(
            params=params,
            use_route_if_available=use_route_if_available,
            draw_sequential_path_if_no_route=draw_sequential_path_if_no_route,
            path_colorscale=path_colorscale,
            path_line_width=path_line_width,
            path_marker_size=path_marker_size,
        )
        return plotly_objs, positions, params

    def build_from_params(
        self,
        params: dict,
        use_route_if_available: bool = True,
        draw_sequential_path_if_no_route: bool = True,
        path_colorscale: str = "Viridis",
        path_line_width: int = 6,
        path_marker_size: int = 1,
    ) -> Tuple[List[go.Trace], np.ndarray]:
        """
        Build camera markers + path from an already-parsed helpers params dict.
        Returns (plotly_objs, positions)
        """
        plotly_objs: List[go.Trace] = []
        positions_list = []

        # Camera markers
        for i, (pos, rot) in enumerate(zip(params["positions"], params["rotations"])):
            yaw = float(rot[0])
            pitch = float(rot[1])
            label = f"Camera {i+1}" if self.label_cameras else None

            cam_traces = generate_camera_3d_thickness(
                origin=pos,
                yaw=yaw,
                pitch=pitch,
                text=label,
                scale=self.camera_scale,
            )
            plotly_objs.extend(cam_traces)
            positions_list.append(pos)

        positions = np.array(positions_list, dtype=float)

        # Route (if waypoint_order present) or fallback sequential path
        added_path = False
        if use_route_if_available and params.get("has_route", False):
            route_trace = create_route_trace(params)
            if route_trace is not None:
                plotly_objs.append(route_trace)
                added_path = True

        if not added_path and draw_sequential_path_if_no_route and len(positions) >= 2:
            n = len(positions)
            color_idx = np.linspace(0.0, 1.0, n)
            path_trace = go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="lines+markers",
                line=dict(
                    color=color_idx,
                    colorscale=path_colorscale,
                    cmin=0,
                    cmax=1,
                    showscale=True,
                    width=path_line_width,
                ),
                marker=dict(
                    size=path_marker_size,
                    color=color_idx,
                    colorscale=path_colorscale,
                    cmin=0,
                    cmax=1,
                    showscale=False,
                    line=dict(width=2, color="white"),
                    symbol="circle",
                ),
                name="Camera Path",
            )
            plotly_objs.append(path_trace)

        return plotly_objs, positions
