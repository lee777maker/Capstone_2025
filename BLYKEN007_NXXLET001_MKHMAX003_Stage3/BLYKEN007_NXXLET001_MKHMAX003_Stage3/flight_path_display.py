from file_manager import File_Manager
from helpers import *
import tkinter as tk
import webview
import tempfile
from plotly.offline import plot
from flight_path_object import FlightPathObjectsBuilder

class Display_Flight_Path:
    def __init__(self):
        self.json_path = None

    def get_filepath(self):
        file_manager = File_Manager()
        self.json_path = file_manager.loadFlightPaths()


    def display_flight(self):
        self.get_filepath()  # sets self.json_path
        builder = FlightPathObjectsBuilder(camera_scale=0.5, label_cameras=True)

        plotly_objs, positions, params = builder.build_from_json(
            self.json_path,
            use_route_if_available=True,                 # use helpers' TSP route when present
            draw_sequential_path_if_no_route=True,       # fallback line if no route
            path_colorscale="Viridis",
            path_line_width=6,
            path_marker_size=1,
        )

        fig = show3d_plotly(plotly_objs, ret=True)
        self.show_plotly_in_tkinter(fig)
        return plotly_objs, positions
    

    def show_plotly_in_tkinter(self, fig):
        # Save Plotly figure to a temporary HTML file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
            plot(fig, filename=tmpfile.name, auto_open=False)
            html_file = tmpfile.name

        # Open in a pywebview window (cross-platform)
        webview.create_window("3D Flight Path", html_file, width=1000, height=800)
        webview.start()


display = Display_Flight_Path()
plot_objs, cam_positions = display.display_flight()