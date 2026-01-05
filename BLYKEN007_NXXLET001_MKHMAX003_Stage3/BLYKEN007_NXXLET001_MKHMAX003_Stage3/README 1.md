# Visualizing and Measuring Optimal UAV Flight Paths

## Description
This project enables visualization of 3D models (PLY/OBJ) and UAV camera flight paths using Python. It supports loading, rendering, and interacting with 3D meshes and flight trajectories, with a Qt-based GUI for an integrated experience. Built with Plotly, Open3D, Trimesh, Tkinter, and PyQt5.

## Requirements
- Python 3.11
- Dependencies: `pip install plotly pywebview open3d trimesh numpy tkinter PyQt5 matplotlib pyyaml pyrender`
- Note: Open3D and Trimesh may require additional setup for 3D rendering.

## Installation
1. Clone the repository:
   ```
   git clone https://gitlab.cs.uct.ac.za/capstone-20255/visualizing-and-measuring-optimal-uav-flight-paths.git
   cd visualizing-and-measuring-optimal-uav-flight-paths
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure a `config.yml` file is present (auto-generated with defaults on first run).

## Usage
### `file_manager.py`
Utility for loading 3D models (PLY/OBJ) and JSON flight paths. Handles file copying and MTL/texture management for OBJ files. Used by other scripts.

### `flight_path_display.py`
Visualize flight paths standalone:
```
python flight_path_display.py
```
- Prompts for a JSON flight path file.
- Displays camera positions and routes in a 3D Plotly view via a webview window.

### `obj_model_display.py`
Visualize 3D models with optional flight paths:
```
python obj_model_display.py
```
- Prompts for a 3D model (PLY/OBJ) and optional JSON flight path.
- Shows mesh, cameras, and routes in a 3D Plotly view.
- Simplifies complex meshes for performance.

### `visualization_system.py`
Main Qt-based GUI application:
```
python visualization_system.py
```
- Features buttons to load meshes (OBJ/PLY/STL) and trajectories.
- Displays interactive 3D Plotly view, waypoint table, and first-person previews.
- Supports mission export to JSON.
- Integrates `file_manager.py`; future integration of other scripts planned.

## Configuration
- Edit `config.yml` to customize scene settings, UI layout, and default cameras.
- The file is auto-generated with defaults on first run.

## Collaborate with your team
1. Max Mkhabela (MKHMAX0003)
2. Kenneth Baloyi (BLYKEN007)
3. Lethabo Neo (NXXLET001)

## Support
For issues or questions, open an issue on the GitLab project.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Project Status
Actively developed as part of a capstone project. `flight_path_display.py` and `obj_model_display.py` are exploratory and will be integrated into `visualization_system.py` in future releases.
