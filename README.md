# UAV Flight Trajectory Visualization System

A comprehensive 3D visualization system for Unmanned Aerial Vehicle (UAV) flight trajectories, designed for photogrammetric surveying of cultural heritage sites and structural inspection missions.

![UAV Visualization System](Screenshot.png)

## Features

### Core Features
- **3D Model Loading**: Support for OBJ, STL, and PLY formats with automatic camera positioning
- **Trajectory Visualization**: Render flight paths with linear and spline interpolation
- **First-Person View (FPV)**: Realistic UAV camera simulation with adjustable parameters
- **Interactive Waypoint Editing**: Add, modify, and delete waypoints with real-time updates
- **Flight Simulation**: Animate UAV movement along trajectories with speed control
- **Multi-Trajectory Comparison**: Compare multiple flight paths with quantitative metrics
- **Video Recording**: Capture simulation sessions to MP4 files
- **Real-time Metrics**: Path length, duration, turning angles, and efficiency ratings

### Advanced Features
- **Keyboard Controls**: Intuitive WASD/QE and arrow key navigation
- **Collapsible UI**: Space-efficient interface organization
- **Multi-threaded Rendering**: Non-blocking UI during intensive operations
- **Batch Operations**: Efficient handling of hundreds of waypoints
- **Flexible Camera Controls**: Adjustable FPV distance and zoom

## Installation

### System Requirements
- **Operating System**: Windows 10/11 or Linux Ubuntu 18.04+
- **Python**: Version 3.11 or newer
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Graphics**: Dedicated GPU with OpenGL 3.3+ support

### Dependencies Installation
```bash
pip install open3d trimesh numpy scipy opencv-python pillow glfw
