# UAV Path Planning Resources

UAV photogrammetry involves capturing overlapping aerial images to reconstruct 3D models of structures. Each waypoint in a flight path represents a specific camera position and orientation optimized for comprehensive coverage. The system uses XYZ coordinates (in meters) for camera positions, with yaw and pitch angles (in degrees) defining camera orientation. Roll angle is typically not used in aerial photography as UAVs maintain level flight for stability.

The coordinate system follows standard conventions: X and Y represent horizontal positioning while Z indicates altitude. Yaw controls horizontal camera rotation (0° points north, 90° east), and pitch controls vertical tilt (negative values point downward toward the target structure).

The starter code uses PyRender for image rendering and Plotly for 3D visualization to demonstrate core concepts. PyRender provides image generation capabilities at reasonable resolutions, which can be useful for first-person view previews from waypoint positions. Alternative rendering libraries such as Open3D's visualization module, VTK, or OpenGL-based solutions may also be considered based on specific application requirements and integration needs. Plotly demonstrates how to visualize trajectories, waypoint markers, camera frustums, and 3D meshes with interactive features. The provided examples illustrate 3D scene composition techniques including camera transformations, route visualization, and mesh rendering. When developing standalone desktop applications, the web-based architecture of Plotly may present integration considerations with traditional GUI frameworks that require evaluation.

The provided code can be viewed as a conceptual foundation that illustrates fundamental approaches rather than a fixed technology stack. The visualization techniques demonstrated with Plotly may be adapted to other 3D graphics libraries, while the rendering concepts shown with PyRender can be applied across different image generation approaches. Key concepts include representing camera positions and orientations in 3D space, generating trajectory curves through interpolation, and designing user interfaces that balance real-time visualization with background processing requirements. Alternative approaches include using Open3D for visualization and rendering, PyQt with OpenGL widgets for desktop integration, or web-based solutions using libraries like Three.js.

## Resources
This directory contains 3D models and flight trajectory data for the UAV path planning visualization system, supporting photogrammetric reconstruction of cultural heritage sites.

```
├── models/                 # 3D reconstruction models
├── waypoints/              # Flight path data
├── config.yml              # Scene configuration
├── helpers.py              # Utility functions
└── shared.py               # Shared state management
```

## Waypoint Data

JSON structure for flight trajectories:

```json
{
  "metadata": {
    "version": "1.0",
    "scene": "scene_name",
    "algorithm": "algorithm_type"
  },
  "cameras": [
    {
      "position": [x, y, z],
      "rotation": [yaw, pitch]
    }
  ],
  "waypoint_order": [0, 1, 2, ...]
}
```