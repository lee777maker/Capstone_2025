import os
import tempfile
import numpy as np
from pathlib import Path
from tkinter import filedialog
from plotly.offline import plot
import plotly.graph_objects as go
import webview

from helpers import *
from file_manager import File_Manager

try:
    import open3d as o3d
    HAS_O3D = True
except:
    HAS_O3D = False
    print("Warning: Open3D not available")

try:
    import trimesh
    HAS_TRIMESH = True
except:
    HAS_TRIMESH = False
    print("Warning: Trimesh not available")


class Display_OBJ_Model:
    def __init__(self):
        self.file_manager = File_Manager()
        self.mesh_path = None
        
    def ensure_initialized(self):
        # initialize helper
        if 'config' not in shared:
            init("config.yml", is_for_photo=True)
        return shared['config']
    
    def load_and_prepare_mesh(self):
        # Get mesh file from user
        mesh_path = self.file_manager.load3DModel()
        if not mesh_path:
            return None, None, None
        
        self.mesh_path = mesh_path
        config = self.ensure_initialized()
        
        # Determine if it's PLY or OBJ
        mesh_path_obj = Path(mesh_path)
        ext = mesh_path_obj.suffix.lower()
        
        print(f"Loading mesh from: {mesh_path}")
        
        if ext == '.ply':
            # Load PLY like in tutorial
            if HAS_O3D:
                msh = o3d.io.read_triangle_mesh(str(mesh_path))
                
                # Extract vertices and faces
                vertices = np.asarray(msh.vertices)
                triangles = np.asarray(msh.triangles)
                
                print(f"Loaded PLY - Vertices: {len(vertices)}, Triangles: {len(triangles)}")
                print(f"Has vertex colors: {msh.has_vertex_colors()}")
                
                # Get vertex colors if available
                vertex_colors = None
                if msh.has_vertex_colors():
                    vertex_colors = np.asarray(msh.vertex_colors)
                    print(f"Vertex colors shape: {vertex_colors.shape}")
                    print(f"Vertex colors range: {vertex_colors.min()} to {vertex_colors.max()}")
                
                return msh, vertices, triangles, vertex_colors
            else:
                print("Error: Open3D required for PLY files")
                return None, None, None, None
                
        else:  # OBJ or other format
            # Try to load with Open3D first
            if HAS_O3D:
                msh = o3d.io.read_triangle_mesh(str(mesh_path))
                
                # Compute normals if missing
                if not msh.has_vertex_normals():
                    msh.compute_vertex_normals()
                
                vertices = np.asarray(msh.vertices)
                triangles = np.asarray(msh.triangles)
                
                print(f"Loaded {ext.upper()} with Open3D - Vertices: {len(vertices)}, Triangles: {len(triangles)}")
                print(f"Has vertex colors: {msh.has_vertex_colors()}")
                
                vertex_colors = None
                if msh.has_vertex_colors():
                    vertex_colors = np.asarray(msh.vertex_colors)
                    # Normalize to 0-1 range if needed
                    if vertex_colors.max() > 1.01:
                        vertex_colors = vertex_colors / 255.0
                
                return msh, vertices, triangles, vertex_colors
                
            elif HAS_TRIMESH:
                # Fallback to trimesh
                print("Using trimesh as fallback")
                tm = trimesh.load(str(mesh_path), force='mesh')
                
                vertices = np.asarray(tm.vertices)
                triangles = np.asarray(tm.faces)
                
                print(f"Loaded {ext.upper()} with Trimesh - Vertices: {len(vertices)}, Triangles: {len(triangles)}")
                
                vertex_colors = None
                if hasattr(tm.visual, 'vertex_colors'):
                    vc = tm.visual.vertex_colors
                    if vc is not None:
                        vertex_colors = np.asarray(vc)
                        if vertex_colors.shape[1] == 4:  # RGBA
                            vertex_colors = vertex_colors[:, :3]
                        if vertex_colors.max() > 1.01:
                            vertex_colors = vertex_colors / 255.0
                
                # Convert to Open3D mesh for consistency
                if HAS_O3D:
                    msh = o3d.geometry.TriangleMesh()
                    msh.vertices = o3d.utility.Vector3dVector(vertices)
                    msh.triangles = o3d.utility.Vector3iVector(triangles)
                    if vertex_colors is not None:
                        msh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                    msh.compute_vertex_normals()
                else:
                    msh = None
                
                return msh, vertices, triangles, vertex_colors
            else:
                print("Error: Need Open3D or Trimesh to load mesh files")
                return None, None, None, None
    
    def simplify_mesh_for_display(self, msh, vertices, triangles, target_triangles=50000):
        # Simplify mesh if too complex
        if len(triangles) <= target_triangles:
            return msh, vertices, triangles
        
        print(f"Simplifying mesh from {len(triangles)} to ~{target_triangles} triangles...")
        
        if HAS_O3D and msh is not None:
            # Use Open3D's simplification 
            voxel_size = 0.05  # Start with small voxel size
            
            simplified = msh.simplify_vertex_clustering(
                voxel_size=voxel_size,
                contraction=o3d.geometry.SimplificationContraction.Average
            )
            
            # If too complex, increase voxel size
            while len(simplified.triangles) > target_triangles and voxel_size < 1.0:
                voxel_size *= 1.5
                simplified = msh.simplify_vertex_clustering(
                    voxel_size=voxel_size,
                    contraction=o3d.geometry.SimplificationContraction.Average
                )
            
            print(f"Simplified to {len(simplified.triangles)} triangles")
            
            vertices = np.asarray(simplified.vertices)
            triangles = np.asarray(simplified.triangles)
            
            return simplified, vertices, triangles
        
        return msh, vertices, triangles
    
    def create_plotly_mesh(self, vertices, triangles, vertex_colors=None):
        # Create Plotly mesh
        print("Creating Plotly mesh visualization...")
        
        # Prepare vertex color strings
        if vertex_colors is not None and len(vertex_colors) == len(vertices):
            # Ensure colors are in 0-1 range
            vc = np.asarray(vertex_colors)
            if vc.max() > 1.01:
                vc = vc / 255.0
            
            # Convert to RGB strings (like tutorial does)
            vertex_color_strings = ['rgb({},{},{})'.format(
                int(c[0]*255), int(c[1]*255), int(c[2]*255)
            ) for c in vc]
            
            print(f"Using {len(vertex_color_strings)} vertex colors")
            
        else:
            # Generate colors based on height 
            print("No vertex colors found, generating based on geometry...")
            
            # Use the helper function from the tutorial approach
            colors_generated = apply_plasma_colormap(vertices)
            
            vertex_color_strings = ['rgb({},{},{})'.format(
                int(c[0]*255), int(c[1]*255), int(c[2]*255)
            ) for c in colors_generated]
            
            print(f"Generated {len(vertex_color_strings)} colors using plasma colormap")
        
        # Create mesh trace (following tutorial's exact approach)
        mesh_trace = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            vertexcolor=vertex_color_strings,
            opacity=1.0,
            lighting=dict(
                ambient=0.7,
                diffuse=0.8,
                specular=0.1,
                roughness=0.5,
                fresnel=0.2
            ),
            lightposition=dict(x=100, y=100, z=100),
            flatshading=False,
            hoverinfo='skip',
            showscale=False,
            name='3D Mesh'
        )
        
        return mesh_trace
    
    def display_obj(self):
        """Main display function following tutorial's approach"""
        # Initialize helpers if needed
        config = self.ensure_initialized()
        
        # Load and prepare mesh
        msh, vertices, triangles, vertex_colors = self.load_and_prepare_mesh()
        
        if vertices is None or triangles is None:
            print("Failed to load mesh")
            return None
        
        # Simplify if needed
        msh, vertices, triangles = self.simplify_mesh_for_display(msh, vertices, triangles)
        
        # Reget vertex colors after simplification if we have the mesh
        if msh is not None and HAS_O3D and msh.has_vertex_colors():
            vertex_colors = np.asarray(msh.vertex_colors)
        else:
            vertex_colors = None
        
        # Create Plotly mesh
        mesh_trace = self.create_plotly_mesh(vertices, triangles, vertex_colors)
        plotly_objs = [mesh_trace]
        
        # Load camera waypoints
        cam_json = filedialog.askopenfilename(
            title="Optional: Select camera waypoints JSON (Cancel to skip)",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*")]
        )
        
        if cam_json:
            try:
                # Use helper functions to load camera parameters
                params = load_camera_parameters(cam_json)
                print(f"Loading {params['num_cameras']} camera waypoints...")
                
                # Add camera markers using helper function
                for i in range(params['num_cameras']):
                    origin = params['positions'][i]
                    yaw = params['rotations'][i, 0]
                    pitch = params['rotations'][i, 1]
                    
                    # Use the helper function to generate camera
                    cam_objs = generate_camera_3d_thickness(
                        origin, yaw, pitch,
                        text=f"Camera {i+1}",
                        scale=0.5
                    )
                    plotly_objs.extend(cam_objs)
                
                # Add route trace if available (using helper function)
                if params.get('has_route', False):
                    route_trace = create_route_trace(params)
                    if route_trace is not None:
                        plotly_objs.append(route_trace)
                        print("Added route visualization")
                        
            except Exception as e:
                print(f"Could not load camera waypoints: {e}")
        
        # Create figure using helper function (exactly like tutorial)
        fig = show3d_plotly(plotly_objs, ret=True)
        
        # Save to HTML and display
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
            plot(fig, filename=tmpfile.name, auto_open=False)
            html_file = tmpfile.name
        
        print("Opening visualization in browser window...")
        webview.create_window("3D Model Viewer", html_file, width=1200, height=900)
        webview.start()
        
        return fig
    
    def display_with_renderer(self):
        config = self.ensure_initialized()
        
        # Load mesh
        mesh_path = self.file_manager.load3DModel()
        if not mesh_path:
            return None
        
        # Determine if we need load_mesh=True (for OBJ) or False (for PLY)
        ext = Path(mesh_path).suffix.lower()
        load_mesh = (ext == '.obj')
        
        try:
            # Initialize renderer (like tutorial does)
            init_renderer(scale=0.5, load_mesh=load_mesh)
            print("Renderer initialized successfully")
            
            return True
            
        except Exception as e:
            print(f"Could not initialize renderer: {e}")
            return False


if __name__ == "__main__":
    # Create display object
    display = Display_OBJ_Model()
    
    # Run the display
    fig = display.display_obj()
    
    if fig is None:
        print("Display failed")
    else:
        print("Display completed successfully")