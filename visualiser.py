# uav_visualiser_fixed.py
import tkinter as tk
import threading
import queue
import time
import numpy as np
import open3d as o3d
from PIL import Image, ImageTk

class ThreadSafeRenderer:
    """Open3D renderer that safely communicates with main thread"""
    def __init__(self, width, height, name):
        self.width = width
        self.height = height
        self.name = name
        self.vis = None
        self.geometries = []
        self.frame_queue = queue.Queue(maxsize=1)
        self.command_queue = queue.Queue()
        self.running = False
        
    def initialize(self):
        """Initialize on main thread once"""
        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name=self.name,
                width=self.width,
                height=self.height,
                visible=True  # Critical: make visible
            )
            return True
        except Exception as e:
            print(f"Failed to initialize {self.name}: {e}")
            return False
    
    def update_geometries(self, geometries):
        """Thread-safe geometry update"""
        self.command_queue.put(('update_geometries', geometries))
    
    def render_frame(self):
        """Render and return frame - called from main thread"""
        try:
            if self.vis:
                self.vis.poll_events()
                self.vis.update_renderer()
                img = self.vis.capture_screen_float_buffer(do_render=True)
                if img:
                    arr = (np.asarray(img) * 255).astype(np.uint8)
                    return arr
        except Exception as e:
            print(f"Render error: {e}")
        return None
    
    def process_commands(self):
        """Process pending commands - call from main thread"""
        try:
            while True:
                cmd, data = self.command_queue.get_nowait()
                if cmd == 'update_geometries' and self.vis:
                    self.vis.clear_geometries()
                    for geom in data:
                        if geom:
                            self.vis.add_geometry(geom, reset_bounding_box=False)
        except queue.Empty:
            pass

class UAVVisualizerFixed:
    def __init__(self, root):
        self.root = root
        self.root.title("UAV Visualizer - Fixed")
        self.root.geometry("1200x800")
        
        # Initialize renderers after UI is set up
        self.root.after(100, self.initialize_renderers)
        
        self.setup_ui()
        self.setup_bindings()
        
    def setup_ui(self):
        # Main canvas for 3D view
        self.canvas = tk.Canvas(self.root, bg='black', width=800, height=600)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = tk.Frame(self.root, width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        tk.Button(control_frame, text="Load Model", command=self.load_model).pack(pady=5)
        tk.Button(control_frame, text="Load Trajectory", command=self.load_trajectory).pack(pady=5)
        
        # FPV display
        self.fpv_label = tk.Label(control_frame, text="FPV View", bg='black', 
                                width=40, height=15)
        self.fpv_label.pack(pady=5)
        
    def setup_bindings(self):
        self.canvas.bind("<Configure>", self.on_resize)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def initialize_renderers(self):
        """Initialize renderers once on main thread"""
        canvas_width = self.canvas.winfo_width() or 800
        canvas_height = self.canvas.winfo_height() or 600
        
        self.main_renderer = ThreadSafeRenderer(canvas_width, canvas_height, "MainView")
        self.fpv_renderer = ThreadSafeRenderer(320, 240, "FPVView")
        
        if self.main_renderer.initialize() and self.fpv_renderer.initialize():
            self.start_rendering_loop()
        else:
            print("Failed to initialize renderers")
            
    def start_rendering_loop(self):
        """Start the main rendering loop"""
        self.rendering = True
        self.render_loop()
        
    def render_loop(self):
        """Main rendering loop - called ~30fps"""
        if not self.rendering:
            return
            
        try:
            # Process pending commands
            self.main_renderer.process_commands()
            self.fpv_renderer.process_commands()
            
            # Render frames
            main_frame = self.main_renderer.render_frame()
            fpv_frame = self.fpv_renderer.render_frame()
            
            # Update displays
            if main_frame is not None:
                self.update_canvas_display(main_frame)
            if fpv_frame is not None:
                self.update_fpv_display(fpv_frame)
                
        except Exception as e:
            print(f"Render loop error: {e}")
            
        # Schedule next frame
        self.root.after(33, self.render_loop)  # ~30fps
        
    def update_canvas_display(self, frame):
        """Update main canvas display"""
        try:
            img = Image.fromarray(frame)
            img = img.resize((self.canvas.winfo_width(), self.canvas.winfo_height()), 
                           Image.LANCZOS)
            self.canvas_image = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_image)
        except Exception as e:
            print(f"Canvas update error: {e}")
            
    def update_fpv_display(self, frame):
        """Update FPV display"""
        try:
            img = Image.fromarray(frame)
            img = img.resize((300, 200), Image.LANCZOS)
            self.fpv_image = ImageTk.PhotoImage(img)
            self.fpv_label.configure(image=self.fpv_image)
        except Exception as e:
            print(f"FPV update error: {e}")
            
    def on_resize(self, event):
        """Handle resize - don't recreate renderers, just update viewport"""
        # Open3D should handle viewport changes automatically
        pass
        
    def load_model(self):
        """Load 3D model - run in background thread"""
        def load_task():
            # Simulate model loading
            time.sleep(1)
            # Create simple geometry for testing
            mesh = o3d.geometry.TriangleMesh.create_sphere()
            mesh.compute_vertex_normals()
            
            # Schedule geometry update on main thread
            self.root.after(0, lambda: self.main_renderer.update_geometries([mesh]))
            
        threading.Thread(target=load_task, daemon=True).start()
        
    def load_trajectory(self):
        """Load trajectory - run in background thread"""
        def load_task():
            # Simulate trajectory loading
            time.sleep(1)
            # Create simple trajectory geometry
            points = np.random.rand(10, 3) * 10
            lines = [[i, i+1] for i in range(len(points)-1)]
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([1, 0, 0])
            
            # Schedule update on main thread
            self.root.after(0, lambda: self.main_renderer.update_geometries([line_set]))
            
        threading.Thread(target=load_task, daemon=True).start()
        
    def on_close(self):
        """Cleanup on close"""
        self.rendering = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = UAVVisualizerFixed(root)
    root.mainloop()