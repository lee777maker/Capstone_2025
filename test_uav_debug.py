#!/usr/bin/env python3
"""Debug and test script for UAV visualizer"""

import sys
import time
import threading
import traceback
import open3d as o3d
import numpy as np
from pathlib import Path

def test_renderer_threading():
    """Test renderer thread safety"""
    print("\n=== TESTING RENDERER THREADING ===")
    
    # Check main thread
    main_thread = threading.main_thread()
    current_thread = threading.current_thread()
    
    print(f"Main thread: {main_thread.name}")
    print(f"Current thread: {current_thread.name}")
    print(f"On main thread: {current_thread == main_thread}")
    
    # Test Open3D on main thread
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=100, height=100)
        vis.poll_events()
        vis.update_renderer()
        vis.destroy_window()
        print("✓ Open3D works on main thread")
    except Exception as e:
        print(f"✗ Open3D failed on main thread: {e}")
    
    print("===================================\n")

def test_geometry_operations():
    """Test geometry creation and bounds"""
    print("\n=== TESTING GEOMETRY OPERATIONS ===")
    
    try:
        # Create test mesh
        mesh = o3d.geometry.TriangleMesh.create_box(width=2, height=2, depth=2)
        mesh.compute_vertex_normals()
        
        # Check bounds
        bbox = mesh.get_axis_aligned_bounding_box()
        print(f"Bounds: {bbox.min_bound} to {bbox.max_bound}")
        print(f"Center: {bbox.get_center()}")
        print(f"Extent: {bbox.get_extent()}")
        
        # Validate bounds
        if np.all(np.isfinite(bbox.min_bound)) and np.all(np.isfinite(bbox.max_bound)):
            print("✓ Bounds are valid")
        else:
            print("✗ Invalid bounds detected")
            
    except Exception as e:
        print(f"✗ Geometry test failed: {e}")
    
    print("====================================\n")

def monitor_app_performance():
    """Monitor app performance metrics"""
    print("\n=== MONITORING PERFORMANCE ===")
    
    import psutil
    import os
    
    pid = os.getpid()
    process = psutil.Process(pid)
    
    print(f"Process ID: {pid}")
    print(f"CPU percent: {process.cpu_percent()}%")
    print(f"Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    print(f"Threads: {process.num_threads()}")
    
    # List threads
    for thread in threading.enumerate():
        print(f"  - {thread.name}: {'alive' if thread.is_alive() else 'dead'}")
    
    print("===============================\n")

def test_render_pipeline():
    """Test complete render pipeline"""
    print("\n=== TESTING RENDER PIPELINE ===")
    
    try:
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=640, height=480)
        
        # Add geometry
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        vis.add_geometry(mesh)
        
        # Set camera
        view_ctrl = vis.get_view_control()
        view_ctrl.set_lookat([0, 0, 0])
        view_ctrl.set_front([1, 1, -1])
        view_ctrl.set_up([0, 0, 1])
        view_ctrl.set_zoom(0.5)
        
        # Render frames
        for i in range(5):
            vis.poll_events()
            vis.update_renderer()
            img = vis.capture_screen_float_buffer(do_render=True)
            
            if img is not None:
                arr = np.asarray(img)
                print(f"  Frame {i}: {arr.shape}, min={arr.min():.2f}, max={arr.max():.2f}")
            else:
                print(f"  Frame {i}: Failed")
            
            time.sleep(0.1)
        
        vis.destroy_window()
        print("✓ Render pipeline works")
        
    except Exception as e:
        print(f"✗ Render pipeline failed: {e}")
        traceback.print_exc()
    
    print("================================\n")

if __name__ == "__main__":
    print("UAV VISUALIZER DEBUG SUITE")
    print("==========================")
    
    test_renderer_threading()
    test_geometry_operations()
    test_render_pipeline()
    monitor_app_performance()
    
    print("\nDebug tests complete!")