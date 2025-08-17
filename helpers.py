import ctypes
import OpenGL
from OpenGL.raw.GL.VERSION.GL_1_1 import glGenTextures as raw_glGenTextures
from OpenGL import GL
import numpy as np
import open3d as o3d
import trimesh
import asyncio
import pickle as pkl
import matplotlib.pyplot as plt
import cv2
import math
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from typing import List, Dict, Union
import json
import cmasher as cmr
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import warnings
import logging
import yaml
from datetime import datetime
from pathlib import Path
import pickle
from IPython.display import display, clear_output

shared={}

def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def patched_glGenTextures(count, textures=None):
    if textures is None:
        arr = (ctypes.c_uint * count)()
        raw_glGenTextures(count, arr)
        if count == 1:
            return arr[0]
        return arr
    else:
        raw_glGenTextures(count, textures)
GL.glGenTextures = patched_glGenTextures

import pyrender

def start_resource_monitor():
    import ipywidgets as widgets
    import threading
    import time
    import psutil
    import GPUtil

    def get_color(percentage):
        if percentage < 70:
            return 'success'
        elif 70 <= percentage < 90:
            return 'warning'
        else:
            return 'danger'

    def create_progress_bar(label_text):
        label = widgets.Label(value=label_text)
        progress = widgets.IntProgress(value=0, min=0, max=100, layout=widgets.Layout(width='200px'))
        progress.bar_style = 'info'
        container = widgets.VBox([label, progress])
        return label, progress, container

    cpu_label, cpu_progress, cpu_container = create_progress_bar("CPU Usage")
    mem_label, mem_progress, mem_container = create_progress_bar("Memory Usage")
    gpu_label, gpu_progress, gpu_container = create_progress_bar("GPU Usage")
    hbox = widgets.HBox([cpu_container, mem_container, gpu_container])
    clear_output(wait=True)
    display(hbox)

    def background_update():
        while True:
            cpu_usage = psutil.cpu_percent(interval=None)
            cpu_progress.value = int(cpu_usage)
            cpu_progress.bar_style = get_color(cpu_usage)
            cpu_label.value = f"CPU: {cpu_usage:.1f}%"

            mem_usage = psutil.virtual_memory().percent
            mem_progress.value = int(mem_usage)
            mem_progress.bar_style = get_color(mem_usage)
            mem_label.value = f"Memory: {mem_usage:.1f}%"

            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].memoryUtil * 100
                gpu_progress.value = int(gpu_usage)
                gpu_progress.bar_style = get_color(gpu_usage)
                gpu_label.value = f"GPU: {gpu_usage:.1f}%"
            else:
                gpu_progress.value = 0
                gpu_progress.bar_style = 'info'
                gpu_label.value = "GPU: N/A"

            time.sleep(1)

    thread = threading.Thread(target=background_update, daemon=True)
    thread.start()

def initialize_logging(config_path, is_for_photo):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    scene_name = config['scene_name']
    shared.update({"config": config})
    shared['config'] = config

def init(config_path, is_for_photo=False):
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    shared['config'] = cfg
    if is_for_photo:
        start_resource_monitor()
    initialize_logging(config_path, is_for_photo)

def show2d(images, path="", dpi=100, layout="h", bgr=False, return_image=False, line_width=2):
    cmap_plasma = plt.get_cmap('plasma')
    img_list = []
    for img in images:
        if img.ndim == 2 or img.shape[2] == 1:
            normalized = img / img.max() if img.max() != 0 else img
            colored = cmap_plasma(normalized)[:, :, :3]
            colored = (colored * 255).astype(np.uint8)
            img_list.append(colored)
        else:
            img_list.append(img.copy())
    if len(img_list) == 1:
        img = img_list[0]
    else:
        line_value = 1.0 if img_list[0].dtype in [np.float32, np.float64] else 255
        if layout == "h":
            heights = [img.shape[0] for img in img_list]
            max_height = max(heights)
            resized_imgs = [cv2.resize(img, (int(img.shape[1] * max_height / img.shape[0]), max_height)) for img in img_list]
            white_line = np.ones((max_height, line_width, 3), dtype=img_list[0].dtype) * line_value
            img = np.hstack([np.hstack([resized_imgs[i], white_line]) for i in range(len(resized_imgs)-1)] + [resized_imgs[-1]])
        elif layout == "v":
            widths = [img.shape[1] for img in img_list]
            max_width = max(widths)
            resized_imgs = [cv2.resize(img, (max_width, int(img.shape[0] * max_width / img.shape[1]))) for img in img_list]
            white_line = np.ones((line_width, max_width, 3), dtype=img_list[0].dtype) * line_value
            img = np.vstack([np.vstack([resized_imgs[i], white_line]) for i in range(len(resized_imgs)-1)] + [resized_imgs[-1]])
    if return_image:
        return img
    if bgr:
        img = img[:, :, ::-1]
    plt.figure(dpi=dpi)
    plt.axis("off")
    if path:
        plt.imshow(img)
        plt.savefig(path, bbox_inches='tight', dpi=dpi, pad_inches=0.0)
    else:
        plt.imshow(img)
        plt.show()

def compute_camera_pose(origin, yaw, pitch):
    a, b = np.array([0, 0, 1], dtype=np.float32), np.array([0, -1, 0], dtype=np.float32)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    cross, dot = np.cross(a, b), np.dot(a, b)
    cross_norm_sq = np.dot(cross, cross)
    skew = np.array([[0, -cross[2], cross[1]],
                    [cross[2], 0, -cross[0]],
                    [-cross[1], cross[0], 0]], dtype=np.float32)
    rotation_initial = np.eye(3, dtype=np.float32) + skew + skew @ skew * ((1 - dot) / cross_norm_sq) if cross_norm_sq >= 1e-16 else np.eye(3, dtype=np.float32)
    yaw_rad, pitch_rad = np.radians(yaw), np.radians(pitch)
    cos_yaw, sin_yaw = np.cos(-yaw_rad), np.sin(-yaw_rad)
    cos_pitch, sin_pitch = np.cos(pitch_rad), np.sin(pitch_rad)
    R_z = np.array([[cos_yaw, -sin_yaw, 0],
                    [sin_yaw,  cos_yaw, 0],
                    [0,        0,       1]], dtype=np.float32)
    R_x = np.array([[1, 0,         0],
                    [0, cos_pitch, -sin_pitch],
                    [0, sin_pitch,  cos_pitch]], dtype=np.float32)
    cam_rot = R_z @ R_x @ rotation_initial
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = cam_rot
    pose[:3, 3] = origin
    return pose

def scale_camera(scale, w, h, fx, fy, cx, cy):
    w_scaled = int(scale * w)
    h_scaled = int(scale * h)
    fx_scaled = int(scale * fx)
    fy_scaled = int(scale * fy)
    cx_scaled = int(scale * cx)
    cy_scaled = int(scale * cy)
    return w_scaled, h_scaled, fx_scaled, fy_scaled, cx_scaled, cy_scaled

def create_camera_mesh(width=5472, height=3648):
    vertices = np.array([
        [1, 1, 1],
        [-1, 1, 1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, -1],
        [0, 0, 1.5],
        [0, 0, 0]
    ], dtype=np.float32)

    aspect_ratio = width / height
    vertices[:4, 0] *= aspect_ratio
    vertices[:4] *= 0.4
    vertices[4:8, 0] *= aspect_ratio
    vertices[:, 2] *= 0.7
    center = np.mean(vertices, axis=0)
    vertices -= center

    face_indices = [
        [0, 1, 3, 1, 2, 3],
        [0, 3, 4, 0, 4, 5],
        [0, 5, 1, 1, 5, 6],
        [1, 6, 2, 2, 6, 7],
        [2, 7, 3, 3, 7, 4],
        [4, 7, 5, 5, 7, 6],
        [0, 1, 8, 1, 2, 8],
        [2, 3, 8, 3, 0, 8],
    ]

    face_colors = [
        [0.2, 0.8, 0.8],
        [0.2, 0.8, 0.8],
        [0.2, 0.8, 0.8],
        [0.2, 0.8, 0.8],
        [0.2, 0.8, 0.8],
        [0.2, 0.8, 0.8],
        [0.9, 0.0, 0.0],
        [0.9, 0.0, 0.0],
    ]

    mesh_faces = []
    mesh_face_colors = []
    for i, indices in enumerate(face_indices):
        tri_indices = np.array(indices).reshape(-1, 3)
        mesh_faces.append(vertices[tri_indices])
        mesh_face_colors.extend([face_colors[i]] * len(tri_indices))

    mesh_faces = np.vstack(mesh_faces)
    mesh_face_colors = np.array(mesh_face_colors)
    hardcoded_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), 
        (4, 5), (5, 6), (6, 7), (7, 4),   
        (0, 5), (1, 6), (2, 7), (3, 4),   
        (0, 8), (1, 8), (2, 8), (3, 8), 
    ]

    xs, ys, zs = [], [], []
    for i1, i2 in hardcoded_edges:
        p1 = vertices[i1]
        p2 = vertices[i2]
        xs.extend([p1[0], p2[0], None])
        ys.extend([p1[1], p2[1], None])
        zs.extend([p1[2], p2[2], None])
    mesh_edges = np.array([xs, ys, zs], dtype=object)
    return mesh_faces, mesh_face_colors, center, mesh_edges

default_camera_faces, default_camera_face_colors, default_camera_origin, default_camera_edges = create_camera_mesh()

def init_renderer(scale=0.05, load_mesh=False):
    config = shared.get('config', {})

    fl_mm = config.get('focal_length_mm', 8.8)
    sensor_mm = config.get('sensor_size_mm', 13.2)
    width_px = config.get('width_px', 5472)
    height_px = config.get('height_px', 3648)
    fl_px = fl_mm / (sensor_mm / width_px)
    pp_x = width_px / 2
    pp_y = height_px / 2

    s_w, s_h, s_fx, s_fy, s_cx, s_cy = scale_camera(
        scale, width_px, height_px, fl_px, fl_px, pp_x, pp_y
    )
    cam_params = [s_w, s_h, s_fx, s_fy, s_cx, s_cy]
    shared.update({"camera_params": cam_params})
    renderer = pyrender.OffscreenRenderer(viewport_width=s_w, viewport_height=s_h)
    scene = pyrender.Scene(ambient_light=True)
    
    scene_name = config['scene_name']
    mesh_dir = '.'  # All files in current directory
    cache_filename = 'cached_scene_LR.pkl' if not load_mesh else 'cached_scene.pkl'
    mesh_filename = f"{scene_name}.ply" if not load_mesh else f"{scene_name}.obj"
    
    cache_scene_path = Path(cache_filename)
    mesh_path = Path(mesh_filename)
    
    if cache_scene_path.exists():
        with cache_scene_path.open('rb') as f:
            scene = pkl.load(f)
    else:
        intrinsics = pyrender.IntrinsicsCamera(fx=s_fx, fy=s_fy, cx=s_cx, cy=s_cy)
        scene = pyrender.Scene(ambient_light=True)

        if not load_mesh:
            msh = o3d.io.read_triangle_mesh(str(mesh_path))
            tm_obj = trimesh.Trimesh(
                vertices=np.asarray(msh.vertices),
                faces=np.asarray(msh.triangles),
                vertex_normals=np.asarray(msh.vertex_normals),
                process=False
            )
            msh_node = pyrender.Mesh.from_trimesh(tm_obj)
            scene.add_node(pyrender.Node(mesh=msh_node))
            del msh, tm_obj, msh_node
        else:
            tm = trimesh.load(str(mesh_path))
            if isinstance(tm, trimesh.Scene):
                for geom in tm.geometry.values():
                    tm_ = pyrender.Mesh.from_trimesh(geom)
                    scene.add_node(pyrender.Node(mesh=tm_))
            else:
                tm_ = pyrender.Mesh.from_trimesh(tm)
                scene.add_node(pyrender.Node(mesh=tm_))
            del tm
            if 'tm_' in locals():
                del tm_

        scene.add(intrinsics, pose=np.eye(4))
        
        with cache_scene_path.open('wb') as f:
            pkl.dump(scene, f)
    
    shared.update({"scene": scene, "renderer": renderer})
    shared.update(dict(zip(['w', 'h', 'fx', 'fy', 'cx', 'cy'], shared['camera_params'])))
    
    del scene, renderer, cam_params

def render_scene(origin, yaw, pitch):
    try:
        # Check if glGenVertexArrays is available
        if not hasattr(GL, 'glGenVertexArrays') or not GL.glGenVertexArrays:
            print("OpenGL glGenVertexArrays not supported. Returning placeholder image.")
            width, height = 640, 480  # Default resolution
            colormap = np.full((height, width, 3), 128, dtype=np.uint8)  # Grey image
            depthmap = np.zeros((height, width), dtype=np.float32)
            return colormap, depthmap

        renderer = shared.get('renderer')
        if renderer is None:
            print("Renderer not initialized. Initializing with default settings.")
            init_renderer(scale=shared['config'].get('render_scale', 0.5))
            renderer = shared.get('renderer')
        scene = shared.get('scene')
        if scene is None:
            raise ValueError("Scene not initialized in shared state")
        
        camera_pose = compute_camera_pose(origin, yaw, pitch)
        scene.set_pose(scene.main_camera_node, pose=camera_pose)
        colormap, depthmap = renderer.render(
            scene, flags=pyrender.RenderFlags.FLAT | pyrender.RenderFlags.RGBA)
        colormap = colormap.copy()
        colormap[depthmap == 0] = [0, 0, 0, 255]
        colormap = colormap[:, :, :3].astype(np.uint8)
        return colormap, depthmap
    except OpenGL.error.NullFunctionError as e:
        print(f"OpenGL error: {e}. Returning placeholder image.")
        width, height = 640, 480
        colormap = np.full((height, width, 3), 128, dtype=np.uint8)  # Grey image
        depthmap = np.zeros((height, width), dtype=np.float32)
        return colormap, depthmap
    except Exception as e:
        print(f"Rendering failed: {e}. Returning placeholder image.")
        width, height = 640, 480
        colormap = np.full((height, width, 3), 128, dtype=np.uint8)  # Grey image
        depthmap = np.zeros((height, width), dtype=np.float32)
        return colormap, depthmap

AXES_CONV_TO_TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
TUPLE_TO_AXES_CONV = {v: k for k, v in AXES_CONV_TO_TUPLE.items()}
NEXT_AXIS_IN_CONV = [1, 2, 0, 1]

def rot_mat_from_vecs(vec1, vec2):
    unit_vec1 = (vec1 / np.linalg.norm(vec1)).reshape(3)
    unit_vec2 = (vec2 / np.linalg.norm(vec2)).reshape(3)
    cross_prod = np.cross(unit_vec1, unit_vec2) 
    dot_prod = np.dot(unit_vec1, unit_vec2)
    cross_prod_norm = np.linalg.norm(cross_prod)
    skew_symm_mat = np.array([[0, -cross_prod[2], cross_prod[1]],
                              [cross_prod[2], 0, -cross_prod[0]],
                              [-cross_prod[1], cross_prod[0], 0]])
    rot_mat = np.eye(3) + skew_symm_mat + skew_symm_mat.dot(skew_symm_mat) * ((1 - dot_prod) / (cross_prod_norm ** 2))
    return rot_mat

def euler_to_mat(ang_i, ang_j, ang_k, axes='sxyz'):
    try:
        first_axis, parity, rep, frame = AXES_CONV_TO_TUPLE[axes]
    except (AttributeError, KeyError):
        TUPLE_TO_AXES_CONV[axes]   
        first_axis, parity, rep, frame = axes
    i = first_axis
    j = NEXT_AXIS_IN_CONV[first_axis + parity]
    k = NEXT_AXIS_IN_CONV[first_axis - parity + 1]
    if frame:
        ang_i, ang_k = ang_k, ang_i
    if parity:
        ang_i, ang_j, ang_k = -ang_i, -ang_j, -ang_k
    si, sj, sk = math.sin(ang_i), math.sin(ang_j), math.sin(ang_k)  
    ci, cj, ck = math.cos(ang_i), math.cos(ang_j), math.cos(ang_k)
    ci_ck, ci_sk = ci * ck, ci * sk
    si_ck, si_sk = si * ck, si * sk
    M = np.identity(4)
    if rep:
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * si_sk + ci_ck
        M[j, k] = -cj * ci_sk - si_ck
        M[k, i] = -sj * ck
        M[k, j] = cj * si_ck + ci_sk
        M[k, k] = cj * ci_ck - si_sk
    else:
        M[i, i] = cj * ck
        M[i, j] = sj * si_ck - ci_sk  
        M[i, k] = sj * ci_ck + si_sk
        M[j, i] = cj * sk
        M[j, j] = sj * si_sk + ci_ck
        M[j, k] = sj * ci_sk - si_ck
        M[k, i] = -sj  
        M[k, j] = cj * si
        M[k, k] = cj * ci
    return M

def compute_camera_rotation(yaw, pitch):
    init_rot = rot_mat_from_vecs([0, 0, 1], [0, -1, 0])
    yaw_pitch_rot = euler_to_mat(
        np.deg2rad(-pitch), np.deg2rad(yaw), 0, axes="rxzx").T[:3, :3]
    cam_rot = yaw_pitch_rot.dot(init_rot)
    return cam_rot

def transform_edges_3n(edges_3n, rot, origin, scale=0.6):
    out_xs, out_ys, out_zs = [], [], []
    for x, y, z in zip(edges_3n[0], edges_3n[1], edges_3n[2]):
        if x is None or y is None or z is None:
            out_xs.append(None)
            out_ys.append(None)
            out_zs.append(None)
        else:
            pt = np.array([x, y, z], dtype=float)
            pt = rot @ pt
            pt = pt * scale + origin
            out_xs.append(pt[0])
            out_ys.append(pt[1])
            out_zs.append(pt[2])
    return np.array([out_xs, out_ys, out_zs], dtype=object)

def create_mesh_element_v2(
    faces: np.ndarray,
    face_colors: np.ndarray,
    text: str = None
) -> go.Mesh3d:
    x = faces[:, :, 0].ravel()
    y = faces[:, :, 1].ravel()
    z = faces[:, :, 2].ravel()

    count = faces.shape[0]
    i = np.arange(0, count*3, 3)
    j = i + 1
    k = i + 2

    face_colors_hex = [
        f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        for r, g, b in face_colors
    ]

    common_kwargs = dict(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        facecolor=face_colors_hex,
        opacity=1,
        lighting=dict(
            ambient=1.0,
            diffuse=0.0,
            specular=0.0,
            roughness=0.0,
            fresnel=0.0
        ),
        lightposition=dict(x=0, y=0, z=0),
        flatshading=True,
        showscale=False,
        name="",
        showlegend=False
    )

    if text:
        text_arr = [text] * (count * 3)
        return go.Mesh3d(
            **common_kwargs,
            text=text_arr,
            hoverinfo="text",
            hovertemplate="%{text}",
            hoverlabel=dict(namelength=0)
        )
    else:
        return go.Mesh3d(
            **common_kwargs,
            hoverinfo="none"
        )

def generate_camera_3d_thickness(
    origin: List[float],
    yaw: float,
    pitch: float,
    text: str = None,
    scale: float = 0.5
) -> List[go.Trace]:
    rot = compute_camera_rotation(yaw, pitch)

    flattened = default_camera_faces.reshape(-1, 3)
    rotated = (rot @ flattened.T).T * scale + np.array(origin)
    faces = rotated.reshape(default_camera_faces.shape)

    mesh = create_mesh_element_v2(faces, default_camera_face_colors, text=text)

    edges = transform_edges_3n(default_camera_edges, rot, origin, scale=scale)
    wireframe = go.Scatter3d(
        x=edges[0],
        y=edges[1],
        z=edges[2],
        mode="lines",
        line=dict(color="black", width=5),
        hoverinfo="none",
        showlegend=False,
        name=""
    )

    return [mesh, wireframe]

def compute_aspect_ratio(
    traces: List[Union[go.Mesh3d, go.Scatter3d]]
) -> Dict[str, float]:
    pts = []
    for tr in traces:
        xs = np.array(tr.x, float)
        ys = np.array(tr.y, float)
        zs = np.array(tr.z, float)
        stacked = np.vstack((xs, ys, zs)).T
        stacked = stacked[np.isfinite(stacked).all(axis=1)]
        if stacked.size:
            pts.append(stacked)
    if not pts:
        return {"x": 1.0, "y": 1.0, "z": 1.0}

    data = np.vstack(pts)
    mins = np.nanmin(data, axis=0)
    maxs = np.nanmax(data, axis=0)
    ranges = maxs - mins
    max_range = np.nanmax(ranges)

    if not np.isfinite(max_range) or max_range <= 0:
        return {"x": 1.0, "y": 1.0, "z": 1.0}

    return {
        "x": float(ranges[0] / max_range),
        "y": float(ranges[1] / max_range),
        "z": float(ranges[2] / max_range),
    }

def inpect_camera_marker():
    origin = [0.0, 0.0, 0.0]
    yaw, pitch, scale = 0.0, 0.0, 0.4
    traces = generate_camera_3d_thickness(origin, yaw, pitch, text="", scale=scale)
    fig = go.Figure()
    for tr in traces:
        fig.add_trace(tr)

    aspect = compute_aspect_ratio(traces)
    fig.update_layout(
        width=800,
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="white",
            camera=dict(eye=dict(x=0.9, y=0.9, z=0.7)),
            aspectmode="manual",
            aspectratio=aspect
        )
    )

    fig.show()

def load_camera_parameters(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cameras = data['cameras']
    
    positions = np.array([cam['position'] for cam in cameras])
    rotations = np.array([cam['rotation'] for cam in cameras])
    
    result = {
        'metadata': data['metadata'],
        'positions': positions,
        'rotations': rotations,
        'num_cameras': len(cameras)
    }
    
    if 'waypoint_order' in data:
        result['waypoint_order'] = np.array(data['waypoint_order'], dtype=np.int32)
        result['has_route'] = True
    else:
        result['waypoint_order'] = None
        result['has_route'] = False
    
    return result
def create_route_trace(params):
    positions = params['positions']
    waypoint_order = params.get('waypoint_order', list(range(len(positions))))
    if not params.get('has_route', False) or waypoint_order is None:
        print("No route information or waypoint order found. Using default sequential order.")
        waypoint_order = list(range(len(positions)))
    
    waypoint_order = np.array(waypoint_order)
    route_points = positions[waypoint_order]
    n = len(route_points)
    color_idx = np.linspace(0, 1, n)
    
    route_trace = go.Scatter3d(
        x=route_points[:, 0],
        y=route_points[:, 1],
        z=route_points[:, 2],
        mode='lines+markers',
        line=dict(
            color=color_idx,
            colorscale=params.get('path_color_scale', 'Plotly3'),
            cmin=0,
            cmax=1,
            showscale=False,
            width=5
        ),
        marker=dict(
            size=1,
            color=color_idx,
            colorscale=params.get('path_color_scale', 'Plotly3'),
            cmin=0,
            cmax=1,
            showscale=False,
            line=dict(width=1, color='white')
        ),
        name='TSP Route',
        hoverinfo='text',
        hovertext=[f'Waypoint {i+1}/{n}<br>Camera Index: {waypoint_order[i]}<br>X: {p[0]:.2f}<br>Y: {p[1]:.2f}<br>Z: {p[2]:.2f}' 
                   for i, p in enumerate(route_points)]
    )
    
    return route_trace
def get_custom_colormaps() -> list:
    colormaps = []
    cmasher_cmaps = [
        'rainforest', 'sunburst', 'ember', 'arctic', 'lavender', 'voltage',
        'redshift', 'toxic', 'neon'
    ]
    for cmap_name in cmasher_cmaps:
        cmap = cmr.cm.cmap_d[cmap_name]
        colormaps.append(ListedColormap(cmap(np.linspace(0, 1, 256)), name=cmap_name))
    matplotlib_cmaps = [
        'viridis', 'plasma', 'inferno', 'magma', 'cool', 'spring'
    ]
    for cmap_name in matplotlib_cmaps:
        cmap = cm.get_cmap(cmap_name)
        colormaps.append(ListedColormap(cmap(np.linspace(0, 1, 256)), name=cmap_name))
    return colormaps

def plot_colormaps(colormaps: list) -> None:
    n_colormaps = len(colormaps)
    ncols = 5
    nrows = (n_colormaps + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10))
    axes = axes.flatten()
    for ax, cmap in zip(axes, colormaps):
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        gradient = np.vstack((gradient, gradient))
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.set_title(cmap.name)
        ax.axis('off')
    for ax in axes[len(colormaps):]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def apply_plasma_colormap(points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-8)
    normalized = (points - mins) / span
    combined = normalized.sum(axis=1)
    vmin, vmax = combined.min(), combined.max()
    normed_vals = (combined - vmin) / (vmax - vmin + 1e-12)
    cmap = plt.colormaps.get_cmap("plasma")
    rgba = cmap(normed_vals)
    return rgba[:, :3]

custom_colormaps = get_custom_colormaps()

def create_mesh_element(faces, face_colors, text=None):
    x = faces[:, :, 0].flatten()
    y = faces[:, :, 1].flatten()
    z = faces[:, :, 2].flatten()
    i = np.arange(0, faces.shape[0] * 3, 3)
    j = i + 1
    k = i + 2
    face_colors_hex = [
        'rgb({},{},{})'.format(int(c[0]*255), int(c[1]*255), int(c[2]*255)) 
        for c in face_colors
    ]
    if text is not None:
        text_array = [text] * (faces.shape[0] * 3)
        mesh_element = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            facecolor=face_colors_hex,
            opacity=1,
            text=text_array,
            hoverinfo='text',
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=0.0, fresnel=0.0),
            lightposition=dict(x=0, y=0, z=0),
            flatshading=True,
            showscale=False
        )
    else:
        mesh_element = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            facecolor=face_colors_hex,
            opacity=1,
            hoverinfo='none',
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=0.0, fresnel=0.0),
            lightposition=dict(x=0, y=0, z=0),
            flatshading=True,
            showscale=False
        )
    return mesh_element

def create_text_element(position, string):
    hover_text = f"Position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
    text_element = go.Scatter3d(
        x=[position[0]], y=[position[1]], z=[position[2]],
        text=[string],
        mode='text',
        textposition='top right',
        textfont=dict(size=7, color='black'),
        hoverinfo='text',
        hovertext=hover_text,
        showlegend=False
    )
    return text_element

def create_points_element(points, colors=None, texts=None, point_size=2, opacity=1.0, symbol='circle', colormap_idx=9):
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    if colors is not None and len(colors) == len(points):
        color_array = [
            'rgb({},{},{})'.format(int(c[0]*255), int(c[1]*255), int(c[2]*255))
            for c in colors
        ]
    else:
        index_vals = np.linspace(0, 1, len(points))
        cmap = custom_colormaps[colormap_idx](index_vals)
        color_array = [
            'rgb({},{},{})'.format(int(c[0]*255), int(c[1]*255), int(c[2]*255))
            for c in cmap
        ]
    scatter = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode='markers',
        marker=dict(
            size=point_size,
            opacity=opacity,
            symbol=symbol,
            color=color_array,
            showscale=False
        ),
        text=texts,
        hoverinfo='text'
    )
    return scatter

def generate_camera_3d(origin, yaw, pitch, text=None, scale=0.5):
    rot = compute_camera_rotation(yaw, pitch)
    flattened_orig = default_camera_faces.reshape(-1, 3).copy()
    rotated_verts = (rot @ flattened_orig.T).T
    rotated_verts = rotated_verts * scale + origin
    camera_faces = rotated_verts.reshape(default_camera_faces.shape)
    if text is not None:
        camera_3d = create_mesh_element(camera_faces, default_camera_face_colors, text=text)
    else:
        camera_3d = create_mesh_element(camera_faces, default_camera_face_colors)
    transformed_edges = transform_edges_3n(default_camera_edges, rot, origin, scale=scale)
    wireframe_3d = go.Scatter3d(
        x=transformed_edges[0],
        y=transformed_edges[1],
        z=transformed_edges[2],
        mode='lines',
        line=dict(color='black', width=1),
        hoverinfo='none',
        showlegend=False
    )
    return [camera_3d, wireframe_3d]

def is_plotly_object(obj):
    plotly_classes = [
        go.Scatter, go.Scatter3d, go.Mesh3d, go.Scattergeo,
        go.Choropleth, go.Heatmap, go.Bar, go.Histogram,
        go.Table, go.Layout, go.Figure
    ]
    for cls in plotly_classes:
        if isinstance(obj, cls):
            return True
    return False

def flatten_plotly_objects(objs):
    plotly_objects = []
    def flatten(obj):
        if isinstance(obj, list):
            for item in obj:
                flatten(item)
        elif is_plotly_object(obj):
            plotly_objects.append(obj)
    for obj in objs:
        flatten(obj)
    return plotly_objects

def show3d_plotly(objects, ret=False):
    objects_plotly = flatten_plotly_objects(objects)
    fig = go.Figure()
    x_vals, y_vals, z_vals = [], [], []
    for obj in objects_plotly:
        if hasattr(obj, 'x'):
            x_vals.extend([val for val in obj.x if val is not None])
        if hasattr(obj, 'y'):
            y_vals.extend([val for val in obj.y if val is not None])
        if hasattr(obj, 'z'):
            z_vals.extend([val for val in obj.z if val is not None])
    if not x_vals or not y_vals or not z_vals:
        if ret:
            return fig
        clear_output(wait=True)
        fig.show(config={
            'scrollZoom': True,
            'displayModeBar': False,
            'staticPlot': False,
            'editable': False,
            'doubleClick': 'reset'
        })
        return
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    z_min, z_max = min(z_vals), max(z_vals)
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range)
    aspectratio = {
        'x': x_range / max_range,
        'y': y_range / max_range,
        'z': z_range / max_range
    }
    for obj in objects_plotly:
        fig.add_trace(obj)
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        scene=dict(
            bgcolor='white',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.7)),
            aspectmode='manual',
            aspectratio=aspectratio
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        template='plotly_white'
    )
    fig.update_traces(hoverinfo='none')
    if not ret:
        clear_output(wait=True)
        fig.show(config={
            'scrollZoom': True,
            'displayModeBar': False,
            'staticPlot': False,
            'editable': False,
            'doubleClick': 'reset'
        })
    else:
        return fig