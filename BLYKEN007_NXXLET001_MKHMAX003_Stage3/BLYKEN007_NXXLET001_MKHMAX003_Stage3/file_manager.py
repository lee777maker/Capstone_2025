import os
import shutil
from pathlib import Path
from tkinter import filedialog
from helpers import shared

class File_Manager:
    
    def __init__(self):
        self.mesh_path = None
        self.json_path = None
    
    def ensure_config(self):
        """Make sure we have config loaded"""
        if 'config' not in shared:
            from helpers import init
            init("config.yml", is_for_photo=True)
        return shared['config']
    
    def load3DModel(self):
        """
        Ask user for a PLY file or OBJ file.
        Copy it to the expected location.
        """
        #for macOS - use separate entries instead of semicolon-separated
        src = filedialog.askopenfilename(
            title="Select a 3D model file (PLY or OBJ)",
            filetypes=[
                ("PLY Files", "*.ply"),
                ("OBJ Files", "*.obj"),
                ("All Files", "*")
            ]
        )
        
        if not src:
            print("No file selected.")
            return None
        
        config = self.ensure_config()
        
        # Get file extension and base name
        src_path = Path(src)
        ext = src_path.suffix.lower()
        base_name = src_path.stem
        
        # Update config with scene name
        config['scene_name'] = base_name
        
        # Create directory structure like in tutorial
        dataset_path = Path(config['dataset_path'])
        scene_path = dataset_path / base_name
        
        if ext == '.ply':
            # For PLY files, use MeshLR directory (like tutorial)
            mesh_dir = scene_path / 'MeshLR'
            mesh_dir.mkdir(parents=True, exist_ok=True)
            dest = mesh_dir / f"{base_name}.ply"
            
            # Copy the file
            if src_path.absolute() != dest.absolute():
                shutil.copy2(src, dest)
                print(f"Copied PLY to: {dest}")
            
            # Remove cached scene files
            for cache_file in ['cached_scene.pkl', 'cached_scene_LR.pkl']:
                cache_path = mesh_dir / cache_file
                if cache_path.exists():
                    cache_path.unlink()
                    
        else:  # OBJ
            # For OBJ files, use Mesh directory
            mesh_dir = scene_path / 'Mesh'
            mesh_dir.mkdir(parents=True, exist_ok=True)
            dest = mesh_dir / f"{base_name}{ext}"
            
            # Copy the OBJ file
            if src_path.absolute() != dest.absolute():
                shutil.copy2(src, dest)
                print(f"Copied {ext.upper()} to: {dest}")
            
            # Copy MTL file if it exists
            mtl_src = src_path.with_suffix('.mtl')
            if mtl_src.exists():
                mtl_dest = mesh_dir / mtl_src.name
                shutil.copy2(mtl_src, mtl_dest)
                print(f"Copied MTL to: {mtl_dest}")
                
                # Copy texture files referenced in MTL
                self._copy_textures(mtl_src, mesh_dir)
            
            # Remove cached scene files
            for cache_file in ['cached_scene.pkl', 'cached_scene_LR.pkl']:
                cache_path = mesh_dir / cache_file
                if cache_path.exists():
                    cache_path.unlink()
        
        self.mesh_path = str(dest)
        return self.mesh_path
    
    def _copy_textures(self, mtl_path, dest_dir):
        """Copy texture files referenced in MTL"""
        mtl_dir = mtl_path.parent
        
        try:
            with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    # Look for texture references
                    for prefix in ['map_Kd', 'map_Ka', 'map_Ks', 'map_d', 'map_Bump']:
                        if line.startswith(prefix):
                            parts = line.split(maxsplit=1)
                            if len(parts) > 1:
                                texture_name = parts[1]
                                texture_path = mtl_dir / texture_name
                                if texture_path.exists():
                                    dest_texture = dest_dir / texture_path.name
                                    if not dest_texture.exists():
                                        shutil.copy2(texture_path, dest_texture)
                                        print(f"Copied texture: {texture_path.name}")
        except Exception as e:
            print(f"Warning: Could not copy textures: {e}")
    
    def loadFlightPaths(self):
        """Load JSON flight path file"""
        fp = filedialog.askopenfilename(
            title="Select flight path JSON (optional)",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*")]
        )
        
        if fp:
            self.json_path = fp
            print(f"Selected flight path: {fp}")
        
        return fp