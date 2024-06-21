import argparse
import os
import pickle
import random

import blenderproc as bproc
import bpy
import h5py
import matplotlib.pyplot as plt
import numpy as np
from blenderproc.python.loader.Front3DLoader import (
    load_front3d_no_furniture,
    load_small_front3d,
)
from PIL import Image, ImageChops

random.seed(5145)

parser = argparse.ArgumentParser()
parser.add_argument("front", help="Path to the 3D front file")
parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
parser.add_argument("front_3D_texture_path", help="Path to the 3D FRONT texture folder.")
parser.add_argument('cc_material_path', nargs='?', default="resources/cctextures", help="Path to CCTextures folder, see the /scripts for the download script.")
parser.add_argument("output_dir", nargs='?', default="examples/datasets/front_3d_with_improved_mat/output", help="Path to where the data should be saved")
args = parser.parse_args()

bproc.init()


if not os.path.exists(args.front) or not os.path.exists(args.future_folder):
    raise Exception("One of the two folders does not exist!")

mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

# set the light bounces
bproc.renderer.set_light_bounces(diffuse_bounces=10, glossy_bounces=10, max_bounces=10,
                                  transmission_bounces=10, transparent_max_bounces=10)

# load the front 3D objects
loaded_objects = load_small_front3d(
    json_path=args.front,
    future_model_path=args.future_folder,
    front_3D_texture_path=args.front_3D_texture_path,
    label_mapping=mapping
)

world = bpy.context.scene.world
world.light_settings.use_ambient_occlusion = True
world.light_settings.ao_factor = 0.9

# # Set a random hdri from the given haven directory as background
haven_hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven("/scratch2/yuxili/BlenderProc/resources/haven")
bproc.world.set_world_background_hdr_img(haven_hdri_path, strength = 15.0)

with open("cam2world_matrix_list.pkl", "rb") as fp:
    cam2world_matrix_list = pickle.load(fp)

cc_materials = bproc.loader.load_ccmaterials(args.cc_material_path, ["Wallpaper","Bricks", "Wood", "Carpet", "Tile", "Marble"])
wood_floor_materials = bproc.filter.by_cp(cc_materials, "asset_name", "WoodFloor.*", regex=True)
floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
for floor in floors:
    # For each material of the object
    for i in range(len(floor.get_materials())):
        # Replace the material with a random one
        floor_material = random.choice(wood_floor_materials)
        floor.set_material(i, floor_material)


baseboards_and_doors = bproc.filter.by_attr(loaded_objects, "name", "Baseboard.*|Door.*", regex=True)
for obj in baseboards_and_doors:
    # For each material of the object
    for i in range(len(obj.get_materials())):
        # Replace the material with a random one
        baseboard_material = random.choice(wood_floor_materials)
        obj.set_material(i, baseboard_material)


walls = bproc.filter.by_attr(loaded_objects, "name", "Wall.*", regex=True)
wallpaper_materials = bproc.filter.by_cp(cc_materials, "asset_name", "Wallpaper.*", regex=True)
wall_material = random.choice(wallpaper_materials)
for wall in walls:
    # For each material of the object
    for i in range(len(wall.get_materials())):
        wall.set_material(i, wall_material)

for cam2world_matrix in cam2world_matrix_list:
    bproc.camera.add_camera_pose(cam2world_matrix)
bproc.camera.set_intrinsics_from_blender_params(lens=np.pi/2, lens_unit="FOV")
# Also render normals
bproc.renderer.enable_normals_output()
bproc.renderer.enable_segmentation_output(map_by=["category_id"])

# render the whole pipeline
data = bproc.renderer.render()

def process_differences(array1, array2):
    # Compute the absolute differences between the arrays
    diff = np.abs(array1 - array2)
    
    # Create a mask where differences are greater than the threshold
    significant_diff = diff == 0
    
    # Set significant differences in the result array to a specific value (e.g., 255)
    array2[significant_diff] = 0  # 255 is commonly used in image contexts to represent white
    
    return array2

# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data)
files = os.listdir(args.output_dir)
for file in files:
    file_name = file.split('.')[0]
    if file.endswith('.hdf5'):
        with h5py.File(os.path.join(args.output_dir, file), 'r') as f:
            colors = f['colors'][:]
            colors_image = Image.fromarray(colors)
            colors_image.save(os.path.join(args.output_dir, f'{file_name}_foreground_removed.png'))
            semantic = f['category_id_segmaps'][:]
            np.save(os.path.join(args.output_dir, f'{file_name}_semantic_foreground_removed.npy'), semantic)
            semantic_reference = np.load(os.path.join(args.output_dir, f'{file_name}_semantic.npy'))
            semantic_diff = process_differences(semantic, semantic_reference)
            print(np.unique(semantic_diff))
            np.save(os.path.join(args.output_dir, f'{file_name}_semantic_mask.npy'), semantic_diff)
            plt.figure(figsize=(10, 6))
            plt.imshow(semantic_diff*255)
            plt.axis('off')  # Hide the axes
            plt.savefig(os.path.join(args.output_dir, f'{file_name}_semantic_map.png'), bbox_inches='tight', pad_inches=0)
            plt.close()  # Close the figure to free up memory