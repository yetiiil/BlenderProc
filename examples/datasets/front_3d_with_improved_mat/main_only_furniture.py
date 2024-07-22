import blenderproc as bproc
import argparse
import os
import numpy as np
import random
import pickle
import bpy
import h5py
from sklearn.cluster import DBSCAN
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import mathutils
from blenderproc.python.loader.Front3DLoader import (
    load_only_furniture,
)

parser = argparse.ArgumentParser()
parser.add_argument("front", help="Path to the 3D front file")
parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
parser.add_argument("front_3D_texture_path", help="Path to the 3D FRONT texture folder.")
parser.add_argument("cc_material_path",nargs="?",default="resources/cctextures",help="Path to CCTextures folder, see the /scripts for the download script.",)
parser.add_argument("output_dir",nargs="?",default="examples/datasets/front_3d_with_improved_mat/output",help="Path to where the data should be saved",)
parser.add_argument("--lamp_light_strength", type=float, default=5, help="Strength of the lamp light")
parser.add_argument("--ceiling_light_strength", type=float, default=0.5, help="Strength of the ceiling light")
parser.add_argument("--word_background_strength", type=float, default=5, help="Strength of the world background")
parser.add_argument("--random_seed", type=int, default=0, help="Random seed")
args = parser.parse_args()

random_seed = args.random_seed
lamp_light_strength = args.lamp_light_strength
ceiling_light_strength = args.ceiling_light_strength
word_background_strength = args.word_background_strength
random.seed(random_seed)
bproc.init()

if not os.path.exists(args.front) or not os.path.exists(args.future_folder):
    raise Exception("One of the two folders does not exist!")

mapping_file = bproc.utility.resolve_resource(
    os.path.join("front_3D", "3D_front_mapping.csv")
)
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

# set the light bounces
bproc.renderer.set_light_bounces(diffuse_bounces=100, glossy_bounces=100, max_bounces=100,
                                  transmission_bounces=100, transparent_max_bounces=100)


# load the front 3D objects
loaded_objects = load_only_furniture(
    json_path=args.front,
    future_model_path=args.future_folder,
    front_3D_texture_path=args.front_3D_texture_path,
    label_mapping=mapping,
    ceiling_light_strength=ceiling_light_strength,
    lamp_light_strength=lamp_light_strength
)

world = bpy.context.scene.world
world.light_settings.use_ambient_occlusion = True
world.light_settings.ao_factor = 0.9

# # Set a random hdri from the given haven directory as background
haven_hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven(
    "/scratch2/yuxili/BlenderProc/resources/haven"
)
bproc.world.set_world_background_hdr_img(haven_hdri_path, strength=word_background_strength)


# save it as blend file
bpy.ops.export_scene.obj('EXEC_DEFAULT', filepath=os.path.join(args.output_dir, "scene_only_furniture.obj"), use_materials=True)

cc_materials = bproc.loader.load_ccmaterials(args.cc_material_path, ["Wallpaper","Bricks", "Wood", "Carpet", "Tile", "Marble"])
wood_floor_materials = bproc.filter.by_cp(cc_materials, "asset_name", "WoodFloor.*", regex=True)
floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
floor_material = random.choice(wood_floor_materials)
for floor in floors:
    # For each material of the object
    for i in range(len(floor.get_materials())):
        # Replace the material with a random one
        floor.set_material(i, floor_material)

baseboards_and_doors = bproc.filter.by_attr(
    loaded_objects, "name", "Baseboard.*|Door.*", regex=True
)
for obj in baseboards_and_doors:
    # For each material of the object
    for i in range(len(obj.get_materials())):
        # Replace the material with a random one
        baseboard_material = random.choice(wood_floor_materials)
        obj.set_material(i, baseboard_material)


walls = bproc.filter.by_attr(loaded_objects, "name", "Wall.*", regex=True)
wallpaper_materials = bproc.filter.by_cp(
    cc_materials, "asset_name", "Wallpaper.*", regex=True
)
wall_material = random.choice(wallpaper_materials)
for wall in walls:
    # For each material of the object
    for i in range(len(wall.get_materials())):
        wall.set_material(i, wall_material)

# Init sampler for sampling locations inside the loaded front3D house
point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)

# Init bvh tree containing all mesh objects
bvh_tree = bproc.object.create_bvh_tree_multi_objects(
    [o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)]
)

poses = 0
tries = 0


def check_fur_name(name):
    for category_name in [
        "chair",
        "sofa",
        "table",
        "bed",
        "cabinet",
        "shelf",
        "pot",
        "plant",
    ]:
        if category_name in name.lower():
            return True
    return False


def check_struc_name(name):
    for category_name in [
        "baseboard",
        "wall",
        "floor",
        "window",
        "ceiling",
        "slab",
        "hole",
        "door",
        "stair",
        "cornice",
        "pocket",
        "light",
    ]:
        if category_name in name.lower():
            return True
    return False

def find_centroid(diff):
    indices = np.argwhere(diff == 1)
    if indices.shape[0] == 0:
        return None  # Handle case with no differences
    return np.mean(indices, axis=0).astype(int)

def gaussian(x, y, x0, y0, sigma):
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

def create_gaussian_mask(shape, centroid, sigma=20):
    x = np.arange(0, shape[1])
    y = np.arange(0, shape[0])
    x, y = np.meshgrid(x, y)
    mask = gaussian(x, y, centroid[1], centroid[0], sigma)
    normalized_mask = (mask / mask.max()) * 255  # Normalize mask to 255
    return normalized_mask

def apply_colored_gaussian_to_image(colors, mask, color='red'):
    colored_mask = np.zeros_like(colors, dtype=np.float64)
    if color == 'red':
        colored_mask[..., 0] = mask  # Red channel
        colored_mask[..., 1] -= mask  
        colored_mask[..., 2] -= mask 
    elif color == 'green':
        colored_mask[..., 1] = mask  # Green channel
        colored_mask[..., 0] -= mask   # Reduce red channel a bit
        colored_mask[..., 2] -= mask   # Reduce blue channel a bit
    elif color == 'blue':
        colored_mask[..., 2] = mask  # Green channel
        colored_mask[..., 1] -= mask   # Reduce red channel a bit
        colored_mask[..., 0] -= mask   # Reduce blue channel a bit
    
    return np.clip(colors + colored_mask, 0, 255).astype(np.uint8)


def process_differences(array1, array2):
    # Compute the absolute differences between the arrays
    diff = np.abs(array1 - array2)
    
    # Create a mask where differences are greater than the threshold
    significant_diff = diff == 0
    
    # Set significant differences in the result array to a specific value (e.g., 255)
    array2[significant_diff] = 0  # 255 is commonly used in image contexts to represent white
    array2[~significant_diff] = 1
    
    return array2

def create_random_sphere_light():
    height = np.random.uniform(1.9, 2.0)
    location = point_sampler.sample(height)
    
    light_data = bpy.data.lights.new(name="SphereLight", type='POINT')
    light_data.energy = 5  # Random light strength
    # Decide the color based on whether it should be warm or white
    if np.random.rand() < 0.5:
        # Warm light: higher red, lower blue, green varies slightly
        red = random.uniform(0.8, 1.0)
        green = random.uniform(0.5, 0.7)
        blue = random.uniform(0.1, 0.3)
    else:
        # White light: equal RGB components, not necessarily pure white to avoid flat lighting
        value = random.uniform(0.8, 1.0)
        red, green, blue = value, value, value

    light_data.color = (red, green, blue)

    # Create light object
    light_object = bpy.data.objects.new(name="SphereLight", object_data=light_data)
    bpy.context.collection.objects.link(light_object)

    # Random position in the scene
    light_object.location = location

    return light_object


with open(os.path.join(args.output_dir, "cam2world_matrix_list.pkl"), "rb") as fp:
    cam2world_matrix_list = pickle.load(fp)

bproc.camera.add_camera_pose(cam2world_matrix_list[0])
bproc.camera.set_intrinsics_from_blender_params(lens=np.pi / 3, lens_unit="FOV")



try:
    bproc.renderer.enable_segmentation_output(map_by=["category_id"])

    # render the whole pipeline
    data = bproc.renderer.render()

    # write the data to a .hdf5 container
    bproc.writer.write_hdf5(args.output_dir, data)

    files = os.listdir(args.output_dir)
    for file in files:
        file_name = args.front.split('/')[-1].split('.')[0]
        if file.endswith(".hdf5"):
            with h5py.File(os.path.join(args.output_dir, file), "r") as f:
                colors = f["colors"][:]
                colors_image = Image.fromarray(colors)
                colors_image.save(
                    os.path.join(
                        args.output_dir,
                        f"{file_name}_empty.png",
                    )
                )
except ReferenceError:
    pass