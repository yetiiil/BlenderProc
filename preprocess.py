import os
import shutil
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# Directory containing your images
source_dir = '/scratch2/yuxili/BlenderProc/output'

# Destination directories
source_destination = '/scratch2/yuxili/Marigold_old/dataset/3DFront_green/source'
target_destination = '/scratch2/yuxili/Marigold_old/dataset/3DFront_green/target'

# Create the destination folders if they do not exist
os.makedirs(source_destination, exist_ok=True)
os.makedirs(target_destination, exist_ok=True)

# Loop through the files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith('_green.png'):
        target_name = filename.replace('_green', '')

        # Define the full path for the original and green image
        original_image_path = os.path.join(source_dir, target_name)
        green_image_path = os.path.join(source_dir, filename)
        
        # Define the destination paths for the images
        original_dest_path = os.path.join(target_destination, target_name)
        green_dest_path = os.path.join(source_destination, filename)
        
        if not os.path.isfile(original_image_path) or not os.path.isfile(green_image_path):
            # print('Original image not found, skipping.')
            # print(original_image_path)
            # print(original_image_path.split("_")[0])
            prefix = filename.split("_")[0]
            # Loop through and delete files with the specified prefix
            for file in os.listdir(source_dir):
                if file.startswith(prefix):
                    os.remove(os.path.join(source_dir, file))
            
            continue
        if os.path.isfile(original_dest_path) and os.path.isfile(green_dest_path):
            print('Destination file already exists, skipping.')
            continue

        # Move the files
        shutil.copy(original_image_path, original_dest_path)
        shutil.copy(green_image_path, green_dest_path)
        print(f'Moved {target_name} to {target_destination} and {filename} to {source_destination}')

print('Processing complete.')