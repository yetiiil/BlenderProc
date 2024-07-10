import os
import shutil
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

source_dir = '/scratch2/yuxili/BlenderProc/output_new'
files = os.listdir(source_dir)
mask_files = [filename for filename in files if filename.endswith('_mask.png')]
for filename in tqdm(mask_files, desc='Processing files', unit='file'):
    try:
        target_name = filename.replace('_mask', '_alpha')
        scene_id, image_id, _ = filename.split('_')
        reference_filename = f'{scene_id}_{int(image_id) - 1}.png'
        
        # Load the original image and the mask
        original_image = np.array(Image.open(os.path.join(source_dir, reference_filename)))
        mask = np.array(Image.open(os.path.join(source_dir, filename))) * 255
        
        # Ensure mask is single channel and has the same dimensions as the original image
        if mask.ndim == 2:  # Single channel
            mask = np.expand_dims(mask, axis=-1)
        
        # Combine the original image with the mask to create an RGBA image
        rgba_image = np.dstack((original_image, mask))
        
        # Save the resulting RGBA image
        result_image = Image.fromarray(rgba_image.astype('uint8'))
        result_image.save(os.path.join(source_dir, target_name))
    except:
        pass