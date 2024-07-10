#!/bin/bash

# Define the directories
source_dir="/scratch2/yuxili/BlenderProc/output"
source_destination="/scratch2/yuxili/ControlNet/training/3DFront/source"
target_destination="/scratch2/yuxili/ControlNet/training/3DFront/target"

# Move all files from source_destination to source_dir
mv "$source_destination"/* "$source_dir"

# Move all files from target_destination to source_dir
mv "$target_destination"/* "$source_dir"