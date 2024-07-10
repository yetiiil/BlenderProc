#!/bin/bash

# Directory containing JSON files
json_dir="/scratch2/yuxili/3D-FRONT"

# Other directories
model_dir="/scratch2/yuxili/3D-FUTURE-model"
texture_dir="/scratch2/yuxili/3D-FRONT-texture"
output_dir="/scratch2/yuxili/BlenderProc/output_new"
resources_dir="resources/cctextures"

# Define light strength sets
# Set 1
lamp_light_strength1=40
ceiling_light_strength1=2
word_background_strength1=15

# Set 2
lamp_light_strength2=40
ceiling_light_strength2=2
word_background_strength2=15

get_base_name() {
    local json_file="$1"
    echo $(basename "$json_file" .json)
}

# Function to check if there are PNG files in output_dir matching the base name
check_png_files_exist() {
    local base_name="$1"
    local png_files=$(ls "$output_dir"/"${base_name}"*.png 2>/dev/null)
    if [ -n "$png_files" ]; then
        return 0  # PNG files exist
    else
        return 1  # No PNG files found
    fi
}

# Function to choose a set based on a random number
choose_set() {
    local random_number=$(awk 'BEGIN { srand(); print rand() }')
    if (( $(echo "$random_number > 0.5" | bc) )); then
        lamp_light_strength=$lamp_light_strength1
        ceiling_light_strength=$ceiling_light_strength1
        word_background_strength=$word_background_strength1
    else
        lamp_light_strength=$lamp_light_strength2
        ceiling_light_strength=$ceiling_light_strength2
        word_background_strength=$word_background_strength2
    fi
}

# Loop through each JSON file in the directory
for json_file in "$json_dir"/*.json; do
    base_name=$(get_base_name "$json_file")

    # Check if there are PNG files in output_dir matching the base name
    if check_png_files_exist "$base_name"; then
        echo "PNG file(s) exist for $json_file in $output_dir. Skipping."
        continue
    fi

    echo "Processing file: $json_file"

    # Choose a set of values
    choose_set

    # Generate a random seed
    random_seed=$RANDOM

    # Run BlenderProc with the chosen parameters
    blenderproc run examples/datasets/front_3d_with_improved_mat/main.py "$json_file" $model_dir $texture_dir $resources_dir $output_dir --lamp_light_strength $lamp_light_strength --ceiling_light_strength $ceiling_light_strength --word_background_strength $word_background_strength --random_seed $random_seed
    blenderproc run examples/datasets/front_3d_with_improved_mat/main_progressive.py "$json_file" $model_dir $texture_dir $resources_dir $output_dir --lamp_light_strength $lamp_light_strength --ceiling_light_strength $ceiling_light_strength --word_background_strength $word_background_strength --random_seed $random_seed
done
