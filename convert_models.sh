#!/bin/bash

# Navigate to your mmsegmentation root directory
cd /home/cpalaskas/diploma_project/mmsegmentation

# Define the list of your existing DeiT model checkpoints
declare -a models=(
    "pretrained/manifold_model/checkpoint_epoch_300.pth"
    "pretrained/cosine_model/checkpoint_epoch_300.pth"
    "pretrained/cosine_intra/checkpoint_epoch_300.pth"
    "pretrained/cosine_multiple_layers/checkpoint_epoch_300.pth"
    "pretrained/gaussian_model/checkpoint_epoch_300.pth"
    "pretrained/gaussian_orth_prjct_model/checkpoint_epoch_300.pth"
)

# Loop through the array and convert each model
for model_path in "${models[@]}"; do
    # Create the new filename by appending '_mmseg' before the .pth extension
    # Example: checkpoint_epoch_300.pth -> checkpoint_epoch_300_mmseg.pth
    new_model_path="${model_path%.pth}_mmseg.pth"
    
    echo "Converting:"
    echo "  From: $model_path"
    echo "  To:   $new_model_path"
    
    # Run the official mmsegmentation conversion script for timm/DeiT ViTs
    python tools/model_converters/vit2mmseg.py "$model_path" "$new_model_path"
    
    echo "---------------------------------------------------"
done

echo "All models have been successfully converted!"