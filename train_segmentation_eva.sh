#!/bin/bash

# Navigate to your mmsegmentation root directory
cd /home/cpalaskas/diploma_project/mmsegmentation

# ADD THIS LINE: Tell Python to look in the current folder for custom modules
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 1. UPDATED: Point to the new config file using the TIMMBackbone
CONFIG_FILE="configs/eva/deit_tiny_upernet.py"

# List of your converted _mmseg.pth models
declare -a models=(
    "/home/cpalaskas/diploma_project/mmsegmentation/pretrained/dinov3_manifold/checkpoint_epoch_300_mmseg.pth"
    "/home/cpalaskas/diploma_project/mmsegmentation/pretrained/dinov3_tiny_cosine/checkpoint_epoch_300_mmseg.pth"
    "/home/cpalaskas/diploma_project/mmsegmentation/pretrained/dinov3_tiny_gaussian/checkpoint_epoch_300_mmseg.pth"
    "/home/cpalaskas/diploma_project/mmsegmentation/pretrained/dinov3_tiny_gaussian_cls/checkpoint_epoch_300_mmseg.pth"
)

# Loop through the array and train each model sequentially
for model_path in "${models[@]}"; do
    # Extract the directory name to use for the output folder (e.g., "dinov3_tiny_gaussian")
    model_name=$(basename $(dirname "$model_path"))
    
    # Define where the results (logs, checkpoints) for this specific model will be saved
    work_dir="work_dirs/ade20k_${model_name}"
    
    # CRITICAL FIX: Removed $(pwd)/ because the path in the array is already absolute
    abs_checkpoint_path="$model_path"
    
    echo "==================================================="
    echo "🚀 Starting training for model: $model_name"
    echo "📦 Backbone weights: $abs_checkpoint_path"
    echo "📁 Saving results to: $work_dir"
    echo "==================================================="
    
    # Run the MMSegmentation training script
    # This correctly passes the checkpoint to the TIMMBackbone via init_cfg
    python tools/train.py "$CONFIG_FILE" \
            --work-dir "$work_dir" \
            --cfg-options \
                model.backbone.init_cfg.type="Pretrained" \
                model.backbone.init_cfg.checkpoint="$abs_checkpoint_path"
        
    echo "✅ Finished training for: $model_name"
    echo ""
done

echo "🎉 All models have been trained successfully!"