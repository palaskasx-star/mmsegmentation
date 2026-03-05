#!/bin/bash

# Navigate to your mmsegmentation root directory
cd /home/cpalaskas/diploma_project/mmsegmentation

# Your base configuration file
CONFIG_FILE="configs/vit/vit_deit-t16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py"

# List of your converted _mmseg.pth models
declare -a models=(
    "pretrained/gaussian_orth_prjct_model/checkpoint_epoch_300_mmseg.pth"
)

# Loop through the array and train each model sequentially
for model_path in "${models[@]}"; do
    # Extract the directory name to use for the output folder (e.g., "cosine_intra")
    model_name=$(basename $(dirname "$model_path"))
    
    # Define where the results (logs, checkpoints) for this specific model will be saved
    work_dir="work_dirs/ade20k_${model_name}"
    
    # Get the absolute path for the checkpoint
    abs_checkpoint_path="$(pwd)/$model_path"
    
    echo "==================================================="
    echo "🚀 Starting training for model: $model_name"
    echo "📦 Backbone weights: $abs_checkpoint_path"
    echo "📁 Saving results to: $work_dir"
    echo "==================================================="
    
    # Run the MMSegmentation training script
    # --work-dir redirects the output to the specific folder
    # --cfg-options overrides the hardcoded 'checkpoint' path in your Python config file
    python tools/train.py "$CONFIG_FILE" \
            --work-dir "$work_dir" \
            --cfg-options \
                model.backbone.init_cfg.type="Pretrained" \
                model.backbone.init_cfg.checkpoint="$abs_checkpoint_path"
        
    echo "✅ Finished training for: $model_name"
    echo ""
done

echo "🎉 All models have been trained successfully!"