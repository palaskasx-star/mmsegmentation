_base_ = '../vit/vit_vit-b16_mln_upernet_8xb2-160k_ade20k-512x512.py'

# 1. Import the file where you wrote your @register_model code
# Assuming you saved it as 'my_custom_models.py' in your root directory
custom_imports = dict(imports=['my_custom_models', 'mmpretrain.models'], allow_failed_imports=False)

model = dict(
    pretrained=None,        # Stops EncoderDecoder from injecting weights
    backbone=dict(
        _delete_=True,      # <--- CRITICAL: Deletes the base config's native ViT settings
        type='mmpretrain.TIMMBackbone',
        model_name='vit_tiny_patch16_dinov3', # Your custom registered name
        features_only=True,
        pretrained=False,   # Keep false, the bash script handles the checkpoint
        out_indices=(2, 5, 8, 11) # Extracts features from layers 3, 6, 9, 12 (assuming 12 layers)
    ),
    neck=dict(
        in_channels=[192, 192, 192, 192], 
        out_channels=192
    ),
    decode_head=dict(
        num_classes=150, 
        in_channels=[192, 192, 192, 192]
    ),
    auxiliary_head=dict(
        num_classes=150, 
        in_channels=192
    )
)


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    # Load annotations before padding so the ground truth mask gets padded too
    dict(type='LoadAnnotations', reduce_zero_label=True),
    # Pad both image and mask to be divisible by 16
    dict(type='Pad', size_divisor=16, pad_val=dict(img=0, seg=255)),
    dict(type='PackSegInputs')
]

# Apply the patched pipeline to validation and testing
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))))
