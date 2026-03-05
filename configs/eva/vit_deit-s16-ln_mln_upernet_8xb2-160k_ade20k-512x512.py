_base_ = '../vit/vit_vit-b16_mln_upernet_8xb2-160k_ade20k-512x512.py'

# 1. Import your specific python file so the registry sees 'vit_small_patch16_dinov3'
# If your file is named my_custom_models.py and is in the same folder, use this:
custom_imports = dict(imports=['my_custom_models'], allow_failed_imports=False)

model = dict(
    pretrained=None,        
    backbone=dict(
        _delete_=True,      
        type='mmpretrain.TIMMBackbone',
        model_name='vit_small_patch16_dinov3', 
        features_only=True,
        pretrained=False,   
        out_indices=(2, 5, 8, 11) 
    ),
    neck=dict(
        # Updated to match DINOv3 Small's embed_dim of 384
        in_channels=[384, 384, 384, 384], 
        out_channels=384 
    ),
    decode_head=dict(
        num_classes=150, 
        # Updated to match the neck's output
        in_channels=[384, 384, 384, 384] 
    ),
    auxiliary_head=dict(
        num_classes=150, 
        # Auxiliary head takes features from the 3rd index (layer 9), which has 384 channels
        in_channels=384 
    )
)
