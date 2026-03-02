_base_ = './vit_vit-b16_mln_upernet_8xb2-160k_ade20k-512x512.py'

model = dict(
    pretrained=None,        # <--- THE MAGIC KEY: Stops EncoderDecoder from injecting weights
    backbone=dict(
        pretrained=None,    # Keep this here just to be absolutely safe
        num_heads=3,
        embed_dims=192,
        drop_path_rate=0.1, 
        final_norm=True
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