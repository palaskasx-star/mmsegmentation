#!/bin/bash

# ==========================================
# ViT (DeiT) Architecture Models
# ==========================================

# 1. ViT Manifold (Tiny)
python tools/dist_train.sh \
    ./configs/vit/vit_deit-t16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py \
    4 \
    --work-dir ../pretrained_models/models/vit_architecture/manifold/Deit3Small_t_DeitTiny_s \
    --cfg-options \
        model.backbone.init_cfg.type="Pretrained" \
        model.backbone.init_cfg.checkpoint="../pretrained_models/models/vit_architecture/manifold/Deit3Small_t_DeitTiny_s/checkpoint_epoch_300_mmseg.pth" \
        train_dataloader.batch_size=4 \
        randomness.seed=42

# 2. ViT Manifold (Small)
python tools/dist_train.sh \
    ./configs/vit/vit_deit-s16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py \
    4 \
    --work-dir ../pretrained_models/models/vit_architecture/manifold/Deit3Base_t_DeitSmall_s \
    --cfg-options \
        model.backbone.init_cfg.type="Pretrained" \
        model.backbone.init_cfg.checkpoint="../pretrained_models/models/vit_architecture/manifold/Deit3Base_t_DeitSmall_s/checkpoint_epoch_300_mmseg.pth" \
        train_dataloader.batch_size=4 \
        randomness.seed=42

# 3. ViT Gaussian Kernel (Tiny)
python tools/dist_train.sh \
    ./configs/vit/vit_deit-t16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py \
    4 \
    --work-dir ../pretrained_models/models/vit_architecture/gaussian_kernel/tiny \
    --cfg-options \
        model.backbone.init_cfg.type="Pretrained" \
        model.backbone.init_cfg.checkpoint="../pretrained_models/models/vit_architecture/gaussian_kernel/tiny/checkpoint_epoch_300_mmseg.pth" \
        train_dataloader.batch_size=4 \
        randomness.seed=42

# 4. ViT Gaussian Kernel (Small)
python tools/dist_train.sh \
    ./configs/vit/vit_deit-s16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py \
    4 \
    --work-dir ../pretrained_models/models/vit_architecture/gaussian_kernel/small \
    --cfg-options \
        model.backbone.init_cfg.type="Pretrained" \
        model.backbone.init_cfg.checkpoint="../pretrained_models/models/vit_architecture/gaussian_kernel/small/checkpoint_epoch_300_mmseg.pth" \
        train_dataloader.batch_size=4 \
        randomness.seed=42

# 5. ViT Distillation Experiments (Tiny)
python tools/dist_train.sh \
    ./configs/vit/vit_deit-t16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py \
    4 \
    --work-dir ../pretrained_models/models/vit_architecture/distillation_experiments/tiny \
    --cfg-options \
        model.backbone.init_cfg.type="Pretrained" \
        model.backbone.init_cfg.checkpoint="../pretrained_models/models/vit_architecture/distillation_experiments/tiny/checkpoint_epoch_300_mmseg.pth" \
        train_dataloader.batch_size=4 \
        randomness.seed=42

# 6. ViT Distillation Experiments (Small)
python tools/dist_train.sh \
    ./configs/vit/vit_deit-s16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py \
    4 \
    --work-dir ../pretrained_models/models/vit_architecture/distillation_experiments/small \
    --cfg-options \
        model.backbone.init_cfg.type="Pretrained" \
        model.backbone.init_cfg.checkpoint="../pretrained_models/models/vit_architecture/distillation_experiments/small/checkpoint_epoch_300_mmseg.pth" \
        train_dataloader.batch_size=4 \
        randomness.seed=42

# 7. ViT DeiT Experiments (Tiny)
python tools/dist_train.sh \
    ./configs/vit/vit_deit-t16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py \
    4 \
    --work-dir ../pretrained_models/models/vit_architecture/deit_experiments/tiny \
    --cfg-options \
        model.backbone.init_cfg.type="Pretrained" \
        model.backbone.init_cfg.checkpoint="../pretrained_models/models/vit_architecture/deit_experiments/tiny/checkpoint_epoch_300_mmseg.pth" \
        train_dataloader.batch_size=4 \
        randomness.seed=42

# 8. ViT DeiT Experiments (Small)
python tools/dist_train.sh \
    ./configs/vit/vit_deit-s16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py \
    4 \
    --work-dir ../pretrained_models/models/vit_architecture/deit_experiments/small \
    --cfg-options \
        model.backbone.init_cfg.type="Pretrained" \
        model.backbone.init_cfg.checkpoint="../pretrained_models/models/vit_architecture/deit_experiments/small/checkpoint_epoch_300_mmseg.pth" \
        train_dataloader.batch_size=4 \
        randomness.seed=42

# 9. ViT Cosine Kernel (Tiny)
python tools/dist_train.sh \
    ./configs/vit/vit_deit-t16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py \
    4 \
    --work-dir ../pretrained_models/models/vit_architecture/cosine_kernel/tiny \
    --cfg-options \
        model.backbone.init_cfg.type="Pretrained" \
        model.backbone.init_cfg.checkpoint="../pretrained_models/models/vit_architecture/cosine_kernel/tiny/checkpoint_epoch_300_mmseg.pth" \
        train_dataloader.batch_size=4 \
        randomness.seed=42

# 10. ViT Cosine Kernel (Small)
python tools/dist_train.sh \
    ./configs/vit/vit_deit-s16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py \
    4 \
    --work-dir ../pretrained_models/models/vit_architecture/cosine_kernel/small \
    --cfg-options \
        model.backbone.init_cfg.type="Pretrained" \
        model.backbone.init_cfg.checkpoint="../pretrained_models/models/vit_architecture/cosine_kernel/small/checkpoint_epoch_300_mmseg.pth" \
        train_dataloader.batch_size=4 \
        randomness.seed=42
# ==========================================
# DinoV3 (EVA) Architecture Models
# ==========================================

# 11. DinoV3 Without Teacher (Tiny)
python tools/dist_train.sh \
    ./configs/eva/vit_deit-t16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py \
    4 \
    --work-dir ../pretrained_models/models/dinov3_architecture/without_teacher/tiny \
    --cfg-options model.backbone.checkpoint_path=../pretrained_models/models/dinov3_architecture/without_teacher/tiny/checkpoint_epoch_300_mmseg.pth train_dataloader.batch_size=4 \
    randomness.seed=42

# 12. DinoV3 Without Teacher (Small)
python tools/dist_train.sh \
    ./configs/eva/vit_deit-s16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py \
    4 \
    --work-dir ../pretrained_models/models/dinov3_architecture/without_teacher/small \
    --cfg-options model.backbone.checkpoint_path=../pretrained_models/models/dinov3_architecture/without_teacher/small/checkpoint_epoch_300_mmseg.pth train_dataloader.batch_size=4 \
    randomness.seed=42

# 13. DinoV3 Manifold (Tiny)
python tools/dist_train.sh \
    ./configs/eva/vit_deit-t16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py \
    4 \
    --work-dir ../pretrained_models/models/dinov3_architecture/manifold/Dinov3Small_t_Dinov3Tiny_s \
    --cfg-options model.backbone.checkpoint_path=../pretrained_models/models/dinov3_architecture/manifold/Dinov3Small_t_Dinov3Tiny_s/checkpoint_epoch_300_mmseg.pth train_dataloader.batch_size=4 \
    randomness.seed=42

# 14. DinoV3 Manifold (Small)
python tools/dist_train.sh \
    ./configs/eva/vit_deit-s16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py \
    4 \
    --work-dir ../pretrained_models/models/dinov3_architecture/manifold/Dinov3Base_t_Dinov3Small_s \
    --cfg-options model.backbone.checkpoint_path=../pretrained_models/models/dinov3_architecture/manifold/Dinov3Base_t_Dinov3Small_s/checkpoint_epoch_300_mmseg.pth train_dataloader.batch_size=4 \
    randomness.seed=42

# 15. DinoV3 Gaussian Kernel (Tiny)
python tools/dist_train.sh \
    ./configs/eva/vit_deit-t16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py \
    4 \
    --work-dir ../pretrained_models/models/dinov3_architecture/gaussian_kernel/tiny \
    --cfg-options model.backbone.checkpoint_path=../pretrained_models/models/dinov3_architecture/gaussian_kernel/tiny/checkpoint_epoch_300_mmseg.pth train_dataloader.batch_size=4 \
    randomness.seed=42

# 16. DinoV3 Gaussian Kernel (Small)
python tools/dist_train.sh \
    ./configs/eva/vit_deit-s16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py \
    4 \
    --work-dir ../pretrained_models/models/dinov3_architecture/gaussian_kernel/small \
    --cfg-options model.backbone.checkpoint_path=../pretrained_models/models/dinov3_architecture/gaussian_kernel/small/checkpoint_epoch_300_mmseg.pth train_dataloader.batch_size=4 \
    randomness.seed=42

# 17. DinoV3 Cosine Kernel (Tiny)
python tools/dist_train.sh \
    ./configs/eva/vit_deit-t16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py \
    4 \
    --work-dir ../pretrained_models/models/dinov3_architecture/cosine_kernel/tiny \
    --cfg-options model.backbone.checkpoint_path=../pretrained_models/models/dinov3_architecture/cosine_kernel/tiny/checkpoint_epoch_300_mmseg.pth train_dataloader.batch_size=4 \
    randomness.seed=42

# 18. DinoV3 Cosine Kernel (Small)
python tools/dist_train.sh \
    ./configs/eva/vit_deit-s16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py \
    4 \
    --work-dir ../pretrained_models/models/dinov3_architecture/cosine_kernel/small \
    --cfg-options model.backbone.checkpoint_path=../pretrained_models/models/dinov3_architecture/cosine_kernel/small/checkpoint_epoch_300_mmseg.pth train_dataloader.batch_size=4 \
    randomness.seed=42
