#!/bin/bash

    # --------------------------
    # CIFAR-100 학습
    # --------------------------
python main.py \
  --mode train \
  --dataset_name cifar100 \
  --batch_size 4 \
  --epochs 10 \
  --lr 1e-3 \
  --clip_model_name "openai/clip-vit-base-patch32" \
  --dino_model_name "facebook/dinov2-base" \
  --clip_scales 1.0 2.0 \
  --dino_scales 1.0 2.0 \
  --clip_num_prefix 1 \
  --dino_num_prefix 1 \
  --checkpoint_dir "./checkpoints/cifar100/both_s2" \
  --resume_checkpoint "./checkpoints/cifar100/both_s2/best_epoch_2.pth" \
  --project_name "test_cifar100_project"
