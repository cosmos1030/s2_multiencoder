#!/bin/bash

MODE=$1  # train or test
DATASET=$2  # cifar10 or imagenet100

if [ "$MODE" = "train" ]; then

  if [ "$DATASET" = "cifar100" ]; then
    # --------------------------
    # CIFAR-100 학습
    # --------------------------
    python main.py \
      --mode train \
      --dataset_name cifar100 \
      --batch_size 128 \
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

  else
    # --------------------------
    # ImageNet100 학습
    # --------------------------
    python main.py \
      --mode train \
      --dataset_name imagenet100 \
      --batch_size 256 \
      --epochs 10 \
      --lr 1e-3 \
      --train_dirs /datasets/imagenet100/train.X1 /datasets/imagenet100/train.X2 \
      --test_dir "/datasets/imagenet100/val.X" \
      --labels_path "/datasets/imagenet100/Labels.json" \
      --clip_model_name "openai/clip-vit-base-patch32" \
      --dino_model_name "facebook/dinov2-base" \
      --clip_scales 1.0 2.0 \
      --dino_scales 1.0 2.0 \
      --clip_num_prefix 1 \
      --dino_num_prefix 1 \
      --checkpoint_dir "./checkpoints/imagenet100" \
      --resume_checkpoint "" \
      --project_name "test_imagenet100_project"
  fi

elif [ "$MODE" = "test" ]; then

  if [ "$DATASET" = "cifar100" ]; then
    # --------------------------
    # CIFAR-100 테스트
    # --------------------------
    python main.py \
      --mode test \
      --dataset_name cifar100 \
      --checkpoint_path "./checkpoints/cifar100/both/best_epoch_3.pth" \
      --batch_size 512 \
      --clip_model_name "openai/clip-vit-base-patch32" \
      --dino_model_name "facebook/dinov2-base" \
      --clip_scales 1.0 \
      --dino_scales 1.0 \
      --clip_num_prefix 1 \
      --dino_num_prefix 1

  else
    # --------------------------
    # ImageNet100 테스트
    # --------------------------
    python main.py \
      --mode test \
      --dataset_name imagenet100 \
      --checkpoint_path "./checkpoints/imagenet100/best_epoch_1.pth" \
      --batch_size 512 \
      --test_dir "/datasets/imagenet100/val.X" \
      --labels_path "/datasets/imagenet100/Labels.json" \
      --clip_model_name "openai/clip-vit-base-patch32" \
      --dino_model_name "facebook/dinov2-base" \
      --clip_scales 1.0 \
      --dino_scales 1.0 \
      --clip_num_prefix 1 \
      --dino_num_prefix 1

  fi

else
  echo "Usage: ./run.sh [train|test] [cifar100|imagenet100]"
  exit 1
fi
