#!/bin/bash

# 기본 실행 예시:
# ./run.sh train
# ./run.sh test

MODE=$1  # train or test

if [ "$MODE" = "train" ]; then
  python main.py \
      --mode train \
      --batch_size 256 \
      --epochs 10 \
      --lr 1e-3 \
      --train_dirs /notebooks/s2_multiencoder/imagenet100/train.X1 /notebooks/s2_multiencoder/imagenet100/train.X2 /notebooks/s2_multiencoder/imagenet100/train.X3 /notebooks/s2_multiencoder/imagenet100/train.X4 \
      --test_dir "/notebooks/s2_multiencoder/imagenet100/val.X" \
      --labels_path "/notebooks/s2_multiencoder/imagenet100/Labels.json" \
      --clip_model_name "openai/clip-vit-base-patch32" \
      --dino_model_name "facebook/dinov2-base" \
      --checkpoint_dir "./checkpoints" \
      --resume_checkpoint "" \
      --project_name "my_earlystop_project"

elif [ "$MODE" = "test" ]; then
  # 체크포인트만 불러와서 테스트
  python main.py \
      --mode test \
      --checkpoint_path "./checkpoints/best_epoch_5.pth" \
      --batch_size 64 \
      --test_dir "/notebooks/s2_multiencoder/imagenet100/val.X" \
      --labels_path "/notebooks/s2_multiencoder/imagenet100/Labels.json" \
      --clip_model_name "openai/clip-vit-base-patch32" \
      --dino_model_name "facebook/dinov2-base"
else
  echo "Usage: ./run.sh [train|test]"
fi
