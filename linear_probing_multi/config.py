import torch
import os

# 하이퍼파라미터
BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 경로 (사용자 환경에 맞게 수정)
BASE_DIR = '/notebooks/s2_multiencoder/imagenet100'
TRAIN_DIRS = [os.path.join(BASE_DIR, f'train.X{i}') for i in range(1, 5)]
VAL_DIR = os.path.join(BASE_DIR, 'val.X')
LABELS_PATH = os.path.join(BASE_DIR, 'Labels.json')

# 모델 이름
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DINO_MODEL_NAME = "facebook/dinov2-base"
