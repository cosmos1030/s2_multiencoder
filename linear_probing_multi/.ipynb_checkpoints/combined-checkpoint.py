import os
import json
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# huggingface
from transformers import (
    CLIPVisionModel, 
    AutoProcessor,          # CLIP용 Processor
    Dinov2Model, 
    AutoImageProcessor      # DINOv2용 Processor
)

##############################
# 1. 하이퍼파라미터 & 경로 설정
##############################
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 예시: ImageNet100 구조
base_dir = '/path/to/imagenet100'
val_dir = os.path.join(base_dir, 'val.X')
train_dirs = [os.path.join(base_dir, f'train.X{i}') for i in range(1, 5)]
labels_path = os.path.join(base_dir, 'Labels.json')

##################################
# 2. 라벨 로드 & id_to_class 매핑
##################################
with open(labels_path, 'r') as f:
    labels = json.load(f)

# class_id(str) -> class_name
id_to_class = {str(k): v for k, v in labels.items()}

#########################################
# 3. Processor (전처리) 정의
#########################################
# 3-1) CLIP용 Processor
clip_name = "openai/clip-vit-base-patch32"
clip_processor = AutoProcessor.from_pretrained(clip_name)

# 3-2) DINOv2용 ImageProcessor
dino_name = "facebook/dinov2-base"
dino_processor = AutoImageProcessor.from_pretrained(dino_name)

# ※ 주의: 두 Processor는 내부적으로 
#   - Resize 크기, 
#   - CenterCrop, 
#   - Normalize (mean, std) 
#   등 설정이 다릅니다.
#   -> 같은 이미지를 넣어도 서로 다른 전처리가 적용됩니다.


#########################################
# 4. Dataset 정의 (MultiFolder -> Dual 변환)
#########################################
class MultiFolderDualDataset(Dataset):
    """
    - 여러 폴더(train.X1, train.X2, ...) 속 (클래스 폴더 -> 이미지) 구조를 순회
    - 각 이미지를 로드하여
        * CLIP Processor로 만든 텐서
        * DINOv2 Processor로 만든 텐서
      두 개를 함께 리턴
    """
    def __init__(self, folders, clip_processor, dino_processor, id_to_class):
        super().__init__()
        self.samples = []
        self.clip_processor = clip_processor
        self.dino_processor = dino_processor
        
        # class_to_idx (문자열 라벨ID -> 0..N-1 인덱스)
        self.class_to_idx = {cls_id: i for i, cls_id in enumerate(sorted(id_to_class.keys()))}

        for folder in folders:
            if not os.path.isdir(folder):
                continue
            for label_id in os.listdir(folder):
                label_path = os.path.join(folder, label_id)
                if not os.path.isdir(label_path):
                    continue
                for file_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, file_name)
                    if os.path.isfile(img_path):
                        self.samples.append((img_path, label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_id = self.samples[idx]
        image = Image.open(path).convert("RGB")

        # CLIP 전처리
        clip_enc = self.clip_processor(images=image, return_tensors="pt")
        # DINOv2 전처리
        dino_enc = self.dino_processor(images=image, return_tensors="pt")

        # 각각 [1, 3, H, W] 형태 -> squeeze로 [3, H, W]
        clip_pixel = clip_enc["pixel_values"].squeeze(0)
        dino_pixel = dino_enc["pixel_values"].squeeze(0)

        label = self.class_to_idx[label_id]
        return clip_pixel, dino_pixel, label


# 검증셋은 ImageFolder 구조도 가능하지만, 여기서는 동일하게 MultiFolderDualDataset 으로 예시
class ImageFolderDualDataset(Dataset):
    """
    - val_dir 내에 class 폴더들이 있고 그 안에 이미지가 있다고 가정
    - CLIP/DINOv2 두 가지 텐서를 같이 리턴
    """
    def __init__(self, folder, clip_processor, dino_processor, id_to_class):
        super().__init__()
        self.samples = []
        self.clip_processor = clip_processor
        self.dino_processor = dino_processor
        
        self.class_to_idx = {cls_id: i for i, cls_id in enumerate(sorted(id_to_class.keys()))}
        
        if os.path.isdir(folder):
            for label_id in os.listdir(folder):
                label_path = os.path.join(folder, label_id)
                if not os.path.isdir(label_path):
                    continue
                for file_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, file_name)
                    if os.path.isfile(img_path):
                        self.samples.append((img_path, label_id))
                    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label_id = self.samples[idx]
        image = Image.open(path).convert("RGB")

        clip_enc = self.clip_processor(images=image, return_tensors="pt")
        dino_enc = self.dino_processor(images=image, return_tensors="pt")

        clip_pixel = clip_enc["pixel_values"].squeeze(0)
        dino_pixel = dino_enc["pixel_values"].squeeze(0)

        label = self.class_to_idx[label_id]
        return clip_pixel, dino_pixel, label


##############################
# 5. DataLoader 정의
##############################
train_dataset = MultiFolderDualDataset(train_dirs, clip_processor, dino_processor, id_to_class)
val_dataset   = ImageFolderDualDataset(val_dir, clip_processor, dino_processor, id_to_class)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


##############################################
# 6. Dual Encoder (CLIP + DINOv2) 모델 정의
##############################################
class DualEncoderLinearProbe(nn.Module):
    def __init__(self,
                 clip_model_name="openai/clip-vit-base-patch32",
                 dino_model_name="facebook/dinov2-base",
                 num_classes=100):
        super().__init__()
        
        # 1) CLIP Vision Model
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name)
        clip_hidden_dim = self.clip_model.config.hidden_size  # 768 for ViT-B/32
        
        # 2) DINOv2 Model
        #    (Hugging Face transformers: Dinov2Model)
        self.dino_model = Dinov2Model.from_pretrained(dino_model_name)
        dino_hidden_dim = self.dino_model.config.hidden_size  # 768 for dinov2-base

        # 두 인코더 파라미터 freeze
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.dino_model.parameters():
            param.requires_grad = False

        # concat(clip_feat, dino_feat) -> [B, clip_dim + dino_dim]
        in_dim = clip_hidden_dim + dino_hidden_dim
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, clip_pixels, dino_pixels):
        """
        clip_pixels: [B, 3, Hc, Wc]
        dino_pixels: [B, 3, Hd, Wd]
        """
        # (1) CLIP Forward
        with torch.no_grad():
            clip_out = self.clip_model(clip_pixels)
            # CLS 토큰 (batch, hidden_size)
            clip_feat = clip_out.last_hidden_state[:, 0, :]

        # (2) DINOv2 Forward
        #     Dinov2Model은 forward(input_ids=None, pixel_values=None, ...) 형태
        with torch.no_grad():
            dino_out = self.dino_model(pixel_values=dino_pixels)
            # cls_token: [B, hidden_size]
            dino_feat = dino_out.last_hidden_state[:, 0, :]

        # (3) Concat -> Linear
        concat_feat = torch.cat([clip_feat, dino_feat], dim=1)  # [B, clip_dim + dino_dim]
        logits = self.linear(concat_feat)
        return logits


###################################
# 7. 학습(Linear Probing) 함수 정의
###################################
def train_dual_encoder_probe(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=DEVICE):
    criterion = nn.CrossEntropyLoss()
    # 오직 model.linear 파라미터만 옵티마이저에 등록 (encoder는 freeze)
    optimizer = optim.Adam(model.linear.parameters(), lr=lr)

    model.to(device)
    best_acc = 0.0

    for epoch in range(epochs):
        #################################
        # (1) Train Loop
        #################################
        model.train()  # linear layer 학습 가능
        train_loss = 0.0
        for (clip_imgs, dino_imgs, labels) in train_loader:
            clip_imgs = clip_imgs.to(device)
            dino_imgs = dino_imgs.to(device)
            labels    = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(clip_imgs, dino_imgs)  # forward
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)

        train_loss_epoch = train_loss / len(train_loader.dataset)

        #################################
        # (2) Validation Loop
        #################################
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for (clip_imgs, dino_imgs, labels) in val_loader:
                clip_imgs = clip_imgs.to(device)
                dino_imgs = dino_imgs.to(device)
                labels    = labels.to(device)

                outputs = model(clip_imgs, dino_imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_loss_epoch = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss_epoch:.4f} "
              f"Val Loss: {val_loss_epoch:.4f} "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc

    print(f"\nBest Validation Accuracy: {best_acc:.4f}")


##############################
# 8. 실행
##############################
if __name__ == "__main__":
    # num_classes 설정 (예: 100 for ImageNet100)
    num_classes = len(id_to_class)

    # 모델 생성
    model = DualEncoderLinearProbe(
        clip_model_name=clip_name,
        dino_model_name=dino_name,
        num_classes=num_classes
    )

    # 학습 및 평가
    train_dual_encoder_probe(model, train_loader, val_loader)
