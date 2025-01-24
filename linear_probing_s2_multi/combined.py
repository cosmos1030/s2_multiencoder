import os
import json
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1) Transformers - CLIP & DINOv2
from transformers import (
    CLIPVisionModel, AutoProcessor,     # CLIP
    Dinov2Model,     AutoImageProcessor # DINOv2
)

# 2) S2 multiscale wrapper (이미 구현되어 있다고 가정)
#    multiscale_forward(forward_fn, x, scales, num_prefix_token, ...)
from s2wrapper import forward as multiscale_forward


##############################
# 1. 하이퍼파라미터 & 경로 설정
##############################
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 예: ImageNet100 구조
base_dir = "/path/to/imagenet100"
train_dirs = [os.path.join(base_dir, f"train.X{i}") for i in range(1, 5)]
val_dir    = os.path.join(base_dir, "val.X")
labels_path = os.path.join(base_dir, "Labels.json")

##################################
# 2. 라벨 로드 & id_to_class 매핑
##################################
with open(labels_path, "r") as f:
    labels = json.load(f)

# str(class_id) -> class_name
id_to_class = {str(k): v for k, v in labels.items()}
num_classes = len(id_to_class)  # 예: 100

#########################################
# 3. 전처리(Processor) - CLIP / DINOv2
#########################################
clip_model_name = "openai/clip-vit-base-patch32"
dino_model_name = "facebook/dinov2-base"

# CLIP 전용 Processor (AutoProcessor)
clip_processor = AutoProcessor.from_pretrained(clip_model_name)
# DINOv2 전용 Processor (AutoImageProcessor)
dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)

##############################################################################
# 4. Dataset: 한 이미지를 로드 -> (clip_tensor, dino_tensor, label) 반환
##############################################################################
class MultiFolderDualDataset(Dataset):
    """
    - 여러 train.X 폴더 내 (클래스 폴더 -> 이미지) 구조를 순회
    - 각 이미지를
        * clip_processor로 변환
        * dino_processor로 변환
      하여 각각의 텐서를 동시에 반환
    """
    def __init__(self, folders, clip_processor, dino_processor, id_to_class):
        super().__init__()
        self.folders = folders
        self.clip_processor = clip_processor
        self.dino_processor = dino_processor

        # class_to_idx: 문자열 라벨ID -> 0..N-1
        self.class_to_idx = {
            cls_id: i for i, cls_id in enumerate(sorted(id_to_class.keys()))
        }

        self.samples = []
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

        # (1) CLIP 전용 전처리
        #     -> {"pixel_values": Tensor(...), ...} 반환
        clip_encoded = self.clip_processor(images=image, return_tensors="pt")
        # (2) DINO 전용 전처리
        dino_encoded = self.dino_processor(images=image, return_tensors="pt")

        # shape: [1, 3, H, W] -> squeeze(0)로 [3, H, W]
        clip_tensor = clip_encoded["pixel_values"].squeeze(0)
        dino_tensor = dino_encoded["pixel_values"].squeeze(0)

        label = self.class_to_idx[label_id]
        return clip_tensor, dino_tensor, label


class ImageFolderDualDataset(Dataset):
    """
    - val_dir 내 class 폴더들에 대해 동일한 방식
    """
    def __init__(self, folder, clip_processor, dino_processor, id_to_class):
        super().__init__()
        self.clip_processor = clip_processor
        self.dino_processor = dino_processor

        self.class_to_idx = {
            cls_id: i for i, cls_id in enumerate(sorted(id_to_class.keys()))
        }

        self.samples = []
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

        # CLIP
        clip_encoded = self.clip_processor(images=image, return_tensors="pt")
        # DINO
        dino_encoded = self.dino_processor(images=image, return_tensors="pt")

        clip_tensor = clip_encoded["pixel_values"].squeeze(0)
        dino_tensor = dino_encoded["pixel_values"].squeeze(0)

        label = self.class_to_idx[label_id]
        return clip_tensor, dino_tensor, label

##############################
# 5. DataLoader
##############################
train_dataset = MultiFolderDualDataset(
    folders=train_dirs,
    clip_processor=clip_processor,
    dino_processor=dino_processor,
    id_to_class=id_to_class
)
val_dataset = ImageFolderDualDataset(
    folder=val_dir,
    clip_processor=clip_processor,
    dino_processor=dino_processor,
    id_to_class=id_to_class
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


########################################
# 6. (CLIP + DINOv2) + S2 모델 정의
########################################
class DualEncoderLinearProbe_S2(nn.Module):
    """
    1) CLIP 인코더 (freeze)
    2) DINO 인코더 (freeze)
    - 각각 S2 multiscale_forward -> CLS 임베딩
    - concat -> Linear
    """
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        clip_scales=[1, 2],
        clip_num_prefix=1,  # CLIP: CLS 토큰 1개

        dino_model_name="facebook/dinov2-base",
        dino_scales=[1, 2],
        dino_num_prefix=1,  # DINO: CLS 토큰 1개

        num_classes=100
    ):
        super().__init__()

        # 6.1) CLIP 인코더
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        clip_hidden_dim = self.clip_model.config.hidden_size  # 예: 768

        self.clip_scales = clip_scales
        self.clip_num_prefix = clip_num_prefix

        # 6.2) DINO 인코더
        self.dino_model = Dinov2Model.from_pretrained(dino_model_name)
        for param in self.dino_model.parameters():
            param.requires_grad = False
        dino_hidden_dim = self.dino_model.config.hidden_size  # 예: 768 (dinov2-base)

        self.dino_scales = dino_scales
        self.dino_num_prefix = dino_num_prefix

        # 최종 임베딩 차원
        out_dim = (clip_hidden_dim * len(clip_scales)) + (dino_hidden_dim * len(dino_scales))
        self.linear = nn.Linear(out_dim, num_classes)

    def forward_clip(self, x):
        # CLIP forward -> last_hidden_state
        return self.clip_model(x).last_hidden_state

    def forward_dino(self, x):
        # DINOv2 forward -> last_hidden_state
        # 주의: DINOv2는 forward()에 pixel_values=인자가 필요
        #       여기서는 x가 이미 [B, 3, H, W] 형태로 들어옴
        return self.dino_model(pixel_values=x).last_hidden_state

    def forward(self, clip_x, dino_x):
        # (1) CLIP + S2
        with torch.no_grad():
            clip_out = multiscale_forward(
                forward_fn=self.forward_clip,
                x=clip_x,
                scales=self.clip_scales,
                num_prefix_token=self.clip_num_prefix
            )
            # clip_out: [B, seq_len, clip_hidden_dim * len(scales)]
            clip_cls = clip_out[:, 0, :]

        # (2) DINO + S2
        with torch.no_grad():
            dino_out = multiscale_forward(
                forward_fn=self.forward_dino,
                x=dino_x,
                scales=self.dino_scales,
                num_prefix_token=self.dino_num_prefix
            )
            # dino_out: [B, seq_len, dino_hidden_dim * len(scales)]
            dino_cls = dino_out[:, 0, :]

        # (3) concat -> Linear
        concat_feat = torch.cat([clip_cls, dino_cls], dim=1)
        logits = self.linear(concat_feat)
        return logits


###################################
# 7. 학습(Linear Probing) 함수 정의
###################################
def train_dual_encoder_probe(
    model,
    train_loader,
    val_loader,
    epochs=EPOCHS,
    lr=LR,
    device=DEVICE
):
    criterion = nn.CrossEntropyLoss()
    # 인코더는 freeze, linear만 학습
    optimizer = optim.Adam(model.linear.parameters(), lr=lr)

    model.to(device)
    best_acc = 0.0

    for epoch in range(epochs):
        #----------------------
        # 7.1) Train Loop
        #----------------------
        model.train()
        train_loss = 0.0
        for clip_imgs, dino_imgs, labels in train_loader:
            clip_imgs = clip_imgs.to(device)
            dino_imgs = dino_imgs.to(device)
            labels    = labels.to(device)

            optimizer.zero_grad()
            outputs = model(clip_imgs, dino_imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)

        train_loss_epoch = train_loss / len(train_loader.dataset)

        #----------------------
        # 7.2) Validation Loop
        #----------------------
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for clip_imgs, dino_imgs, labels in val_loader:
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
# 8. 실행 (main)
##############################
if __name__ == "__main__":
    model = DualEncoderLinearProbe_S2(
        clip_model_name="openai/clip-vit-base-patch32",
        clip_scales=[1,2],
        clip_num_prefix=1,

        dino_model_name="facebook/dinov2-base",
        dino_scales=[1,2],
        dino_num_prefix=1,

        num_classes=num_classes
    )

    train_dual_encoder_probe(
        model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        lr=LR,
        device=DEVICE
    )
