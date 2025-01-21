import os
import json
from PIL import Image
from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import CLIPVisionModel, AutoProcessor
from s2wrapper import forward as multiscale_forward


##############################
# 1. 하이퍼파라미터 & 경로 설정
##############################
BATCH_SIZE = 512
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet100 (사용자 환경에 맞게 수정)
base_dir = "/home/dyk6208/Projects/s2_multiencoder/imagenet100"
val_dir = os.path.join(base_dir, "val.X")
train_dirs = [os.path.join(base_dir, f"train.X{i}") for i in range(1, 5)]
labels_path = os.path.join(base_dir, "Labels.json")

##################################
# 2. 라벨 로드 & id_to_class 매핑
##################################
with open(labels_path, "r") as f:
    labels = json.load(f)
# str(class_id) -> class_name
id_to_class = {str(k): v for k, v in labels.items()}

##############################
# 3. 데이터셋 및 전처리 정의
##############################
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# MultiFolderDataset 예시
class MultiFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folders, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(id_to_class.keys()))}
        
        for folder in folders:
            if not os.path.isdir(folder):
                continue
            for label_id in os.listdir(folder):
                full_path = os.path.join(folder, label_id)
                if not os.path.isdir(full_path):
                    continue
                files = os.listdir(full_path)
                for filename in files:
                    img_path = os.path.join(full_path, filename)
                    if os.path.isfile(img_path):
                        self.samples.append((img_path, label_id))
                        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_id = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_idx[label_id]
        return image, label


##############################
# 4. Datasets & Dataloaders
##############################
train_dataset = MultiFolderDataset(train_dirs, transform=transform)
val_dataset   = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

########################################
# 5. CLIP 모델 + Linear Probe (S2 적용)
########################################
class CLIPLinearProbe_S2(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", scales=[1, 2], num_prefix_token=1, num_classes=100):
        """
        scales: [1, 2, ...] 여러 배율
        num_prefix_token: CLS 토큰 개수 (CLIP는 기본 1)
        """
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # CLIP 모델은 freeze (가중치 고정)
        for param in self.clip.parameters():
            param.requires_grad = False
        
        # hidden_size: clip-vit-base-patch32의 경우 768
        self.hidden_size = self.clip.config.hidden_size
        self.scales = scales
        self.num_prefix_token = num_prefix_token
        
        # S2 multiscale 시, 최종 임베딩 차원은 hidden_size * len(scales)
        self.output_dim = self.hidden_size * len(scales)
        
        # Linear Layer: (hidden_size * #scales) -> num_classes
        self.linear = nn.Linear(self.output_dim, num_classes)

    def forward_features(self, inputs):
        # CLIP forward -> last_hidden_state
        # shape: [B, seq_len, hidden_size]
        return self.clip(inputs).last_hidden_state

    def forward(self, x):
        """
        x: [B, 3, 224, 224]
        """
        with torch.no_grad():
            # multiscale_forward(forward_fn, x, scales, num_prefix_token)
            # 결과 shape: [B, seq_len, hidden_size * len(scales)]
            out = multiscale_forward(self.forward_features, x, scales=self.scales, num_prefix_token=self.num_prefix_token)
        
        # CLS 토큰만 사용 (첫 번째 토큰)
        # out: [B, seq_len, hidden_size * #scales]
        # -> out[:, 0, :] : [B, hidden_size * #scales]
        cls_feat = out[:, 0, :]
        
        # Linear 분류
        logits = self.linear(cls_feat)  # [B, num_classes]
        return logits


###################################
# 6. 학습(Linear Probing) 함수 정의
###################################
def train_linear_probe(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=DEVICE):
    criterion = nn.CrossEntropyLoss()
    # Linear Layer만 학습 -> model.linear 파라미터만 옵티마이저에 등록
    optimizer = optim.Adam(model.linear.parameters(), lr=lr)

    model.to(device)
    best_acc = 0.0

    for epoch in range(epochs):
        #######################
        # 6.1. Train Loop
        #######################
        model.train()  # Linear는 train 모드
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)    # forward
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)

        #######################
        # 6.2. Validation Loop
        #######################
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        train_loss_epoch = train_loss / len(train_loader.dataset)
        val_loss_epoch   = val_loss   / len(val_loader.dataset)
        val_acc = correct / total

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss_epoch:.4f} "
              f"Val Loss: {val_loss_epoch:.4f} "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc

    print(f"\nBest Validation Accuracy: {best_acc:.4f}")


##############################
# 7. 실행
##############################
if __name__ == "__main__":
    # num_classes: ImageNet100 이므로 100
    num_classes = len(id_to_class)  # = 100 (가정)
    
    # 모델 생성 (S2)
    # scales=[1,2] -> 최종 feature dim = 768 * 2 = 1536
    # num_prefix_token=1 -> CLIP Vision 모델의 [CLS] 토큰 수가 1개
    model = CLIPLinearProbe_S2(
        model_name="openai/clip-vit-base-patch32",
        scales=[1, 2],
        num_prefix_token=1,
        num_classes=num_classes
    )

    # 학습 및 평가
    train_linear_probe(model, train_loader, val_loader)
