import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
from transformers import CLIPVisionModel, AutoProcessor

##############################
# 1. 하이퍼파라미터 & 경로 설정
##############################
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ImageNet100 (사용자 경로 맞게 수정)
base_dir = '/home/dyk6208/Projects/s2_multiencoder/imagenet100'
val_dir = os.path.join(base_dir, 'val.X')
train_dirs = [os.path.join(base_dir, f'train.X{i}') for i in range(1, 5)]
labels_path = os.path.join(base_dir, 'Labels.json')

##################################
# 2. 라벨 로드 & id_to_class 매핑
##################################
with open(labels_path, 'r') as f:
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
                # label_id가 폴더인지(클래스 디렉토리) 확인
                full_path = os.path.join(folder, label_id)
                if not os.path.isdir(full_path):
                    continue
                # 파일 목록
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


# 실제 train, val dataset
train_dataset = MultiFolderDataset(train_dirs, transform=transform)
val_dataset   = datasets.ImageFolder(val_dir, transform=transform)
# 위에서 ImageFolder로 만든 val_dataset에서도 class_to_idx가 자동 생성되니, 
# train_dataset과 idx 매핑이 맞는지 확인 필요 (라벨 구조 동일하다면 문제없음)

##############################
# 4. DataLoader 정의
##############################
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

########################################
# 5. CLIP 모델 + Linear Probe 모델 정의
########################################
class CLIPLinearProbe(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", num_classes=100):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(model_name)
        
        # CLIP Vision Model 파라미터는 freeze
        for param in self.clip.parameters():
            param.requires_grad = False
        
        # CLIP 모델의 hidden_size는 768 (base-patch32 기준)
        self.hidden_size = 768
        
        # 최종 Linear Layer
        self.linear = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        """
        x: [B, 3, 224, 224]
        out.last_hidden_state: [B, seq_len, hidden_size]
        """
        with torch.no_grad():
            out = self.clip(x)
            
        # (1) CLS 토큰(첫 번째 토큰) 사용
        cls_feat = out.last_hidden_state[:, 0, :]
        # (2) 혹은 평균 풀링
        # cls_feat = out.last_hidden_state.mean(dim=1)
        
        # 이제 cls_feat의 grad는 None이지만,
        # 여기서 linear로 넘어갈 때부터는 grad가 생깁니다.
        logits = self.linear(cls_feat)
        return logits


###################################
# 6. 학습(Linear Probing) 함수 정의
###################################
def train_linear_probe(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=DEVICE):
    criterion = nn.CrossEntropyLoss()
    # Linear Layer만 학습하기 위해 model.linear 파라미터만 옵티마이저에 등록
    optimizer = optim.Adam(model.linear.parameters(), lr=lr)

    model.to(device)
    best_acc = 0.0

    for epoch in range(epochs):
        #################################
        # 6.1. Train Loop
        #################################
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)  # forward
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)

        #################################
        # 6.2. Validation Loop
        #################################
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
    # 모델 생성
    num_classes = len(id_to_class)  # 예: 100
    model = CLIPLinearProbe(num_classes=num_classes)

    # 학습 및 평가
    train_linear_probe(model, train_loader, val_loader)
