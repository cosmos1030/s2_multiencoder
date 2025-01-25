import os
import json
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoImageProcessor

class MultiFolderDualDataset(Dataset):
    """
    여러 'train.X' 폴더 구조에서 이미지 + (CLIP/DINOv2) 전처리
    """
    def __init__(self, folders, clip_processor, dino_processor, id_to_class):
        super().__init__()
        self.samples = []
        self.clip_processor = clip_processor
        self.dino_processor = dino_processor
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
        clip_pixels = clip_enc["pixel_values"].squeeze(0)

        # DINOv2 전처리
        dino_enc = self.dino_processor(images=image, return_tensors="pt")
        dino_pixels = dino_enc["pixel_values"].squeeze(0)

        label = self.class_to_idx[label_id]
        return clip_pixels, dino_pixels, label

class ImageFolderDualDataset(Dataset):
    """
    단일 폴더 (예: test_dir) 구조에서 이미지 + (CLIP/DINOv2) 전처리
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
        clip_pixels = clip_enc["pixel_values"].squeeze(0)

        dino_enc = self.dino_processor(images=image, return_tensors="pt")
        dino_pixels = dino_enc["pixel_values"].squeeze(0)

        label = self.class_to_idx[label_id]
        return clip_pixels, dino_pixels, label

def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    id_to_class = {str(k): v for k, v in labels.items()}
    return id_to_class
