import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    id_to_class = {str(k): v for k, v in labels.items()}
    return id_to_class

def safe_stack(tensor_list):
    if all(t is None for t in tensor_list):
        return None
    valid_tensors = [t for t in tensor_list if t is not None]
    return torch.stack(valid_tensors, dim=0)

def dual_collate_fn(batch):
    """
    batch: [(clip_tensor_or_None, dino_tensor_or_None, label), ...]
    """
    clip_list, dino_list, label_list = zip(*batch)
    clip_imgs = safe_stack(clip_list)
    dino_imgs = safe_stack(dino_list)
    labels = torch.tensor(label_list, dtype=torch.long)
    return clip_imgs, dino_imgs, labels


# ----------------------------
# ImageNet-100 전용 Dataset
# ----------------------------
class MultiFolderDualDataset(Dataset):
    """
    여러 'train.X' 폴더에서 이미지 로드 + (CLIP/DINO) 전처리
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

        clip_pixels = None
        if self.clip_processor is not None:
            clip_enc = self.clip_processor(images=image, return_tensors="pt")
            clip_pixels = clip_enc["pixel_values"].squeeze(0)

        dino_pixels = None
        if self.dino_processor is not None:
            dino_enc = self.dino_processor(images=image, return_tensors="pt")
            dino_pixels = dino_enc["pixel_values"].squeeze(0)

        label = self.class_to_idx[label_id]
        return clip_pixels, dino_pixels, label


class ImageFolderDualDataset(Dataset):
    """
    단일 폴더 (ex: val.X) 구조에서 이미지 로드 + (CLIP/DINO) 전처리
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

        clip_pixels = None
        if self.clip_processor is not None:
            clip_enc = self.clip_processor(images=image, return_tensors="pt")
            clip_pixels = clip_enc["pixel_values"].squeeze(0)

        dino_pixels = None
        if self.dino_processor is not None:
            dino_enc = self.dino_processor(images=image, return_tensors="pt")
            dino_pixels = dino_enc["pixel_values"].squeeze(0)

        label = self.class_to_idx[label_id]
        return clip_pixels, dino_pixels, label


# ----------------------------
# CIFAR-100 전용 Dataset
# ----------------------------
class CIFAR100DualDataset(Dataset):
    """
    torchvision.datasets.CIFAR100 로드 + (CLIP/DINO) 전처리
    """
    def __init__(self, split="train", clip_processor=None, dino_processor=None, download=True):
        super().__init__()
        # train=True/False 에 따라 split 결정
        is_train = (split == "train")

        # torchvision CIFAR100
        self.dataset = datasets.CIFAR100(
            root="./cifar_data", 
            train=is_train, 
            download=download
        )

        self.clip_processor = clip_processor
        self.dino_processor = dino_processor

        # CIFAR-100 classes: 100개
        # ['apple','aquarium_fish','baby','bear','beaver', ... 'worm']
        self.id_to_class = {str(i): c for i, c in enumerate(self.dataset.classes)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]  # (PIL Image, int)
        
        clip_pixels = None
        if self.clip_processor is not None:
            clip_enc = self.clip_processor(images=img, return_tensors="pt")
            clip_pixels = clip_enc["pixel_values"].squeeze(0)

        dino_pixels = None
        if self.dino_processor is not None:
            dino_enc = self.dino_processor(images=img, return_tensors="pt")
            dino_pixels = dino_enc["pixel_values"].squeeze(0)

        return clip_pixels, dino_pixels, label


# ----------------------------
# CIFAR-10 전용 Dataset
# ----------------------------
class CIFAR10DualDataset(Dataset):
    """
    torchvision.datasets.CIFAR100 로드 + (CLIP/DINO) 전처리
    """
    def __init__(self, split="train", clip_processor=None, dino_processor=None, download=True):
        super().__init__()
        # train=True/False 에 따라 split 결정
        is_train = (split == "train")

        # torchvision CIFAR10
        self.dataset = datasets.CIFAR10(
            root="./cifar_data", 
            train=is_train, 
            download=download
        )

        self.clip_processor = clip_processor
        self.dino_processor = dino_processor

        # CIFAR-100 classes: 100개
        # ['apple','aquarium_fish','baby','bear','beaver', ... 'worm']
        self.id_to_class = {str(i): c for i, c in enumerate(self.dataset.classes)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]  # (PIL Image, int)
        
        clip_pixels = None
        if self.clip_processor is not None:
            clip_enc = self.clip_processor(images=img, return_tensors="pt")
            clip_pixels = clip_enc["pixel_values"].squeeze(0)

        dino_pixels = None
        if self.dino_processor is not None:
            dino_enc = self.dino_processor(images=img, return_tensors="pt")
            dino_pixels = dino_enc["pixel_values"].squeeze(0)

        return clip_pixels, dino_pixels, label
