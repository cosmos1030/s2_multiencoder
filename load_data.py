import json
import os
import random
import matplotlib.pyplot as plt
import cv2

# Load labels from JSON file
# with open('/home/pv/Project/s2/imagenet100/Labels.json', 'r') as f:
with open('/home/dyk6208/Projects/s2_multiencoder/imagenet100/Labels.json', 'r') as f:
    labels = json.load(f)

# Create a mapping from class ID to class name
id_to_class = {str(k): v for k, v in labels.items()}

# Function to display an image with its label
# def show_image_with_label(image_path, label):
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     plt.imshow(image)
#     plt.title(f'Label: {label}')
#     plt.axis('off')
#     plt.show()

# Directories containing the images
# base_dir = '/home/pv/Project/s2/imagenet100'
base_dir = '/home/dyk6208/Projects/s2_multiencoder/imagenet100'


val_dir = os.path.join(base_dir, 'val.X')
train_dirs = [os.path.join(base_dir, f'train.X{i}') for i in range(1, 5)]

# Helper function to get a random sample of images from a directory
# def get_random_images(data_dirs, num_samples=5):
#     images = []
#     all_files = []
#     for data_dir in data_dirs:
#         for label_id in os.listdir(data_dir):
#             class_name = id_to_class.get(label_id, 'Unknown')
#             files = os.listdir(os.path.join(data_dir, label_id))
#             all_files.extend([(os.path.join(data_dir, label_id, filename), class_name) for filename in files])

#     selected_files = random.sample(all_files, num_samples)
#     for image_path, class_name in selected_files:
#         images.append((image_path, class_name))
#     return images

# # Get random images from validation set
# val_images = get_random_images([val_dir], 5)

# # Get random images from all training directories
# train_images = get_random_images(train_dirs, 5)

# # Display random validation images
# print("Random Validation Images:")
# for image_path, label in val_images:
#     show_image_with_label(image_path, label)

# # Display random training images
# print("Random Training Images:")
# for image_path, label in train_images:
#     show_image_with_label(image_path, label)

# Creating the transforms
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create a custom dataset class
class MultiFolderDataset(Dataset):
    def __init__(self, folders, transform=None):
        self.samples = []
        self.transform = transform
        for folder in folders:
            for label_id in os.listdir(folder):
                class_name = id_to_class.get(label_id, 'Unknown')
                files = os.listdir(os.path.join(folder, label_id))
                self.samples.extend([(os.path.join(folder, label_id, filename), label_id) for filename in files])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(id_to_class.keys()))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_idx[label]
        return image, label

    
import torchvision.datasets as datasets
# Creating the Datasets
train_dataset = MultiFolderDataset(train_dirs, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)