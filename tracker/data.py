import os
import random
import numpy as np
import torch 
import torch.nn as nn
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms,models,datasets

from sklearn.model_selection import train_test_split

SEED = 42
BATCH_SIZE = 16
NUM_CLASSES = 4
IMAGE_SIZE = 260

CLASS_NAMES = [
    "HAPPY", "NEUTRAL", "CONFUSED", "FRUSTRATED"
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),

    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "game_data"

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "validation",
    transform=val_transform
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Classes:", train_dataset.classes)
print("Class mapping:", train_dataset.class_to_idx)
print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))