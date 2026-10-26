import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from preprocess import preprocess_dataset


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
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class NPYImageFolder(Dataset):
    """ImageFolder-compatible dataset that reads preprocessed NumPy arrays."""

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted(
            path.name for path in self.root.iterdir() if path.is_dir()
        )
        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.classes)
        }
        self.samples = [
            (path, self.class_to_idx[path.parent.name])
            for class_name in self.classes
            for path in sorted((self.root / class_name).glob("*.npy"))
        ]
        self.targets = [target for _, target in self.samples]

        if not self.samples:
            raise FileNotFoundError(
                f"No .npy files found in dataset directory: {self.root}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image_array = np.load(path, allow_pickle=False)
        image = Image.fromarray(image_array)

        if self.transform is not None:
            image = self.transform(image)

        return image, target


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "game_data"
NPY_DATA_DIR = BASE_DIR / "game_data_npy"

preprocess_dataset(
    data_dir=DATA_DIR,
    output_dir=NPY_DATA_DIR,
    image_size=IMAGE_SIZE,
)

train_dataset = NPYImageFolder(
    NPY_DATA_DIR / "train",
    transform=train_transform,
)
val_dataset = NPYImageFolder(
    NPY_DATA_DIR / "validation",
    transform=val_transform,
)
test_dataset = NPYImageFolder(
    NPY_DATA_DIR / "test",
    transform=val_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("Classes:", train_dataset.classes)
print("Class mapping:", train_dataset.class_to_idx)
print("Train NumPy samples:", len(train_dataset))
print("Validation NumPy samples:", len(val_dataset))
print("Test NumPy samples:", len(test_dataset))
