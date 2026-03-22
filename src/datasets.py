import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from src.config import (
    TRAIN_DIR, TEST_DIR, TEST_CSV, DATA_ROOT,
    IMG_SIZE, BATCH_SIZE, NUM_WORKERS, SEED
)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

class GTSRBTrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        class_folders = [
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        class_folders = sorted(class_folders, key=lambda x: int(x))

        for class_name in class_folders:
            class_id = int(class_name)
            class_path = os.path.join(root_dir, class_name)

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.samples.append((img_path, class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

class GTSRBTestDataset(Dataset):
    def __init__(self, csv_file, test_dir, data_root=None, transform=None):
        self.df = pd.read_csv(csv_file)
        self.test_dir = test_dir
        self.data_root = data_root
        self.transform = transform

        if "Path" in self.df.columns:
            self.image_col = "Path"
        elif "Filename" in self.df.columns:
            self.image_col = "Filename"
        else:
            raise ValueError("Test.csv must contain 'Path' or 'Filename' column.")

        if "ClassId" not in self.df.columns:
            raise ValueError("Test.csv must contain 'ClassId' column.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row["ClassId"])

        if self.image_col == "Path":
            if self.data_root is None:
                raise ValueError("data_root must be provided when CSV uses 'Path'.")
            img_path = os.path.join(self.data_root, row["Path"])
        else:
            img_path = os.path.join(self.test_dir, row["Filename"])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

def create_dataloaders():
    full_train_dataset = GTSRBTrainDataset(TRAIN_DIR, transform=train_transform)
    test_dataset = GTSRBTestDataset(
        TEST_CSV,
        TEST_DIR,
        data_root=DATA_ROOT,
        transform=test_transform
    )

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    split_gen = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=split_gen
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return full_train_dataset, train_loader, val_loader, test_loader
