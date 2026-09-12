"""torch Dataset wrappers for the train/val/test splits."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F_tf


def _safe_open(path, img_size: int) -> Image.Image:
    """Open an image as RGB; fall back to a blank tile so one corrupt file can't kill a run."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (img_size, img_size))


class TrainValDataset(Dataset):
    """Loads (image, label) pairs from a catalog DataFrame with columns [filepath, label]."""

    def __init__(self, dataframe: pd.DataFrame, label_mapping: dict, img_size: int, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_mapping = label_mapping
        self.img_size = img_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.loc[idx]
        try:
            image = _safe_open(row["filepath"], self.img_size)
            # Resize image if it is too large to save memory
            if max(image.size) > self.img_size * 2:
                image.thumbnail((self.img_size * 2, self.img_size * 2))
            
            # Albumentations expects numpy array
            image_np = np.array(image)
            
            if self.transform:
                transformed = self.transform(image=image_np)
                image_tensor = transformed["image"]
            else:
                image_tensor = F_tf.to_tensor(image)
        except (Exception, MemoryError) as e:
            import logging
            logging.getLogger(__name__).warning("Gagal memuat gambar %s (%s). Menggunakan blank tile.", row["filepath"], e)
            image_tensor = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
            
        return image_tensor, self.label_mapping[row["label"]]


class IndexedDataset(Dataset):
    """Wraps a dataset so each item is paired with its original catalog row index.

    Used for validation folds: `DataLoader` batches/reorders items, so predictions
    need their original index carried along to be scattered into the right OOF slot.
    """

    def __init__(self, base_dataset: Dataset, indices: Sequence[int]):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[idx]
        return image, label, self.indices[idx]


class TestDataset(Dataset):
    """Loads unlabeled test images named after the `id` column of the submission template."""

    def __init__(self, dataframe: pd.DataFrame, test_dir: Path, img_size: int, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.test_dir = Path(test_dir)
        self.img_size = img_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        img_id = self.dataframe.loc[idx, "id"]
        img_name = str(img_id) if str(img_id).endswith(".jpg") else f"{img_id}.jpg"
        filepath = self.test_dir / img_name
        try:
            image = _safe_open(filepath, self.img_size)
            # Resize image if it is too large to save memory
            if max(image.size) > self.img_size * 2:
                image.thumbnail((self.img_size * 2, self.img_size * 2))
                
            # Albumentations expects numpy array
            image_np = np.array(image)
            
            if self.transform:
                transformed = self.transform(image=image_np)
                image_tensor = transformed["image"]
            else:
                image_tensor = F_tf.to_tensor(image)
        except (Exception, MemoryError) as e:
            import logging
            logging.getLogger(__name__).warning("Gagal memuat gambar uji %s (%s). Menggunakan blank tile.", filepath, e)
            image_tensor = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
            
        return image_tensor, img_id
