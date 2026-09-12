"""Image transform pipelines for train/eval using Albumentations."""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int, train: bool, rotate_limit: int = 15) -> A.Compose:
    """Train: advanced Albumentations (crop, flip, color jitter, CLAHE, blur, coarse dropout).
    Eval: plain resize + normalize.
    """
    if train:
        return A.Compose([
            # 1. Resize & Crop
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), p=1.0),
            
            # 2. Flips (Horizontal flip is primary; we can add vertical flip option or keep it moderate)
            A.HorizontalFlip(p=0.5),
            
            # 3. Kecerahan, Kontras, Saturasi, & Hue
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            
            # 4. Kestabilan Kontras Spasial (CLAHE)
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            
            # 5. Geometri Affine (Rotasi/Skala)
            A.Affine(translate_percent=(-0.0625, 0.0625), scale=(0.9, 1.1), rotate=(-rotate_limit, rotate_limit), border_mode=0, p=0.5),
            
            # 6. Blur acak
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            
            # 7. Coarse Dropout (Simulasi objek terpotong/sampah lain menutupi)
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.02, 0.1), hole_width_range=(0.02, 0.1), p=0.3),
            
            # 8. Normalisasi ImageNet
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            
            # 9. Konversi ke PyTorch Tensor
            ToTensorV2(),
        ])
        
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

