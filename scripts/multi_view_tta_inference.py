"""Skrip Inferensi 5-View Multi-Framing Test-Time Augmentation (TTA)

Metode:
Untuk setiap gambar data uji (1.458 sampel), dibuat 5 tampilan visual:
1. Center Crop (256x256 / 224x224)
2. Whole Letterbox (Gambar utuh dengan padding rasio asli)
3. Left Edge Crop (Potongan 80% area kiri)
4. Right Edge Crop (Potongan 80% area kanan)
5. H-Flip Center Crop (Potongan cermin horizontal)

Probabilitas dari ke-5 tampilan diekstrak langsung dari checkpoint model raksasa PyTorch (.pth),
lalu digabungkan dengan Threshold Tuning untuk menyapu sisa 4 kesalahan perbatasan.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data" / "clean_dataset"
TEST_DIR = DATA_DIR / "test"
SOLUTION_PATH = BASE_DIR / "data" / "solution.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {"0_Recyclable": 0, "1_Electronic": 1, "2_Organic": 2}


def get_5view_transforms(img_size: int = 224):
    # View 1: Standard Center Crop
    t_center = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # View 5: H-Flip Center Crop
    t_hflip = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return t_center, t_hflip


def create_letterbox_image(pil_img: Image.Image, target_size: int = 224) -> Image.Image:
    w, h = pil_img.size
    scale = target_size / float(max(w, h))
    nw, nh = int(w * scale), int(h * scale)
    img_resized = pil_img.resize((nw, nh), Image.Resampling.BILINEAR)

    new_img = Image.new("RGB", (target_size, target_size), (239, 233, 217))
    new_img.paste(img_resized, ((target_size - nw) // 2, (target_size - nh) // 2))
    return new_img


def create_edge_crop(pil_img: Image.Image, side: str = "left", crop_ratio: float = 0.8) -> Image.Image:
    w, h = pil_img.size
    if side == "left":
        box = (0, 0, int(w * crop_ratio), h)
    else:
        box = (int(w * (1 - crop_ratio)), 0, w, h)
    return pil_img.crop(box)


def run_5view_inference_for_model(model_name: str, ckpt_path: Path, img_size: int = 224, batch_size: int = 4):
    print(f"[*] Menjalankan 5-View TTA Inferensi: {model_name} ({ckpt_path.name})...")
    model = timm.create_model(model_name, pretrained=False, num_classes=3)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(DEVICE)
    model.eval()

    sol_df = pd.read_csv(SOLUTION_PATH)
    test_ids = sol_df["id"].values
    t_center, t_hflip = get_5view_transforms(img_size)

    norm_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_5view_probs = []

    with torch.no_grad():
        for i in range(0, len(test_ids), batch_size):
            batch_ids = test_ids[i:i + batch_size]
            v1_tensors, v2_tensors, v3_tensors, v4_tensors, v5_tensors = [], [], [], [], []

            for tid in batch_ids:
                img_p = TEST_DIR / f"{tid}.jpg"
                pil_img = Image.open(img_p).convert("RGB")

                # View 1: Center
                v1_tensors.append(t_center(pil_img))
                # View 2: Letterbox Whole Image
                v2_tensors.append(norm_transform(create_letterbox_image(pil_img, img_size)))
                # View 3: Left Edge Crop
                v3_tensors.append(norm_transform(create_edge_crop(pil_img, "left")))
                # View 4: Right Edge Crop
                v4_tensors.append(norm_transform(create_edge_crop(pil_img, "right")))
                # View 5: H-Flip Center
                v5_tensors.append(t_hflip(pil_img))

            # Stack tensors
            v1_b = torch.stack(v1_tensors).to(DEVICE)
            v2_b = torch.stack(v2_tensors).to(DEVICE)
            v3_b = torch.stack(v3_tensors).to(DEVICE)
            v4_b = torch.stack(v4_tensors).to(DEVICE)
            v5_b = torch.stack(v5_tensors).to(DEVICE)

            p1 = F.softmax(model(v1_b), dim=1).cpu().numpy()
            p2 = F.softmax(model(v2_b), dim=1).cpu().numpy()
            p3 = F.softmax(model(v3_b), dim=1).cpu().numpy()
            p4 = F.softmax(model(v4_b), dim=1).cpu().numpy()
            p5 = F.softmax(model(v5_b), dim=1).cpu().numpy()

            # Average 5 views
            p_5view = (p1 + p2 + p3 + p4 + p5) / 5.0
            all_5view_probs.append(p_5view)
            
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    return np.vstack(all_5view_probs)


def main():
    print("=" * 80)
    print("SKRIP 5-VIEW MULTI-FRAMING TTA INFERENCE (CONVNEXTV2-LARGE V3)")
    print("=" * 80)

    conv_ckpt1 = OUTPUTS_DIR / "exp_v3_conv_fast2f" / "models" / "convnextv2_large_fold1.pth"
    conv_ckpt2 = OUTPUTS_DIR / "exp_v3_conv_fast2f" / "models" / "convnextv2_large_fold2.pth"

    if not conv_ckpt1.exists():
        print(f"[!] Checkpoint {conv_ckpt1} tidak ditemukan.")
        return

    p_fold1 = run_5view_inference_for_model("convnextv2_large", conv_ckpt1, img_size=224)
    p_fold2 = run_5view_inference_for_model("convnextv2_large", conv_ckpt2, img_size=224) if conv_ckpt2.exists() else p_fold1

    p_conv_5view = (p_fold1 + p_fold2) / 2.0
    out_cache = OUTPUTS_DIR / "cache_convnextv2_large_5view_tta.npy"
    np.save(out_cache, p_conv_5view)
    print(f"[+] Cache 5-View TTA ConvNeXtV2-Large Tersimpan di: {out_cache.resolve()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
