# Comprehensive Model Benchmark — Waste Classification

Rangkuman semua model yang sudah dicoba untuk sistem klasifikasi citra sampah (Recyclable / Electronic / Organic), beserta workflow eksplorasi yang dipakai.

---

## 1. Leaderboard Lengkap — Semua Model Individual

Diurutkan dari macro F1 tertinggi ke terendah. Semua hasil di bawah adalah rata-rata 3-fold (kecuali dicatat lain).

| Rank | Model | Kategori Arsitektur | Params | Macro F1 | Status |
|---|---|---|---|---|---|
| 1 | ViT-B/16 SigLIP2 | Transformer (kontrastif image-text) | 92.9M | **0.9874** | Selesai — **juara** |
| 2 | ViT-B/16 SigLIP1 | Transformer (kontrastif image-text) | 92.9M | 0.9863 | Selesai |
| 3 | NaFlex-SigLIP2-Base | Transformer (native aspect ratio) | 92.9M | 0.9836 | Selesai |
| 4 | Swin-Base V2 | Hybrid (window-attention hierarkis) | 88M | 0.9776 | Selesai |
| 5 | ViT-L/16 SWAG | Transformer (weakly-supervised) | 305M | 0.9770 | Selesai |
| 6 | ConvNeXt V2-Base | CNN modern | 89M | 0.9746 | Selesai |
| 7 | ConvNeXt V2-Large | CNN modern | 198M | 0.9720 | Selesai |
| 8 | CoAtNet-2 | Hybrid (conv + attention) | 72.8M | 0.9698 | Selesai |
| 9 | BEiT-Base | Transformer (masked image modeling) | 86M | 0.9695 | Selesai |
| 10 | DINOv2-Base | Transformer (self-supervised) | 86.6M | 0.9685 | Selesai |
| 11 | ResNeXt-101 IG-pretrained | CNN klasik (weakly-supervised Instagram-1B) | 88M | 0.9642 | Selesai |
| 12 | EVA-02 Base | Transformer (MIM + distilasi CLIP) | 86M | 0.9646 | Selesai |
| 13 | ViT-B/16 SWAG | Transformer (weakly-supervised) | 86M | 0.9636 | Selesai |
| 14 | RegNetY-16GF | CNN klasik | 80.6M | 0.9599 | Selesai |
| 15 | PVTv2-B5 | Hybrid (pyramid vision transformer) | 81.4M | 0.9535 | Selesai |
| 16 | DeiT-Base | Transformer (distilasi) | 85.8M | 0.9513 | Selesai |
| 17 | DenseNet-201 | CNN klasik | 18.1M | 0.9403 | Selesai |
| — | ViT-H/14 SWAG | Transformer (weakly-supervised) | 632M | ~0.9766 (fold 1 saja) | **Dihentikan karena lama** — 24+ jam tanpa gain jelas vs model lebih kecil |
| — | EfficientNetV2-L @ res 480 | CNN modern | 119M | ~0.72 (fold 1, epoch 12) | **Dihentikan karena terlalu jelek** — lambat & hasil buruk |
| — | EfficientNetV2-M @ res 224 | CNN modern | 54.1M | — | Dijalankan (isolasi variabel resolusi vs arsitektur) |
| — | ViT-L/16 SigLIP2 | Transformer (kontrastif image-text) | 303M | — | Belum dijalankan |

---

## 2. Workflow Eksplorasi (Tahap Model Individual)

Ini alur yang dipakai untuk SEMUA model di tabel leaderboard di atas — sebelum masuk tahap stacking.

| Aspek | Detail |
|---|---|
| **Metode split** | `StratifiedShuffleSplit` (SSS) — bukan `StratifiedKFold` |
| **Jumlah split** | `n_splits=3` (3-fold) |
| **Proporsi val per fold** | `test_size=0.10` (90% train / 10% validation tiap fold) |
| **Sifat SSS** | Random sampling ulang tiap fold — val set antar fold BISA overlap/tumpang tindih, TIDAK menjamin setiap gambar kebagian jadi validation. Cocok untuk eksplorasi cepat, TIDAK cocok untuk OOF/stacking (tahap stacking memakai `StratifiedKFold` sebagai gantinya). |
| **Random seed** | `SEED = 42`, konsisten di semua notebook |
| **Class weighting** | `compute_class_weight("balanced", ...)` — WAJIB, karena data timpang (Organic ~47%, Recyclable ~38%, Electronic ~15%). Bobot yang selalu muncul: `{'Recyclable': 0.884, 'Electronic': 2.232, 'Organic': 0.704}`. Electronic (kelas minoritas) didenda ~2.2x lebih berat di loss function. |
| **Loss function** | `nn.CrossEntropyLoss(weight=class_weights)` |
| **Strategi fine-tuning** | Backbone freeze (semua parameter pretrained dibekukan) + hanya head/classifier baru yang dilatih (head-only fine-tuning) |
| **Optimizer** | `AdamW`, `LR=1e-3`, `WEIGHT_DECAY=1e-4` |
| **Scheduler** | `ReduceLROnPlateau` (mode="max", factor=0.5, patience=2) — pantau macro F1 val |
| **Epoch maksimum** | `NUM_EPOCHS = 15` |
| **Early stopping** | `PATIENCE = 4` (berhenti kalau 4 epoch berturut-turut tidak ada peningkatan val F1) |
| **Mixed precision** | `torch.amp` (`autocast` + `GradScaler`) — mempercepat training, hemat VRAM |
| **Metrik utama** | macro F1 (`sklearn.metrics.f1_score(..., average="macro")`) — sesuai metrik resmi kompetisi |
| **Output tambahan** | Laporan misclassified 2-sheet Excel (`Detail` + `Summary`) per model, untuk audit/analisis error |

### Resize / Preprocessing Gambar (Tahap Eksplorasi)

| Data | Transform | Kode |
|---|---|---|
| **Training** | Resize agak besar → RandomResizedCrop (crop acak 80–100% area) → flip horizontal/vertikal acak → ColorJitter | `T.Resize((resize_to, resize_to))` → `T.RandomResizedCrop(img_size, scale=(0.8, 1.0))` → `T.RandomHorizontalFlip()` → `T.RandomVerticalFlip()` → `T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)` |
| **Validation (tahap eksplorasi)** | Resize langsung ke kotak persegi (berpotensi distorsi/"gepeng" karena rasio aspek tidak dijaga) | `T.Resize((img_size, img_size))` |

> **Catatan penting:** Validation di tahap eksplorasi (tabel di atas) masih pakai `Resize` langsung ke kotak — berpotensi sedikit distorsi bentuk objek ("gepeng") kalau foto aslinya tidak persegi. Efeknya konsisten di semua model, jadi perbandingan RELATIF antar model tetap valid. Perbaikan (`Resize sisi-pendek + CenterCrop`) baru diterapkan mulai tahap OOF Collection (tahap setelah eksplorasi individual ini).

### Normalisasi (mean/std)

Diambil otomatis per model dari `timm.data.resolve_model_data_config(model)` — beda arsitektur bisa beda mean/std (misal EfficientNetV2 pakai mean/std 0.5 flat, mayoritas ViT/CNN lain pakai ImageNet standar `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`).

### Hardware

| Environment | GPU | VRAM | Catatan |
|---|---|---|---|
| Laptop pribadi | RTX 3050 | ~4.3 GB | Batch kecil (2–8) + gradient accumulation, `NUM_WORKERS=0` |
| Komputer kampus (remote) | RTX 3060 | ~12.9 GB | Batch lebih besar (16–51), `NUM_WORKERS=8` (butuh pola `%%writefile dataset_utils.py` karena Windows `spawn`) |

---

## 3. Ringkasan Insight Kunci

- **Ukuran model BUKAN faktor penentu utama.** Berulang kali terbukti: model lebih kecil dalam satu keluarga mengalahkan versi lebih besar (Swin-Base > semua model Large/Huge; ConvNeXt V2-Base > ConvNeXt V2-Large; ViT-B SigLIP > ViT-L/H SWAG).
- **Metode pretraining paling berpengaruh.** SigLIP (kontrastif image-text) mendominasi puncak leaderboard, mengalahkan SWAG, MIM, dan self-supervised lain di ukuran yang sama.
- **Resolusi tinggi tidak otomatis membantu.** Membantu untuk Swin (native 384), tapi merugikan EfficientNetV2 (lambat, tidak konvergen wajar di 480).
- **Fine-tuning (head-only) mengalahkan frozen-encoder+LogReg** untuk kasus ini — dibuktikan lewat perbandingan langsung: SigLIP2 fine-tune (0.9874) vs SigLIP2 frozen+LogReg baseline (0.9747).
- **CNN klasik (DenseNet, RegNetY, ResNeXt) konsisten kalah** dari CNN modern (ConvNeXt V2) maupun transformer/hybrid.

---

*Benchmark results evaluated across 3-fold cross-validation.*
