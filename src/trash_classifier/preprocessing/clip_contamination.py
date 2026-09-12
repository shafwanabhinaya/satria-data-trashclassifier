"""Deteksi kontaminasi gambar train (diagram, chart, teks, grafis) menggunakan model CLIP zero-shot."""
from __future__ import annotations

import logging
from pathlib import Path
import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm

from ..catalog import ImageCatalog

logger = logging.getLogger(__name__)

def audit_clip_contamination(
    catalog: ImageCatalog, 
    model_name: str = "openai/clip-vit-base-patch32", 
    batch_size: int = 64, 
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> pd.DataFrame:
    """Scan seluruh katalog gambar menggunakan CLIP dan hitung skor probabilitas kontaminasi.
    
    Prompts:
      - Trash: "a photo of a trash, garbage, recyclable, or organic waste object" (In-Distribution)
      - Document/Chart: "a document, chart, diagram, or page of text" (Contamination 1)
      - Graphic/Illustration: "a computer graphic, logo, banner, or digital illustration" (Contamination 2)
    """
    from transformers import CLIPProcessor, CLIPModel
    
    logger.info("Loading CLIP model: %s on %s...", model_name, device)
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    prompts = [
        "a photo of trash, garbage, recyclable, or organic waste",
        "a document, chart, diagram, or page of written text",
        "a computer graphic, logo, banner, or digital illustration"
    ]
    
    df = catalog.dataframe
    filepaths = df["filepath"].tolist()
    
    trash_probs = []
    doc_probs = []
    graphic_probs = []
    
    logger.info("Scoring %d images with CLIP...", len(filepaths))
    
    # Nonaktifkan warning tokenizer parallelism agar log tidak berisik
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Gunakan torch.no_grad() untuk efisiensi
    with torch.no_grad():
        # Memproses dalam batch
        for i in tqdm(range(0, len(filepaths), batch_size)):
            batch_paths = filepaths[i:i+batch_size]
            batch_images = []
            valid_indices = []
            
            for idx, p in enumerate(batch_paths):
                try:
                    # Buka gambar secara aman
                    img = Image.open(p).convert("RGB")
                    batch_images.append(img)
                    valid_indices.append(idx)
                except Exception as exc:
                    logger.warning("Gagal membaca gambar %s: %s", p, exc)
                    
            if not batch_images:
                for _ in batch_paths:
                    trash_probs.append(1.0)  # Default aman
                    doc_probs.append(0.0)
                    graphic_probs.append(0.0)
                continue
                
            inputs = processor(text=prompts, images=batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            # Dapatkan probabilitas softmax di sepanjang sumbu teks (dim=1)
            logits_per_image = outputs.logits_per_image  # shape: (batch_size, 3)
            probs = logits_per_image.softmax(dim=1).cpu().numpy()
            
            probs_idx = 0
            for idx in range(len(batch_paths)):
                if idx in valid_indices:
                    p_vals = probs[probs_idx]
                    trash_probs.append(float(p_vals[0]))
                    doc_probs.append(float(p_vals[1]))
                    graphic_probs.append(float(p_vals[2]))
                    probs_idx += 1
                else:
                    trash_probs.append(1.0)
                    doc_probs.append(0.0)
                    graphic_probs.append(0.0)
                    
    report = df.copy()
    report["clip_trash_prob"] = trash_probs
    report["clip_document_prob"] = doc_probs
    report["clip_graphic_prob"] = graphic_probs
    report["clip_contamination_score"] = report["clip_document_prob"] + report["clip_graphic_prob"]
    
    return report.sort_values("clip_contamination_score", ascending=False).reset_index(drop=True)
