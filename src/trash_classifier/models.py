"""Backbone construction with partial (discriminative-LR) fine-tuning using timm."""
from __future__ import annotations

import torch
import torch.nn as nn
import timm


class ModelFactory:
    """Membangun backbone pretrained secara dinamis menggunakan timm dengan partial fine-tuning."""

    def __init__(self, device: str):
        self.device = device

    def build(self, model_name: str, num_classes: int, unfreeze_blocks: int):
        """Membangun model dari timm, mengganti head klasifikasi, dan melakukan partial freezing."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        try:
            # dynamic_img_size=True: sebagian ViT (mis. DINOv2) menolak resolusi input yg
            # beda dari native pretraining-nya (assertion error) kecuali flag ini diaktifkan
            # (memicu interpolasi position embedding). Arsitektur yg tidak kenal kwarg ini
            # (ConvNeXt, Swin, dll) akan TypeError -> fallback tanpa flag di bawah.
            model = timm.create_model(model_name, pretrained=True, num_classes=num_classes, dynamic_img_size=True)
        except TypeError:
            model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        except Exception as e:
            import logging
            logging.warning("Load pretrained awal terhenti (%s). Mencoba ulang dengan gc.collect()...", e)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                model = timm.create_model(model_name, pretrained=True, num_classes=num_classes, dynamic_img_size=True)
            except TypeError:
                model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        
        # Bekukan seluruh parameter terlebih dahulu
        for p in model.parameters():
            p.requires_grad = False
            
        # Unfreeze N block terakhir sesuai tipe arsitektur secara dinamis
        self._unfreeze_backbone_tail(model, model_name, unfreeze_blocks)
        
        # Pastikan parameter classifier head baru selalu dapat dilatih
        head = model.get_classifier()
        for p in head.parameters():
            p.requires_grad = True
            
        # Pisahkan parameter head dan backbone untuk discriminative learning rate
        head_params = list(head.parameters())
        head_param_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_param_ids]
        
        return model.to(self.device), head_params, backbone_params

    def _unfreeze_backbone_tail(self, model, model_name: str, n_unfreeze: int):
        """Logika generik unfreeze block berdasarkan arsitektur timm."""
        if n_unfreeze <= 0:
            return
            
        # Unfreeze final pooling and normalization layers (norm, attn_pool, fc_norm)
        # karena layer ini adalah transisi krusial menuju classification head
        for name in ["norm", "attn_pool", "fc_norm"]:
            if hasattr(model, name):
                layer = getattr(model, name)
                if isinstance(layer, nn.Module):
                    for p in layer.parameters():
                        p.requires_grad = True

        # Tipe 1: Vision Transformers standar (vit, deit, beit, siglip, EVA)
        if hasattr(model, "blocks"):
            for block in model.blocks[-n_unfreeze:]:
                for p in block.parameters():
                    p.requires_grad = True
        # Tipe 2: Swin Transformers
        elif hasattr(model, "layers") and not hasattr(model, "stages"):
            for layer in model.layers[-n_unfreeze:]:
                for p in layer.parameters():
                    p.requires_grad = True
        # Tipe 3: ConvNeXt (memiliki model.stages)
        elif hasattr(model, "stages"):
            for stage in model.stages[-n_unfreeze:]:
                for p in stage.parameters():
                    p.requires_grad = True
        # Tipe 4: EfficientNet/MobileNet (memiliki model.blocks sebagai sequential)
        elif hasattr(model, "blocks") and isinstance(model.blocks, nn.Sequential):
            for block in model.blocks[-n_unfreeze:]:
                for p in block.parameters():
                    p.requires_grad = True
        # Fallback default
        else:
            child_modules = list(model.children())
            if len(child_modules) >= 2:
                for p in child_modules[-2].parameters():
                    p.requires_grad = True

    def build_optimizer(self, model_name: str, head_params, backbone_params,
                         head_lr: float, backbone_lr_factor: float, num_epochs: int,
                         scheduler_type: str = "cosine", train_loader_len: int = 100):
        param_groups = [{"params": head_params, "lr": head_lr}]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": head_lr * backbone_lr_factor})

        # ViT menggunakan Adam, arsitektur lain menggunakan AdamW
        if "vit" in model_name:
            optimizer = torch.optim.Adam(param_groups)
        else:
            optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler_type == "cosine_restarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_max=max(1, num_epochs // 2), T_mult=1)
        elif scheduler_type == "one_cycle":
            max_lrs = [group["lr"] for group in param_groups]
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=max_lrs, epochs=num_epochs, steps_per_epoch=train_loader_len
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        return optimizer, scheduler
