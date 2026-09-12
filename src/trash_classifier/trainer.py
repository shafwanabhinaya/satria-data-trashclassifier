"""Trains one backbone across K stratified folds and collects OOF + test predictions."""
from __future__ import annotations

import gc
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .datasets import IndexedDataset, TestDataset, TrainValDataset
from .models import ModelFactory
from .transforms import build_transforms
from .utils import plot_training_curves

logger = logging.getLogger(__name__)


class SingleModelTrainer:
    """Runs K-fold partial fine-tuning for a single architecture end to end."""

    def __init__(self, config, model_factory: ModelFactory | None = None):
        self.config = config
        self.model_factory = model_factory or ModelFactory(config.device)
        self._use_amp = config.device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self._use_amp else None

    def run(self, model_name: str, catalog_df: pd.DataFrame, label_mapping: dict,
            class_weights: torch.Tensor, test_df: pd.DataFrame, skf):
        """Returns (oof_probs, test_probs, test_ids) for `model_name`.

        `oof_probs` is aligned to `catalog_df`'s row order; `test_probs` is the
        fold-averaged (and optionally TTA-averaged) softmax over `test_df`.
        """
        cfg = self.config
        num_classes = len(label_mapping)
        img_size = cfg.img_size[model_name]

        oof_probs = np.zeros((len(catalog_df), num_classes))
        test_probs = np.zeros((len(test_df), num_classes))
        test_ids: list = []

        test_loader = self._build_test_loader(test_df, img_size)

        config_hash = cfg.get_model_config_hash(model_name)
        safe_model_name = model_name.replace("/", "_").replace(".", "_")

        for fold, (train_idx, val_idx) in enumerate(skf.split(catalog_df, catalog_df["label"]), start=1):
            fold_cache_path = cfg.cache_dir / f"cache_{safe_model_name}_{config_hash}_fold{fold}.npz"
            model_dir = cfg.experiment_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{safe_model_name}_fold{fold}.pth"

            # Check if fold is already completed
            if fold_cache_path.exists() and model_path.exists():
                logger.info("       [RESUME] Memuat hasil FOLD %d dari cache...", fold)
                data = np.load(fold_cache_path, allow_pickle=True)
                best_val_orig_idx = data["val_orig_idx"]
                best_val_probs = data["val_probs"]
                oof_probs[best_val_orig_idx] = best_val_probs
                
                fold_test_probs = data["test_probs"]
                test_probs += fold_test_probs / cfg.k_folds
                if not test_ids:
                    test_ids = data["test_ids"].tolist()
                continue

            logger.info("  ---> FOLD %d/%d", fold, cfg.k_folds)

            train_loader = self._build_train_loader(catalog_df.iloc[train_idx], label_mapping, img_size)
            val_loader = self._build_val_loader(catalog_df.iloc[val_idx], label_mapping, img_size, val_idx)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)

            epoch_cache_path = cfg.cache_dir / f"checkpoint_{safe_model_name}_{config_hash}_fold{fold}_epoch.pth"
            resumed_stage2 = epoch_cache_path.exists()

            # Stage 1: Freeze whole backbone, only head is trainable
            if cfg.use_progressive_ft and not resumed_stage2:
                logger.info("       [Stage 1] Inisialisasi model dengan backbone beku...")
                model, head_params, backbone_params = self.model_factory.build(
                    model_name, num_classes, 0  # 0 unfreeze blocks (completely frozen)
                )
                if cfg.use_ema:
                    from timm.utils import ModelEmaV2
                    model_ema = ModelEmaV2(model, decay=0.9999)
                    logger.info("       [EMA] Inisialisasi ModelEmaV2 untuk Stage 1...")
                else:
                    model_ema = None
                
                # Build Stage 1 optimizer (only training head_params)
                optimizer_s1, scheduler_s1 = self.model_factory.build_optimizer(
                    model_name, head_params, [],
                    cfg.head_lr[model_name], cfg.backbone_lr_factor, cfg.stage1_epochs,
                    scheduler_type=cfg.scheduler_type, train_loader_len=len(train_loader)
                )
                
                logger.info("       [Stage 1] Melatih classifier head selama %d epoch...", cfg.stage1_epochs)
                for epoch in range(1, cfg.stage1_epochs + 1):
                    train_loss, train_f1 = self._train_epoch(model, train_loader, criterion, optimizer_s1, scheduler_s1, cfg.grad_accum_steps, model_ema=model_ema)
                    val_loss, val_probs, val_orig_idx = self._evaluate_epoch(model, val_loader, criterion, model_ema=model_ema)
                    
                    if not isinstance(scheduler_s1, torch.optim.lr_scheduler.OneCycleLR):
                        scheduler_s1.step()
                        
                    current_lr = optimizer_s1.param_groups[0]['lr']
                    y_val = catalog_df["label"].map(label_mapping).values[val_orig_idx]
                    val_f1 = f1_score(y_val, np.argmax(val_probs, axis=1), average="macro")
                    logger.info(
                        "       [Stage 1] Epoch %d/%d | Train Loss: %.4f | Train F1: %.4f | Val Loss: %.4f | Val F1: %.4f | LR: %.6f",
                        epoch, cfg.stage1_epochs, train_loss, train_f1, val_loss, val_f1, current_lr
                    )

            # Stage 2 Setup / Standard Training Setup
            if cfg.use_progressive_ft:
                if not resumed_stage2:
                    logger.info("       [Stage 2] Unfreeze backbone...")
                    # Free Stage 1 optimizer memory and empty CUDA cache
                    del optimizer_s1, scheduler_s1
                    if torch.cuda.is_available():
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                    if cfg.stage2_unfreeze_all:
                        for p in model.parameters():
                            p.requires_grad = True
                    else:
                        self.model_factory._unfreeze_backbone_tail(model, model_name, cfg.unfreeze_blocks[model_name])
                        head = model.get_classifier()
                        for p in head.parameters():
                            p.requires_grad = True
                            
                    head = model.get_classifier()
                    head_params = list(head.parameters())
                    head_param_ids = {id(p) for p in head_params}
                    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_param_ids]
                else:
                    logger.info("       [RESUME] Memuat arsitektur model Stage 2 langsung...")
                    model, head_params, backbone_params = self.model_factory.build(
                        model_name, num_classes, cfg.unfreeze_blocks[model_name] if not cfg.stage2_unfreeze_all else 0
                    )
                    if cfg.stage2_unfreeze_all:
                        for p in model.parameters():
                            p.requires_grad = True
                    else:
                        self.model_factory._unfreeze_backbone_tail(model, model_name, cfg.unfreeze_blocks[model_name])
                        head = model.get_classifier()
                        for p in head.parameters():
                            p.requires_grad = True
                            
                    head = model.get_classifier()
                    head_params = list(head.parameters())
                    head_param_ids = {id(p) for p in head_params}
                    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_param_ids]
                    
                    if cfg.use_ema:
                        from timm.utils import ModelEmaV2
                        model_ema = ModelEmaV2(model, decay=0.9999)
                    else:
                        model_ema = None
            else:
                # Standard training (No Stage 1)
                model, head_params, backbone_params = self.model_factory.build(
                    model_name, num_classes, cfg.unfreeze_blocks[model_name]
                )
                if cfg.use_ema:
                    from timm.utils import ModelEmaV2
                    model_ema = ModelEmaV2(model, decay=0.9999)
                    logger.info("       [EMA] Inisialisasi ModelEmaV2 untuk Standard Training...")
                else:
                    model_ema = None
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()

            optimizer, scheduler = self.model_factory.build_optimizer(
                model_name, head_params, backbone_params,
                cfg.head_lr[model_name], cfg.backbone_lr_factor, cfg.num_epochs,
                scheduler_type=cfg.scheduler_type, train_loader_len=len(train_loader)
            )

            best_val_f1 = -1.0
            early_stop_counter = 0
            best_val_probs = None
            best_val_orig_idx = None
            history = []
            start_epoch = 1

            if resumed_stage2:
                try:
                    logger.info("       [RESUME] Memuat progress epoch terakhir dari cache...")
                    checkpoint = torch.load(epoch_cache_path, map_location=cfg.device, weights_only=False)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    if model_ema is not None and "model_ema_state_dict" in checkpoint:
                        model_ema.module.load_state_dict(checkpoint["model_ema_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    
                    best_val_f1 = checkpoint["best_val_f1"]
                    best_val_probs = checkpoint["best_val_probs"]
                    best_val_orig_idx = checkpoint["best_val_orig_idx"]
                    
                    if torch.is_tensor(best_val_probs):
                        best_val_probs = best_val_probs.cpu().numpy()
                    if isinstance(best_val_orig_idx, list):
                        best_val_orig_idx = [x.item() if hasattr(x, "item") else x for x in best_val_orig_idx]
                    elif torch.is_tensor(best_val_orig_idx):
                        best_val_orig_idx = best_val_orig_idx.cpu().numpy()
                    early_stop_counter = checkpoint["early_stop_counter"]
                    history = checkpoint["history"]
                    start_epoch = checkpoint["epoch"] + 1
                    logger.info("       [RESUME] Sukses memuat! Melanjutkan dari Epoch %d", start_epoch)
                except Exception as e:
                    logger.warning("       [RESUME] Gagal memuat checkpoint epoch (%s). Memulai ulang fold ini.", e)

            for epoch in range(start_epoch, cfg.num_epochs + 1):
                train_loss, train_f1 = self._train_epoch(model, train_loader, criterion, optimizer, scheduler, cfg.grad_accum_steps, model_ema=model_ema)
                val_loss, val_probs, val_orig_idx = self._evaluate_epoch(model, val_loader, criterion, model_ema=model_ema)
                
                # Step scheduler per epoch if not OneCycleLR
                if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()

                # Get learning rate
                current_lr = optimizer.param_groups[0]['lr']
                
                # Compute val F1
                y_val = catalog_df["label"].map(label_mapping).values[val_orig_idx]
                val_f1 = f1_score(y_val, np.argmax(val_probs, axis=1), average="macro")
                
                logger.info(
                    "       [Stage 2] Epoch %d/%d | Train Loss: %.4f | Train F1: %.4f | Val Loss: %.4f | Val F1: %.4f | LR: %.6f",
                    epoch, cfg.num_epochs, train_loss, train_f1, val_loss, val_f1, current_lr
                )

                history.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_f1": train_f1,
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                    "lr": current_lr
                })

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_val_probs = val_probs
                    best_val_orig_idx = val_orig_idx
                    early_stop_counter = 0
                    # Save best checkpoint (save EMA weights if available)
                    if model_ema is not None:
                        torch.save(model_ema.module.state_dict(), model_path)
                    else:
                        torch.save(model.state_dict(), model_path)
                    logger.info("       [SAVED] Checkpoint model terbaik diperbarui di: %s", model_path)
                else:
                    early_stop_counter += 1

                # Save intermediate epoch checkpoint to disk
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_f1": best_val_f1,
                    "best_val_probs": best_val_probs,
                    "best_val_orig_idx": best_val_orig_idx,
                    "early_stop_counter": early_stop_counter,
                    "history": history,
                }
                if model_ema is not None:
                    checkpoint["model_ema_state_dict"] = model_ema.module.state_dict()
                torch.save(checkpoint, epoch_cache_path)

                if cfg.early_stopping_patience > 0 and early_stop_counter >= cfg.early_stopping_patience:
                    logger.info("       [EARLY STOPPING] Dihentikan lebih awal pada epoch %d karena F1 tidak membaik.", epoch)
                    break

            # Save history to CSV
            history_df = pd.DataFrame(history)
            history_csv_path = cfg.experiment_dir / f"history_{safe_model_name}_fold{fold}.csv"
            history_df.to_csv(history_csv_path, index=False)
            
            # Plot curves
            curve_path = cfg.experiment_dir / f"curve_{safe_model_name}_fold{fold}.png"
            plot_training_curves(history_df, curve_path)

            # Assign best OOF predictions
            if torch.is_tensor(best_val_probs):
                best_val_probs = best_val_probs.cpu().numpy()
            if isinstance(best_val_orig_idx, list):
                best_val_orig_idx = [x.item() if hasattr(x, "item") else x for x in best_val_orig_idx]
            elif torch.is_tensor(best_val_orig_idx):
                best_val_orig_idx = best_val_orig_idx.cpu().numpy()
            oof_probs[best_val_orig_idx] = best_val_probs
            logger.info("       Fold %d Best macro-F1 (val): %.4f", fold, best_val_f1)

            # Load best model checkpoint for predicting test set
            model.load_state_dict(torch.load(model_path, map_location=cfg.device, weights_only=True))
            fold_test_probs, fold_test_ids = self._predict_probs(model, test_loader, tta=cfg.use_tta)
            test_probs += fold_test_probs / cfg.k_folds
            if not test_ids:
                test_ids = fold_test_ids

            # Save fold-level results to cache
            np.savez(
                fold_cache_path,
                val_orig_idx=best_val_orig_idx,
                val_probs=best_val_probs,
                test_probs=fold_test_probs,
                test_ids=np.array(fold_test_ids)
            )
            logger.info("       [CACHE] Hasil FOLD %d berhasil disimpan ke cache.", fold)

            # Clean up intermediate epoch-level checkpoint to save space
            if epoch_cache_path.exists():
                epoch_cache_path.unlink()

            del model, optimizer, train_loader, val_loader
            if self._use_amp:
                torch.cuda.empty_cache()
            gc.collect()

        return oof_probs, test_probs, test_ids

    # --- loader builders -------------------------------------------------

    def _build_train_loader(self, df: pd.DataFrame, label_mapping: dict, img_size: int) -> DataLoader:
        dataset = TrainValDataset(df, label_mapping, img_size, transform=build_transforms(img_size, train=True, rotate_limit=self.config.rotate_limit))
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers, pin_memory=True)

    def _build_val_loader(self, df: pd.DataFrame, label_mapping: dict, img_size: int, val_idx) -> DataLoader:
        base = TrainValDataset(df, label_mapping, img_size, transform=build_transforms(img_size, train=False))
        dataset = IndexedDataset(base, val_idx.tolist())
        return DataLoader(dataset, batch_size=self.config.batch_size * 2, shuffle=False, num_workers=self.config.num_workers, pin_memory=True)

    def _build_test_loader(self, test_df: pd.DataFrame, img_size: int) -> DataLoader:
        dataset = TestDataset(test_df, self.config.test_dir, img_size, transform=build_transforms(img_size, train=False))
        return DataLoader(dataset, batch_size=self.config.batch_size * 2, shuffle=False, num_workers=self.config.num_workers, pin_memory=True)

    # --- train / inference loops -----------------------------------------

    def _train_epoch(self, model, loader, criterion, optimizer, scheduler=None, accum_steps: int = 1, model_ema=None):
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        
        optimizer.zero_grad(set_to_none=True)
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
            
            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) / accum_steps
                self.scaler.scale(loss).backward()
                if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                    if self.config.grad_clip_max_norm > 0:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip_max_norm, foreach=False)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if model_ema is not None:
                        model_ema.update(model)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels) / accum_steps
                loss.backward()
                if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                    if self.config.grad_clip_max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip_max_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if model_ema is not None:
                        model_ema.update(model)

            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                    scheduler.step()

            total_loss += loss.item() * accum_steps * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        train_f1 = f1_score(all_labels, all_preds, average="macro")
        return avg_loss, train_f1

    @torch.no_grad()
    def _evaluate_epoch(self, model, loader, criterion, model_ema=None):
        eval_model = model_ema.module if model_ema is not None else model
        eval_model.eval()
        total_loss = 0.0
        all_probs, all_ids = [], []
        for batch in loader:
            inputs = batch[0].to(self.config.device)
            labels = batch[1].to(self.config.device)
            batch_ids = batch[-1]
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                outputs = eval_model(inputs)
                loss = criterion(outputs, labels)
                probs = F.softmax(outputs, dim=1)

            total_loss += loss.item() * inputs.size(0)
            all_probs.append(probs.float().cpu().numpy())
            if torch.is_tensor(batch_ids):
                all_ids.extend(batch_ids.cpu().tolist())
            else:
                all_ids.extend(batch_ids)

        avg_loss = total_loss / len(loader.dataset)
        probs = np.concatenate(all_probs, axis=0)
        return avg_loss, probs, all_ids

    @torch.no_grad()
    def _predict_probs(self, model, loader, tta: bool):
        """Softmax probs for every item in `loader`; averages with H-flip and V-flip passes if `tta`."""
        model.eval()
        all_probs, all_ids = [], []
        for batch in loader:
            inputs = batch[0].to(self.config.device)
            batch_ids = batch[-1]
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                probs = F.softmax(model(inputs), dim=1)
                if tta:
                    # Horizontal Flip
                    h_flipped = torch.flip(inputs, dims=[3])
                    probs_h = F.softmax(model(h_flipped), dim=1)
                    
                    if self.config.use_vertical_flip_tta:
                        # Vertical Flip
                        v_flipped = torch.flip(inputs, dims=[2])
                        probs_v = F.softmax(model(v_flipped), dim=1)
                        probs = (probs + probs_h + probs_v) / 3.0
                    else:
                        probs = (probs + probs_h) / 2.0
            all_probs.append(probs.float().cpu().numpy())
            if torch.is_tensor(batch_ids):
                all_ids.extend(batch_ids.cpu().tolist())
            else:
                all_ids.extend(batch_ids)
        return np.concatenate(all_probs, axis=0), all_ids
