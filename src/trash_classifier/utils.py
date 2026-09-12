"""Small standalone helpers shared across the pipeline."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python/numpy/torch RNGs for reproducible folds and weight init."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def plot_training_curves(history_df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot Loss vs Epoch, F1 vs Epoch, and LR vs Epoch, and save the figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        epochs = history_df["epoch"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Loss vs Epoch
        axes[0].plot(epochs, history_df["train_loss"], label="Train Loss", marker="o")
        axes[0].plot(epochs, history_df["val_loss"], label="Val Loss", marker="o")
        axes[0].set_title("Loss vs Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)

        # 2. F1 vs Epoch
        axes[1].plot(epochs, history_df["train_f1"], label="Train F1-Macro", marker="o")
        axes[1].plot(epochs, history_df["val_f1"], label="Val F1-Macro", marker="o")
        axes[1].set_title("F1-Macro vs Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("F1-Macro")
        axes[1].legend()
        axes[1].grid(True)

        # 3. LR vs Epoch
        axes[2].plot(epochs, history_df["lr"], label="Learning Rate", color="g", marker="o")
        axes[2].set_title("Learning Rate vs Epoch")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Gagal menggambar kurva: %s", e)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], output_path: str | Path) -> None:
    """Plot and save confusion matrix using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import itertools

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix (OOF)')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Draw values in the boxes
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Gagal membuat plot confusion matrix: %s", e)


def update_experiment_summary(summary_csv_path: str | Path, exp_name: str, folds: int, oof_f1: float, config_dict: dict) -> None:
    """Updates a global experiment summary CSV with results of the current run."""
    import os
    import csv
    from pathlib import Path
    
    summary_csv_path = Path(summary_csv_path)
    file_exists = os.path.exists(summary_csv_path)
    
    row = {
        "Exp": exp_name,
        "Folds": folds,
        "OOF F1": f"{oof_f1:.4f}",
        "Models": ",".join(config_dict.get("models", [])),
        "Seed": config_dict.get("seed", 42),
        "Scheduler": config_dict.get("scheduler", "cosine"),
        "ProgressiveFT": config_dict.get("progressive_ft", True)
    }
    
    headers = ["Exp", "Folds", "OOF F1", "Models", "Seed", "Scheduler", "ProgressiveFT"]
    
    try:
        # Create directory if not exists
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Gagal menulis/memperbarui file experiment_summary.csv: %s", e)



