"""Audit kandidat salah-label lewat cleanlab, memakai OOF probs dari SingleModelTrainer.

Memanggil `SingleModelTrainer.run()` apa adanya (satu backbone, K-fold seperti
biasa) semata untuk mendapatkan out-of-fold probabilities -- tidak ada
config/trainer training yang diubah, tidak ada bobot model atau submission
yang dihasilkan/disimpan di sini.
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from ..catalog import ImageCatalog
from ..config import PipelineConfig
from ..trainer import SingleModelTrainer

logger = logging.getLogger(__name__)


def audit_label_issues(config: PipelineConfig, catalog: ImageCatalog, model_name: str) -> pd.DataFrame:
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores

    label_mapping = catalog.label_mapping
    classes = catalog.classes
    df = catalog.dataframe
    y_true = df["label"].map(label_mapping).values

    class_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_true),
        dtype=torch.float32, device=config.device,
    )

    # test_df cuma dibutuhkan karena SingleModelTrainer.run() menghasilkan test_probs
    # sekalian -- kita abaikan hasilnya, cukup buat dari nama file yang ada.
    test_ids = sorted(p.stem for p in Path(config.test_dir).glob("*.*"))
    test_df = pd.DataFrame({"id": test_ids})

    skf = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)
    trainer = SingleModelTrainer(config)
    logger.info("Melatih %s (%d epoch x %d fold) untuk OOF probs...", model_name, config.num_epochs, config.k_folds)
    oof_probs, _, _ = trainer.run(model_name, df, label_mapping, class_weights, test_df, skf)

    issue_idx = find_label_issues(labels=y_true, pred_probs=oof_probs, return_indices_ranked_by="self_confidence")
    quality_scores = get_label_quality_scores(labels=y_true, pred_probs=oof_probs)

    pred_label_idx = oof_probs.argmax(axis=1)
    report = pd.DataFrame({
        "filepath": df["filepath"],
        "given_label": df["label"],
        "predicted_label": [classes[i] for i in pred_label_idx],
        "label_quality_score": quality_scores,
        "oof_prob_given_label": oof_probs[np.arange(len(df)), y_true],
        "oof_prob_predicted_label": oof_probs[np.arange(len(df)), pred_label_idx],
        "is_flagged_label_issue": False,
    })
    report.loc[issue_idx, "is_flagged_label_issue"] = True
    return report.sort_values("label_quality_score").reset_index(drop=True)
