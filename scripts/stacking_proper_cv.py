"""Stacking meta-learner (LogisticRegression/RidgeClassifier) via StratifiedKFold PROPER,
CPU-only, murni dari OOF probs train (train_idx fit, val_idx predict -- tidak pernah lihat
sampelnya sendiri). solution.csv HANYA diagnostik info-only di akhir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score

BASE = Path(__file__).resolve().parent.parent
CLASSES = ["0_Recyclable", "1_Electronic", "2_Organic"]

OOF_PATHS = {
    "conv_l": "outputs/_reindexed_convnextv2_large_fast2f_oof.npy",
    "conv_b": "outputs/_reindexed_convnextv2_base_v3clean_oof.npy",
    "swin_b": "outputs/_reindexed_swinv2_base_v3clean_oof.npy",
    "eva02_full": "outputs/full_eva02/run1/cache/cache_eva02_base_patch14_224_mim_in22k_81e63d15_oof.npy",
    "swinv2_large": "outputs/cache_swinv2_large_window12to16_192to256_ms_in22k_ft_in1k_5d671fe2_oof.npy",
    "siglip_384": "outputs/exp_v5_siglip_384/cache/cache_vit_base_patch16_siglip_384_9abc1159_oof.npy",
    "swin384_fast": "outputs/cache_fast_highres_swin384_oof.npy",
}
TTA_TEST_PATHS = {
    "conv_l": "outputs/tta/cache_conv_l_tta5_test.npy",
    "conv_b": "outputs/tta/cache_conv_b_tta5_test.npy",
    "swin_b": "outputs/tta/cache_swin_b_tta5_test.npy",
    "eva02_full": "outputs/tta/cache_eva02_full_tta5_test.npy",
    "swinv2_large": "outputs/tta/cache_swinv2_large_tta5_test.npy",
    "siglip_384": "outputs/tta/cache_siglip_384_tta5_test.npy",
    "swin384_fast": "outputs/tta/cache_swin384_fast_tta5_test.npy",
}
TTA_IDS_PATHS = {k: v.replace("_test.npy", "_ids.npy") for k, v in TTA_TEST_PATHS.items()}
MODEL_KEYS = list(OOF_PATHS.keys())


def load_catalog_labels() -> np.ndarray:
    cat = pd.read_csv(BASE / "outputs" / "_v3_catalog_order.csv")
    return cat["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()


def load_oof_matrix() -> np.ndarray:
    arrs = [np.load(BASE / OOF_PATHS[k]) for k in MODEL_KEYS]
    for a, k in zip(arrs, MODEL_KEYS):
        assert a.shape[0] == 25693, f"{k}: {a.shape}"
    return np.concatenate(arrs, axis=1)  # (n_samples, n_models*3)


def load_test_matrix(order_ids: np.ndarray) -> np.ndarray:
    out = []
    for k in MODEL_KEYS:
        probs = np.load(BASE / TTA_TEST_PATHS[k])
        ids = np.load(BASE / TTA_IDS_PATHS[k])
        idx = {int(i): j for j, i in enumerate(ids)}
        reorder = np.array([idx[int(i)] for i in order_ids])
        out.append(probs[reorder])
    return np.concatenate(out, axis=1)


def nested_cv_stack_score(X, y, clf_builder, n_splits=5, seed=42):
    """Proper nested CV: fit meta-learner on train_idx fold, predict val_idx fold. Never sees val labels."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(y), dtype=int)
    for train_idx, val_idx in skf.split(X, y):
        clf = clf_builder()
        clf.fit(X[train_idx], y[train_idx])
        oof_pred[val_idx] = clf.predict(X[val_idx])
    return f1_score(y, oof_pred, average="macro"), oof_pred


def main():
    y_train = load_catalog_labels()
    X = load_oof_matrix()
    print(f"X shape: {X.shape} (n_samples, {len(MODEL_KEYS)} models x 3 classes)")

    print("=" * 80)
    print("[baseline] equal-weight + threshold OOF-F1 = 0.99286 (referensi)")

    print("\n[stacking] LogisticRegression meta-learner, 5-fold nested CV (proper, no leakage)")
    for C in [0.1, 1.0, 10.0]:
        score, _ = nested_cv_stack_score(
            X, y_train, lambda C=C: LogisticRegression(max_iter=2000, C=C)
        )
        print(f"    C={C:<6} nested-CV macro-F1 = {score:.5f}")

    print("\n[stacking] RidgeClassifier meta-learner, 5-fold nested CV")
    for alpha in [0.1, 1.0, 10.0]:
        score, _ = nested_cv_stack_score(X, y_train, lambda alpha=alpha: RidgeClassifier(alpha=alpha))
        print(f"    alpha={alpha:<6} nested-CV macro-F1 = {score:.5f}")

    print("\n" + "=" * 80)
    print("Kesimpulan: bandingkan angka di atas vs baseline 0.99286.")
    print("Jika TIDAK ada yg > 0.99286 + margin wajar (~0.001), stacking TIDAK worth dipakai --")
    print("model linear tidak menemukan interaksi non-linear yg berguna, dan risiko overfit ke 25693")
    print("sampel (n_models*3=21 fitur) lebih besar drpd potensi gain-nya.")


if __name__ == "__main__":
    main()
