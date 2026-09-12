"""Cari BOBOT per-model (bukan cuma threshold per-kelas) via optimasi OOF murni, CPU-only.
Parametrisasi: softmax(logits) -> weights (otomatis non-negatif & sum=1), + 3 threshold multiplier.
solution.csv HANYA dipakai sbg diagnostik info-only di akhir, TIDAK PERNAH utk fit/tuning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
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
N_MODELS = len(MODEL_KEYS)


def load_catalog_labels() -> np.ndarray:
    cat = pd.read_csv(BASE / "outputs" / "_v3_catalog_order.csv")
    return cat["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()


def load_oof_stack() -> np.ndarray:
    arrs = [np.load(BASE / OOF_PATHS[k]) for k in MODEL_KEYS]
    for a, k in zip(arrs, MODEL_KEYS):
        assert a.shape[0] == 25693, f"{k}: {a.shape}"
    return np.stack(arrs, axis=0)  # (n_models, n_samples, 3)


def load_test_stack(order_ids: np.ndarray) -> np.ndarray:
    out = []
    for k in MODEL_KEYS:
        probs = np.load(BASE / TTA_TEST_PATHS[k])
        ids = np.load(BASE / TTA_IDS_PATHS[k])
        idx = {int(i): j for j, i in enumerate(ids)}
        reorder = np.array([idx[int(i)] for i in order_ids])
        out.append(probs[reorder])
    return np.stack(out, axis=0)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def blend(stack: np.ndarray, weights: np.ndarray, mult: np.ndarray) -> np.ndarray:
    # stack: (n_models, n_samples, 3), weights: (n_models,)
    combo = np.tensordot(weights, stack, axes=(0, 0))  # (n_samples, 3)
    return combo * mult


def neg_f1(params: np.ndarray, y_true: np.ndarray, stack: np.ndarray) -> float:
    logits, mult = params[:N_MODELS], params[N_MODELS:]
    w = softmax(logits)
    probs = blend(stack, w, mult)
    pred = probs.argmax(1)
    return -f1_score(y_true, pred, average="macro")


def optimize_multistart(y_true, stack, n_starts=20, seed=0):
    rng = np.random.default_rng(seed)
    best_score, best_params = -1, None
    x0_equal = np.concatenate([np.zeros(N_MODELS), np.ones(3)])
    starts = [x0_equal] + [
        np.concatenate([rng.uniform(-1.5, 1.5, N_MODELS), rng.uniform(0.7, 1.3, 3)])
        for _ in range(n_starts - 1)
    ]
    for x0 in starts:
        res = minimize(
            neg_f1, x0=x0, args=(y_true, stack), method="Nelder-Mead",
            options=dict(xatol=1e-4, fatol=1e-6, maxiter=3000, maxfev=3000),
        )
        score = -res.fun
        if score > best_score:
            best_score, best_params = score, res.x
    return best_score, best_params


def main():
    y_train = load_catalog_labels()
    oof_stack = load_oof_stack()

    print("=" * 80)
    print("[baseline] equal-weight + threshold (dari skrip sebelumnya) OOF-F1 = 0.99286 (referensi)")

    print("\n[weighted-blend] optimasi bobot per-model + threshold per-kelas (multi-start, OOF murni)")
    best_score, best_params = optimize_multistart(y_train, oof_stack, n_starts=24)
    logits, mult = best_params[:N_MODELS], best_params[N_MODELS:]
    w = softmax(logits)
    print(f"    Best OOF-F1 = {best_score:.5f}")
    print("    Bobot per model:")
    for k, wi in sorted(zip(MODEL_KEYS, w), key=lambda t: -t[1]):
        print(f"      {k:15} w={wi:.4f}")
    print(f"    Threshold mult = {mult}")

    delta = best_score - 0.99286
    print(f"\n    Delta vs equal-weight baseline: {delta:+.5f}")

    if delta <= 0.0002:
        print("    -> Perbaikan TIDAK signifikan / berisiko overfit ke OOF (10 parameter di ruang optimasi).")
        print("    -> REKOMENDASI: TETAP PAKAI equal-weight (lebih robust, less overfit risk), jangan ganti submission.")
        return

    print("    -> Perbaikan cukup besar, worth dicoba sbg kandidat baru (tapi tetap verifikasi via info-only).")

    ref_ids = np.load(BASE / TTA_IDS_PATHS["conv_l"])
    ref_ids_sorted = np.sort(ref_ids)
    test_stack = load_test_stack(ref_ids_sorted)
    test_probs = blend(test_stack, w, mult)
    test_pred = test_probs.argmax(1)
    dist = np.bincount(test_pred, minlength=3)
    print(f"    Test distribution: {dict(zip(CLASSES, dist.tolist()))}")

    sol_path = BASE / "data" / "solution.csv"
    if sol_path.exists():
        sol = pd.read_csv(sol_path)
        sol_map = dict(zip(sol.iloc[:, 0].astype(int), sol.iloc[:, 1].astype(int)))
        y_proxy_idx = np.array([sol_map.get(int(i), -1) for i in ref_ids_sorted])
        mask = y_proxy_idx >= 0
        info_f1 = f1_score(y_proxy_idx[mask], test_pred[mask], average="macro")
        print(f"\n[INFO-ONLY] macro-F1 vs solution.csv proxy = {info_f1:.6f} (n={mask.sum()})")

    out_df = pd.DataFrame({"id": ref_ids_sorted.astype(int), "predicted": test_pred.astype(int)})
    out_path = BASE / "outputs" / "submission_v3_weighted_blend_TTAfull.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nTersimpan (kandidat, BELUM di-copy ke submission resmi): {out_path}")


if __name__ == "__main__":
    main()
