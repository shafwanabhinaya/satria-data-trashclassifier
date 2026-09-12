"""CLI entry point for the BDC Satria Data 2026 trash-classification pipeline.

Usage:
    python main.py --base-dir "D:\\Downloads\\BDC2026" --epochs 10 --folds 3
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from trash_classifier import PipelineConfig, TrainingPipeline
from trash_classifier.config import DEFAULT_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latih ensemble klasifikasi sampah BDC Satria Data 2026.")
    parser.add_argument("--base-dir", type=str, default=None,
                         help="Folder berisi train/, test/, dan submission.csv template "
                              "(default: env BDC_DATA_DIR atau D:\\Downloads\\BDC2026).")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--exp-name", type=str, default="exp001", help="Nama eksperimen / subfolder output.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=list(DEFAULT_MODELS),
                         help="Backbone yang dilatih dan diensemble.")
    parser.add_argument("--no-tta", action="store_true", help="Nonaktifkan horizontal-flip TTA saat prediksi test.")
    parser.add_argument("--stacking", action="store_true", help="Gunakan Stacking (Logistic Regression) sebagai ganti weighted blend.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (0 untuk menonaktifkan).")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "cosine_restarts", "one_cycle"],
                         help="Tipe learning rate scheduler.")
    parser.add_argument("--no-deterministic", action="store_true", help="Nonaktifkan setting CuDNN deterministic.")
    
    # Progressive Fine-Tuning, TTA, EMA, & Augmentations options (Phase 2)
    parser.add_argument("--no-progressive-ft", action="store_true", help="Nonaktifkan progressive fine-tuning.")
    parser.add_argument("--stage1-epochs", type=int, default=2, help="Epoch pelatihan head di Stage 1.")
    parser.add_argument("--no-unfreeze-all", action="store_true", help="Hanya unfreeze block ekor di Stage 2.")
    parser.add_argument("--vflip-tta", action="store_true", help="Aktifkan vertical-flip TTA.")
    parser.add_argument("--accum-steps", type=int, default=1, help="Langkah akumulasi gradien.")
    parser.add_argument("--ema", action="store_true", help="Aktifkan Exponential Moving Average (ModelEmaV2).")
    parser.add_argument("--rotate-limit", type=int, default=15, help="Batas rotasi acak pada augmentasi citra.")
    parser.add_argument("--clip-grad", type=float, default=1.0, help="Batas norm maksimum untuk gradient clipping (0.0 untuk menonaktifkan).")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Nilai label smoothing (misal: 0.05 atau 0.1).")
    parser.add_argument("--workers", type=int, default=4, help="Jumlah worker process untuk DataLoader.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    config_kwargs = dict(
        output_dir=args.output_dir,
        experiment_name=args.exp_name,
        num_epochs=args.epochs,
        k_folds=args.folds,
        batch_size=args.batch_size,
        models_to_train=tuple(args.models),
        use_tta=not args.no_tta,
        use_stacking=args.stacking,
        seed=args.seed,
        early_stopping_patience=args.patience,
        scheduler_type=args.scheduler,
        cudnn_deterministic=not args.no_deterministic,
        use_progressive_ft=not args.no_progressive_ft,
        stage1_epochs=args.stage1_epochs,
        stage2_unfreeze_all=not args.no_unfreeze_all,
        use_vertical_flip_tta=args.vflip_tta,
        grad_accum_steps=args.accum_steps,
        use_ema=args.ema,
        rotate_limit=args.rotate_limit,
        grad_clip_max_norm=args.clip_grad,
        label_smoothing=args.label_smoothing,
        num_workers=args.workers,
    )
    if args.base_dir:
        config_kwargs["base_dir"] = args.base_dir

    config = PipelineConfig(**config_kwargs)
    submission_path = TrainingPipeline(config).run()
    logging.info("Selesai. Submission: %s", submission_path)


if __name__ == "__main__":
    main()
