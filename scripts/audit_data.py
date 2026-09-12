"""CLI: audit gambar train/ -- kandidat salah label (cleanlab) + kualitas gambar (cleanvision).

Orkestrasi & logic ada di `trash_classifier.preprocessing` (lihat `AuditPipeline`
untuk detail alur, dan modul-modul di dalamnya untuk catatan penting, mis.
kenapa cleanvision "light" bukan deteksi saturasi HSV, dan kenapa kasus
ambigu tidak ada pengecekan otomatis). Skrip ini cuma argparse + wiring
config + panggil `AuditPipeline(config).run(...)`.

Usage:
    python scripts/audit_data.py --base-dir data --model vit_b --epochs 3
    python scripts/audit_data.py --base-dir data --skip-labels   # cleanvision saja, cepat
    python scripts/audit_data.py --base-dir data --skip-quality  # cleanlab saja
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trash_classifier.config import DEFAULT_MODELS, PipelineConfig
from trash_classifier.preprocessing import AuditPipeline
from trash_classifier.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit data train: cari kandidat salah label (cleanlab) dan "
                     "masalah kualitas gambar (cleanvision).")
    parser.add_argument("--base-dir", type=str, default=None,
                         help="Folder berisi train/ dan test/ (mis. 'data' kalau dijalankan dari root proyek ini).")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--model", type=str, default="vit_b", choices=list(DEFAULT_MODELS),
                         help="Satu backbone saja untuk hasilkan OOF probs -- tidak perlu training ensemble penuh.")
    parser.add_argument("--epochs", type=int, default=3,
                         help="Epoch per fold. Rendah karena tujuannya mengurutkan kecurigaan, bukan model final "
                              "-- naikkan kalau hasil rankingnya terasa berisik.")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-labels", action="store_true", help="Lewati audit label (cleanlab).")
    parser.add_argument("--skip-quality", action="store_true", help="Lewati audit kualitas gambar (cleanvision).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    config_kwargs = dict(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        k_folds=args.folds,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    if args.base_dir:
        config_kwargs["base_dir"] = args.base_dir
    config = PipelineConfig(**config_kwargs)
    set_seed(config.seed)

    AuditPipeline(config).run(model_name=args.model, skip_labels=args.skip_labels, skip_quality=args.skip_quality)


if __name__ == "__main__":
    main()
