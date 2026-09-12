"""CLI: cari & (opsional) karantina gambar train yang duplikat piksel-identik.

Logic sebenarnya ada di `trash_classifier.preprocessing.duplicates` (lihat
modul itu untuk detail, termasuk kenapa grup bentrok label tidak pernah
di-auto-handle). Skrip ini cuma argparse + wiring config/catalog + tulis CSV.

Default-nya cuma menulis laporan CSV (dry run) -- tidak ada file yang
dipindah/dihapus kecuali kamu tambahkan --apply. Yang dipindah pun bukan
dihapus permanen, cuma dikarantina ke folder terpisah supaya gampang
dikembalikan.

Usage:
    python scripts/dedupe_exact_duplicates.py --base-dir data                # laporan saja
    python scripts/dedupe_exact_duplicates.py --base-dir data --apply        # karantina yang aman
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trash_classifier.catalog import ImageCatalog
from trash_classifier.config import PipelineConfig
from trash_classifier.preprocessing import (
    apply_quarantine,
    build_duplicate_report,
    find_exact_duplicate_groups,
    log_apply_verification,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deteksi (dan opsional karantina) gambar train duplikat piksel-identik.")
    parser.add_argument("--base-dir", type=str, default=None, help="Folder berisi train/.")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--quarantine-dir", type=str, default=None,
                         help="Default: <base-dir>/_duplicates_removed")
    parser.add_argument("--apply", action="store_true",
                         help="Benar-benar pindahkan file 'drop' (non-konflik) ke quarantine-dir. "
                              "Tanpa flag ini, skrip cuma menulis laporan CSV.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    config_kwargs = dict(output_dir=args.output_dir)
    if args.base_dir:
        config_kwargs["base_dir"] = args.base_dir
    config = PipelineConfig(**config_kwargs)

    catalog = ImageCatalog.load_or_build(config.train_dir, config.duckdb_path)
    logger.info("Total gambar train: %d", len(catalog))

    groups = find_exact_duplicate_groups(catalog)
    report = build_duplicate_report(catalog, groups)

    audit_dir = config.output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_path = audit_dir / "exact_duplicates_review.csv"
    report.to_csv(out_path, index=False)

    n_groups = report["group_id"].nunique()
    n_conflict_groups = report.loc[report["cross_label_conflict"], "group_id"].nunique()
    n_drop_safe = len(report[(report["action"] == "drop") & (~report["cross_label_conflict"])])
    logger.info("Grup duplikat: %d (%d di antaranya BENTROK LABEL -- tidak disentuh)",
                n_groups, n_conflict_groups)
    logger.info("Kandidat aman untuk dibuang (1 label per grup): %d gambar -> %s", n_drop_safe, out_path)

    if n_conflict_groups:
        logger.warning("%d grup duplikat punya label berbeda-beda -- cek manual dulu di %s "
                        "(cross_label_conflict=True), skrip ini tidak akan menyentuhnya.",
                        n_conflict_groups, out_path)

    if args.apply:
        quarantine_dir = Path(args.quarantine_dir) if args.quarantine_dir else config.train_dir.parent / "_duplicates_removed"
        before_counts = catalog.label_counts()
        moved = apply_quarantine(report, quarantine_dir)
        logger.info("Dipindahkan %d gambar duplikat ke %s", moved, quarantine_dir)
        log_apply_verification(report, before_counts, config.train_dir)
        logger.info("PENTING: hapus katalog lama (%s) atau jalankan ulang build-catalog "
                     "supaya training berikutnya tidak lagi mengacu ke file yang sudah dipindah.",
                     config.duckdb_path)
    else:
        logger.info("Dry run -- tidak ada file dipindah. Tinjau %s lalu jalankan ulang dengan --apply.", out_path)


if __name__ == "__main__":
    main()
