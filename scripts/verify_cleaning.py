"""CLI: verifikasi menyeluruh hasil siklus pembersihan data -- pastikan tidak sia-sia.

Read-only, TIDAK mengubah apapun -- aman dijalankan kapan saja, di remote
maupun laptop. Mengecek:

  1. Kelengkapan katalog -- jumlah file mentah (os.walk) vs yang terdeteksi
     ImageCatalog (dipakai training sungguhan). Ada preseden nyata: file
     berpath kepanjangan bisa diam-diam terlewat dari katalog tanpa error.
  2. File dengan path > 259 karakter (batas MAX_PATH Windows).
  3. File kontaminasi (kategori B / CONTAMINATION) yang seharusnya sudah
     dibuang -- pastikan benar-benar tidak ada lagi.
  4. File relabel ada di folder kelas yang benar (bukan folder lama).
  5. Duplikat exact tersisa (scan ulang penuh, paling lama -- bisa dilewati
     dengan --skip-dedupe-scan kalau cuma mau cek cepat).

Exit code 0 kalau semua lolos, 1 kalau ada masalah -- bisa dipakai di script
lain sebagai gerbang sebelum training.

Usage:
    python scripts/verify_cleaning.py --base-dir data
    python scripts/verify_cleaning.py --base-dir data --skip-dedupe-scan   # cepat
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
    check_catalog_completeness,
    check_contamination_removed,
    check_duplicates,
    check_long_paths,
    check_relabel_placement,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verifikasi menyeluruh hasil pembersihan data train.")
    parser.add_argument("--base-dir", type=str, default=None, help="Folder berisi train/.")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--label-review-csv", type=str, default=None,
                         help="Default: <output-dir>/audit/label_review.csv")
    parser.add_argument("--relabel-review-csv", type=str, default=None,
                         help="Default: <output-dir>/audit/relabel_review.csv")
    parser.add_argument("--skip-dedupe-scan", action="store_true",
                         help="Lewati scan duplikat exact penuh (paling lama, ~2-3 menit buat 26 ribu gambar).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    config_kwargs = dict(output_dir=args.output_dir)
    if args.base_dir:
        config_kwargs["base_dir"] = args.base_dir
    config = PipelineConfig(**config_kwargs)

    label_review_csv = Path(args.label_review_csv) if args.label_review_csv else config.output_dir / "audit" / "label_review.csv"
    relabel_review_csv = Path(args.relabel_review_csv) if args.relabel_review_csv else config.output_dir / "audit" / "relabel_review.csv"

    all_ok = True

    logger.info("=== 1. Kelengkapan katalog (file mentah vs ImageCatalog) ===")
    mismatches = check_catalog_completeness(config.train_dir)
    if mismatches:
        all_ok = False
        for label, (raw, cat) in mismatches.items():
            logger.error("  TIDAK COCOK %s: file mentah=%d, ImageCatalog=%d (selisih %d -- kemungkinan path kepanjangan)",
                         label, raw, cat, raw - cat)
    else:
        logger.info("  OK -- semua file mentah terdeteksi ImageCatalog, tidak ada yang terlewat diam-diam.")

    logger.info("=== 2. File dengan path > 259 karakter (batas MAX_PATH Windows) ===")
    long_paths = check_long_paths(config.train_dir)
    if long_paths:
        all_ok = False
        logger.error("  %d file path kepanjangan (rename ke nama pendek, JANGAN ubah isi file):", len(long_paths))
        for p in long_paths:
            logger.error("    %s", p)
    else:
        logger.info("  OK -- 0 file melebihi batas.")

    if label_review_csv.exists() or relabel_review_csv.exists():
        logger.info("=== 3. File kontaminasi sudah benar-benar dibuang ===")
        still_present = check_contamination_removed(label_review_csv, relabel_review_csv, config.train_dir)
        if still_present:
            all_ok = False
            logger.error("  %d file kontaminasi MASIH ADA (seharusnya sudah dibuang):", len(still_present))
            for f in still_present:
                logger.error("    %s", f)
        else:
            logger.info("  OK -- semua file kontaminasi yang tercatat sudah tidak ada.")

        logger.info("=== 4. File relabel ada di folder yang benar ===")
        missing, wrong_folder = check_relabel_placement(relabel_review_csv, config.train_dir)
        if missing or wrong_folder:
            all_ok = False
            for f in missing:
                logger.error("  HILANG (seharusnya sudah direlabel, tidak ketemu sama sekali): %s", f)
            for fname, expected, actual in wrong_folder:
                logger.error("  SALAH FOLDER: %s -- seharusnya di %s, ketemu di %s", fname, expected, actual)
        else:
            logger.info("  OK -- semua file relabel ada di folder yang benar.")
    else:
        logger.warning("=== 3-4. Dilewati -- label_review.csv/relabel_review.csv tidak ditemukan di %s ===",
                       label_review_csv.parent)

    if not args.skip_dedupe_scan:
        logger.info("=== 5. Scan ulang duplikat exact PENUH (bisa beberapa menit) ===")
        if long_paths:
            all_ok = False
            logger.error("  DILEWATI -- scan ini crash kalau ada file path kepanjangan "
                         "(lihat cek #2). Perbaiki dulu (rename), baru jalankan ulang.")
        else:
            try:
                catalog = ImageCatalog.from_folders(config.train_dir)
                groups = check_duplicates(catalog)
            except Exception as exc:
                all_ok = False
                logger.error("  GAGAL scan duplikat: %s: %s", type(exc).__name__, exc)
            else:
                if groups:
                    all_ok = False
                    logger.error("  %d grup duplikat DITEMUKAN (seharusnya 0):", len(groups))
                    for g in groups[:20]:
                        logger.error("    %s", g)
                else:
                    logger.info("  OK -- 0 duplikat.")
    else:
        logger.info("=== 5. Dilewati (--skip-dedupe-scan) ===")

    logger.info("=" * 60)
    if all_ok:
        logger.info("SEMUA CEK LOLOS -- data bersih dan lengkap, aman dipakai training.")
    else:
        logger.error("ADA MASALAH -- lihat detail di atas sebelum lanjut training.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
