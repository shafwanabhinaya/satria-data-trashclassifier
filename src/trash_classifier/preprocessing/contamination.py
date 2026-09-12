"""Eksekusi keputusan kontaminasi non-trash dari hasil review manual.

Sumber utama: outputs/audit/label_review.csv, kolom `decision == "B"`
(putaran pertama). Bisa digabung dengan sumber tambahan (mis.
relabel_review.csv, kolom `decision == "CONTAMINATION"`) untuk kasus yang
ketahuan kontaminasi belakangan di putaran kedua -- lihat
scripts/remove_contamination.py --extra-review-csv.

File dipindah (bukan dihapus) ke folder karantina terpisah -- sama seperti
`duplicates.py`, supaya reversibel. Aman dijalankan ulang (idempoten): file
yang sudah pernah dipindah otomatis dilewati, tidak dihitung dobel di
verifikasi before/after.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from ..catalog import ImageCatalog

logger = logging.getLogger(__name__)


def load_confirmed_contamination(review_csv_path: Path, decision_value: str = "B") -> pd.DataFrame:
    df = pd.read_csv(review_csv_path)
    return df[df["decision"] == decision_value].reset_index(drop=True)


def apply_contamination_removal(confirmed: pd.DataFrame, quarantine_dir: Path) -> pd.DataFrame:
    """Pindahkan file ke karantina; return subset `confirmed` yang BENAR-BENAR dipindah.

    File yang sudah tidak ada (mis. sudah dipindah di run sebelumnya) dilewati
    dengan warning, bukan dihitung sebagai "seharusnya berkurang" -- supaya
    verifikasi before/after tetap akurat walau skrip ini dijalankan ulang.
    """
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    moved_rows = []
    for _, row in confirmed.iterrows():
        src = Path(row["filepath"])
        if not src.exists():
            logger.warning("Sudah tidak ada, dilewati: %s", src)
            continue
        dest = quarantine_dir / f"{row['given_label']}__{src.name}"
        try:
            shutil.move(str(src), str(dest))
            moved_rows.append(row)
        except OSError as exc:
            logger.warning("Gagal pindahkan %s -> %s: %s", src, dest, exc)
    return pd.DataFrame(moved_rows) if moved_rows else confirmed.iloc[0:0]


def log_removal_verification(moved: pd.DataFrame, before_counts: pd.Series, train_dir: Path) -> None:
    """Rescan `train_dir` dari disk dan bandingkan jumlah per kelas -- sama pola dengan dedupe.

    `moved` harus berupa hasil `apply_contamination_removal` (yang BENAR-BENAR
    dipindah), bukan daftar kandidat mentah -- supaya re-run parsial tidak
    memicu false-alarm mismatch.
    """
    after_counts = ImageCatalog.from_folders(train_dir).label_counts()
    expected_drop = moved.groupby("given_label").size() if len(moved) else pd.Series(dtype=int)

    logger.info("Perbandingan jumlah file per kelas (sebelum -> sesudah buang kontaminasi):")
    any_mismatch = False
    for label in sorted(set(before_counts.index) | set(after_counts.index)):
        before = int(before_counts.get(label, 0))
        after = int(after_counts.get(label, 0))
        expected = int(expected_drop.get(label, 0))
        actual_drop = before - after
        flag = ""
        if actual_drop != expected:
            flag = f"  <-- TIDAK SESUAI (harusnya berkurang {expected})"
            any_mismatch = True
        logger.info("  %-15s %6d -> %6d  (-%d)%s", label, before, after, actual_drop, flag)

    if any_mismatch:
        logger.warning(
            "Jumlah file yang berkurang TIDAK sesuai dengan yang seharusnya dipindah "
            "(%d gambar benar-benar dipindah run ini). Selidiki sebelum lanjut training.",
            int(expected_drop.sum()),
        )
