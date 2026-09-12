"""Eksekusi relabel: pindah file ke folder kelas yang benar hasil review manual.

Sumber: outputs/audit/relabel_review.csv, kolom `decision` (kelas tujuan
yang benar), kolom `given_label` (kelas asal yang salah). Baris
`decision == "CONTAMINATION"` dilewati -- itu domain `contamination.py`
(lihat scripts/remove_contamination.py --extra-review-csv), bukan relabel.

File BENAR-BENAR dipindah (bukan disalin/karantina) ke folder kelas yang
benar; nama file TIDAK diubah. `relabel_review.csv` sendiri sudah berfungsi
sebagai jejak audit/undo (given_label = lokasi lama, decision = lokasi baru)
-- tidak perlu mekanisme karantina terpisah seperti dedupe/kontaminasi,
karena ini murni pindah folder, bukan pengeluaran dari dataset.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from ..catalog import ImageCatalog

logger = logging.getLogger(__name__)

CONTAMINATION_DECISION = "CONTAMINATION"


def load_confirmed_relabels(relabel_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(relabel_csv_path)
    return df[df["decision"] != CONTAMINATION_DECISION].reset_index(drop=True)


def apply_relabel(confirmed: pd.DataFrame, train_dir: Path) -> pd.DataFrame:
    """Pindahkan tiap file ke folder `decision`-nya. Return subset yang BENAR-BENAR dipindah."""
    moved_rows = []
    for _, row in confirmed.iterrows():
        src = Path(row["filepath"])
        if not src.exists():
            logger.warning("Sudah tidak ada, dilewati: %s", src)
            continue

        dest_dir = train_dir / row["decision"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if dest.exists():
            logger.warning("Tujuan sudah ada file dengan nama sama, dilewati (tangani manual): %s", dest)
            continue

        try:
            shutil.move(str(src), str(dest))
            moved_rows.append(row)
        except OSError as exc:
            logger.warning("Gagal pindahkan %s -> %s: %s", src, dest, exc)
    return pd.DataFrame(moved_rows) if moved_rows else confirmed.iloc[0:0]


def log_relabel_verification(moved: pd.DataFrame, before_counts: pd.Series, train_dir: Path) -> None:
    """Rescan `train_dir` dari disk dan bandingkan NET perubahan per kelas.

    Beda dari dedupe/kontaminasi (cuma berkurang): relabel dua arah -- kelas
    asal berkurang, kelas tujuan bertambah -- jadi yang dibandingkan net
    (masuk dikurangi keluar), bukan cuma pengurangan.
    """
    after_counts = ImageCatalog.from_folders(train_dir).label_counts()
    out_counts = moved.groupby("given_label").size() if len(moved) else pd.Series(dtype=int)
    in_counts = moved.groupby("decision").size() if len(moved) else pd.Series(dtype=int)

    logger.info("Perbandingan jumlah file per kelas (sebelum -> sesudah relabel):")
    any_mismatch = False
    for label in sorted(set(before_counts.index) | set(after_counts.index)):
        before = int(before_counts.get(label, 0))
        after = int(after_counts.get(label, 0))
        expected_net = int(in_counts.get(label, 0)) - int(out_counts.get(label, 0))
        actual_net = after - before
        flag = ""
        if actual_net != expected_net:
            flag = f"  <-- TIDAK SESUAI (harusnya net {expected_net:+d})"
            any_mismatch = True
        logger.info("  %-15s %6d -> %6d  (%+d)%s", label, before, after, actual_net, flag)

    if any_mismatch:
        logger.warning(
            "Perubahan jumlah file per kelas TIDAK sesuai ekspektasi. Selidiki sebelum lanjut training.")
