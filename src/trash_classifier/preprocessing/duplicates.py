"""Deteksi & karantina gambar train yang duplikat piksel-identik (cleanvision exact_duplicates).

"Duplikat" di sini = md5 dari array piksel hasil decode gambar (jadi
re-encode/format beda tapi isi piksel sama tetap kehitung duplikat; bukan
sekadar file-byte identik).

Grup duplikat yang labelnya BERBEDA (gambar sama persis tapi dilabel dua
kelas berbeda) bukan kasus "buang salah satu" -- itu sinyal label bentrok
yang perlu ditinjau manual. `apply_quarantine` TIDAK PERNAH menyentuh grup
semacam itu.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from ..catalog import ImageCatalog

logger = logging.getLogger(__name__)


def find_exact_duplicate_groups(catalog: ImageCatalog) -> list[list[str]]:
    from cleanvision import Imagelab

    imagelab = Imagelab(filepaths=catalog.dataframe["filepath"].tolist())
    imagelab.find_issues({"exact_duplicates": {}})
    return imagelab.info["exact_duplicates"]["sets"]


def build_duplicate_report(catalog: ImageCatalog, groups: list[list[str]]) -> pd.DataFrame:
    label_by_path = catalog.dataframe.set_index("filepath")["label"]

    rows = []
    for group_id, filepaths in enumerate(groups):
        filepaths_sorted = sorted(filepaths)
        labels = {label_by_path[fp] for fp in filepaths_sorted}
        conflict = len(labels) > 1
        keep_path = filepaths_sorted[0]
        for fp in filepaths_sorted:
            rows.append({
                "group_id": group_id,
                "filepath": fp,
                "given_label": label_by_path[fp],
                "cross_label_conflict": conflict,
                "action": "keep" if fp == keep_path else "drop",
            })

    columns = ["group_id", "filepath", "given_label", "cross_label_conflict", "action"]
    report = pd.DataFrame(rows, columns=columns)
    # Grup konflik ditaruh di atas -- itu yang paling butuh perhatian manual.
    return report.sort_values(["cross_label_conflict", "group_id"], ascending=[False, True]).reset_index(drop=True)


def apply_quarantine(report: pd.DataFrame, quarantine_dir: Path) -> int:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    to_move = report[(report["action"] == "drop") & (~report["cross_label_conflict"])]

    moved = 0
    for _, row in to_move.iterrows():
        src = Path(row["filepath"])
        if not src.exists():
            logger.warning("Sudah tidak ada, dilewati: %s", src)
            continue
        dest = quarantine_dir / f"{row['given_label']}__{src.name}"
        try:
            shutil.move(str(src), str(dest))
            moved += 1
        except OSError as exc:
            logger.warning("Gagal pindahkan %s -> %s: %s", src, dest, exc)
    return moved


def log_apply_verification(report: pd.DataFrame, before_counts: pd.Series, train_dir: Path) -> None:
    """Bandingkan jumlah file per kelas sebelum vs sesudah `apply_quarantine`, scan ulang dari disk.

    Rescan dari disk (bukan cuma andalkan hitungan `moved`) supaya kalau ada
    yang tidak beres (mis. insiden filename berubah di luar dugaan), langsung
    ketahuan di log -- bukan baru ketahuan belakangan lewat File Explorer.
    """
    after_counts = ImageCatalog.from_folders(train_dir).label_counts()
    expected_drop = (
        report[(report["action"] == "drop") & (~report["cross_label_conflict"])]
        .groupby("given_label").size()
    )

    logger.info("Perbandingan jumlah file per kelas (sebelum -> sesudah apply):")
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
            "(%d gambar, dihitung dari CSV). Jangan lanjut training sebelum ini diselidiki "
            "-- kemungkinan ada file lain yang ikut terpengaruh di luar 61 kandidat duplikat.",
            int(expected_drop.sum()),
        )
