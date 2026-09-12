"""Verifikasi menyeluruh hasil siklus pembersihan data (dedupe, kontaminasi, relabel).

Read-only -- tidak mengubah apapun. Dibuat supaya bisa dijalankan ulang kapan
saja (remote atau laptop) untuk memastikan hasil --apply sebelumnya benar-benar
tersimpan, tidak hilang/tertimpa oleh proses lain (transfer, sync, rebuild
katalog, dst).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from ..catalog import ImageCatalog
from .duplicates import find_exact_duplicate_groups

MAX_PATH_LIMIT = 259


def _basename(filepath: str) -> str:
    return filepath.replace("\\", "/").split("/")[-1]


def check_catalog_completeness(train_dir: Path) -> dict[str, tuple[int, int]]:
    """Bandingkan jumlah file mentah (os.walk, akurat) vs yang terdeteksi
    `ImageCatalog.from_folders` (pathlib glob) per kelas.

    Ada preseden nyata: pathlib glob/rglob bisa DIAM-DIAM melewatkan file yang
    path-nya melebihi batas MAX_PATH Windows -- kalau ada selisih di sini,
    berarti training akan diam-diam kehilangan gambar tanpa error apapun,
    persis pola bug yang pernah ditemukan (lihat DATA_AUDIT_LOG.md).

    Return {label: (jumlah_mentah, jumlah_katalog)} HANYA untuk label yang
    tidak cocok (dict kosong = semua cocok).
    """
    exts = (".jpg", ".jpeg", ".png")
    catalog = ImageCatalog.from_folders(train_dir)
    catalog_counts = catalog.dataframe["label"].value_counts().to_dict()

    mismatches: dict[str, tuple[int, int]] = {}
    for label_dir in sorted(p for p in Path(train_dir).iterdir() if p.is_dir()):
        raw_count = sum(
            1 for _, _, files in os.walk(str(label_dir)) for f in files if f.lower().endswith(exts)
        )
        cat_count = int(catalog_counts.get(label_dir.name, 0))
        if raw_count != cat_count:
            mismatches[label_dir.name] = (raw_count, cat_count)
    return mismatches


def check_long_paths(train_dir: Path, max_len: int = MAX_PATH_LIMIT) -> list[str]:
    """Return path file yang melebihi batas MAX_PATH Windows.

    Sengaja pakai `os.walk` (bukan `Path.rglob`) -- rglob terbukti bisa diam-diam
    melewatkan entri yang path gabungannya kepanjangan, os.walk lebih bisa
    diandalkan buat enumerasi kasus ini.
    """
    root = str(Path(train_dir).resolve())
    long_paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if len(full) > max_len:
                long_paths.append(full)
    return long_paths


def check_contamination_removed(label_review_csv: Path, relabel_review_csv: Path, train_dir: Path) -> list[str]:
    """Return nama file kontaminasi yang SEHARUSNYA sudah dibuang tapi masih ada di `train_dir`."""
    existing_names = {p.name for p in Path(train_dir).rglob("*") if p.is_file()}

    contam_files: set[str] = set()
    if Path(label_review_csv).exists():
        df = pd.read_csv(label_review_csv)
        contam_files |= set(df.loc[df["decision"] == "B", "filepath"].apply(_basename))
    if Path(relabel_review_csv).exists():
        df = pd.read_csv(relabel_review_csv)
        contam_files |= set(df.loc[df["decision"] == "CONTAMINATION", "filepath"].apply(_basename))

    return sorted(f for f in contam_files if f in existing_names)


def check_relabel_placement(
    relabel_review_csv: Path, train_dir: Path
) -> tuple[list[str], list[tuple[str, str, list[str]]]]:
    """Return (hilang, salah_folder) untuk file yang seharusnya sudah direlabel.

    `hilang`: nama file yang tercatat direlabel tapi tidak ketemu sama sekali.
    `salah_folder`: (nama_file, folder_seharusnya, folder_ketemu) untuk yang
    ada tapi bukan di folder tujuan yang tercatat.
    """
    if not Path(relabel_review_csv).exists():
        return [], []

    file_folders: dict[str, list[str]] = {}
    for p in Path(train_dir).rglob("*"):
        if p.is_file():
            file_folders.setdefault(p.name, []).append(p.parent.name)

    df = pd.read_csv(relabel_review_csv)
    df = df[df["decision"] != "CONTAMINATION"]

    missing: list[str] = []
    wrong_folder: list[tuple[str, str, list[str]]] = []
    for _, row in df.iterrows():
        fname = _basename(row["filepath"])
        expected = row["decision"]
        if fname not in file_folders:
            missing.append(fname)
            continue
        if expected not in file_folders[fname]:
            wrong_folder.append((fname, expected, file_folders[fname]))

    return missing, wrong_folder


def check_duplicates(catalog: ImageCatalog) -> list[list[str]]:
    """Return sisa grup duplikat exact (list kosong = bersih)."""
    return find_exact_duplicate_groups(catalog)
