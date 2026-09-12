"""Antrian tinjau manual multi-reviewer untuk kandidat label-issue.

Dipisah dari Flask (lihat scripts/review_labels.py) supaya logic klaim/rilis/
simpan keputusan bisa diuji tanpa perlu jalankan web server sungguhan.

Aman dipakai concurrent (banyak reviewer sekaligus lewat ngrok): satu
`threading.Lock` melindungi seluruh state (antrian, klaim, penulisan CSV).
"""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import pandas as pd

RUBRIC: dict[str, str] = {
    "A": "Salah label -- gambar berisi objek sampah asli, tapi kategorinya salah.",
    "B": "Kontaminasi -- gambar bukan foto sampah sama sekali (chart, foto tidak relevan, dll).",
    "C": "Ambigu -- objek sampah asli, wajar dibingungkan; label given tetap dipertahankan.",
    "D": "Model error -- label given sudah benar, model/cleanlab saja yang salah tebak.",
}

REVIEW_CSV_COLUMNS = [
    "filepath", "given_label", "predicted_label", "label_quality_score",
    "reviewer", "decision", "note", "reviewed_at",
]


class ReviewQueue:
    """Antrian kandidat + status klaim, dengan persist tiap keputusan ke CSV.

    `decision_options` (kode -> deskripsi) menentukan nilai `decision` yang
    valid -- default `RUBRIC` (A/B/C/D), tapi bisa diganti (mis. nama kelas
    tujuan) untuk putaran review lain yang pakai `ReviewQueue` yang sama.
    """

    def __init__(self, candidates: pd.DataFrame, review_csv_path: Path,
                 decision_options: dict[str, str] | None = None, stale_after_seconds: int = 900):
        self._lock = threading.Lock()
        self._review_csv_path = review_csv_path
        self._stale_after = stale_after_seconds
        self.decision_options = decision_options or RUBRIC

        self._candidates: dict[str, dict] = {
            row["filepath"]: row for row in candidates.to_dict("records")
        }
        self._reviewed: dict[str, dict] = {}
        self._in_progress: dict[str, dict] = {}  # filepath -> {"reviewer": str, "claimed_at": float}

        self._load_existing_reviews()

    def _load_existing_reviews(self) -> None:
        if not self._review_csv_path.exists():
            self._review_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._review_csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=REVIEW_CSV_COLUMNS).writeheader()
            return
        existing = pd.read_csv(self._review_csv_path)
        for row in existing.to_dict("records"):
            self._reviewed[row["filepath"]] = row

    def _release_stale_claims(self) -> None:
        now = time.time()
        stale = [fp for fp, claim in self._in_progress.items()
                 if now - claim["claimed_at"] > self._stale_after]
        for fp in stale:
            del self._in_progress[fp]

    def claim_next(self, reviewer: str) -> dict | None:
        """Kunci gambar berikutnya yang belum ditinjau/diklaim untuk `reviewer` ini."""
        with self._lock:
            self._release_stale_claims()
            for filepath, row in self._candidates.items():
                if filepath in self._reviewed or filepath in self._in_progress:
                    continue
                self._in_progress[filepath] = {"reviewer": reviewer, "claimed_at": time.time()}
                return row
            return None

    def release(self, filepath: str) -> None:
        """Lepas klaim tanpa menyimpan keputusan (mis. tombol 'lewati')."""
        with self._lock:
            self._in_progress.pop(filepath, None)

    def submit_decision(self, filepath: str, reviewer: str, decision: str, note: str) -> dict:
        if decision not in self.decision_options:
            raise ValueError(f"decision harus salah satu dari {list(self.decision_options)}, dapat {decision!r}")

        with self._lock:
            existing = self._reviewed.get(filepath)
            if existing is not None:
                # Idempoten: submit ganda (double-klik, network retry, dst) untuk gambar yang
                # sama tidak boleh menulis baris baru -- kembalikan keputusan yang sudah ada.
                self._in_progress.pop(filepath, None)
                return existing

            row = self._candidates.get(filepath)
            if row is None:
                raise KeyError(f"filepath tidak ada di antrian kandidat: {filepath}")

            record = {
                "filepath": filepath,
                "given_label": row["given_label"],
                "predicted_label": row["predicted_label"],
                "label_quality_score": row["label_quality_score"],
                "reviewer": reviewer,
                "decision": decision,
                "note": note,
                "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(self._review_csv_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=REVIEW_CSV_COLUMNS).writerow(record)

            self._reviewed[filepath] = record
            self._in_progress.pop(filepath, None)
            return record

    def progress(self) -> dict:
        with self._lock:
            by_reviewer: dict[str, int] = {}
            for record in self._reviewed.values():
                by_reviewer[record["reviewer"]] = by_reviewer.get(record["reviewer"], 0) + 1
            return {
                "reviewed": len(self._reviewed),
                "total": len(self._candidates),
                "in_progress": len(self._in_progress),
                "by_reviewer": by_reviewer,
            }
