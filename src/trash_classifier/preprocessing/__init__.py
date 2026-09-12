"""Audit & pembersihan data train -- terpisah dari alur training/inferensi inti.

Semua fungsi di sini murni membaca (atau, untuk `apply_quarantine`, memindah
file ke karantina) -- tidak ada yang mengubah config/trainer/pipeline training.
"""
from .contamination import apply_contamination_removal, load_confirmed_contamination, log_removal_verification
from .clip_contamination import audit_clip_contamination
from .duplicates import apply_quarantine, build_duplicate_report, find_exact_duplicate_groups, log_apply_verification
from .label_issues import audit_label_issues
from .pipeline import AuditPipeline
from .quality import audit_image_quality
from .relabel import apply_relabel, load_confirmed_relabels, log_relabel_verification
from .review import RUBRIC, ReviewQueue
from .review_app import build_review_app
from .verify import (
    check_catalog_completeness,
    check_contamination_removed,
    check_duplicates,
    check_long_paths,
    check_relabel_placement,
)

__all__ = [
    "audit_image_quality",
    "audit_label_issues",
    "audit_clip_contamination",
    "find_exact_duplicate_groups",
    "build_duplicate_report",
    "apply_quarantine",
    "log_apply_verification",
    "load_confirmed_contamination",
    "apply_contamination_removal",
    "log_removal_verification",
    "load_confirmed_relabels",
    "apply_relabel",
    "log_relabel_verification",
    "AuditPipeline",
    "ReviewQueue",
    "RUBRIC",
    "build_review_app",
    "check_catalog_completeness",
    "check_long_paths",
    "check_contamination_removed",
    "check_relabel_placement",
    "check_duplicates",
]
