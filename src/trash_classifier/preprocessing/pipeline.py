"""Orkestrasi audit data train -- peran yang sama seperti TrainingPipeline untuk main.py.

`scripts/audit_data.py` seharusnya cuma argparse + wiring config + panggil
`AuditPipeline(config).run(...)`, sama tipisnya dengan `main.py` yang cuma
panggil `TrainingPipeline(config).run()`.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..catalog import ImageCatalog
from ..config import PipelineConfig
from .label_issues import audit_label_issues
from .quality import audit_image_quality

logger = logging.getLogger(__name__)


class AuditPipeline:
    """Membangun katalog lalu menjalankan audit label/kualitas yang diminta."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, model_name: str, skip_labels: bool = False, skip_quality: bool = False) -> dict[str, Path]:
        cfg = self.config
        catalog = ImageCatalog.load_or_build(cfg.train_dir, cfg.duckdb_path)
        logger.info("Total gambar train: %d", len(catalog))

        audit_dir = cfg.output_dir / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, Path] = {}
        if not skip_labels:
            results["label_issues"] = self._run_label_audit(catalog, model_name, audit_dir)
        if not skip_quality:
            results["image_quality"] = self._run_quality_audit(catalog, audit_dir)
        return results

    def _run_label_audit(self, catalog: ImageCatalog, model_name: str, audit_dir: Path) -> Path:
        report = audit_label_issues(self.config, catalog, model_name)
        out_path = audit_dir / "label_issues.csv"
        report.to_csv(out_path, index=False)
        n_flagged = int(report["is_flagged_label_issue"].sum())
        logger.info("Label issues: %d/%d gambar ditandai cleanlab -> %s", n_flagged, len(report), out_path)
        return out_path

    def _run_quality_audit(self, catalog: ImageCatalog, audit_dir: Path) -> Path:
        report = audit_image_quality(catalog)
        out_path = audit_dir / "image_quality.csv"
        report.to_csv(out_path, index=False)
        n_any_issue = int((report["num_quality_issues"] > 0).sum())
        logger.info("Quality issues: %d/%d gambar punya >=1 masalah kualitas -> %s",
                     n_any_issue, len(report), out_path)
        return out_path
