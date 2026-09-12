"""Audit kualitas gambar train lewat cleanvision (blur, gelap, terlalu terang, duplikat, dst)."""
from __future__ import annotations

import pandas as pd

from ..catalog import ImageCatalog


def audit_image_quality(catalog: ImageCatalog) -> pd.DataFrame:
    from cleanvision import Imagelab

    filepaths = catalog.dataframe["filepath"].tolist()
    imagelab = Imagelab(filepaths=filepaths)
    imagelab.find_issues()

    issues = imagelab.issues.reset_index().rename(columns={"index": "filepath"})
    label_by_path = catalog.dataframe.set_index("filepath")["label"]
    issues["given_label"] = issues["filepath"].map(label_by_path)

    issue_cols = [c for c in issues.columns if c.startswith("is_") and c.endswith("_issue")]
    issues["num_quality_issues"] = issues[issue_cols].sum(axis=1)
    return issues.sort_values("num_quality_issues", ascending=False).reset_index(drop=True)
