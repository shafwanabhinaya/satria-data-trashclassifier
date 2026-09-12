"""Builds and persists the train-image catalog (filepath -> label) used by the pipeline."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TABLE_NAME = "image_metadata"


class ImageCatalog:
    """A (filepath, label) DataFrame over the train set, with DuckDB-backed persistence."""

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.dataframe)

    @classmethod
    def from_folders(cls, train_dir: Path) -> "ImageCatalog":
        """Walk `train_dir/<label>/*.jpg` and build the catalog from disk."""
        records = []
        label_dirs = sorted(p for p in Path(train_dir).iterdir() if p.is_dir())
        for label_dir in label_dirs:
            for img_path in label_dir.glob("*.*"):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append({"filepath": str(img_path), "label": label_dir.name})
        return cls(pd.DataFrame.from_records(records, columns=["filepath", "label"]))

    @classmethod
    def from_duckdb(cls, db_path: Path, table: str = TABLE_NAME) -> "ImageCatalog":
        con = duckdb.connect(str(db_path))
        try:
            df = con.execute(f"SELECT filepath, label FROM {table}").df()
        finally:
            con.close()
        return cls(df)

    @classmethod
    def load_or_build(cls, train_dir: Path, db_path: Path) -> "ImageCatalog":
        """Reuse a previously persisted catalog if present, otherwise scan disk and persist it."""
        if Path(db_path).exists():
            catalog = cls.from_duckdb(db_path)
            if len(catalog) > 0:
                return catalog
        catalog = cls.from_folders(train_dir)
        catalog.save_to_duckdb(db_path)
        return catalog

    def save_to_duckdb(self, db_path: Path, table: str = TABLE_NAME) -> None:
        df = self.dataframe  # noqa: F841 - referenced by name inside the SQL below
        con = duckdb.connect(str(db_path))
        try:
            con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df")
        finally:
            con.close()

    def label_counts(self) -> pd.Series:
        return self.dataframe["label"].value_counts().sort_index()

    @property
    def classes(self) -> list[str]:
        """Sorted class names; sorting keeps the `0_/1_/2_` folder prefixes as the label codes
        required by the submission format (0=Recyclable, 1=Electronic, 2=Organic)."""
        return sorted(self.dataframe["label"].unique().tolist())

    @property
    def label_mapping(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.classes)}
