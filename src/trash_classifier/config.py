"""Configuration for the trash-classification training pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

DEFAULT_MODELS: tuple[str, ...] = (
    "vit_base_patch16_siglip_384",
    "vit_base_patch16_siglip_256",
    "swinv2_base_window12to16_192to256_22kft1k",
    "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
    "convnextv2_base",
    "convnextv2_large",
    "efficientnetv2_rw_m",
    # Kandidat skrining lintas-arsitektur (2026-07-22): keluarga pretraining beda
    # dari yang sudah dipakai (bukan siglip/convnext/swin/efficientnet lagi).
    "eva02_base_patch14_224.mim_in22k",
    "vit_base_patch14_dinov2.lvd142m",
    "beit_base_patch16_224.in22k_ft_in22k_in1k",
    "maxvit_base_tf_224.in1k",
    # Push kapasitas (2026-07-29): EVA-02 base adalah satu2nya kandidat skrining yg genuinely
    # menaikkan ensemble (LOO delta -0.0007 kalau dibuang) -- coba versi Large, paradigma MIM
    # sama yg sudah terbukti, kapasitas ~3.5x lebih besar (304M vs 86M param).
    "eva02_large_patch14_224.mim_in22k",
    # Push kapasitas #2: ConvNeXt-Large pretrain CLIP (paradigma kontrastif teks-gambar skala
    # besar) -- satu2nya paradigma CLIP yg sudah terbukti positif di ensemble (siglip_384),
    # ini versi skala jauh lebih besar & arsitektur beda (conv, bukan ViT).
    "convnext_large_mlp.clip_laion2b_augreg_ft_in1k",
)

DEFAULT_IMG_SIZE = {
    "vit_base_patch16_siglip_384": 384,
    "vit_base_patch16_siglip_256": 256,
    "swinv2_base_window12to16_192to256_22kft1k": 256,
    "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k": 256,
    "convnextv2_base": 224,
    "convnextv2_large": 224,
    "efficientnetv2_rw_m": 224,
    "eva02_base_patch14_224.mim_in22k": 224,
    "vit_base_patch14_dinov2.lvd142m": 224,  # native 518, dipaksa 224 utk kecepatan skrining
    "beit_base_patch16_224.in22k_ft_in22k_in1k": 224,
    "maxvit_base_tf_224.in1k": 224,
    "eva02_large_patch14_224.mim_in22k": 224,
    "convnext_large_mlp.clip_laion2b_augreg_ft_in1k": 256,
}
DEFAULT_UNFREEZE_BLOCKS = {
    "vit_base_patch16_siglip_384": 4,
    "vit_base_patch16_siglip_256": 4,
    "swinv2_base_window12to16_192to256_22kft1k": 2,
    "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k": 2,
    "convnextv2_base": 2,
    "convnextv2_large": 2,
    "efficientnetv2_rw_m": 2,
    "eva02_base_patch14_224.mim_in22k": 4,
    "vit_base_patch14_dinov2.lvd142m": 4,
    "beit_base_patch16_224.in22k_ft_in22k_in1k": 4,
    "maxvit_base_tf_224.in1k": 2,
    "eva02_large_patch14_224.mim_in22k": 3,  # sedikit dikurangi dari base (4) -- model besar, batasi waktu
    "convnext_large_mlp.clip_laion2b_augreg_ft_in1k": 2,
}
DEFAULT_HEAD_LR = {
    "vit_base_patch16_siglip_384": 5e-4,
    "vit_base_patch16_siglip_256": 5e-4,
    "swinv2_base_window12to16_192to256_22kft1k": 2e-4,
    "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k": 2e-4,
    "convnextv2_base": 5e-4,
    "convnextv2_large": 5e-4,
    "efficientnetv2_rw_m": 5e-4,
    "eva02_base_patch14_224.mim_in22k": 3e-4,
    "vit_base_patch14_dinov2.lvd142m": 3e-4,
    "beit_base_patch16_224.in22k_ft_in22k_in1k": 3e-4,
    "maxvit_base_tf_224.in1k": 5e-4,
    "eva02_large_patch14_224.mim_in22k": 2e-4,  # model besar -> LR head sedikit lbh rendah utk stabilitas
    "convnext_large_mlp.clip_laion2b_augreg_ft_in1k": 3e-4,
}


@dataclass
class PipelineConfig:
    """Paths and hyperparameters needed to run the training pipeline.

    `base_dir` must contain a `train/<label>/*.jpg` folder tree and the
    `submission.csv` (id, predicted) template shipped by the panitia.
    """

    base_dir: Path = field(default_factory=lambda: Path(os.environ.get("BDC_DATA_DIR", r"D:\Downloads\BDC2026")))
    output_dir: Path = Path("outputs")

    train_dirname: str = "train"
    test_dirname: str = "test"
    sample_submission_name: str = "submission.csv"

    duckdb_name: str = "dataset_catalog.duckdb"
    submission_name: str = "submission.csv"

    batch_size: int = 16
    num_epochs: int = 10
    k_folds: int = 3
    seed: int = 42

    models_to_train: tuple[str, ...] = DEFAULT_MODELS
    img_size: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_IMG_SIZE))
    unfreeze_blocks: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_UNFREEZE_BLOCKS))
    head_lr: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_HEAD_LR))
    backbone_lr_factor: float = 0.05
    grad_accum_steps: int = 1

    label_smoothing: float = 0.1
    use_tta: bool = True
    ensemble_search_step: float = 0.05
    use_stacking: bool = False

    experiment_name: str = "exp001"
    early_stopping_patience: int = 3
    scheduler_type: str = "cosine"  # "cosine", "cosine_restarts", "one_cycle"
    cudnn_deterministic: bool = True

    use_progressive_ft: bool = True
    stage1_epochs: int = 2
    stage2_unfreeze_all: bool = True
    use_vertical_flip_tta: bool = False
    aug_version: str = "aug_v1"
    rotate_limit: int = 15
    grad_clip_max_norm: float = 1.0
    use_ema: bool = False

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 4

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "cache").mkdir(parents=True, exist_ok=True)

    @property
    def experiment_dir(self) -> Path:
        return self.output_dir / self.experiment_name

    @property
    def cache_dir(self) -> Path:
        return self.experiment_dir / "cache"

    @property
    def train_dir(self) -> Path:
        return self.base_dir / self.train_dirname

    @property
    def test_dir(self) -> Path:
        return self.base_dir / self.test_dirname

    @property
    def sample_submission_path(self) -> Path:
        return self.base_dir / self.sample_submission_name

    @property
    def duckdb_path(self) -> Path:
        return self.output_dir / self.duckdb_name

    @property
    def submission_path(self) -> Path:
        return self.experiment_dir / self.submission_name

    def save_to_json(self, path: Path | str) -> None:
        import json
        data = {
            "experiment_name": self.experiment_name,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "k_folds": self.k_folds,
            "seed": self.seed,
            "models_to_train": list(self.models_to_train),
            "img_size": self.img_size,
            "unfreeze_blocks": self.unfreeze_blocks,
            "head_lr": self.head_lr,
            "backbone_lr_factor": self.backbone_lr_factor,
            "grad_accum_steps": self.grad_accum_steps,
            "label_smoothing": self.label_smoothing,
            "use_tta": self.use_tta,
            "ensemble_search_step": self.ensemble_search_step,
            "use_stacking": self.use_stacking,
            "early_stopping_patience": self.early_stopping_patience,
            "scheduler_type": self.scheduler_type,
            "cudnn_deterministic": self.cudnn_deterministic,
            "use_progressive_ft": self.use_progressive_ft,
            "stage1_epochs": self.stage1_epochs,
            "stage2_unfreeze_all": self.stage2_unfreeze_all,
            "use_vertical_flip_tta": self.use_vertical_flip_tta,
            "aug_version": self.aug_version,
            "rotate_limit": self.rotate_limit,
            "grad_clip_max_norm": self.grad_clip_max_norm,
            "use_ema": self.use_ema,
            "num_workers": self.num_workers,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def get_model_config_hash(self, model_name: str) -> str:
        import hashlib
        import json
        config_dict = {
            "model_name": model_name,
            "img_size": self.img_size[model_name],
            "unfreeze_blocks": self.unfreeze_blocks[model_name],
            "head_lr": self.head_lr[model_name],
            "backbone_lr_factor": self.backbone_lr_factor,
            "grad_accum_steps": self.grad_accum_steps,
            "num_epochs": self.num_epochs,
            "k_folds": self.k_folds,
            "seed": self.seed,
            "use_progressive_ft": self.use_progressive_ft,
            "stage1_epochs": self.stage1_epochs,
            "stage2_unfreeze_all": self.stage2_unfreeze_all,
            "scheduler_type": self.scheduler_type,
            "early_stopping_patience": self.early_stopping_patience,
            "aug_version": self.aug_version,
            "rotate_limit": self.rotate_limit,
            "grad_clip_max_norm": self.grad_clip_max_norm,
            "use_ema": self.use_ema,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:8]

