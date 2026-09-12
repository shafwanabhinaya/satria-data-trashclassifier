"""End-to-end orchestration: build catalog, train each backbone via K-fold, ensemble, submit."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from .catalog import ImageCatalog
from .config import PipelineConfig
from .ensemble import EnsembleWeightSearcher
from .trainer import SingleModelTrainer
from .utils import set_seed

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Runs the full multi-backbone ensemble training pipeline and writes the submission CSV."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        set_seed(config.seed)

    def run(self) -> Path:
        cfg = self.config
        # Dump config for reproducibility
        cfg.save_to_json(cfg.experiment_dir / "config.json")
        
        catalog = ImageCatalog.load_or_build(cfg.train_dir, cfg.duckdb_path)
        logger.info("Distribusi kelas train:\n%s", catalog.label_counts())

        classes = catalog.classes
        label_mapping = catalog.label_mapping
        y_true = catalog.dataframe["label"].map(label_mapping).values

        test_df = pd.read_csv(cfg.sample_submission_path)
        class_weights = self._compute_class_weights(catalog.dataframe, label_mapping, len(classes))

        skf = StratifiedKFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)
        trainer = SingleModelTrainer(cfg)

        oof_probs_by_model: dict[str, np.ndarray] = {}
        test_probs_by_model: dict[str, np.ndarray] = {}
        test_ids: list = []

        for model_name in cfg.models_to_train:
            safe_model_name = model_name.replace("/", "_").replace(".", "_")
            config_hash = cfg.get_model_config_hash(model_name)
            oof_cache = cfg.cache_dir / f"cache_{safe_model_name}_{config_hash}_oof.npy"
            test_cache = cfg.cache_dir / f"cache_{safe_model_name}_{config_hash}_test.npy"
            ids_cache = cfg.cache_dir / f"cache_{safe_model_name}_{config_hash}_ids.npy"

            if oof_cache.exists() and test_cache.exists() and ids_cache.exists():
                logger.info("Memuat %s dari cache (sudah pernah dilatih dengan config ini)...", model_name.upper())
                oof_probs = np.load(oof_cache)
                test_probs = np.load(test_cache)
                ids = np.load(ids_cache).tolist()
            else:
                size = cfg.img_size[model_name]
                logger.info("MELATIH %s (resolusi %dx%d, unfreeze %d block terakhir, hash %s)",
                            model_name.upper(), size, size, cfg.unfreeze_blocks[model_name], config_hash)

                oof_probs, test_probs, ids = trainer.run(
                    model_name, catalog.dataframe, label_mapping, class_weights, test_df, skf)
                
                # Simpan ke cache agar jika crash/resume tidak hilang
                cfg.cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(oof_cache, oof_probs)
                np.save(test_cache, test_probs)
                np.save(ids_cache, np.array(ids))

            oof_probs_by_model[model_name] = oof_probs
            test_probs_by_model[model_name] = test_probs
            if not test_ids:
                test_ids = ids

            oof_preds = np.argmax(oof_probs, axis=1)
            model_f1 = f1_score(y_true, oof_preds, average="macro")
            logger.info("%s -- OOF macro-F1 keseluruhan: %.4f", model_name.upper(), model_f1)
            logger.info("\n%s", classification_report(y_true, oof_preds, target_names=classes, zero_division=0))

        if cfg.use_stacking:
            logger.info("Melatih Stacking Meta-Classifier (Logistic Regression)...")
            from .ensemble import StackingEnsemble
            stacker = StackingEnsemble(random_state=cfg.seed)
            best_oof_f1 = stacker.fit(oof_probs_by_model, y_true)
            logger.info("Stacking Meta-Classifier trained | OOF macro-F1: %.4f", best_oof_f1)
            
            X_meta = np.hstack([oof_probs_by_model[name] for name in sorted(oof_probs_by_model.keys())])
            oof_final_preds = stacker.meta_clf.predict(X_meta)
            
            final_preds = stacker.predict(test_probs_by_model)
        else:
            logger.info("Mencari bobot ensemble terbaik via OOF grid search...")
            searcher = EnsembleWeightSearcher(step=cfg.ensemble_search_step)
            best_weights, best_oof_f1 = searcher.search(oof_probs_by_model, y_true)
            logger.info("Bobot terbaik: %s | OOF macro-F1: %.4f", best_weights, best_oof_f1)
            
            # Blended OOF & test probs
            blended_oof_probs = sum(oof_probs_by_model[m] * w for m, w in best_weights.items())
            final_test_probs = sum(test_probs_by_model[m] * w for m, w in best_weights.items())
            
            # Decision threshold tuning
            logger.info("Mengoptimalkan decision thresholds (Nelder-Mead)...")
            from .ensemble import ThresholdTuner
            tuner = ThresholdTuner()
            tuner.fit(blended_oof_probs, y_true)
            logger.info("Multipliers optimal: %s", tuner.multipliers)
            
            # Calculate F1 after tuning
            oof_final_preds = tuner.predict(blended_oof_probs)
            tuned_oof_f1 = f1_score(y_true, oof_final_preds, average="macro")
            logger.info("OOF macro-F1 setelah Threshold Tuning: %.4f (selisih: %+.4f)", 
                        tuned_oof_f1, tuned_oof_f1 - best_oof_f1)
            best_oof_f1 = tuned_oof_f1
            
            final_preds = tuner.predict(final_test_probs)

        # --- Error Analysis (Phase 1) ---
        from .utils import plot_confusion_matrix
        plot_confusion_matrix(
            y_true, oof_final_preds, classes, cfg.experiment_dir / "confusion_matrix.png"
        )
        
        # Save classification report as text
        report = classification_report(y_true, oof_final_preds, target_names=classes, zero_division=0)
        with open(cfg.experiment_dir / "classification_report.txt", "w") as f:
            f.write(report)
            
        # Save list of misclassified instances
        misclassified_mask = oof_final_preds != y_true
        df_mis = catalog.dataframe[misclassified_mask].copy()
        df_mis["true_label"] = df_mis["label"]
        idx_to_label = {v: k for k, v in label_mapping.items()}
        df_mis["predicted_label"] = [idx_to_label[p] for p in oof_final_preds[misclassified_mask]]
        misclassified_csv_path = cfg.experiment_dir / "misclassified_instances.csv"
        df_mis.to_csv(misclassified_csv_path, index=False)
        logger.info("Analisis Error disimpan di %s", cfg.experiment_dir)

        # Save experiment results
        results = {
            "best_oof_f1": float(best_oof_f1),
            "use_stacking": cfg.use_stacking,
        }
        if not cfg.use_stacking:
            results["best_weights"] = {m: float(w) for m, w in best_weights.items()}
            results["multipliers"] = [float(x) for x in tuner.multipliers]
            
        import json
        with open(cfg.experiment_dir / "experiment_results.json", "w") as f:
            json.dump(results, f, indent=4)

        # Update global experiment summary CSV
        from .utils import update_experiment_summary
        summary_dict = {
            "models": list(cfg.models_to_train),
            "seed": cfg.seed,
            "scheduler": cfg.scheduler_type,
            "progressive_ft": cfg.use_progressive_ft
        }
        update_experiment_summary(
            cfg.output_dir / "experiment_summary.csv",
            cfg.experiment_name,
            cfg.k_folds,
            best_oof_f1,
            summary_dict
        )
        logger.info("Rangkuman eksperimen diperbarui di %s", cfg.output_dir / "experiment_summary.csv")

        clean_ids = [int(x.item()) if torch.is_tensor(x) else int(x) for x in test_ids]
        submission = pd.DataFrame({"id": clean_ids, "predicted": final_preds})
        submission.to_csv(cfg.submission_path, index=False)
        logger.info("Submission tersimpan di %s (estimasi OOF macro-F1 ~ %.4f)", cfg.submission_path, best_oof_f1)
        return cfg.submission_path

    def _compute_class_weights(self, df: pd.DataFrame, label_mapping: dict, num_classes: int) -> torch.Tensor:
        y = df["label"].map(label_mapping).values
        weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y)
        return torch.tensor(weights, dtype=torch.float32, device=self.config.device)
