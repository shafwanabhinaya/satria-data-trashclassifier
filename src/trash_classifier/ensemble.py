"""Grid search for ensemble blend weights that maximize OOF macro-F1."""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.metrics import f1_score


class EnsembleWeightSearcher:
    """Finds per-model blend weights (non-negative, summing to 1) maximizing macro-F1 on OOF probs."""

    def __init__(self, step: float = 0.05):
        self.step = step

    def search(self, oof_probs: dict[str, np.ndarray], y_true: np.ndarray):
        """Returns (best_weights: dict[model_name, float], best_f1: float) using Optuna (fallback to grid search)."""
        names = list(oof_probs.keys())
        num_models = len(names)

        try:
            import optuna

            # Disable optuna logs to avoid cluttering terminal
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                weights = []
                for i in range(num_models):
                    weights.append(trial.suggest_float(f"w_{i}", 0.0, 1.0))

                w_sum = sum(weights)
                if w_sum == 0:
                    normalized_weights = [1.0 / num_models] * num_models
                else:
                    normalized_weights = [w / w_sum for w in weights]

                blended = sum(oof_probs[name] * normalized_weights[i] for i, name in enumerate(names))
                preds = np.argmax(blended, axis=1)
                return f1_score(y_true, preds, average="macro")

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=300, timeout=15)

            best_params = study.best_params
            best_w = [best_params[f"w_{i}"] for i in range(num_models)]
            w_sum = sum(best_w)
            if w_sum == 0:
                best_w = [1.0 / num_models] * num_models
            else:
                best_w = [w / w_sum for w in best_w]

            best_weights = {name: float(best_w[i]) for i, name in enumerate(names)}
            best_f1 = study.best_value
            return best_weights, best_f1

        except ImportError:
            # Fallback to Grid Search if optuna is not installed
            import logging
            logging.getLogger(__name__).warning("Optuna tidak terinstal. Menggunakan Grid Search diskrit...")

            grid = np.round(np.arange(0, 1 + 1e-9, self.step), 4)
            best_f1, best_weights = -1.0, None
            for combo in itertools.product(grid, repeat=num_models - 1):
                remainder = round(1 - sum(combo), 4)
                if remainder < 0 or remainder > 1:
                    continue
                weights = dict(zip(names, (*combo, remainder)))

                blended = sum(oof_probs[name] * w for name, w in weights.items())
                preds = np.argmax(blended, axis=1)
                f1 = f1_score(y_true, preds, average="macro")

                if f1 > best_f1:
                    best_f1, best_weights = f1, weights

            return best_weights, best_f1


class StackingEnsemble:
    """Melatih meta-classifier (Logistic Regression) pada probabilitas OOF dari model base."""

    def __init__(self, random_state: int = 42):
        from sklearn.linear_model import LogisticRegression
        # L2-regularized logistic regression adalah pilihan standar yang kokoh untuk stacking probabilitas
        self.meta_clf = LogisticRegression(max_iter=1000, random_state=random_state, C=1.0)

    def fit(self, oof_probs: dict[str, np.ndarray], y_true: np.ndarray) -> float:
        """Melatih meta-classifier pada probabilitas OOF. Mengembalikan skor OOF F1."""
        X_meta = np.hstack([oof_probs[name] for name in sorted(oof_probs.keys())])
        self.meta_clf.fit(X_meta, y_true)
        
        oof_preds = self.meta_clf.predict(X_meta)
        from sklearn.metrics import f1_score
        return f1_score(y_true, oof_preds, average="macro")

    def predict(self, test_probs: dict[str, np.ndarray]) -> np.ndarray:
        """Memprediksi kelas final untuk test set."""
        X_meta_test = np.hstack([test_probs[name] for name in sorted(test_probs.keys())])
        return self.meta_clf.predict(X_meta_test)


class ThresholdTuner:
    """Mencari pengali probabilitas kelas optimal untuk memaksimalkan Macro-F1."""

    def __init__(self):
        self.multipliers = None

    def fit(self, oof_probs: np.ndarray, y_true: np.ndarray):
        """Mencari pengali optimal menggunakan optimasi Nelder-Mead pada probabilitas OOF."""
        from scipy.optimize import minimize
        from sklearn.metrics import f1_score
        num_classes = oof_probs.shape[1]
        
        def objective(multipliers):
            weighted_probs = oof_probs * multipliers
            preds = np.argmax(weighted_probs, axis=1)
            return -f1_score(y_true, preds, average="macro")
            
        init_multipliers = np.ones(num_classes)
        bounds = [(0.1, 10.0) for _ in range(num_classes)]
        
        res = minimize(objective, init_multipliers, method="Nelder-Mead", bounds=bounds)
        self.multipliers = res.x
        return self.multipliers

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Menerapkan pengali yang telah dipelajari pada data probabilitas baru."""
        if self.multipliers is None:
            return np.argmax(probs, axis=1)
        weighted_probs = probs * self.multipliers
        return np.argmax(weighted_probs, axis=1)
