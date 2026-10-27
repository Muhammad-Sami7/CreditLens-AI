from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


class CreditLensEvaluator:
    """Evaluate the already-trained CreditLens-AI models."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = (
            Path(root_dir)
            if root_dir is not None
            else Path(__file__).resolve().parents[2]
        )

        self.models_dir = self.root_dir / "models"
        self.data_dir = self.root_dir / "data" / "processed"
        self.reports_dir = self.root_dir / "reports"

        self.feature_columns = self._load_json("feature_columns.json")
        self.metadata = self._load_json("model_metadata.json")
        self.ensemble_config = self._load_json("ensemble_config.json")

        self.target_column = self.metadata["dataset"]["target_column"]
        self.random_state = 42
        self.validation_size = 0.20

        self.weights = {
            "lightgbm": self.ensemble_config["ensemble"]["weights"]["lightgbm"],
            "xgboost": self.ensemble_config["ensemble"]["weights"]["xgboost"],
            "catboost": self.ensemble_config["ensemble"]["weights"]["catboost"],
        }

        self.raw_threshold = float(
            self.ensemble_config["calibration"]["raw_ensemble_threshold"]
        )
        self.calibrated_threshold = float(
            self.ensemble_config["calibration"]["calibrated_threshold"]
        )

        self.models = self._load_models()
        self.calibrator = self._load_calibrator()

    def _load_json(self, filename: str) -> dict[str, Any] | list[str]:
        path = self.models_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _load_models(self) -> dict[str, Any]:
        paths = {
            "lightgbm": self.models_dir / "regularized_lightgbm.pkl",
            "xgboost": self.models_dir / "optimized_xgboost.pkl",
            "catboost": self.models_dir / "optimized_catboost.pkl",
        }

        for name, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Required {name} model not found: {path}"
                )

        return {
            "lightgbm": joblib.load(paths["lightgbm"]),
            "xgboost": joblib.load(paths["xgboost"]),
            "catboost": joblib.load(paths["catboost"]),
        }

    def _load_calibrator(self) -> Any:
        path = self.models_dir / "isotonic_calibrator.pkl"

        if not path.exists():
            raise FileNotFoundError(
                f"Required isotonic calibrator not found: {path}"
            )

        return joblib.load(path)

    def load_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load the labeled training dataset."""

        path = self.data_dir / "train_processed.parquet"

        if not path.exists():
            raise FileNotFoundError(
                f"Processed training dataset not found: {path}"
            )

        df = pd.read_parquet(path)

        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' is missing."
            )

        missing_features = [
            feature
            for feature in self.feature_columns
            if feature not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing {len(missing_features)} production features. "
                f"Examples: {missing_features[:10]}"
            )

        X = df[self.feature_columns].copy()
        y = df[self.target_column].astype(int)

        return X, y

    def create_validation_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Reproduce the original stratified 80/20 validation split."""

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=self.validation_size,
            random_state=self.random_state,
            stratify=y,
        )

        return X_train, X_valid, y_train, y_valid

    def predict_probabilities(
        self,
        X_valid: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """Generate probabilities from the three trained models."""

        lightgbm_probability = self.models["lightgbm"].predict_proba(
            X_valid
        )[:, 1]

        xgboost_probability = self.models["xgboost"].predict_proba(
            X_valid
        )[:, 1]

        catboost_probability = self.models["catboost"].predict_proba(
            X_valid
        )[:, 1]

        return {
            "lightgbm": lightgbm_probability,
            "xgboost": xgboost_probability,
            "catboost": catboost_probability,
        }

    def calculate_raw_ensemble(
        self,
        model_probabilities: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Calculate weighted soft-voting ensemble probability."""

        return (
            self.weights["lightgbm"] * model_probabilities["lightgbm"]
            + self.weights["xgboost"] * model_probabilities["xgboost"]
            + self.weights["catboost"] * model_probabilities["catboost"]
        )

    def calculate_calibrated_probability(
        self,
        raw_ensemble_probability: np.ndarray,
    ) -> np.ndarray:
        """Apply the existing production isotonic calibrator."""

        calibrated = self.calibrator.predict(raw_ensemble_probability)

        return np.asarray(calibrated, dtype=float)

    @staticmethod
    def _classification_metrics(
        y_true: pd.Series,
        probability: np.ndarray,
        threshold: float,
    ) -> dict[str, Any]:
        prediction = (probability >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            prediction,
            labels=[0, 1],
        ).ravel()

        return {
            "threshold": float(threshold),
            "roc_auc": float(roc_auc_score(y_true, probability)),
            "pr_auc": float(average_precision_score(y_true, probability)),
            "accuracy": float(accuracy_score(y_true, prediction)),
            "precision": float(
                precision_score(y_true, prediction, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, prediction, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_true, prediction, zero_division=0)
            ),
            "brier_score": float(
                brier_score_loss(y_true, probability)
            ),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
            "positive_predictions": int(prediction.sum()),
            "negative_predictions": int((prediction == 0).sum()),
        }

    def evaluate(
        self,
        save_report: bool = True,
    ) -> dict[str, Any]:
        """Run the complete development/validation evaluation."""

        X, y = self.load_dataset()

        X_train, X_valid, y_train, y_valid = self.create_validation_split(
            X,
            y,
        )

        model_probabilities = self.predict_probabilities(X_valid)

        raw_ensemble = self.calculate_raw_ensemble(
            model_probabilities
        )

        calibrated_ensemble = self.calculate_calibrated_probability(
            raw_ensemble
        )

        individual_metrics = {
            "lightgbm": self._classification_metrics(
                y_valid,
                model_probabilities["lightgbm"],
                self.metadata["individual_models"]["Regularized LightGBM"][
                    "threshold"
                ],
            ),
            "xgboost": self._classification_metrics(
                y_valid,
                model_probabilities["xgboost"],
                self.metadata["individual_models"]["Optimized XGBoost"][
                    "threshold"
                ],
            ),
            "catboost": self._classification_metrics(
                y_valid,
                model_probabilities["catboost"],
                self.metadata["individual_models"]["Optimized CatBoost"][
                    "threshold"
                ],
            ),
        }

        raw_ensemble_metrics = self._classification_metrics(
            y_valid,
            raw_ensemble,
            self.raw_threshold,
        )

        calibrated_ensemble_metrics = self._classification_metrics(
            y_valid,
            calibrated_ensemble,
            self.calibrated_threshold,
        )

        report = {
            "evaluation_type": "development_validation",
            "warning": (
                "The validation set was used during model development. "
                "The existing isotonic calibrator and calibrated threshold "
                "were also fitted/optimized using this validation set, so "
                "calibrated metrics are not an independent final test."
            ),
            "dataset": {
                "total_rows": int(len(X)),
                "training_rows": int(len(X_train)),
                "validation_rows": int(len(X_valid)),
                "feature_count": int(len(self.feature_columns)),
                "positive_cases": int(y_valid.sum()),
                "negative_cases": int((y_valid == 0).sum()),
                "positive_rate": float(y_valid.mean()),
            },
            "ensemble_configuration": {
                "lightgbm_weight": self.weights["lightgbm"],
                "xgboost_weight": self.weights["xgboost"],
                "catboost_weight": self.weights["catboost"],
                "raw_threshold": self.raw_threshold,
                "calibrated_threshold": self.calibrated_threshold,
            },
            "individual_models": individual_metrics,
            "raw_ensemble": raw_ensemble_metrics,
            "calibrated_ensemble": calibrated_ensemble_metrics,
        }

        if save_report:
            self.reports_dir.mkdir(parents=True, exist_ok=True)

            report_path = (
                self.reports_dir / "development_validation_evaluation.json"
            )

            with report_path.open("w", encoding="utf-8") as file:
                json.dump(report, file, indent=4)

        return report
