"""Leakage-safe temporal tree baselines for labeled BATADAL data."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.detection.evaluate import DetectionEvaluator


def _class_counts(labels: np.ndarray) -> dict[int, int]:
    return {int(label): int(count) for label, count in Counter(labels).items()}


def build_temporal_features(
    frame,
    sensor_columns: list[str],
    timestamp_column: str = "DATETIME",
) -> "pd.DataFrame":
    """Build causal lag, rolling, and diurnal predictors without future access."""
    import pandas as pd

    sensor_data = frame[sensor_columns].astype(float)
    features = {}
    for sensor in sensor_columns:
        series = sensor_data[sensor]
        features[sensor] = series
        features[f"{sensor}__diff_1"] = series.diff()
        for window in (3, 12):
            rolling = series.rolling(window=window, min_periods=1)
            features[f"{sensor}__rolling_mean_{window}"] = rolling.mean()
            features[f"{sensor}__rolling_std_{window}"] = rolling.std(ddof=0)

    timestamps = pd.to_datetime(frame[timestamp_column])
    fractional_hour = timestamps.dt.hour + timestamps.dt.minute / 60.0
    angle = 2.0 * np.pi * fractional_hour / 24.0
    features["hour_sin"] = np.sin(angle)
    features["hour_cos"] = np.cos(angle)
    return pd.DataFrame(features, index=frame.index)


def chronological_smote_split(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    k_neighbors: int = 5,
    split_index: int | None = None,
) -> dict[str, Any]:
    """Split ordered rows first, then fit preprocessing and SMOTE on train only."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels).ravel()

    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("features must be 2D and have the same number of rows as labels.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if np.isinf(features).any():
        raise ValueError("features must not contain infinite values.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must contain both binary classes 0 and 1.")

    if split_index is None:
        split_index = int(len(features) * (1.0 - test_size))
    if split_index <= 0 or split_index >= len(features):
        raise ValueError("chronological split must leave rows in both partitions.")

    train_features = features[:split_index]
    train_labels = labels[:split_index].astype(int)
    test_features = features[split_index:]
    test_labels = labels[split_index:].astype(int)

    if len(np.unique(train_labels)) != 2:
        raise ValueError("chronological training partition must contain both classes for SMOTE.")

    minority_count = min(Counter(train_labels).values())
    if minority_count <= k_neighbors:
        raise ValueError(
            f"training partition has {minority_count} minority rows; "
            f"SMOTE with k_neighbors={k_neighbors} requires at least {k_neighbors + 1}."
        )

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_imputed = imputer.fit_transform(train_features)
    test_imputed = imputer.transform(test_features)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    test_scaled = scaler.transform(test_imputed)

    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    train_resampled, labels_resampled = smote.fit_resample(train_scaled, train_labels)

    return {
        "X_train": train_resampled,
        "y_train": labels_resampled,
        "X_test": test_scaled,
        "y_test": test_labels,
        "imputer": imputer,
        "scaler": scaler,
        "split_index": split_index,
        "train_rows_before_smote": len(train_labels),
        "test_rows": len(test_labels),
        "train_class_counts_before_smote": _class_counts(train_labels),
        "train_class_counts_after_smote": _class_counts(labels_resampled),
        "test_class_counts_untouched": _class_counts(test_labels),
    }


def evaluate_smote_random_forest(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> dict[str, Any]:
    """Fit a supervised RF baseline on resampled train rows; evaluate untouched test rows."""
    prepared = chronological_smote_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        k_neighbors=k_neighbors,
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )
    classifier.fit(prepared["X_train"], prepared["y_train"])
    test_probabilities = classifier.predict_proba(prepared["X_test"])[:, 1]
    metrics = DetectionEvaluator(threshold=0.5).evaluate_predictions(
        prepared["y_test"], test_probabilities
    )

    return {
        "method": "Random Forest with training-only SMOTE",
        "split_policy": "chronological; split before imputation, scaling, and resampling",
        "test_rows_resampled": 0,
        "preprocessing_fit_rows": prepared["train_rows_before_smote"],
        "smote_fit_rows": prepared["train_rows_before_smote"],
        "train_rows_before_smote": prepared["train_rows_before_smote"],
        "train_rows_after_smote": len(prepared["y_train"]),
        "test_rows": prepared["test_rows"],
        "train_class_counts_before_smote": prepared["train_class_counts_before_smote"],
        "train_class_counts_after_smote": prepared["train_class_counts_after_smote"],
        "test_class_counts_untouched": prepared["test_class_counts_untouched"],
        "metrics": metrics,
    }


def evaluate_smote_tree_baselines(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    k_neighbors: int = 5,
    split_index: int | None = None,
    n_estimators: int = 300,
) -> dict[str, Any]:
    """Compare RF and XGBoost using the same untouched chronological test set."""
    prepared = chronological_smote_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        k_neighbors=k_neighbors,
        split_index=split_index,
    )
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight=None,
        ),
        "xgboost": XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        ),
    }
    reports = {}
    for model_name, classifier in models.items():
        classifier.fit(prepared["X_train"], prepared["y_train"])
        test_probabilities = classifier.predict_proba(prepared["X_test"])[:, 1]
        metrics = DetectionEvaluator(threshold=0.5).evaluate_predictions(
            prepared["y_test"], test_probabilities
        )
        reports[model_name] = {
            "method": f"{model_name.replace('_', ' ').title()} with training-only SMOTE",
            "split_policy": "chronological; split before imputation, scaling, and resampling",
            "test_rows_resampled": 0,
            "preprocessing_fit_rows": prepared["train_rows_before_smote"],
            "smote_fit_rows": prepared["train_rows_before_smote"],
            "train_rows_before_smote": prepared["train_rows_before_smote"],
            "train_rows_after_smote": len(prepared["y_train"]),
            "test_rows": prepared["test_rows"],
            "train_class_counts_before_smote": prepared["train_class_counts_before_smote"],
            "train_class_counts_after_smote": prepared["train_class_counts_after_smote"],
            "test_class_counts_untouched": prepared["test_class_counts_untouched"],
            "metrics": metrics,
        }
    return {
        "feature_count": int(np.asarray(features).shape[1]),
        "split_index": prepared["split_index"],
        "split_policy": "chronological; split before imputation, scaling, and resampling",
        "models": reports,
    }