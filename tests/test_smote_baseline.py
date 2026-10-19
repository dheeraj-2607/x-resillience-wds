from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.explainability.xai import compute_shap_explanations
from src.detection.smote_baseline import (
    build_temporal_features,
    chronological_smote_split,
    evaluate_smote_tree_baselines,
)


def test_chronological_split_fits_scaler_and_smote_on_training_rows_only():
    row_count = 40
    features = np.column_stack((np.arange(row_count, dtype=float), np.ones(row_count)))
    features[5, 1] = np.nan
    features[35, 1] = np.nan
    labels = np.array([1] * 8 + [0] * 24 + [0] * 7 + [1])

    result = chronological_smote_split(features, labels, test_size=0.2, random_state=7)

    assert result["split_index"] == 32
    assert result["train_rows_before_smote"] == 32
    assert result["test_rows"] == 8
    assert result["scaler"].mean_[0] == np.mean(np.arange(32, dtype=float))
    assert result["test_class_counts_untouched"] == {0: 7, 1: 1}
    assert result["train_class_counts_after_smote"] == {1: 24, 0: 24}
    assert result["X_train"].shape[0] == 48
    assert result["X_test"].shape[0] == 8
    assert np.isfinite(result["X_train"]).all()
    assert np.isfinite(result["X_test"]).all()


def test_chronological_smote_split_rejects_insufficient_training_minority():
    features = np.arange(40, dtype=float).reshape(20, 2)
    labels = np.array([1, 0] + [0] * 18)

    try:
        chronological_smote_split(features, labels, test_size=0.2)
    except ValueError as error:
        assert "minority rows" in str(error)
    else:
        raise AssertionError("Expected SMOTE to reject an undersized training minority class.")


def test_temporal_features_are_trailing_and_include_diurnal_encoding():
    frame = pd.DataFrame(
        {
            "DATETIME": pd.date_range("2026-01-01", periods=15, freq="h"),
            "P_J280": np.arange(15, dtype=float),
        }
    )
    features = build_temporal_features(frame, ["P_J280"])

    assert features.loc[1, "P_J280__diff_1"] == 1.0
    assert features.loc[2, "P_J280__rolling_mean_3"] == 1.0
    assert np.isclose(features.loc[2, "P_J280__rolling_std_3"], np.std([0.0, 1.0, 2.0]))
    assert np.isclose(features.loc[12, "P_J280__rolling_mean_12"], 6.5)
    assert np.isclose(features.loc[0, "hour_sin"], 0.0)
    assert np.isclose(features.loc[6, "hour_sin"], 1.0)
    assert np.isclose(features.loc[12, "hour_cos"], -1.0)
    assert features.loc[0, "P_J280__diff_1"] != features.loc[0, "P_J280__diff_1"]


def test_two_tree_baselines_keep_test_partition_untouched():
    rng = np.random.default_rng(22)
    features = rng.normal(size=(100, 4))
    labels = np.zeros(100, dtype=int)
    labels[:8] = 1
    labels[80:84] = 1

    report = evaluate_smote_tree_baselines(
        features,
        labels,
        split_index=80,
        random_state=12,
        n_estimators=10,
    )

    assert set(report["models"]) == {"random_forest", "xgboost"}
    for model_report in report["models"].values():
        assert model_report["test_rows"] == 20
        assert model_report["test_rows_resampled"] == 0
        assert model_report["train_class_counts_after_smote"] == {0: 72, 1: 72}
        assert model_report["metrics"]["samples"] == 20


def test_tree_shap_branch_returns_finite_feature_attributions():
    rng = np.random.default_rng(5)
    features = rng.normal(size=(80, 3))
    labels = (features[:, 0] + features[:, 1] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=20, random_state=5).fit(features, labels)

    attributions = compute_shap_explanations(
        model,
        background_data=features[:20],
        instance=features[20:21],
        feature_names=["P_J280", "L_T1", "F_PU3"],
    )

    assert len(attributions) == 3
    assert np.isfinite(attributions["SHAP Value"]).all()