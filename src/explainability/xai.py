"""Library-backed sensor attribution for the dashboard demonstration."""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any, Callable, Sequence

import lime.lime_tabular
import numpy as np
import pandas as pd
import shap
import torch
from scipy.stats import pearsonr, rankdata

from src.localization.sensor_mapping import resolve_sensor_component

logger = logging.getLogger(__name__)


def _as_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite one- or two-dimensional array.")
    return matrix


def compute_shap_explanations(
    model: Any,
    background_data: np.ndarray,
    instance: np.ndarray,
    feature_names: Sequence[str] | None = None,
    prediction_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    random_state: int = 42,
    max_evals: int | None = None,
) -> pd.DataFrame:
    """Compute TreeSHAP for tree estimators or permutation SHAP for black boxes."""
    background = _as_matrix(background_data, "background_data")
    sample = _as_matrix(instance, "instance")
    if sample.shape[0] != 1 or sample.shape[1] != background.shape[1]:
        raise ValueError("instance must have one row and match the background feature count.")
    names = list(feature_names or [f"feature_{i}" for i in range(sample.shape[1])])
    if len(names) != sample.shape[1]:
        raise ValueError("feature_names must match the input feature count.")

    if prediction_fn is None:
        if model is None or not hasattr(model, "predict"):
            raise ValueError("Provide a fitted tree model or a model-agnostic prediction_fn.")
        explanation = shap.TreeExplainer(model)(sample)
    else:
        explainer = shap.PermutationExplainer(
            prediction_fn,
            shap.maskers.Independent(background[:1]),
            feature_names=names,
            seed=random_state,
        )
        explanation = explainer(
            sample,
            max_evals=max_evals or 2 * (sample.shape[1] + 1),
            silent=True,
        )

    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[..., -1]
    values = values.reshape(sample.shape[0], -1)[0]
    frame = pd.DataFrame({"Sensor": names, "SHAP Value": values})
    frame["Abs SHAP"] = frame["SHAP Value"].abs()
    return frame.sort_values("Abs SHAP", ascending=False).reset_index(drop=True)


def compute_lime_explanations(
    training_data: np.ndarray,
    instance: np.ndarray,
    prediction_fn: Callable[[np.ndarray], np.ndarray],
    feature_names: Sequence[str] | None = None,
    random_state: int = 42,
    num_samples: int = 512,
) -> pd.DataFrame:
    """Fit a LIME local surrogate using normal-calibration observations."""
    training = _as_matrix(training_data, "training_data")
    sample = _as_matrix(instance, "instance")
    if sample.shape[0] != 1 or sample.shape[1] != training.shape[1]:
        raise ValueError("instance must have one row and match training_data features.")
    names = list(feature_names or [f"feature_{i}" for i in range(sample.shape[1])])
    if len(names) != sample.shape[1]:
        raise ValueError("feature_names must match the input feature count.")

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=training,
        mode="regression",
        feature_names=names,
        feature_selection="none",
        discretize_continuous=False,
        sample_around_instance=True,
        random_state=random_state,
    )
    explanation = explainer.explain_instance(
        sample[0],
        prediction_fn,
        num_features=len(names),
        num_samples=num_samples,
    )
    local_weights = dict(explanation.local_exp.get(0, []))
    frame = pd.DataFrame(
        {
            "Sensor": names,
            "LIME Weight": [float(local_weights.get(i, 0.0)) for i in range(len(names))],
        }
    )
    frame["Abs LIME"] = frame["LIME Weight"].abs()
    return frame.sort_values("Abs LIME", ascending=False).reset_index(drop=True)


def _rank_stability(attributions: Sequence[np.ndarray]) -> float:
    correlations = []
    for first, second in combinations(attributions, 2):
        first_ranks = rankdata(first)
        second_ranks = rankdata(second)
        if np.std(first_ranks) == 0 or np.std(second_ranks) == 0:
            correlation = 1.0 if np.array_equal(first_ranks, second_ranks) else 0.0
        else:
            correlation = float(pearsonr(first_ranks, second_ranks).statistic)
        correlations.append(correlation if np.isfinite(correlation) else 0.0)
    return float(np.mean(correlations)) if correlations else 1.0


def compute_xai_explanations(
    demo_pipe: dict[str, Any],
    normal_window_raw: pd.DataFrame,
    disturbed_window: pd.DataFrame,
    target_sensor: str,
) -> dict[str, Any]:
    """Explain sensor disturbances with seeded SHAP/LIME and masking metrics."""
    model = demo_pipe["model"]
    preprocessor = demo_pipe["preprocessor"]
    adjacency = demo_pipe["adj_matrix"]
    variance = np.asarray(demo_pipe["mu_sq_h"], dtype=np.float32)
    sensor_columns = list(demo_pipe["sensor_cols"])
    normal_target_raw = demo_pipe["normal_target_raw"]
    calibration_windows = np.asarray(demo_pipe["xai_calibration_windows"], dtype=np.float32)
    component_map = demo_pipe.get("component_map", {})

    normal_x = preprocessor.transform_features(normal_window_raw, sensor_columns)[np.newaxis].astype(np.float32)
    normal_y = preprocessor.transform_features(normal_target_raw, sensor_columns)[np.newaxis].astype(np.float32)
    disturbed_x = preprocessor.transform_features(disturbed_window, sensor_columns)[np.newaxis].astype(np.float32)
    if normal_x.shape != disturbed_x.shape or normal_x.shape[1] != 12:
        raise ValueError("XAI inputs must be matching 12-hour windows.")

    sensor_count = len(sensor_columns)
    delta = disturbed_x - normal_x
    direction = delta[0]
    denominator = np.sum(np.square(direction), axis=0)
    differences = calibration_windows - normal_x
    numerators = np.sum(differences * direction[np.newaxis, :, :], axis=1)
    calibration_coefficients = np.divide(
        numerators,
        denominator[np.newaxis, :],
        out=np.zeros_like(numerators),
        where=denominator[np.newaxis, :] > 1e-12,
    )
    if len(calibration_coefficients) < 2:
        calibration_coefficients = np.repeat(calibration_coefficients, 2, axis=0)

    def evaluate_batch(batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            predictions = model(torch.from_numpy(batch.astype(np.float32)), adj_matrix=adjacency).numpy()
        residuals = np.square(predictions[:, 11, :] - normal_y[:, 11, :]) / variance[11]
        return np.mean(residuals, axis=1)

    def predict_sensor_coefficients(coefficients: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(coefficients, dtype=np.float32)
        batch = normal_x + coefficients[:, np.newaxis, :] * delta
        return evaluate_batch(batch)

    seeds = [17, 29, 43, 71, 101]
    shap_runs = []
    lime_runs = []
    for seed in seeds:
        shap_frame = compute_shap_explanations(
            model=None,
            background_data=np.zeros((1, sensor_count), dtype=float),
            instance=np.ones((1, sensor_count), dtype=float),
            feature_names=sensor_columns,
            prediction_fn=predict_sensor_coefficients,
            random_state=seed,
            max_evals=2 * (sensor_count + 1) * 4,
        )
        lime_frame = compute_lime_explanations(
            training_data=calibration_coefficients,
            instance=np.ones((1, sensor_count), dtype=float),
            prediction_fn=predict_sensor_coefficients,
            feature_names=sensor_columns,
            random_state=seed,
            num_samples=512,
        )
        shap_runs.append(shap_frame.set_index("Sensor")["SHAP Value"].reindex(sensor_columns).to_numpy())
        lime_runs.append(lime_frame.set_index("Sensor")["LIME Weight"].reindex(sensor_columns).to_numpy())

    shap_values = np.mean(shap_runs, axis=0)
    lime_weights = np.mean(lime_runs, axis=0)
    shap_frame = pd.DataFrame({"Sensor": sensor_columns, "SHAP Value": shap_values})
    shap_frame["Abs SHAP"] = shap_frame["SHAP Value"].abs()
    shap_frame["Direction"] = np.where(
        shap_frame["SHAP Value"] >= 0, "Increased anomaly score", "Reduced anomaly score"
    )
    shap_frame = shap_frame.sort_values("Abs SHAP", ascending=False).reset_index(drop=True)

    lime_frame = pd.DataFrame({"Sensor": sensor_columns, "LIME Weight": lime_weights})
    lime_frame["Abs LIME"] = lime_frame["LIME Weight"].abs()
    lime_frame = lime_frame.sort_values("Abs LIME", ascending=False).reset_index(drop=True)

    shap_top = set(shap_frame["Sensor"].iloc[:4])
    lime_top = set(lime_frame["Sensor"].iloc[:4])
    intersection = len(shap_top & lime_top)
    union = len(shap_top | lime_top)
    jaccard = float(intersection / union) if union else 1.0
    agreement = "High" if jaccard >= 0.75 else "Moderate" if jaccard >= 0.5 else "Low"

    disturbed_score = float(predict_sensor_coefficients(np.ones((1, sensor_count)))[0])
    normal_score = float(evaluate_batch(normal_x)[0])
    top_indices = [sensor_columns.index(sensor) for sensor in shap_frame["Sensor"].iloc[:4]]

    def masked_score(masked_sensors: Sequence[int]) -> float:
        masked = disturbed_x.copy()
        masked[:, :, list(masked_sensors)] = normal_x[:, :, list(masked_sensors)]
        return float(evaluate_batch(masked)[0])

    top4_reduction = disturbed_score - masked_score(top_indices)
    random_reductions = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        random_indices = rng.choice(sensor_count, size=min(4, sensor_count), replace=False)
        random_reductions.append(disturbed_score - masked_score(random_indices))
    fidelity = {
        "top4_score_reduction": float(top4_reduction),
        "random4_mean_score_reduction": float(np.mean(random_reductions)),
        "top4_gain_over_random": float(top4_reduction - np.mean(random_reductions)),
        "random4_score_reductions": [float(value) for value in random_reductions],
        "masking_baseline": "restore selected sensor traces to the normal calibration window",
    }
    stability = {
        "metric": "mean pairwise Pearson correlation of sensor attribution ranks over 5 seeds",
        "shap": _rank_stability([np.abs(values) for values in shap_runs]),
        "lime": _rank_stability([np.abs(values) for values in lime_runs]),
    }

    mapping = []
    for sensor in shap_frame["Sensor"].iloc[:5]:
        resolved = resolve_sensor_component(sensor, component_map)
        mapping.append(
            {
                "Sensor": sensor,
                "Physical Component": f"{resolved['component_type']} {resolved['component_id']}",
                "Topology Region": resolved["topology_region"],
            }
        )

    metrics = {
        "fidelity": fidelity,
        "stability": stability,
        "agreement_jaccard": jaccard,
        "shap_top4_sensors": sorted(shap_top),
        "lime_top4_sensors": sorted(lime_top),
    }
    numeric_metrics = [top4_reduction, *random_reductions, stability["shap"], stability["lime"], jaccard]
    if not np.isfinite(numeric_metrics).all():
        raise FloatingPointError("XAI metrics contain non-finite values.")
    logger.info("XAI validation metrics: %s", metrics)

    return {
        "df_shap": shap_frame,
        "df_lime": lime_frame,
        "agreement": agreement,
        "overlap_count": intersection,
        "metrics": metrics,
        "xai_mapping": mapping,
        "s_dist": disturbed_score,
        "s_norm": normal_score,
    }