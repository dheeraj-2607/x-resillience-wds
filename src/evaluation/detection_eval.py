"""Model-agnostic detection scoring (same rule as scratch/run_graph_topology_ablation.py).

score(window) = mean over sensors of (last-horizon squared residual / calibrated variance),
threshold     = 99.8th percentile of calibration scores (normal data only).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from src.detection.evaluate import DetectionEvaluator
from src.detection.scoring import calibrated_sensor_variance
from src.evaluation.data_prep import Splits

THRESHOLD_PERCENTILE = 99.8


def calibrate(predict_fn: Callable[[np.ndarray], np.ndarray], splits: Splits):
    x, y = splits.calibration
    errors = np.square(predict_fn(x) - y)
    variance = calibrated_sensor_variance(errors, splits.sensors, splits.preprocessor.scaler)
    scores = np.mean(errors[:, -1, :] / variance[-1], axis=1)
    return variance, float(np.percentile(scores, THRESHOLD_PERCENTILE))


def window_scores(predict_fn, windows, targets, variance) -> np.ndarray:
    return np.mean(np.square(predict_fn(windows) - targets)[:, -1, :] / variance[-1], axis=1)


def attack_delays(timestamps: pd.Series, flags: np.ndarray, attacks: pd.DataFrame) -> dict[int, float | None]:
    out = {}
    for attack in attacks.itertuples(index=False):
        start = pd.Timestamp(attack.start_time)
        inside = timestamps.between(start, pd.Timestamp(attack.end_time)).to_numpy()
        hits = np.flatnonzero(inside & flags)
        out[int(attack.attack_id)] = (
            float((timestamps.iloc[hits[0]] - start).total_seconds() / 3600) if len(hits) else None
        )
    return out


def evaluate_detection(predict_fn, splits: Splits, attacks: pd.DataFrame) -> dict[str, Any]:
    variance, threshold = calibrate(predict_fn, splits)
    evaluator = DetectionEvaluator(threshold=threshold)
    report: dict[str, Any] = {"threshold": threshold}
    for name, (x, y, ts) in splits.evaluation.items():
        scores = window_scores(predict_fn, x, y, variance)
        metrics = evaluator.evaluate_predictions(splits.labels[name], scores)
        delays = attack_delays(ts, scores >= threshold, attacks[attacks.dataset == name])
        report[name] = {
            "metrics": metrics,
            "attacks_detected": sum(v is not None for v in delays.values()),
            "attacks_total": len(delays),
            "delay_hours": delays,
        }
    report["variance"] = variance
    return report


def summarise_seeds(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """runs: [{'seed', 'adjacency', 'report'}] -> mean ± sd table per adjacency and dataset."""
    rows = []
    for adjacency in sorted({r["adjacency"] for r in runs}):
        group = [r for r in runs if r["adjacency"] == adjacency]
        for dataset in ("Dataset04", "Test Dataset"):
            row = {"adjacency": adjacency, "dataset": dataset, "seeds": len(group)}
            for metric in ("f1_score", "recall", "precision", "roc_auc", "pr_auc", "false_positive_rate"):
                values = np.array([g["report"][dataset]["metrics"][metric] for g in group], dtype=float)
                row[metric] = f"{values.mean():.3f} ± {values.std(ddof=1) if len(values) > 1 else 0:.3f}"
            row["attacks_detected"] = "/".join(str(g["report"][dataset]["attacks_detected"]) for g in group)
            rows.append(row)
    return pd.DataFrame(rows)
