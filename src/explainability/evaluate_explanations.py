"""Evaluate sensor explanation fidelity, seed stability, and SHAP/LIME agreement."""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.detection.graph_model import SpatioTemporalGNN, build_adjacency_matrix_from_graph
from src.evaluation.data_prep import build_splits
from src.evaluation.detection_eval import calibrate
from src.evaluation.shapley import masked_window_value_fn, permutation_shapley
from src.explainability.xai import compute_lime_explanations
from src.network.ctown_loader import load_ctown_network
from src.network.graph_builder import build_network_graph

SEEDS = (42, 123, 2024, 3407, 9999)
TOP_K = 5
RANDOM_MASKS = 20


def _jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def _load_model(seed: int, sensor_count: int, checkpoint_dir: Path) -> SpatioTemporalGNN:
    model = SpatioTemporalGNN(
        in_features=sensor_count,
        window_size=12,
        hidden_dim=64,
        dropout=0.2,
        forecast_horizon=12,
    )
    path = checkpoint_dir / f"detection_gnn_A_CTOWN_seed{seed}.pt"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def _predict_fn(model: SpatioTemporalGNN, adjacency: torch.Tensor):
    def predict(windows: np.ndarray) -> np.ndarray:
        outputs = []
        with torch.no_grad():
            for start in range(0, len(windows), 1024):
                batch = torch.from_numpy(np.ascontiguousarray(windows[start:start + 1024], dtype=np.float32))
                outputs.append(model(batch, adj_matrix=adjacency).numpy())
        return np.concatenate(outputs, axis=0)
    return predict


def _sensor_coefficients(
    calibration_windows: np.ndarray,
    reference: np.ndarray,
    attack_window: np.ndarray,
) -> np.ndarray:
    delta = attack_window - reference
    denominator = np.square(delta).sum(axis=0)
    differences = calibration_windows - reference
    numerator = np.sum(differences * delta[None, :, :], axis=1)
    return np.divide(
        numerator,
        denominator[None, :],
        out=np.zeros_like(numerator),
        where=denominator[None, :] > 1e-12,
    )


def _evaluate_window(
    *,
    predict_fn,
    calibration_windows: np.ndarray,
    attack_window: np.ndarray,
    attack_target: np.ndarray,
    variance: np.ndarray,
    sensors: list[str],
    seed: int,
    n_permutations: int,
) -> dict[str, Any]:
    distances = np.square(calibration_windows - attack_window).mean(axis=(1, 2))
    reference = calibration_windows[int(np.argmin(distances))]
    value_fn = masked_window_value_fn(predict_fn, reference, attack_window, attack_target, variance)
    shap_values = permutation_shapley(
        value_fn,
        len(sensors),
        n_permutations=n_permutations,
        rng=np.random.default_rng(seed),
    )
    shap_order = np.argsort(-np.abs(shap_values))
    shap_top = {sensors[index] for index in shap_order[:TOP_K]}
    full_mask = np.ones((1, len(sensors)), dtype=float)
    full_score = float(value_fn(full_mask)[0])
    shap_mask = full_mask.copy()
    shap_mask[0, list(shap_order[:TOP_K])] = 0.0
    shap_drop = full_score - float(value_fn(shap_mask)[0])

    rng = np.random.default_rng(seed + 100_003)
    random_drops = []
    for _ in range(RANDOM_MASKS):
        random_mask = full_mask.copy()
        random_indices = rng.choice(len(sensors), size=TOP_K, replace=False)
        random_mask[0, random_indices] = 0.0
        random_drops.append(full_score - float(value_fn(random_mask)[0]))

    coefficients = _sensor_coefficients(calibration_windows, reference, attack_window)
    delta = attack_window - reference

    def predict_coefficients(coefficient_rows: np.ndarray) -> np.ndarray:
        coefficient_rows = np.asarray(coefficient_rows, dtype=float)
        return value_fn(coefficient_rows)

    lime_frame = compute_lime_explanations(
        training_data=coefficients,
        instance=np.ones((1, len(sensors)), dtype=float),
        prediction_fn=predict_coefficients,
        feature_names=sensors,
        random_state=seed,
        num_samples=512,
    )
    lime_top = set(lime_frame["Sensor"].head(TOP_K).astype(str))
    return {
        "shap_top5": sorted(shap_top),
        "lime_top5": sorted(lime_top),
        "shap_lime_overlap_count": len(shap_top & lime_top),
        "shap_lime_jaccard": _jaccard(shap_top, lime_top),
        "full_anomaly_score": full_score,
        "top5_shap_score_drop": float(shap_drop),
        "random5_score_drops": [float(value) for value in random_drops],
        "top5_shap_mean_absolute_value": float(np.abs(shap_values[shap_order[:TOP_K]]).sum()),
    }


def evaluate_explanation_quality(
    *,
    root: str | Path,
    attacks: pd.DataFrame,
    seeds: tuple[int, ...] = SEEDS,
    n_permutations: int = 50,
) -> dict[str, Any]:
    """Measure explanation quality on one ensemble-peak window per test attack."""
    root = Path(root)
    splits = build_splits(root, attacks, harmonise=True)
    network = load_ctown_network(root / "data" / "raw" / "CTOWN.INP")
    graph = build_network_graph(network)
    adjacency = build_adjacency_matrix_from_graph(graph, splits.sensors, max_hops=3)
    windows, targets, timestamps = splits.evaluation["Test Dataset"]
    records = []
    checkpoint_dir = root / "models" / "seeds"
    evaluations: dict[int, dict[str, Any]] = {}

    for seed in seeds:
        model = _load_model(seed, len(splits.sensors), checkpoint_dir)
        predict_fn = _predict_fn(model, adjacency)
        variance, threshold = calibrate(predict_fn, splits)
        sensor_scores = np.square(predict_fn(windows) - targets)[:, -1, :] / variance[-1]
        window_scores = sensor_scores.mean(axis=1)
        evaluations[seed] = {
            "predict_fn": predict_fn,
            "variance": variance,
            "threshold": threshold,
            "window_scores": window_scores,
        }

    for attack in attacks[attacks["dataset"] == "Test Dataset"].itertuples(index=False):
        inside = timestamps.between(
            pd.Timestamp(attack.start_time), pd.Timestamp(attack.end_time)
        ).to_numpy()
        pool = np.flatnonzero(inside)
        if not len(pool):
            raise ValueError(f"Attack {attack.attack_id} has no target-aligned test windows.")
        ensemble_scores = np.mean(np.stack([
            evaluations[seed]["window_scores"][pool] for seed in seeds
        ]), axis=0)
        window_index = int(pool[np.argmax(ensemble_scores)])
        for seed in seeds:
            evaluation = evaluations[seed]
            quality = _evaluate_window(
                predict_fn=evaluation["predict_fn"],
                calibration_windows=splits.calibration[0],
                attack_window=windows[window_index],
                attack_target=targets[window_index],
                variance=evaluation["variance"],
                sensors=splits.sensors,
                seed=seed + int(attack.attack_id) * 1009,
                n_permutations=n_permutations,
            )
            records.append({
                "seed": seed,
                "attack_id": int(attack.attack_id),
                "attack_type": str(attack.attack_type),
                "target": f"{attack.component_type} {attack.target_component}",
                "target_timestamp": str(timestamps.iloc[window_index]),
                "detected_at_threshold": bool(
                    evaluation["window_scores"][window_index] >= evaluation["threshold"]
                ),
                **quality,
            })

    by_attack: dict[str, Any] = {}
    stability_scores = []
    agreement_scores = []
    for attack_id in sorted({record["attack_id"] for record in records}):
        attack_records = [record for record in records if record["attack_id"] == attack_id]
        stability = [
            _jaccard(set(first["shap_top5"]), set(second["shap_top5"]))
            for first, second in combinations(attack_records, 2)
        ]
        stability_scores.extend(stability)
        agreement_scores.extend(record["shap_lime_jaccard"] for record in attack_records)
        by_attack[str(attack_id)] = {
            "target": attack_records[0]["target"],
            "shap_top5_seed_sets": {str(record["seed"]): record["shap_top5"] for record in attack_records},
            "pairwise_shap_top5_jaccard_mean": float(np.mean(stability)) if stability else 1.0,
            "shap_lime_top5_jaccard_mean": float(np.mean([
                record["shap_lime_jaccard"] for record in attack_records
            ])),
            "shap_lime_overlap_count_mean": float(np.mean([
                record["shap_lime_overlap_count"] for record in attack_records
            ])),
        }

    top5_drops = np.asarray([record["top5_shap_score_drop"] for record in records], dtype=float)
    random_drops = np.asarray([
        value for record in records for value in record["random5_score_drops"]
    ], dtype=float)
    return {
        "study": "Test Dataset ensemble-peak residual window per attack; same timestamp across model seeds",
        "seeds": list(seeds),
        "attack_count": len(by_attack),
        "n_permutations": n_permutations,
        "top_k": TOP_K,
        "random_masks_per_window": RANDOM_MASKS,
        "fidelity": {
            "masking": "restore selected sensor traces to the nearest normal calibration window",
            "top5_shap_mean_score_drop": float(np.mean(top5_drops)),
            "top5_shap_score_drop_std": float(np.std(top5_drops, ddof=1)) if len(top5_drops) > 1 else 0.0,
            "random5_mean_score_drop": float(np.mean(random_drops)),
            "random5_score_drop_std": float(np.std(random_drops, ddof=1)) if len(random_drops) > 1 else 0.0,
            "shap_gain_over_random": float(np.mean(top5_drops) - np.mean(random_drops)),
        },
        "stability": {
            "metric": "pairwise Jaccard overlap of SHAP top-5 sensor sets across model seeds, per attack",
            "mean_pairwise_jaccard": float(np.mean(stability_scores)) if stability_scores else 1.0,
            "pair_count": len(stability_scores),
        },
        "agreement": {
            "metric": "SHAP/LIME top-5 set intersection and Jaccard",
            "mean_overlap_count": float(np.mean([record["shap_lime_overlap_count"] for record in records])),
            "mean_jaccard": float(np.mean(agreement_scores)) if agreement_scores else 1.0,
        },
        "by_attack": by_attack,
        "records": records,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    attacks = pd.read_csv(root / "data" / "ground_truth_attacks.csv")
    report = evaluate_explanation_quality(root=root, attacks=attacks)
    output_path = root / "results" / "explanation_quality_report.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "path": str(output_path),
        "fidelity": report["fidelity"],
        "stability": report["stability"],
        "agreement": report["agreement"],
    }, indent=2))


if __name__ == "__main__":
    main()