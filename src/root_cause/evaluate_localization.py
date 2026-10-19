"""Benchmark sensor- and graph-based localization on the 14 BATADAL attacks."""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.detection.dataset_loader import BATADALDatasetLoader
from src.detection.graph_model import (
    SpatioTemporalGNN,
    build_adjacency_matrix_from_graph,
    resolve_sensor_locations,
)
from src.detection.preprocessing import DataPreprocessor
from src.detection.scoring import calibrated_sensor_variance
from src.explainability.xai import compute_shap_explanations
from src.network.ctown_loader import load_ctown_network
from src.network.graph_builder import build_directed_hydraulic_graph, build_network_graph
from src.network.zoning import assign_zones
from src.root_cause.assessment import rank_upstream_components

logger = logging.getLogger(__name__)
ATTACKS_PATH = ROOT_DIR / "data" / "ground_truth_attacks.csv"
OUTPUT_PATH = ROOT_DIR / "results" / "localization_benchmark_report.json"
METHODS = ("baseline_a_raw_deviation", "baseline_b_top1_shap", "proposed_shap_reversed_flow_ppr")


def _component_nodes(
    graph: nx.Graph, component_type: str, component_id: str
) -> list[str]:
    if component_type == "Tank":
        return [component_id] if component_id in graph else []
    for source, destination, data in graph.edges(data=True):
        if str(data.get("id")) == component_id and str(data.get("type", "")).title() == component_type:
            return [source, destination]
    return []


def _sensor_component(
    sensor: str,
    sensor_nodes: dict[str, set[str]],
    graph: nx.Graph,
    zones: dict[str, int],
) -> dict[str, Any]:
    if sensor.startswith("P_"):
        component_type, component_id = "Junction", sensor[2:]
    elif sensor.startswith("L_"):
        component_type, component_id = "Tank", sensor[2:]
    elif sensor.startswith(("F_PU", "S_PU")):
        component_type, component_id = "Pump", sensor.split("_", 1)[1]
    elif sensor.startswith(("F_V", "S_V")):
        component_type, component_id = "Valve", sensor.split("_", 1)[1]
    else:
        component_type, component_id = "Unknown", sensor
    nodes = list(sensor_nodes.get(sensor, ()))
    return {
        "component_type": component_type,
        "component_id": component_id,
        "nodes": nodes,
        "dma_zone": _majority_zone(nodes, zones),
    }


def _majority_zone(nodes: list[str], zones: dict[str, int]) -> int | None:
    values = [int(zones[node]) for node in nodes if node in zones and int(zones[node]) >= 0]
    return Counter(values).most_common(1)[0][0] if values else None


def _predict_dataset(
    frame: pd.DataFrame,
    sensor_columns: list[str],
    preprocessor: DataPreprocessor,
    model: SpatioTemporalGNN,
    adjacency: torch.Tensor,
    variance: np.ndarray,
    dataset_name: str,
) -> dict[str, Any]:
    cleaned = preprocessor.handle_missing_values(frame, sensor_columns)
    features = preprocessor.transform_features(cleaned, sensor_columns)
    windows, targets, _ = preprocessor.create_forecasting_windows(
        features, forecast_horizon=12
    )
    with torch.no_grad():
        predictions = model(torch.from_numpy(windows), adj_matrix=adjacency).numpy()
    normalized_residuals = np.square(predictions - targets) / variance
    sensor_scores = normalized_residuals[:, -1, :]
    window_scores = np.mean(sensor_scores, axis=1)
    target_timestamps = cleaned["DATETIME"].iloc[23:].reset_index(drop=True)
    if len(target_timestamps) != len(window_scores):
        raise RuntimeError(f"{dataset_name} target timestamps do not align to H=12 forecasts.")
    return {
        "windows": windows,
        "targets": targets,
        "sensor_scores": sensor_scores,
        "window_scores": window_scores,
        "target_timestamps": target_timestamps,
    }


def _shap_attributions(
    model: SpatioTemporalGNN,
    adjacency: torch.Tensor,
    variance: np.ndarray,
    normal_windows: np.ndarray,
    attack_window: np.ndarray,
    attack_target: np.ndarray,
    sensor_columns: list[str],
    seed: int,
) -> pd.DataFrame:
    distances = np.mean(np.square(normal_windows - attack_window), axis=(1, 2))
    reference = normal_windows[int(np.argmin(distances)) : int(np.argmin(distances)) + 1]
    delta = attack_window - reference

    def score_masked(coefficients: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(coefficients, dtype=np.float32)
        masked = reference + coefficients[:, np.newaxis, :] * delta
        with torch.no_grad():
            predictions = model(torch.from_numpy(masked), adj_matrix=adjacency).numpy()
        actual_targets = np.repeat(attack_target, len(coefficients), axis=0)
        residual = np.square(predictions[:, -1, :] - actual_targets[:, -1, :]) / variance[-1]
        return np.mean(residual, axis=1)

    return compute_shap_explanations(
        model=None,
        background_data=np.zeros((1, len(sensor_columns)), dtype=float),
        instance=np.ones((1, len(sensor_columns)), dtype=float),
        feature_names=sensor_columns,
        prediction_fn=score_masked,
        random_state=seed,
        max_evals=2 * (len(sensor_columns) + 1),
    )


def _ranked_sensor_components(
    ranked_sensors: list[str],
    sensor_nodes: dict[str, set[str]],
    graph: nx.Graph,
    zones: dict[str, int],
) -> list[dict[str, Any]]:
    results = []
    seen = set()
    for sensor in ranked_sensors:
        item = _sensor_component(sensor, sensor_nodes, graph, zones)
        key = (item["component_type"], item["component_id"])
        if item["component_type"] in {"Pump", "Valve", "Tank", "Junction"} and key not in seen:
            seen.add(key)
            results.append(item)
    return results


def _component_hop_distance(
    predicted: dict[str, Any],
    target_type: str,
    target_id: str,
    undirected_graph: nx.Graph,
) -> float | None:
    if predicted["component_type"] == target_type and predicted["component_id"] == target_id:
        return 0.0
    source_nodes = predicted.get("nodes") or _component_nodes(
        undirected_graph, predicted["component_type"], predicted["component_id"]
    )
    target_nodes = _component_nodes(undirected_graph, target_type, target_id)
    distances = [
        nx.shortest_path_length(undirected_graph, source, target)
        for source in source_nodes
        for target in target_nodes
        if source in undirected_graph
        and target in undirected_graph
        and nx.has_path(undirected_graph, source, target)
    ]
    return float(min(distances)) if distances else None


def _score_method(
    predictions: list[dict[str, Any]], method_name: str, undirected_graph: nx.Graph
) -> dict[str, Any]:
    count = len(predictions)
    top1 = sum(
        bool(row["predicted_components"])
        and row["predicted_components"][0]["component_type"] == row["target_component_type"]
        and row["predicted_components"][0]["component_id"] == row["target_component"]
        for row in predictions
    )
    top3 = sum(
        any(
            candidate["component_type"] == row["target_component_type"]
            and candidate["component_id"] == row["target_component"]
            for candidate in row["predicted_components"][:3]
        )
        for row in predictions
    )
    dma = sum(
        row["predicted_dma_zone"] is not None
        and row["predicted_dma_zone"] == row["target_dma_zone"]
        for row in predictions
    )
    distances = [row["topological_hop_distance"] for row in predictions if row["topological_hop_distance"] is not None]
    by_attack_type = {}
    for attack_type in sorted({row["attack_type"] for row in predictions}):
        subset = [row for row in predictions if row["attack_type"] == attack_type]
        subset_top1 = sum(
            bool(row["predicted_components"])
            and row["predicted_components"][0]["component_type"] == row["target_component_type"]
            and row["predicted_components"][0]["component_id"] == row["target_component"]
            for row in subset
        )
        subset_top3 = sum(
            any(
                candidate["component_type"] == row["target_component_type"]
                and candidate["component_id"] == row["target_component"]
                for candidate in row["predicted_components"][:3]
            )
            for row in subset
        )
        by_attack_type[attack_type] = {
            "attacks": len(subset),
            "top1_component_accuracy_pct": 100.0 * subset_top1 / len(subset),
            "top3_component_accuracy_pct": 100.0 * subset_top3 / len(subset),
        }

    return {
        "method": method_name,
        "attacks": count,
        "top1_component_accuracy_pct": 100.0 * top1 / count,
        "top3_component_accuracy_pct": 100.0 * top3 / count,
        "dma_zone_accuracy_pct": 100.0 * dma / count,
        "mean_topological_hop_distance": float(np.mean(distances)) if distances else None,
        "by_attack_type": by_attack_type,
    }


def run_localization_benchmark() -> dict[str, Any]:
    """Run the 14-attack localization comparison and save its JSON report."""
    config = yaml.safe_load((ROOT_DIR / "configs" / "config.yaml").read_text(encoding="utf-8"))
    loader = BATADALDatasetLoader()
    dataset03, sensor_columns, _ = loader.load_csv(ROOT_DIR / "data" / "raw" / "BATADAL_dataset03.csv")
    split = int(0.8 * len(dataset03))
    preprocessor = DataPreprocessor(window_size=12)
    train_frame = preprocessor.handle_missing_values(dataset03.iloc[:split], sensor_columns)
    calibration_frame = preprocessor.handle_missing_values(dataset03.iloc[split:], sensor_columns)
    preprocessor.fit_scaler(train_frame, sensor_columns)
    calibration_features = preprocessor.transform_features(calibration_frame, sensor_columns)
    calibration_windows, calibration_targets, _ = preprocessor.create_forecasting_windows(
        calibration_features, forecast_horizon=12
    )

    net = load_ctown_network(ROOT_DIR / config["paths"]["inp_file"])
    undirected_graph = build_network_graph(net)
    directed_graph = build_directed_hydraulic_graph(net)
    adjacency = build_adjacency_matrix_from_graph(undirected_graph, sensor_columns, max_hops=3)
    sensor_nodes = {
        sensor: set(nodes)
        for sensor, nodes in resolve_sensor_locations(undirected_graph, sensor_columns).items()
    }
    dma_zones = assign_zones(undirected_graph, method="dma")

    model_path = ROOT_DIR / "models" / "detection_gnn_h12_candidate.pt"
    model = SpatioTemporalGNN(
        in_features=len(sensor_columns),
        window_size=12,
        hidden_dim=64,
        dropout=0.2,
        forecast_horizon=12,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        calibration_predictions = model(torch.from_numpy(calibration_windows), adj_matrix=adjacency).numpy()
    variance = calibrated_sensor_variance(
        np.square(calibration_predictions - calibration_targets), sensor_columns, preprocessor.scaler
    )

    attacks = pd.read_csv(ATTACKS_PATH)
    results_by_dataset = {}
    for dataset_name, filename in (
        ("Dataset04", "BATADAL_dataset04.csv"),
        ("Test Dataset", "BATADAL_test_dataset.csv"),
    ):
        frame, dataset_sensors, _ = loader.load_csv(ROOT_DIR / "data" / "raw" / filename)
        common_sensors = [sensor for sensor in sensor_columns if sensor in dataset_sensors]
        if common_sensors != sensor_columns:
            raise ValueError(f"{filename} does not contain the complete trained sensor set.")
        prediction_data = _predict_dataset(
            frame,
            sensor_columns,
            preprocessor,
            model,
            adjacency,
            variance,
            dataset_name,
        )
        dataset_attacks = attacks[attacks["dataset"] == dataset_name]
        attack_records = []
        for attack in dataset_attacks.itertuples(index=False):
            start = pd.Timestamp(attack.start_time)
            end = pd.Timestamp(attack.end_time)
            in_attack = prediction_data["target_timestamps"].between(start, end).to_numpy()
            candidate_indices = np.flatnonzero(in_attack)
            if not len(candidate_indices):
                raise ValueError(f"Attack {attack.attack_id} has no forecast window in its interval.")
            peak_index = int(candidate_indices[np.argmax(prediction_data["window_scores"][candidate_indices])])
            raw_rank = np.argsort(prediction_data["sensor_scores"][peak_index])[::-1]
            raw_sensors = [sensor_columns[index] for index in raw_rank]
            peak_shap_frame = _shap_attributions(
                model,
                adjacency,
                variance,
                calibration_windows,
                prediction_data["windows"][peak_index : peak_index + 1],
                prediction_data["targets"][peak_index : peak_index + 1],
                sensor_columns,
                seed=42 + int(attack.attack_id),
            )
            shap_sensors = peak_shap_frame["Sensor"].astype(str).tolist()
            ranked_interval_indices = candidate_indices[
                np.argsort(prediction_data["window_scores"][candidate_indices])[-3:]
            ]
            interval_shap_frames = [peak_shap_frame]
            for interval_index in ranked_interval_indices:
                if interval_index == peak_index:
                    continue
                interval_shap_frames.append(
                    _shap_attributions(
                        model,
                        adjacency,
                        variance,
                        calibration_windows,
                        prediction_data["windows"][interval_index : interval_index + 1],
                        prediction_data["targets"][interval_index : interval_index + 1],
                        sensor_columns,
                        seed=42 + int(attack.attack_id) + int(interval_index),
                    )
                )
            ppr_shap_values = np.max(
                [
                    frame.set_index("Sensor").reindex(sensor_columns)["SHAP Value"].abs().to_numpy()
                    for frame in interval_shap_frames
                ],
                axis=0,
            )
            ppr_shap_frame = pd.DataFrame(
                {"Sensor": sensor_columns, "SHAP Value": ppr_shap_values}
            )
            ppr_shap_frame["Abs SHAP"] = ppr_shap_frame["SHAP Value"].abs()
            ppr_shap_frame = ppr_shap_frame.sort_values("Abs SHAP", ascending=False).reset_index(drop=True)
            ppr_result = rank_upstream_components(
                directed_graph,
                sensor_nodes,
                dict(zip(ppr_shap_frame["Sensor"], ppr_shap_frame["SHAP Value"], strict=True)),
                dma_zones,
            )
            method_candidates = {
                "baseline_a_raw_deviation": _ranked_sensor_components(
                    raw_sensors, sensor_nodes, undirected_graph, dma_zones
                ),
                "baseline_b_top1_shap": _ranked_sensor_components(
                    shap_sensors, sensor_nodes, undirected_graph, dma_zones
                )[:1],
                "proposed_shap_reversed_flow_ppr": ppr_result["ranked_components"],
            }
            for method, candidates in method_candidates.items():
                prediction = candidates[0] if candidates else None
                predicted_zone = prediction.get("dma_zone") if prediction else None
                if method == "proposed_shap_reversed_flow_ppr":
                    predicted_zone = ppr_result["affected_dma_zone"]
                hop_distance = (
                    _component_hop_distance(
                        prediction,
                        str(attack.component_type),
                        str(attack.target_component),
                        undirected_graph,
                    )
                    if prediction
                    else None
                )
                attack_records.append(
                    {
                        "method": method,
                        "attack_id": int(attack.attack_id),
                        "dataset": dataset_name,
                        "attack_type": str(attack.attack_type),
                        "target_component_type": str(attack.component_type),
                        "target_component": str(attack.target_component),
                        "target_dma_zone": int(str(attack.dma_zone).replace("DMA", "")),
                        "timestamp_of_peak_window": str(prediction_data["target_timestamps"].iloc[peak_index]),
                        "raw_deviation_top_sensor": raw_sensors[0],
                        "shap_top_sensor": shap_sensors[0],
                        "shap_top_sensors": shap_sensors[:5],
                        "ppr_interval_shap_top_sensors": ppr_shap_frame["Sensor"].head(5).tolist(),
                        "predicted_components": candidates[:3],
                        "predicted_dma_zone": predicted_zone,
                        "topological_hop_distance": hop_distance,
                        "ppr_propagation_path": ppr_result["propagation_path"]
                        if method == "proposed_shap_reversed_flow_ppr"
                        else None,
                    }
                )

        results_by_dataset[dataset_name] = {
            method: _score_method(
                [row for row in attack_records if row["method"] == method],
                method,
                undirected_graph,
            )
            for method in METHODS
        }
        results_by_dataset[dataset_name]["attack_results"] = attack_records

    combined_records = [
        row
        for dataset_result in results_by_dataset.values()
        for row in dataset_result["attack_results"]
    ]
    combined_metrics = {
        method: _score_method(
            [row for row in combined_records if row["method"] == method], method, undirected_graph
        )
        for method in METHODS
    }
    report = {
        "benchmark": "BATADAL 14-attack component localization",
        "model_checkpoint": "models/detection_gnn_h12_candidate.pt",
        "ground_truth": "data/ground_truth_attacks.csv",
        "hydraulic_graph": {
            "simulation": "WNTR EPANET baseline simulation from CTOWN.INP",
            "directed_edges": directed_graph.number_of_edges(),
            "orientation": "predominant nonzero signed flow; equipment orientation for zero-flow links",
            "ppr_alpha": 0.5,
            "ppr_direct_shap_weight": 0.5,
            "candidate_score_normalization": "separate max normalization of PPR and direct component SHAP mass, then equal-weight fusion",
        },
        "evidence_selection": "Peak normalized-residual H=12 window whose target timestamp falls within each attack interval.",
        "methods": {
            "baseline_a_raw_deviation": "Rank sensors by final-horizon instrument-normalized squared residual; map to directly sensed components.",
            "baseline_b_top1_shap": "Rank sensors by real SHAP permutation attribution; map the top sensor directly to its component.",
            "proposed_shap_reversed_flow_ppr": "Maximum absolute sensor SHAP magnitude over the three highest-score event windows, then personalize PageRank on the reversed simulated-flow graph.",
        },
        "metrics": combined_metrics,
        "by_dataset": results_by_dataset,
        "all_attack_results": combined_records,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = run_localization_benchmark()
    print(json.dumps(report["metrics"], indent=2))
    print(f"Saved localization benchmark report to {OUTPUT_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
