"""Compare H=12 forecasting with C-Town versus identity spatial adjacency."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.detection.dataset_loader import BATADALDatasetLoader
from src.detection.evaluate import DetectionEvaluator
from src.detection.graph_model import SpatioTemporalGNN, build_adjacency_matrix_from_graph
from src.detection.preprocessing import DataPreprocessor
from src.detection.scoring import calibrated_sensor_variance
from src.network.ctown_loader import load_ctown_network
from src.network.graph_builder import build_network_graph

WINDOW_SIZE = 12
FORECAST_HORIZON = 12
RANDOM_SEED = 42
EPOCHS = 50
BATCH_SIZE = 64
THRESHOLD_PERCENTILE = 99.8


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _attack_delays(
    timestamps: pd.Series,
    predicted_flags: np.ndarray,
    attacks: pd.DataFrame,
) -> dict[str, float | None]:
    delays = {}
    for attack in attacks.itertuples(index=False):
        start = pd.Timestamp(attack.start_time)
        end = pd.Timestamp(attack.end_time)
        in_attack = timestamps.between(start, end).to_numpy()
        detections = np.flatnonzero(in_attack & predicted_flags.astype(bool))
        delay_key = f"attack_{attack.attack_id}"
        if len(detections):
            first_detection = timestamps.iloc[detections[0]]
            delays[delay_key] = float((first_detection - start).total_seconds() / 3600.0)
        else:
            delays[delay_key] = None
    return delays


def _train_and_score(
    adjacency: torch.Tensor,
    adjacency_name: str,
    train_windows: np.ndarray,
    train_targets: np.ndarray,
    calibration_windows: np.ndarray,
    calibration_targets: np.ndarray,
    validation_windows: np.ndarray,
    validation_targets: np.ndarray,
    validation_labels: np.ndarray,
    validation_timestamps: pd.Series,
    sensor_columns: list[str],
    scaler,
    attacks: pd.DataFrame,
) -> dict:
    set_seed(RANDOM_SEED)
    model = SpatioTemporalGNN(
        in_features=len(sensor_columns),
        window_size=WINDOW_SIZE,
        hidden_dim=64,
        dropout=0.2,
        forecast_horizon=FORECAST_HORIZON,
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_windows), torch.from_numpy(train_targets)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for _ in range(EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x, adj_matrix=adjacency), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        calibration_predictions = model(torch.from_numpy(calibration_windows), adj_matrix=adjacency).numpy()
        validation_predictions = model(torch.from_numpy(validation_windows), adj_matrix=adjacency).numpy()

    calibration_errors = np.square(calibration_predictions - calibration_targets)
    validation_errors = np.square(validation_predictions - validation_targets)
    variance = calibrated_sensor_variance(calibration_errors, sensor_columns, scaler)
    calibration_scores = np.mean(calibration_errors[:, -1, :] / variance[-1], axis=1)
    validation_scores = np.mean(validation_errors[:, -1, :] / variance[-1], axis=1)
    threshold = float(np.percentile(calibration_scores, THRESHOLD_PERCENTILE))
    evaluator = DetectionEvaluator(threshold=threshold)
    metrics = evaluator.evaluate_predictions(validation_labels, validation_scores)
    predicted_flags = validation_scores >= threshold
    delays = _attack_delays(validation_timestamps, predicted_flags, attacks)
    observed_delays = [delay for delay in delays.values() if delay is not None]

    return {
        "adjacency": adjacency_name,
        "threshold": threshold,
        "metrics": metrics,
        "detected_attack_count": len(observed_delays),
        "total_attack_count": len(delays),
        "mean_detection_delay_hours": float(np.mean(observed_delays)) if observed_delays else None,
        "attack_detection_delay_hours": delays,
    }


def main() -> None:
    config = yaml.safe_load((ROOT_DIR / "configs" / "config.yaml").read_text(encoding="utf-8"))
    loader = BATADALDatasetLoader()
    dataset03, sensors03, _ = loader.load_csv(ROOT_DIR / "data" / "raw" / "BATADAL_dataset03.csv")
    dataset04, sensors04, label_column = loader.load_csv(ROOT_DIR / "data" / "raw" / "BATADAL_dataset04.csv")
    sensor_columns = [sensor for sensor in sensors03 if sensor in sensors04]

    split_index = int(0.8 * len(dataset03))
    preprocessor = DataPreprocessor(window_size=WINDOW_SIZE)
    train_frame = preprocessor.handle_missing_values(dataset03.iloc[:split_index], sensor_columns)
    calibration_frame = preprocessor.handle_missing_values(dataset03.iloc[split_index:], sensor_columns)
    validation_frame = preprocessor.handle_missing_values(dataset04, sensor_columns)
    preprocessor.fit_scaler(train_frame, sensor_columns)

    train_features = preprocessor.transform_features(train_frame, sensor_columns)
    calibration_features = preprocessor.transform_features(calibration_frame, sensor_columns)
    validation_features = preprocessor.transform_features(validation_frame, sensor_columns)
    train_windows, train_targets, _ = preprocessor.create_forecasting_windows(
        train_features, forecast_horizon=FORECAST_HORIZON
    )
    calibration_windows, calibration_targets, _ = preprocessor.create_forecasting_windows(
        calibration_features, forecast_horizon=FORECAST_HORIZON
    )
    validation_windows, validation_targets, _ = preprocessor.create_forecasting_windows(
        validation_features, forecast_horizon=FORECAST_HORIZON
    )

    label_offset = WINDOW_SIZE + FORECAST_HORIZON - 1
    validation_timestamps = validation_frame[loader.timestamp_col].iloc[label_offset:].reset_index(drop=True)
    if label_column is None:
        raise ValueError("Dataset04 must contain its partial label column.")
    attacks = pd.read_csv(ROOT_DIR / "data" / "ground_truth_attacks.csv")
    attacks = attacks[attacks["dataset"] == "Dataset04"]
    validation_labels = np.zeros(len(validation_timestamps), dtype=np.int8)
    for attack in attacks.itertuples(index=False):
        validation_labels[
            validation_timestamps.between(pd.Timestamp(attack.start_time), pd.Timestamp(attack.end_time))
        ] = 1
    if len(validation_labels) != len(validation_windows):
        raise RuntimeError("Validation labels and H=12 score timestamps are misaligned.")

    network = load_ctown_network(ROOT_DIR / config["paths"]["inp_file"])
    graph = build_network_graph(network)
    ctown_adjacency = build_adjacency_matrix_from_graph(graph, sensor_columns, max_hops=3)
    identity_adjacency = torch.eye(len(sensor_columns), dtype=ctown_adjacency.dtype)
    results = [
        _train_and_score(
            adjacency,
            adjacency_name,
            train_windows,
            train_targets,
            calibration_windows,
            calibration_targets,
            validation_windows,
            validation_targets,
            validation_labels,
            validation_timestamps,
            sensor_columns,
            preprocessor.scaler,
            attacks,
        )
        for adjacency_name, adjacency in (
            ("A_CTOWN", ctown_adjacency),
            ("Identity", identity_adjacency),
        )
    ]
    comparison_table = [
        {
            "adjacency": result["adjacency"],
            "F1": result["metrics"]["f1_score"],
            "PR_AUC": result["metrics"]["pr_auc"],
            "detection_delay_hours": result["mean_detection_delay_hours"],
            "detected_attacks": result["detected_attack_count"],
            "total_attacks": result["total_attack_count"],
        }
        for result in results
    ]
    report = {
        "experiment": "H=12 C-Town spatial adjacency versus identity adjacency",
        "architecture": {
            "model": "SpatioTemporalGNN",
            "input_features": len(sensor_columns),
            "window_size": WINDOW_SIZE,
            "forecast_horizon": FORECAST_HORIZON,
            "hidden_dim": 64,
            "dropout": 0.2,
            "epochs": EPOCHS,
            "random_seed": RANDOM_SEED,
        },
        "calibration": {
            "dataset": "Dataset03 chronological final 20% normal calibration partition",
            "variance_floor": "sensor-specific instrument precision floors",
            "threshold_percentile": THRESHOLD_PERCENTILE,
        },
        "evaluation": {
            "dataset": "Dataset04",
            "ground_truth_source": "data/ground_truth_attacks.csv",
            "label_rule": "positive within published attack intervals; normal outside those intervals",
            "raw_partial_ATT_FLAG_used_for_metrics": False,
            "labels_aligned_to_horizon": FORECAST_HORIZON,
            "attack_intervals": len(attacks),
        },
        "comparison_table": comparison_table,
        "runs": results,
    }
    output_path = ROOT_DIR / "results" / "graph_topology_ablation_report.json"
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(comparison_table, indent=2))
    print(f"Saved report to {output_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
