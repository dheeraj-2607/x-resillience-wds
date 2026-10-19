"""
train.py

Normal-Only Spatio-Temporal GNN Forecasting Anomaly Detection Pipeline.
STRICT DATA GATING ENFORCED: Refuses to execute model training when a real
BATADAL dataset is not supplied. Synthetic mock data is strictly rejected for
training to preserve research integrity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.detection.dataset_loader import BATADALDatasetLoader, verify_dataset_is_real
from src.detection.evaluate import DetectionEvaluator
from src.detection.graph_model import SpatioTemporalGNN, build_adjacency_matrix_from_graph, resolve_sensor_locations
from src.detection.preprocessing import DataPreprocessor
from src.network.ctown_loader import load_ctown_network
from src.network.graph_builder import build_network_graph

logger = logging.getLogger(__name__)


def train_model(
    config: Dict[str, Any],
    dataset_path: Optional[Union[str, Path]] = None,
    allow_synthetic: bool = False,
) -> Dict[str, Any]:
    """Train entry point for SpatioTemporalGNN forecasting anomaly detection model.

    Trains exclusively on normal operational baseline data (BATADAL_dataset03.csv),
    calibrates anomaly threshold on held-out normal calibration split, and evaluates
    detection performance on labeled operational data (BATADAL_dataset04.csv).

    Args:
        config: Configuration dictionary (typically loaded from config.yaml).
        dataset_path: Optional path to single dataset CSV file. If None, uses paths from config.
        allow_synthetic: Explicit override flag allowing synthetic data for testing ONLY.

    Returns:
        Dict containing trained model, history, validation metrics, threshold, and status.

    Raises:
        RuntimeError: If a real dataset is not provided and allow_synthetic is False.
        ValueError: If specified dataset files lack required columns or rows.
    """
    # Reproducibility seeds
    seed = config.get("random_seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    detection_cfg = config.get("detection", {})
    allow_synth_cfg = detection_cfg.get("allow_synthetic_training", False) or allow_synthetic

    train_p = dataset_path or detection_cfg.get("train_path") or detection_cfg.get("data_path")
    val_p = detection_cfg.get("val_path")

    # Check multi-file vs single-file real data availability
    is_train_real = verify_dataset_is_real(train_p)
    is_val_real = verify_dataset_is_real(val_p) if val_p else False
    is_real = is_train_real or is_val_real

    # STRICT DATA GATING CHECK
    if not is_real and not allow_synth_cfg:
        err_msg = (
            f"MODEL TRAINING REFUSED: No real BATADAL dataset found at path '{train_p}'. "
            f"Per research integrity guidelines, training cannot run on unverified or "
            f"synthetic data to produce research results. Supply valid BATADAL dataset CSV files "
            f"before running model training."
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    if not is_real and allow_synth_cfg:
        logger.warning(
            "ALLOW_SYNTHETIC FLAG IS SET: Running training pipeline against non-real/synthetic data "
            "FOR SOFTWARE PIPELINE VERIFICATION ONLY. DO NOT REPORT THESE RESULTS AS RESEARCH PERFORMANCE!"
        )

    loader = BATADALDatasetLoader(
        timestamp_col=detection_cfg.get("timestamp_col", "DATETIME"),
        label_col=detection_cfg.get("label_col", "ATT_FLAG"),
    )

    window_size = detection_cfg.get("window_size", 12)
    preprocessor = DataPreprocessor(
        scaler_type=detection_cfg.get("scaler_type", "standard"),
        window_size=window_size,
        stride=detection_cfg.get("stride", 1),
    )

    # Multi-file support (separate train file + val file) vs single file
    if is_train_real and is_val_real:
        logger.info("Multi-file dataset detected. Train: %s | Val: %s", train_p, val_p)
        train_df_raw, train_sensor_cols, train_lbl = loader.load_csv(train_p)
        val_df_raw, val_sensor_cols, val_lbl = loader.load_csv(val_p)

        sensor_cols = [c for c in train_sensor_cols if c in val_sensor_cols]

        full_train_df = preprocessor.handle_missing_values(train_df_raw, sensor_cols)
        val_df = preprocessor.handle_missing_values(val_df_raw, sensor_cols)

        val_label_col = val_lbl or "ATT_FLAG"
        if val_label_col not in val_df.columns:
            val_df[val_label_col] = 0

        # Chronological 80% train / 20% calibration split of normal training data
        n_tr = len(full_train_df)
        split_idx = int(0.8 * n_tr)
        train_df = full_train_df.iloc[:split_idx].copy()
        calib_df = full_train_df.iloc[split_idx:].copy()

    else:
        logger.info("Single-file dataset mode for path: %s", train_p)
        df_raw, sensor_cols, label_col = loader.load_csv(train_p)

        df_clean = preprocessor.handle_missing_values(df_raw, sensor_cols)
        train_df, calib_df, val_df = preprocessor.temporal_split(
            df_clean,
            train_ratio=detection_cfg.get("train_ratio", 0.7),
            val_ratio=detection_cfg.get("val_ratio", 0.15),
        )
        val_label_col = label_col or "ATT_FLAG"
        if val_label_col not in val_df.columns:
            val_df[val_label_col] = 0

    logger.info(
        "Normal Data Partitions: Train=%d, Calibration=%d | Validation=%d",
        len(train_df),
        len(calib_df),
        len(val_df),
    )

    # FIT SCALER STRICTLY ON THE 80% NORMAL TRAINING PARTITION
    preprocessor.fit_scaler(train_df, sensor_cols)

    train_X_arr = preprocessor.transform_features(train_df, sensor_cols)
    calib_X_arr = preprocessor.transform_features(calib_df, sensor_cols)
    val_X_arr = preprocessor.transform_features(val_df, sensor_cols)

    val_y_arr = val_df[val_label_col].values

    requested_horizon = int(detection_cfg.get("forecast_horizon", 12))
    available_horizon = max(1, min(len(train_df), len(calib_df), len(val_df)) - window_size)
    forecast_horizon = min(requested_horizon, available_horizon)

    if forecast_horizon != requested_horizon:
        logger.warning(
            "Forecast horizon reduced from H=%d to H=%d because the available sequence length is too short for the configured production horizon.",
            requested_horizon,
            forecast_horizon,
        )

    # Construct forecasting sliding windows: X (B, W, F), Y_target (B, H, F)
    X_train, Y_train, _ = preprocessor.create_forecasting_windows(
        train_X_arr,
        forecast_horizon=forecast_horizon,
    )
    X_calib, Y_calib, _ = preprocessor.create_forecasting_windows(
        calib_X_arr,
        forecast_horizon=forecast_horizon,
    )
    X_val, Y_val, val_y_lbls = preprocessor.create_forecasting_windows(
        val_X_arr,
        val_y_arr,
        forecast_horizon=forecast_horizon,
    )

    # Load C-Town NetworkX graph topology and construct 43x43 sensor adjacency matrix
    inp_file_path = config.get("paths", {}).get("inp_file", "data/raw/CTOWN.INP")
    adj_matrix = None
    sensor_map_info = {}
    if Path(inp_file_path).exists():
        try:
            net = load_ctown_network(inp_file_path)
            g = build_network_graph(net)
            adj_matrix = build_adjacency_matrix_from_graph(g, sensor_cols, max_hops=3)
            sensor_nodes = resolve_sensor_locations(g, sensor_cols)
            sensor_map_info = {s: sorted(list(nodes)) for s, nodes in sensor_nodes.items()}
            logger.info(
                "Built GNN adjacency matrix from C-Town graph topology: shape %s, non-zero=%d",
                adj_matrix.shape,
                int(torch.count_nonzero(adj_matrix)),
            )
        except Exception as e:
            logger.error("Failed to build C-Town graph adjacency matrix: %s", str(e))
            raise e

    # PyTorch DataLoaders for forecasting
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    calib_dataset = TensorDataset(torch.from_numpy(X_calib), torch.from_numpy(Y_calib))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))

    batch_size = detection_cfg.get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model instantiation
    model = SpatioTemporalGNN(
        in_features=len(sensor_cols),
        window_size=window_size,
        hidden_dim=detection_cfg.get("hidden_dim", 64),
        dropout=detection_cfg.get("dropout", 0.2),
        forecast_horizon=forecast_horizon,
    )

    optimizer = optim.Adam(model.parameters(), lr=detection_cfg.get("learning_rate", 0.001))
    criterion = nn.MSELoss()

    epochs = detection_cfg.get("epochs", 20)
    history = {"train_loss": [], "val_loss": []}

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x, adj_matrix=adj_matrix)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_loss / len(train_dataset)
        history["train_loss"].append(epoch_train_loss)

        # Calibration loss check
        model.eval()
        calib_running_loss = 0.0
        with torch.no_grad():
            for c_x, c_y in calib_loader:
                c_preds = model(c_x, adj_matrix=adj_matrix)
                c_loss = criterion(c_preds, c_y)
                calib_running_loss += c_loss.item() * c_x.size(0)

        epoch_calib_loss = calib_running_loss / len(calib_dataset)
        history["val_loss"].append(epoch_calib_loss)
        model.train()

        logger.info(
            "Epoch %d/%d - Train MSE Loss: %.6f, Calib MSE Loss: %.6f",
            epoch,
            epochs,
            epoch_train_loss,
            epoch_calib_loss,
        )

    # THRESHOLD CALIBRATION ON HELD-OUT NORMAL CALIBRATION SPLIT ONLY
    model.eval()
    all_calib_scores = []
    with torch.no_grad():
        for c_x, c_y in calib_loader:
            c_preds = model(c_x, adj_matrix=adj_matrix)
            # Aggregate the forecast error across all horizon steps and sensor features.
            sample_mse = torch.mean((c_preds - c_y) ** 2, dim=(1, 2)).cpu().numpy()
            all_calib_scores.extend(sample_mse)

    calib_scores_arr = np.array(all_calib_scores)
    percentile_val = detection_cfg.get("threshold_percentile", 99.5)
    calibrated_threshold = float(np.percentile(calib_scores_arr, percentile_val))
    logger.info(
        "Calibrated Anomaly Threshold (%.1fth percentile on normal calibration data): %.6f",
        percentile_val,
        calibrated_threshold,
    )

    # FINAL VALIDATION EVALUATION ON BATADAL_dataset04.csv
    all_val_scores = []
    with torch.no_grad():
        for v_x, v_y in val_loader:
            v_preds = model(v_x, adj_matrix=adj_matrix)
            sample_mse = torch.mean((v_preds - v_y) ** 2, dim=(1, 2)).cpu().numpy()
            all_val_scores.extend(sample_mse)

    val_scores_arr = np.array(all_val_scores)

    evaluator = DetectionEvaluator(threshold=calibrated_threshold)
    val_metrics = evaluator.evaluate_predictions(val_y_lbls, val_scores_arr, threshold=calibrated_threshold)

    # Save model checkpoint
    models_dir = Path(config.get("paths", {}).get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / "detection_gnn.pt"
    torch.save(model.state_dict(), save_path)
    logger.info("Saved trained model state to %s", save_path)

    # Save metrics and artifacts to results directory
    results_dir = Path(config.get("paths", {}).get("results_dir", "results"))
    processed_dir = Path(config.get("paths", {}).get("processed_dir", "data/processed"))
    results_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "detection_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(results_dir / "calibrated_threshold.json", "w") as f:
        json.dump(
            {
                "threshold": calibrated_threshold,
                "percentile": percentile_val,
                "num_calibration_samples": len(calib_scores_arr),
                "calibration_score_mean": float(np.mean(calib_scores_arr)),
                "calibration_score_std": float(np.std(calib_scores_arr)),
                "calibration_score_max": float(np.max(calib_scores_arr)),
            },
            f,
            indent=2,
        )

    topology_export = {
        "num_sensors": len(sensor_cols),
        "sensor_columns": sensor_cols,
        "sensor_to_ctown_nodes": sensor_map_info,
        "adjacency_nonzero_count": int(torch.count_nonzero(adj_matrix)) if adj_matrix is not None else 0,
    }
    with open(results_dir / "sensor_topology_info.json", "w") as f:
        json.dump(topology_export, f, indent=2)
    with open(processed_dir / "sensor_topology_info.json", "w") as f:
        json.dump(topology_export, f, indent=2)

    logger.info("Saved validation metrics, calibrated threshold, and topology info to %s", results_dir)

    return {
        "model": model,
        "history": history,
        "validation_metrics": val_metrics,
        "calibrated_threshold": calibrated_threshold,
        "status": "COMPLETED",
        "checkpoint_path": str(save_path),
        "is_real_dataset": is_real,
        "num_sensors": len(sensor_cols),
        "num_train_windows": len(X_train),
        "num_calib_windows": len(X_calib),
        "num_val_windows": len(X_val),
        "adjacency_nonzero_count": topology_export["adjacency_nonzero_count"],
    }

