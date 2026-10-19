"""
audit_test_inference.py

Systematic Audit Script for Phase 5 Held-Out Test Inference.
Inspects raw test data, timestamps, extreme scores, sensor values, and pipeline consistency.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

import sys
sys.path.insert(0, ".")

from src.detection.dataset_loader import BATADALDatasetLoader
from src.detection.graph_model import SpatioTemporalGNN, build_adjacency_matrix_from_graph
from src.detection.preprocessing import DataPreprocessor
from src.detection.scoring import calibrated_sensor_variance
from src.network.ctown_loader import load_ctown_network
from src.network.graph_builder import build_network_graph


def main():
    print("=====================================================================")
    print("           AUDIT STEP 1: TEST DATA TIMELINE & PARSING AUDIT           ")
    print("=====================================================================")

    raw_path = Path("data/raw/BATADAL_test_dataset.csv")
    df_raw = pd.read_csv(raw_path)
    df_raw.columns = df_raw.columns.astype(str).str.strip()

    num_raw_rows = len(df_raw)
    print(f"1. Raw CSV Row Count: {num_raw_rows}")

    # Inspect raw DATETIME column strings
    raw_timestamps = df_raw["DATETIME"].astype(str).values
    print(f"   First 5 raw strings: {raw_timestamps[:5]}")
    print(f"   Last 5 raw strings : {raw_timestamps[-5:]}")

    # Compare default pd.to_datetime vs dayfirst=True vs explicit format
    parsed_default = pd.to_datetime(df_raw["DATETIME"])
    parsed_dayfirst = pd.to_datetime(df_raw["DATETIME"], dayfirst=True)
    
    # Check monotonicity / ordering
    is_mono_default = parsed_default.is_monotonic_increasing
    is_mono_dayfirst = parsed_dayfirst.is_monotonic_increasing

    print(f"   Default pd.to_datetime() Monotonic Increasing: {is_mono_default}")
    print(f"   pd.to_datetime(dayfirst=True) Monotonic Increasing: {is_mono_dayfirst}")

    # Find out-of-order jumps in default parsing
    diffs_default = parsed_default.diff()
    negative_jumps = np.where(diffs_default < pd.Timedelta(0))[0]
    print(f"   Number of backwards timestamp jumps in default parsing: {len(negative_jumps)}")
    if len(negative_jumps) > 0:
        for idx in negative_jumps[:5]:
            print(f"     Row {idx}: Raw='{raw_timestamps[idx]}', Parsed Default='{parsed_default.iloc[idx]}', Prev Parsed='{parsed_default.iloc[idx-1]}'")

    # Check hourly cadence with dayfirst=True
    time_diffs_df = parsed_dayfirst.diff()
    non_1h = time_diffs_df[time_diffs_df != pd.Timedelta(hours=1)][1:]
    print(f"   Non-1-hour gaps with dayfirst=True: {len(non_1h)}")
    if len(non_1h) > 0:
        print(f"     Gaps sample: {non_1h.head(5)}")

    print(f"   First timestamp (dayfirst=True): {parsed_dayfirst.iloc[0]}")
    print(f"   Last timestamp (dayfirst=True) : {parsed_dayfirst.iloc[-1]}")

    print("\n=====================================================================")
    print("           AUDIT STEP 2: EXTREME ANOMALY SCORE INVESTIGATION         ")
    print("=====================================================================")

    # Load Dataset03 and fit scaler as in training
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    loader = BATADALDatasetLoader()
    df03, s03, _ = loader.load_csv("data/raw/BATADAL_dataset03.csv")
    df_test, s_test, _ = loader.load_csv("data/raw/BATADAL_test_dataset.csv")
    sensor_cols = [c for c in s03 if c in s_test]

    preprocessor = DataPreprocessor(window_size=12)
    n_tr = len(df03)
    split_idx = int(0.8 * n_tr)

    train_df = preprocessor.handle_missing_values(df03.iloc[:split_idx], sensor_cols)
    calib_df = preprocessor.handle_missing_values(df03.iloc[split_idx:], sensor_cols)
    test_df = preprocessor.handle_missing_values(df_test, sensor_cols)

    preprocessor.fit_scaler(train_df, sensor_cols)

    # Inspect raw values of F_PU3, S_PU3, S_PU1, F_PU6, S_PU6, L_T3, L_T1 in raw test dataset
    print("\n   Checking raw values in test dataset for target sensors:")
    target_sensors = ["F_PU3", "S_PU3", "S_PU1", "F_PU6", "S_PU6", "L_T3", "L_T1"]
    for s_col in target_sensors:
        if s_col in df_raw.columns:
            vals = df_raw[s_col].values
            print(f"     {s_col:<8}: min={np.min(vals)}, max={np.max(vals)}, mean={np.mean(vals):.2f}, std={np.std(vals):.2f}, NaNs={np.isnan(vals).sum()}, -999s={(vals == -999).sum()}")

    # Inspect Scaler mean & scale for these sensors
    print("\n   Checking Scaler parameters fitted on Dataset03 Train (80%):")
    scaler = preprocessor.scaler
    for s_col in target_sensors:
        if s_col in sensor_cols:
            idx = sensor_cols.index(s_col)
            mean_val = scaler.mean_[idx]
            scale_val = scaler.scale_[idx]
            print(f"     {s_col:<8}: Scaler Mean = {mean_val:.4f}, Scaler Std (Scale) = {scale_val:.6f}")

    # Transform features
    calib_X = preprocessor.transform_features(calib_df, sensor_cols)
    test_X = preprocessor.transform_features(test_df, sensor_cols)

    H = 12
    X_cal, Y_cal, _ = preprocessor.create_forecasting_windows(calib_X, forecast_horizon=H)
    X_test, Y_test, _ = preprocessor.create_forecasting_windows(test_X, forecast_horizon=H)

    net = load_ctown_network(config["paths"]["inp_file"])
    g = build_network_graph(net)
    adj_matrix = build_adjacency_matrix_from_graph(g, sensor_cols, max_hops=3)

    model = SpatioTemporalGNN(in_features=len(sensor_cols), window_size=12, hidden_dim=64, forecast_horizon=H)
    model.load_state_dict(torch.load("models/detection_gnn_h12_candidate.pt", weights_only=True))
    model.eval()

    with torch.no_grad():
        c_preds = model(torch.from_numpy(X_cal), adj_matrix=adj_matrix).numpy()
        t_preds = model(torch.from_numpy(X_test), adj_matrix=adj_matrix).numpy()

    E_cal = (c_preds - Y_cal) ** 2
    E_test = (t_preds - Y_test) ** 2

    mu_sq_h = calibrated_sensor_variance(E_cal, sensor_cols, preprocessor.scaler)
    Z_cal = E_cal / mu_sq_h
    Z_test = E_test / mu_sq_h

    # Check mu_sq_h for target sensors at step H-1 (index 11)
    print("\n   Checking Normal Calibration Error Baseline mu_sq_h (step H=12):")
    for s_col in target_sensors:
        if s_col in sensor_cols:
            idx = sensor_cols.index(s_col)
            mu_val = mu_sq_h[11, idx]
            print(f"     {s_col:<8}: Normal Calib MSE mu_sq_h[11] = {mu_val:.8f}")

    # Locate peak timesteps in Z_test[:, 11, :]
    S_test_final = np.mean(Z_test[:, 11, :], axis=1)
    peak_win_idx = np.argmax(S_test_final)
    peak_score = S_test_final[peak_win_idx]
    peak_raw_idx = peak_win_idx + 12

    print(f"\n   PEAK ANOMALY WINDOW: Index {peak_win_idx} (Raw row {peak_raw_idx})")
    print(f"     Raw Timestamp (dayfirst=True): {parsed_dayfirst.iloc[peak_raw_idx]}")
    print(f"     Score: {peak_score:e}")

    # Top contributing sensors at peak
    peak_Z = Z_test[peak_win_idx, 11, :]
    top_sensors_peak_idx = np.argsort(peak_Z)[::-1][:5]
    print("     Top contributing sensors at peak:")
    for k in top_sensors_peak_idx:
        s_col = sensor_cols[k]
        s_z = peak_Z[k]
        raw_val = df_test.iloc[peak_raw_idx][s_col]
        pred_val_norm = t_preds[peak_win_idx, 11, k]
        target_val_norm = Y_test[peak_win_idx, 11, k]
        sq_err = E_test[peak_win_idx, 11, k]
        mu = mu_sq_h[11, k]
        print(f"       {s_col:<8}: Z={s_z:e} | Raw Val={raw_val} | Norm Target={target_val_norm:.4f} | Norm Pred={pred_val_norm:.4f} | SqErr={sq_err:.4f} | mu={mu:.8f}")

    print("\n=====================================================================")
    print("           AUDIT STEP 3: TRAINING/CALIBRATION PIPELINE CONSISTENCY   ")
    print("=====================================================================")

    # Verify feature ordering
    print(f"   Dataset03 Sensor Count: {len(s03)} | Test Sensor Count: {len(s_test)}")
    print(f"   Identical Sensor List: {s03 == s_test}")
    print(f"   Sensor Columns Count: {len(sensor_cols)}")

    print("\nAUDIT SCRIPT COMPLETED.")


if __name__ == "__main__":
    main()
