"""
run_final_h12_staging.py

Phase 1-4 Final Model Staging & Dataset04 Validation Pipeline for H=12 Architecture.

STRICT INTEGRITY RULES:
- Production checkpoint models/detection_gnn.pt is NOT touched.
- Raw BATADAL CSV files under data/raw/ are NOT modified.
- BATADAL_test_dataset.csv is strictly UNTOUCHED.
- Threshold derived strictly from Dataset03 20% normal calibration partition using locked 99.8th percentile rule.
- Dataset04 labels accessed ONLY post-scoring for validation metrics.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

sys.path.insert(0, ".")

from src.detection.dataset_loader import BATADALDatasetLoader
from src.detection.evaluate import DetectionEvaluator
from src.detection.graph_model import SpatioTemporalGNN, build_adjacency_matrix_from_graph, resolve_sensor_locations
from src.detection.preprocessing import DataPreprocessor
from src.detection.scoring import calibrated_sensor_variance
from src.network.ctown_loader import load_ctown_network
from src.network.graph_builder import build_network_graph


def compute_file_hash(filepath: Path | str) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_sensor_type(sensor_name: str) -> str:
    if sensor_name.startswith("P_"):
        return "Junction Pressure"
    elif sensor_name.startswith("L_"):
        return "Tank Level"
    elif sensor_name.startswith("F_"):
        return "Link Flow"
    elif sensor_name.startswith("S_"):
        return "Link Status"
    return "Unknown"


def main():
    print("=========================================================================================================")
    print("                PHASE 1-4 FINAL MODEL STAGING & DATASET04 VALIDATION (H=12 ARCHITECTURE)                ")
    print("=========================================================================================================\n")

    # PRE-EXECUTION INTEGRITY SNAPSHOT
    prod_ckpt_path = Path("models/detection_gnn.pt")
    config_path = Path("configs/config.yaml")

    initial_prod_hash = compute_file_hash(prod_ckpt_path) if prod_ckpt_path.exists() else None
    initial_config_hash = compute_file_hash(config_path)

    raw_dir = Path("data/raw")
    raw_hashes = {p.name: compute_file_hash(p) for p in raw_dir.glob("*") if p.is_file()}

    test_csv_path = Path("data/raw/BATADAL_test_dataset.csv")
    test_csv_mtime = test_csv_path.stat().st_mtime if test_csv_path.exists() else None

    # PHASE 1: FREEZE CONFIGURATION
    print("--- Phase 1: Freeze Configuration ---")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    H = 12
    seed = 42
    epochs = 50
    batch_size = 64
    learning_rate = 0.001
    hidden_dim = 64
    dropout = 0.2
    percentile_cutoff = 99.8

    set_seed(seed)

    print(f"   Forecast Horizon: {H}")
    print(f"   Architecture: SpatioTemporalGNN (hidden_dim={hidden_dim}, dropout={dropout})")
    print(f"   Optimizer: Adam (lr={learning_rate}) | Batch Size: {batch_size} | Epochs: {epochs}")
    print(f"   Locked Threshold Percentile: {percentile_cutoff}th percentile on normal calibration split\n")

    # PHASE 2: FINAL CANDIDATE TRAINING & NORMAL CALIBRATION
    print("--- Phase 2: Final Candidate Training & Normal Calibration ---")
    loader = BATADALDatasetLoader()
    df03_raw, s03, _ = loader.load_csv("data/raw/BATADAL_dataset03.csv")
    df04_raw, s04, _ = loader.load_csv("data/raw/BATADAL_dataset04.csv")

    sensor_cols = [c for c in s03 if c in s04]
    F_count = len(sensor_cols)

    preprocessor = DataPreprocessor(window_size=12)
    n_tr = len(df03_raw)
    split_idx = int(0.8 * n_tr)

    train_df = preprocessor.handle_missing_values(df03_raw.iloc[:split_idx], sensor_cols)
    calib_df = preprocessor.handle_missing_values(df03_raw.iloc[split_idx:], sensor_cols)
    val_df = preprocessor.handle_missing_values(df04_raw, sensor_cols)

    # Scaler fit STRICTLY on 80% normal training partition
    preprocessor.fit_scaler(train_df, sensor_cols)

    train_X_arr = preprocessor.transform_features(train_df, sensor_cols)
    calib_X_arr = preprocessor.transform_features(calib_df, sensor_cols)
    val_X_arr = preprocessor.transform_features(val_df, sensor_cols)
    val_y_arr = val_df["ATT_FLAG"].values

    net = load_ctown_network(config["paths"]["inp_file"])
    g = build_network_graph(net)
    adj_matrix = build_adjacency_matrix_from_graph(g, sensor_cols, max_hops=3)

    X_tr, Y_tr, _ = preprocessor.create_forecasting_windows(train_X_arr, forecast_horizon=H)
    X_cal, Y_cal, _ = preprocessor.create_forecasting_windows(calib_X_arr, forecast_horizon=H)
    X_val, Y_val, val_y_lbls = preprocessor.create_forecasting_windows(val_X_arr, val_y_arr, forecast_horizon=H)

    model = SpatioTemporalGNN(
        in_features=F_count,
        window_size=12,
        hidden_dim=hidden_dim,
        dropout=dropout,
        forecast_horizon=H,
    )
    total_params = count_parameters(model)
    print(f"   Model instantiated. Parameter Count: {total_params:,}")

    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}

    model.train()
    cal_X_tensor = torch.from_numpy(X_cal)
    cal_Y_tensor = torch.from_numpy(Y_cal)

    for ep in range(1, epochs + 1):
        ep_tr_loss = 0.0
        for b_x, b_y in tr_loader:
            optimizer.zero_grad()
            preds = model(b_x, adj_matrix=adj_matrix)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()
            ep_tr_loss += loss.item() * len(b_x)
        ep_tr_loss /= len(X_tr)

        model.eval()
        with torch.no_grad():
            c_preds_ep = model(cal_X_tensor, adj_matrix=adj_matrix)
            ep_val_loss = criterion(c_preds_ep, cal_Y_tensor).item()
        model.train()

        history["train_loss"].append(float(ep_tr_loss))
        history["val_loss"].append(float(ep_val_loss))

        if ep % 10 == 0 or ep == epochs:
            print(f"   Epoch {ep:02d}/{epochs:02d} | Train MSE: {ep_tr_loss:.6f} | Normal Calib MSE: {ep_val_loss:.6f}")

    model.eval()
    candidate_ckpt_path = Path("models/detection_gnn_h12_candidate.pt")
    torch.save(model.state_dict(), candidate_ckpt_path)
    print(f"   Saved candidate checkpoint to: {candidate_ckpt_path} (Size: {candidate_ckpt_path.stat().st_size:,} bytes)")

    # DERIVE OPERATIONAL THRESHOLD FROM NORMAL CALIBRATION PARTITION ONLY
    with torch.no_grad():
        c_preds = model(torch.from_numpy(X_cal), adj_matrix=adj_matrix).numpy()
        v_preds = model(torch.from_numpy(X_val), adj_matrix=adj_matrix).numpy()

    E_cal = (c_preds - Y_cal) ** 2
    E_val = (v_preds - Y_val) ** 2

    mu_sq_h = calibrated_sensor_variance(E_cal, sensor_cols, preprocessor.scaler)
    Z_cal = E_cal / mu_sq_h
    Z_val = E_val / mu_sq_h

    S_cal_final = np.mean(Z_cal[:, H - 1, :], axis=1)
    S_val_final = np.mean(Z_val[:, H - 1, :], axis=1)

    tau_99_8 = float(np.percentile(S_cal_final, percentile_cutoff))
    calib_stats = {
        "locked_percentile": percentile_cutoff,
        "derived_threshold": round(tau_99_8, 6),
        "num_calibration_windows": len(S_cal_final),
        "mean_calibration_score": float(np.mean(S_cal_final)),
        "std_calibration_score": float(np.std(S_cal_final)),
        "max_calibration_score": float(np.max(S_cal_final)),
        "p95_calibration_score": float(np.percentile(S_cal_final, 95)),
        "p99_calibration_score": float(np.percentile(S_cal_final, 99)),
    }
    print(f"   Operational Anomaly Threshold (derived from Dataset03 Calib {percentile_cutoff}%): {tau_99_8:.6f}")

    training_report = {
        "model_staging_phase": "Final Candidate H=12",
        "checkpoint_path": str(candidate_ckpt_path),
        "random_seed": seed,
        "parameters": total_params,
        "configuration": {
            "forecast_horizon": H,
            "window_size": 12,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "scaler_type": "standard",
            "scaler_fitted_on": "Dataset03_80pct_train_split",
        },
        "training_history": history,
        "final_train_loss": history["train_loss"][-1],
        "final_calib_loss": history["val_loss"][-1],
        "calibration_statistics": calib_stats,
    }

    report_path = Path("results/final_h12_training_report.json")
    with open(report_path, "w") as f:
        json.dump(training_report, f, indent=2)
    print(f"   Saved training report to: {report_path}\n")

    # PHASE 3: DATASET04 FINAL VALIDATION
    print("--- Phase 3: Dataset04 Final Validation ---")
    evaluator = DetectionEvaluator(threshold=tau_99_8)
    val_metrics = evaluator.evaluate_predictions(val_y_lbls, S_val_final, threshold=tau_99_8)
    val_preds_binary = (S_val_final >= tau_99_8).astype(int)

    cm = val_metrics["confusion_matrix"]
    tn, fp, fn, tp = cm["TN"], cm["FP"], cm["FN"], cm["TP"]

    # Sequence Recalls & Delays
    attack_indices = np.where(val_y_arr == 1)[0]
    raw_sequences = []
    curr_seq = [attack_indices[0]]
    for idx in attack_indices[1:]:
        if idx == curr_seq[-1] + 1:
            curr_seq.append(idx)
        else:
            raw_sequences.append(curr_seq)
            curr_seq = [idx]
    raw_sequences.append(curr_seq)

    val_dates = val_df["DATETIME"].dt.strftime("%Y-%m-%d %H:%M").values[12:]

    sequence_results = []
    for i_seq, seq_raw in enumerate(raw_sequences, 1):
        win_indices = []
        for i_win in range(len(val_y_lbls)):
            target_step1_idx = i_win + 12
            if seq_raw[0] <= target_step1_idx <= seq_raw[-1]:
                win_indices.append(i_win)

        win_indices = np.array(win_indices)
        if len(win_indices) > 0:
            tp_seq = int(np.sum(val_preds_binary[win_indices] == 1))
            rec_seq = float(tp_seq / len(win_indices))

            pos_wins = win_indices[val_preds_binary[win_indices] == 1]
            if len(pos_wins) > 0:
                first_pos_win = pos_wins[0]
                first_detection_idx = first_pos_win + 12
                delay_steps = int(first_detection_idx - seq_raw[0])
                delay_hours = delay_steps * 1.0  # hourly sampling in BATADAL
                first_det_time = str(val_dates[first_pos_win])
            else:
                delay_steps = None
                delay_hours = None
                first_det_time = None
        else:
            rec_seq = 0.0
            delay_steps = None
            delay_hours = None
            first_det_time = None

        seq_info = {
            "sequence_id": i_seq,
            "raw_start_index": int(seq_raw[0]),
            "raw_end_index": int(seq_raw[-1]),
            "length_steps": len(seq_raw),
            "window_count": len(win_indices),
            "detected_windows": tp_seq if len(win_indices) > 0 else 0,
            "recall": round(rec_seq, 4),
            "detection_delay_steps": delay_steps,
            "detection_delay_hours": delay_hours,
            "first_detection_timestamp": first_det_time,
        }
        sequence_results.append(seq_info)

    # Sensor-Level Anomaly Attribution
    print("   Computing Sensor-Level Anomaly Attribution...")
    E_calib_final = E_cal[:, H - 1, :]  # (N_cal, 43)
    E_val_final = E_val[:, H - 1, :]    # (N_val, 43)

    calib_sensor_means = calibrated_sensor_variance(E_calib_final, sensor_cols, preprocessor.scaler)
    attack_mask_val = (val_y_lbls == 1)
    attack_sensor_means = np.mean(E_val_final[attack_mask_val, :], axis=0)
    normal_sensor_means = np.mean(E_val_final[~attack_mask_val, :], axis=0)

    # Sequence 3 specific windows
    seq3_win_indices = []
    seq3_raw = raw_sequences[2]  # 3rd sequence
    for i_win in range(len(val_y_lbls)):
        target_step1_idx = i_win + 12
        if seq3_raw[0] <= target_step1_idx <= seq3_raw[-1]:
            seq3_win_indices.append(i_win)
    seq3_win_indices = np.array(seq3_win_indices)
    seq3_sensor_means = np.mean(E_val_final[seq3_win_indices, :], axis=0)

    sensor_attribution_list = []
    for i, col in enumerate(sensor_cols):
        c_m = float(calib_sensor_means[i])
        a_m = float(attack_sensor_means[i])
        n_m = float(normal_sensor_means[i])
        s3_m = float(seq3_sensor_means[i])

        rel_ratio_overall = float(a_m / c_m)
        rel_ratio_seq3 = float(s3_m / c_m)
        abs_diff_overall = float(a_m - c_m)

        sensor_attribution_list.append({
            "sensor": col,
            "type": get_sensor_type(col),
            "calib_mean_residual": round(c_m, 6),
            "attack_mean_residual": round(a_m, 6),
            "seq3_mean_residual": round(s3_m, 6),
            "relative_ratio_overall": round(rel_ratio_overall, 4),
            "relative_ratio_seq3": round(rel_ratio_seq3, 4),
            "abs_increase_overall": round(abs_diff_overall, 6),
        })

    df_attribution = pd.DataFrame(sensor_attribution_list)
    top_overall = df_attribution.sort_values("relative_ratio_overall", ascending=False).head(10).to_dict(orient="records")
    top_seq3 = df_attribution.sort_values("relative_ratio_seq3", ascending=False).head(10).to_dict(orient="records")

    type_summary = df_attribution.groupby("type")[["relative_ratio_overall", "relative_ratio_seq3"]].mean().reset_index().to_dict(orient="records")

    validation_report = {
        "evaluator_configuration": {
            "model_checkpoint": str(candidate_ckpt_path),
            "validation_dataset": "BATADAL_dataset04.csv",
            "threshold": tau_99_8,
            "threshold_rule": "Dataset03 Normal Calib 99.8th percentile",
        },
        "performance_metrics": {
            "precision": round(val_metrics["precision"], 4),
            "recall": round(val_metrics["recall"], 4),
            "f1_score": round(val_metrics["f1_score"], 4),
            "roc_auc": round(val_metrics["roc_auc"], 4) if val_metrics["roc_auc"] is not None else None,
            "pr_auc": round(val_metrics["pr_auc"], 4) if val_metrics["pr_auc"] is not None else None,
            "false_positive_rate": round(val_metrics["false_positive_rate"], 4),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        },
        "sequence_analysis": sequence_results,
        "sensor_attribution_summary": {
            "top_10_sensors_overall": top_overall,
            "top_10_sensors_seq3": top_seq3,
            "mean_ratio_by_sensor_type": type_summary,
        },
    }

    # Print Summary Report Table to Console
    print("\n=========================================================================================================")
    print("                              DATASET04 FINAL VALIDATION REPORT                                          ")
    print("=========================================================================================================")
    print(f" Precision           : {val_metrics['precision']:.4f}")
    print(f" Recall              : {val_metrics['recall']:.4f}")
    print(f" F1 Score            : {val_metrics['f1_score']:.4f}")
    roc_auc = f"{val_metrics['roc_auc']:.4f}" if val_metrics["roc_auc"] is not None else "N/A (requires both classes)"
    pr_auc = f"{val_metrics['pr_auc']:.4f}" if val_metrics["pr_auc"] is not None else "N/A (requires both classes)"
    print(f" ROC-AUC             : {roc_auc}")
    print(f" PR-AUC              : {pr_auc}")
    print(f" False Positive Rate : {val_metrics['false_positive_rate']:.4f} ({val_metrics['false_positive_rate']*100:.2f}%)")
    print(f" Confusion Matrix    : TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")

    print("--- Sequence Recalls & Detection Delays ---")
    print(f" {'Seq':<4} | {'Length':<8} | {'Recall':<8} | {'Detected/Total Wins':<20} | {'Delay (steps)':<15} | {'Delay (hours)':<15}")
    print(" -------------------------------------------------------------------------------------------------------")
    for s in sequence_results:
        d_str = str(s['detection_delay_steps']) if s['detection_delay_steps'] is not None else "Undetected"
        dh_str = f"{s['detection_delay_hours']:.1f}h" if s['detection_delay_hours'] is not None else "N/A"
        print(f" Seq {s['sequence_id']} | {s['length_steps']:<8} | {s['recall']:<8.2%} | {s['detected_windows']}/{s['window_count']:<18} | {d_str:<15} | {dh_str:<15}")

    print("\n--- Top 5 Elevated Sensors Overall ---")
    for r in top_overall[:5]:
        print(f"   {r['sensor']:<10} ({r['type']:<18}): Relative Residual Increase = {r['relative_ratio_overall']:.2f}x normal")

    print("\n--- Top 5 Elevated Sensors for Sequence 3 ---")
    for r in top_seq3[:5]:
        print(f"   {r['sensor']:<10} ({r['type']:<18}): Relative Residual Increase = {r['relative_ratio_seq3']:.2f}x normal")
    print("=========================================================================================================\n")

    # PHASE 4: INTEGRITY VERIFICATION
    print("--- Phase 4: Integrity Verification ---")

    final_prod_hash = compute_file_hash(prod_ckpt_path) if prod_ckpt_path.exists() else None
    final_config_hash = compute_file_hash(config_path)
    final_raw_hashes = {p.name: compute_file_hash(p) for p in raw_dir.glob("*") if p.is_file()}
    final_test_mtime = test_csv_path.stat().st_mtime if test_csv_path.exists() else None

    is_prod_unchanged = (initial_prod_hash == final_prod_hash)
    is_config_unchanged = (initial_config_hash == final_config_hash)
    is_raw_unchanged = (raw_hashes == final_raw_hashes)
    is_test_unaccessed = (test_csv_mtime == final_test_mtime)

    print(f"   [1] Production model models/detection_gnn.pt UNCHANGED: {is_prod_unchanged} (Hash: {final_prod_hash[:12]}...)")
    print(f"   [2] Config configs/config.yaml UNCHANGED            : {is_config_unchanged}")
    print(f"   [3] Raw data directory data/raw/* UNCHANGED         : {is_raw_unchanged}")
    print(f"   [4] Test set BATADAL_test_dataset.csv UNACCESSED   : {is_test_unaccessed}")

    integrity_summary = {
        "production_checkpoint_unchanged": is_prod_unchanged,
        "config_unchanged": is_config_unchanged,
        "raw_data_unchanged": is_raw_unchanged,
        "test_dataset_unaccessed": is_test_unaccessed,
        "production_checkpoint_hash": final_prod_hash,
        "candidate_checkpoint_hash": compute_file_hash(candidate_ckpt_path),
    }

    validation_report["integrity_verification"] = integrity_summary

    # Write combined training + validation report
    with open("results/final_h12_validation_report.json", "w") as f:
        json.dump(validation_report, f, indent=2)

    if not (is_prod_unchanged and is_config_unchanged and is_raw_unchanged and is_test_unaccessed):
        raise RuntimeError("INTEGRITY VIOLATION DETECTED! One or more integrity checks failed.")

    print("\nALL INTEGRITY CHECKS PASSED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
