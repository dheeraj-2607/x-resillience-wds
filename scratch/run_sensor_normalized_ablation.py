"""
run_sensor_normalized_ablation.py

Performs an offline diagnostic ablation study evaluating sensor-specific normal residual
normalization to determine whether baseline variance differences across sensors hide subtle
localized attacks.

Methods evaluated:
1. Baseline: Global Unnormalized Mean MSE
2. Sensor-Normalized Global Mean: mean of (E_i / mean_calib_E_i) across all 43 sensors
3. Sensor-Normalized Top-5 Mean: mean of 5 largest (E_i / mean_calib_E_i)
4. Sensor Z-Score Top-5 Mean: mean of 5 largest ((E_i - mean_calib_E_i) / std_calib_E_i)

Thresholds are calibrated ONLY on 20% normal calibration split of BATADAL_dataset03.csv
using 99.5th percentile scores.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, ".")

from src.detection.dataset_loader import BATADALDatasetLoader
from src.detection.evaluate import DetectionEvaluator
from src.detection.graph_model import SpatioTemporalGNN, build_adjacency_matrix_from_graph
from src.detection.preprocessing import DataPreprocessor
from src.detection.scoring import calibrated_sensor_variance, sensor_variance_floor
from src.network.ctown_loader import load_ctown_network
from src.network.graph_builder import build_network_graph


def main():
    config_path = Path("configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    loader = BATADALDatasetLoader()
    df03_raw, s03, _ = loader.load_csv("data/raw/BATADAL_dataset03.csv")
    df04_raw, s04, _ = loader.load_csv("data/raw/BATADAL_dataset04.csv")

    sensor_cols = [c for c in s03 if c in s04]

    preprocessor = DataPreprocessor(window_size=12)
    n_tr = len(df03_raw)
    split_idx = int(0.8 * n_tr)
    train_df = preprocessor.handle_missing_values(df03_raw.iloc[:split_idx], sensor_cols)
    calib_df = preprocessor.handle_missing_values(df03_raw.iloc[split_idx:], sensor_cols)
    val_df = preprocessor.handle_missing_values(df04_raw, sensor_cols)

    # Fit scaler ONLY on 80% train
    preprocessor.fit_scaler(train_df, sensor_cols)

    calib_X_arr = preprocessor.transform_features(calib_df, sensor_cols)
    val_X_arr = preprocessor.transform_features(val_df, sensor_cols)
    val_y_arr = val_df["ATT_FLAG"].values

    X_calib, Y_calib, _ = preprocessor.create_forecasting_windows(calib_X_arr)
    X_val, Y_val, val_y_lbls = preprocessor.create_forecasting_windows(val_X_arr, val_y_arr)

    net = load_ctown_network(config["paths"]["inp_file"])
    g = build_network_graph(net)
    adj_matrix = build_adjacency_matrix_from_graph(g, sensor_cols, max_hops=3)

    model = SpatioTemporalGNN(in_features=len(sensor_cols), window_size=12, hidden_dim=64)
    model.load_state_dict(torch.load("models/detection_gnn.pt"))
    model.eval()

    with torch.no_grad():
        c_preds = model(torch.from_numpy(X_calib), adj_matrix=adj_matrix).numpy()
        v_preds = model(torch.from_numpy(X_val), adj_matrix=adj_matrix).numpy()

    # Per-sensor squared residuals E: shape (N, 43)
    E_calib = (c_preds - Y_calib) ** 2
    E_val = (v_preds - Y_val) ** 2

    # ESTIMATE EACH SENSOR'S NORMAL RESIDUAL DISTRIBUTION ON CALIBRATION DATA ONLY
    sensor_normal_mean_mse = calibrated_sensor_variance(E_calib, sensor_cols, preprocessor.scaler)
    sensor_normal_std_mse = np.maximum(
        np.std(E_calib, axis=0), sensor_variance_floor(sensor_cols, preprocessor.scaler)
    )

    # SENSOR NORMALIZATION TENSORS
    # 1. Ratio to normal mean MSE: (N, 43)
    Z_ratio_calib = E_calib / sensor_normal_mean_mse
    Z_ratio_val = E_val / sensor_normal_mean_mse

    # 2. Z-Score of squared error: (N, 43)
    Z_std_calib = (E_calib - sensor_normal_mean_mse) / sensor_normal_std_mse
    Z_std_val = (E_val - sensor_normal_mean_mse) / sensor_normal_std_mse

    # Define Scoring Functions
    scoring_configs = {
        "Global Mean MSE (Baseline)": (
            np.mean(E_calib, axis=1),
            np.mean(E_val, axis=1),
        ),
        "Sensor-Normalized Mean": (
            np.mean(Z_ratio_calib, axis=1),
            np.mean(Z_ratio_val, axis=1),
        ),
        "Sensor-Normalized Top-5 Mean": (
            np.mean(np.sort(Z_ratio_calib, axis=1)[:, -5:], axis=1),
            np.mean(np.sort(Z_ratio_val, axis=1)[:, -5:], axis=1),
        ),
        "Sensor Z-Score Top-5 Mean": (
            np.mean(np.sort(Z_std_calib, axis=1)[:, -5:], axis=1),
            np.mean(np.sort(Z_std_val, axis=1)[:, -5:], axis=1),
        ),
    }

    # Extract 5 contiguous attack sequences in dataset04
    attack_indices = np.where(val_y_lbls == 1)[0]
    sequences = []
    curr_seq = [attack_indices[0]]
    for idx in attack_indices[1:]:
        if idx == curr_seq[-1] + 1:
            curr_seq.append(idx)
        else:
            sequences.append(curr_seq)
            curr_seq = [idx]
    sequences.append(curr_seq)

    results_report = {}

    print("=================================================================")
    print("   SENSOR-NORMALIZED ANOMALY SCORE ABLATION STUDY REPORT        ")
    print("=================================================================\n")

    for method_name, (s_calib, s_val) in scoring_configs.items():
        # Calibrate threshold at 99.5th percentile on normal calibration split ONLY
        threshold = float(np.percentile(s_calib, 99.5))

        evaluator = DetectionEvaluator(threshold=threshold)
        metrics = evaluator.evaluate_predictions(val_y_lbls, s_val, threshold=threshold)

        val_preds_binary = (s_val >= threshold).astype(int)
        seq_recalls = {}
        for i, seq in enumerate(sequences, 1):
            tp = np.sum(val_preds_binary[seq] == 1)
            rec = float(tp / len(seq))
            seq_recalls[f"Sequence_{i}"] = {
                "length_hours": len(seq),
                "tp": int(tp),
                "fn": int(len(seq) - tp),
                "recall": rec,
            }

        norm_val_scores = s_val[val_y_lbls == 0]
        att_val_scores = s_val[val_y_lbls == 1]

        results_report[method_name] = {
            "method": method_name,
            "threshold_99_5": threshold,
            "calibration_scores_stats": {
                "mean": float(np.mean(s_calib)),
                "std": float(np.std(s_calib)),
                "p99_5": threshold,
                "max": float(np.max(s_calib)),
            },
            "validation_metrics": metrics,
            "sequence_recalls": seq_recalls,
            "seq2_recall": seq_recalls["Sequence_2"]["recall"],
            "seq3_recall": seq_recalls["Sequence_3"]["recall"],
            "score_distribution": {
                "normal_mean": float(np.mean(norm_val_scores)),
                "normal_median": float(np.median(norm_val_scores)),
                "normal_p95": float(np.percentile(norm_val_scores, 95)),
                "attack_mean": float(np.mean(att_val_scores)),
                "attack_median": float(np.median(att_val_scores)),
                "attack_p95": float(np.percentile(att_val_scores, 95)),
            },
        }

    ranked_methods = sorted(
        scoring_configs.keys(),
        key=lambda m: results_report[m]["validation_metrics"]["f1_score"],
        reverse=True,
    )

    print("RANKED SENSOR-NORMALIZED RESULTS (BY F1-SCORE):")
    print("---------------------------------------------------------------------------------------------------------")
    print(f"{'Rank':<5} | {'Scoring Method':<28} | {'Threshold':<10} | {'Precision':<9} | {'Recall':<8} | {'F1':<8} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'FPR':<8}")
    print("---------------------------------------------------------------------------------------------------------")
    for rank, m in enumerate(ranked_methods, 1):
        met = results_report[m]["validation_metrics"]
        tau = results_report[m]["threshold_99_5"]
        print(
            f"{rank:<5} | {m:<28} | {tau:<10.4f} | {met['precision']:<9.4f} | {met['recall']:<8.4f} | {met['f1_score']:<8.4f} | {met['roc_auc']:<8.4f} | {met['pr_auc']:<8.4f} | {met['false_positive_rate']:<8.4f}"
        )
    print("---------------------------------------------------------------------------------------------------------\n")

    print("PER-ATTACK SEQUENCE RECALL COMPARISON:")
    print("---------------------------------------------------------------------------------------------------------")
    print(f"{'Scoring Method':<28} | {'Seq 1 (42h)':<11} | {'Seq 2 (60h)':<11} | {'Seq 3 (37h)':<11} | {'Seq 4 (7h)':<11} | {'Seq 5 (73h)':<11} | {'Overall Recall':<14}")
    print("---------------------------------------------------------------------------------------------------------")
    for m in scoring_configs.keys():
        sr = results_report[m]["sequence_recalls"]
        ov_rec = results_report[m]["validation_metrics"]["recall"]
        print(
            f"{m:<28} | {sr['Sequence_1']['recall']:<11.2%} | {sr['Sequence_2']['recall']:<11.2%} | {sr['Sequence_3']['recall']:<11.2%} | {sr['Sequence_4']['recall']:<11.2%} | {sr['Sequence_5']['recall']:<11.2%} | {ov_rec:<14.2%}"
        )
    print("---------------------------------------------------------------------------------------------------------\n")

    report_path = Path("results/sensor_normalized_ablation_report.json")
    with open(report_path, "w") as f:
        json.dump(results_report, f, indent=2)

    print(f"Saved complete sensor-normalized ablation report to: {report_path.resolve()}")
    print("=================================================================")


if __name__ == "__main__":
    main()
