"""
run_h12_threshold_ablation.py

Controlled Threshold-Calibration Experiment for H=12 Architecture across 5 Seeds.

Evaluates percentile thresholds (99.5%, 99.7%, 99.8%, 99.9%) derived ONLY
from Dataset03 20% normal calibration partition on BATADAL_dataset04 validation set.

STRICT INTEGRITY RULES:
- Production checkpoint models/detection_gnn.pt is NOT touched.
- Raw BATADAL CSV files are NOT modified.
- BATADAL_test_dataset.csv is strictly UNTOUCHED.
- Models are NOT retrained (loads existing models/experiments/robustness/gnn_h12_seed*.pt).
- Thresholds calculated strictly from Dataset03 normal calibration set.
- Dataset04 labels accessed ONLY post-scoring for validation metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from src.detection.dataset_loader import BATADALDatasetLoader
from src.detection.evaluate import DetectionEvaluator
from src.detection.graph_model import SpatioTemporalGNN, build_adjacency_matrix_from_graph
from src.detection.preprocessing import DataPreprocessor
from src.detection.scoring import calibrated_sensor_variance
from src.network.ctown_loader import load_ctown_network
from src.network.graph_builder import build_network_graph


def main():
    config_path = Path("configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    Path("results").mkdir(parents=True, exist_ok=True)

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

    # Fit scaler strictly on 80% train split
    preprocessor.fit_scaler(train_df, sensor_cols)

    calib_X_arr = preprocessor.transform_features(calib_df, sensor_cols)
    val_X_arr = preprocessor.transform_features(val_df, sensor_cols)
    val_y_arr = val_df["ATT_FLAG"].values

    net = load_ctown_network(config["paths"]["inp_file"])
    g = build_network_graph(net)
    adj_matrix = build_adjacency_matrix_from_graph(g, sensor_cols, max_hops=3)

    seeds = [42, 123, 2024, 3407, 9999]
    percentiles = [99.5, 99.7, 99.8, 99.9]
    H = 12

    # Extract 5 contiguous attack sequences in dataset04
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

    # Create forecasting windows for H=12
    X_cal, Y_cal, _ = preprocessor.create_forecasting_windows(calib_X_arr, forecast_horizon=H)
    X_val, Y_val, val_y_lbls = preprocessor.create_forecasting_windows(val_X_arr, val_y_arr, forecast_horizon=H)

    all_seed_results = []

    print("=========================================================================================================")
    print("             H=12 CONTROLLED THRESHOLD ABLATION EXPERIMENT (5 SEEDS x 4 THRESHOLDS)                    ")
    print("=========================================================================================================\n")

    for s in seeds:
        ckpt_path = f"models/experiments/robustness/gnn_h12_seed{s}.pt"
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found!")

        model = SpatioTemporalGNN(
            in_features=F_count,
            window_size=12,
            hidden_dim=64,
            forecast_horizon=H,
        )
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.eval()

        with torch.no_grad():
            c_preds = model(torch.from_numpy(X_cal), adj_matrix=adj_matrix).numpy()
            v_preds = model(torch.from_numpy(X_val), adj_matrix=adj_matrix).numpy()

        E_cal = (c_preds - Y_cal) ** 2
        E_val = (v_preds - Y_val) ** 2

        # Per-horizon step, per-sensor normal calibration mean squared error
        mu_sq_h = calibrated_sensor_variance(E_cal, sensor_cols, preprocessor.scaler)
        Z_cal = E_cal / mu_sq_h
        Z_val = E_val / mu_sq_h

        # Final Horizon Score: step H (index H-1)
        S_cal_final = np.mean(Z_cal[:, H - 1, :], axis=1)
        S_val_final = np.mean(Z_val[:, H - 1, :], axis=1)

        for p in percentiles:
            tau = float(np.percentile(S_cal_final, p))
            expected_fp_pct = round(100.0 - p, 2)
            expected_fp_fraction = round((100.0 - p) / 100.0, 4)

            evaluator = DetectionEvaluator(threshold=tau)
            metrics = evaluator.evaluate_predictions(val_y_lbls, S_val_final, threshold=tau)
            val_preds_binary = (S_val_final >= tau).astype(int)

            cm = metrics["confusion_matrix"]

            # Sequence Recalls & Detection Delays
            seq_recalls = {}
            seq_delays = {}
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

                    # Detection delay: first window index in sequence where prediction == 1
                    pos_wins = win_indices[val_preds_binary[win_indices] == 1]
                    if len(pos_wins) > 0:
                        first_pos_win = pos_wins[0]
                        first_detection_idx = first_pos_win + 12
                        delay_steps = int(first_detection_idx - seq_raw[0])
                    else:
                        delay_steps = None
                else:
                    rec_seq = 0.0
                    delay_steps = None

                seq_recalls[f"seq{i_seq}_recall"] = round(rec_seq, 4)
                seq_delays[f"seq{i_seq}_delay_steps"] = delay_steps

            rec_dict = {
                "seed": s,
                "percentile": p,
                "expected_fp_pct": expected_fp_pct,
                "expected_fp_fraction": expected_fp_fraction,
                "threshold_val": round(tau, 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1_score": round(metrics["f1_score"], 4),
                "roc_auc": round(metrics["roc_auc"], 4) if metrics["roc_auc"] else None,
                "pr_auc": round(metrics["pr_auc"], 4) if metrics["pr_auc"] else None,
                "fpr": round(metrics["false_positive_rate"], 4),
                "TN": cm["TN"],
                "FP": cm["FP"],
                "FN": cm["FN"],
                "TP": cm["TP"],
                **seq_recalls,
                **seq_delays,
            }
            all_seed_results.append(rec_dict)

            print(
                f"[Seed {s} | {p}% Cutoff (tau={tau:.4f})] F1={metrics['f1_score']:.4f}, Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}, FPR={metrics['false_positive_rate']:.4f}, Seq3 Rec={seq_recalls['seq3_recall']:.2%}"
            )

    df_all = pd.DataFrame(all_seed_results)

    # Compute Summary Statistics Per Threshold Percentile across 5 Seeds
    summary_rows = []
    threshold_summary = {}

    for p in percentiles:
        sub = df_all[df_all["percentile"] == p]
        p_key = f"p_{p}".replace(".", "_")

        stat_dict = {
            "percentile": p,
            "expected_fp_pct": float(sub["expected_fp_pct"].iloc[0]),
            "expected_fp_fraction": float(sub["expected_fp_fraction"].iloc[0]),
            "mean_threshold": round(float(sub["threshold_val"].mean()), 4),
            "std_threshold": round(float(sub["threshold_val"].std(ddof=1)), 4),
            "mean_precision": round(float(sub["precision"].mean()), 4),
            "std_precision": round(float(sub["precision"].std(ddof=1)), 4),
            "mean_recall": round(float(sub["recall"].mean()), 4),
            "std_recall": round(float(sub["recall"].std(ddof=1)), 4),
            "mean_f1": round(float(sub["f1_score"].mean()), 4),
            "std_f1": round(float(sub["f1_score"].std(ddof=1)), 4),
            "mean_roc_auc": round(float(sub["roc_auc"].mean()), 4),
            "std_roc_auc": round(float(sub["roc_auc"].std(ddof=1)), 4),
            "mean_pr_auc": round(float(sub["pr_auc"].mean()), 4),
            "std_pr_auc": round(float(sub["pr_auc"].std(ddof=1)), 4),
            "mean_fpr": round(float(sub["fpr"].mean()), 4),
            "std_fpr": round(float(sub["fpr"].std(ddof=1)), 4),
            "mean_seq1_recall": round(float(sub["seq1_recall"].mean()), 4),
            "mean_seq2_recall": round(float(sub["seq2_recall"].mean()), 4),
            "mean_seq3_recall": round(float(sub["seq3_recall"].mean()), 4),
            "mean_seq4_recall": round(float(sub["seq4_recall"].mean()), 4),
            "mean_seq5_recall": round(float(sub["seq5_recall"].mean()), 4),
        }

        # Average delays across seeds for detected sequences
        for i_seq in range(1, 6):
            delays = sub[f"seq{i_seq}_delay_steps"].dropna()
            stat_dict[f"mean_seq{i_seq}_delay"] = round(float(delays.mean()), 2) if len(delays) > 0 else "Undetected"

        threshold_summary[p_key] = stat_dict
        summary_rows.append(stat_dict)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("results/h12_threshold_ablation.csv", index=False)

    report_content = {
        "experiment": "H=12 Controlled Threshold Ablation Study",
        "seeds_evaluated": seeds,
        "percentiles_evaluated": percentiles,
        "expected_false_positive_fractions": {
            "99.5%": "0.5% (0.005)",
            "99.7%": "0.3% (0.003)",
            "99.8%": "0.2% (0.002)",
            "99.9%": "0.1% (0.001)",
        },
        "summary_by_percentile": threshold_summary,
        "per_seed_runs": all_seed_results,
    }

    with open("results/h12_threshold_ablation_report.json", "w") as f:
        json.dump(report_content, f, indent=2)

    # Plot Generation: h12_threshold_comparison.png
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    p_labels = [f"{p}%" for p in percentiles]
    x = np.arange(len(percentiles))

    # Subplot 1: F1, Precision, Recall vs Threshold Percentile
    axes[0, 0].plot(x, df_summary["mean_f1"], "o-", color="#2ca02c", linewidth=2.5, label="F1 Score")
    axes[0, 0].plot(x, df_summary["mean_precision"], "s--", color="#d62728", linewidth=2, label="Precision")
    axes[0, 0].plot(x, df_summary["mean_recall"], "^--", color="#1f77b4", linewidth=2, label="Recall")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(p_labels)
    axes[0, 0].set_xlabel("Calibration Percentile Threshold", fontsize=10)
    axes[0, 0].set_ylabel("Score", fontsize=10)
    axes[0, 0].set_title("H=12 Performance Metrics vs Threshold Cutoff", fontsize=11, fontweight="bold")
    axes[0, 0].legend(loc="best")
    axes[0, 0].grid(True, alpha=0.3)

    # Subplot 2: False Positive Rate (FPR) vs Threshold Percentile
    axes[0, 1].bar(x, df_summary["mean_fpr"] * 100, width=0.4, color="#e377c2", alpha=0.85)
    axes[0, 1].axhline(2.5, color="red", linestyle=":", linewidth=1.5, label="2.5% FPR Target")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(p_labels)
    axes[0, 1].set_xlabel("Calibration Percentile Threshold", fontsize=10)
    axes[0, 1].set_ylabel("False Positive Rate (%)", fontsize=10)
    axes[0, 1].set_title("H=12 False Positive Rate (FPR) vs Threshold Cutoff", fontsize=11, fontweight="bold")
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].grid(True, alpha=0.3)

    # Subplot 3: Sequence 3 Recall vs Threshold Percentile
    axes[1, 0].bar(x, df_summary["mean_seq3_recall"] * 100, width=0.4, color="#ff7f0e", alpha=0.85)
    axes[1, 0].axhline(9.73, color="blue", linestyle="--", linewidth=1.5, label="H=1 Baseline Seq3 (9.73%)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(p_labels)
    axes[1, 0].set_xlabel("Calibration Percentile Threshold", fontsize=10)
    axes[1, 0].set_ylabel("Sequence 3 Recall (%)", fontsize=10)
    axes[1, 0].set_title("H=12 Sequence 3 Recall vs Threshold Cutoff", fontsize=11, fontweight="bold")
    axes[1, 0].legend(loc="upper right")
    axes[1, 0].grid(True, alpha=0.3)

    # Subplot 4: Per-Sequence Recalls across Threshold Percentiles
    for i_seq in range(1, 6):
        axes[1, 1].plot(
            x,
            df_summary[f"mean_seq{i_seq}_recall"] * 100,
            "o-",
            linewidth=1.8,
            label=f"Seq {i_seq}",
        )
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(p_labels)
    axes[1, 1].set_xlabel("Calibration Percentile Threshold", fontsize=10)
    axes[1, 1].set_ylabel("Sequence Recall (%)", fontsize=10)
    axes[1, 1].set_title("H=12 Per-Sequence Recalls vs Threshold Cutoff", fontsize=11, fontweight="bold")
    axes[1, 1].legend(loc="lower left", fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/h12_threshold_comparison.png", dpi=300)
    plt.close()

    print("\n=== SUMMARY TABLE ACROSS THRESHOLD CUTOFFS (MEAN ACROSS 5 SEEDS) ===")
    print(f"{'Percentile':<10} | {'Tau Value':<10} | {'Expected FP':<12} | {'F1':<8} | {'Prec':<8} | {'Rec':<8} | {'FPR (%)':<8} | {'Seq3 Rec (%)':<12}")
    print("---------------------------------------------------------------------------------------------------")
    for _, r in df_summary.iterrows():
        print(
            f"{r['percentile']}%{'':<5} | {r['mean_threshold']:<10.4f} | {r['expected_fp_pct']}%{'':<7} | {r['mean_f1']:<8.4f} | {r['mean_precision']:<8.4f} | {r['mean_recall']:<8.4f} | {r['mean_fpr']*100:<8.2f} | {r['mean_seq3_recall']*100:<12.2f}"
        )
    print("---------------------------------------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
