"""
run_capacity_control.py

Controlled Capacity-Matched Experiment comparing H=1 (Capacity-Matched: 95,001 params)
vs H=12 (Existing Architecture: 95,236 params) across 5 independent random seeds.

STRICT INTEGRITY RULES:
- Production checkpoint models/detection_gnn.pt is NOT touched.
- Raw BATADAL CSV files are NOT modified.
- BATADAL_test_dataset.csv is strictly UNTOUCHED.
- Checkpoints saved strictly under models/experiments/capacity_control/.
- Thresholds calculated strictly from Dataset03 20% normal calibration partition using 99.8th percentile.
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
from torch.utils.data import DataLoader, TensorDataset
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, ".")

from src.detection.dataset_loader import BATADALDatasetLoader
from src.detection.evaluate import DetectionEvaluator
from src.detection.graph_model import SpatioTemporalGNN, build_adjacency_matrix_from_graph
from src.detection.preprocessing import DataPreprocessor
from src.detection.scoring import calibrated_sensor_variance
from src.network.ctown_loader import load_ctown_network
from src.network.graph_builder import build_network_graph


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    config_path = Path("configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    out_dir = Path("models/experiments/capacity_control")
    out_dir.mkdir(parents=True, exist_ok=True)
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

    train_X_arr = preprocessor.transform_features(train_df, sensor_cols)
    calib_X_arr = preprocessor.transform_features(calib_df, sensor_cols)
    val_X_arr = preprocessor.transform_features(val_df, sensor_cols)
    val_y_arr = val_df["ATT_FLAG"].values

    net = load_ctown_network(config["paths"]["inp_file"])
    g = build_network_graph(net)
    adj_matrix = build_adjacency_matrix_from_graph(g, sensor_cols, max_hops=3)

    seeds = [42, 123, 2024, 3407, 9999]
    epochs = 50
    batch_size = 64
    learning_rate = 0.001
    percentile_cutoff = 99.8

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

    all_run_records = []

    print("=========================================================================================================")
    print("        CAPACITY-MATCHED EXPERIMENT: H=1 (95,001 params) vs H=12 (95,236 params) Across 5 Seeds        ")
    print("=========================================================================================================\n")

    configs = [
        {"horizon": 1, "hidden_dim": 79, "label": "H=1 (Cap-Matched)"},
        {"horizon": 12, "hidden_dim": 64, "label": "H=12"},
    ]

    for s in seeds:
        for cfg in configs:
            H = cfg["horizon"]
            hd = cfg["hidden_dim"]
            label = cfg["label"]

            set_seed(s)
            run_id = f"H={H}_hd={hd}_seed={s}"
            print(f"--- Processing {run_id} ({label}) ---", flush=True)

            X_tr, Y_tr, _ = preprocessor.create_forecasting_windows(train_X_arr, forecast_horizon=H)
            X_cal, Y_cal, _ = preprocessor.create_forecasting_windows(calib_X_arr, forecast_horizon=H)
            X_val, Y_val, val_y_lbls = preprocessor.create_forecasting_windows(val_X_arr, val_y_arr, forecast_horizon=H)

            model = SpatioTemporalGNN(
                in_features=F_count,
                window_size=12,
                hidden_dim=hd,
                forecast_horizon=H,
            )
            params = count_parameters(model)

            ckpt_name = f"gnn_h{H}_hd{hd}_seed{s}.pt"
            ckpt_path = out_dir / ckpt_name

            if ckpt_path.exists():
                print(f"   Loading checkpoint {ckpt_name}...", flush=True)
                model.load_state_dict(torch.load(ckpt_path, weights_only=True))
                train_loss_val = 0.0
                calib_loss_val = 0.0
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                criterion = nn.MSELoss()
                tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
                tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

                cal_X_tensor = torch.from_numpy(X_cal)
                cal_Y_tensor = torch.from_numpy(Y_cal)

                train_losses = []
                val_losses = []

                model.train()
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
                        c_preds = model(cal_X_tensor, adj_matrix=adj_matrix)
                        ep_val_loss = criterion(c_preds, cal_Y_tensor).item()
                    model.train()

                    train_losses.append(ep_tr_loss)
                    val_losses.append(ep_val_loss)

                model.eval()
                torch.save(model.state_dict(), ckpt_path)
                train_loss_val = train_losses[-1]
                calib_loss_val = val_losses[-1]

            model.eval()
            with torch.no_grad():
                c_preds = model(torch.from_numpy(X_cal), adj_matrix=adj_matrix).numpy()
                v_preds = model(torch.from_numpy(X_val), adj_matrix=adj_matrix).numpy()

            if H == 1:
                c_preds = c_preds[:, np.newaxis, :]
                Y_cal = Y_cal[:, np.newaxis, :]
                v_preds = v_preds[:, np.newaxis, :]
                Y_val = Y_val[:, np.newaxis, :]

            E_cal = (c_preds - Y_cal) ** 2
            E_val = (v_preds - Y_val) ** 2

            mu_sq_h = calibrated_sensor_variance(E_cal, sensor_cols, preprocessor.scaler)
            Z_cal = E_cal / mu_sq_h
            Z_val = E_val / mu_sq_h

            S_cal_final = np.mean(Z_cal[:, H - 1, :], axis=1)
            S_val_final = np.mean(Z_val[:, H - 1, :], axis=1)

            tau_99_8 = float(np.percentile(S_cal_final, percentile_cutoff))

            evaluator = DetectionEvaluator(threshold=tau_99_8)
            metrics = evaluator.evaluate_predictions(val_y_lbls, S_val_final, threshold=tau_99_8)
            val_preds_binary = (S_val_final >= tau_99_8).astype(int)

            seq_recalls = {}
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
                else:
                    rec_seq = 0.0

                seq_recalls[f"seq{i_seq}_recall"] = round(rec_seq, 4)

            cm = metrics["confusion_matrix"]
            rec_dict = {
                "seed": int(s),
                "horizon": int(H),
                "hidden_dim": int(hd),
                "model_label": label,
                "parameters": int(params),
                "threshold_99_8": round(float(tau_99_8), 4),
                "precision": round(float(metrics["precision"]), 4),
                "recall": round(float(metrics["recall"]), 4),
                "f1_score": round(float(metrics["f1_score"]), 4),
                "roc_auc": round(float(metrics["roc_auc"]), 4) if metrics["roc_auc"] else None,
                "pr_auc": round(float(metrics["pr_auc"]), 4) if metrics["pr_auc"] else None,
                "fpr": round(float(metrics["false_positive_rate"]), 4),
                "TN": int(cm["TN"]),
                "FP": int(cm["FP"]),
                "FN": int(cm["FN"]),
                "TP": int(cm["TP"]),
                **seq_recalls,
            }
            all_run_records.append(rec_dict)
            print(
                f"   [{label} Seed {s}] params={params}, F1={metrics['f1_score']:.4f}, Rec={metrics['recall']:.4f}, FPR={metrics['false_positive_rate']:.4f}, Seq3 Rec={seq_recalls['seq3_recall']:.2%}",
                flush=True,
            )

    df_runs = pd.DataFrame(all_run_records)
    df_runs.to_csv("results/capacity_control_metrics.csv", index=False)

    df_h1 = df_runs[df_runs["horizon"] == 1].sort_values("seed").reset_index(drop=True)
    df_h12 = df_runs[df_runs["horizon"] == 12].sort_values("seed").reset_index(drop=True)

    paired_diffs = []
    seq3_rows = []

    for i in range(len(seeds)):
        s_val = int(df_h1.loc[i, "seed"])
        h1_f1 = float(df_h1.loc[i, "f1_score"])
        h12_f1 = float(df_h12.loc[i, "f1_score"])

        h1_rec = float(df_h1.loc[i, "recall"])
        h12_rec = float(df_h12.loc[i, "recall"])

        h1_fpr = float(df_h1.loc[i, "fpr"])
        h12_fpr = float(df_h12.loc[i, "fpr"])

        h1_seq3 = float(df_h1.loc[i, "seq3_recall"])
        h12_seq3 = float(df_h12.loc[i, "seq3_recall"])

        diff_f1 = float(h12_f1 - h1_f1)
        diff_rec = float(h12_rec - h1_rec)
        diff_fpr = float(h12_fpr - h1_fpr)
        diff_seq3 = float(h12_seq3 - h1_seq3)

        paired_diffs.append({
            "seed": s_val,
            "h1_cap_params": int(df_h1.loc[i, "parameters"]),
            "h12_params": int(df_h12.loc[i, "parameters"]),
            "h1_f1": h1_f1,
            "h12_f1": h12_f1,
            "diff_f1": round(diff_f1, 4),
            "h1_rec": h1_rec,
            "h12_rec": h12_rec,
            "diff_rec": round(diff_rec, 4),
            "h1_fpr": h1_fpr,
            "h12_fpr": h12_fpr,
            "diff_fpr": round(diff_fpr, 4),
            "h1_seq3": h1_seq3,
            "h12_seq3": h12_seq3,
            "diff_seq3": round(diff_seq3, 4),
        })

        seq3_rows.append({
            "seed": s_val,
            "h1_cap_seq3_recall": f"{h1_seq3:.2%}",
            "h12_seq3_recall": f"{h12_seq3:.2%}",
            "improvement": f"{diff_seq3:+.2%}",
        })

    df_paired = pd.DataFrame(paired_diffs)
    df_seq3 = pd.DataFrame(seq3_rows)
    df_seq3.to_csv("results/capacity_control_sequence3.csv", index=False)

    metrics_to_stat = ["precision", "recall", "f1_score", "roc_auc", "pr_auc", "fpr", "seq1_recall", "seq2_recall", "seq3_recall", "seq4_recall", "seq5_recall"]
    summary_dict = {}
    summary_df_rows = []

    for m in metrics_to_stat:
        h1_m, h1_s = float(np.mean(df_h1[m])), float(np.std(df_h1[m], ddof=1))
        h12_m, h12_s = float(np.mean(df_h12[m])), float(np.std(df_h12[m], ddof=1))
        m_diff = float(np.mean(df_h12[m] - df_h1[m]))

        summary_dict[m] = {
            "h1_cap_mean": round(h1_m, 4),
            "h1_cap_std": round(h1_s, 4),
            "h1_cap_min": round(float(np.min(df_h1[m])), 4),
            "h1_cap_max": round(float(np.max(df_h1[m])), 4),
            "h12_mean": round(h12_m, 4),
            "h12_std": round(h12_s, 4),
            "h12_min": round(float(np.min(df_h12[m])), 4),
            "h12_max": round(float(np.max(df_h12[m])), 4),
            "mean_improvement": round(m_diff, 4),
        }

        summary_df_rows.append({
            "Metric": m,
            "H1_CapMatched_Mean": round(h1_m, 4),
            "H1_CapMatched_Std": round(h1_s, 4),
            "H1_CapMatched_Min": round(float(np.min(df_h1[m])), 4),
            "H1_CapMatched_Max": round(float(np.max(df_h1[m])), 4),
            "H12_Mean": round(h12_m, 4),
            "H12_Std": round(h12_s, 4),
            "H12_Min": round(float(np.min(df_h12[m])), 4),
            "H12_Max": round(float(np.max(df_h12[m])), 4),
            "Mean_Improvement": round(m_diff, 4),
        })

    df_summary = pd.DataFrame(summary_df_rows)
    df_summary.to_csv("results/capacity_control_summary.csv", index=False)

    _, p_val_f1 = stats.ttest_rel(df_h12["f1_score"], df_h1["f1_score"])
    _, p_val_rec = stats.ttest_rel(df_h12["recall"], df_h1["recall"])
    _, p_val_seq3 = stats.ttest_rel(df_h12["seq3_recall"], df_h1["seq3_recall"])
    _, p_val_roc = stats.ttest_rel(df_h12["roc_auc"], df_h1["roc_auc"])

    complete_report = {
        "experiment": "Capacity-Matched Control Experiment (H=1 Cap-Matched vs H=12)",
        "parameter_counts": {
            "H1_capacity_matched": int(df_h1["parameters"].iloc[0]),
            "H12_baseline": int(df_h12["parameters"].iloc[0]),
            "difference": int(df_h12["parameters"].iloc[0] - df_h1["parameters"].iloc[0]),
        },
        "seeds_evaluated": seeds,
        "locked_threshold_percentile": 99.8,
        "summary_statistics": summary_dict,
        "paired_differences": df_paired.to_dict(orient="records"),
        "statistical_tests": {
            "paired_ttest_f1_pvalue": float(p_val_f1),
            "paired_ttest_recall_pvalue": float(p_val_rec),
            "paired_ttest_seq3_pvalue": float(p_val_seq3),
            "paired_ttest_roc_auc_pvalue": float(p_val_roc),
        },
        "all_runs": all_run_records,
    }

    with open("results/capacity_control_report.json", "w") as f:
        json.dump(complete_report, f, indent=2)

    # Plot 1: F1 Score Comparison (capacity_control_f1.png)
    x_pos = np.arange(len(seeds))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x_pos - width/2, df_h1["f1_score"], width, label="H=1 Capacity-Matched (95.0k params)", color="#1f77b4", alpha=0.85)
    plt.bar(x_pos + width/2, df_h12["f1_score"], width, label="H=12 (95.2k params)", color="#2ca02c", alpha=0.85)
    plt.xticks(x_pos, [f"Seed {s}" for s in seeds])
    plt.ylabel("F1 Score", fontsize=11)
    plt.title("Capacity-Matched F1 Comparison: H=1 (95.0k) vs H=12 (95.2k) at 99.8% Cutoff", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/capacity_control_f1.png", dpi=300)
    plt.close()

    # Plot 2: Sequence 3 Recall Comparison (capacity_control_seq3.png)
    plt.figure(figsize=(9, 5))
    plt.bar(x_pos - width/2, df_h1["seq3_recall"] * 100, width, label="H=1 Capacity-Matched (95.0k params)", color="#1f77b4", alpha=0.85)
    plt.bar(x_pos + width/2, df_h12["seq3_recall"] * 100, width, label="H=12 (95.2k params)", color="#ff7f0e", alpha=0.85)
    plt.xticks(x_pos, [f"Seed {s}" for s in seeds])
    plt.ylabel("Sequence 3 Recall (%)", fontsize=11)
    plt.title("Capacity-Matched Sequence 3 Recall: H=1 (95.0k) vs H=12 (95.2k)", fontsize=12, fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/capacity_control_seq3.png", dpi=300)
    plt.close()

    # Plot 3: False Positive Rate Comparison (capacity_control_fpr.png)
    plt.figure(figsize=(9, 5))
    plt.bar(x_pos - width/2, df_h1["fpr"] * 100, width, label="H=1 Capacity-Matched (95.0k params)", color="#1f77b4", alpha=0.85)
    plt.bar(x_pos + width/2, df_h12["fpr"] * 100, width, label="H=12 (95.2k params)", color="#d62728", alpha=0.85)
    plt.xticks(x_pos, [f"Seed {s}" for s in seeds])
    plt.ylabel("False Positive Rate (%)", fontsize=11)
    plt.title("Capacity-Matched FPR Comparison: H=1 (95.0k) vs H=12 (95.2k) at 99.8% Cutoff", fontsize=12, fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/capacity_control_fpr.png", dpi=300)
    plt.close()

    print("\n=========================================================================================================")
    print("                    CAPACITY-MATCHED EXPERIMENT SUMMARY TABLE                                           ")
    print("=========================================================================================================\n")
    print(f"{'Metric':<14} | {'H=1 Cap (95.0k)':<18} | {'H=12 (95.2k)':<18} | {'Mean Diff':<12}")
    print("---------------------------------------------------------------------------------------------------------")
    for m in metrics_to_stat:
        h1_m, h1_s = summary_dict[m]["h1_cap_mean"], summary_dict[m]["h1_cap_std"]
        h12_m, h12_s = summary_dict[m]["h12_mean"], summary_dict[m]["h12_std"]
        m_diff = summary_dict[m]["mean_improvement"]
        print(f"{m:<14} | {h1_m:.4f} ({h1_s:.4f})     | {h12_m:.4f} ({h12_s:.4f})     | {m_diff:<+12.4f}")
    print("---------------------------------------------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
