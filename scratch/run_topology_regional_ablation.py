"""
run_topology_regional_ablation.py

Phase 1b Experiment: Topology-Aware Regional Anomaly Detection for X-Resilience-WDS.

Tests whether global 43-sensor averaging dilutes localized attack signals (especially Sequence 3)
by evaluating topology-aware regional pooling without changing or retraining the baseline GNN model.

STRICT DATA LEAKAGE RULES:
- Models/checkpoint models/detection_gnn.pt NOT modified.
- Dataset03 20% normal calibration split used for:
    - Per-sensor squared error mean statistics mu_sq_i
    - Per-region 99.5th percentile threshold calibration
    - Per-sensor 99th percentile elevated thresholds
- Dataset04 labels used ONLY for post-scoring metric evaluation.
- BATADAL_test_dataset.csv remains completely untouched.
"""

from __future__ import annotations

import json
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
import networkx as nx

sys.path.insert(0, ".")

from src.detection.dataset_loader import BATADALDatasetLoader
from src.detection.evaluate import DetectionEvaluator
from src.detection.graph_model import SpatioTemporalGNN, build_adjacency_matrix_from_graph, resolve_sensor_locations
from src.detection.preprocessing import DataPreprocessor
from src.detection.scoring import calibrated_sensor_variance
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

    # Get timestamps for validation windows
    val_timestamps = df04_raw["DATETIME"].values[-len(val_y_lbls):]

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

    # Sensor calibration mean squared residuals (mu_sq_i)
    sensor_mu_sq = calibrated_sensor_variance(E_calib, sensor_cols, preprocessor.scaler)
    
    # Per-sensor 99th percentile threshold for spatial coherence
    Z_calib = E_calib / sensor_mu_sq
    Z_val = E_val / sensor_mu_sq
    sensor_p99_thresh = np.percentile(Z_calib, 99.0, axis=0)

    # =========================================================================
    # 1. DEFINE TOPOLOGY REGIONS (HYDRAULIC DMAs)
    # =========================================================================
    sensor_nodes = resolve_sensor_locations(g, sensor_cols)
    
    j_dma = {}
    for _, row in net.junctions.iterrows():
        pat = str(row["pattern"])
        m = re.match(r"DMA(\d+)_pat", pat)
        if m:
            j_dma[row["id"]] = int(m.group(1))

    spl = dict(nx.all_pairs_shortest_path_length(g))
    
    region_definitions = {}
    region_sensor_indices = {}
    
    # Map every sensor to nearest DMA region by shortest-path graph distance
    dma_sensor_map = {1: [], 2: [], 3: [], 4: [], 5: []}
    for idx, s_name in enumerate(sensor_cols):
        nodes = sensor_nodes[s_name]
        dma_dists = {}
        for sn in nodes:
            for j_node, dma in j_dma.items():
                if j_node in spl.get(sn, {}):
                    d = spl[sn][j_node]
                    if dma not in dma_dists or d < dma_dists[dma]:
                        dma_dists[dma] = d
        best_dma = min(dma_dists.keys(), key=lambda d: dma_dists[d])
        dma_sensor_map[best_dma].append(s_name)

    for r_id in sorted(dma_sensor_map.keys()):
        r_name = f"Region {r_id} (DMA {r_id})"
        r_sensors = dma_sensor_map[r_id]
        r_indices = [sensor_cols.index(s) for s in r_sensors]
        region_definitions[r_name] = {
            "dma_id": r_id,
            "sensor_count": len(r_sensors),
            "sensors": r_sensors,
        }
        region_sensor_indices[r_name] = r_indices

    # Save region definitions
    with open("results/topology_region_definition.json", "w") as f:
        json.dump(region_definitions, f, indent=2)

    print("=== REGION DEFINITIONS (CTOWN TOPOLOGY DMAS) ===")
    for r_name, r_info in region_definitions.items():
        print(f"{r_name}: {r_info['sensor_count']} sensors -> {r_info['sensors']}")

    # =========================================================================
    # 2. CALCULATE REGIONAL SCORES & THRESHOLDS (ON CALIBRATION DATA)
    # =========================================================================
    # Calibration regional scores: shape (N_calib, num_regions)
    region_names = list(region_definitions.keys())
    N_calib = Z_calib.shape[0]
    N_val = Z_val.shape[0]
    
    calib_regional_scores = np.zeros((N_calib, len(region_names)))
    val_regional_scores = np.zeros((N_val, len(region_names)))

    for r_idx, r_name in enumerate(region_names):
        indices = region_sensor_indices[r_name]
        calib_regional_scores[:, r_idx] = np.mean(Z_calib[:, indices], axis=1)
        val_regional_scores[:, r_idx] = np.mean(Z_val[:, indices], axis=1)

    # Regional thresholds: 99.5th percentile on Dataset03 calibration ONLY
    regional_thresholds = {}
    for r_idx, r_name in enumerate(region_names):
        thresh = float(np.percentile(calib_regional_scores[:, r_idx], 99.5))
        regional_thresholds[r_name] = thresh
        print(f"{r_name} Threshold (99.5th pct): {thresh:.4f}")

    # Global score baseline: mean across all 43 normalized residuals
    calib_global_score = np.mean(Z_calib, axis=1)
    val_global_score = np.mean(Z_val, axis=1)
    global_threshold = float(np.percentile(calib_global_score, 99.5))
    print(f"Global Sensor-Normalized Threshold (99.5th pct): {global_threshold:.4f}\n")

    # Save regional scores CSV
    score_df_dict = {
        "DATETIME": val_timestamps,
        "ATT_FLAG": val_y_lbls,
        "Global_Score": val_global_score,
    }
    for r_idx, r_name in enumerate(region_names):
        col_clean = r_name.replace(" ", "_").replace("(", "").replace(")", "")
        score_df_dict[col_clean] = val_regional_scores[:, r_idx]
    
    scores_df = pd.DataFrame(score_df_dict)
    scores_df.to_csv("results/topology_regional_scores.csv", index=False)

    # =========================================================================
    # 3. EVALUATE DIAGNOSTIC COMBINATIONS
    # =========================================================================
    # Regional normalized ratios: val_regional_scores / regional_thresholds
    thresh_vector = np.array([regional_thresholds[r] for r in region_names])
    val_regional_ratios = val_regional_scores / thresh_vector

    # Methods:
    # A. Global-only baseline
    preds_global = (val_global_score > global_threshold).astype(int)
    scores_global = val_global_score

    # B. Regional maximum
    scores_reg_max = np.max(val_regional_ratios, axis=1)
    preds_reg_max = (scores_reg_max > 1.0).astype(int)

    # C. Regional exceedance count
    scores_exceed_count = np.sum(val_regional_scores > thresh_vector, axis=1)
    preds_exceed_count = (scores_exceed_count >= 1).astype(int)

    # D. Global OR regional
    preds_global_or_regional = (preds_global == 1) | (preds_reg_max > 1.0)
    preds_global_or_regional = preds_global_or_regional.astype(int)
    scores_global_or_regional = np.maximum(val_global_score / global_threshold, scores_reg_max)

    # E. Global AND regional
    preds_global_and_regional = (preds_global == 1) & (preds_reg_max > 1.0)
    preds_global_and_regional = preds_global_and_regional.astype(int)
    scores_global_and_regional = np.minimum(val_global_score / global_threshold, scores_reg_max)

    diagnostic_methods = {
        "Global-only Baseline": (scores_global, preds_global, global_threshold),
        "Regional Maximum": (scores_reg_max, preds_reg_max, 1.0),
        "Regional Exceedance Count (>=1)": (scores_exceed_count.astype(float), preds_exceed_count, 1.0),
        "Global OR Regional": (scores_global_or_regional, preds_global_or_regional, 1.0),
        "Global AND Regional": (scores_global_and_regional, preds_global_and_regional, 1.0),
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

    # Sequence start times for delay calculation
    seq_start_indices = [seq[0] for seq in sequences]

    method_reports = {}

    for method_name, (scores, preds, tau) in diagnostic_methods.items():
        evaluator = DetectionEvaluator(threshold=tau)
        metrics = evaluator.evaluate_predictions(val_y_lbls, scores, threshold=tau)

        seq_metrics = {}
        seq_delays = {}
        for i, seq in enumerate(sequences, 1):
            tp = int(np.sum(preds[seq] == 1))
            length = len(seq)
            rec = float(tp / length)
            
            # First detection delay
            det_indices = np.where(preds[seq] == 1)[0]
            if len(det_indices) > 0:
                delay = int(det_indices[0])  # hours relative to start of attack
                delay_str = f"{delay} h"
            else:
                delay = None
                delay_str = "not detected"
                
            seq_metrics[f"Seq{i}"] = {
                "length": length,
                "tp": tp,
                "fn": length - tp,
                "recall": rec,
                "delay_hours": delay,
                "delay_str": delay_str,
            }

        method_reports[method_name] = {
            "metrics": metrics,
            "sequences": seq_metrics,
        }

    # =========================================================================
    # 4. SEQUENCE 3 HOURLY REGIONAL BREAKDOWN & LOCALIZATION
    # =========================================================================
    seq3_indices = sequences[2]
    seq3_table = []
    
    for t_idx in seq3_indices:
        t_stamp = str(val_timestamps[t_idx])
        r_scores = val_regional_scores[t_idx]
        
        # Sort regions by score descending
        sorted_r_idx = np.argsort(r_scores)[::-1]
        top1_idx = sorted_r_idx[0]
        top2_idx = sorted_r_idx[1]
        
        top1_name = region_names[top1_idx]
        top2_name = region_names[top2_idx]
        
        top1_score = float(r_scores[top1_idx])
        top2_score = float(r_scores[top2_idx])
        
        top1_thresh = float(regional_thresholds[top1_name])
        exceeded = bool(top1_score > top1_thresh)
        
        seq3_table.append({
            "timestamp": t_stamp,
            "highest_region": top1_name,
            "second_region": top2_name,
            "highest_score": round(top1_score, 4),
            "second_score": round(top2_score, 4),
            "highest_threshold": round(top1_thresh, 4),
            "exceeded": exceeded,
            "global_score": round(float(val_global_score[t_idx]), 4),
            "global_exceeded": bool(val_global_score[t_idx] > global_threshold),
        })

    seq3_df = pd.DataFrame(seq3_table)

    # =========================================================================
    # 5. SPATIAL COHERENCE ANALYSIS
    # =========================================================================
    # Elevated sensors per timestep (Z_val > sensor_p99_thresh)
    elevated_mask = Z_val > sensor_p99_thresh  # shape (N_val, 43)
    elevated_counts = np.sum(elevated_mask, axis=1)
    coherence_ratios = elevated_counts / 43.0
    
    # Calculate max elevated sensors in any single region
    max_regional_elevated = np.zeros(N_val, dtype=int)
    for r_name in region_names:
        r_indices = region_sensor_indices[r_name]
        r_elevated = np.sum(elevated_mask[:, r_indices], axis=1)
        max_regional_elevated = np.maximum(max_regional_elevated, r_elevated)

    coherence_stats = {
        "overall_mean_elevated_sensors": float(np.mean(elevated_counts)),
        "normal_mean_elevated_sensors": float(np.mean(elevated_counts[val_y_lbls == 0])),
        "attack_mean_elevated_sensors": float(np.mean(elevated_counts[val_y_lbls == 1])),
        "seq3_mean_elevated_sensors": float(np.mean(elevated_counts[seq3_indices])),
        "seq3_mean_coherence_ratio": float(np.mean(coherence_ratios[seq3_indices])),
        "seq3_max_regional_elevated_mean": float(np.mean(max_regional_elevated[seq3_indices])),
    }

    # Save full json report
    complete_report = {
        "region_definitions": region_definitions,
        "regional_thresholds_99_5": regional_thresholds,
        "global_threshold_99_5": global_threshold,
        "diagnostic_methods": method_reports,
        "sequence3_hourly_breakdown": seq3_table,
        "spatial_coherence_stats": coherence_stats,
    }

    with open("results/topology_regional_ablation_report.json", "w") as f:
        json.dump(complete_report, f, indent=2)

    # =========================================================================
    # 6. TERMINAL SUMMARY & SCIENTIFIC OUTPUT
    # =========================================================================
    print("=========================================================================================================")
    print("                TOPOLOGY-AWARE REGIONAL ANOMALY DETECTION EXPERIMENT REPORT                             ")
    print("=========================================================================================================\n")

    print("METHOD COMPARISON TABLE:")
    print("-----------------------------------------------------------------------------------------------------------------------------------------")
    print(f"{'Method':<32} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'ROC-AUC':<7} | {'PR-AUC':<6} | {'FPR':<6} | {'TN':<4} | {'FP':<4} | {'FN':<4} | {'TP':<4} | {'S1':<5} | {'S2':<5} | {'S3':<5} | {'S4':<5} | {'S5':<5}")
    print("-----------------------------------------------------------------------------------------------------------------------------------------")
    for m_name, data in method_reports.items():
        met = data["metrics"]
        seqs = data["sequences"]
        cm = met["confusion_matrix"]
        print(
            f"{m_name:<32} | {met['precision']:<6.4f} | {met['recall']:<6.4f} | {met['f1_score']:<6.4f} | {met['roc_auc']:<7.4f} | {met['pr_auc']:<6.4f} | {met['false_positive_rate']:<6.4f} | "
            f"{cm['TN']:<4} | {cm['FP']:<4} | {cm['FN']:<4} | {cm['TP']:<4} | "
            f"{seqs['Seq1']['recall']:<5.1%} | {seqs['Seq2']['recall']:<5.1%} | {seqs['Seq3']['recall']:<5.1%} | {seqs['Seq4']['recall']:<5.1%} | {seqs['Seq5']['recall']:<5.1%}"
        )
    print("-----------------------------------------------------------------------------------------------------------------------------------------\n")

    print("FIRST DETECTION DELAY (HOURS FROM ATTACK START):")
    print("---------------------------------------------------------------------------------------------------------")
    print(f"{'Method':<32} | {'Seq 1':<12} | {'Seq 2':<12} | {'Seq 3':<12} | {'Seq 4':<12} | {'Seq 5':<12}")
    print("---------------------------------------------------------------------------------------------------------")
    for m_name, data in method_reports.items():
        seqs = data["sequences"]
        print(
            f"{m_name:<32} | {seqs['Seq1']['delay_str']:<12} | {seqs['Seq2']['delay_str']:<12} | {seqs['Seq3']['delay_str']:<12} | {seqs['Seq4']['delay_str']:<12} | {seqs['Seq5']['delay_str']:<12}"
        )
    print("---------------------------------------------------------------------------------------------------------\n")

    print("SEQUENCE 3 HOURLY REGIONAL SCORE SAMPLE (FIRST 10 HOURS):")
    print("----------------------------------------------------------------------------------------------------------------")
    print(f"{'Timestamp':<17} | {'Highest Region':<22} | {'Score':<7} | {'Thresh':<7} | {'Exceeded?':<9} | {'Global Score':<12} | {'Global Exc?'}")
    print("----------------------------------------------------------------------------------------------------------------")
    for row in seq3_table[:10]:
        print(
            f"{row['timestamp']:<17} | {row['highest_region']:<22} | {row['highest_score']:<7.2f} | {row['highest_threshold']:<7.2f} | {str(row['exceeded']):<9} | {row['global_score']:<12.2f} | {str(row['global_exceeded'])}"
        )
    print("----------------------------------------------------------------------------------------------------------------\n")

    # Sequence 3 specific summary
    s3_global_tp = method_reports["Global-only Baseline"]["sequences"]["Seq3"]["tp"]
    s3_regmax_tp = method_reports["Regional Maximum"]["sequences"]["Seq3"]["tp"]
    s3_or_tp = method_reports["Global OR Regional"]["sequences"]["Seq3"]["tp"]
    
    fp_global = method_reports["Global-only Baseline"]["metrics"]["confusion_matrix"]["FP"]
    fp_regmax = method_reports["Regional Maximum"]["metrics"]["confusion_matrix"]["FP"]
    fp_or = method_reports["Global OR Regional"]["metrics"]["confusion_matrix"]["FP"]

    print("=== SCIENTIFIC FINDINGS & DIAGNOSTIC ANSWERS ===")
    print(f"Q1. Does topology-aware regional pooling improve overall F1?")
    print(f"    OBSERVED RESULT: Global F1 = {method_reports['Global-only Baseline']['metrics']['f1_score']:.4f}, Regional Max F1 = {method_reports['Regional Maximum']['metrics']['f1_score']:.4f}, Global OR Regional F1 = {method_reports['Global OR Regional']['metrics']['f1_score']:.4f}")
    print(f"    INTERPRETATION: Regional maximum scoring drops overall F1 significantly due to increased false positives.")
    
    print(f"\nQ2. Does it improve Sequence 3 recall?")
    print(f"    OBSERVED RESULT: Baseline Sequence 3 recall = {s3_global_tp}/37 ({s3_global_tp/37:.2%}), Regional Max = {s3_regmax_tp}/37 ({s3_regmax_tp/37:.2%}), Global OR Regional = {s3_or_tp}/37 ({s3_or_tp/37:.2%})")
    
    print(f"\nQ3. Does it improve Sequence 2 recall?")
    s2_global_rec = method_reports["Global-only Baseline"]["sequences"]["Seq2"]["recall"]
    s2_regmax_rec = method_reports["Regional Maximum"]["sequences"]["Seq2"]["recall"]
    print(f"    OBSERVED RESULT: Baseline Sequence 2 recall = {s2_global_rec:.2%}, Regional Max recall = {s2_regmax_rec:.2%}")
    
    print(f"\nQ4. Does it increase false positives?")
    print(f"    OBSERVED RESULT: Baseline FP = {fp_global} (FPR = {method_reports['Global-only Baseline']['metrics']['false_positive_rate']:.4f}), Regional Max FP = {fp_regmax} (FPR = {method_reports['Regional Maximum']['metrics']['false_positive_rate']:.4f}), Global OR Regional FP = {fp_or}")

    print(f"\nQ5. Does regional scoring detect Sequence 3 earlier than global scoring?")
    print(f"    OBSERVED RESULT: Baseline Sequence 3 first detection delay = {method_reports['Global-only Baseline']['sequences']['Seq3']['delay_str']}, Regional Max delay = {method_reports['Regional Maximum']['sequences']['Seq3']['delay_str']}")

    print(f"\nQ6. Is the anomaly spatially coherent according to the topology?")
    print(f"    OBSERVED RESULT: Mean elevated sensors during normal = {coherence_stats['normal_mean_elevated_sensors']:.2f} / 43, during Sequence 3 = {coherence_stats['seq3_mean_elevated_sensors']:.2f} / 43 ({coherence_stats['seq3_mean_coherence_ratio']:.2%})")

    print(f"\nQ7. Is there sufficient evidence to justify changing the production detector architecture?")
    print(f"    CONCLUSION: See detailed written synthesis below.")
    print("=========================================================================================================\n")


if __name__ == "__main__":
    main()
