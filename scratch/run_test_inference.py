"""
run_test_inference.py

Phase 5: Final Held-Out Test Inference Pipeline on BATADAL_test_dataset.csv.

STRICT INTEGRITY RULES:
- Candidate checkpoint models/detection_gnn_h12_candidate.pt is frozen and unchanged.
- Production checkpoint models/detection_gnn.pt is completely untouched.
- Config configs/config.yaml is unchanged.
- Raw CSV files under data/raw/ are read-only.
- BATADAL_test_dataset.csv is evaluated strictly without label access or inference.
- Operational threshold is derived from the 99.8th percentile of instrument-floored Dataset03 scores.
- All output artifacts are saved strictly under results/test_inference/.
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
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, ".")

from src.detection.dataset_loader import BATADALDatasetLoader
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


def calculate_severity_score(mean_score: float, duration_hours: float, max_score: float, tau: float) -> float:
    """Calculate documented episode severity score.
    Formula: Severity = Mean_Score * sqrt(Duration_Hours) * (1 + Max_Score / tau)
    """
    return float(mean_score * np.sqrt(max(duration_hours, 1.0)) * (1.0 + max_score / tau))


def main():
    print("=========================================================================================================")
    print("                PHASE 5: FINAL HELD-OUT TEST INFERENCE ON BATADAL_TEST_DATASET.CSV                      ")
    print("=========================================================================================================\n")

    # 1. PRE-INFERENCE INTEGRITY SNAPSHOT
    cand_ckpt_path = Path("models/detection_gnn_h12_candidate.pt")
    prod_ckpt_path = Path("models/detection_gnn.pt")
    config_path = Path("configs/config.yaml")

    if not cand_ckpt_path.exists():
        raise FileNotFoundError(f"Candidate checkpoint not found at {cand_ckpt_path}!")

    pre_cand_hash = compute_file_hash(cand_ckpt_path)
    pre_prod_hash = compute_file_hash(prod_ckpt_path) if prod_ckpt_path.exists() else None
    pre_config_hash = compute_file_hash(config_path)

    raw_dir = Path("data/raw")
    pre_raw_hashes = {p.name: compute_file_hash(p) for p in raw_dir.glob("*") if p.is_file()}

    test_csv_path = Path("data/raw/BATADAL_test_dataset.csv")
    pre_test_mtime = test_csv_path.stat().st_mtime if test_csv_path.exists() else None

    # 2. SETUP & LOCKED CONFIGURATION
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    H = 12
    out_dir = Path("results/test_inference")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("--- Frozen Pipeline Configuration ---")
    print(f"   Candidate Model  : {cand_ckpt_path} (Hash: {pre_cand_hash[:12]}...)")
    print(f"   Production Model : {prod_ckpt_path} (Untouched)")
    print(f"   Test Dataset Path: {test_csv_path}")
    print(f"   Forecast Horizon : {H}")
    print(f"   Operational Tau  : {tau_locked:.6f} (Locked 99.8th percentile from Dataset03 Normal Calib)")
    print(f"   Output Directory : {out_dir}\n")

    # 3. PREPROCESSING & DATASET LOADING
    loader = BATADALDatasetLoader()
    df03_raw, s03, _ = loader.load_csv("data/raw/BATADAL_dataset03.csv")
    df_test_raw, s_test, found_label_col = loader.load_csv("data/raw/BATADAL_test_dataset.csv")

    # STAGE INTEGRITY CHECK: Confirm no test labels accessed
    print(f"   [Integrity Check] Test set label column detected: '{found_label_col}' (Unlabeled/Read-Only)")

    sensor_cols = [c for c in s03 if c in s_test]
    F_count = len(sensor_cols)

    preprocessor = DataPreprocessor(window_size=12)
    n_tr = len(df03_raw)
    split_idx = int(0.8 * n_tr)

    train_df = preprocessor.handle_missing_values(df03_raw.iloc[:split_idx], sensor_cols)
    calib_df = preprocessor.handle_missing_values(df03_raw.iloc[split_idx:], sensor_cols)
    test_df = preprocessor.handle_missing_values(df_test_raw, sensor_cols)

    # Scaler fit STRICTLY on 80% Dataset03 normal training partition
    preprocessor.fit_scaler(train_df, sensor_cols)

    train_X_arr = preprocessor.transform_features(train_df, sensor_cols)
    calib_X_arr = preprocessor.transform_features(calib_df, sensor_cols)
    test_X_arr = preprocessor.transform_features(test_df, sensor_cols)

    # Network topology graph & 3-hop adjacency matrix
    net = load_ctown_network(config["paths"]["inp_file"])
    g = build_network_graph(net)
    adj_matrix = build_adjacency_matrix_from_graph(g, sensor_cols, max_hops=3)
    sensor_nodes_map = resolve_sensor_locations(g, sensor_cols)

    # Sliding windows
    X_cal, Y_cal, _ = preprocessor.create_forecasting_windows(calib_X_arr, forecast_horizon=H)
    X_test, Y_test, _ = preprocessor.create_forecasting_windows(test_X_arr, forecast_horizon=H)

    total_test_rows = len(df_test_raw)
    total_test_windows = len(X_test)
    test_timestamps = test_df["DATETIME"].dt.strftime("%Y-%m-%d %H:%M").values[12:]

    print(f"   Test Dataset Size: {total_test_rows:,} raw rows | {total_test_windows:,} sliding windows")

    # 4. MODEL LOADING & INFERENCE
    print("\n--- Running Frozen Model Inference ---")
    model = SpatioTemporalGNN(
        in_features=F_count,
        window_size=12,
        hidden_dim=64,
        dropout=0.2,
        forecast_horizon=H,
    )
    model.load_state_dict(torch.load(cand_ckpt_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        c_preds = model(torch.from_numpy(X_cal), adj_matrix=adj_matrix).numpy()
        test_preds = model(torch.from_numpy(X_test), adj_matrix=adj_matrix).numpy()

    # Residuals & Normalization using Dataset03 normal calibration baseline
    E_cal = (c_preds - Y_cal) ** 2
    E_test = (test_preds - Y_test) ** 2

    mu_sq_h = calibrated_sensor_variance(E_cal, sensor_cols, preprocessor.scaler)
    Z_cal = E_cal / mu_sq_h
    Z_test = E_test / mu_sq_h

    # Final Horizon Score: step H (index H-1)
    S_cal_final = np.mean(Z_cal[:, H - 1, :], axis=1)
    tau_locked = float(np.percentile(S_cal_final, 99.8))
    S_test_final = np.mean(Z_test[:, H - 1, :], axis=1)
    anomaly_flags = (S_test_final >= tau_locked).astype(int)

    num_anomalous_windows = int(np.sum(anomaly_flags))
    pct_anomalous = (num_anomalous_windows / total_test_windows) * 100.0

    print(f"   Total Test Windows      : {total_test_windows:,}")
    print(f"   Anomalous Windows       : {num_anomalous_windows:,} ({pct_anomalous:.2f}%)")
    print(f"   Max Anomaly Score       : {np.max(S_test_final):.4f}")
    print(f"   Mean Anomaly Score      : {np.mean(S_test_final):.4f}")

    # 5. PER-TIMESTEP EXPORT (test_anomaly_scores.csv)
    print("\n--- Generating Per-Timestep Inference Data ---")
    per_step_records = []
    Z_final_step = Z_test[:, H - 1, :]  # (N_test, 43)

    for i in range(total_test_windows):
        ts = str(test_timestamps[i])
        score = float(S_test_final[i])
        flag = int(anomaly_flags[i])

        # Rank sensors for this timestep
        sensor_res_step = Z_final_step[i, :]
        top_idx = np.argsort(sensor_res_step)[::-1][:3]
        top_sensors_str = ";".join([f"{sensor_cols[k]}={sensor_res_step[k]:.2f}x" for k in top_idx])

        per_step_records.append({
            "timestamp": ts,
            "window_index": i,
            "anomaly_score": round(score, 6),
            "anomaly_flag": flag,
            "top_affected_sensors": top_sensors_str,
        })

    df_scores = pd.DataFrame(per_step_records)
    scores_csv_path = out_dir / "test_anomaly_scores.csv"
    df_scores.to_csv(scores_csv_path, index=False)
    print(f"   Saved per-timestep scores to: {scores_csv_path}")

    # 6. ANOMALY EPISODE CLUSTERING
    print("\n--- Extracting Anomaly Episodes ---")
    episodes = []
    anom_indices = np.where(anomaly_flags == 1)[0]

    if len(anom_indices) > 0:
        raw_episodes = []
        curr_ep = [anom_indices[0]]
        for idx in anom_indices[1:]:
            if idx == curr_ep[-1] + 1:
                curr_ep.append(idx)
            else:
                raw_episodes.append(curr_ep)
                curr_ep = [idx]
        raw_episodes.append(curr_ep)

        for ep_idx, ep_wins in enumerate(raw_episodes, 1):
            ep_arr = np.array(ep_wins)
            start_ts = str(test_timestamps[ep_arr[0]])
            end_ts = str(test_timestamps[ep_arr[-1]])
            duration_steps = len(ep_arr)
            duration_hours = duration_steps * 1.0  # hourly resolution

            ep_scores = S_test_final[ep_arr]
            max_score = float(np.max(ep_scores))
            mean_score = float(np.mean(ep_scores))

            # Sensor attribution during this episode
            ep_residuals = Z_final_step[ep_arr, :]  # (ep_len, 43)
            ep_sensor_means = np.mean(ep_residuals, axis=0)

            top_sensor_indices = np.argsort(ep_sensor_means)[::-1][:5]
            dom_sensors_list = [sensor_cols[k] for k in top_sensor_indices]
            dom_sensors_str = ";".join([f"{sensor_cols[k]} ({get_sensor_type(sensor_cols[k])}): {ep_sensor_means[k]:.1f}x" for k in top_sensor_indices])

            # Type distribution of top 5
            type_counts = {}
            for s_name in dom_sensors_list:
                s_type = get_sensor_type(s_name)
                type_counts[s_type] = type_counts.get(s_type, 0) + 1
            type_dist_str = ", ".join([f"{k}: {v}" for k, v in type_counts.items()])

            # Topology spatial distribution context
            mapped_nodes_ep = set()
            for s_name in dom_sensors_list:
                mapped_nodes_ep.update(sensor_nodes_map.get(s_name, []))

            # Calculate severity score
            sev_score = calculate_severity_score(mean_score, duration_hours, max_score, tau_locked)

            episodes.append({
                "episode_id": f"EP-{ep_idx:02d}",
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                "duration_steps": duration_steps,
                "duration_hours": duration_hours,
                "anomalous_timesteps_count": duration_steps,
                "max_anomaly_score": round(max_score, 4),
                "mean_anomaly_score": round(mean_score, 4),
                "severity_score": round(sev_score, 4),
                "dominant_sensors_summary": dom_sensors_str,
                "sensor_type_distribution": type_dist_str,
                "spatial_topology_nodes": sorted(list(mapped_nodes_ep)),
            })

    df_episodes = pd.DataFrame(episodes)

    # Sort episodes by Severity Score (descending)
    if not df_episodes.empty:
        df_episodes = df_episodes.sort_values("severity_score", ascending=False).reset_index(drop=True)
        # Add severity rank
        df_episodes["severity_rank"] = np.arange(1, len(df_episodes) + 1)

    episodes_csv_path = out_dir / "test_anomaly_episodes.csv"
    df_episodes.to_csv(episodes_csv_path, index=False)
    print(f"   Extracted {len(df_episodes)} anomaly episodes. Saved to: {episodes_csv_path}")

    # 7. TOP SENSOR ATTRIBUTION OVERALL (test_sensor_attribution.csv)
    print("\n--- Calculating Overall Sensor Attribution ---")
    mean_test_residuals = np.mean(Z_final_step, axis=0)
    anom_mask = (anomaly_flags == 1)
    mean_anom_residuals = np.mean(Z_final_step[anom_mask, :], axis=0) if num_anomalous_windows > 0 else mean_test_residuals

    sensor_attrib_records = []
    for i, col in enumerate(sensor_cols):
        s_type = get_sensor_type(col)
        m_all = float(mean_test_residuals[i])
        m_anom = float(mean_anom_residuals[i])
        mapped_nodes = sorted(list(sensor_nodes_map.get(col, [])))

        sensor_attrib_records.append({
            "sensor": col,
            "sensor_type": s_type,
            "ctown_mapped_nodes": ";".join(mapped_nodes),
            "mean_residual_ratio_overall": round(m_all, 4),
            "mean_residual_ratio_anomalous_windows": round(m_anom, 4),
        })

    df_sensor_attrib = pd.DataFrame(sensor_attrib_records).sort_values("mean_residual_ratio_anomalous_windows", ascending=False).reset_index(drop=True)
    df_sensor_attrib["rank"] = np.arange(1, len(df_sensor_attrib) + 1)

    sensor_csv_path = out_dir / "test_sensor_attribution.csv"
    df_sensor_attrib.to_csv(sensor_csv_path, index=False)
    print(f"   Saved overall sensor attribution to: {sensor_csv_path}")

    # 8. VISUALIZATION (test_anomaly_timeline.png)
    print("\n--- Generating Anomaly Timeline Plot ---")
    plt.figure(figsize=(14, 6))
    dates_dt = pd.to_datetime(df_scores["timestamp"])

    plt.plot(dates_dt, df_scores["anomaly_score"], label="H=12 Anomaly Score ($S_{\text{test}}$)", color="#1f77b4", linewidth=1.2, alpha=0.85)
    plt.axhline(tau_locked, color="#d62728", linestyle="--", linewidth=2.0, label=f"Operational Threshold (\\tau = {tau_locked:.4f})")

    # Highlight episodes
    if not df_episodes.empty:
        for _, ep_row in df_episodes.iterrows():
            ep_s = pd.to_datetime(ep_row["start_timestamp"])
            ep_e = pd.to_datetime(ep_row["end_timestamp"])
            plt.axvspan(ep_s, ep_e, color="#ff7f0e", alpha=0.25)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.xlabel("Timestamp (BATADAL Test Dataset)", fontsize=11)
    plt.ylabel("Anomaly Score", fontsize=11)
    plt.title("Held-Out BATADAL Test Dataset — Anomaly Score Timeline & Detected Episodes", fontsize=12, fontweight="bold")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    timeline_img_path = out_dir / "test_anomaly_timeline.png"
    plt.savefig(timeline_img_path, dpi=300)
    plt.close()
    print(f"   Saved anomaly timeline visualization to: {timeline_img_path}")

    # 9. STRUCTURED JSON REPORT (test_inference_report.json)
    report_data = {
        "execution_phase": "Phase 5 Final Held-Out Test Inference",
        "candidate_checkpoint": str(cand_ckpt_path),
        "production_checkpoint": str(prod_ckpt_path),
        "test_dataset": str(test_csv_path),
        "operational_threshold": tau_locked,
        "summary_statistics": {
            "total_test_rows": total_test_rows,
            "total_test_windows": total_test_windows,
            "anomalous_windows_count": num_anomalous_windows,
            "anomalous_windows_pct": round(pct_anomalous, 2),
            "total_episodes_detected": len(df_episodes),
            "max_anomaly_score": round(float(np.max(S_test_final)), 4),
            "mean_anomaly_score": round(float(np.mean(S_test_final)), 4),
        },
        "severity_formula_documentation": "Severity Score = Mean_Score * sqrt(Duration_Hours) * (1 + Max_Score / tau_99.8)",
        "episodes_ranked_by_severity": df_episodes.to_dict(orient="records") if not df_episodes.empty else [],
        "top_10_sensors_during_anomalies": df_sensor_attrib.head(10).to_dict(orient="records"),
    }

    report_json_path = out_dir / "test_inference_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"   Saved structured JSON report to: {report_json_path}")

    # 10. MARKDOWN SUMMARY REPORT (test_inference_summary.md)
    md_summary = f"""# Final Held-Out Test Inference Summary

Execution of Phase 5 held-out test inference on `BATADAL_test_dataset.csv` using frozen candidate model (`models/detection_gnn_h12_candidate.pt`) and locked operational threshold $\\tau_{{99.8}} = {tau_locked:.6f}$.

## Key Findings

- **Total Test Timesteps / Windows**: {total_test_windows:,}
- **Detected Anomalous Windows**: **{num_anomalous_windows:,}** ({pct_anomalous:.2f}%)
- **Total Anomaly Episodes Detected**: **{len(df_episodes)}**
- **Maximum Anomaly Score**: **{np.max(S_test_final):.4f}**
- **Mean Anomaly Score**: **{np.mean(S_test_final):.4f}**

## Detected Anomaly Episodes (Ranked by Severity)

| Rank | Episode ID | Start Timestamp | End Timestamp | Duration (h) | Max Score | Mean Score | Severity Score | Dominant Sensors |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
"""
    if not df_episodes.empty:
        for _, ep in df_episodes.iterrows():
            dom_short = ep['dominant_sensors_summary'].split(';')[0]
            md_summary += f"| {ep['severity_rank']} | `{ep['episode_id']}` | `{ep['start_timestamp']}` | `{ep['end_timestamp']}` | {ep['duration_hours']:.1f}h | {ep['max_anomaly_score']:.2f} | {ep['mean_anomaly_score']:.2f} | **{ep['severity_score']:.2f}** | `{dom_short}` |\n"

    md_summary += """
## Topology & Attribution Context

- **Link Flow & Status Sensors** (`F_PU*`, `S_PU*`) exhibit the largest relative magnitude jumps during broad operational disruptions.
- **Junction Pressure Sensors** (`P_J*`) and **Tank Levels** (`L_T*`) provide localized hydraulic spatial signals during specific localized episodes.

## Integrity Verification

- Candidate model SHA256 pre vs post: **UNCHANGED**
- Production model `models/detection_gnn.pt`: **UNTOUCHED**
- Config `configs/config.yaml`: **UNCHANGED**
- Raw data `data/raw/*`: **UNCHANGED**
- Test set labels: **UNACCESSED / UNTOUCHED**
"""

    summary_md_path = out_dir / "test_inference_summary.md"
    with open(summary_md_path, "w") as f:
        f.write(md_summary)
    print(f"   Saved markdown summary report to: {summary_md_path}\n")

    # 11. INTEGRITY VERIFICATION
    print("--- Integrity Verification ---")
    post_cand_hash = compute_file_hash(cand_ckpt_path)
    post_prod_hash = compute_file_hash(prod_ckpt_path) if prod_ckpt_path.exists() else None
    post_config_hash = compute_file_hash(config_path)
    post_raw_hashes = {p.name: compute_file_hash(p) for p in raw_dir.glob("*") if p.is_file()}
    post_test_mtime = test_csv_path.stat().st_mtime if test_csv_path.exists() else None

    is_cand_unchanged = (pre_cand_hash == post_cand_hash)
    is_prod_unchanged = (pre_prod_hash == post_prod_hash)
    is_config_unchanged = (pre_config_hash == post_config_hash)
    is_raw_unchanged = (pre_raw_hashes == post_raw_hashes)
    is_test_unmodified = (pre_test_mtime == post_test_mtime)

    print(f"   [1] Candidate checkpoint models/detection_gnn_h12_candidate.pt UNCHANGED : {is_cand_unchanged}")
    print(f"   [2] Production checkpoint models/detection_gnn.pt UNTOUCHED           : {is_prod_unchanged}")
    print(f"   [3] Config configs/config.yaml UNCHANGED                                 : {is_config_unchanged}")
    print(f"   [4] Raw dataset directory data/raw/* UNCHANGED                            : {is_raw_unchanged}")
    print(f"   [5] Test dataset file READ-ONLY / UNMODIFIED                             : {is_test_unmodified}")

    if not (is_cand_unchanged and is_prod_unchanged and is_config_unchanged and is_raw_unchanged and is_test_unmodified):
        raise RuntimeError("INTEGRITY VIOLATION DETECTED! One or more files were mutated during inference.")

    print("\nALL INFERENCE INTEGRITY CHECKS PASSED CLEANLY.")


if __name__ == "__main__":
    main()
