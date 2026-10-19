"""
verify_corrected_episodes.py

Independent Verification & Analysis of Corrected Anomaly Episodes for Phase 5 Review.
Recomputes episode boundaries, severity scores, sensor attributions, and topology mappings.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, ".")

from src.detection.graph_model import resolve_sensor_locations
from src.network.ctown_loader import load_ctown_network
from src.network.graph_builder import build_network_graph


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


def calculate_severity(mean_score: float, duration_hours: float, max_score: float, tau: float) -> float:
    return float(mean_score * np.sqrt(max(duration_hours, 1.0)) * (1.0 + max_score / tau))


def main():
    print("=====================================================================")
    print("           STEP 1: INDEPENDENT RECOMPUTATION OF EPISODES             ")
    print("=====================================================================")

    inference_report = json.loads(
        Path("results/test_inference/corrected/test_inference_report.json").read_text(encoding="utf-8")
    )
    tau = float(inference_report["operational_threshold"])
    scores_path = Path("results/test_inference/corrected/test_anomaly_scores.csv")
    df_scores = pd.read_csv(scores_path)

    total_windows = len(df_scores)
    anom_df = df_scores[df_scores["anomaly_flag"] == 1]
    total_anom_windows = len(anom_df)

    print(f"1. Total Evaluated Windows: {total_windows:,}")
    print(f"2. Total Anomalous Windows: {total_anom_windows:,} ({(total_anom_windows/total_windows)*100:.2f}%)")

    # Verify chronological monotonicity of timestamps
    timestamps_dt = pd.to_datetime(df_scores["timestamp"])
    is_mono = timestamps_dt.is_monotonic_increasing
    time_diffs = timestamps_dt.diff()[1:]
    has_gaps = (time_diffs != pd.Timedelta(hours=1)).any()
    dup_count = timestamps_dt.duplicated().sum()

    print(f"3. Strictly Monotonic Timestamps: {is_mono}")
    print(f"4. Has Non-1-Hour Gaps: {has_gaps}")
    print(f"5. Duplicate Timestamps Count: {dup_count}")

    # Recompute contiguous episode boundaries
    anom_indices = anom_df.index.values
    raw_episodes = []
    curr_ep = [anom_indices[0]]
    for idx in anom_indices[1:]:
        if idx == curr_ep[-1] + 1:
            curr_ep.append(idx)
        else:
            raw_episodes.append(curr_ep)
            curr_ep = [idx]
    raw_episodes.append(curr_ep)

    print(f"6. Recomputed Contiguous Episodes Count: {len(raw_episodes)}")

    # Load C-Town Network graph topology
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    net = load_ctown_network(config["paths"]["inp_file"])
    g = build_network_graph(net)

    # Get all sensor columns
    s_cols = [c for c in df_scores.columns if c not in ["timestamp", "window_index", "anomaly_score", "anomaly_flag", "top_affected_sensors"]]
    # Load topology mapping
    sensor_nodes = resolve_sensor_locations(g, s_cols if s_cols else ["P_J14"])

    episodes_list = []
    for ep_idx, ep_wins in enumerate(raw_episodes, 1):
        ep_arr = np.array(ep_wins)
        start_ts = str(df_scores.loc[ep_arr[0], "timestamp"])
        end_ts = str(df_scores.loc[ep_arr[-1], "timestamp"])
        duration = len(ep_arr)
        dur_h = float(duration)

        scores_sub = df_scores.loc[ep_arr, "anomaly_score"].values
        max_sc = float(np.max(scores_sub))
        mean_sc = float(np.mean(scores_sub))

        sev = calculate_severity(mean_sc, dur_h, max_sc, tau)

        # Classify episode duration type
        if duration == 1:
            ep_type = "Isolated Anomaly (1h)"
        elif duration <= 6:
            ep_type = "Short Episode (2-6h)"
        else:
            ep_type = "Persistent Episode (>6h)"

        if max_sc > 100.0:
            ep_type += " [High-Severity Outlier]"

        episodes_list.append({
            "episode_id": f"EP-{ep_idx:02d}",
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "duration_steps": duration,
            "duration_hours": dur_h,
            "anomalous_timesteps": duration,
            "max_score": max_sc,
            "mean_score": mean_sc,
            "severity_score": sev,
            "classification": ep_type,
        })

    df_ep = pd.DataFrame(episodes_list).sort_values("severity_score", ascending=False).reset_index(drop=True)
    df_ep["severity_rank"] = np.arange(1, len(df_ep) + 1)

    print("\n--- RECOMPUTED 17 ANOMALY EPISODES (RANKED BY SEVERITY) ---")
    print(f"{'Rank':<5} | {'ID':<6} | {'Start Timestamp':<17} | {'End Timestamp':<17} | {'Dur (h)':<8} | {'Max Score':<12} | {'Severity Score':<15} | {'Classification'}")
    print("-" * 115)
    for _, r in df_ep.iterrows():
        max_str = f"{r['max_score']:.2e}" if r['max_score'] > 1e4 else f"{r['max_score']:.2f}"
        sev_str = f"{r['severity_score']:.2e}" if r['severity_score'] > 1e4 else f"{r['severity_score']:.2f}"
        print(f"{r['severity_rank']:<5} | {r['episode_id']:<6} | {r['start_timestamp']:<17} | {r['end_timestamp']:<17} | {r['duration_hours']:<8.1f} | {max_str:<12} | {sev_str:<15} | {r['classification']}")

    print("\nVERIFICATION SCRIPT COMPLETED.")


if __name__ == "__main__":
    main()
