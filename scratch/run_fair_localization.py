"""Fair localization benchmark over the seed checkpoints from run_detection_multiseed.py.

Usage (from the repository root, after run_detection_multiseed.py):
    python scratch/run_fair_localization.py                  # 5 seeds, 50 SHAP permutations
    python scratch/run_fair_localization.py --permutations 1 # reproduces the old noisy SHAP setting
Outputs:
    results/fair_localization_records.json   (one record per attack x seed x timing)
    results/fair_localization_report.json    (summary)
    results/fair_localization_table_oracle.csv, results/fair_localization_table_detector.csv
Runtime: a few minutes on a laptop CPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scratch"))

from run_detection_multiseed import SEEDS, make_predict_fn  # noqa: E402
from src.detection.graph_model import SpatioTemporalGNN, build_adjacency_matrix_from_graph, resolve_sensor_locations
from src.evaluation.data_prep import build_splits
from src.evaluation.detection_eval import calibrate
from src.evaluation.fair_localization import evaluate_model, method_e_comparison_table, results_table, summarise
from src.network.ctown_loader import load_ctown_network
from src.network.graph_builder import build_directed_hydraulic_graph, build_network_graph
from src.network.zoning import assign_zones
from src.root_cause.sensitivity_signatures import build_sensitivity_signatures, load_sensitivity_signatures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-harmonise", action="store_true", help="must match the detection run")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--signatures", type=Path, default=ROOT_DIR / "results" / "sensitivity_signatures.npz")
    args = parser.parse_args()

    attacks = pd.read_csv(ROOT_DIR / "data" / "ground_truth_attacks.csv")
    splits = build_splits(ROOT_DIR, attacks, harmonise=not args.no_harmonise)
    if not args.signatures.exists():
        print(f"Building hydraulic sensitivity signatures at {args.signatures}", flush=True)
        build_sensitivity_signatures(
            ROOT_DIR / "data" / "raw" / "CTOWN.INP",
            ROOT_DIR / "data" / "raw" / "BATADAL_dataset03.csv",
            args.signatures,
        )
    hydraulic_signatures = load_sensitivity_signatures(args.signatures)
    if hydraulic_signatures["sensors"] != splits.sensors:
        raise ValueError("Sensitivity signature sensors do not match the benchmark datasets.")

    net = load_ctown_network(ROOT_DIR / "data" / "raw" / "CTOWN.INP")
    undirected = build_network_graph(net)
    directed = build_directed_hydraulic_graph(net)
    adjacency = build_adjacency_matrix_from_graph(undirected, splits.sensors, max_hops=3)
    sensor_nodes = {s: set(n) for s, n in resolve_sensor_locations(undirected, splits.sensors).items()}
    zones = assign_zones(undirected, method="dma")

    records = []
    for seed in (int(s) for s in args.seeds.split(",")):
        model = SpatioTemporalGNN(in_features=len(splits.sensors), window_size=12, hidden_dim=64,
                                  dropout=0.2, forecast_horizon=12)
        path = ROOT_DIR / "models" / "seeds" / f"detection_gnn_A_CTOWN_seed{seed}.pt"
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        predict = make_predict_fn(model, adjacency)
        variance, threshold = calibrate(predict, splits)
        records += evaluate_model(
            predict_fn=predict, variance=variance, threshold=threshold,
            calibration_windows=splits.calibration[0], datasets=splits.evaluation, attacks=attacks,
            sensors=splits.sensors, directed_graph=directed, sensor_nodes=sensor_nodes, dma_zones=zones,
            hydraulic_signatures=hydraulic_signatures,
            seed=seed, n_permutations=args.permutations,
        )
        print(f"seed {seed} done", flush=True)

    summary = summarise(records, attacks)
    summary.update({
        "harmonised_precision": not args.no_harmonise,
        "shap_permutations": args.permutations,
        "hydraulic_signature_file": str(args.signatures.relative_to(ROOT_DIR)),
        "hydraulic_signature_shape": list(hydraulic_signatures["signatures"].shape),
        "method_e_evidence": "absolute permutation-SHAP mass; target asset sensor channels masked for spoof/concealment",
    })
    out = ROOT_DIR / "results"
    (out / "fair_localization_records.json").write_text(json.dumps(records, indent=1, default=str) + "\n")
    (out / "fair_localization_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    for timing in ("oracle", "detector"):
        table = results_table(summary, timing)
        table.to_csv(out / f"fair_localization_table_{timing}.csv", index=False)
        print(f"\n== {timing} timing ==\n{table.to_string(index=False)}")
    table_r4 = method_e_comparison_table(summary)
    table_r4.to_csv(out / "fair_localization_table_r4.csv", index=False)
    print(f"\n== R4: SHAP vs hydraulic signatures ==\n{table_r4.to_string(index=False)}")
    print("\nD vs B (top-3, oracle):", summary["timing"]["oracle"]["D_vs_B_top3"])


if __name__ == "__main__":
    main()
