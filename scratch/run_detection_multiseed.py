"""Train the H=12 SpatioTemporalGNN with C-Town and identity adjacency over 5 seeds,
score Dataset04 (ground-truth intervals) and the test set (official ATTACK_FLAG),
and save one checkpoint per seed for the fair localization benchmark.

Usage (from the repository root):
    python scratch/run_detection_multiseed.py                  # precision-harmonised (recommended)
    python scratch/run_detection_multiseed.py --no-harmonise   # reproduce the old data handling
Outputs:
    models/seeds/detection_gnn_<adjacency>_seed<seed>.pt
    results/multiseed_detection_report.json
    results/multiseed_detection_table.csv
Runtime: about 1-2 minutes per model on a laptop CPU (10 models).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.detection.graph_model import SpatioTemporalGNN, build_adjacency_matrix_from_graph
from src.evaluation.data_prep import build_splits
from src.evaluation.detection_eval import evaluate_detection, summarise_seeds
from src.network.ctown_loader import load_ctown_network
from src.network.graph_builder import build_network_graph

SEEDS = (42, 123, 2024, 3407, 9999)
EPOCHS, BATCH_SIZE, LR = 50, 64, 1e-3


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(seed: int, adjacency: torch.Tensor, x: np.ndarray, y: np.ndarray, n_sensors: int) -> SpatioTemporalGNN:
    set_seed(seed)
    model = SpatioTemporalGNN(in_features=n_sensors, window_size=12, hidden_dim=64, dropout=0.2, forecast_horizon=12)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    for _ in range(EPOCHS):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x, adj_matrix=adjacency), batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def make_predict_fn(model: SpatioTemporalGNN, adjacency: torch.Tensor):
    def predict(windows: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(windows), 2048):
                out.append(model(torch.from_numpy(np.ascontiguousarray(windows[i:i + 2048], dtype=np.float32)),
                                 adj_matrix=adjacency).numpy())
        return np.concatenate(out)
    return predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-harmonise", action="store_true")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    attacks = pd.read_csv(ROOT_DIR / "data" / "ground_truth_attacks.csv")
    splits = build_splits(ROOT_DIR, attacks, harmonise=not args.no_harmonise)
    graph = build_network_graph(load_ctown_network(ROOT_DIR / "data" / "raw" / "CTOWN.INP"))
    a_ctown = build_adjacency_matrix_from_graph(graph, splits.sensors, max_hops=3)
    adjacencies = {"A_CTOWN": a_ctown, "Identity": torch.eye(len(splits.sensors), dtype=a_ctown.dtype)}

    (ROOT_DIR / "models" / "seeds").mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in seeds:
        for name, adjacency in adjacencies.items():
            model = train(seed, adjacency, *splits.train, n_sensors=len(splits.sensors))
            torch.save(model.state_dict(), ROOT_DIR / "models" / "seeds" / f"detection_gnn_{name}_seed{seed}.pt")
            report = evaluate_detection(make_predict_fn(model, adjacency), splits, attacks)
            report.pop("variance")
            runs.append({"seed": seed, "adjacency": name, "report": report})
            d04, test = report["Dataset04"]["metrics"], report["Test Dataset"]["metrics"]
            print(f"seed {seed} {name:8s} | D04 F1 {d04['f1_score']:.3f} AUC {d04['roc_auc']:.3f} "
                  f"| TEST F1 {test['f1_score']:.3f} AUC {test['roc_auc']:.3f}", flush=True)

    table = summarise_seeds(runs)
    print(table.to_string(index=False))
    table.to_csv(ROOT_DIR / "results" / "multiseed_detection_table.csv", index=False)
    (ROOT_DIR / "results" / "multiseed_detection_report.json").write_text(
        json.dumps({"harmonised_precision": not args.no_harmonise, "seeds": seeds, "runs": runs},
                   indent=2, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
