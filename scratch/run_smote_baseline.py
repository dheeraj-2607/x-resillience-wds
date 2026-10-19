"""Run a leakage-safe supervised RF baseline on BATADAL Dataset04."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.detection.dataset_loader import BATADALDatasetLoader
from src.detection.smote_baseline import build_temporal_features, evaluate_smote_tree_baselines


DATASET_PATH = ROOT_DIR / "data" / "raw" / "BATADAL_dataset04.csv"
OUTPUT_PATH = ROOT_DIR / "results" / "smote_baseline_report.json"


def main() -> None:
    loader = BATADALDatasetLoader()
    dataset04, sensors04, label04 = loader.load_csv(DATASET_PATH)
    if label04 is None:
        raise ValueError("Dataset04 must contain ground-truth labels for supervised evaluation.")

    sensor_columns = sensors04
    dataset04 = dataset04.sort_values(loader.timestamp_col, kind="stable").reset_index(drop=True)
    attacks = pd.read_csv(ROOT_DIR / "data" / "ground_truth_attacks.csv")
    attacks = attacks[attacks["dataset"] == "Dataset04"]
    timestamps = dataset04[loader.timestamp_col]
    labels = np.zeros(len(dataset04), dtype=int)
    for attack in attacks.itertuples(index=False):
        labels[timestamps.between(pd.Timestamp(attack.start_time), pd.Timestamp(attack.end_time)).to_numpy()] = 1
    features = build_temporal_features(dataset04, sensor_columns, loader.timestamp_col).to_numpy()

    report = evaluate_smote_tree_baselines(
        features,
        labels,
        test_size=0.2,
        random_state=42,
    )
    report["datasets"] = {
        "supervised_source": str(DATASET_PATH.relative_to(ROOT_DIR)),
        "ground_truth_source": "data/ground_truth_attacks.csv",
    }
    report["input_sensor_count"] = len(sensor_columns)
    report["total_rows"] = len(dataset04)
    report["train_end_timestamp"] = str(dataset04[loader.timestamp_col].iloc[report["split_index"] - 1])
    report["test_start_timestamp"] = str(dataset04[loader.timestamp_col].iloc[report["split_index"]])
    report["source_partial_labels"] = {
        "unlabelled_ATT_FLAG_rows": int(dataset04[label04].isna().sum()),
        "metric_labels_from": "complete published Dataset04 attack intervals; outside intervals is normal",
    }
    report["feature_engineering"] = [
        "raw sensor values",
        "one-hour backward differences",
        "3-hour and 12-hour trailing rolling means and population standard deviations",
        "hour-of-day sine and cosine encoding",
    ]
    report["source_partition_rows"] = {
        "dataset04_train": report["split_index"],
        "dataset04_test": len(dataset04) - report["split_index"],
        "train_normal_intervals": int(np.sum(labels[: report["split_index"]] == 0)),
        "train_attack_intervals": int(np.sum(labels[: report["split_index"]] == 1)),
        "test_normal_intervals": int(np.sum(labels[report["split_index"] :] == 0)),
        "test_attack_intervals": int(np.sum(labels[report["split_index"] :] == 1)),
    }
    report["note"] = (
        "This is a separate supervised tree baseline on Dataset04. The authoritative table supplies "
        "all seven attack intervals; timestamps within them are positive and timestamps outside are normal. "
        "The partial ATT_FLAG sentinel is not treated as a class label. Dataset04 is split chronologically "
        "before fitting imputation, scaling, or SMOTE. "
        "It does not alter or replace the normal-only SpatioTemporalGNN results."
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to {OUTPUT_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()