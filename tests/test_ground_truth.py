"""Scientific-integrity checks for BATADAL attack metadata and scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.evaluate import DetectionEvaluator
from src.network.ctown_loader import load_ctown_network

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_ground_truth_attacks_match_ctown_components():
    attacks = pd.read_csv(ROOT_DIR / "data" / "ground_truth_attacks.csv")
    expected_columns = [
        "attack_id",
        "dataset",
        "start_time",
        "end_time",
        "target_component",
        "component_type",
        "attack_type",
        "dma_zone",
        "affected_sensors",
    ]
    assert list(attacks.columns) == expected_columns
    assert len(attacks) == 14
    assert attacks["attack_id"].is_unique
    assert attacks["dataset"].value_counts().to_dict() == {
        "Dataset04": 7,
        "Test Dataset": 7,
    }
    assert pd.to_datetime(attacks["start_time"], format="%Y-%m-%d %H:%M", errors="raise").notna().all()
    assert pd.to_datetime(attacks["end_time"], format="%Y-%m-%d %H:%M", errors="raise").notna().all()
    assert attacks["target_component"].str.strip().ne("").all()
    assert attacks["affected_sensors"].str.strip().ne("").all()

    network = load_ctown_network(ROOT_DIR / "data" / "raw" / "CTOWN.INP")
    components_by_type = {
        "Tank": set(network.wn.tank_name_list),
        "Pump": set(network.wn.pump_name_list),
        "Valve": set(network.wn.valve_name_list),
    }
    for attack in attacks.itertuples(index=False):
        assert attack.target_component in components_by_type[attack.component_type]


def test_unlabelled_labels_are_excluded_from_evaluation():
    evaluator = DetectionEvaluator(threshold=0.5)
    results = evaluator.evaluate_predictions(
        np.array([0, 1, -999, np.nan]),
        np.array([0.1, 0.9, 0.99, 0.99]),
    )

    assert results["samples"] == 2
    assert results["attack_samples"] == 1
    assert results["precision"] == 1.0
    assert results["recall"] == 1.0
    assert results["f1_score"] == 1.0
    assert results["confusion_matrix"] == {"TN": 1, "FP": 0, "FN": 0, "TP": 1}