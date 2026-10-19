"""Correctness tests for src/evaluation (these test the code, not the research outcome)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.data_prep import harmonise_precision
from src.evaluation.labels import (
    TARGET_OFFSET, attacked_components, interval_labels, sensor_component, target_timestamps,
)
from src.evaluation.shapley import permutation_shapley


def test_shapley_is_exact_for_additive_game_and_sums_to_total():
    weights = np.array([3.0, -1.0, 0.5, 2.0])
    value = lambda masks: masks @ weights
    phi = permutation_shapley(value, 4, n_permutations=3, rng=np.random.default_rng(1))
    assert np.allclose(phi, weights)


def test_shapley_efficiency_for_interaction_game():
    value = lambda m: m[:, 0] * m[:, 1] * 4.0 + m[:, 2]
    phi = permutation_shapley(value, 3, n_permutations=20, rng=np.random.default_rng(0))
    assert np.isclose(phi.sum(), 5.0)          # v(all) - v(empty)
    assert np.isclose(phi[0], phi[1])          # symmetric players


def test_target_timestamps_use_last_horizon_offset():
    frame = pd.DataFrame({"DATETIME": pd.date_range("2017-01-01", periods=40, freq="h")})
    ts = target_timestamps(frame)
    assert TARGET_OFFSET == 23 and len(ts) == 17 and ts.iloc[0] == frame.DATETIME.iloc[23]


def test_interval_labels_are_inclusive():
    ts = pd.Series(pd.date_range("2017-02-24 03:00", periods=6, freq="h"))
    attacks = pd.DataFrame({"start_time": ["2017-02-24 05:00"], "end_time": ["2017-02-24 07:00"]})
    assert interval_labels(ts, attacks).tolist() == [0, 0, 1, 1, 1, 0]


def test_sensor_component_mapping():
    assert sensor_component("L_T7") == "Tank T7"
    assert sensor_component("F_PU10") == "Pump PU10"
    assert sensor_component("S_V2") == "Valve V2"
    assert sensor_component("P_J280") is None


def test_attacked_components_from_affected_sensors():
    row = pd.Series({"component_type": "Tank", "target_component": "T7",
                     "affected_sensors": "L_T7;F_PU10;S_PU10;P_J14"})
    assert attacked_components(row) == {"Tank T7", "Pump PU10"}


def test_harmonise_precision_rounds_only_sensor_columns():
    frame = pd.DataFrame({"DATETIME": ["a"], "P_J280": [2.9764], "ATT_FLAG": [0]})
    out = harmonise_precision(frame, ["P_J280"])
    assert out.P_J280.item() == 2.98 and out.ATT_FLAG.item() == 0


def test_ground_truth_attack_12_matches_official_labels():
    gt = pd.read_csv("data/ground_truth_attacks.csv")
    assert gt.loc[gt.attack_id == 12, "end_time"].item() == "2017-02-28 08:00"
