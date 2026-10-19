from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from src.evaluation.fair_localization import METHODS, evaluate_model


def test_method_e_is_ranked_and_masks_spoofed_asset_sensor():
    graph = nx.DiGraph()
    graph.add_node("T1", type="tank")
    graph.add_nodes_from(["J1", "J2", "J3"])
    graph.add_edge("J1", "J2", type="pump", id="PU1", weight=1.0)
    graph.add_edge("J2", "J3", type="valve", id="V2", weight=1.0)
    sensors = ["L_T1", "F_PU1", "F_V2"]
    sensor_nodes = {"L_T1": {"T1"}, "F_PU1": {"J1", "J2"}, "F_V2": {"J2", "J3"}}
    timestamps = pd.Series([pd.Timestamp("2017-01-16 12:00")])
    windows = np.ones((1, 12, 3), dtype=np.float32)
    targets = np.zeros_like(windows)
    attacks = pd.DataFrame([{
        "dataset": "Test Dataset",
        "attack_id": 8,
        "attack_type": "Sensor Spoofing/Concealment",
        "start_time": "2017-01-16 00:00",
        "end_time": "2017-01-17 00:00",
        "target_component": "T1",
        "component_type": "Tank",
        "affected_sensors": "L_T1",
    }])
    assets = ["Tank T1", "Pump PU1", "Valve V2"]
    signatures = {
        "signatures": np.array([[1.0, 0.2, 0.1], [0.1, 1.0, 0.2], [0.1, 0.2, 1.0]]),
        "asset_labels": assets,
        "sensors": sensors,
    }

    records = evaluate_model(
        predict_fn=lambda values: values * 0.5,
        variance=np.ones((12, 3)),
        threshold=0.0,
        calibration_windows=np.zeros((2, 12, 3), dtype=np.float32),
        datasets={"Test Dataset": (windows, targets, timestamps)},
        attacks=attacks,
        sensors=sensors,
        directed_graph=graph,
        sensor_nodes=sensor_nodes,
        dma_zones={},
        hydraulic_signatures=signatures,
        seed=42,
        n_permutations=1,
    )

    assert "E_hydraulic_signature" in METHODS
    assert len(records) == 2
    assert all(record["E_masked_sensor_channels"] == ["L_T1"] for record in records)
    assert all(len(record["E_hydraulic_cosine_scores"]) == 3 for record in records)
    assert all("E_hydraulic_signature_top3" in record for record in records)
