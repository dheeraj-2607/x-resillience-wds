"""Phase 1 exit gate for BATADAL component-localization benchmarking."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT_DIR / "results" / "localization_benchmark_report.json"
PROPOSED = "proposed_shap_reversed_flow_ppr"
METHODS = {
    "baseline_a_raw_deviation",
    "baseline_b_top1_shap",
    PROPOSED,
}


def test_localization_report_has_complete_methods_and_valid_outputs():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    metrics = report["metrics"]
    rows = [
        row
        for dataset in report["by_dataset"].values()
        for row in dataset["attack_results"]
    ]

    assert set(metrics) == METHODS
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        assert metrics[method]["attacks"] == 14
        assert len(method_rows) == 14
        assert all(
            math.isfinite(row["topological_hop_distance"])
            and row["topological_hop_distance"] >= 0
            for row in method_rows
        )

    proposed_scores = [
        component["score"]
        for row in rows
        if row["method"] == PROPOSED
        for component in row["predicted_components"]
    ]
    assert proposed_scores
    assert all(math.isfinite(score) and 0.0 <= score <= 1.0 for score in proposed_scores)


def test_localization_report_contains_attack_type_breakdowns_and_paths():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    rows = [
        row
        for dataset in report["by_dataset"].values()
        for row in dataset["attack_results"]
        if row["method"] == PROPOSED
    ]

    assert len(rows) == 14
    assert {row["attack_type"] for row in rows} == {
        "Actuator Manipulation",
        "Sensor Spoofing/Concealment",
    }
    assert all("nodes" in row["ppr_propagation_path"] for row in rows)
    assert all("links" in row["ppr_propagation_path"] for row in rows)
