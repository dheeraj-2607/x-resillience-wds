"""Phase 1 exit gate for BATADAL component-localization benchmarking."""

from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT_DIR / "results" / "localization_benchmark_report.json"
PROPOSED = "proposed_shap_reversed_flow_ppr"
BASELINES = ("baseline_a_raw_deviation", "baseline_b_top1_shap")


def test_proposed_localization_beats_both_baselines_on_top1_and_top3():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    metrics = report["metrics"]
    proposed = metrics[PROPOSED]

    assert proposed["attacks"] == 14
    for baseline_name in BASELINES:
        baseline = metrics[baseline_name]
        assert baseline["attacks"] == 14
        assert proposed["top1_component_accuracy_pct"] > baseline["top1_component_accuracy_pct"]
        assert proposed["top3_component_accuracy_pct"] > baseline["top3_component_accuracy_pct"]


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
