import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_no_unsupported_rf_table_is_reported():
    """Ensure the repository only publishes metrics reproduced by its own scripts."""
    doc_paths = [
        ROOT_DIR / "README.md",
        ROOT_DIR / "results" / "README.md",
        ROOT_DIR / "results" / "final_results_summary.md",
    ]

    forbidden_markers = [
        "rf f1 = 0.9532",
        "roc-auc = 0.9922",
        "random forest f1 = 0.9532",
        "table 5: rf",
    ]

    for doc_path in doc_paths:
        text = doc_path.read_text(encoding="utf-8", errors="ignore").lower()
        for marker in forbidden_markers:
            assert marker not in text, f"Unsupported benchmark claim found in {doc_path.name}: {marker}"


def test_release_readme_metrics_match_result_artifacts():
    readme = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
    results = ROOT_DIR / "results"
    detection = pd.read_csv(results / "multiseed_detection_table.csv")

    def f1(adjacency: str, dataset: str) -> str:
        row = detection[(detection["adjacency"] == adjacency) & (detection["dataset"] == dataset)].iloc[0]
        return str(row["f1_score"])

    assert f1("A_CTOWN", "Dataset04") in readme
    assert f1("A_CTOWN", "Test Dataset") in readme
    assert f1("Identity", "Dataset04") in readme
    assert f1("Identity", "Test Dataset") in readme

    r4 = pd.read_csv(results / "fair_localization_table_r4.csv")
    oracle_e = r4[(r4["timing"] == "oracle") & (r4["method"] == "E_hydraulic_signature")].iloc[0]
    assert str(oracle_e["top1"]) in readme
    assert str(oracle_e["top3"]) in readme

    cpis = json.loads((results / "cpis_simulation_report.json").read_text(encoding="utf-8"))
    assert f"{cpis['weight_sensitivity']['kendall_tau_vs_default_mean']:.3f}" in readme
    synthetic = json.loads((results / "synthetic_attacks" / "synthetic_attack_pipeline_report.json").read_text(encoding="utf-8"))
    assert str(synthetic["scenario_count"]) in readme
    threshold = json.loads(
        (results / "test_inference" / "corrected" / "test_inference_report.json").read_text(encoding="utf-8")
    )["operational_threshold"]
    assert f"{threshold:.6f}" in readme

    synthetic = json.loads(
        (results / "synthetic_attacks" / "synthetic_attack_pipeline_report.json").read_text(encoding="utf-8")
    )
    master = pd.read_csv(results / "synthetic_attack_ground_truth.csv")
    assert synthetic["scenario_count"] == 71
    assert len(master) == synthetic["scenario_count"]
    case = next(
        scenario for scenario in synthetic["scenarios"]
        if scenario["scenario_id"] == synthetic["recommended_case_study"]
    )
    assert case["detection"]["mean_recall"] > 0.0
    assert case["recovery"]["status"] != "NOT_DETECTED"
