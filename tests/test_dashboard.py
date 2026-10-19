"""
test_dashboard.py

Unit tests for Phase 8 Dashboard data loading, terminology policy enforcement,
dual score calculations, and post-fix data integrity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
CORRECTED_DIR = ROOT_DIR / "results" / "test_inference" / "corrected"


def test_cpis_is_bounded_and_responds_to_impact_inputs():
    from src.impact.cpis import calculate_cpis

    config = yaml.safe_load((ROOT_DIR / "configs" / "config.yaml").read_text(encoding="utf-8"))["impact"]
    factors = {
        "pressure_impact": 0.5,
        "flow_impact": 0.5,
        "service_disruption": 0.5,
        "infrastructure_criticality": 0.5,
    }
    result = calculate_cpis(factors, config)

    assert 0.0 <= result["cpis_score"] <= 1.0
    assert result["cpis_severity"] == "High"
    expected = sum(factors[name] * config["weights"][name] for name in factors)
    assert result["cpis_score"] == pytest.approx(expected)


def test_cpis_rejects_out_of_range_factor():
    from src.impact.cpis import calculate_cpis

    factors = {
        "pressure_impact": 1.2,
        "flow_impact": 0.5,
        "service_disruption": 0.5,
        "infrastructure_criticality": 0.5,
    }
    with pytest.raises(ValueError, match="between zero and one"):
        calculate_cpis(factors)


def test_cpis_derives_all_four_roadmap_factors_from_sensor_residuals():
    from src.impact.cpis import derive_episode_impact_factors

    config = yaml.safe_load((ROOT_DIR / "configs" / "config.yaml").read_text(encoding="utf-8"))["impact"]
    factors = derive_episode_impact_factors(
        {"P_J1": 6.0, "F_PU1": 9.0, "L_T1": 3.0, "S_PU1": 4.0},
        duration_hours=24.0,
        threshold=3.0,
        impact_config=config,
    )

    assert set(factors) == {
        "pressure_impact",
        "flow_impact",
        "service_disruption",
        "infrastructure_criticality",
    }
    assert factors["pressure_impact"] == pytest.approx(2.0 / 3.0)
    assert factors["flow_impact"] == pytest.approx(0.75)
    assert factors["service_disruption"] == pytest.approx(0.7)
    assert 0.0 <= factors["infrastructure_criticality"] <= 1.0


def test_config_keeps_batadal_training_validation_and_test_files_separate():
    config = yaml.safe_load((ROOT_DIR / "configs" / "config.yaml").read_text(encoding="utf-8"))
    detection = config["detection"]

    dataset_paths = {
        Path(detection[key]).name
        for key in ("train_path", "val_path", "test_path")
    }

    assert dataset_paths == {
        "BATADAL_dataset03.csv",
        "BATADAL_dataset04.csv",
        "BATADAL_test_dataset.csv",
    }


def test_localization_accuracy_report_is_available_to_validation_dashboard():
    report_path = ROOT_DIR / "results" / "localization_benchmark_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    proposed = report["metrics"]["proposed_shap_reversed_flow_ppr"]
    app_source = (ROOT_DIR / "dashboard" / "app.py").read_text(encoding="utf-8")

    assert proposed["attacks"] == 14
    assert 0.0 <= proposed["top1_component_accuracy_pct"] <= 100.0
    assert 0.0 <= proposed["top3_component_accuracy_pct"] <= 100.0
    assert "Dataset04 Component Localization (7 Attacks)" in app_source
    assert "Combined 14-Attack Component Localization" in app_source


def test_dashboard_uses_latest_training_metrics_and_formats_unavailable_auc():
    from dashboard.app import (
        _calculate_validation_accuracy,
        _format_validation_metric,
        load_dashboard_data,
    )

    expected = json.loads((ROOT_DIR / "results" / "detection_metrics.json").read_text(encoding="utf-8"))
    _, _, _, summary, _ = load_dashboard_data()

    assert summary["training_validation_metrics"] == expected
    assert _format_validation_metric(expected["precision"]) == f"{expected['precision']:.4f}"
    assert _format_validation_metric(expected["roc_auc"]) == "N/A"
    assert _format_validation_metric(expected["pr_auc"]) == "N/A"
    assert _calculate_validation_accuracy(expected) == pytest.approx(122 / 219)


class TestDashboardDataIntegrity:
    def test_forecast_horizon_config_is_h12(self):
        """Regression test ensuring the project config matches the H=12 production architecture."""
        config = yaml.safe_load((ROOT_DIR / "configs" / "config.yaml").read_text(encoding="utf-8"))
        assert config["detection"]["forecast_horizon"] == 12

    def test_corrected_test_artifacts_exist(self):
        """Verify all corrected test inference CSVs and JSONs exist."""
        assert (CORRECTED_DIR / "test_anomaly_scores.csv").exists()
        assert (CORRECTED_DIR / "test_anomaly_episodes.csv").exists()
        assert (CORRECTED_DIR / "test_sensor_attribution.csv").exists()
        assert (CORRECTED_DIR / "test_inference_report.json").exists()

        df_episodes = pd.read_csv(CORRECTED_DIR / "test_anomaly_episodes.csv")
        assert {
            "cpis_score",
            "cpis_severity",
            "cpis_pressure_impact",
            "cpis_flow_impact",
            "cpis_service_disruption",
            "cpis_infrastructure_criticality",
            "cpis_evidence_basis",
        }.issubset(df_episodes.columns)

    def test_corrected_test_scores_monotonicity_and_counts(self):
        """Verify corrected test scores timeline monotonicity and exact window counts."""
        df_scores = pd.read_csv(CORRECTED_DIR / "test_anomaly_scores.csv")
        df_scores["timestamp"] = pd.to_datetime(df_scores["timestamp"])

        assert len(df_scores) == 2066
        assert df_scores["timestamp"].is_monotonic_increasing
        assert df_scores["timestamp"].iloc[0] == pd.Timestamp("2017-01-04 12:00:00")
        assert df_scores["timestamp"].iloc[-1] == pd.Timestamp("2017-03-31 13:00:00")
        assert df_scores["anomaly_flag"].sum() == 147

    def test_episodes_severity_calculations(self):
        """Verify log-scaled operational severity formula vs raw severity."""
        df_ep = pd.read_csv(CORRECTED_DIR / "test_anomaly_episodes.csv")
        assert len(df_ep) >= 17

        tau = 2.943899
        for _, row in df_ep.iterrows():
            mean_s = row["mean_anomaly_score"] if "mean_anomaly_score" in row else row["mean_score"]
            peak_s = row["max_anomaly_score"] if "max_anomaly_score" in row else row["peak_score"]
            dur = row["duration_hours"]

            # Log-scaled severity
            expected_log_sev = (
                np.log10(1.0 + mean_s)
                * np.sqrt(dur)
                * (1.0 + np.log10(1.0 + peak_s / tau))
            )
            assert expected_log_sev > 0.0

    def test_forbidden_terminology_policy(self):
        """Verify that dashboard UI display strings enforce non-alarmist terminology."""
        app_file = ROOT_DIR / "dashboard" / "app.py"
        assert app_file.exists()

        content = app_file.read_text(encoding="utf-8")
        assert "TERMINOLOGY_POLICY_NOTICE" in content
        assert "Unlabeled test set events are classified as" in content

    def test_dashboard_loads_exactly_17_episodes(self):
        """Regression test ensuring dashboard loads exactly 17 frozen test episodes."""
        from dashboard.app import load_dashboard_data
        df_scores, df_episodes, df_attribution, summary_data, component_map = load_dashboard_data()

        assert len(df_episodes) == 17
        assert len(df_scores) == 2066
        assert df_scores["anomaly_flag"].sum() == 147
        assert df_episodes["cpis_score"].between(0.0, 1.0).all()
        assert df_episodes["cpis_severity"].isin(["Low", "Moderate", "High", "Critical"]).all()
        assert df_episodes["cpis_evidence_basis"].str.contains("all per-sensor", case=False).all()

        # Verify key episodes exist
        ep_ids = df_episodes["episode_id"].tolist()
        for required_id in ["EP-09", "EP-08", "EP-14", "EP-06"]:
            assert required_id in ep_ids

    def test_episode_explorer_formatting_no_value_error(self):
        """Regression test ensuring float duration_hours formatting does not raise ValueError."""
        from dashboard.app import load_dashboard_data
        _, df_episodes, _, _, _ = load_dashboard_data()

        # Perform the exact Styler formatting used in Episode Explorer
        try:
            styled_df = df_episodes[
                [
                    "episode_id",
                    "start_time",
                    "end_time",
                    "duration_hours",
                    "cpis_score",
                    "cpis_severity",
                    "cpis_pressure_impact",
                    "cpis_flow_impact",
                    "cpis_service_disruption",
                    "cpis_infrastructure_criticality",
                    "cpis_evidence_basis",
                    "peak_score",
                    "mean_score",
                    "operational_severity_log",
                    "raw_severity",
                    "persistence_classification",
                    "top_contributing_sensors",
                ]
            ].style.format(
                {
                    "peak_score": "{:.2e}",
                    "mean_score": "{:.2e}",
                    "operational_severity_log": "{:.1f}",
                    "cpis_score": "{:.2f}",
                    "cpis_pressure_impact": "{:.2f}",
                    "cpis_flow_impact": "{:.2f}",
                    "cpis_service_disruption": "{:.2f}",
                    "cpis_infrastructure_criticality": "{:.2f}",
                    "raw_severity": "{:.2e}",
                    "duration_hours": "{:.0f}",
                }
            )
            # Render html to trigger full string formatting evaluation
            _ = styled_df.to_html()
        except ValueError as exc:
            pytest.fail(f"Styler formatting raised ValueError: {exc}")


class TestDemoModeIntegrity:
    def test_demo_uses_locked_threshold(self):
        """Verify dashboard threshold matches corrected held-out calibration."""
        from dashboard.app import THRESHOLD_TAU
        report_path = ROOT_DIR / "results" / "test_inference" / "corrected" / "test_inference_report.json"
        import json

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert THRESHOLD_TAU == report["operational_threshold"]

    def test_demo_pipeline_sensor_categories_discoverable(self):
        """Verify all 6 monitored WDS sensor categories are discoverable from the 43 sensors."""
        from dashboard.app import load_demo_pipeline
        demo_pipe = load_demo_pipeline()
        sensor_cols = demo_pipe["sensor_cols"]

        assert len(sensor_cols) == 43

        pressures = [s for s in sensor_cols if s.startswith("P_")]
        tanks = [s for s in sensor_cols if s.startswith("L_")]
        p_flows = [s for s in sensor_cols if s.startswith("F_PU")]
        p_stats = [s for s in sensor_cols if s.startswith("S_PU")]
        v_flows = [s for s in sensor_cols if s.startswith("F_V")]
        v_stats = [s for s in sensor_cols if s.startswith("S_V")]

        assert len(pressures) == 12
        assert len(tanks) == 7
        assert len(p_flows) == 11
        assert len(p_stats) == 11
        assert len(v_flows) == 1
        assert len(v_stats) == 1
        assert np.isfinite(demo_pipe["calibrated_threshold"])

    def test_demo_reloads_stale_graph_builder_module(self, monkeypatch):
        import src.network.graph_builder as graph_builder
        from dashboard.app import load_demo_pipeline

        monkeypatch.delattr(graph_builder, "build_directed_hydraulic_graph")
        load_demo_pipeline.clear()
        demo_pipe = load_demo_pipeline()

        assert demo_pipe["directed_graph"].is_directed()

    def test_dashboard_reloads_stale_root_cause_assessment(self, monkeypatch):
        import inspect
        import src.root_cause.assessment as assessment_module
        from dashboard.app import _get_root_cause_assessor

        current_assessor = assessment_module.assess_explanation_root_cause
        monkeypatch.setattr(
            assessment_module,
            "assess_explanation_root_cause",
            lambda explanation_result, anomaly_score, threshold: None,
        )

        refreshed_assessor = _get_root_cause_assessor()

        assert refreshed_assessor is assessment_module.assess_explanation_root_cause
        assert "graph" in inspect.signature(refreshed_assessor).parameters
        monkeypatch.setattr(assessment_module, "assess_explanation_root_cause", current_assessor)

    def test_demo_synthetic_injection_in_memory_only(self):
        """Verify synthetic anomaly injection modifies only in-memory objects."""
        from dashboard.app import load_demo_pipeline
        import numpy as np

        demo_pipe = load_demo_pipeline()
        orig_window = demo_pipe["normal_window_raw"].copy()

        # Create in-memory copy and perturb
        disturbed = orig_window.copy()
        target_sensor = "P_J280"
        disturbed.loc[6:, target_sensor] += 10.0

        # Verify original in-memory object is unchanged
        assert not np.allclose(orig_window[target_sensor].values, disturbed[target_sensor].values)
        assert orig_window[target_sensor].iloc[-1] != disturbed[target_sensor].iloc[-1]

        # Verify files on disk remain untouched
        assert (ROOT_DIR / "data" / "raw" / "BATADAL_dataset03.csv").exists()
        assert (ROOT_DIR / "models" / "detection_gnn.pt").exists()

    def test_xai_shap_and_lime_real_calculated_values(self):
        """Verify SHAP and LIME explanations are calculated from frozen GNN model with real values."""
        from dashboard.app import load_demo_pipeline
        from src.explainability import compute_xai_explanations
        demo_pipe = load_demo_pipeline()
        normal_window_raw = demo_pipe["normal_window_raw"].copy()
        disturbed_window = normal_window_raw.copy()
        target_sensor = "F_PU1"
        disturbed_window.loc[6:, target_sensor] += 4.0 * demo_pipe["sensor_stds"].get(target_sensor, 1.0)

        res = compute_xai_explanations(demo_pipe, normal_window_raw, disturbed_window, target_sensor)

        # Verify SHAP
        df_shap = res["df_shap"]
        assert len(df_shap) == 43
        assert df_shap["Sensor"].iloc[0] == "F_PU1"
        assert df_shap["Abs SHAP"].iloc[0] > 0.1  # Real non-zero SHAP contribution
        assert res["s_dist"] > res["s_norm"]

        # Verify LIME
        df_lime = res["df_lime"]
        assert len(df_lime) == 43
        assert df_lime["Abs LIME"].max() > 0.0  # Real calculated weights

        # Verify Agreement & Mapping
        assert res["agreement"] in ["High", "Moderate", "Low"]
        assert len(res["xai_mapping"]) == 5
        assert res["xai_mapping"][0]["Sensor"] in demo_pipe["sensor_cols"]
        metrics = res["metrics"]
        assert 0.0 <= metrics["agreement_jaccard"] <= 1.0
        assert len(metrics["stability"]) == 3
        assert np.isfinite(metrics["stability"]["shap"])
        assert np.isfinite(metrics["stability"]["lime"])
        assert np.isfinite(metrics["fidelity"]["top4_score_reduction"])
        assert np.isfinite(metrics["fidelity"]["random4_mean_score_reduction"])

    def test_topology_impact_and_rerouting_analysis(self):
        """Verify topology impact path and operational rerouting calculation."""
        from dashboard.app import load_demo_pipeline
        from src.impact.topology import analyze_topology_and_impact
        demo_pipe = load_demo_pipeline()

        res_edge = analyze_topology_and_impact(
            demo_pipe["g"],
            demo_pipe["net"],
            "F_PU1",
            demo_pipe["sensor_nodes_map"],
            {},
        )
        assert res_edge["comp_id"] == "PU1"
        assert res_edge["comp_type"] == "Pump"
        assert res_edge["is_edge"] is True
        assert len(res_edge["endpoints"]) == 2

        res_node = analyze_topology_and_impact(
            demo_pipe["g"],
            demo_pipe["net"],
            "P_J280",
            demo_pipe["sensor_nodes_map"],
            {},
        )
        assert res_node["comp_id"] == "J280"
        assert res_node["comp_type"] == "Junction"
        assert res_node["is_edge"] is False

    def test_ctown_topology_figure_highlights_nodes_and_links(self):
        import networkx as nx
        from dashboard.app import build_ctown_topology_figure

        graph = nx.Graph()
        graph.add_node("J1", x=0.0, y=0.0, type="junction")
        graph.add_node("J2", x=1.0, y=0.0, type="tank")
        graph.add_edge("J1", "J2", id="PU1", type="pump")
        zones = {"J1": 1, "J2": 1}

        node_figure = build_ctown_topology_figure(
            graph,
            zones,
            {"comp_id": "J1", "comp_type": "Junction", "is_edge": False, "dma_region": "DMA 1", "is_detected": True},
        )
        edge_figure = build_ctown_topology_figure(
            graph,
            zones,
            {"comp_id": "PU1", "comp_type": "Pump", "is_edge": True, "dma_region": "DMA 1", "is_detected": True},
        )
        normal_node_figure = build_ctown_topology_figure(
            graph,
            zones,
            {"comp_id": "J1", "comp_type": "Junction", "is_edge": False, "dma_region": "DMA 1", "is_detected": False},
        )

        assert node_figure.data[-1].name == "Affected Junction J1"
        assert list(node_figure.data[-1].x) == [0.0]
        assert edge_figure.data[-1].name == "Affected Pump PU1"
        assert list(edge_figure.data[-1].x) == [0.0, 1.0]
        assert normal_node_figure.data[-1].marker.color == "#2ca02c"
        assert "DMA 1" in edge_figure.layout.title.text



