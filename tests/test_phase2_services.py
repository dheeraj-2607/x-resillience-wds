from __future__ import annotations

import networkx as nx
import pytest

from src.impact.topology import analyze_topology_and_impact
from src.localization.sensor_mapping import map_sensors_to_components, resolve_sensor_component
from src.localization.zone_embeddings import compute_zone_embeddings
from src.recovery.mitigation import get_operational_mitigation, get_scenario_name
from src.root_cause.assessment import assess_explanation_root_cause, rank_upstream_components


def _synthetic_network() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edge("J1", "J2", id="PU1", type="pump")
    graph.add_edge("J1", "J3", id="P1", type="pipe")
    graph.add_edge("J3", "J2", id="P2", type="pipe")
    graph.add_edge("J1", "J4", id="P3", type="pipe")
    return graph


def test_localization_maps_sensor_prefix_and_explicit_component_nodes():
    component_map = {"F_PU1": ["J1", "J2"]}

    mapped = resolve_sensor_component("F_PU1", component_map)
    multiple = map_sensors_to_components(["P_J1", "L_T1"], component_map)

    assert (mapped["component_type"], mapped["component_id"]) == ("Pump", "PU1")
    assert mapped["mapped_nodes"] == ["J1", "J2"]
    assert [entry["component_type"] for entry in multiple] == ["Junction", "Tank"]


def test_zone_embeddings_rank_attribution_mass_and_split_link_membership():
    import pandas as pd

    explanation = {
        "df_shap": pd.DataFrame(
            [
                {"Sensor": "F_PU1", "Abs SHAP": 8.0},
                {"Sensor": "P_J2", "Abs SHAP": 2.0},
                {"Sensor": "F_V1", "Abs SHAP": 1.0},
            ]
        ),
        "df_lime": pd.DataFrame(
            [
                {"Sensor": "F_PU1", "Abs LIME": 6.0},
                {"Sensor": "P_J2", "Abs LIME": 2.0},
                {"Sensor": "F_V1", "Abs LIME": 1.0},
            ]
        ),
    }
    sensor_nodes = {"F_PU1": ["J1", "J3"], "P_J2": ["J2"], "F_V1": ["J4"]}
    dma_zones = {"J1": 1, "J2": 1, "J3": 2, "J4": 2}

    result = compute_zone_embeddings(explanation, sensor_nodes, dma_zones)

    assert result["top_zone"] == 1
    assert [row["zone_id"] for row in result["rankings"]] == [1, 2]
    assert result["rankings"][0]["mapped_sensor_weight"] == 1.5
    assert abs(sum(row["localization_score"] for row in result["rankings"]) - 1.0) < 1e-6
    assert len(result["rankings"][0]["embedding"]) == len(result["embedding_features"]) == 8


def test_zone_embedding_handles_zero_lime_mass():
    import pandas as pd

    result = compute_zone_embeddings(
        {
            "df_shap": pd.DataFrame([{"Sensor": "P_J1", "Abs SHAP": 2.0}]),
            "df_lime": pd.DataFrame([{"Sensor": "P_J1", "Abs LIME": 0.0}]),
        },
        {"P_J1": ["J1"]},
        {"J1": 3},
    )

    assert result["top_zone"] == 3
    assert result["rankings"][0]["localization_score"] == 1.0


def test_topology_impact_finds_neighbors_dma_and_alternate_path():
    graph = _synthetic_network()
    result = analyze_topology_and_impact(
        graph,
        net=None,
        target_sensor="F_PU1",
        sensor_nodes_map={"P_J1": ["J1"], "P_J3": ["J3"]},
        component_map={},
        dma_zones={"J1": 1, "J2": 1},
    )

    assert result["comp_type"] == "Pump"
    assert result["endpoints"] == ["J1", "J2"]
    assert result["nearby_sensors"] == ["P_J1", "P_J3"]
    assert result["dma_region"] == "DMA 1"
    assert result["has_reroute"] is True
    assert result["reroute_path"] == ["J1", "J3", "J2"]


def test_root_cause_candidate_comes_from_shap_not_demo_injection():
    import pandas as pd

    selected_demo_sensor = "F_PU1"
    explanation = {
        "df_shap": pd.DataFrame(
            [
                {"Sensor": "F_PU2", "SHAP Value": 8.0, "Abs SHAP": 8.0, "Direction": "Increased anomaly score"},
                {"Sensor": "F_PU1", "SHAP Value": 1.0, "Abs SHAP": 1.0, "Direction": "Increased anomaly score"},
            ]
        ),
        "df_lime": pd.DataFrame({"Sensor": ["F_PU2", "P_J1"], "Abs LIME": [3.0, 2.0]}),
        "xai_mapping": [
            {"Sensor": "F_PU2", "Physical Component": "Pump PU2", "Topology Region": "DMA / Node J2"}
        ],
        "agreement": "Moderate",
        "overlap_count": 2,
    }
    assessment = assess_explanation_root_cause(explanation, 4.0, 3.0)
    inconclusive = assess_explanation_root_cause(explanation, 2.0, 3.0)

    assert assessment["detected"] is True
    assert assessment["candidate_sensor"] == "F_PU2"
    assert assessment["candidate_sensor"] != selected_demo_sensor
    assert assessment["candidate_component"] == "Pump PU2"
    assert assessment["lime_supports_candidate"] is True
    assert assessment["confidence"] == "Moderate"
    assert assessment["evidence"]
    assert assessment["scope_note"] == "Explanation-derived candidate only; not a confirmed real-world root cause."
    assert "injected_cause" not in assessment
    assert inconclusive["detected"] is False
    assert inconclusive["confidence"] == "Inconclusive"


def test_root_cause_assessment_rejects_missing_explanation():
    with pytest.raises(ValueError, match="df_shap"):
        assess_explanation_root_cause({}, 4.0, 3.0)


def test_reversed_flow_pagerank_ranks_upstream_asset_and_path():
    graph = nx.DiGraph()
    graph.add_node("J1", type="junction")
    graph.add_node("J2", type="junction")
    graph.add_node("T1", type="tank")
    graph.add_edge("J1", "J2", id="PU1", type="pump")
    graph.add_edge("J2", "T1", id="P1", type="pipe")

    result = rank_upstream_components(
        graph,
        {"P_J2": ["J2"]},
        {"P_J2": 2.0},
        {"J1": 1, "J2": 1, "T1": 1},
    )

    assert result["candidate_component"] == "Pump PU1"
    assert result["affected_dma_zone"] == 1
    assert result["propagation_path"]["nodes"] == ["J2", "J1"]
    assert result["propagation_path"]["links"] == ["PU1"]


def test_recovery_recommendations_are_category_specific_and_validated():
    pump_actions = get_operational_mitigation("Pump Flow")
    pressure_actions = get_operational_mitigation("Junction Pressure")

    assert pump_actions and pressure_actions
    assert pump_actions != pressure_actions
    assert get_scenario_name("Pump Flow", "Flow Increase") == "Simulated Pump Flow Manipulation"
    with pytest.raises(ValueError, match="Unsupported"):
        get_operational_mitigation("Unknown")