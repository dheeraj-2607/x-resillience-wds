"""Explanation-derived root-cause candidate assessment."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx


def rank_upstream_components(
    graph: nx.DiGraph,
    sensor_nodes: Mapping[str, Sequence[str]],
    shap_attributions: Mapping[str, float],
    dma_zones: Mapping[str, int] | None = None,
    alpha: float = 0.5,
    direct_shap_weight: float = 0.75,
) -> dict[str, Any]:
    """Rank upstream pumps, valves, and tanks using reversed-flow PPR."""
    if not graph.is_directed():
        raise ValueError("rank_upstream_components requires a directed hydraulic graph.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    if not 0.0 <= direct_shap_weight <= 1.0:
        raise ValueError("direct_shap_weight must be between 0 and 1.")

    expanded_graph = nx.DiGraph()
    expanded_graph.add_nodes_from((node, dict(data)) for node, data in graph.nodes(data=True))
    asset_graph_nodes: dict[tuple[str, str], str] = {}
    asset_endpoints: dict[tuple[str, str], tuple[str, str]] = {}
    link_asset_nodes: dict[str, str] = {}
    for source, destination, data in graph.edges(data=True):
        component_type = str(data.get("type", "")).title()
        component_id = str(data.get("id", f"{source}-{destination}"))
        edge_data = dict(data)
        if component_type in {"Pump", "Valve"}:
            asset_node = f"__asset__:{component_type}:{component_id}"
            expanded_graph.add_node(
                asset_node,
                type=component_type.lower(),
                component_id=component_id,
                virtual_asset_node=True,
            )
            asset_graph_nodes[(component_type, component_id)] = asset_node
            asset_endpoints[(component_type, component_id)] = (source, destination)
            link_asset_nodes[component_id] = asset_node
            expanded_graph.add_edge(source, asset_node, **edge_data)
            expanded_graph.add_edge(asset_node, destination, **edge_data)
        else:
            expanded_graph.add_edge(source, destination, **edge_data)

    personalization = {node: 0.0 for node in expanded_graph}
    direct_asset_attributions: Counter[tuple[str, str]] = Counter()
    evidence_nodes_by_sensor: dict[str, list[str]] = {}
    for sensor, attribution in shap_attributions.items():
        weight = abs(float(attribution))
        if not math.isfinite(weight):
            raise ValueError(f"SHAP attribution for {sensor!r} must be finite.")
        connected_nodes = [node for node in sensor_nodes.get(sensor, ()) if node in graph]
        asset_id = None
        if sensor.startswith(("F_PU", "S_PU", "F_V", "S_V")):
            asset_id = sensor.split("_", 1)[1]
        mapped_asset_node = link_asset_nodes.get(asset_id) if asset_id else None
        if mapped_asset_node is not None:
            connected_nodes = [mapped_asset_node]
        evidence_nodes_by_sensor[sensor] = connected_nodes
        if weight and connected_nodes:
            share = weight / len(connected_nodes)
            for node in connected_nodes:
                personalization[node] += share
        if sensor.startswith("L_"):
            direct_asset_attributions[("Tank", sensor[2:])] += weight
        elif sensor.startswith(("F_PU", "S_PU")):
            direct_asset_attributions[("Pump", sensor.split("_", 1)[1])] += weight
        elif sensor.startswith(("F_V", "S_V")):
            direct_asset_attributions[("Valve", sensor.split("_", 1)[1])] += weight

    total_mass = math.fsum(personalization.values())
    if total_mass <= 0.0:
        raise ValueError("At least one nonzero SHAP attribution must map to the hydraulic graph.")
    personalization = {node: value / total_mass for node, value in personalization.items()}
    reversed_graph = expanded_graph.reverse(copy=True)
    pagerank = nx.pagerank(
        reversed_graph,
        alpha=alpha,
        personalization=personalization,
        dangling=personalization,
        weight="weight",
    )

    candidates = []
    for component, data in expanded_graph.nodes(data=True):
        if data.get("type") == "tank":
            candidates.append(
                {
                    "component_id": str(component),
                    "component_type": "Tank",
                    "score": float(pagerank[component]),
                    "nodes": [component],
                    "graph_node": component,
                }
            )
    for (component_type, component_id), graph_node in asset_graph_nodes.items():
        source, destination = asset_endpoints[(component_type, component_id)]
        candidates.append(
            {
                "component_id": component_id,
                "component_type": component_type,
                "score": float(pagerank[graph_node]),
                "nodes": [source, destination],
                "upstream_node": source,
                "downstream_node": destination,
                "graph_node": graph_node,
            }
        )
    if not candidates:
        raise ValueError("The directed hydraulic graph has no pump, valve, or tank candidates.")
    ppr_max = max(candidate["score"] for candidate in candidates)
    direct_max = max(
        direct_asset_attributions[(candidate["component_type"], candidate["component_id"])]
        for candidate in candidates
    )
    for candidate in candidates:
        ppr_score = candidate["score"] / ppr_max if ppr_max else 0.0
        direct_score = (
            direct_asset_attributions[(candidate["component_type"], candidate["component_id"])]
            / direct_max
            if direct_max
            else 0.0
        )
        candidate["ppr_score"] = ppr_score
        candidate["direct_shap_mass"] = direct_asset_attributions[
            (candidate["component_type"], candidate["component_id"])
        ]
        candidate["score"] = (
            (1.0 - direct_shap_weight) * ppr_score + direct_shap_weight * direct_score
        )
    candidates.sort(key=lambda item: (-item["score"], item["component_type"], item["component_id"]))

    top_candidate = candidates[0]
    evidence_nodes = sorted(
        (node for node, weight in personalization.items() if weight > 0.0),
        key=lambda node: (-personalization[node], str(node)),
    )
    route_options = []
    route_target = top_candidate["graph_node"]
    for evidence_node in evidence_nodes:
        if nx.has_path(reversed_graph, evidence_node, route_target):
            path = nx.shortest_path(reversed_graph, evidence_node, route_target)
            route_options.append((len(path), -personalization[evidence_node], path))
    route_options.sort(key=lambda option: (option[0], option[1], [str(node) for node in option[2]]))
    node_path = route_options[0][2] if route_options else []
    if node_path and "upstream_node" in top_candidate:
        node_path = [*node_path, top_candidate["upstream_node"]]
    asset_node_to_name = {
        graph_node: f"{component_type} {component_id}"
        for (component_type, component_id), graph_node in asset_graph_nodes.items()
    }
    display_path = [asset_node_to_name.get(node, node) for node in node_path]
    link_path = []
    for source, destination in zip(node_path, node_path[1:]):
        link_id = reversed_graph[source][destination].get("id")
        if link_id and (not link_path or link_path[-1] != link_id):
            link_path.append(link_id)
    if top_candidate["component_type"] in {"Pump", "Valve"}:
        if not link_path or link_path[-1] != top_candidate["component_id"]:
            link_path.append(top_candidate["component_id"])

    zone_candidates = [
        int(dma_zones[node])
        for node in top_candidate["nodes"]
        if dma_zones is not None and node in dma_zones and int(dma_zones[node]) >= 0
    ]
    affected_zone = Counter(zone_candidates).most_common(1)[0][0] if zone_candidates else None
    return {
        "candidate_component": f"{top_candidate['component_type']} {top_candidate['component_id']}",
        "candidate_component_id": top_candidate["component_id"],
        "candidate_component_type": top_candidate["component_type"],
        "candidate_score": top_candidate["score"],
        "affected_dma_zone": affected_zone,
        "propagation_path": {"nodes": display_path, "links": link_path},
        "ranked_components": candidates,
        "pagerank": pagerank,
        "personalization": personalization,
    }


def assess_explanation_root_cause(
    explanation_result: dict[str, Any],
    anomaly_score: float,
    threshold: float,
    *,
    graph: nx.DiGraph | None = None,
    sensor_nodes: Mapping[str, Sequence[str]] | None = None,
    dma_zones: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Rank a candidate from SHAP output only; never accept an injected sensor.

    This reports the sensor most responsible for the model explanation, not a
    confirmed physical cause. Ground-truth demo injection details are excluded
    from both the API and returned assessment.
    """
    shap_frame = explanation_result.get("df_shap")
    lime_frame = explanation_result.get("df_lime")
    required_columns = {"Sensor", "SHAP Value", "Abs SHAP", "Direction"}
    if shap_frame is None or not required_columns.issubset(shap_frame.columns) or shap_frame.empty:
        raise ValueError("explanation_result.df_shap must contain ranked SHAP sensor contributions.")
    if not math.isfinite(float(anomaly_score)) or not math.isfinite(float(threshold)) or threshold <= 0:
        raise ValueError("anomaly_score must be finite and threshold must be finite and positive.")

    ranked_shap = shap_frame.sort_values("Abs SHAP", ascending=False)
    top = ranked_shap.iloc[0]
    candidate_sensor = str(top["Sensor"])
    shap_value = float(top["SHAP Value"])
    shap_magnitude = float(top["Abs SHAP"])
    if not math.isfinite(shap_value) or not math.isfinite(shap_magnitude):
        raise ValueError("SHAP contributions must be finite numbers.")

    mapping_by_sensor = {
        item.get("Sensor"): item for item in explanation_result.get("xai_mapping", [])
    }
    component_mapping = mapping_by_sensor.get(candidate_sensor, {})
    graph_result = None
    if graph is not None:
        if sensor_nodes is None:
            raise ValueError("sensor_nodes is required when graph-based PPR is enabled.")
        shap_attributions = {
            str(row["Sensor"]): float(row["SHAP Value"])
            for _, row in ranked_shap.iterrows()
        }
        graph_result = rank_upstream_components(graph, sensor_nodes, shap_attributions, dma_zones)
        component_mapping = {
            **component_mapping,
            "Physical Component": graph_result["candidate_component"],
        }
    lime_top_sensors = []
    if lime_frame is not None and "Sensor" in lime_frame.columns:
        lime_top_sensors = lime_frame["Sensor"].head(4).astype(str).tolist()
    lime_supports_candidate = candidate_sensor in lime_top_sensors
    detected = float(anomaly_score) >= float(threshold)

    evidence = [
        f"Anomaly score {float(anomaly_score):.4f} versus threshold {float(threshold):.4f}.",
        f"{candidate_sensor} has the largest absolute SHAP contribution ({shap_magnitude:.4g}; {top['Direction']}).",
        f"Top-four SHAP/LIME overlap: {int(explanation_result.get('overlap_count', 0))}/4.",
    ]
    if lime_top_sensors:
        evidence.append(
            "LIME " + ("also ranks" if lime_supports_candidate else "does not rank")
            + f" {candidate_sensor} in its top four."
        )
    if graph_result is not None:
        evidence.append(
            f"Reversed-flow PPR ranks {graph_result['candidate_component']} upstream of the attributed sensors."
        )

    if detected and shap_magnitude > 0:
        hypothesis = f"{candidate_sensor} is the strongest explanation-derived candidate for the observed statistical anomaly."
        confidence = "Moderate" if lime_supports_candidate else "SHAP-only; low agreement"
    else:
        hypothesis = f"{candidate_sensor} is the top explanation contributor, but the score does not establish an anomaly cause."
        confidence = "Inconclusive"

    result = {
        "detected": detected,
        "candidate_sensor": candidate_sensor,
        "candidate_component": component_mapping.get("Physical Component", "Unmapped component"),
        "candidate_topology_region": component_mapping.get("Topology Region", "Unmapped"),
        "shap_value": shap_value,
        "shap_magnitude": shap_magnitude,
        "lime_supports_candidate": lime_supports_candidate,
        "explanation_agreement": explanation_result.get("agreement", "Unavailable"),
        "evidence": evidence,
        "hypothesis": hypothesis,
        "confidence": confidence,
        "scope_note": "Explanation-derived candidate only; not a confirmed real-world root cause.",
    }
    if graph_result is not None:
        result.update(
            {
                "candidate_component": graph_result["candidate_component"],
                "candidate_component_id": graph_result["candidate_component_id"],
                "candidate_component_type": graph_result["candidate_component_type"],
                "candidate_component_score": graph_result["candidate_score"],
                "affected_dma_zone": graph_result["affected_dma_zone"],
                "propagation_path": graph_result["propagation_path"],
                "ranked_upstream_components": graph_result["ranked_components"],
            }
        )
    return result