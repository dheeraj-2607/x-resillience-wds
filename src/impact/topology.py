"""Topology footprint and alternate-path analysis for affected WDS components."""

from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd

from src.localization.sensor_mapping import resolve_sensor_component


def analyze_topology_and_impact(
    graph: nx.Graph,
    net: Any,
    target_sensor: str,
    sensor_nodes_map: dict[str, list[str]],
    component_map: dict[str, list[str]],
    dma_zones: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Resolve affected component, neighboring sensors, DMA, and alternate path."""
    resolved = resolve_sensor_component(target_sensor, component_map)
    component_id = resolved["component_id"]
    component_type = resolved["component_type"]
    is_edge = component_type in {"Pump", "Valve"}

    endpoints: list[str] = []
    neighbors: list[str] = []
    if is_edge:
        found_edge = next(
            ((start, end) for start, end, data in graph.edges(data=True) if data.get("id") == component_id),
            None,
        )
        if found_edge:
            start, end = found_edge
            endpoints = [start, end]
            neighbors = sorted((set(graph.neighbors(start)) | set(graph.neighbors(end))) - {start, end})
        else:
            endpoints = [component_id]
    elif component_id in graph:
        endpoints = [component_id]
        neighbors = sorted(graph.neighbors(component_id))

    footprint_nodes = set(endpoints + neighbors)
    nearby_sensors = sorted(
        sensor
        for sensor, nodes in sensor_nodes_map.items()
        if sensor != target_sensor and footprint_nodes.intersection(nodes)
    )

    zones = sorted(
        {
            dma_zones[node]
            for node in endpoints
            if dma_zones and node in dma_zones and dma_zones[node] > 0
        }
    )
    if zones:
        dma_region = " / ".join(f"DMA {zone}" for zone in zones)
        if len(zones) > 1:
            dma_region += " boundary"
    else:
        dma_region = "Unassigned DMA region"
        junctions = getattr(net, "junctions", None)
        if junctions is not None and "pattern" in junctions.columns:
            for endpoint in endpoints:
                match = junctions[junctions["id"] == endpoint]
                if not match.empty and pd.notna(match.iloc[0]["pattern"]):
                    pattern = str(match.iloc[0]["pattern"])
                    if "DMA" in pattern:
                        dma_region = f"DMA / {pattern.replace('_pat', '').upper()}"
                        break

    reroute_path: list[str] = []
    if is_edge and len(endpoints) == 2:
        start, end = endpoints
        alternate_graph = graph.copy()
        if alternate_graph.has_edge(start, end):
            alternate_graph.remove_edge(start, end)
        if nx.has_path(alternate_graph, start, end):
            path = nx.shortest_path(alternate_graph, start, end)
            if len(path) <= 12:
                reroute_path = path
    elif not is_edge and endpoints and len(neighbors) >= 2:
        alternate_graph = graph.copy()
        alternate_graph.remove_node(endpoints[0])
        start, end = neighbors[:2]
        if nx.has_path(alternate_graph, start, end):
            path = nx.shortest_path(alternate_graph, start, end)
            if len(path) <= 12:
                reroute_path = path

    return {
        "comp_id": component_id,
        "comp_type": component_type,
        "is_edge": is_edge,
        "endpoints": endpoints,
        "nbr_nodes": neighbors,
        "nearby_sensors": nearby_sensors,
        "dma_region": dma_region,
        "has_reroute": bool(reroute_path),
        "reroute_path": reroute_path,
        "reroute_path_str": " -> ".join(reroute_path)
        if reroute_path
        else "No validated alternative route identified from the available topology.",
        "affected_topology_nodes": sorted(footprint_nodes),
    }