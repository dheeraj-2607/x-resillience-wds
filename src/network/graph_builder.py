"""
graph_builder.py

Builds a NetworkX graph representation of the parsed C-Town network:
nodes = junctions/tanks/reservoirs, edges = pipes/pumps/valves.

Status: IMPLEMENTED.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
import wntr

from src.network.ctown_loader import CTownNetwork

logger = logging.getLogger(__name__)

NODE_TYPES = {"junction", "tank", "reservoir"}
EDGE_TYPES = {"pipe", "pump", "valve"}


def build_network_graph(
    net: CTownNetwork,
    directed: bool = False,
    hydraulic_results: Any | None = None,
) -> nx.Graph | nx.DiGraph:
    """Build a C-Town graph, optionally orienting links by simulated flow.

    Node attributes: type ('junction'/'tank'/'reservoir'), plus hydraulic
    properties available for that component type (elevation, demand,
    tank levels, etc.), plus x/y coordinates when available.

    Edge attributes: type ('pipe'/'pump'/'valve'), id, and available
    hydraulic properties (length, diameter, roughness for pipes;
    pump_type for pumps; valve_type/setting for valves).

    Args:
        net: A CTownNetwork produced by ctown_loader.load_ctown_network.
        directed: Run a baseline WNTR simulation and orient each edge in its
            predominant flow direction. The default preserves the undirected
            graph used by existing topology and GNN code.
        hydraulic_results: Optional WNTR simulation results to reuse when
            ``directed`` is True.

    Returns:
        A networkx.Graph with hydraulic connectivity and metadata.
    """
    if directed and hydraulic_results is None:
        hydraulic_results = wntr.sim.EpanetSimulator(net.wn).run_sim()
    flow_rates = hydraulic_results.link["flowrate"] if directed else None
    g: nx.Graph | nx.DiGraph = nx.DiGraph() if directed else nx.Graph()
    coords = net.coordinates.set_index("id") if len(net.coordinates) else None

    def _edge_data(link_id: str, node1: str, node2: str) -> tuple[str, str, dict[str, Any]]:
        if not directed:
            return node1, node2, {}

        if link_id not in flow_rates.columns:
            raise ValueError(f"Simulation results do not contain flow data for link {link_id!r}.")
        values = flow_rates[link_id].to_numpy(dtype=float)
        active_values = values[np.isfinite(values) & (np.abs(values) > 1e-10)]
        if len(active_values) == 0:
            return node1, node2, {
                "flow_direction_source": "equipment_orientation_no_baseline_flow",
                "predominant_flow_fraction": None,
                "mean_flow": float(np.nanmean(values)) if np.isfinite(values).any() else 0.0,
                "weight": 1e-6,
            }

        positive = int(np.count_nonzero(active_values > 0))
        negative = len(active_values) - positive
        flow_is_forward = positive >= negative
        start, end = (node1, node2) if flow_is_forward else (node2, node1)
        return start, end, {
            "flow_direction_source": "predominant_simulated_flow",
            "predominant_flow_fraction": float(max(positive, negative) / len(active_values)),
            "mean_flow": float(np.mean(values)),
            "predominant_flow": float(np.median(active_values[active_values > 0]))
            if flow_is_forward
            else float(np.median(active_values[active_values < 0])),
            "weight": float(np.median(np.abs(active_values))),
        }

    def _coord(node_id: str) -> dict[str, Any]:
        if coords is not None and node_id in coords.index:
            return {"x": float(coords.loc[node_id, "x"]), "y": float(coords.loc[node_id, "y"])}
        return {}

    for _, row in net.junctions.iterrows():
        g.add_node(
            row["id"],
            type="junction",
            elevation=row["elevation"],
            base_demand=row["base_demand"],
            pattern=row["pattern"],
            **_coord(row["id"]),
        )

    for _, row in net.tanks.iterrows():
        g.add_node(
            row["id"],
            type="tank",
            elevation=row["elevation"],
            init_level=row["init_level"],
            min_level=row["min_level"],
            max_level=row["max_level"],
            diameter=row["diameter"],
            **_coord(row["id"]),
        )

    for _, row in net.reservoirs.iterrows():
        g.add_node(row["id"], type="reservoir", base_head=row["base_head"], **_coord(row["id"]))

    for _, row in net.pipes.iterrows():
        start, end, flow_metadata = _edge_data(str(row["id"]), row["node1"], row["node2"])
        g.add_edge(
            start,
            end,
            id=row["id"],
            type="pipe",
            length=row["length"],
            diameter=row["diameter"],
            roughness=row["roughness"],
            **flow_metadata,
        )

    for _, row in net.pumps.iterrows():
        start, end, flow_metadata = _edge_data(str(row["id"]), row["node1"], row["node2"])
        g.add_edge(
            start,
            end,
            id=row["id"],
            type="pump",
            pump_type=row["pump_type"],
            **flow_metadata,
        )

    for _, row in net.valves.iterrows():
        start, end, flow_metadata = _edge_data(str(row["id"]), row["node1"], row["node2"])
        g.add_edge(
            start,
            end,
            id=row["id"],
            type="valve",
            valve_type=row["valve_type"],
            setting=row["setting"],
            diameter=row["diameter"],
            **flow_metadata,
        )

    component_count = (
        nx.number_weakly_connected_components(g)
        if isinstance(g, nx.DiGraph)
        else nx.number_connected_components(g)
    )
    logger.info(
        "Built graph: %d nodes, %d edges (%d connected components)",
        g.number_of_nodes(),
        g.number_of_edges(),
        component_count,
    )
    return g


def build_directed_hydraulic_graph(
    net: CTownNetwork, hydraulic_results: Any | None = None
) -> nx.DiGraph:
    """Build a directed graph aligned with predominant simulated water flow."""
    graph = build_network_graph(net, directed=True, hydraulic_results=hydraulic_results)
    if not isinstance(graph, nx.DiGraph):
        raise TypeError("Directed hydraulic graph construction returned an undirected graph.")
    return graph


def node_metadata_table(g: nx.Graph):
    """Return node metadata as a list of dicts (for CSV/JSON export)."""
    return [dict(id=n, **d) for n, d in g.nodes(data=True)]


def edge_metadata_table(g: nx.Graph):
    """Return edge metadata as a list of dicts (for CSV/JSON export)."""
    return [dict(node1=u, node2=v, **d) for u, v, d in g.edges(data=True)]


def component_to_sensor_mapping_interface(g: nx.Graph) -> dict[str, list[str]]:
    """Placeholder interface: component id -> list of sensor ids.

    STATUS: PLACEHOLDER. The supplied project documents do not specify
    which physical sensors exist in the BATADAL/C-Town deployment or how
    they map to components. This function currently maps every node to
    itself (assumes one virtual "sensor" per monitored node) so downstream
    modules have a consistent interface to call. Replace with the real
    sensor deployment map when available.
    """
    return {n: [n] for n, d in g.nodes(data=True) if d.get("type") in {"tank", "junction"}}
