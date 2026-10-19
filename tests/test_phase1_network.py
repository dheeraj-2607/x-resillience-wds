"""
Unit tests for Phase 1: ctown_loader, graph_builder, zoning.

Run with: pytest tests/test_phase1_network.py -v
(run from the project root so `data/raw/CTOWN.INP` resolves)
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.network.ctown_loader import load_ctown_network
from src.network.graph_builder import (
    build_directed_hydraulic_graph,
    build_network_graph,
    component_to_sensor_mapping_interface,
)
from src.network.zoning import DMAZoning, LouvainZoning, assign_zones

INP_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "CTOWN.INP"


@pytest.fixture(scope="module")
def net():
    return load_ctown_network(INP_PATH)


@pytest.fixture(scope="module")
def graph(net):
    return build_network_graph(net)


class TestCTownLoader:
    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_ctown_network("does_not_exist.inp")

    def test_parses_all_component_types(self, net):
        assert len(net.junctions) > 0
        assert len(net.tanks) > 0
        assert len(net.reservoirs) > 0
        assert len(net.pipes) > 0
        assert len(net.pumps) > 0
        assert len(net.valves) > 0

    def test_component_counts_match_raw_file(self, net):
        # Cross-check against a raw independent count of the .inp text,
        # so we know the parser isn't silently dropping/duplicating rows.
        text = INP_PATH.read_text()

        def count_section(section: str) -> int:
            lines = text.split(f"[{section}]")[1].split("[")[0].splitlines()
            return sum(1 for l in lines if l.strip() and not l.strip().startswith(";"))

        assert len(net.junctions) == count_section("JUNCTIONS")
        assert len(net.pipes) == count_section("PIPES")
        assert len(net.pumps) == count_section("PUMPS")
        assert len(net.tanks) == count_section("TANKS")

    def test_coordinates_present(self, net):
        assert len(net.coordinates) > 0

    def test_summary_counts_are_consistent(self, net):
        s = net.summary()
        assert s["total_nodes"] == s["junctions"] + s["tanks"] + s["reservoirs"]


class TestGraphBuilder:
    def test_graph_node_count_matches_network(self, net, graph):
        s = net.summary()
        assert graph.number_of_nodes() == s["total_nodes"]

    def test_graph_edge_count_matches_network(self, net, graph):
        s = net.summary()
        assert graph.number_of_edges() == s["total_links"]

    def test_graph_is_connected(self, graph):
        # A hydraulically valid WDS should be a single connected component.
        assert nx.number_connected_components(graph) == 1

    def test_node_types_labeled(self, graph):
        types = {d["type"] for _, d in graph.nodes(data=True)}
        assert types == {"junction", "tank", "reservoir"}

    def test_edge_types_labeled(self, graph):
        types = {d["type"] for _, _, d in graph.edges(data=True)}
        assert types <= {"pipe", "pump", "valve"}

    def test_sensor_mapping_interface_covers_junctions_and_tanks(self, graph):
        mapping = component_to_sensor_mapping_interface(graph)
        n_junctions_tanks = sum(
            1 for _, d in graph.nodes(data=True) if d.get("type") in {"junction", "tank"}
        )
        assert len(mapping) == n_junctions_tanks

    def test_directed_graph_follows_predominant_simulated_flow(self, net):
        graph = build_directed_hydraulic_graph(net)

        assert isinstance(graph, nx.DiGraph)
        assert graph.has_edge("J336", "J335")
        assert not graph.has_edge("J335", "J336")
        reverse_pipe = graph["J336"]["J335"]
        assert reverse_pipe["id"] == "P10"
        assert reverse_pipe["flow_direction_source"] == "predominant_simulated_flow"
        assert reverse_pipe["predominant_flow_fraction"] == 1.0
        assert reverse_pipe["weight"] > 0.0
        assert graph.has_edge("J290", "J291")


class TestZoning:
    def test_dma_zoning_resolves_all_nodes(self, graph):
        zones = DMAZoning().assign_zones(graph)
        assert len(zones) == graph.number_of_nodes()

    def test_dma_zoning_finds_five_zones(self, graph):
        # CTOWN.INP defines DMA1_pat..DMA5_pat.
        zones = DMAZoning().assign_zones(graph)
        assert len({z for z in zones.values() if z != -1}) == 5

    def test_dma_zoning_no_unresolved_nodes(self, graph):
        # The network is fully connected, so BFS propagation should reach
        # every node from at least one DMA-labeled neighbor.
        zones = DMAZoning().assign_zones(graph)
        assert -1 not in zones.values()

    def test_louvain_zoning_partitions_all_nodes(self, graph):
        zones = LouvainZoning(random_state=42).assign_zones(graph)
        assert len(zones) == graph.number_of_nodes()
        assert len(set(zones.values())) > 1

    def test_dispatcher_default_is_dma(self, graph):
        zones = assign_zones(graph)
        assert len({z for z in zones.values() if z != -1}) == 5

    def test_dispatcher_rejects_unknown_method(self, graph):
        with pytest.raises(ValueError):
            assign_zones(graph, method="not_a_real_method")
