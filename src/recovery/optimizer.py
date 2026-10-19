"""WNTR-simulated recovery action evaluation and weighted plan selection."""

from __future__ import annotations

import copy
import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wntr
import networkx as nx


DEFAULT_RECOVERY_CONFIG: dict[str, Any] = {
    "weights": {
        "service_continuity": 0.30,
        "hydraulic_stability": 0.25,
        "recovery_time": 0.20,
        "operational_cost": 0.10,
        "criticality_protection": 0.15,
    },
    "candidate_actions": [
        "valve_isolation",
        "pump_switching",
        "flow_rerouting",
        "zone_containment",
        "emergency_adjustment",
    ],
    "simulation_duration_hours": 24,
    "hydraulic_timestep_minutes": 60,
    "demand_model": "PDA",
    "minimum_pressure_m": 0.0,
    "required_pressure_m": 20.0,
    "hydraulic_stability_pressure_scale_m": 10.0,
    "service_continuity_target": 0.95,
    "hydraulic_stability_target": 0.90,
    "recovery_persistence_steps": 2,
    "pump_speed_increase": 0.10,
    "max_pump_speed": 1.20,
    "action_costs": {
        "restore_component": 0.1,
        "valve_isolation": 0.5,
        "pump_switching": 0.4,
        "flow_rerouting": 0.3,
        "zone_containment": 0.7,
        "emergency_adjustment": 0.8,
    },
    "max_candidate_actions": 12,
}


def generate_recovery_candidates(
    incident_model: Any,
    target_link_id: str,
    dma_zones: Mapping[str, int] | None = None,
    affected_nodes: Sequence[str] = (),
    recovery_config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate feasible link/pump/zone response candidates from the WNTR model."""
    config = _merge_recovery_config(recovery_config)
    network = incident_model
    target = network.get_link(target_link_id)
    target_endpoints = {target.start_node_name, target.end_node_name}
    area_nodes = set(affected_nodes) | target_endpoints
    target_zone_ids = {
        int(dma_zones[node])
        for node in area_nodes
        if dma_zones and node in dma_zones and int(dma_zones[node]) > 0
    }
    requested_types = set(config.get("candidate_actions", []))
    costs = config.get("action_costs", {})
    candidates: list[dict[str, Any]] = []

    hydraulic_graph = nx.Graph()
    for node_name in network.node_name_list:
        hydraulic_graph.add_node(node_name)
    for link_id in network.link_name_list:
        link = network.get_link(link_id)
        hydraulic_graph.add_edge(link.start_node_name, link.end_node_name, id=link_id)

    def add(action_id: str, action_type: str, label: str, operations: list[dict[str, Any]]) -> None:
        if action_type not in requested_types or not operations:
            return
        candidates.append(
            {
                "action_id": action_id,
                "action_type": action_type,
                "label": label,
                "operations": operations,
                "operational_cost": float(costs.get(action_type, 0.5)),
            }
        )

    add(
        f"restore_{target_link_id}",
        "valve_isolation" if target_link_id in network.valve_name_list else "pump_switching",
        f"Restore {target_link_id} to service",
        [{"operation": "set_link_status", "link_id": target_link_id, "status": "Open"}],
    )

    nearby_links = []
    for link_id in network.link_name_list:
        if link_id == target_link_id:
            continue
        link = network.get_link(link_id)
        endpoints = {link.start_node_name, link.end_node_name}
        if endpoints & area_nodes:
            nearby_links.append((link_id, link, endpoints))

    if "pump_switching" in requested_types and target_link_id in network.pump_name_list:
        backup_pumps = []
        for backup_id in network.pump_name_list:
            if backup_id == target_link_id:
                continue
            backup = network.get_link(backup_id)
            backup_nodes = {backup.start_node_name, backup.end_node_name}
            try:
                distance = min(
                    nx.shortest_path_length(hydraulic_graph, target_node, backup_node)
                    for target_node in target_endpoints
                    for backup_node in backup_nodes
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                distance = math.inf
            backup_zones = {
                int(dma_zones[node])
                for node in backup_nodes
                if dma_zones and node in dma_zones and int(dma_zones[node]) > 0
            }
            zone_mismatch = bool(target_zone_ids and backup_zones and not target_zone_ids.intersection(backup_zones))
            backup_pumps.append((zone_mismatch, distance, backup_id))
        for _, _, backup_id in sorted(backup_pumps)[:3]:
            add(
                f"switch_{target_link_id}_to_{backup_id}",
                "pump_switching",
                f"Switch from {target_link_id} to {backup_id}",
                [
                    {"operation": "set_link_status", "link_id": target_link_id, "status": "Closed"},
                    {"operation": "set_link_status", "link_id": backup_id, "status": "Open"},
                ],
            )

    candidate_zone_nodes = {
        node for node, zone_id in (dma_zones or {}).items() if int(zone_id) in target_zone_ids
    } | area_nodes
    reroute_links = []
    for link_id in network.valve_name_list:
        if link_id == target_link_id:
            continue
        link = network.get_link(link_id)
        endpoints = {link.start_node_name, link.end_node_name}
        if endpoints & candidate_zone_nodes and link.initial_status == wntr.network.LinkStatus.Closed:
            reroute_links.append((link_id, link, endpoints))
    for link_id, link, _ in reroute_links:
        if "flow_rerouting" in requested_types:
            add(
                f"open_route_{link_id}",
                "flow_rerouting",
                f"Open alternate valve route {link_id}",
                [{"operation": "set_link_status", "link_id": link_id, "status": "Open"}],
            )

    for link_id, link, _ in nearby_links:
        status = getattr(link, "initial_status", None)
        if status != wntr.network.LinkStatus.Closed:
            continue
        if link_id in network.pump_name_list and "pump_switching" in requested_types:
            add(
                f"start_{link_id}",
                "pump_switching",
                f"Start adjacent pump {link_id}",
                [{"operation": "set_link_status", "link_id": link_id, "status": "Open"}],
            )

    if target_zone_ids and "zone_containment" in requested_types:
        boundary_valves = []
        for valve_id in network.valve_name_list:
            if valve_id == target_link_id:
                continue
            valve = network.get_link(valve_id)
            start_zone = (dma_zones or {}).get(valve.start_node_name, -1)
            end_zone = (dma_zones or {}).get(valve.end_node_name, -1)
            if (start_zone in target_zone_ids) != (end_zone in target_zone_ids):
                boundary_valves.append(valve_id)
        add(
            "contain_affected_zone",
            "zone_containment",
            f"Contain DMA {','.join(map(str, sorted(target_zone_ids)))}",
            [
                {"operation": "set_link_status", "link_id": valve_id, "status": "Closed"}
                for valve_id in boundary_valves
            ],
        )

    if "emergency_adjustment" in requested_types:
        pump_operations = []
        for pump_id in network.pump_name_list:
            pump = network.get_link(pump_id)
            pump_nodes = {pump.start_node_name, pump.end_node_name}
            in_zone = any(node in candidate_zone_nodes for node in pump_nodes)
            if in_zone and hasattr(pump, "speed_timeseries"):
                current_speed = float(pump.speed_timeseries.base_value)
                new_speed = min(
                    current_speed + float(config["pump_speed_increase"]),
                    float(config["max_pump_speed"]),
                )
                if new_speed > current_speed:
                    pump_operations.append(
                        {"operation": "set_pump_speed", "link_id": pump_id, "speed": new_speed}
                    )
        add(
            "boost_adjacent_pumps",
            "emergency_adjustment",
            "Increase speed of affected-area pumps within configured limit",
            pump_operations,
        )

    max_actions = max(1, int(config["max_candidate_actions"]))
    return candidates[:max_actions]


def optimize_component_outage(
    reference_model: Any,
    target_link_id: str,
    dma_zones: Mapping[str, int] | None = None,
    affected_nodes: Sequence[str] = (),
    critical_nodes: Sequence[str] = (),
    recovery_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Model a link outage, generate recovery alternatives, and optimize them.

    This is a hydraulic component-outage contingency, distinct from arbitrary
    sensor-value injection in presentation demo mode.
    """
    incident_model = copy.deepcopy(reference_model)
    if target_link_id not in incident_model.link_name_list:
        raise ValueError(f"Unknown target link for outage scenario: {target_link_id}")
    incident_model.get_link(target_link_id).initial_status = wntr.network.LinkStatus.Closed
    actions = generate_recovery_candidates(
        incident_model,
        target_link_id,
        dma_zones=dma_zones,
        affected_nodes=affected_nodes,
        recovery_config=recovery_config,
    )
    result = optimize_recovery_actions(
        reference_model,
        incident_model,
        actions,
        critical_nodes=critical_nodes,
        recovery_config=recovery_config,
    )
    result["scenario"] = "simulated_component_outage"
    result["target_link_id"] = target_link_id
    result["candidate_action_count"] = len(actions)
    return result


def optimize_recovery_actions(
    reference_model: Any,
    incident_model: Any,
    candidate_actions: Sequence[Mapping[str, Any]],
    critical_nodes: Sequence[str] = (),
    recovery_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate candidate actions and rank by weighted resilience objectives.

    `reference_model` is the healthy hydraulic reference and `incident_model`
    is a separately prepared incident-state copy. Candidate operations are
    applied to clones, never to either input model or the raw INP file.
    """
    config = _merge_recovery_config(recovery_config)
    if not candidate_actions:
        return {"status": "NO_CANDIDATES", "recommended_action": None, "evaluations": []}

    reference_results = _simulate(copy.deepcopy(reference_model), config, "reference")
    baseline_junctions = list(reference_model.junction_name_list)
    critical = [node for node in critical_nodes if node in baseline_junctions]
    if not critical:
        critical = baseline_junctions

    evaluations = []
    for candidate in candidate_actions:
        action_id = str(candidate.get("action_id", "unnamed_action"))
        try:
            scenario = copy.deepcopy(incident_model)
            _apply_operations(scenario, candidate.get("operations", []))
            result = _simulate(scenario, config, action_id)
            metrics = _score_simulation(
                reference_results,
                result,
                baseline_junctions,
                critical,
                candidate,
                config,
            )
            evaluations.append({**candidate, **metrics, "status": "SIMULATED"})
        except Exception as error:
            evaluations.append(
                {
                    **candidate,
                    "status": "SIMULATION_FAILED",
                    "failure_reason": str(error),
                    "objective_score": None,
                    "feasible": False,
                }
            )

    successful = [row for row in evaluations if row["status"] == "SIMULATED"]
    if not successful:
        return {"status": "NO_SIMULATIONS_SUCCEEDED", "recommended_action": None, "evaluations": evaluations}
    successful.sort(key=lambda row: (not row["feasible"], -row["objective_score"], row["action_id"]))
    feasible = [row for row in successful if row["feasible"]]
    return {
        "status": "OPTIMIZED" if feasible else "NO_FEASIBLE_ACTION",
        "recommended_action": successful[0] if feasible else None,
        "best_available_action": successful[0],
        "evaluations": evaluations,
        "evaluation_note": "WNTR hydraulic simulations; estimated scenario metrics, not field-validated outcomes.",
    }


def _merge_recovery_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_RECOVERY_CONFIG)
    if config:
        for key, value in config.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
    return merged


def _apply_operations(network: Any, operations: Sequence[Mapping[str, Any]]) -> None:
    status_names = {"Open": wntr.network.LinkStatus.Open, "Closed": wntr.network.LinkStatus.Closed}
    for operation in operations:
        operation_type = operation.get("operation")
        link_id = operation.get("link_id")
        if link_id not in network.link_name_list:
            raise ValueError(f"Unknown WNTR link: {link_id}")
        link = network.get_link(link_id)
        if operation_type == "set_link_status":
            status = operation.get("status")
            if status not in status_names:
                raise ValueError(f"Unsupported link status for recovery operation: {status}")
            link.initial_status = status_names[status]
        elif operation_type == "set_pump_speed":
            if link_id not in network.pump_name_list or not hasattr(link, "speed_timeseries"):
                raise ValueError(f"Link is not a controllable pump: {link_id}")
            speed = float(operation["speed"])
            if not math.isfinite(speed) or speed <= 0:
                raise ValueError("Pump speed must be a finite positive multiplier.")
            link.speed_timeseries.base_value = speed
        elif operation_type == "set_valve_setting":
            if link_id not in network.valve_name_list or not hasattr(link, "initial_setting"):
                raise ValueError(f"Link is not a configurable valve: {link_id}")
            setting = float(operation["setting"])
            if not math.isfinite(setting):
                raise ValueError("Valve setting must be finite.")
            link.initial_setting = setting
        else:
            raise ValueError(f"Unsupported recovery operation: {operation_type}")


def _simulate(network: Any, config: Mapping[str, Any], run_name: str) -> Any:
    duration_hours = int(config["simulation_duration_hours"])
    timestep_seconds = int(config["hydraulic_timestep_minutes"]) * 60
    if duration_hours <= 0 or timestep_seconds <= 0:
        raise ValueError("Recovery simulation duration and hydraulic timestep must be positive.")
    network.options.time.duration = duration_hours * 3600
    network.options.time.hydraulic_timestep = timestep_seconds
    network.options.time.report_timestep = timestep_seconds
    network.options.hydraulic.demand_model = config["demand_model"]
    network.options.hydraulic.minimum_pressure = float(config["minimum_pressure_m"])
    network.options.hydraulic.required_pressure = float(config["required_pressure_m"])
    with tempfile.TemporaryDirectory(prefix="xres-recovery-") as temp_dir:
        prefix = Path(temp_dir) / run_name
        return wntr.sim.EpanetSimulator(network).run_sim(file_prefix=str(prefix))


def _score_simulation(
    reference_results: Any,
    candidate_results: Any,
    junctions: Sequence[str],
    critical_nodes: Sequence[str],
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ref_pressure = reference_results.node["pressure"][list(junctions)]
    cand_pressure = candidate_results.node["pressure"][list(junctions)]
    ref_demand = reference_results.node["demand"][list(junctions)].clip(lower=0)
    cand_demand = candidate_results.node["demand"][list(junctions)].clip(lower=0)

    common_index = ref_pressure.index.intersection(cand_pressure.index)
    common_nodes = [node for node in junctions if node in ref_pressure.columns and node in cand_pressure.columns]
    ref_pressure = ref_pressure.loc[common_index, common_nodes]
    cand_pressure = cand_pressure.loc[common_index, common_nodes]
    ref_demand = ref_demand.loc[common_index, common_nodes]
    cand_demand = cand_demand.loc[common_index, common_nodes]

    demand_total = float(ref_demand.to_numpy().sum())
    if demand_total <= 0:
        raise ValueError("Reference simulation has no positive junction demand.")
    service_continuity = float(np.minimum(cand_demand.to_numpy(), ref_demand.to_numpy()).sum() / demand_total)
    service_continuity = float(np.clip(service_continuity, 0.0, 1.0))

    pressure_scale = float(config["hydraulic_stability_pressure_scale_m"])
    if pressure_scale <= 0:
        raise ValueError("hydraulic_stability_pressure_scale_m must be positive.")
    pressure_delta = np.abs(cand_pressure.to_numpy() - ref_pressure.to_numpy())
    pressure_quality = np.exp(-pressure_delta / pressure_scale)
    floor_m = float(config["minimum_pressure_m"])
    pressure_valid = cand_pressure.to_numpy() >= floor_m
    hydraulic_stability = float(np.mean(pressure_quality * pressure_valid))
    hydraulic_stability = float(np.clip(hydraulic_stability, 0.0, 1.0))

    critical_nodes = [node for node in critical_nodes if node in common_nodes]
    if critical_nodes:
        critical_ref_demand = ref_demand[critical_nodes].to_numpy()
        critical_cand_demand = cand_demand[critical_nodes].to_numpy()
        critical_total = float(critical_ref_demand.sum())
        critical_service = (
            float(np.minimum(critical_cand_demand, critical_ref_demand).sum() / critical_total)
            if critical_total > 0
            else 1.0
        )
        critical_pressure_stability = float(
            np.mean(np.exp(-np.abs(cand_pressure[critical_nodes].to_numpy() - ref_pressure[critical_nodes].to_numpy()) / pressure_scale))
        )
        criticality_protection = 0.5 * critical_service + 0.5 * critical_pressure_stability
    else:
        criticality_protection = 0.5 * service_continuity + 0.5 * hydraulic_stability

    service_by_step = np.minimum(cand_demand.to_numpy(), ref_demand.to_numpy()).sum(axis=1) / np.maximum(
        ref_demand.to_numpy().sum(axis=1), 1e-12
    )
    stability_by_step = np.mean(pressure_quality * pressure_valid, axis=1)
    step_seconds = int(config["hydraulic_timestep_minutes"]) * 60
    required_steps = max(1, int(config["recovery_persistence_steps"]))
    target_service = float(config["service_continuity_target"])
    target_stability = float(config["hydraulic_stability_target"])
    recovered_at = None
    for index in range(max(0, len(common_index) - required_steps + 1)):
        window = slice(index, index + required_steps)
        if np.all(service_by_step[window] >= target_service) and np.all(stability_by_step[window] >= target_stability):
            recovered_at = index
            break
    simulation_hours = float(config["simulation_duration_hours"])
    recovery_time = (
        float(common_index[recovered_at]) * step_seconds / 3600.0
        if recovered_at is not None
        else simulation_hours
    )
    recovery_time_score = float(np.clip(1.0 - recovery_time / simulation_hours, 0.0, 1.0))

    action_cost = max(0.0, float(candidate.get("operational_cost", 0.0)))
    cost_scale = max(float(config.get("maximum_operational_cost", 1.0)), 1e-12)
    operational_cost_score = float(np.clip(1.0 - action_cost / cost_scale, 0.0, 1.0))
    weights = {key: float(value) for key, value in config["weights"].items()}
    weight_total = sum(weights.values())
    if weight_total <= 0 or any(value < 0 or not math.isfinite(value) for value in weights.values()):
        raise ValueError("Recovery objective weights must be finite, nonnegative, and have positive sum.")
    objective_score = (
        weights["service_continuity"] * service_continuity
        + weights["hydraulic_stability"] * hydraulic_stability
        + weights["recovery_time"] * recovery_time_score
        + weights["operational_cost"] * operational_cost_score
        + weights["criticality_protection"] * criticality_protection
    ) / weight_total
    feasible = (
        service_continuity >= target_service
        and hydraulic_stability >= target_stability
        and recovered_at is not None
    )
    return {
        "service_continuity": round(service_continuity, 4),
        "hydraulic_stability": round(hydraulic_stability, 4),
        "recovery_time_hours": round(recovery_time, 2),
        "recovery_time_score": round(recovery_time_score, 4),
        "operational_cost": round(action_cost, 4),
        "operational_cost_score": round(operational_cost_score, 4),
        "criticality_protection": round(float(np.clip(criticality_protection, 0.0, 1.0)), 4),
        "objective_score": round(float(objective_score), 4),
        "feasible": feasible,
    }


def _snapshot_at_results(network: Any, results: Any, result_index: int, timestamp: pd.Timestamp) -> Any:
    """Rebase a WNTR model to the hydraulic state at one simulated result row."""
    snapshot = copy.deepcopy(network)
    for tank_id in snapshot.tank_name_list:
        tank = snapshot.get_node(tank_id)
        level = float(results.node["head"][tank_id].iloc[result_index]) - float(tank.elevation)
        tank.init_level = float(np.clip(level, tank.min_level, tank.max_level))
    for link_id in snapshot.link_name_list:
        status_value = float(results.link["status"][link_id].iloc[result_index])
        snapshot.get_link(link_id).initial_status = (
            wntr.network.LinkStatus.Open if status_value >= 0.5 else wntr.network.LinkStatus.Closed
        )
    snapshot.options.time.start_clocktime = float(timestamp.hour * 3600 + timestamp.minute * 60)
    snapshot.options.time.pattern_start = float(
        (timestamp.dayofweek * 24 + timestamp.hour) * 3600 + timestamp.minute * 60
    )
    return snapshot


def _relative_reduction(baseline: float, candidate: float) -> float | None:
    if baseline <= 1e-12:
        return None
    return 100.0 * (baseline - candidate) / baseline


def _nearest_controllable_link(network: Any, node_id: str) -> str | None:
    graph = nx.Graph()
    graph.add_nodes_from(network.node_name_list)
    for link_id in network.link_name_list:
        link = network.get_link(link_id)
        graph.add_edge(link.start_node_name, link.end_node_name)
    controllable = sorted(set(network.pump_name_list) | set(network.valve_name_list))
    ranked = []
    for link_id in controllable:
        link = network.get_link(link_id)
        try:
            distance = min(
                nx.shortest_path_length(graph, node_id, endpoint)
                for endpoint in (link.start_node_name, link.end_node_name)
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        ranked.append((distance, link_id))
    return min(ranked)[1] if ranked else None


def _attack_anchor_link(network: Any, attack: Any) -> str | None:
    if attack.target_component in network.pump_name_list or attack.target_component in network.valve_name_list:
        return str(attack.target_component)
    for sensor in str(attack.affected_sensors).split(";"):
        if sensor.startswith(("F_", "S_")):
            link_id = sensor[2:]
            if link_id in network.pump_name_list or link_id in network.valve_name_list:
                return link_id
    return _nearest_controllable_link(network, str(attack.target_component))


def _scenario_report_metrics(
    incident_model: Any,
    actions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    dma_zones: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from src.impact.cpis import _pda_physical_metrics

    baseline_results = _simulate(copy.deepcopy(incident_model), config, "unmitigated")
    baseline = _pda_physical_metrics(
        incident_model, baseline_results, dma_zones, float(config["required_pressure_m"])
    )
    scenarios = {"baseline_0_no_action": baseline}
    for name, operations in actions:
        scenario = copy.deepcopy(incident_model)
        _apply_operations(scenario, operations)
        result = _simulate(scenario, config, name)
        metrics = _pda_physical_metrics(scenario, result, dma_zones, float(config["required_pressure_m"]))
        scenarios[name] = metrics
    return baseline, scenarios


def run_dynamic_recovery_benchmark(
    root: str | Path,
    recovery_config: Mapping[str, Any] | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Compare optimized actions and mandatory baselines from each attack's detected state."""
    from src.evaluation.labels import sensor_component
    from src.impact.cpis import _apply_observed_actuator_replay, _prepare_attack_network
    from src.network.ctown_loader import load_ctown_network
    from src.network.graph_builder import build_network_graph
    from src.network.zoning import assign_zones

    root = Path(root)
    attacks = pd.read_csv(root / "data" / "ground_truth_attacks.csv")
    detection_report = json.loads((root / "results" / "multiseed_detection_report.json").read_text(encoding="utf-8"))
    localization_records = json.loads((root / "results" / "fair_localization_records.json").read_text(encoding="utf-8"))
    configured = _merge_recovery_config(recovery_config)
    rng = np.random.default_rng(random_seed)
    data_files = {
        "Dataset04": root / "data" / "raw" / "BATADAL_dataset04.csv",
        "Test Dataset": root / "data" / "raw" / "BATADAL_test_dataset.csv",
    }
    data_frames = {}
    for name, path in data_files.items():
        frame = pd.read_csv(path)
        frame["DATETIME"] = pd.to_datetime(frame["DATETIME"], dayfirst=True)
        data_frames[name] = frame

    metadata = load_ctown_network(root / "data" / "raw" / "CTOWN.INP")
    graph = build_network_graph(metadata)
    zones = assign_zones(graph, method="dma")
    rows = []

    for attack in attacks.itertuples(index=False):
        dataset_name = str(attack.dataset)
        available_delays = []
        for run in detection_report["runs"]:
            if run["adjacency"] != "A_CTOWN":
                continue
            delay = run["report"][dataset_name]["delay_hours"].get(str(attack.attack_id))
            if delay is not None:
                available_delays.append((float(delay), int(run["seed"])))
        base_row = {
            "attack_id": int(attack.attack_id),
            "dataset": dataset_name,
            "attack_type": str(attack.attack_type),
            "target": f"{attack.component_type} {attack.target_component}",
        }
        if not available_delays:
            rows.append({**base_row, "status": "NOT_DETECTED", "detection_timestamp": None,
                         "baseline_0_no_action": None, "baseline_1_naive_isolation": None,
                         "baseline_2_random_feasible": None, "optimizer_best": None})
            continue

        available_delays.sort()
        delay_hours, detection_seed = available_delays[len(available_delays) // 2]
        start_time = pd.Timestamp(attack.start_time)
        end_time = pd.Timestamp(attack.end_time)
        detection_time = start_time + pd.to_timedelta(delay_hours, unit="h")
        elapsed_seconds = max(0.0, (detection_time - start_time).total_seconds())
        frame = data_frames[dataset_name]
        baseline_start, _, _ = _prepare_attack_network(
            root / "data" / "raw" / "CTOWN.INP", attack, frame, start_time, end_time,
            replay_actuators=False, timestep_minutes=int(configured["hydraulic_timestep_minutes"]),
        )
        incident_start, _, replayed_sensors = _prepare_attack_network(
            root / "data" / "raw" / "CTOWN.INP", attack, frame, start_time, end_time,
            replay_actuators=True, timestep_minutes=int(configured["hydraulic_timestep_minutes"]),
        )

        if elapsed_seconds > 0:
            pre_config = {**configured, "simulation_duration_hours": max(1, int(math.ceil(delay_hours)))}
            baseline_pre = _simulate(copy.deepcopy(baseline_start), pre_config, "baseline_to_detection")
            incident_pre = _simulate(copy.deepcopy(incident_start), pre_config, "incident_to_detection")
            baseline_index = int(np.argmin(np.abs(np.asarray(baseline_pre.node["pressure"].index) - elapsed_seconds)))
            incident_index = int(np.argmin(np.abs(np.asarray(incident_pre.node["pressure"].index) - elapsed_seconds)))
            baseline_state = _snapshot_at_results(baseline_start, baseline_pre, baseline_index, detection_time)
            incident_state = _snapshot_at_results(incident_start, incident_pre, incident_index, detection_time)
        else:
            baseline_state = copy.deepcopy(baseline_start)
            incident_state = copy.deepcopy(incident_start)

        for control_name in list(incident_state.control_name_list):
            if control_name.startswith("attack_replay_"):
                incident_state.remove_control(control_name)
        remaining_hours = max(1, int(math.ceil((end_time - detection_time).total_seconds() / 3600.0)))
        sim_config = {
            **configured,
            "simulation_duration_hours": remaining_hours,
            "hydraulic_timestep_minutes": 15,
            "demand_model": "PDA",
            "minimum_pressure_m": 0.0,
            "required_pressure_m": 20.0,
        }
        _configure_network_time(baseline_state, detection_time, remaining_hours, 15)
        _configure_network_time(incident_state, detection_time, remaining_hours, 15)
        replayed_after_detection = _apply_observed_actuator_replay(
            incident_state, attack, frame, detection_time, end_time
        )
        anchor_link = _attack_anchor_link(incident_state, attack)
        if anchor_link is None:
            rows.append({**base_row, "status": "NO_CONTROLLABLE_LINK", "detection_seed": detection_seed,
                         "detection_delay_hours": delay_hours, "detection_timestamp": detection_time.isoformat(),
                         "actuator_sensors_replayed": replayed_sensors,
                         "baseline_0_no_action": None, "baseline_1_naive_isolation": None,
                         "baseline_2_random_feasible": None, "optimizer_best": None})
            continue

        anchor_link_obj = incident_state.get_link(anchor_link)
        affected_nodes = [anchor_link_obj.start_node_name, anchor_link_obj.end_node_name]
        candidates = generate_recovery_candidates(
            incident_state, anchor_link, dma_zones=zones, affected_nodes=affected_nodes,
            recovery_config=configured,
        )
        detector_record = next((record for record in localization_records
                                if record.get("attack_id") == int(attack.attack_id)
                                and record.get("seed") == detection_seed
                                and record.get("dataset") == dataset_name
                                and record.get("timing") == "detector"), {})
        top_shap_sensors = detector_record.get("shap_top5_sensors", [])
        top_shap_sensor = top_shap_sensors[0] if top_shap_sensors else ""
        shap_component = sensor_component(top_shap_sensor) if top_shap_sensor else None
        if shap_component and shap_component.startswith(("Pump ", "Valve ")):
            naive_link = shap_component.split(" ", maxsplit=1)[1]
        elif top_shap_sensor.startswith(("L_", "P_")):
            naive_link = _nearest_controllable_link(incident_state, top_shap_sensor[2:]) or anchor_link
        else:
            naive_link = anchor_link
        if naive_link not in incident_state.link_name_list:
            naive_link = anchor_link
        naive_action = [{"operation": "set_link_status", "link_id": naive_link, "status": "Closed"}]

        candidate_list = candidates or []
        optimization = optimize_recovery_actions(
            baseline_state,
            incident_state,
            candidate_list,
            critical_nodes=[node for node, zone in zones.items() if zone == int(str(attack.dma_zone).replace("DMA", ""))],
            recovery_config=sim_config,
        )
        feasible_ids = {
            row["action_id"] for row in optimization.get("evaluations", [])
            if row.get("status") == "SIMULATED" and row.get("feasible") is True
        }
        feasible_candidates = [candidate for candidate in candidates if candidate["action_id"] in feasible_ids]
        random_action = (
            feasible_candidates[int(rng.integers(0, len(feasible_candidates)))]
            if feasible_candidates
            else None
        )
        best_action = optimization.get("recommended_action") or optimization.get("best_available_action")
        scenarios_to_run = [("baseline_1_naive_isolation", naive_action)]
        if random_action:
            scenarios_to_run.append(("baseline_2_random_feasible", random_action.get("operations", [])))
        if best_action:
            scenarios_to_run.append(("optimizer_best", best_action.get("operations", [])))
        baseline_metrics, scenario_metrics = _scenario_report_metrics(
            incident_state, scenarios_to_run, sim_config, zones
        )

        def report_scenario(name: str) -> dict[str, Any] | None:
            metrics = scenario_metrics.get(name)
            if metrics is None:
                return None
            return {
                "unmet_demand_volume_m3": metrics["unmet_demand_volume_m3"],
                "pressure_violation_junction_hours": metrics["pressure_violation_junction_hours"],
                "unmet_demand_reduction_pct_vs_no_action": _relative_reduction(
                    baseline_metrics["unmet_demand_volume_m3"], metrics["unmet_demand_volume_m3"]
                ),
                "pressure_violation_hours_reduction_pct_vs_no_action": _relative_reduction(
                    baseline_metrics["pressure_violation_junction_hours"],
                    metrics["pressure_violation_junction_hours"],
                ),
            }

        rows.append({
            **base_row,
            "status": "EVALUATED",
            "detection_seed": detection_seed,
            "detection_delay_hours": delay_hours,
            "detection_timestamp": detection_time.isoformat(),
            "simulation_horizon_after_detection_hours": remaining_hours,
            "actuator_sensors_replayed": sorted(set(replayed_sensors + replayed_after_detection)),
            "shap_top5_sensors": top_shap_sensors,
            "naive_isolation_sensor": top_shap_sensor or None,
            "naive_isolated_link": naive_link,
            "random_action_id": random_action.get("action_id") if random_action else None,
            "random_feasible_action_status": "SELECTED" if random_action else "NO_FEASIBLE_ACTION_AVAILABLE",
            "optimizer_status": optimization.get("status"),
            "optimizer_best_action_id": best_action.get("action_id") if best_action else None,
            "baseline_0_no_action": {
                "unmet_demand_volume_m3": baseline_metrics["unmet_demand_volume_m3"],
                "pressure_violation_junction_hours": baseline_metrics["pressure_violation_junction_hours"],
                "pressure_head_deficit_mh": baseline_metrics["pressure_head_deficit_mh"],
            },
            "baseline_1_naive_isolation": report_scenario("baseline_1_naive_isolation"),
            "baseline_2_random_feasible": report_scenario("baseline_2_random_feasible"),
            "optimizer_best": report_scenario("optimizer_best"),
        })

    def mean_reduction(key: str, metric: str) -> float | None:
        values = [row[key][metric] for row in rows if row.get(key) and row[key].get(metric) is not None]
        return float(np.mean(values)) if values else None

    return {
        "simulation": {
            "engine": "WNTR EpanetSimulator",
            "demand_model": "PDA",
            "minimum_pressure_m": 0.0,
            "required_pressure_m": 20.0,
            "detection_policy": "Median observed detection delay among C-Town 5-seed runs; state is snapshotted at that exact timestamp before response simulation.",
            "unit_definitions": {"unmet_demand_volume_m3": "m^3", "pressure_violation_junction_hours": "junction-hours", "pressure_head_deficit_mh": "m*h"},
        },
        "baseline_definitions": {
            "baseline_0_no_action": "Continue observed incident status replay after detection.",
            "baseline_1_naive_isolation": "Close the highest-ranked SHAP controllable component; fall back to the incident anchor link if the top sensor has no isolatable link.",
            "baseline_2_random_feasible": "One seeded random action from the optimizer's feasible candidate set.",
        },
        "summary": {
            "attack_count": len(rows),
            "evaluated_attack_count": sum(row["status"] == "EVALUATED" for row in rows),
            "not_detected_attack_count": sum(row["status"] == "NOT_DETECTED" for row in rows),
            "mean_optimizer_unmet_demand_reduction_pct": mean_reduction("optimizer_best", "unmet_demand_reduction_pct_vs_no_action"),
            "mean_optimizer_pressure_violation_hours_reduction_pct": mean_reduction("optimizer_best", "pressure_violation_hours_reduction_pct_vs_no_action"),
        },
        "table_r6": rows,
    }


def _configure_network_time(network: Any, start_time: pd.Timestamp, duration_hours: int, timestep_minutes: int) -> None:
    network.options.time.duration = int(duration_hours) * 3600
    network.options.time.hydraulic_timestep = int(timestep_minutes) * 60
    network.options.time.report_timestep = int(timestep_minutes) * 60
    network.options.time.start_clocktime = float(start_time.hour * 3600 + start_time.minute * 60)
    network.options.time.pattern_start = float(
        (start_time.dayofweek * 24 + start_time.hour) * 3600 + start_time.minute * 60
    )


def main() -> None:
    import yaml

    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text(encoding="utf-8"))
    report = run_dynamic_recovery_benchmark(root, config.get("recovery", {}))
    output = root / "results" / "recovery_benchmark_report.json"
    output.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"Wrote Table R6 to {output} ({report['summary']['attack_count']} attacks)")


if __name__ == "__main__":
    main()