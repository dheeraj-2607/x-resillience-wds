"""Generate C-Town synthetic cyber-physical scenarios and run the Stage 2 pipeline.

Default run creates 71 BATADAL-format CSV scenarios, a master attack table, and
five-seed detection, Method E localization, PDA CPIS, and recovery metrics with
scenario-bootstrap 95% confidence intervals.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
import wntr
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.dataset_loader import BATADALDatasetLoader
from src.detection.graph_model import SpatioTemporalGNN, build_adjacency_matrix_from_graph
from src.detection.scoring import calibrated_sensor_variance
from src.evaluation.data_prep import build_splits, harmonise_precision
from src.evaluation.detection_eval import calibrate
from src.evaluation.labels import TARGET_OFFSET, sensor_component
from src.impact.cpis import (
    CPIS_FACTOR_NAMES,
    _pda_physical_metrics,
    calculate_cpis,
)
from src.network.ctown_loader import load_ctown_network
from src.network.graph_builder import build_network_graph
from src.network.zoning import assign_zones
from src.recovery.optimizer import (
    _apply_operations,
    _merge_recovery_config,
    _nearest_controllable_link,
    _score_simulation,
    _simulate,
    _snapshot_at_results,
    generate_recovery_candidates,
    optimize_recovery_actions,
)
from src.root_cause.sensitivity_signatures import load_sensitivity_signatures, signature_ranking

BASE_START = pd.Timestamp("2017-01-04 00:00:00")
HORIZON_HOURS = 168
DURATION_CHOICES = (6, 12, 24, 48, 72)
SEEDS = (42, 123, 2024, 3407, 9999)


def _sensor_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith(("P_", "L_", "F_", "S_"))]


def _scenario_specs(network: Any, count: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    specs: list[dict[str, Any]] = []
    for link_id in sorted(network.pump_name_list):
        for mode in ("force_on", "force_off"):
            specs.append({"attack_type": "Actuator Manipulation", "component_type": "Pump",
                          "target_component": link_id, "mode": mode,
                          "affected_sensors": f"F_{link_id};S_{link_id}"})
    for link_id in sorted(network.valve_name_list):
        if link_id != "V2":
            continue
        for mode in ("force_open", "force_closed"):
            specs.append({"attack_type": "Actuator Manipulation", "component_type": "Valve",
                          "target_component": link_id, "mode": mode,
                          "affected_sensors": f"F_{link_id};S_{link_id}"})

    rule_tanks = set()
    for control_name in network.control_name_list:
        control = network.get_control(control_name)
        source = getattr(getattr(control, "_condition", None), "_source_obj", None)
        if source is not None and source.name in network.tank_name_list:
            rule_tanks.add(source.name)
    for tank_id in sorted(rule_tanks):
        linked_assets = []
        for control_name in network.control_name_list:
            control = network.get_control(control_name)
            source = getattr(getattr(control, "_condition", None), "_source_obj", None)
            if source is None or source.name != tank_id:
                continue
            for action in list(getattr(control, "_then_actions", ())) + list(getattr(control, "_else_actions", ())):
                target = getattr(action, "_target_obj", None)
                if target is not None and target.name in network.link_name_list:
                    linked_assets.append(target.name)
        affected = [f"L_{tank_id}"]
        for link_id in sorted(set(linked_assets)):
            affected.extend((f"F_{link_id}", f"S_{link_id}"))
        for mode in ("raise_low_thresholds", "lower_high_thresholds"):
            specs.append({"attack_type": "Control-Threshold Tampering", "component_type": "Tank",
                          "target_component": tank_id, "mode": mode,
                          "affected_sensors": ";".join(affected),
                          "associated_link_id": sorted(set(linked_assets))[0] if linked_assets else None})

    for tank_id in sorted(network.tank_name_list):
        associated = next((spec for spec in specs
                           if spec["component_type"] == "Tank" and spec["target_component"] == tank_id), None)
        drain_link = associated.get("associated_link_id") if associated else None
        if not drain_link:
            drain_link = _nearest_controllable_link(network, tank_id)
        specs.append({"attack_type": "Sensor Spoofing/Concealment", "component_type": "Tank",
                      "target_component": tank_id, "mode": "static_normal_level_during_depletion",
                      "affected_sensors": f"L_{tank_id}", "concealed_sensor": f"L_{tank_id}",
                      "associated_link_id": drain_link})

    if len(specs) < count:
        seeds = [
            spec for spec in specs
            if spec["attack_type"] in {"Actuator Manipulation", "Control-Threshold Tampering"}
        ]
        while len(specs) < count:
            source = seeds[(len(specs) - len(seeds)) % len(seeds)]
            clone = dict(source)
            clone["mode"] = source["mode"]
            clone["replicate"] = (len(specs) - len(seeds)) // len(seeds) + 1
            specs.append(clone)
    specs = specs[:count]
    for index, spec in enumerate(specs):
        duration = DURATION_CHOICES[index % len(DURATION_CHOICES)]
        latest_start = HORIZON_HOURS - duration
        start = int(rng.integers(24, latest_start + 1)) if latest_start >= 24 else 0
        spec.update({
            "attack_id": index + 1,
            "scenario_id": f"SYN-{index + 1:03d}",
            "dataset": "Synthetic Dataset",
            "duration_hours": duration,
            "start_hour": start,
            "end_hour_exclusive": start + duration,
            "start_time": (BASE_START + pd.Timedelta(hours=start)).isoformat(),
            "end_time": (BASE_START + pd.Timedelta(hours=start + duration - 1)).isoformat(),
        })
    return specs


def _configure_simulation(network: Any, duration_hours: int, start_hour: int) -> None:
    network.options.time.duration = int(duration_hours) * 3600
    network.options.time.hydraulic_timestep = 900
    network.options.time.report_timestep = 3600
    network.options.time.start_clocktime = float((start_hour % 24) * 3600)
    network.options.time.pattern_start = float(start_hour * 3600)
    network.options.hydraulic.demand_model = "PDA"
    network.options.hydraulic.minimum_pressure = 0.0
    network.options.hydraulic.required_pressure = 20.0


def _sensor_frame(network: Any, results: Any, sensors: list[str], start_hour: int) -> pd.DataFrame:
    times = pd.to_timedelta(np.asarray(results.node["pressure"].index, dtype=float), unit="s")
    frame = pd.DataFrame({"DATETIME": BASE_START + pd.Timedelta(hours=start_hour) + times})
    for sensor in sensors:
        if sensor.startswith("P_"):
            frame[sensor] = results.node["pressure"][sensor[2:]].to_numpy(dtype=float)
        elif sensor.startswith("L_"):
            tank = network.get_node(sensor[2:])
            frame[sensor] = results.node["head"][sensor[2:]].to_numpy(dtype=float) - float(tank.elevation)
        elif sensor.startswith("F_"):
            frame[sensor] = results.link["flowrate"][sensor[2:]].to_numpy(dtype=float) * 1000.0
        else:
            frame[sensor] = results.link["status"][sensor[2:]].to_numpy(dtype=float)
    return frame


def _set_initial_state(network: Any, results: Any, row_index: int) -> None:
    for tank_id in network.tank_name_list:
        tank = network.get_node(tank_id)
        level = float(results.node["head"][tank_id].iloc[row_index]) - float(tank.elevation)
        tank.init_level = float(np.clip(level, tank.min_level, tank.max_level))
    for link_id in network.link_name_list:
        status = float(results.link["status"][link_id].iloc[row_index])
        network.get_link(link_id).initial_status = (
            wntr.network.LinkStatus.Open if status >= 0.5 else wntr.network.LinkStatus.Closed
        )


def _remove_target_controls(network: Any, link_id: str) -> None:
    for name in list(network.control_name_list):
        control = network.get_control(name)
        actions = list(getattr(control, "_then_actions", ())) + list(getattr(control, "_else_actions", ()))
        if any(getattr(getattr(action, "_target_obj", None), "name", None) == link_id for action in actions):
            network.remove_control(name)


def _apply_scenario_fault(network: Any, spec: dict[str, Any]) -> dict[str, Any]:
    details: dict[str, Any] = {"mode": spec["mode"]}
    if spec["attack_type"] == "Actuator Manipulation":
        link_id = spec["target_component"]
        _remove_target_controls(network, link_id)
        force_open = spec["mode"] in {"force_on", "force_open"}
        network.get_link(link_id).initial_status = (
            wntr.network.LinkStatus.Open if force_open else wntr.network.LinkStatus.Closed
        )
        details["actuated_link"] = link_id
        return details

    if spec["attack_type"] == "Control-Threshold Tampering":
        tank_id = spec["target_component"]
        tank = network.get_node(tank_id)
        changed = []
        for control_name in network.control_name_list:
            control = network.get_control(control_name)
            condition = getattr(control, "_condition", None)
            source = getattr(condition, "_source_obj", None)
            if source is None or source.name != tank_id:
                continue
            relation = getattr(getattr(condition, "_relation", None), "name", "")
            threshold = float(condition._threshold)
            shift = 1.0
            if spec["mode"] == "raise_low_thresholds" and relation in {"lt", "le"}:
                condition._threshold = min(float(tank.max_level), threshold + shift)
            elif spec["mode"] == "lower_high_thresholds" and relation in {"gt", "ge"}:
                condition._threshold = max(float(tank.min_level), threshold - shift)
            else:
                continue
            changed.append(control_name)
        if not changed:
            direction = 1.0 if spec["mode"] == "raise_low_thresholds" else -1.0
            tank.init_level = float(np.clip(tank.init_level + direction, tank.min_level, tank.max_level))
            details["fallback"] = "initial tank level; CTOWN has no matching rule"
        details["changed_controls"] = changed
        return details

    link_id = spec.get("associated_link_id")
    if link_id not in network.link_name_list:
        link_id = _nearest_controllable_link(network, spec["target_component"])
    if link_id:
        _remove_target_controls(network, link_id)
        network.get_link(link_id).initial_status = wntr.network.LinkStatus.Open
        details["depletion_link"] = link_id
    return details


def _slice_results(results: Any, start_seconds: int, end_seconds: int) -> Any:
    return SimpleNamespace(
        node={key: table.loc[(table.index >= start_seconds) & (table.index <= end_seconds)]
              for key, table in results.node.items()},
        link={key: table.loc[(table.index >= start_seconds) & (table.index <= end_seconds)]
              for key, table in results.link.items()},
    )


def _simulate_scenario(base_network: Any, baseline_results: Any, spec: dict[str, Any], sensors: list[str]):
    start = int(spec["start_hour"])
    duration = int(spec["duration_hours"])
    end = start + duration
    fault_network = copy.deepcopy(base_network)
    _set_initial_state(fault_network, baseline_results, start)
    _configure_simulation(fault_network, duration, start)
    fault_details = _apply_scenario_fault(fault_network, spec)
    fault_results = wntr.sim.EpanetSimulator(fault_network).run_sim()

    base_sensor_frame = _sensor_frame(base_network, baseline_results, sensors, 0)
    prefix = base_sensor_frame.iloc[:start].copy()
    attack_sensor_frame = _sensor_frame(fault_network, fault_results, sensors, start)
    middle = attack_sensor_frame.iloc[:-1].copy()

    remainder = HORIZON_HOURS - end
    suffix = pd.DataFrame(columns=["DATETIME", *sensors])
    if remainder > 0:
        suffix_network = copy.deepcopy(base_network)
        _set_initial_state(suffix_network, fault_results, -1)
        _configure_simulation(suffix_network, remainder, end)
        suffix_results = wntr.sim.EpanetSimulator(suffix_network).run_sim()
        suffix = _sensor_frame(suffix_network, suffix_results, sensors, end).copy()
    frame = pd.concat([prefix, middle, suffix], ignore_index=True)
    frame = frame.drop_duplicates("DATETIME", keep="last").sort_values("DATETIME").reset_index(drop=True)
    frame[sensors] = frame[sensors].astype(float).round(2)

    labels = ((frame["DATETIME"] >= BASE_START + pd.Timedelta(hours=start))
              & (frame["DATETIME"] < BASE_START + pd.Timedelta(hours=end))).astype(np.int8)
    frame["ATT_FLAG"] = labels
    if spec["attack_type"] == "Sensor Spoofing/Concealment":
        sensor = spec["concealed_sensor"]
        baseline_value = float(base_sensor_frame.loc[start, sensor])
        frame.loc[labels == 1, sensor] = round(baseline_value, 2)

    physical_baseline = _slice_results(baseline_results, start * 3600, end * 3600)
    physical_fault = fault_results
    return frame, fault_network, fault_results, physical_baseline, physical_fault, fault_details


def _score_cpis(baseline_network: Any, fault_network: Any, baseline_results: Any, fault_results: Any,
                zones: dict[str, int], duration_hours: int) -> dict[str, Any]:
    baseline = _pda_physical_metrics(baseline_network, baseline_results, zones)
    incident = _pda_physical_metrics(fault_network, fault_results, zones)
    zone_count = len({zone for zone in zones.values() if zone > 0})
    excess_pressure = max(0.0, incident["pressure_violation_junction_hours"] - baseline["pressure_violation_junction_hours"])
    excess_unmet = max(0.0, incident["unmet_demand_volume_m3"] - baseline["unmet_demand_volume_m3"])
    excess_tank_loss = max(0.0, incident["tank_net_volume_loss_m3"] - baseline["tank_net_volume_loss_m3"])
    pressure = min(1.0, excess_pressure / max(1.0, incident["junction_count"] * duration_hours))
    flow = min(1.0, excess_tank_loss / max(incident["total_tank_capacity_m3"], 1e-12))
    service = min(1.0, excess_unmet / max(incident["required_demand_volume_m3"], 1e-12))
    zones_hit = []
    for zone in sorted(set(baseline["zone_unmet_demand_volume_m3"]) | set(incident["zone_unmet_demand_volume_m3"])):
        demand_delta = incident["zone_unmet_demand_volume_m3"].get(zone, 0.0) - baseline["zone_unmet_demand_volume_m3"].get(zone, 0.0)
        pressure_delta = incident["zone_pressure_head_deficit_mh"].get(zone, 0.0) - baseline["zone_pressure_head_deficit_mh"].get(zone, 0.0)
        if demand_delta > 1e-6 or pressure_delta > 1e-6:
            zones_hit.append(zone)
    zone_factor = len(zones_hit) / max(zone_count, 1)
    factors = {"pressure_impact": pressure, "flow_impact": flow,
               "service_disruption": service, "infrastructure_criticality": zone_factor}
    return {
        "baseline": baseline,
        "incident": incident,
        "excess_unmet_demand_volume_m3": excess_unmet,
        "excess_tank_storage_loss_m3": excess_tank_loss,
        "impacted_dma_zones": zones_hit,
        "impact_factors": factors,
        **calculate_cpis(factors),
    }


def _model_predict(model: SpatioTemporalGNN, adjacency: torch.Tensor, windows: np.ndarray) -> np.ndarray:
    parts = []
    with torch.no_grad():
        for start in range(0, len(windows), 1024):
            batch = torch.from_numpy(np.ascontiguousarray(windows[start:start + 1024], dtype=np.float32))
            parts.append(model(batch, adj_matrix=adjacency).numpy())
    return np.concatenate(parts, axis=0)


def _bootstrap_ci(values: list[float], rng: np.random.Generator, samples: int = 2000) -> dict[str, float | None]:
    values = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(values) == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "scenario_count": 0}
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)), "scenario_count": int(len(values))}


def _recovery_for_scenario(spec: dict[str, Any], incident_network: Any, fault_results: Any,
                           detection_offset: int, sensors: list[str], zones: dict[str, int],
                           recovery_config: dict[str, Any], rng: np.random.Generator,
                           top_shap_sensor: str | None) -> dict[str, Any]:
    from src.recovery.optimizer import _snapshot_at_results

    detection_timestamp = BASE_START + pd.Timedelta(hours=spec["start_hour"] + detection_offset)
    index = int(np.argmin(np.abs(np.asarray(fault_results.node["pressure"].index) - detection_offset * 3600)))
    incident = _snapshot_at_results(incident_network, fault_results, index, detection_timestamp)
    target = spec["target_component"]
    if spec["component_type"] in {"Pump", "Valve"}:
        anchor_link = target
    elif spec.get("associated_link_id"):
        anchor_link = spec["associated_link_id"]
    elif top_shap_sensor and top_shap_sensor.startswith(("F_", "S_")):
        anchor_link = top_shap_sensor[2:]
    else:
        anchor_link = _nearest_controllable_link(incident, target)
    if anchor_link not in incident.link_name_list:
        anchor_link = _nearest_controllable_link(incident, str(target))
    if not anchor_link:
        return {"status": "NO_CONTROLLABLE_LINK", "detection_timestamp": detection_timestamp.isoformat()}

    anchor = incident.get_link(anchor_link)
    affected_nodes = [anchor.start_node_name, anchor.end_node_name]
    candidates = generate_recovery_candidates(incident, anchor_link, zones, affected_nodes, recovery_config)
    reference = copy.deepcopy(incident)
    duration = max(1, HORIZON_HOURS - spec["start_hour"] - detection_offset)
    config = {**recovery_config, "simulation_duration_hours": duration,
              "hydraulic_timestep_minutes": 60, "demand_model": "PDA",
              "minimum_pressure_m": 0.0, "required_pressure_m": 20.0}
    optimized = optimize_recovery_actions(reference, incident, candidates, affected_nodes, recovery_config=config)
    evaluations = optimized.get("evaluations", [])
    feasible = [row for row in evaluations if row.get("status") == "SIMULATED" and row.get("feasible")]
    random_action = feasible[int(rng.integers(0, len(feasible)))] if feasible else None
    baseline_results = _simulate(copy.deepcopy(incident), config, "synthetic_no_action")
    baseline = _pda_physical_metrics(incident, baseline_results, zones)

    if top_shap_sensor and top_shap_sensor.startswith(("F_", "S_")):
        naive_link = top_shap_sensor[2:]
    elif top_shap_sensor and top_shap_sensor.startswith(("L_", "P_")):
        naive_link = _nearest_controllable_link(incident, top_shap_sensor[2:])
    else:
        naive_link = anchor_link
    if naive_link not in incident.link_name_list:
        naive_link = anchor_link
    scenarios = {
        "baseline_1_naive_isolation": [{"operation": "set_link_status", "link_id": naive_link, "status": "Closed"}],
    }
    if random_action:
        scenarios["baseline_2_random_feasible"] = random_action.get("operations", [])
    best = optimized.get("recommended_action") or optimized.get("best_available_action")
    if best:
        scenarios["optimizer_best"] = best.get("operations", [])

    outcomes = {}
    for name, operations in scenarios.items():
        model = copy.deepcopy(incident)
        _apply_operations(model, operations)
        results = _simulate(model, config, f"synthetic_{name}")
        metrics = _pda_physical_metrics(model, results, zones)
        outcomes[name] = {
            "action_id": (random_action if name == "baseline_2_random_feasible" else best if name == "optimizer_best" else None),
            "unmet_demand_volume_m3": metrics["unmet_demand_volume_m3"],
            "pressure_violation_junction_hours": metrics["pressure_violation_junction_hours"],
            "unmet_demand_reduction_pct": 100 * (baseline["unmet_demand_volume_m3"] - metrics["unmet_demand_volume_m3"]) / max(baseline["unmet_demand_volume_m3"], 1e-12),
            "pressure_violation_reduction_pct": 100 * (baseline["pressure_violation_junction_hours"] - metrics["pressure_violation_junction_hours"]) / max(baseline["pressure_violation_junction_hours"], 1e-12),
        }
        if outcomes[name]["action_id"] is not None:
            outcomes[name]["action_id"] = outcomes[name]["action_id"].get("action_id")
    return {
        "status": optimized.get("status"),
        "detection_timestamp": detection_timestamp.isoformat(),
        "anchor_link": anchor_link,
        "naive_isolated_link": naive_link,
        "candidate_action_count": len(candidates),
        "ranked_actions": [
            {key: evaluation.get(key) for key in ("action_id", "action_type", "label", "objective_score", "service_continuity", "hydraulic_stability", "feasible", "status")}
            for evaluation in sorted(evaluations, key=lambda row: (row.get("status") != "SIMULATED", not bool(row.get("feasible")), -(row.get("objective_score") or -1)))
        ],
        "baseline_0_no_action": {"unmet_demand_volume_m3": baseline["unmet_demand_volume_m3"],
                                 "pressure_violation_junction_hours": baseline["pressure_violation_junction_hours"]},
        **outcomes,
    }


def run_pipeline(root: Path, output_dir: Path, count: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    loader = BATADALDatasetLoader()
    d03, sensors, _ = loader.load_csv(root / "data/raw/BATADAL_dataset03.csv")
    d03 = harmonise_precision(d03, sensors)
    original_attacks = pd.read_csv(root / "data/ground_truth_attacks.csv")
    splits = build_splits(root, original_attacks, harmonise=True)
    network_metadata = load_ctown_network(root / "data/raw/CTOWN.INP")
    base_network = copy.deepcopy(network_metadata.wn)
    graph = build_network_graph(network_metadata)
    adjacency = build_adjacency_matrix_from_graph(graph, sensors, max_hops=3)
    zones = assign_zones(graph, method="dma")
    baseline_network = copy.deepcopy(base_network)
    _configure_simulation(baseline_network, HORIZON_HOURS, 0)
    baseline_results = wntr.sim.EpanetSimulator(baseline_network).run_sim()
    baseline_sensor_frame = _sensor_frame(baseline_network, baseline_results, sensors, 0)

    specs = _scenario_specs(base_network, count, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios_dir = output_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    master_rows = []
    scenario_simulations = {}
    for spec in specs:
        frame, incident_network, fault_results, physical_baseline, physical_fault, fault_details = _simulate_scenario(
            base_network, baseline_results, spec, sensors
        )
        frame.to_csv(scenarios_dir / f"{spec['scenario_id']}.csv", index=False,
                     date_format="%d/%m/%y %H", float_format="%.4f")
        scenario_simulations[spec["scenario_id"]] = (incident_network, fault_results)
        master_rows.append({key: spec.get(key) for key in (
            "attack_id", "scenario_id", "dataset", "attack_type", "component_type", "target_component",
            "mode", "duration_hours", "start_hour", "start_time", "end_time", "affected_sensors",
            "associated_link_id", "replicate",
        )} | {"csv_path": f"results/synthetic_attacks/scenarios/{spec['scenario_id']}.csv",
            "fault_details": json.dumps(fault_details)})
    pd.DataFrame(master_rows).to_csv(output_dir.parent / "synthetic_attack_ground_truth.csv", index=False)

    signature_path = root / "results/sensitivity_signatures.npz"
    signatures = load_sensitivity_signatures(signature_path)
    splits = build_splits(root, original_attacks, harmonise=True)
    model_data = []
    for model_seed in SEEDS:
        model = SpatioTemporalGNN(in_features=len(sensors), window_size=12, hidden_dim=64, dropout=0.2, forecast_horizon=12)
        model_path = root / "models/seeds" / f"detection_gnn_A_CTOWN_seed{model_seed}.pt"
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()
        adjacency_cpu = adjacency.cpu()

        def predict(windows, model=model, adjacency_cpu=adjacency_cpu):
            outputs = []
            with torch.no_grad():
                for offset in range(0, len(windows), 1024):
                    batch = torch.from_numpy(np.ascontiguousarray(windows[offset:offset + 1024], dtype=np.float32))
                    outputs.append(model(batch, adj_matrix=adjacency_cpu).numpy())
            return np.concatenate(outputs, axis=0)

        variance, threshold = calibrate(predict, splits)
        model_data.append({"seed": model_seed, "predict": predict, "variance": variance, "threshold": threshold})

    recovery_config = _merge_recovery_config(yaml_safe_load(root / "configs/config.yaml").get("recovery", {}))
    scenario_records = []
    for spec in specs:
        incident_network, fault_results = scenario_simulations[spec["scenario_id"]]
        csv_path = scenarios_dir / f"{spec['scenario_id']}.csv"
        frame = pd.read_csv(csv_path)
        frame["DATETIME"] = pd.to_datetime(frame["DATETIME"], dayfirst=True)
        normalized = splits.preprocessor.handle_missing_values(frame.copy(), sensors)
        x, y, _ = splits.preprocessor.create_forecasting_windows(
            splits.preprocessor.transform_features(normalized, sensors), forecast_horizon=12
        )
        labels = frame["ATT_FLAG"].to_numpy(dtype=np.int8)[TARGET_OFFSET:]
        evidence_rows = []
        detection_records = []
        localization_records = []
        for trained in model_data:
            predictions = trained["predict"](x)
            residual = np.square(predictions - y)[:, -1, :] / trained["variance"][-1]
            scores = residual.mean(axis=1)
            predicted = scores >= trained["threshold"]
            f1 = float(f1_score(labels, predicted, zero_division=0))
            precision = float(precision_score(labels, predicted, zero_division=0))
            recall = float(recall_score(labels, predicted, zero_division=0))
            attack_indices = np.flatnonzero(labels)
            peak_index = int(attack_indices[np.argmax(scores[attack_indices])]) if len(attack_indices) else int(np.argmax(scores))
            evidence_rows.append({"seed": trained["seed"], "score": float(scores[peak_index]),
                                  "residual": residual[peak_index], "f1": f1, "precision": precision,
                                  "recall": recall, "predicted": predicted,
                                  "detection_indices": np.flatnonzero(predicted & (labels == 1))})
            detection_records.append({"seed": trained["seed"], "f1": f1, "precision": precision, "recall": recall,
                                      "attack_window_recall": float(np.mean(predicted[labels == 1])) if labels.sum() else 0.0})

        e = np.mean([row["residual"] for row in evidence_rows], axis=0)
        mask = {spec["concealed_sensor"]} if spec.get("concealed_sensor") else set()
        rankings = signature_ranking(e, signatures["signatures"], signatures["asset_labels"], sensors, excluded_sensors=mask)
        target = f"{spec['component_type']} {spec['target_component']}"
        localization_records.append({"top1": float(rankings[0][0] == target),
                                     "top3": float(target in [asset for asset, _ in rankings[:3]]),
                                     "top3_assets": [asset for asset, _ in rankings[:3]],
                                     "masked_sensors": sorted(mask)})

        start = spec["start_hour"]
        duration = spec["duration_hours"]
        base_slice = _slice_results(baseline_results, start * 3600, (start + duration) * 3600)
        cpis = _score_cpis(baseline_network, incident_network, base_slice, physical_fault, zones, duration)
        found_indices = [int(index) for row in evidence_rows for index in row["detection_indices"]]
        if found_indices:
            target_index = int(np.median(found_indices)) + TARGET_OFFSET
            detect_offset = max(0, min(duration, target_index - start))
            recovery = _recovery_for_scenario(
                spec, incident_network, fault_results, detect_offset, sensors, zones,
                recovery_config, rng, str(sensors[int(np.argmax(e))]),
            )
        else:
            recovery = {"status": "NOT_DETECTED", "detection_timestamp": None,
                        "baseline_0_no_action": None, "baseline_1_naive_isolation": None,
                        "baseline_2_random_feasible": None, "optimizer_best": None}

        scenario_records.append({
            "scenario_id": spec["scenario_id"],
            "attack_id": spec["attack_id"],
            "attack_type": spec["attack_type"],
            "target": target,
            "duration_hours": duration,
            "start_time": spec["start_time"],
            "detection": {
                "seed_metrics": detection_records,
                "mean_f1": float(np.mean([row["f1"] for row in detection_records])),
                "mean_precision": float(np.mean([row["precision"] for row in detection_records])),
                "mean_recall": float(np.mean([row["recall"] for row in detection_records])),
            },
            "sensitivity_localization": localization_records[0],
            "cpis": cpis,
            "recovery": recovery,
        })

    def scenario_means(key_path, subkey=None):
        vals = []
        for row in scenario_records:
            value = row
            for key in key_path:
                value = value.get(key) if isinstance(value, dict) else None
            if subkey:
                value = value.get(subkey) if isinstance(value, dict) else None
            if value is not None and np.isfinite(float(value)):
                vals.append(float(value))
        return _bootstrap_ci(vals, rng)

    output = {
        "generator": {"scenario_count": count, "random_seed": seed, "duration_hours": list(DURATION_CHOICES),
                      "horizon_hours": HORIZON_HOURS, "model_seeds": list(SEEDS),
                      "attack_families": ["Actuator Manipulation", "Control-Threshold Tampering", "Sensor Spoofing/Concealment"]},
        "confidence_intervals_95": {
            "detection_f1": scenario_means(["detection"], "mean_f1"),
            "detection_precision": scenario_means(["detection"], "mean_precision"),
            "detection_recall": scenario_means(["detection"], "mean_recall"),
            "method_e_top1": _bootstrap_ci([row["sensitivity_localization"]["top1"] for row in scenario_records], rng),
            "method_e_top3": _bootstrap_ci([row["sensitivity_localization"]["top3"] for row in scenario_records], rng),
            "physical_cpis": _bootstrap_ci([row["cpis"]["cpis_score"] for row in scenario_records], rng),
            "recovery_unmet_demand_reduction_pct": _bootstrap_ci([
                row["recovery"].get("optimizer_best", {}).get("unmet_demand_reduction_pct")
                for row in scenario_records if row["recovery"].get("optimizer_best")
            ], rng),
            "recovery_pressure_violation_reduction_pct": _bootstrap_ci([
                row["recovery"].get("optimizer_best", {}).get("pressure_violation_reduction_pct")
                for row in scenario_records if row["recovery"].get("optimizer_best")
            ], rng),
        },
        "units": {"unmet_demand_volume": "m^3", "pressure_head_deficit": "m", "pressure_violation": "junction-hours"},
        "scenario_count": len(scenario_records),
        "master_ground_truth_csv": "results/synthetic_attack_ground_truth.csv",
        "scenario_csv_directory": "results/synthetic_attacks/scenarios",
        "recommended_case_study": max(
            [row for row in scenario_records
             if row["recovery"].get("status") != "NOT_DETECTED"
             and row["detection"]["mean_recall"] > 0],
            key=lambda row: row["cpis"]["cpis_score"],
        )["scenario_id"],
        "scenarios": scenario_records,
    }
    report_path = output_dir / "synthetic_attack_pipeline_report.json"
    report_path.write_text(json.dumps(output, indent=2, default=float) + "\n", encoding="utf-8")
    return output


def yaml_safe_load(path: Path) -> dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-count", type=int, default=71)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "synthetic_attacks")
    args = parser.parse_args()
    if not 50 <= args.scenario_count <= 100:
        parser.error("--scenario-count must be between 50 and 100")
    report = run_pipeline(ROOT, args.output_dir, args.scenario_count, args.seed)
    print(json.dumps({"scenario_count": report["scenario_count"],
                      "ci95": report["confidence_intervals_95"],
                      "recommended_case_study": report["recommended_case_study"]}, indent=2))


if __name__ == "__main__":
    main()