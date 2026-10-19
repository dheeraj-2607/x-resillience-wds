"""Configurable Cyber-Physical Impact Score factors and severity categories."""

from __future__ import annotations

import ast
import copy
import json
import math
import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wntr
from scipy.stats import kendalltau, spearmanr
from wntr.network.controls import Comparison, Control, ControlAction, ControlPriority, SimTimeCondition

from src.network.ctown_loader import load_ctown_network
from src.network.graph_builder import build_network_graph
from src.network.zoning import assign_zones


CPIS_FACTOR_NAMES = (
    "pressure_impact",
    "flow_impact",
    "service_disruption",
    "infrastructure_criticality",
)
DEFAULT_IMPACT_CONFIG = {
    "weights": {
        "pressure_impact": 0.30,
        "flow_impact": 0.30,
        "service_disruption": 0.25,
        "infrastructure_criticality": 0.15,
    },
    "severity_thresholds": {"low": 0.25, "moderate": 0.50, "high": 0.75},
    "asset_criticality": {"pressure": 0.55, "tank": 0.75, "flow": 0.85, "status": 0.90},
}


def _saturate(value: float, reference: float) -> float:
    value = max(0.0, float(value))
    return value / (value + reference) if value > 0 else 0.0


def parse_dominant_sensor_scores(summary: str) -> dict[str, float]:
    """Parse saved `sensor (type): residualx` episode summaries."""
    scores = {}
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    for entry in str(summary).split(";"):
        match = re.match(rf"\s*([^\s]+)\s+\([^)]*\):\s*({number})x\s*$", entry)
        if match:
            scores[match.group(1)] = float(match.group(2))
    return scores


def derive_episode_impact_factors(
    sensor_scores: Mapping[str, float],
    duration_hours: float,
    threshold: float,
    impact_config: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Derive bounded pressure, flow, service and asset-criticality factors.

    Residuals are the episode-average sensor-normalized scores. Service
    disruption is estimated from episode duration and active sensor classes;
    it is a proxy because BATADAL provides no measured service-outage labels.
    Infrastructure criticality uses the configured sensor-class criticality
    prior, also provisional until a utility asset criticality inventory exists.
    """
    config = {**DEFAULT_IMPACT_CONFIG, **(impact_config or {})}
    criticality = {**DEFAULT_IMPACT_CONFIG["asset_criticality"], **config.get("asset_criticality", {})}
    if not math.isfinite(float(threshold)) or threshold <= 0:
        raise ValueError("threshold must be finite and greater than zero.")
    if not math.isfinite(float(duration_hours)) or duration_hours < 0:
        raise ValueError("duration_hours must be finite and nonnegative.")

    groups: dict[str, list[float]] = {name: [] for name in ("pressure", "tank", "flow", "status")}
    for sensor, raw_score in sensor_scores.items():
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(f"Sensor residual score must be finite: {sensor}")
        if sensor.startswith("P_"):
            groups["pressure"].append(max(0.0, score))
        elif sensor.startswith("L_"):
            groups["tank"].append(max(0.0, score))
        elif sensor.startswith("F_"):
            groups["flow"].append(max(0.0, score))
        elif sensor.startswith("S_"):
            groups["status"].append(max(0.0, score))

    group_means = {
        group: sum(values) / len(values) if values else 0.0
        for group, values in groups.items()
    }
    pressure_impact = _saturate(group_means["pressure"], threshold)
    flow_impact = _saturate(group_means["flow"], threshold)

    active_groups = sum(value >= threshold for value in group_means.values())
    sensor_class_coverage = active_groups / len(group_means)
    duration_factor = float(duration_hours) / (float(duration_hours) + 24.0)
    service_disruption = 0.6 * duration_factor + 0.4 * sensor_class_coverage

    asset_scores = []
    for group, values in groups.items():
        class_criticality = float(criticality.get(group, 0.0))
        if not 0.0 <= class_criticality <= 1.0:
            raise ValueError(f"asset_criticality.{group} must be between zero and one.")
        for value in values:
            if value >= threshold:
                asset_scores.append(class_criticality * _saturate(value, threshold))
    infrastructure_criticality = sum(asset_scores) / len(asset_scores) if asset_scores else 0.0

    return {
        "pressure_impact": pressure_impact,
        "flow_impact": flow_impact,
        "service_disruption": service_disruption,
        "infrastructure_criticality": infrastructure_criticality,
    }


def count_topology_nodes(value: object) -> int:
    """Count distinct mapped topology nodes from a CSV list or iterable."""
    nodes = value
    if isinstance(nodes, str):
        try:
            nodes = ast.literal_eval(nodes)
        except (SyntaxError, ValueError):
            nodes = nodes.split(";")

    if not isinstance(nodes, Iterable) or isinstance(nodes, (bytes, dict)):
        return 0

    return len({str(node).strip() for node in nodes if str(node).strip()})


def calculate_cpis(
    impact_factors: Mapping[str, float],
    impact_config: Mapping[str, Any] | None = None,
) -> dict[str, float | str]:
    """Compute weighted CPIS and roadmap severity category from four factors."""
    config = {**DEFAULT_IMPACT_CONFIG, **(impact_config or {})}
    weights = {**DEFAULT_IMPACT_CONFIG["weights"], **config.get("weights", {})}
    thresholds = {**DEFAULT_IMPACT_CONFIG["severity_thresholds"], **config.get("severity_thresholds", {})}
    if set(impact_factors) != set(CPIS_FACTOR_NAMES):
        raise ValueError(f"impact_factors must contain exactly: {', '.join(CPIS_FACTOR_NAMES)}.")

    normalized_weights = {name: float(weights[name]) for name in CPIS_FACTOR_NAMES}
    if any(not math.isfinite(value) or value < 0 for value in normalized_weights.values()):
        raise ValueError("CPIS weights must be finite and nonnegative.")
    weight_total = sum(normalized_weights.values())
    if weight_total <= 0:
        raise ValueError("At least one CPIS weight must be greater than zero.")
    normalized_weights = {name: value / weight_total for name, value in normalized_weights.items()}

    factors = {name: float(impact_factors[name]) for name in CPIS_FACTOR_NAMES}
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in factors.values()):
        raise ValueError("Every CPIS impact factor must be finite and between zero and one.")

    severity_cutoffs = [float(thresholds[key]) for key in ("low", "moderate", "high")]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in severity_cutoffs):
        raise ValueError("CPIS severity thresholds must be between zero and one.")
    if severity_cutoffs != sorted(severity_cutoffs):
        raise ValueError("CPIS severity thresholds must be ordered low <= moderate <= high.")

    score = sum(factors[name] * normalized_weights[name] for name in CPIS_FACTOR_NAMES)
    if score < severity_cutoffs[0]:
        category = "Low"
    elif score < severity_cutoffs[1]:
        category = "Moderate"
    elif score < severity_cutoffs[2]:
        category = "High"
    else:
        category = "Critical"

    return {
        "cpis_score": round(score, 4),
        "cpis_severity": category,
        **{f"cpis_{name}": round(value, 4) for name, value in factors.items()},
    }


def _configure_pda(network: Any, duration_hours: int, timestep_minutes: int = 15) -> None:
    network.options.time.duration = int(duration_hours) * 3600
    network.options.time.hydraulic_timestep = int(timestep_minutes) * 60
    network.options.time.report_timestep = int(timestep_minutes) * 60
    network.options.hydraulic.demand_model = "PDA"
    network.options.hydraulic.minimum_pressure = 0.0
    network.options.hydraulic.required_pressure = 20.0


def _remove_link_controls(network: Any, link_id: str) -> None:
    for name in list(network.control_name_list):
        control = network.get_control(name)
        actions = list(getattr(control, "_then_actions", ())) + list(getattr(control, "_else_actions", ()))
        if any(getattr(getattr(action, "_target_obj", None), "name", None) == link_id for action in actions):
            network.remove_control(name)


def _apply_observed_actuator_replay(
    network: Any,
    attack: Any,
    source_frame: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> list[str]:
    """Replay observed status-channel changes for actuator attacks only."""
    if "actuator" not in str(attack.attack_type).lower():
        return []
    affected = {sensor.strip() for sensor in str(attack.affected_sensors).split(";")}
    status_sensors = sorted(sensor for sensor in affected if sensor.startswith("S_"))
    tank_sensor = f"L_{attack.target_component}" if attack.component_type == "Tank" else None
    replay_sensors = set(status_sensors)
    if tank_sensor and tank_sensor in affected:
        replay_sensors.add(tank_sensor)
    if not replay_sensors:
        return []
    frame = source_frame[(source_frame["DATETIME"] >= start_time) & (source_frame["DATETIME"] <= end_time)]
    if frame.empty:
        return []

    replayed = []
    if tank_sensor in replay_sensors and tank_sensor in frame:
        tank_values = frame[["DATETIME", tank_sensor]].dropna().sort_values("DATETIME")
        if not tank_values.empty:
            tank = network.get_node(attack.target_component)
            tank.init_level = float(np.clip(
                float(tank_values.iloc[0][tank_sensor]),
                float(tank.min_level),
                float(tank.max_level),
            ))
            replayed.append(tank_sensor)
    for sensor in status_sensors:
        link_id = sensor[2:]
        if link_id not in network.link_name_list or sensor not in frame:
            continue
        observed = frame[["DATETIME", sensor]].dropna().drop_duplicates("DATETIME")
        if observed.empty:
            continue
        _remove_link_controls(network, link_id)
        first_status = wntr.network.LinkStatus.Open if float(observed.iloc[0][sensor]) >= 0.5 else wntr.network.LinkStatus.Closed
        network.get_link(link_id).initial_status = first_status
        previous = float(observed.iloc[0][sensor])
        for row in observed.iloc[1:].itertuples(index=False):
            observed_time = pd.Timestamp(row[0])
            value = float(row[1])
            if (value >= 0.5) == (previous >= 0.5):
                continue
            elapsed = max(0.0, (observed_time - start_time).total_seconds())
            status = wntr.network.LinkStatus.Open if value >= 0.5 else wntr.network.LinkStatus.Closed
            event = Control(
                SimTimeCondition(network, Comparison.ge, elapsed),
                ControlAction(network.get_link(link_id), "status", status),
                priority=ControlPriority.high,
            )
            network.add_control(f"attack_replay_{link_id}_{int(elapsed)}", event)
            previous = value
        replayed.append(sensor)
    return replayed


def _prepare_attack_network(
    inp_path: Path,
    attack: Any,
    source_frame: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    replay_actuators: bool,
    timestep_minutes: int,
) -> tuple[Any, int, list[str]]:
    network = wntr.network.WaterNetworkModel(str(inp_path))
    duration_hours = max(1, int(math.ceil((end_time - start_time).total_seconds() / 3600.0)))
    _configure_pda(network, duration_hours, timestep_minutes)
    network.options.time.start_clocktime = float((start_time.hour * 3600) + (start_time.minute * 60))
    network.options.time.pattern_start = float(
        (start_time.dayofweek * 24 + start_time.hour) * 3600 + start_time.minute * 60
    )
    replayed = (
        _apply_observed_actuator_replay(network, attack, source_frame, start_time, end_time)
        if replay_actuators
        else []
    )
    return network, duration_hours, replayed


def _tank_volume(network: Any, results: Any, tank_id: str, index: int) -> float:
    tank = network.get_node(tank_id)
    level = float(results.node["head"][tank_id].iloc[index]) - float(tank.elevation)
    return float(tank.get_volume(level))


def _pda_physical_metrics(
    network: Any,
    results: Any,
    zones: Mapping[str, int],
    required_pressure_m: float = 20.0,
) -> dict[str, Any]:
    junctions = list(network.junction_name_list)
    pressure = results.node["pressure"][junctions].to_numpy(dtype=float)
    delivered_rate = results.node["demand"][junctions].clip(lower=0).to_numpy(dtype=float)
    time_seconds = np.asarray(results.node["pressure"].index, dtype=float)
    step_seconds = np.diff(time_seconds, append=time_seconds[-1])
    step_hours = step_seconds / 3600.0

    required_rate = np.zeros_like(delivered_rate)
    for column_index, junction_id in enumerate(junctions):
        junction = network.get_node(junction_id)
        for demand in junction.demand_timeseries_list:
            required_rate[:, column_index] += np.asarray([demand.at(float(t)) for t in time_seconds])
    required_rate *= float(network.options.hydraulic.demand_multiplier)
    unmet_rate = np.maximum(required_rate - delivered_rate, 0.0)
    required_volume = float(np.sum(required_rate * step_seconds[:, None]))
    delivered_volume = float(np.sum(delivered_rate * step_seconds[:, None]))
    unmet_volume = float(np.sum(unmet_rate * step_seconds[:, None]))

    violation = pressure < required_pressure_m
    pressure_violation_junction_hours = float(np.sum(violation * step_hours[:, None]))
    head_deficit_m = np.maximum(required_pressure_m - pressure, 0.0)
    pressure_head_deficit_mh = float(np.sum(head_deficit_m * step_hours[:, None]))
    violating_values = head_deficit_m[violation]

    zone_unmet: dict[int, float] = {}
    zone_pressure_deficit: dict[int, float] = {}
    for column_index, junction_id in enumerate(junctions):
        zone = int(zones.get(junction_id, -1))
        if zone <= 0:
            continue
        zone_unmet[zone] = zone_unmet.get(zone, 0.0) + float(
            np.sum(unmet_rate[:, column_index] * step_seconds)
        )
        zone_pressure_deficit[zone] = zone_pressure_deficit.get(zone, 0.0) + float(
            np.sum(head_deficit_m[:, column_index] * step_hours)
        )

    tank_losses = {}
    tank_capacity = 0.0
    for tank_id in network.tank_name_list:
        tank = network.get_node(tank_id)
        tank_capacity += float(tank.get_volume(float(tank.max_level)) - tank.get_volume(float(tank.min_level)))
        first_volume = _tank_volume(network, results, tank_id, 0)
        last_volume = _tank_volume(network, results, tank_id, -1)
        tank_losses[tank_id] = first_volume - last_volume

    return {
        "duration_hours": float(step_hours.sum()),
        "junction_count": len(junctions),
        "required_demand_volume_m3": required_volume,
        "delivered_demand_volume_m3": delivered_volume,
        "unmet_demand_volume_m3": unmet_volume,
        "unmet_demand_fraction": float(np.clip(unmet_volume / required_volume, 0.0, 1.0)) if required_volume else 0.0,
        "pressure_violation_junction_hours": pressure_violation_junction_hours,
        "pressure_head_deficit_mh": pressure_head_deficit_mh,
        "mean_pressure_head_deficit_m": float(np.mean(violating_values)) if len(violating_values) else 0.0,
        "max_pressure_head_deficit_m": float(np.max(violating_values)) if len(violating_values) else 0.0,
        "tank_net_volume_loss_m3": float(max(0.0, sum(tank_losses.values()))),
        "tank_gross_depletion_m3": float(sum(max(0.0, loss) for loss in tank_losses.values())),
        "tank_net_volume_change_by_tank_m3": tank_losses,
        "total_tank_capacity_m3": tank_capacity,
        "zone_unmet_demand_volume_m3": zone_unmet,
        "zone_pressure_head_deficit_mh": zone_pressure_deficit,
    }


def run_pda_attack_replays(
    root: str | Path,
    impact_config: Mapping[str, Any] | None = None,
    weight_samples: int = 1000,
    random_seed: int = 42,
    timestep_minutes: int = 15,
) -> dict[str, Any]:
    """Replay each published attack against a matching normal PDA baseline."""
    root = Path(root)
    attack_table = pd.read_csv(root / "data" / "ground_truth_attacks.csv")
    dataset_paths = {
        "Dataset04": root / "data" / "raw" / "BATADAL_dataset04.csv",
        "Test Dataset": root / "data" / "raw" / "BATADAL_test_dataset.csv",
    }
    dataset_frames = {}
    for dataset_name, path in dataset_paths.items():
        frame = pd.read_csv(path)
        frame.columns = frame.columns.astype(str).str.strip()
        frame["DATETIME"] = pd.to_datetime(frame["DATETIME"], dayfirst=True)
        dataset_frames[dataset_name] = frame

    network_metadata = load_ctown_network(root / "data" / "raw" / "CTOWN.INP")
    hydraulic_graph = build_network_graph(network_metadata)
    zones = assign_zones(hydraulic_graph, method="dma")
    impact_config = dict(impact_config or DEFAULT_IMPACT_CONFIG)
    weights = {**DEFAULT_IMPACT_CONFIG["weights"], **impact_config.get("weights", {})}
    zone_count = len({zone for zone in zones.values() if zone > 0})
    rows = []

    for attack in attack_table.itertuples(index=False):
        if attack.dataset not in dataset_frames:
            continue
        start_time = pd.Timestamp(attack.start_time)
        end_time = pd.Timestamp(attack.end_time)
        frame = dataset_frames[attack.dataset]
        baseline_network, duration_hours, _ = _prepare_attack_network(
            root / "data" / "raw" / "CTOWN.INP", attack, frame, start_time, end_time,
            replay_actuators=False, timestep_minutes=timestep_minutes,
        )
        fault_network, _, replayed_sensors = _prepare_attack_network(
            root / "data" / "raw" / "CTOWN.INP", attack, frame, start_time, end_time,
            replay_actuators=True, timestep_minutes=timestep_minutes,
        )
        baseline_results = wntr.sim.EpanetSimulator(baseline_network).run_sim()
        fault_results = wntr.sim.EpanetSimulator(fault_network).run_sim()
        baseline = _pda_physical_metrics(baseline_network, baseline_results, zones)
        incident = _pda_physical_metrics(fault_network, fault_results, zones)

        pressure_denominator = max(1.0, incident["junction_count"] * duration_hours)
        excess_pressure_hours = max(
            0.0,
            incident["pressure_violation_junction_hours"] - baseline["pressure_violation_junction_hours"],
        )
        pressure_factor = float(np.clip(excess_pressure_hours / pressure_denominator, 0.0, 1.0))
        excess_tank_loss = max(0.0, incident["tank_net_volume_loss_m3"] - baseline["tank_net_volume_loss_m3"])
        storage_factor = float(np.clip(excess_tank_loss / max(incident["total_tank_capacity_m3"], 1e-12), 0.0, 1.0))
        excess_unmet_volume = max(
            0.0,
            incident["unmet_demand_volume_m3"] - baseline["unmet_demand_volume_m3"],
        )
        service_factor = float(np.clip(
            excess_unmet_volume / max(incident["required_demand_volume_m3"], 1e-12), 0.0, 1.0
        ))
        impacted_zones = []
        for zone in sorted(set(baseline["zone_unmet_demand_volume_m3"]) | set(incident["zone_unmet_demand_volume_m3"])):
            unmet_delta = incident["zone_unmet_demand_volume_m3"].get(zone, 0.0) - baseline["zone_unmet_demand_volume_m3"].get(zone, 0.0)
            pressure_delta = incident["zone_pressure_head_deficit_mh"].get(zone, 0.0) - baseline["zone_pressure_head_deficit_mh"].get(zone, 0.0)
            if unmet_delta > 1e-6 or pressure_delta > 1e-6:
                impacted_zones.append(zone)
        zone_factor = float(np.clip(len(impacted_zones) / max(zone_count, 1), 0.0, 1.0))
        factors = {
            "pressure_impact": pressure_factor,
            "flow_impact": storage_factor,
            "service_disruption": service_factor,
            "infrastructure_criticality": zone_factor,
        }
        physical_cpis = calculate_cpis(factors, impact_config)
        rows.append({
            "attack_id": int(attack.attack_id),
            "dataset": attack.dataset,
            "attack_type": attack.attack_type,
            "target": f"{attack.component_type} {attack.target_component}",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "actuator_status_sensors_replayed": replayed_sensors,
            "pda_baseline": baseline,
            "pda_incident": incident,
            "excess_pressure_violation_junction_hours": excess_pressure_hours,
            "excess_unmet_demand_volume_m3": excess_unmet_volume,
            "excess_tank_storage_loss_m3": excess_tank_loss,
            "impacted_dma_zones": impacted_zones,
            "impact_factors": factors,
            **physical_cpis,
        })

    factor_matrix = np.asarray([
        [row["impact_factors"][name] for name in CPIS_FACTOR_NAMES] for row in rows
    ], dtype=float)
    default_weight_vector = np.asarray([float(weights[name]) for name in CPIS_FACTOR_NAMES], dtype=float)
    default_weight_vector /= default_weight_vector.sum()
    default_scores = factor_matrix @ default_weight_vector
    unmet = factor_matrix[:, CPIS_FACTOR_NAMES.index("service_disruption")]
    rng = np.random.default_rng(random_seed)
    dirichlet_weights = rng.dirichlet(np.ones(len(CPIS_FACTOR_NAMES)), size=weight_samples)
    taus = []
    for vector in dirichlet_weights:
        sampled_scores = factor_matrix @ vector
        tau = kendalltau(default_scores, sampled_scores).statistic
        taus.append(float(tau) if np.isfinite(tau) else 0.0)
    spearman = spearmanr(default_scores, unmet).statistic if len(rows) > 1 else 1.0
    return {
        "simulation": {
            "engine": "WNTR EpanetSimulator",
            "demand_model": "PDA",
            "minimum_pressure_m": 0.0,
            "required_pressure_m": 20.0,
            "hydraulic_timestep_minutes": timestep_minutes,
            "attack_count": len(rows),
            "actuator_replay_policy": "Observed S_* status changes replayed only for Actuator Manipulation intervals; sensor-only attacks leave plant controls unchanged.",
        },
        "units": {
            "unmet_demand_volume_m3": "m^3",
            "required_demand_volume_m3": "m^3",
            "delivered_demand_volume_m3": "m^3",
            "tank_net_volume_loss_m3": "m^3",
            "pressure_violation_junction_hours": "junction-hours",
            "pressure_head_deficit_mh": "m*h",
            "mean_pressure_head_deficit_m": "m",
            "impact_factors": "dimensionless ratios in [0, 1]",
        },
        "default_weights": {name: float(weights[name]) for name in CPIS_FACTOR_NAMES},
        "weight_sensitivity": {
            "distribution": "Dirichlet(1,1,1,1)",
            "sample_count": int(weight_samples),
            "kendall_tau_vs_default_mean": float(np.mean(taus)) if taus else 1.0,
            "kendall_tau_vs_default_median": float(np.median(taus)) if taus else 1.0,
            "kendall_tau_vs_default_min": float(np.min(taus)) if taus else 1.0,
            "kendall_tau_vs_default_max": float(np.max(taus)) if taus else 1.0,
            "spearman_default_cpis_vs_unmet_demand": float(spearman) if np.isfinite(spearman) else 0.0,
        },
        "table_r5": rows,
    }


def main() -> None:
    import yaml

    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text(encoding="utf-8"))
    report = run_pda_attack_replays(root, config.get("impact", {}))
    output = root / "results" / "cpis_simulation_report.json"
    output.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"Wrote Table R5 to {output} ({len(report['table_r5'])} attacks)")


if __name__ == "__main__":
    main()