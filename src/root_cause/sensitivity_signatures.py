"""Hydraulic sensitivity signatures for controllable C-Town assets."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wntr
from wntr.network.controls import Comparison

from src.detection.dataset_loader import BATADALDatasetLoader
from src.detection.preprocessing import DataPreprocessor
from src.detection.scoring import PHYSICAL_VARIANCE_FLOORS

SIMULATION_HORIZON_SECONDS = 24 * 60 * 60
START_OFFSETS_HOURS = tuple(range(0, 7 * 24, 21))
FLOW_TO_LITRES_PER_SECOND = 1000.0
TANK_THRESHOLD_SHIFT_FRACTION = 0.10


def candidate_asset_labels(network: Any) -> list[str]:
    """Return all controllable assets in stable type/name order."""
    return [
        *(f"Tank {name}" for name in sorted(network.tank_name_list)),
        *(f"Pump {name}" for name in sorted(network.pump_name_list)),
        *(f"Valve {name}" for name in sorted(network.valve_name_list)),
    ]


def hydraulic_sensor_vector(results: Any, network: Any, sensors: list[str]) -> np.ndarray:
    """Average a 24-hour WNTR result as the BATADAL 43-sensor vector.

    WNTR reports link flow in m^3/s; BATADAL flow channels are in L/s.
    The first result row is the initial condition and is omitted from the time mean.
    """
    values = []
    for sensor in sensors:
        if sensor.startswith("P_"):
            series = results.node["pressure"][sensor[2:]]
        elif sensor.startswith("L_"):
            tank = network.get_node(sensor[2:])
            series = results.node["head"][sensor[2:]] - float(tank.elevation)
        elif sensor.startswith("F_"):
            series = results.link["flowrate"][sensor[2:]] * FLOW_TO_LITRES_PER_SECOND
        elif sensor.startswith("S_"):
            series = results.link["status"][sensor[2:]]
        else:
            raise ValueError(f"Unsupported BATADAL sensor channel: {sensor}")
        channel = np.asarray(series.iloc[1:], dtype=float)
        values.append(float(np.nanmean(channel)))
    vector = np.asarray(values, dtype=float)
    if vector.shape != (len(sensors),) or not np.isfinite(vector).all():
        raise ValueError("Hydraulic result did not produce one finite mean per sensor.")
    return vector


def signature_ranking(
    evidence: np.ndarray,
    signatures: np.ndarray,
    asset_labels: list[str],
    sensors: list[str],
    excluded_sensors: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Rank assets by cosine similarity, optionally masking compromised sensors."""
    evidence = np.asarray(evidence, dtype=float)
    signatures = np.asarray(signatures, dtype=float)
    if evidence.shape != (len(sensors),):
        raise ValueError("Evidence vector length must match the sensor list.")
    if signatures.shape != (len(asset_labels), len(sensors)):
        raise ValueError("Signature matrix must have one row per asset and column per sensor.")
    if not np.isfinite(evidence).all() or not np.isfinite(signatures).all():
        raise ValueError("Evidence and signatures must contain only finite values.")

    keep = np.ones(len(sensors), dtype=bool)
    for sensor in excluded_sensors or set():
        if sensor in sensors:
            keep[sensors.index(sensor)] = False
    evidence = evidence[keep]
    signatures = signatures[:, keep]
    evidence_norm = float(np.linalg.norm(evidence))
    signature_norms = np.linalg.norm(signatures, axis=1)
    denominators = evidence_norm * signature_norms
    similarities = np.divide(
        signatures @ evidence,
        denominators,
        out=np.zeros(len(asset_labels), dtype=float),
        where=denominators > 0,
    )
    order = sorted(range(len(asset_labels)), key=lambda index: (-similarities[index], asset_labels[index]))
    return [(asset_labels[index], float(similarities[index])) for index in order]


def _control_actions(control: Any) -> list[Any]:
    return list(getattr(control, "_then_actions", ())) + list(getattr(control, "_else_actions", ()))


def _remove_asset_controls(network: Any, asset_id: str) -> list[str]:
    removed = []
    for name in list(network.control_name_list):
        control = network.get_control(name)
        if any(getattr(getattr(action, "_target_obj", None), "name", None) == asset_id
               for action in _control_actions(control)):
            network.remove_control(name)
            removed.append(name)
    return removed


def _apply_asset_fault(network: Any, asset_type: str, asset_id: str, mode: str) -> dict[str, Any]:
    if asset_type in {"Pump", "Valve"}:
        removed_controls = _remove_asset_controls(network, asset_id)
        status = wntr.network.LinkStatus.Open if mode.endswith("on") or mode.endswith("open") else wntr.network.LinkStatus.Closed
        network.get_link(asset_id).initial_status = status
        return {"mode": mode, "removed_controls": removed_controls}

    tank = network.get_node(asset_id)
    threshold_shift = (float(tank.max_level) - float(tank.min_level)) * TANK_THRESHOLD_SHIFT_FRACTION
    changed_controls = []
    relation_target = Comparison.lt if mode == "raise_low_thresholds" else Comparison.gt
    adjacent_relation = Comparison.le if mode == "raise_low_thresholds" else Comparison.ge
    for name in network.control_name_list:
        control = network.get_control(name)
        condition = getattr(control, "_condition", None)
        source = getattr(condition, "_source_obj", None)
        if getattr(source, "name", None) != asset_id:
            continue
        relation = getattr(condition, "_relation", None)
        if relation not in {relation_target, adjacent_relation}:
            continue
        threshold = float(condition._threshold)
        if mode == "raise_low_thresholds":
            condition._threshold = min(float(tank.max_level), threshold + threshold_shift)
        else:
            condition._threshold = max(float(tank.min_level), threshold - threshold_shift)
        changed_controls.append(name)

    if changed_controls:
        return {"mode": mode, "changed_controls": changed_controls, "threshold_shift": threshold_shift}

    if asset_id not in network.tank_name_list:
        raise ValueError(f"Tank {asset_id} does not exist in the hydraulic network.")
    direction = 1.0 if mode == "raise_low_thresholds" else -1.0
    original_level = float(tank.init_level)
    tank.init_level = float(np.clip(
        original_level + direction * threshold_shift,
        float(tank.min_level),
        float(tank.max_level),
    ))
    return {
        "mode": mode,
        "fallback": "initial tank level; no PLC rule references this tank",
        "initial_level_before": original_level,
        "initial_level_after": float(tank.init_level),
    }


def _simulate_day(base_network: Any, start_hour: int) -> tuple[Any, Any]:
    network = copy.deepcopy(base_network)
    network.options.time.duration = SIMULATION_HORIZON_SECONDS
    network.options.time.pattern_start = float(start_hour * 3600)
    network.options.time.start_clocktime = float((start_hour % 24) * 3600)
    results = wntr.sim.EpanetSimulator(network).run_sim()
    return network, results


def _calibration_sigmas(dataset_path: Path, sensors: list[str]) -> np.ndarray:
    loader = BATADALDatasetLoader()
    frame, dataset_sensors, _ = loader.load_csv(dataset_path)
    if dataset_sensors != sensors:
        if set(dataset_sensors) != set(sensors):
            raise ValueError("Calibration dataset sensor columns do not match the hydraulic signature sensors.")
        frame = frame[[*frame.columns.difference(sensors), *sensors]]
    frame[sensors] = frame[sensors].astype(float).round(2)
    calibration_start = int(0.8 * len(frame))
    preprocessor = DataPreprocessor(window_size=12)
    calibration = preprocessor.handle_missing_values(frame.iloc[calibration_start:].copy(), sensors)
    empirical_sigma = calibration[sensors].std(ddof=0).to_numpy(dtype=float)
    floor_sigma = np.asarray([
        np.sqrt(PHYSICAL_VARIANCE_FLOORS[sensor[:2]]) for sensor in sensors
    ])
    return np.maximum(empirical_sigma, floor_sigma)


def build_sensitivity_signatures(
    inp_path: str | Path,
    calibration_dataset_path: str | Path,
    output_path: str | Path,
    start_offsets_hours: tuple[int, ...] = START_OFFSETS_HOURS,
) -> dict[str, Any]:
    """Run balanced 24-hour asset faults and save a 22 x 43 magnitude matrix."""
    inp_path = Path(inp_path)
    calibration_dataset_path = Path(calibration_dataset_path)
    output_path = Path(output_path)
    network = wntr.network.WaterNetworkModel(str(inp_path))
    _, sensors, _ = BATADALDatasetLoader().load_csv(calibration_dataset_path)
    assets = candidate_asset_labels(network)
    if len(assets) != 22:
        raise ValueError(f"Expected 22 controllable assets, found {len(assets)}.")
    sigma = _calibration_sigmas(calibration_dataset_path, sensors)

    baselines: dict[int, np.ndarray] = {}
    for start_hour in start_offsets_hours:
        baseline_network, baseline_results = _simulate_day(network, start_hour)
        baselines[start_hour] = hydraulic_sensor_vector(baseline_results, baseline_network, sensors)

    signatures = []
    fault_metadata: dict[str, list[dict[str, Any]]] = {}
    for asset in assets:
        asset_type, asset_id = asset.split(" ", maxsplit=1)
        if asset_type in {"Pump", "Valve"}:
            modes = ("force_on", "force_off") if asset_type == "Pump" else ("force_open", "force_closed")
        else:
            modes = ("raise_low_thresholds", "lower_high_thresholds")

        deviations = []
        mode_details = []
        for mode in modes:
            for start_hour in start_offsets_hours:
                fault_network = copy.deepcopy(network)
                mode_details.append(_apply_asset_fault(fault_network, asset_type, asset_id, mode))
                fault_network.options.time.duration = SIMULATION_HORIZON_SECONDS
                fault_network.options.time.pattern_start = float(start_hour * 3600)
                fault_network.options.time.start_clocktime = float((start_hour % 24) * 3600)
                fault_results = wntr.sim.EpanetSimulator(fault_network).run_sim()
                fault_vector = hydraulic_sensor_vector(fault_results, fault_network, sensors)
                deviation = (fault_vector - baselines[start_hour]) / sigma
                deviations.append(np.abs(deviation))
        signatures.append(np.mean(np.stack(deviations), axis=0))
        fault_metadata[asset] = mode_details

    matrix = np.stack(signatures)
    if matrix.shape != (22, 43) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid sensitivity matrix shape or values: {matrix.shape}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        signatures=matrix,
        asset_labels=np.asarray(assets, dtype=str),
        sensors=np.asarray(sensors, dtype=str),
        sigma_calib=sigma,
        start_offsets_hours=np.asarray(start_offsets_hours, dtype=np.int64),
        simulation_horizon_hours=np.asarray(24, dtype=np.int64),
        aggregation=np.asarray("mean absolute standardized response across fault modes and starts"),
        fault_metadata_json=np.asarray(json.dumps(fault_metadata)),
    )
    return {
        "path": str(output_path),
        "shape": matrix.shape,
        "assets": assets,
        "sensors": sensors,
        "start_offsets_hours": list(start_offsets_hours),
        "fault_metadata": fault_metadata,
    }


def load_sensitivity_signatures(path: str | Path) -> dict[str, Any]:
    """Load and validate a previously generated sensitivity matrix."""
    with np.load(path, allow_pickle=False) as data:
        signatures = np.asarray(data["signatures"], dtype=float)
        asset_labels = data["asset_labels"].astype(str).tolist()
        sensors = data["sensors"].astype(str).tolist()
        sigma = np.asarray(data["sigma_calib"], dtype=float)
    if signatures.shape != (len(asset_labels), len(sensors)):
        raise ValueError("Stored signature matrix dimensions do not match its labels.")
    if not np.isfinite(signatures).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("Stored signatures and calibration scales must be finite; scales must be positive.")
    return {"signatures": signatures, "asset_labels": asset_labels, "sensors": sensors, "sigma_calib": sigma}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    result = build_sensitivity_signatures(
        root / "data/raw/CTOWN.INP",
        root / "data/raw/BATADAL_dataset03.csv",
        root / "results/sensitivity_signatures.npz",
    )
    print(f"Saved {result['shape']} sensitivity matrix to {result['path']}")