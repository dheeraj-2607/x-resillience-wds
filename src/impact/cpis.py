"""Configurable Cyber-Physical Impact Score factors and severity categories."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any
import re


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


def parse_dominant_sensor_scores(summary: str) -> dict[str, float]:
    """Parse saved `sensor (type): residualx` episode summaries."""
    scores = {}
    for entry in str(summary).split(";"):
        match = re.match(r"\s*([^\s]+)\s+\([^)]*\):\s*([+-]?[\d.eE]+)x\s*$", entry)
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