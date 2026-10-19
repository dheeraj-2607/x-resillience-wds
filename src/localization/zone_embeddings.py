"""Attribution-pooled zone embeddings for post-detection localization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


SENSOR_TYPES = ("pressure", "tank", "flow", "status")
EMBEDDING_FEATURES = tuple(f"shap_{kind}" for kind in SENSOR_TYPES) + tuple(
    f"lime_{kind}" for kind in SENSOR_TYPES
)


def _sensor_type(sensor: str) -> str | None:
    if sensor.startswith("P_"):
        return "pressure"
    if sensor.startswith("L_"):
        return "tank"
    if sensor.startswith("F_"):
        return "flow"
    if sensor.startswith("S_"):
        return "status"
    return None


def compute_zone_embeddings(
    explanation_result: Mapping[str, Any],
    sensor_nodes_map: Mapping[str, Sequence[str]],
    dma_zones: Mapping[str, int],
    shap_weight: float = 0.7,
) -> dict[str, Any]:
    """Pool signed explanation magnitudes into DMA-specific sensor-type vectors.

    Link sensors spanning multiple DMAs contribute fractionally to each mapped
    zone. The returned score is the weighted share of explanation mass by zone;
    it is a ranking, not a calibrated probability or localization accuracy.
    """
    if not 0.0 <= shap_weight <= 1.0:
        raise ValueError("shap_weight must be between 0 and 1.")

    shap_frame = explanation_result.get("df_shap")
    lime_frame = explanation_result.get("df_lime")
    if shap_frame is None or not {"Sensor", "Abs SHAP"}.issubset(shap_frame.columns):
        raise ValueError("explanation_result.df_shap must include Sensor and Abs SHAP columns.")
    if lime_frame is None or not {"Sensor", "Abs LIME"}.issubset(lime_frame.columns):
        raise ValueError("explanation_result.df_lime must include Sensor and Abs LIME columns.")

    shap_magnitudes = dict(zip(shap_frame["Sensor"].astype(str), shap_frame["Abs SHAP"].astype(float)))
    lime_magnitudes = dict(zip(lime_frame["Sensor"].astype(str), lime_frame["Abs LIME"].astype(float)))

    zone_vectors: dict[int, np.ndarray] = {}
    zone_sensor_weights: dict[int, float] = {}
    for sensor, nodes in sensor_nodes_map.items():
        kind = _sensor_type(sensor)
        if kind is None:
            continue
        zones = sorted({int(dma_zones[node]) for node in nodes if node in dma_zones and int(dma_zones[node]) > 0})
        if not zones:
            continue

        shap_value = shap_magnitudes.get(sensor, 0.0)
        lime_value = lime_magnitudes.get(sensor, 0.0)
        if not np.isfinite(shap_value) or not np.isfinite(lime_value):
            raise ValueError(f"Explanation magnitudes must be finite for sensor {sensor}.")

        vector = np.zeros(len(EMBEDDING_FEATURES), dtype=float)
        type_index = SENSOR_TYPES.index(kind)
        vector[type_index] = max(0.0, shap_value)
        vector[len(SENSOR_TYPES) + type_index] = max(0.0, lime_value)
        membership_weight = 1.0 / len(zones)
        for zone in zones:
            zone_vectors.setdefault(zone, np.zeros(len(EMBEDDING_FEATURES), dtype=float))
            zone_vectors[zone] += vector * membership_weight
            zone_sensor_weights[zone] = zone_sensor_weights.get(zone, 0.0) + membership_weight

    if not zone_vectors:
        return {
            "method": "xai_attribution_pooled_dma_embedding",
            "embedding_features": list(EMBEDDING_FEATURES),
            "rankings": [],
            "top_zone": None,
        }

    shap_mass = {zone: float(vector[:4].sum()) for zone, vector in zone_vectors.items()}
    lime_mass = {zone: float(vector[4:].sum()) for zone, vector in zone_vectors.items()}
    total_shap = sum(shap_mass.values())
    total_lime = sum(lime_mass.values())
    active_shap_weight = shap_weight if total_shap > 0 else 0.0
    active_lime_weight = (1.0 - shap_weight) if total_lime > 0 else 0.0
    active_weight_total = active_shap_weight + active_lime_weight
    if active_weight_total > 0:
        active_shap_weight /= active_weight_total
        active_lime_weight /= active_weight_total

    rankings = []
    for zone, vector in zone_vectors.items():
        shap_share = shap_mass[zone] / total_shap if total_shap > 0 else 0.0
        lime_share = lime_mass[zone] / total_lime if total_lime > 0 else 0.0
        score = active_shap_weight * shap_share + active_lime_weight * lime_share
        rankings.append(
            {
                "zone_id": zone,
                "localization_score": round(score, 6),
                "shap_share": round(shap_share, 6),
                "lime_share": round(lime_share, 6),
                "mapped_sensor_weight": round(zone_sensor_weights[zone], 4),
                "embedding": [round(float(value), 6) for value in vector],
            }
        )

    rankings.sort(key=lambda row: (-row["localization_score"], row["zone_id"]))
    return {
        "method": "xai_attribution_pooled_dma_embedding",
        "embedding_features": list(EMBEDDING_FEATURES),
        "rankings": rankings,
        "top_zone": rankings[0]["zone_id"],
    }