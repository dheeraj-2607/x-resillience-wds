"""Ground-truth labels for BATADAL evaluation.

Two sources are supported:
  * data/ground_truth_attacks.csv  -> hour-level attack intervals (Dataset04 and test set)
  * the official labelled test file -> its ATTACK_FLAG column (hour-level)

Labels are aligned to the forecast TARGET timestamp used by the H=12 detector:
window i covers rows [i, i+12) and its scored target is row i + 12 + 12 - 1 = i + 23.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WINDOW = 12
HORIZON = 12
TARGET_OFFSET = WINDOW + HORIZON - 1  # 23

COMPONENT_PREFIXES = (("L_", "Tank"), ("F_PU", "Pump"), ("S_PU", "Pump"), ("F_V", "Valve"), ("S_V", "Valve"))


def target_timestamps(frame: pd.DataFrame, timestamp_col: str = "DATETIME") -> pd.Series:
    """Timestamp of the scored (last-horizon) target for every forecasting window."""
    return frame[timestamp_col].iloc[TARGET_OFFSET:].reset_index(drop=True)


def interval_labels(timestamps: pd.Series, attacks: pd.DataFrame) -> np.ndarray:
    """1 where a target timestamp lies inside any [start_time, end_time] interval (inclusive)."""
    labels = np.zeros(len(timestamps), dtype=np.int8)
    for attack in attacks.itertuples(index=False):
        inside = timestamps.between(pd.Timestamp(attack.start_time), pd.Timestamp(attack.end_time))
        labels[inside.to_numpy()] = 1
    return labels


def official_test_labels(path: str | Path) -> np.ndarray:
    """ATTACK_FLAG of the official labelled BATADAL test file, aligned to window targets."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    if "ATTACK_FLAG" not in raw.columns:
        raise ValueError(
            f"{path} has no ATTACK_FLAG column. Use the labelled BATADAL test file "
            "(SHA-256 8a786183003ed3fe0fcec287ab4c585c5ec9c32adadc79586720070583c2e59c)."
        )
    return raw["ATTACK_FLAG"].to_numpy()[TARGET_OFFSET:].astype(np.int8)


def sensor_component(sensor: str) -> str | None:
    """Map a BATADAL sensor to the controllable component it measures ('Tank T1', 'Pump PU3', ...)."""
    for prefix, kind in COMPONENT_PREFIXES:
        if sensor.startswith(prefix):
            return f"{kind} {sensor.split('_', 1)[1]}"
    return None  # pressure sensors (P_J*) measure junctions, never attack targets


def attacked_components(attack_row) -> set[str]:
    """All controllable components involved in an attack (for lenient scoring).

    Uses the `attacked_components` column if present (semicolon list such as
    'Tank T7;Pump PU10;Pump PU11'); otherwise the primary target plus the components
    of `affected_sensors`.
    """
    primary = f"{attack_row.component_type} {attack_row.target_component}"
    explicit = getattr(attack_row, "attacked_components", None)
    if isinstance(explicit, str) and explicit.strip():
        return {c.strip() for c in explicit.split(";") if c.strip()} | {primary}
    sensors = str(getattr(attack_row, "affected_sensors", "") or "").split(";")
    return {c for c in map(sensor_component, (s.strip() for s in sensors)) if c} | {primary}
