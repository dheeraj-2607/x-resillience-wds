"""Shared sensor-aware residual normalization utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

PHYSICAL_VARIANCE_FLOORS = {
    "P_": 0.05**2,
    "L_": 0.01**2,
    "F_": 0.50**2,
    "S_": 0.01,
}


def sensor_variance_floor(
    sensor_columns: Sequence[str], scaler: StandardScaler | MinMaxScaler
) -> np.ndarray:
    """Return physical instrument variance floors in the scaler's feature space.

    StandardScaler variance divides by ``scale_ ** 2``. MinMaxScaler transforms
    physical units by multiplying by ``scale_``, so its variance floor uses the
    corresponding multiplication.
    """
    physical_floors = []
    for sensor in sensor_columns:
        prefix = sensor[:2]
        if prefix not in PHYSICAL_VARIANCE_FLOORS:
            raise ValueError(f"No instrument variance floor configured for sensor {sensor!r}.")
        physical_floors.append(PHYSICAL_VARIANCE_FLOORS[prefix])

    physical_floors_array = np.asarray(physical_floors, dtype=float)
    scales = np.asarray(scaler.scale_, dtype=float)
    if scales.shape != physical_floors_array.shape or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("The fitted scaler must have one finite positive scale per sensor.")

    if isinstance(scaler, StandardScaler):
        return physical_floors_array / np.square(scales)
    return physical_floors_array * np.square(scales)


def calibrated_sensor_variance(
    squared_errors: np.ndarray,
    sensor_columns: Sequence[str],
    scaler: StandardScaler | MinMaxScaler,
) -> np.ndarray:
    """Floor empirical mean squared residuals at instrument precision limits."""
    empirical_variance = np.mean(np.asarray(squared_errors), axis=0)
    floor = sensor_variance_floor(sensor_columns, scaler)
    if empirical_variance.shape[-1] != len(sensor_columns):
        raise ValueError("The residual feature dimension must match sensor_columns.")
    return np.maximum(empirical_variance, floor)