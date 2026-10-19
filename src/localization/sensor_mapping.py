"""Map BATADAL sensor identifiers to physical C-Town components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def resolve_sensor_component(sensor_name: str, component_map: Mapping[str, Sequence[str]] | None = None) -> dict[str, Any]:
    """Resolve a sensor prefix to a component type and its mapped topology nodes."""
    if sensor_name.startswith("P_"):
        component_type, component_id = "Junction", sensor_name[2:]
    elif sensor_name.startswith("L_"):
        component_type, component_id = "Tank", sensor_name[2:]
    elif sensor_name.startswith(("F_PU", "S_PU")):
        component_type, component_id = "Pump", sensor_name.split("_", 1)[1]
    elif sensor_name.startswith(("F_V", "S_V")):
        component_type, component_id = "Valve", sensor_name.split("_", 1)[1]
    else:
        component_type, component_id = "Unknown", sensor_name

    mapped_nodes = list((component_map or {}).get(sensor_name, ()))
    return {
        "sensor": sensor_name,
        "component_type": component_type,
        "component_id": component_id,
        "mapped_nodes": mapped_nodes,
        "topology_region": f"DMA / Node {mapped_nodes[0]}" if mapped_nodes else "Unmapped",
    }


def map_sensors_to_components(
    sensor_names: Sequence[str], component_map: Mapping[str, Sequence[str]] | None = None
) -> list[dict[str, Any]]:
    """Resolve multiple sensor identifiers while preserving their input order."""
    return [resolve_sensor_component(name, component_map) for name in sensor_names]