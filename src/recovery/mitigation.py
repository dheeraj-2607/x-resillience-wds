"""Defensive operational response suggestions for WDS component categories."""

from __future__ import annotations


MITIGATION_STEPS = {
    "Pump Flow": [
        "Verify pump operating state and flow telemetry readings.",
        "Verify commanded flow rate and pump speed setpoints.",
        "Isolate the abnormal pump if operationally appropriate.",
        "Transfer demand to an available parallel or alternative supply route if supported by network topology.",
        "Monitor downstream junction pressure and connected tank storage levels.",
        "Restore normal operating conditions after resolving component disturbance.",
    ],
    "Pump Status": [
        "Verify pump controller status and power supply integrity.",
        "Compare commanded control state vs observed operational state.",
        "Switch to an available redundant pump if operationally appropriate.",
        "Monitor resulting pressure and flow dynamics in connected sub-networks.",
    ],
    "Valve Flow": [
        "Verify valve position actuator and flow control settings.",
        "Inspect valve assembly for unexpected throttling or bypass flow.",
        "Isolate affected valve section if operationally safe.",
        "Utilize alternative supply path where network topology supports it.",
    ],
    "Valve Status": [
        "Verify valve position command signal and actual mechanical state.",
        "Inspect PLC control logic and manual override status.",
        "Use an alternative supply route/path if physically available.",
        "Monitor downstream pressure sensors for hydraulic equilibrium.",
    ],
    "Junction Pressure": [
        "Verify pressure control equipment (PRVs) and pressure transducers.",
        "Inspect nearby pumps and valves for uncommanded state changes.",
        "Isolate affected network section if severe pressure deviation persists.",
        "Use alternative supply routing if supported by network topology.",
    ],
    "Tank Level": [
        "Verify tank level sensor calibration and physical inlet/outlet valve status.",
        "Inspect connected supply pumps for abnormal duty cycles.",
        "Balance hydraulic supply through available network storage routes.",
        "Maintain minimum operational reserve levels for emergency storage.",
    ],
}

SCENARIO_NAMES = {
    "Junction Pressure": "Simulated Pressure Manipulation / Pressure Setpoint Deviation",
    "Tank Level": "Simulated Tank-Level Manipulation / Level Sensor Deviation",
    "Pump Flow": "Simulated Pump Flow Manipulation",
    "Pump Status": "Simulated Pump State Manipulation",
    "Valve Flow": "Simulated Valve Flow Manipulation",
    "Valve Status": "Simulated Valve State Manipulation",
}


def get_operational_mitigation(category: str) -> list[str]:
    """Return a defensive checklist; unknown component categories fail explicitly."""
    try:
        return list(MITIGATION_STEPS[category])
    except KeyError as error:
        raise ValueError(f"Unsupported WDS component category: {category}") from error


def get_scenario_name(category: str, disturbance_type: str) -> str:
    """Return the display title for a synthetic disturbance scenario."""
    base_name = SCENARIO_NAMES.get(category)
    if base_name is None:
        raise ValueError(f"Unsupported WDS component category: {category}")
    return base_name