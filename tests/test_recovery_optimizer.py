from __future__ import annotations

import copy

import wntr

from src.recovery.optimizer import optimize_recovery_actions


def _recovery_config() -> dict:
    return {
        "weights": {
            "service_continuity": 0.30,
            "hydraulic_stability": 0.25,
            "recovery_time": 0.20,
            "operational_cost": 0.10,
            "criticality_protection": 0.15,
        },
        "simulation_duration_hours": 3,
        "hydraulic_timestep_minutes": 60,
        "demand_model": "PDA",
        "minimum_pressure_m": 0.0,
        "required_pressure_m": 20.0,
        "hydraulic_stability_pressure_scale_m": 10.0,
        "service_continuity_target": 0.95,
        "hydraulic_stability_target": 0.9,
        "recovery_persistence_steps": 1,
        "maximum_operational_cost": 1.0,
    }


def _single_supply_network():
    network = wntr.network.WaterNetworkModel()
    network.add_reservoir("R1", base_head=100.0)
    network.add_junction("J1", base_demand=0.01, elevation=0.0)
    network.add_pipe("P1", "R1", "J1", length=1000.0, diameter=0.25, roughness=100.0)
    return network


def test_optimizer_simulates_actions_ranks_feasible_recovery_and_preserves_inputs():
    reference = _single_supply_network()
    incident = copy.deepcopy(reference)
    incident.get_link("P1").initial_status = wntr.network.LinkStatus.Closed
    actions = [
        {
            "action_id": "restore_supply",
            "action_type": "restore_component",
            "label": "Restore P1",
            "operational_cost": 0.1,
            "operations": [{"operation": "set_link_status", "link_id": "P1", "status": "Open"}],
        },
        {
            "action_id": "no_op",
            "action_type": "emergency_adjustment",
            "label": "No hydraulic change",
            "operational_cost": 0.0,
            "operations": [],
        },
    ]

    result = optimize_recovery_actions(reference, incident, actions, ["J1"], _recovery_config())

    assert result["status"] == "OPTIMIZED"
    assert result["recommended_action"]["action_id"] == "restore_supply"
    assert result["recommended_action"]["service_continuity"] == 1.0
    assert 0.0 <= result["recommended_action"]["hydraulic_stability"] <= 1.0
    assert result["recommended_action"]["recovery_time_hours"] == 0.0
    assert any(row["action_id"] == "no_op" and row["feasible"] is False for row in result["evaluations"])
    assert reference.get_link("P1").initial_status == wntr.network.LinkStatus.Open
    assert incident.get_link("P1").initial_status == wntr.network.LinkStatus.Closed


def test_pda_metrics_report_unmet_demand_as_physical_volume():
    from src.impact.cpis import _configure_pda, _pda_physical_metrics

    network = _single_supply_network()
    network.get_link("P1").initial_status = wntr.network.LinkStatus.Closed
    _configure_pda(network, duration_hours=1, timestep_minutes=60)
    results = wntr.sim.EpanetSimulator(network).run_sim()

    metrics = _pda_physical_metrics(network, results, {"J1": 1})

    assert metrics["required_demand_volume_m3"] == 36.0
    assert metrics["delivered_demand_volume_m3"] >= 0.0
    assert metrics["unmet_demand_volume_m3"] > 0.0
    assert metrics["pressure_violation_junction_hours"] >= 0.0
    assert metrics["mean_pressure_head_deficit_m"] >= 0.0