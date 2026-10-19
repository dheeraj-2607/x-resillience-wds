from __future__ import annotations

import numpy as np
import pytest

from src.root_cause.sensitivity_signatures import signature_ranking


def test_signature_ranking_uses_cosine_similarity_and_excludes_compromised_sensor():
    sensors = ["L_T1", "F_PU1", "P_J1"]
    assets = ["Tank T1", "Pump PU1", "Valve V2"]
    signatures = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])

    assert signature_ranking(np.array([1.0, 0.0, 0.0]), signatures, assets, sensors)[0][0] == "Tank T1"
    masked = signature_ranking(
        np.array([1.0, 0.0, 0.0]),
        signatures,
        assets,
        sensors,
        excluded_sensors={"L_T1"},
    )
    assert {asset for asset, _ in masked} == set(assets)
    assert all(score == 0.0 for _, score in masked)


def test_signature_ranking_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="Evidence vector length"):
        signature_ranking(np.ones(2), np.ones((1, 3)), ["Tank T1"], ["L_T1", "P_J1", "F_PU1"])