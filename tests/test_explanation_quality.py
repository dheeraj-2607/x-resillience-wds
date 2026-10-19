from __future__ import annotations

import numpy as np
import pytest

from src.explainability.evaluate_explanations import _jaccard, _sensor_coefficients


def test_jaccard_overlap_for_top_five_sets():
    assert _jaccard({"P1", "P2", "P3"}, {"P2", "P3", "P4"}) == pytest.approx(0.5)
    assert _jaccard(set(), set()) == 1.0


def test_calibration_coefficients_reconstruct_channelwise_projection():
    reference = np.zeros((2, 2), dtype=float)
    attack = np.ones((2, 2), dtype=float)
    calibration = np.array([
        [[0.25, 0.5], [0.25, 0.5]],
        [[0.75, 0.1], [0.75, 0.1]],
    ])

    coefficients = _sensor_coefficients(calibration, reference, attack)
    assert np.allclose(coefficients, [[0.25, 0.5], [0.75, 0.1]])