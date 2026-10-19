"""Permutation Shapley values with antithetic sampling.

This is the estimator behind shap.PermutationExplainer: each sampled permutation is
walked forward and backward and the marginal contributions are averaged.
`n_permutations=1` reproduces the project's earlier max_evals = 2*(n+1) setting,
which is too noisy (two runs on the same window agree on only 3-4 of the top-5
sensors). Use n_permutations >= 50 for reported results.

The value function receives a (k, n) array of 0/1 coalition masks and must return
k scores. Calls are batched: one call per walk of n+1 coalitions.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def permutation_shapley(
    value_fn: Callable[[np.ndarray], np.ndarray],
    n_features: int,
    n_permutations: int = 50,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    phi = np.zeros(n_features)
    for _ in range(n_permutations):
        perm = rng.permutation(n_features)
        for order in (perm, perm[::-1]):
            masks = np.zeros((n_features + 1, n_features))
            for position, feature in enumerate(order):
                masks[position + 1:, feature] = 1.0
            values = np.asarray(value_fn(masks), dtype=float)
            phi[order] += np.diff(values)
    return phi / (2 * n_permutations)


def masked_window_value_fn(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    reference_window: np.ndarray,
    attack_window: np.ndarray,
    attack_target: np.ndarray,
    variance: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Score of the window where sensors in the coalition take their attack values and
    the rest keep the nearest-normal reference values (same game as the project's
    `_shap_attributions`). Score = mean over sensors of the last-horizon squared
    residual divided by the calibrated variance."""
    delta = attack_window - reference_window

    def value_fn(masks: np.ndarray) -> np.ndarray:
        masked = reference_window[None] + masks[:, None, :].astype(np.float32) * delta[None]
        predictions = predict_fn(masked.astype(np.float32))
        residual = np.square(predictions[:, -1, :] - attack_target[-1][None]) / variance[-1]
        return residual.mean(axis=1)

    return value_fn
