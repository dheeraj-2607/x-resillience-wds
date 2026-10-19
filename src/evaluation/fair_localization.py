"""Fair component-localization benchmark for the 14 BATADAL attacks.

Fixes the comparison problems found in src/root_cause/evaluate_localization.py:
  1. every method returns a FULL ranking, so top-3 is meaningful for all of them
     (baseline B was previously cut to one candidate);
  2. every method uses the SAME evidence: the 3 highest-scoring windows of the attack;
  3. every method ranks the SAME candidate set: the tanks, pumps and valves scored by
     rank_upstream_components (junctions are never attack targets);
  4. evidence windows come either from the true attack interval ("oracle" timing, what
     the project did before) or only from windows the detector flagged ("detector"
     timing, the realistic setting); an undetected attack counts as a miss;
  5. a random-guess baseline and a lenient score (any attacked component counts) are
     reported, and results are aggregated over several trained models (seeds).

Methods
  A_raw_residual            component score = max normalized residual of its own sensors
  B_shap_direct_no_graph    component score = sum |SHAP| of its own sensors (no graph)
  C_ppr_only_graph          reversed-flow personalized PageRank only (direct_shap_weight=0)
  D_proposed_ppr_plus_shap  the project's method (direct_shap_weight=0.5)
B vs D isolates what the graph adds; C shows the graph on its own.

The core is model-agnostic: pass any `predict_fn(windows)->(N, 12, F)`.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from math import comb
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from src.evaluation.labels import attacked_components, sensor_component
from src.evaluation.shapley import masked_window_value_fn, permutation_shapley
from src.root_cause.assessment import rank_upstream_components
from src.root_cause.sensitivity_signatures import signature_ranking

METHODS = (
    "A_raw_residual",
    "B_shap_direct_no_graph",
    "C_ppr_only_graph",
    "D_proposed_ppr_plus_shap",
    "E_hydraulic_signature",
)


def _explain(predict_fn, calibration_windows, window, target, variance, n_perm, seed):
    distances = np.mean(np.square(calibration_windows - window), axis=(1, 2))
    reference = calibration_windows[int(np.argmin(distances))]
    value_fn = masked_window_value_fn(predict_fn, reference, window, target, variance)
    return permutation_shapley(value_fn, window.shape[1], n_perm, np.random.default_rng(seed))


def _label(component_tuple) -> str:
    return f"{component_tuple[0]} {component_tuple[1]}"


def evaluate_model(
    *,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    variance: np.ndarray,
    threshold: float,
    calibration_windows: np.ndarray,
    datasets: Mapping[str, tuple[np.ndarray, np.ndarray, pd.Series]],
    attacks: pd.DataFrame,
    sensors: Sequence[str],
    directed_graph: nx.DiGraph,
    sensor_nodes: Mapping[str, set],
    dma_zones: Mapping[str, int],
    hydraulic_signatures: Mapping[str, Any],
    seed: int,
    n_permutations: int = 50,
    evidence_windows: int = 3,
) -> list[dict[str, Any]]:
    """Run all methods for one trained model. Returns one record per attack x timing."""
    signature_matrix = np.asarray(hydraulic_signatures["signatures"], dtype=float)
    signature_assets = list(hydraulic_signatures["asset_labels"])
    signature_sensors = list(hydraulic_signatures["sensors"])
    if signature_sensors != list(sensors):
        raise ValueError("Hydraulic signature sensor order must match the benchmark sensor order.")
    if signature_matrix.shape != (len(signature_assets), len(signature_sensors)):
        raise ValueError("Hydraulic signature dimensions do not match asset and sensor labels.")

    records = []
    for dataset_name, (windows, targets, timestamps) in datasets.items():
        predictions = predict_fn(windows)
        sensor_scores = np.square(predictions - targets)[:, -1, :] / variance[-1]
        window_scores = sensor_scores.mean(axis=1)
        for attack in attacks[attacks["dataset"] == dataset_name].itertuples(index=False):
            target = f"{attack.component_type} {attack.target_component}"
            involved = attacked_components(attack)
            in_attack = timestamps.between(
                pd.Timestamp(attack.start_time), pd.Timestamp(attack.end_time)
            ).to_numpy()
            for timing in ("oracle", "detector"):
                pool_mask = in_attack if timing == "oracle" else in_attack & (window_scores >= threshold)
                pool = np.flatnonzero(pool_mask)
                record = {
                    "seed": seed, "dataset": dataset_name, "attack_id": int(attack.attack_id),
                    "attack_type": str(attack.attack_type), "target": target,
                    "attacked_components": sorted(involved), "timing": timing,
                    "detected": bool(len(pool)),
                }
                if not len(pool):
                    for method in METHODS:
                        record.update({f"{method}_top1": False, f"{method}_top3": False,
                                       f"{method}_lenient_top1": False, f"{method}_lenient_top3": False})
                    records.append(record)
                    continue
                top = pool[np.argsort(window_scores[pool])[-evidence_windows:]]
                shap_abs = np.max(
                    [np.abs(_explain(predict_fn, calibration_windows, windows[i], targets[i],
                                     variance, n_permutations, seed + int(i))) for i in top],
                    axis=0,
                )
                raw = np.max(sensor_scores[top], axis=0)

                def ppr(attribution, weight):
                    result = rank_upstream_components(
                        directed_graph, sensor_nodes, dict(zip(sensors, attribution)),
                        dma_zones, direct_shap_weight=weight,
                    )
                    return [_label((c["component_type"], c["component_id"])) for c in result["ranked_components"]]

                proposed = ppr(shap_abs, 0.5)
                universe = list(proposed)

                def direct(attribution):
                    mass = Counter()
                    for sensor, value in zip(sensors, attribution):
                        component = sensor_component(sensor)
                        if component in universe:
                            mass[component] += float(value)
                    return sorted(universe, key=lambda c: (-mass[c], proposed.index(c)))

                rankings = {
                    "A_raw_residual": direct(raw),
                    "B_shap_direct_no_graph": direct(shap_abs),
                    "C_ppr_only_graph": ppr(shap_abs, 0.0),
                    "D_proposed_ppr_plus_shap": proposed,
                }
                record["n_candidates"] = len(universe)
                if set(universe) != set(signature_assets):
                    raise ValueError("Hydraulic signature assets must match the benchmark candidate set.")
                is_concealment = "concealment" in str(attack.attack_type).lower() or "spoof" in str(attack.attack_type).lower()
                masked_sensors = (
                    {
                        sensor for sensor in sensors
                        if sensor_component(sensor) == target
                    }
                    if is_concealment
                    else set()
                )
                signature_scores = signature_ranking(
                    shap_abs,
                    signature_matrix,
                    signature_assets,
                    list(sensors),
                    excluded_sensors=masked_sensors,
                )
                rankings["E_hydraulic_signature"] = [asset for asset, _ in signature_scores]
                record["E_hydraulic_cosine_scores"] = dict(signature_scores)
                record["E_masked_sensor_channels"] = sorted(masked_sensors)
                record["shap_top5_sensors"] = [sensors[i] for i in np.argsort(-shap_abs)[:5]]
                for method, ranking in rankings.items():
                    record[f"{method}_top3_list"] = ranking[:3]
                    record[f"{method}_rank_of_target"] = ranking.index(target) + 1 if target in ranking else None
                    record[f"{method}_top1"] = ranking[0] == target
                    record[f"{method}_top3"] = target in ranking[:3]
                    record[f"{method}_lenient_top1"] = ranking[0] in involved
                    record[f"{method}_lenient_top3"] = bool(set(ranking[:3]) & involved)
                records.append(record)
    return records


def summarise(records: Sequence[Mapping[str, Any]], attacks: pd.DataFrame) -> dict[str, Any]:
    """Mean and standard deviation over seeds, per timing; plus random-guess baselines."""
    seeds = sorted({r["seed"] for r in records})
    n_candidates = max(r.get("n_candidates", 0) for r in records)
    summary: dict[str, Any] = {"seeds": seeds, "n_candidates": n_candidates, "timing": {}}
    for timing in ("oracle", "detector"):
        rows = [r for r in records if r["timing"] == timing]
        block = {"attack_runs": len(rows), "detected_runs": sum(r["detected"] for r in rows), "methods": {}}
        for method in METHODS:
            stats = {}
            for metric in ("top1", "top3", "lenient_top1", "lenient_top3"):
                per_seed = [
                    100.0 * np.mean([bool(r[f"{method}_{metric}"]) for r in rows if r["seed"] == s])
                    for s in seeds
                ]
                stats[metric] = {"mean": float(np.mean(per_seed)),
                                 "std": float(np.std(per_seed, ddof=1)) if len(seeds) > 1 else 0.0}
            for kind in ("Actuator", "Spoof"):
                subset = [r for r in rows if kind in r["attack_type"]]
                for metric in ("top1", "top3"):
                    stats[f"{metric}_{kind.lower()}"] = 100.0 * np.mean(
                        [bool(r[f"{method}_{metric}"]) for r in subset]
                    )
            block["methods"][method] = stats
        b = [r for r in rows if r["detected"]]
        block["D_vs_B_top3"] = {
            "D_only": sum(r["D_proposed_ppr_plus_shap_top3"] and not r["B_shap_direct_no_graph_top3"] for r in b),
            "B_only": sum(r["B_shap_direct_no_graph_top3"] and not r["D_proposed_ppr_plus_shap_top3"] for r in b),
            "same": sum(r["B_shap_direct_no_graph_top3"] == r["D_proposed_ppr_plus_shap_top3"] for r in b),
        }
        summary["timing"][timing] = block
    sizes = [len(attacked_components(a)) for a in attacks.itertuples(index=False)]
    k = n_candidates
    summary["random_guess_pct"] = {
        "top1": 100.0 / k, "top3": 300.0 / k,
        "lenient_top1": 100.0 * float(np.mean([s / k for s in sizes])),
        "lenient_top3": 100.0 * float(np.mean([1 - comb(k - s, 3) / comb(k, 3) for s in sizes])),
    }
    return summary


def results_table(summary: Mapping[str, Any], timing: str = "oracle") -> pd.DataFrame:
    """Summary table (mean ± sd over seeds, %)."""
    rows = []
    for method, stats in summary["timing"][timing]["methods"].items():
        fmt = lambda m: f"{stats[m]['mean']:.1f} ± {stats[m]['std']:.1f}"
        rows.append({"method": method, "top1": fmt("top1"), "top3": fmt("top3"),
                     "lenient_top1": fmt("lenient_top1"), "lenient_top3": fmt("lenient_top3"),
                     "top1_actuator": round(stats["top1_actuator"], 1),
                     "top3_actuator": round(stats["top3_actuator"], 1),
                     "top1_concealment": round(stats["top1_spoof"], 1),
                     "top3_concealment": round(stats["top3_spoof"], 1)})
    r = summary["random_guess_pct"]
    rows.append({"method": "random_guess", "top1": f"{r['top1']:.1f}", "top3": f"{r['top3']:.1f}",
                 "lenient_top1": f"{r['lenient_top1']:.1f}", "lenient_top3": f"{r['lenient_top3']:.1f}",
                 "top1_actuator": None, "top3_actuator": None,
                 "top1_concealment": None, "top3_concealment": None})
    return pd.DataFrame(rows)


def method_e_comparison_table(summary: Mapping[str, Any]) -> pd.DataFrame:
    """Return R4: strict SHAP-vs-signature accuracy and concealment performance."""
    rows = []
    for timing in ("oracle", "detector"):
        methods = summary["timing"][timing]["methods"]
        for method in ("B_shap_direct_no_graph", "E_hydraulic_signature"):
            stats = methods[method]
            rows.append({
                "timing": timing,
                "method": method,
                "top1": f"{stats['top1']['mean']:.1f} ± {stats['top1']['std']:.1f}",
                "top3": f"{stats['top3']['mean']:.1f} ± {stats['top3']['std']:.1f}",
                "concealment_top1": round(stats["top1_spoof"], 1),
                "concealment_top3": round(stats["top3_spoof"], 1),
            })
    return pd.DataFrame(rows)
