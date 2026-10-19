# Stage 2 Results

## Scope and experimental setup

Stage 2 adds physics-informed localization, pressure-dependent-demand (PDA) impact analysis, dynamic recovery comparisons, and a synthetic attack pipeline. The underlying network is `data/raw/CTOWN.INP`; WNTR simulations use `EpanetSimulator`, PDA demand, minimum pressure 0 m, and required pressure 20 m. Synthetic data are research scenarios, not field observations or attack evidence from an operating utility.

The generator produced 71 scenarios with seed `20261019`, sampled start times over a 168-hour horizon, and durations of 6, 12, 24, 48, or 72 hours. Scenario families were actuator manipulation (41), PLC tank-level threshold tampering (23), and tank-level concealment (7). Each sensor CSV uses BATADAL channel names, hourly timestamps, and `ATT_FLAG`; ground truth and per-scenario CSVs are in `synthetic_attack_ground_truth.csv` and `synthetic_attacks/scenarios/`.

Detection uses the five Stage 1 C-Town checkpoints (seeds 42, 123, 2024, 3407, 9999). Hydraulic sensitivity signatures use 22 assets, 43 channels, two fault modes per asset, and eight weekly start offsets. Tank T6 has no PLC threshold rule in the source INP, so its sensitivity case uses a documented initial-level perturbation fallback. For historical attacks, actuator replay uses available status telemetry and tank-level onset state; attacks without an observed actuator channel are treated as sensor/plant replay without invented commands.

## Table R4: SHAP and hydraulic signature localization

Source: `fair_localization_table_r4.csv`. Values are five-seed mean +/- standard deviation for strict Top-1/Top-3; concealment columns are the attack-type subset.

| Timing | Method | Top-1 (%) | Top-3 (%) | Concealment Top-1 (%) | Concealment Top-3 (%) |
|---|---|---:|---:|---:|---:|
| Oracle | B: SHAP alone | 17.1 +/- 6.4 | 28.6 +/- 5.1 | 3.3 | 13.3 |
| Oracle | E: Hydraulic signature | 21.4 +/- 5.1 | 41.4 +/- 6.0 | 16.7 | 36.7 |
| Detector | B: SHAP alone | 15.7 +/- 3.2 | 21.4 +/- 5.1 | 3.3 | 10.0 |
| Detector | E: Hydraulic signature | 17.1 +/- 3.9 | 32.9 +/- 6.4 | 6.7 | 30.0 |

Method E ranks the 22 assets by cosine similarity between the observed residual evidence and each hydraulic response signature. For spoof/concealment cases, the directly attacked sensor channel is removed from both vectors before matching. The signature matrix is stored in `sensitivity_signatures.npz`.

## Table R5: PDA physical CPIS and weight sensitivity

Source: `cpis_simulation_report.json`. Four dimensionless CPIS factors represent pressure-violation junction-hours, excess net tank-storage loss, excess unmet demand, and newly impacted DMA zones, normalized to [0, 1] against each paired normal replay. Physical volumes are retained in m^3; pressure head deficit is reported in m and m·h.

| Measure | Result |
|---|---:|
| Historical attack replays | 14 |
| Dirichlet weight vectors | 1,000 |
| Kendall tau vs default CPIS ranking, mean | 0.9929 |
| Kendall tau vs default CPIS ranking, median | 1.0000 |
| Kendall tau range | 0.9211–1.0000 |
| Spearman correlation, CPIS vs excess unmet-demand factor | 0.8639 |

Only attacks that changed simulated plant behavior relative to the paired baseline receive positive excess-impact factors. Sensor-only spoofing rows with no observed actuator change therefore have zero physical hydraulic delta in this replay design; they are not assigned fabricated hydraulic consequences.

## Table R6: Detection-timed recovery

Source: `recovery_benchmark_report.json`. Response begins from a WNTR state snapshot at the median observed C-Town detection delay across the five seeds. Baseline 0 continues the incident; Baseline 1 isolates the top-SHAP component (or a nearest isolatable link when needed); Baseline 2 samples an action only from optimizer actions actually marked feasible.

| Measure | Result |
|---|---:|
| Historical attacks | 14 |
| Attacks with a detection timestamp | 12 |
| Attacks not detected | 2 |
| Mean optimizer reduction in unmet-demand volume vs no action | -0.0015% |
| Mean optimizer reduction in pressure-violation junction-hours vs no action | 0.1025% |

The measured mean response benefit is negligible: the optimizer did not materially reduce unmet demand in these scenarios. Missed attacks have no post-detection response comparison; random feasible baselines are marked unavailable when no candidate meets the configured feasibility checks. These results do not establish operational recovery effectiveness.

## Synthetic pipeline: scenario-level 95% confidence intervals

Source: `synthetic_attacks/synthetic_attack_pipeline_report.json`. Intervals are percentile bootstrap intervals over 71 scenarios (2,000 resamples), with detection statistics averaged over five trained C-Town seeds per scenario.

| Metric | Mean | 95% CI |
|---|---:|---:|
| Detection F1 | 0.2381 | 0.1732–0.3104 |
| Detection precision | 0.3102 | 0.2362–0.3856 |
| Detection recall | 0.3249 | 0.2354–0.4190 |
| Method E localization Top-1 | 32.4% | 22.5–43.7% |
| Method E localization Top-3 | 53.5% | 42.3–64.8% |
| Physical CPIS | 0.0178 | 0.0127–0.0232 |
| Optimizer unmet-demand reduction | -0.0003% | -0.0007–0.0001% |
| Optimizer pressure-violation reduction | -0.3716% | [-1.0757%, -0.0025%] |

The synthetic results are conditional on this generated scenario mix and should not be read as generalization estimates for real attacks. Mean recovery effects are negligible and slightly adverse in the simulation. The selected dashboard walkthrough is `SYN-011`, chosen from scenarios with both detector hits and a completed recovery evaluation; it is not the maximum-impact scenario if that scenario was not detected.

## Explanation quality

`explanation_quality_report.json` reports top-five SHAP masking fidelity against 20 random five-sensor masks, pairwise SHAP top-five Jaccard across five seeds on a shared ensemble-selected attack window, and SHAP/LIME overlap. Mean SHAP gain over random masking was 190.51 anomaly-score units; mean cross-seed Jaccard was 0.470; mean SHAP/LIME Jaccard was 0.220 (mean intersection 1.69 of 5 sensors). Large score-drop dispersion is retained in the JSON rather than hidden by a single point estimate.

## Reproduction

Run from the repository root with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe scripts\generate_synthetic_attacks.py --scenario-count 71 --seed 20261019
.\.venv\Scripts\python.exe -m src.explainability.evaluate_explanations
.\.venv\Scripts\pytest.exe tests\ -v
```

The synthetic pipeline requires the Stage 1 C-Town seed checkpoints and `results/sensitivity_signatures.npz`. The historical Stage 1 results are documented in `STAGE1_RESULTS.md`; source benchmark artifacts and current summaries are indexed in `README.md` and `results/README.md`.
