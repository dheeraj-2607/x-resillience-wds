# X-Resilience-WDS Results & Research Artifact Index

This index distinguishes the verified five-seed Stage 1 benchmarks, Stage 2 physical studies, held-out inference, and historical ablations. Numeric claims in release documentation should be sourced from the JSON/CSV artifacts listed here.

## Stage 1: Detection and Localization

- `multiseed_detection_report.json` and `multiseed_detection_table.csv`: five seeds (42, 123, 2024, 3407, 9999), each with C-Town and Identity adjacency; Dataset04 and test detection metrics. Mean F1 is 0.480 / 0.315 for C-Town and 0.502 / 0.333 for Identity (Dataset04 / test).
- `fair_localization_report.json`, `fair_localization_records.json`, `fair_localization_table_oracle.csv`, and `fair_localization_table_detector.csv`: four localization methods and random baseline on 22 assets, under oracle and detector timing.
- `STAGE1_RESULTS.md`: setup, Tables R1–R3, measured findings, and figures.
- `smote_baseline_report.json`: separate supervised Random Forest Dataset04 holdout; not comparable as a normal-only GNN result.

## Stage 2: Hydraulic Impact, Sensitivity, and Recovery

- `sensitivity_signatures.npz`: 22 controllable-asset by 43 sensor hydraulic sensitivity signatures; 8 weekly time offsets and 24-hour PDA simulations.
- `fair_localization_table_r4.csv`: Method E hydraulic signature results versus Method B SHAP, including concealment-specific performance. Oracle Top-1/Top-3: E 21.4% / 41.4%, B 17.1% / 28.6%.
- `cpis_simulation_report.json`: 14 historical attack replays with paired normal PDA baselines, physical CPIS factors, m³ volumes, meter-based head deficits, and 1,000 Dirichlet weight draws.
- `recovery_benchmark_report.json`: dynamic recovery at observed detection time, comparing no action, top-SHAP naive isolation, random feasible response, and optimizer ranking. Two of 14 attacks were not detected; they have no response result.
- `explanation_quality_report.json`: SHAP top-five fidelity versus random masking, five-seed Jaccard stability, and SHAP/LIME agreement.
- `STAGE2_RESULTS.md`: physical-study setup, Tables R4–R6, synthetic pipeline statistics, limitations, and reproducibility commands.

## Stage 3: Synthetic Pipeline

- `synthetic_attack_ground_truth.csv`: master table for 71 scenarios.
- `synthetic_attacks/scenarios/SYN-*.csv`: BATADAL-formatted sensor traces and attack flags.
- `synthetic_attacks/synthetic_attack_pipeline_report.json`: 5-seed detection, Method E localization, PDA CPIS, recovery outcomes, and scenario-bootstrap 95% confidence intervals.
- Generated scenario families: actuator manipulation, tank control-threshold tampering, and tank-level concealment; durations span 6–72 hours.

## Held-Out Test Inference

`test_inference/corrected/` contains the saved test scores, contiguous episodes, sensor attribution, and calibrated threshold. The current threshold is 2.943899. Test inference is unsupervised; ground-truth intervals are used only in separate offline benchmarks.

Historical ablations and promotion audits remain available in their existing JSON/CSV/figure artifacts. Their numeric results are not interchangeable with the Stage 1 five-seed table. CPIS is a configured prioritization score, not a probability; synthetic hydraulic and recovery simulations are not field-validated operational outcomes.

## Verification

Run the full test suite from the repository root with `pytest tests/ -v`. Current release documentation is checked by `tests/test_reporting_integrity.py`.
