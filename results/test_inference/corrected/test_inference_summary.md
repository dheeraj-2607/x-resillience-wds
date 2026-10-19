# Corrected Held-Out Test Inference Summary

Execution of corrected held-out test inference on `BATADAL_test_dataset.csv` saved to `results/test_inference/corrected/`.

## Key Findings (Corrected Timeline)

- **Total Test Timesteps / Windows**: 2,066
- **Timestamp Range**: `2017-01-04 12:00` to `2017-04-01 00:00` (Strictly Monotonic)
- **Detected Anomalous Windows**: **147** (7.12%)
- **True Contiguous Anomaly Episodes**: **17** (Corrected from 23 fragmented pseudo-episodes)
- **Maximum Anomaly Score**: **1442.5920**

## Corrected Anomaly Episodes (Ranked by Severity)

CPIS is the configured weighted score of pressure impact, flow impact, service disruption, and infrastructure criticality. Categories use the configured Low/Moderate/High/Critical cutoffs. Values are prioritization aids, not probabilities; this run derives factors from all per-sensor episode-average residuals.

| Rank | Episode ID | Duration (h) | CPIS | Severity | Pressure | Flow | Service | Infrastructure |
| :---: | :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| 1 | `EP-09` | 52.0h | **0.902** | Critical | 0.987 | 0.999 | 0.810 | 0.691 |
| 2 | `EP-08` | 52.0h | **0.903** | Critical | 0.989 | 0.998 | 0.810 | 0.693 |
| 3 | `EP-14` | 2.0h | **0.551** | High | 0.437 | 0.953 | 0.246 | 0.481 |
| 4 | `EP-03` | 18.0h | **0.619** | High | 0.732 | 0.582 | 0.557 | 0.572 |
| 5 | `EP-10` | 3.0h | **0.465** | Moderate | 0.839 | 0.355 | 0.167 | 0.435 |
| 6 | `EP-15` | 6.0h | **0.546** | High | 0.637 | 0.583 | 0.420 | 0.502 |
| 7 | `EP-12` | 2.0h | **0.576** | High | 0.589 | 0.658 | 0.446 | 0.600 |
| 8 | `EP-04` | 2.0h | **0.488** | Moderate | 0.783 | 0.397 | 0.246 | 0.484 |
| 9 | `EP-05` | 2.0h | **0.455** | Moderate | 0.562 | 0.493 | 0.246 | 0.510 |
| 10 | `EP-11` | 1.0h | **0.440** | Moderate | 0.777 | 0.335 | 0.124 | 0.505 |
| 11 | `EP-02` | 1.0h | **0.392** | Moderate | 0.764 | 0.177 | 0.124 | 0.526 |
| 12 | `EP-13` | 1.0h | **0.482** | Moderate | 0.646 | 0.497 | 0.224 | 0.552 |
| 13 | `EP-06` | 1.0h | **0.413** | Moderate | 0.762 | 0.267 | 0.124 | 0.487 |
| 14 | `EP-16` | 1.0h | **0.499** | Moderate | 0.551 | 0.556 | 0.324 | 0.573 |
| 15 | `EP-17` | 1.0h | **0.407** | Moderate | 0.684 | 0.224 | 0.224 | 0.523 |
| 16 | `EP-01` | 1.0h | **0.468** | Moderate | 0.613 | 0.517 | 0.224 | 0.486 |
| 17 | `EP-07` | 1.0h | **0.424** | Moderate | 0.688 | 0.309 | 0.224 | 0.460 |
