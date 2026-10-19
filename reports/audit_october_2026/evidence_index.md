# Evidence Index & Artifact Provenance Manifest

**Project**: X-Resilience-WDS  
**Audit Date**: 19 October 2026  
**Auditor**: Senior Systems Architect & Evidence Auditor  

This manifest catalogs all verified artifacts, SHA-256 checksums, timestamps, and the specific scientific claims supported by each artifact.

---

## 1. Raw Datasets & Hydraulic Input Files

| Artifact Path | Format | Size | SHA-256 Checksum | Supported Claim |
|:---|:---:|:---:|:---|:---|
| `data/raw/BATADAL_dataset03.csv` | CSV | 2.94 MB | `8ca6cb851242254d2605b5a53ba3ba5009e2213a545d65396615f1ce0b426e1b` | 1-year normal baseline operational data (8,737 hours; 0 attacks) |
| `data/raw/BATADAL_dataset04.csv` | CSV | 1.23 MB | `4746beab2cfcdb5e68c7fa197a7522aecb70189d61dcc3eb1f870d2ce387068b` | 6-month operational validation set with 7 real historical attacks |
| `data/raw/BATADAL_test_dataset.csv` | CSV | 383 KB | `184dfc1581df52193b2acb79164f7ce094f2bb00b1bc2e3c2e70050095065fa5` | 3-month evaluation set (2,089 hours; 7 published test attacks) |
| `data/raw/CTOWN.INP` | INP | 117 KB | `9198b0bb43f551b29bfb550e206ff95f1df220118fc0068c7f130cec78d58d9e` | Official EPANET C-Town network model (396 nodes, 444 links) |
| `data/ground_truth_attacks.csv` | CSV | 1.63 KB | `b3484c24388e25d2b67ad7e7a83d3f9b2c34455f6974db37b78bdfba3e6a9288` | Authoritative 14 BATADAL attacks table from Taormina et al. (2018) |

*Hash verification script: `scripts/verify_datasets.py` (Verified 100% MATCH on 19 Oct 2026).*

---

## 2. Model Checkpoints & Checkpoint Provenance

| Artifact Path | Parameter Count | SHA-256 Checksum | Supported Claim |
|:---|:---:|:---|:---|
| `models/detection_gnn.pt` | 95,236 | `8e85de2b871ff49bb0894f2eb97e3d9fd6087151e8f0ba7ee35006f890a93667` | Production $H=12$ SpatioTemporalGNN checkpoint |
| `models/detection_gnn_h12_candidate.pt` | 95,236 | `8e85de2b871ff49bb0894f2eb97e3d9fd6087151e8f0ba7ee35006f890a93667` | Candidate model verified byte-identical to production |

---

## 3. Experimental Reports & Output Artifacts

| Artifact Path | Format | Size | Key Scientific Finding / Supported Claim |
|:---|:---:|:---:|:---|
| `results/final_h12_validation_report.json` | JSON | 8.2 KB | $H=12$ GNN on Dataset04: Precision=0.5635, Recall=0.7900, F1=0.6578, ROC-AUC=0.9685 |
| `results/smote_baseline_report.json` | JSON | 1.5 KB | Leakage-free chronological SMOTE RF on Dataset04: Precision=0.0, Recall=0.0, F1=0.0, ROC-AUC=0.5230 |
| `results/localization_benchmark_report.json` | JSON | 134 KB | 14-attack localization: Proposed Top-1=21.4%, Top-3=21.4%, Mean Hop Distance=17.93 vs Baseline B=21.57 |
| `results/calibrated_threshold.json` | JSON | 222 B | Calibration threshold $\tau = 3.013382$ (99.8th percentile on normal calibration split) |
| `results/test_inference/corrected/test_anomaly_scores.csv` | CSV | 120 KB | 2,066 evaluated test windows; 147 flagged anomalies using instrument variance floors |
| `results/test_inference/corrected/test_anomaly_episodes.csv` | CSV | 4.8 KB | 17 contiguous operational anomaly episodes detected on the held-out test set |
| `results/sensor_topology_info.json` | JSON | 357 B | Topological sensor inventory: 12 Pressures, 7 Levels, 12 Flows, 12 Statuses = 43 Sensors |

---

## 4. Key Source Code & Test Suite Implementations

| Source File | Role & Subsystem | Test File Covering Module |
|:---|:---|:---|
| `src/detection/graph_model.py` | SpatioTemporalGNN PyTorch implementation | `tests/test_phase1b_detection.py` |
| `src/detection/preprocessing.py` | Chronological splitting & normal-only scaling | `tests/test_phase1b_detection.py` |
| `src/detection/smote_baseline.py` | Leakage-free chronological tree baselines | `tests/test_smote_baseline.py` |
| `src/explainability/xai.py` | Permutation SHAP attribution engine | `tests/test_phase2_services.py` |
| `src/network/graph_builder.py` | Directed flow graph from WNTR simulation | `tests/test_phase1_network.py` |
| `src/root_cause/assessment.py` | Reversed-Flow Personalized PageRank | `tests/test_phase2_services.py` |
| `src/root_cause/evaluate_localization.py` | 14-attack localization benchmark engine | `tests/test_localization_benchmark.py` |
| `src/impact/cpis.py` | Cyber-Physical Impact Score calculator | `tests/test_phase2_services.py` |
| `src/recovery/optimizer.py` | Multi-objective WNTR PDA recovery optimizer | `tests/test_recovery_optimizer.py` |
| `dashboard/app.py` | Streamlit interactive operational cockpit | `tests/test_dashboard.py` |
| `tests/test_reporting_integrity.py` | Gate enforcing zero fraudulent Table 5 claims | Run in automated test suite |
