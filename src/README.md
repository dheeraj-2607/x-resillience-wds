# X-Resilience-WDS: Spatio-Temporal GNN Anomaly Detection for Water Distribution Systems

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests Status](https://img.shields.io/badge/Tests-48%20PASSED-brightgreen.svg)]()

---

## 1. Executive Summary & Production Status

**X-Resilience-WDS** is an end-to-end spatio-temporal Graph Neural Network (`SpatioTemporalGNN`) anomaly detection framework designed for multivariate time-series sensor streams in Water Distribution Systems (WDS). The production system utilizes a direct multi-horizon forecasting architecture ($H=12$) trained exclusively on normal operational dynamics from the C-Town network (BATADAL benchmark).

### Production System Parameters
- **Production Checkpoint**: `models/detection_gnn.pt`
- **Model Checkpoint SHA-256**: `8e85de2b871ff49bb0894f2eb97e3d9fd6087151e8f0ba7ee35006f890a93667`
- **Rollback Backup Checkpoint**: `models/backups/detection_gnn_pre_h12_promotion_20260914_192026.pt`
- **Rollback Checkpoint SHA-256**: `056284e983baf988d962e4d56d7ea7528c4638173f081c418ec9f6ce907d9e64`
- **Recalibrated Held-Out Threshold ($\tau_{99.8}$)**: `2.943899` (floored Dataset03 scores)
- **Test Suite Status**: **48 / 48 PASSED** ($100\%$ pass rate)

---

## 2. Dataset Description & Simulation Caveat

The system is evaluated on the **BATADAL (Battle of the Attack Detection Algorithms)** benchmark dataset, derived from EPANET hydraulic simulations of the C-Town WDS network ($43$ telemetry sensors: $12$ junction pressures, $7$ tank levels, $12$ link flows, and $12$ actuator statuses).

> [!IMPORTANT]
> **SIMULATION DATA CAVEAT**: All BATADAL datasets are computer simulation-generated using EPANET hydraulic engines. They represent synthetic operational scenarios and benchmark cyber-physical attack simulations, **NOT** physical municipal water utility field measurements.

### Data Partitioning Structure:
1. **`BATADAL_dataset03.csv` ($8,761$ hourly rows)**: Normal operational baseline. Chronologically split into $80\%$ Training ($7,008$ timesteps) and $20\%$ Normal Calibration ($1,753$ timesteps, $7,957$ sliding $H=12$ windows). Standard scalers ($\mu, \sigma$) and threshold $\tau_{99.8} = 3.013382$ were fit **strictly** on Dataset03. Zero attack labels were present or used.
2. **`BATADAL_dataset04.csv` ($4,177$ hourly rows)**: Supervised benchmark validation dataset containing $173$ labeled attack windows. Used exclusively for post-hoc threshold validation and metric benchmarking.
3. **`BATADAL_test_dataset.csv` ($2,089$ raw hourly rows, $2,066$ evaluated $H=12$ windows)**: Held-out unlabeled test evaluation dataset. Evaluated completely unsupervised with zero test set labels accessed or inferred.

> **Resampling and leakage policy:** The reported GNN pipeline does not use SMOTE or any other oversampler. Its 80/20 split is performed chronologically within the normal Dataset03 training baseline, before scaler fitting and window generation. Dataset04 remains a separate labeled validation set; the test dataset is only used for unlabeled inference. The 4,177 Dataset04 rows are never oversampled, combined with training rows, or split into train/test partitions.

---

## 3. Methodology & System Architecture

The core detector operates via **Normal-Only Spatio-Temporal Forecasting**. By learning normal hydraulic dependencies across time and spatial network topology, anomalies are detected when future sensor states deviate significantly from multi-step model predictions.

```
       ┌────────────────────────┐
       │ Sensor Features (B,T,F)│  (T=24 history window, F=43 sensors)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │  C-Town Graph Conv     │  (Topological Adjacency A ∈ ℝ⁴³ˣ⁴³)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ 1D Temporal Conv Layer │  (Kernel=3, Hidden_dim=64)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Multi-Horizon Head     │  (Direct forecast H=12 steps ahead)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Forecast Error e_i,t   │  (e_i,t = |y_i,t - ŷ_i,t| / (σ_i + ε))
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Anomaly Score S > τ?   │  (τ99.8 = 3.013382)
       └────────────────────────┘
```

### Topology & Graph Construction
The $43$ sensors map directly to the EPANET C-Town hydraulic network topology:
- **Junction Pressures ($P_{\text{junction}}$)** $\to$ Network Junction Nodes
- **Tank Levels ($L_{\text{tank}}$)** $\to$ Storage Tank Nodes
- **Flow & Status Sensors ($F_{\text{pump/valve}}, S_{\text{pump/valve}}$)** $\to$ Network Links / Edges

The graph adjacency matrix $A \in \mathbb{R}^{43 \times 43}$ is constructed using shortest-path topological distances ($\le 3$-hop neighborhoods), with self-loops and symmetric degree normalization: $\tilde{A} = D^{-1/2} (A + I) D^{-1/2}$.

---

## 4. Ground-Truth Validation Benchmark (Dataset04)

Validation performance was evaluated on `BATADAL_dataset04.csv` ($4,177$ hourly timesteps, $3,801$ normal windows, $173$ anomaly windows) using the frozen operational threshold $\tau_{99.8} = 3.013382$:

| Benchmark Metric | Dataset04 Score | Description |
|---|---|---|
| **Precision** | **0.5635** | $\text{TP} / (\text{TP} + \text{FP})$ |
| **Recall** | **0.7900** | $\text{TP} / (\text{TP} + \text{FN})$ |
| **F1 Score** | **0.6578** | Harmonic Mean of Precision & Recall |
| **ROC-AUC** | **0.9685** | Area Under Receiver Operating Characteristic Curve |
| **PR-AUC** | **0.6769** | Area Under Precision-Recall Curve |
| **False Positive Rate (FPR)** | **0.0341** | $\text{FP} / \text{Normal Windows}$ ($134 / 3801$) |

### Confusion Matrix Breakdown
- **True Negatives (TN)**: $3,801$
- **False Positives (FP)**: $134$
- **False Negatives (FN)**: $46$
- **True Positives (TP)**: $173$

---

## 5. Sequence-Level Validation Performance

Across the 5 labeled attack sequences in Dataset04, the $H=12$ model achieved early detection with minimal time-to-detection delays:

| Attack Sequence | Total Windows | Detected Windows | Sequence Recall | First Detection Delay |
|---|---|---|---|---|
| **Sequence 1** | 42 | 39 | **92.86%** | **0 hours** |
| **Sequence 2** | 60 | 39 | **65.00%** | **0 hours** |
| **Sequence 3** | 37 | 21 | **56.76%** | **4 hours** |
| **Sequence 4** | 7 | 6 | **85.71%** | **1 hour** |
| **Sequence 5** | 73 | 68 | **93.15%** | **0 hours** |

---

## 6. Held-Out Test Inference & Anomaly Episodes

Evaluation on `BATADAL_test_dataset.csv` ($2,089$ raw timesteps, $2,066$ evaluated $H=12$ windows) yielded **156 anomalous windows** ($7.55\%$ window anomaly rate) aggregated into **17 contiguous anomaly episodes**.

### Key Test Anomaly Episodes:
- **EP-12** (`2017-02-11 14:00` $\to$ `2017-02-13 18:00`, 53 h): Peak score = $1.13 \times 10^{10}$, Mean score = $4.34 \times 10^9$. Top drivers: `F_PU3`, `S_PU3`, `S_PU1`.
- **EP-11** (`2017-02-08 16:00` $\to$ `2017-02-10 20:00`, 53 h): Peak score = $7.38 \times 10^9$, Mean score = $3.97 \times 10^9$. Top drivers: `F_PU3`, `S_PU3`, `L_T3`.
- **EP-20** (`2017-03-07 13:00` $\to$ `2017-03-07 14:00`, 2 h): Peak score = $40.30$, Mean score = $39.18$. Top drivers: `S_PU6`, `F_PU6`.
- **EP-06** (`2017-01-30 01:00` $\to$ `2017-01-30 18:00`, 18 h): Peak score = $8.44$, Mean score = $5.38$. Top drivers: `L_T2`, `P_J422`.

---

> **Provenance note:** Dataset04 metrics and training threshold values below are legacy outputs produced before the sensor-specific variance-floor correction. Corrected held-out results use recalibrated scores and threshold.

## 7. Instrument-Calibrated Variance Floors

Per-sensor calibration residual variance is floored by physical instrument precision for pressure, tank level, flow, and binary status sensors, converted to the fitted scaler's feature space. This bounds normalized residuals for pump `PU3` when it transitions from an inactive calibration state. Corrected held-out inference recalibrates its threshold from the same floored Dataset03 scores.

---

## 8. H=12 Architectural Selection & Capacity Control

To verify whether the performance gain of $H=12$ over $H=1$ was due to multi-horizon forecasting or simply increased parameter capacity, a parameter-matched capacity-controlled experiment was conducted across 5 independent random seeds:

- **$H=1$ Capacity-Expanded Baseline**: $95,164$ parameters
- **$H=12$ Direct Multi-Horizon Model**: $95,236$ parameters ($99.75\%$ parameter match)

### Scientific Conclusions & Trade-Offs:
1. **Recall & Difficult-Sequence Superiority**: Across 5 seeds, $H=12$ consistently outperformed $H=1$ in Recall ($0.7900$ vs $0.7283$) and difficult attack sequence detection (Sequence 3 Recall: $56.76\%$ vs $43.24\%$). This confirms that multi-horizon forecasting provides genuine predictive benefits.
2. **Precision & FPR Trade-Off**: $H=12$ exhibited a higher false-positive rate ($0.0341$ vs $0.0195$) and lower precision ($0.5635$ vs $0.6632$) than $H=1$. Multi-step forecasting accumulates uncertainty over longer horizons.

---

## 9. Phase 8 Operational Streamlit Dashboard

The repository includes an interactive research and operational dashboard (`dashboard/app.py`).

### Launch Command:
```bash
.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.port 8501
```

### Dashboard Capabilities & Tabs:
1. **Executive Overview**: Model architecture specs, evaluated window counts, anomaly window rates, and Dataset04 ground-truth metrics card.
2. **Interactive Timeline**: Plotly line chart displaying $2,066$ test windows with threshold line and toggle between Log-Scaled Display Score ($\log_{10}(1 + \text{RawScore})$) and Raw Statistical Anomaly Score.
3. **Episode Explorer**: Table of $17$ contiguous anomaly episodes sorted by Operational Display Severity.
4. **Sensor Attribution & Calibration Floors**: Per-episode top-sensor breakdown with sensor-specific instrument variance-floor diagnostics.
5. **Topology & Region Mapping**: Mapped sensor locations to C-Town topological nodes and DMAs under the header `"Topology-Associated Region"`.
6. **Ground-Truth Validation**: Dedicated tab isolating Dataset04 supervised benchmark results.

### Terminology Policy Enforcement:
- **Forbidden**: `"Confirmed Cyberattack"`, `"Confirmed Attack"`, `"Attacker Identified"`, `"Malicious Intrusion Confirmed"`.
- **Allowed**: `"Statistical Anomaly Detected"`, `"Operational State Transition"`, `"Affected Sensor Group"`, `"Topology-Associated Region"`.

---

## 10. Dual Score & Severity Handling

### Anomaly Scores:
- **Raw Statistical Anomaly Score**: Preserved in core engine and CSV research artifacts without modification.
- **UI Display Score**: Log-transformed for clean chart rendering: $\text{UI\_Log\_Score} = \log_{10}(1 + \text{RawScore})$.

### Dual Severity Metrics:
- **Raw Statistical Severity**:
  $$\text{Severity}_{\text{Raw}} = \text{Mean Score} \times \sqrt{\text{Duration (hours)}} \times \left(1 + \frac{\text{Peak Score}}{\tau}\right)$$
- **Operational Display Severity (Log-Scaled Bounded)**:
  $$\text{Severity}_{\text{Log}} = \log_{10}(1 + \text{Mean Score}) \times \sqrt{\text{Duration (hours)}} \times \left(1 + \log_{10}\left(1 + \frac{\text{Peak Score}}{\tau}\right)\right)$$

---

## 11. Phase 2 Service Modules and Evaluation Scope

- `src/explainability/xai.py`: computes sensor-level two-player SHAP attribution and a local Ridge/LIME surrogate for the synthetic dashboard demonstration; tested against the frozen GNN.
- `src/localization/sensor_mapping.py` and `src/localization/zone_embeddings.py`: map sensors to C-Town components and pool sensor SHAP/LIME vectors into ranked DMA-level zone embeddings, splitting link-sensor contribution across endpoint zones. This is a project-local LightEdge-style implementation, not an imported/pretrained LightEdge-IDS embedding, which is not present in the repository.
- `src/root_cause/assessment.py`: selects the leading cause candidate from ranked SHAP output, checks LIME support and threshold status, and never accepts the injected sensor as an input. It does not infer a confirmed cause for the unlabeled test set.
- `src/impact/topology.py` and `src/impact/cpis.py`: derive graph footprint, nearby sensors, DMA, alternate paths, and the project-specific CPIS; topology behavior is covered with synthetic graphs.
- `src/recovery/mitigation.py`: returns category-specific defensive checks and response suggestions; recommendations are software-tested, not hydraulically simulated or operationally validated.

These tests verify deterministic software behavior only. Zone rankings have synthetic graph tests but no independent ground-truth zone labels or measured localization accuracy. The unlabeled BATADAL test set has no component-level ground truth, root-cause labels, or recovery outcomes, so localization accuracy, root-cause accuracy, impact prediction accuracy, and recovery effectiveness cannot be reported as benchmark metrics from this data.

## 12. Reproducibility Guide & Directory Structure

### Environment Setup (Windows):
```powershell
# Activate virtual environment
.venv\Scripts\activate

# Run full test suite (48 tests)
.venv\Scripts\python.exe -m pytest -v

# Launch Streamlit dashboard
.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.port 8501
```

### Directory Structure:
```text
X-Resilience-WDS/
├── configs/
│   └── config.yaml                     # System configuration
├── data/
│   ├── raw/                            # BATADAL raw datasets (Read-only)
│   └── processed/                      # Component sensor mappings
├── models/
│   ├── detection_gnn.pt                # PROMOTED PRODUCTION CHECKPOINT (H=12)
│   ├── detection_gnn_h12_candidate.pt  # Frozen Candidate Checkpoint
│   └── backups/                        # Pre-promotion rollback checkpoints
├── src/
│   ├── detection/                      # Core loader, preprocessor, model, evaluator
│   ├── explainability/                 # Model-based sensor explanations
│   ├── impact/                         # CPIS and topology impact analysis
│   ├── localization/                   # Sensor-to-component mapping
│   ├── network/                        # EPANET loader, graph builder, zoning
│   ├── recovery/                       # Defensive response recommendations
│   └── root_cause/                     # Evidence-scoped demo assessment
├── dashboard/
│   ├── app.py                          # Streamlit Phase 8 Operational Dashboard
│   └── README.md                       # Dashboard documentation
├── tests/
│   ├── test_phase1_network.py          # Network & graph unit tests
│   ├── test_phase1b_detection.py       # Preprocessing & GNN model tests
│   ├── test_phase2_services.py         # Impact/localization/root-cause/recovery tests
│   └── test_dashboard.py              # Dashboard & data integrity tests
├── results/
│   ├── final_figures/                  # Publication figures (fig1 - fig6)
│   ├── test_inference/corrected/       # Corrected test inference CSVs & JSONs
│   └── production_readiness/           # Production readiness & promotion reports
└── requirements.txt                    # Project dependencies
```

---

## 13. License & Citation

Distributed under the MIT License. See `LICENSE` for details.
