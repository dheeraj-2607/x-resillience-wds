# X-Resilience-WDS Operational Anomaly Dashboard

The `dashboard/app.py` Streamlit application provides an interactive research and operational interface for inspecting spatio-temporal GNN anomaly detection results on the C-Town Water Distribution System (BATADAL benchmark).

---

## Quick Start

Launch the Streamlit dashboard using the virtual environment Python executable:

```bash
.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Or directly via streamlit if PATH is configured:

```bash
streamlit run dashboard/app.py
```

---

## Data Sources

The dashboard loads pre-computed, verified test inference artifacts from the repository:

- `results/test_inference/corrected/test_anomaly_scores.csv`: Per-window anomaly scores, flags, and top sensor drivers across $2,066$ evaluated test windows.
- `results/test_inference/corrected/test_anomaly_episodes.csv`: $17$ contiguous anomaly episodes with timestamps, durations, peak/mean scores, and severity metrics.
- `results/test_inference/corrected/test_sensor_attribution.csv`: Per-timestep residual scores and sensor-to-topology mappings.
- `results/final_results_summary.json`: Ground-truth Dataset04 validation metrics.
- `data/processed/component_sensor_map.json`: EPANET node/link component mapping for C-Town topology.

In the synthetic demonstration, zone localization ranks C-Town DMAs using sensor SHAP/LIME attribution pooled into topology-aware zone vectors. Link sensors are fractionally assigned to the DMAs at their endpoints. These are project-generated attribution embeddings, not pretrained LightEdge-IDS embeddings or calibrated probabilities; the repository contains no LightEdge-IDS embedding model or zone-level attack labels.

---

## Ground-Truth Validation vs. Unlabeled Test Data Distinction

To maintain complete scientific integrity, the dashboard strictly distinguishes between:

1. **Dataset04 Ground-Truth Validation**: Labeled benchmark performance metrics (Precision = $0.5635$, Recall = $0.7900$, F1 = $0.6578$, ROC-AUC = $0.9685$, PR-AUC = $0.6769$). Supervised metrics were computed **only** on Dataset04.
2. **Unlabeled Test Set Evaluation**: Inferred on `BATADAL_test_dataset.csv` ($2,089$ raw timesteps, $2,066$ evaluated $H=12$ forecasting windows, $156$ anomalous windows across $17$ contiguous episodes). Zero test set labels were accessed or inferred.

---

## Dual Score & Severity Definitions

### 1. Anomaly Scores
- **Raw Statistical Anomaly Score**: Sum of normalized forecasting residuals across $H=12$ steps and $43$ sensors.
- **Log-Scaled Display Score**: $\text{UI\_Log\_Score} = \log_{10}(1 + \text{Raw Anomaly Score})$. Recommended for UI display due to pump PU3 score amplification.

### 2. Operational Severity Metrics
- **Raw Statistical Severity**:
  $$\text{Severity}_{\text{Raw}} = \text{Mean Score} \times \sqrt{\text{Duration (hours)}} \times \left(1 + \frac{\text{Peak Score}}{\tau}\right)$$
- **Operational Display Severity (Log-Scaled Bounded)**:
  $$\text{Severity}_{\text{Log}} = \log_{10}(1 + \text{Mean Score}) \times \sqrt{\text{Duration (hours)}} \times \left(1 + \log_{10}\left(1 + \frac{\text{Peak Score}}{\tau}\right)\right)$$

The dashboard uses **Operational Display Severity** for ranking episodes by default to prevent single pump state transitions from dominating UI rankings.

---

## Instrument-Calibrated Variance Floors

Calibration residual variances are bounded below by sensor-class instrument precision, transformed into the fitted scaler's feature space.

The flow floor prevents inactive pump `PU3` from producing unbounded normalized residuals when its operational state changes. Held-out inference recalibrates its threshold from the same floored Dataset03 scores.

---

## Scientific Terminology Policy

The dashboard strictly enforces non-alarmist terminology:

- **FORBIDDEN TERMS**: "Confirmed Cyberattack", "Confirmed Attack", "Attacker Identified", "Malicious Intrusion Confirmed", "Attack Mechanism Confirmed".
- **REQUIRED OPERATIONAL TERMS**: "Statistical Anomaly Detected", "Operational State Transition", "Affected Sensor Group", "Topology-Associated Region", "Persistent Anomaly", "Forecasting Deviation".
