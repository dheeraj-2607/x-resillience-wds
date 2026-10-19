# X-Resilience-WDS: Final Project Results & Model Staging Summary

> [!NOTE]
> This summary contains legacy Dataset04 and held-out metrics from before sensor-specific variance floors. Current corrected held-out calibration and anomaly counts are in `results/test_inference/corrected/test_inference_report.json`.

This document presents the complete, integrated findings for the **topology-aware SpatioTemporalGNN architecture ($H=12$)** applied to the BATADAL C-Town Water Distribution System benchmark.

---

## Executive Summary & Key Milestones

- **Leading Architecture**: SpatioTemporalGNN with $H=12$ forecasting horizon ($95,236$ parameters).
- **Candidate Checkpoint Staged**: [`models/detection_gnn_h12_candidate.pt`](file:///c:/Users/newuser/Downloads/pro_X/X-Resilience-WDS/models/detection_gnn_h12_candidate.pt)
- **Production Checkpoint**: [`models/detection_gnn.pt`](file:///c:/Users/newuser/Downloads/pro_X/X-Resilience-WDS/models/detection_gnn.pt) remains **COMPLETELY UNTOUCHED** (SHA256: `056284e983ba...`).
- **Operational Anomaly Threshold ($	au_{99.8}$)**: **$3.013382$** (Derived strictly from Dataset03 normal calibration 99.8th percentile).
- **Dataset04 Validation Performance**: **F1 = $0.6578$**, **Recall = $0.7900$**, **Precision = $0.5635$**, **ROC-AUC = $0.9685$**, **PR-AUC = $0.6769$**, **FPR = $0.0341$**.
- **Capacity-Matched Comparison**: $H=12$ achieves a **$+22.47\%$ recall gain** and **$+0.0723$ F1 gain** over capacity-matched $H=1$ ($95,001$ parameters) across 5 random seeds.
- **Corrected Held-Out Test Set Observations**: Evaluated $2,066$ sliding windows ($2,089$ raw rows) on a strictly monotonic chronological timeline (`2017-01-04 12:00` to `2017-04-01 00:00`). Detected **$156$ anomalous timesteps** ($7.55\%$) forming **$17$ true contiguous anomaly episodes**.

---

## 1. Pipeline & Architecture Specifications

### A. Network Topology & Sensors
- **Water Distribution System**: C-Town Network ($43$ sensor channels: $12$ Junction Pressures `P_J*`, $7$ Tank Levels `L_T*`, $12$ Link Flows `F_*`, $12$ Actuator Statuses `S_*`).
- **Graph Structure**: 3-hop shortest-path spatial adjacency matrix derived directly from EPANET file `CTOWN.INP`.

### B. Preprocessing & Feature Normalization
- **Scaler**: `StandardScaler` fitted **strictly** on the 80% normal training split of `BATADAL_dataset03.csv` ($N=6,990$ windows). Zero validation or test data entered scaling parameters.
- **Sliding Window**: Window size $W=12$ (12 hours of past history), stride $=1$, forecasting horizon $H=12$ (predicting 12 future time-steps simultaneously).

### C. SpatioTemporalGNN Architecture
- **Input Dimensions**: $(B, W=12, F=43)$
- **Layers**: 1D Temporal Convolution $
ightarrow$ Topology Graph Convolution (Spatial Message Passing over C-Town Graph) $
ightarrow$ Dropout ($0.2$) $
ightarrow$ Multi-step Forecasting Linear Output Header $(B, H=12, F=43)$.
- **Total Trainable Parameters**: $95,236$

### D. Operational Threshold Calibration Rule
- Threshold $	au_{99.8} = 3.013382$ was derived **exclusively** from the 99.8th percentile of normalized forecasting error scores on the held-out 20% normal calibration split of `BATADAL_dataset03.csv` ($N=1,747$ windows). Zero attack labels from Dataset04 or test set were used to tune the threshold.

---

## 2. PART A � VALIDATED RESULTS (Supervised Performance)

> [!NOTE]
> All metrics in Part A are evaluated on ground-truth labeled validation data (`BATADAL_dataset04.csv`, $N=4,154$ windows, $219$ attack windows across 5 attack sequences).

### A. Dataset04 Final Candidate Validation Performance

| Metric | Candidate $H=12$ Value | Standard Baseline |
| :--- | :---: | :---: |
| **Precision** | **$0.5635$** | $0.5402$ |
| **Recall** | **$0.7900$** | $0.6438$ |
| **F1 Score** | **$0.6578$** | $0.5875$ |
| **ROC-AUC** | **$0.9685$** | $0.9482$ |
| **PR-AUC** | **$0.6769$** | $0.5932$ |
| **False Positive Rate (FPR)** | **$0.0341$ ($3.41\%$)** | $0.0244$ |
| **Confusion Matrix** | $\mathbf{\text{TN}=3801, \text{FP}=134, \text{FN}=46, \text{TP}=173}$ | � |

### B. Per-Sequence Recalls & Detection Delays (Dataset04)

| Sequence ID | Duration (steps) | Candidate Recall (%) | Detected / Total Windows | Detection Delay (steps) | Detection Delay (hours) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sequence 1** | $42$ | **$92.86\%$** | $39 / 42$ | $0$ steps | **$0.0$ h** |
| **Sequence 2** | $60$ | **$65.00\%$** | $39 / 60$ | $0$ steps | **$0.0$ h** |
| **Sequence 3** | $37$ | **$56.76\%$** | $21 / 37$ | $4$ steps | **$4.0$ h** |
| **Sequence 4** | $7$ | **$85.71\%$** | $6 / 7$ | $1$ step | **$1.0$ h** |
| **Sequence 5** | $73$ | **$93.15\%$** | $68 / 73$ | $0$ steps | **$0.0$ h** |

### C. Capacity-Controlled Controlled Experiment ($H=1$ vs $H=12$ across 5 Random Seeds)

To rule out parameter count as a confounding variable, $H=1$ was scaled up to $\text{hidden\_dim}=79$ ($95,001$ parameters) and compared against $H=12$ ($	ext{hidden\_dim}=64$, $95,236$ parameters):

| Metric | H=1 Capacity-Matched (95k params) | H=12 Candidate (95k params) | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Mean F1 Score** | $0.5600 \pm 0.0480$ | **$0.6323 \pm 0.0295$** | **$+0.0723$** |
| **Mean Recall** | $0.4932 \pm 0.0784$ | **$0.7178 \pm 0.0521$** | **$+0.2247$ ($+22.47\%$)** |
| **Mean ROC-AUC** | $0.8824 \pm 0.0043$ | **$0.9567 \pm 0.0074$** | **$+0.0743$** |
| **Mean PR-AUC** | $0.5397 \pm 0.0086$ | **$0.6292 \pm 0.0310$** | **$+0.0895$** |
| **Sequence 3 Recall** | $9.73\%$ | **$36.76\%$** | **$+27.03\%$** |

---

## 3. PART B � UNSUPERVISED TEST OBSERVATIONS (`BATADAL_test_dataset.csv`)

> [!IMPORTANT]
> Ground-truth labels for `BATADAL_test_dataset.csv` are unavailable. Therefore, supervised performance metrics (F1, Precision, Recall, ROC-AUC) CANNOT be calculated for the test set. Findings in Part B represent model-detected abnormal behavior.

### A. Test Set Summary & Corrected Timeline
- **Raw Test Rows**: $2,089$
- **Evaluated $H=12$ Windows**: $2,066$ timesteps (`2017-01-04 12:00` to `2017-04-01 00:00`, strictly monotonic)
- **Anomalous Windows Detected**: **$156$ timesteps** ($7.55\%$)
- **True Contiguous Anomaly Episodes**: **$17$ episodes** (corrected from 23 fragmented pseudo-episodes caused by default date parsing)

### B. Corrected Anomaly Episode Breakdown (Ranked by Severity)

$$\text{Severity Score} = \text{Mean Anomaly Score} \times \sqrt{\text{Duration in Hours}} \times \left(1 + \frac{\text{Peak Anomaly Score}}{\tau_{99.8}}\right)$$

| Rank | Episode ID | Start Timestamp | End Timestamp | Duration (h) | Max Score | Mean Score | Severity Score | Episode Category |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | `EP-12` | `2017-02-11 14:00` | `2017-02-13 18:00` | $53.0$ h | $1.13 \times 10^{10}$ | $4.34 \times 10^9$ | **$1.19 \times 10^{20}$** | Persistent Episode (>6h) [High-Severity Outlier] |
| **2** | `EP-11` | `2017-02-08 16:00` | `2017-02-10 20:00` | $53.0$ h | $7.38 \times 10^9$ | $3.97 \times 10^9$ | **$7.08 \times 10^{19}$** | Persistent Episode (>6h) [High-Severity Outlier] |
| **3** | `EP-20` | `2017-03-07 13:00` | `2017-03-07 14:00` | $2.0$ h | $40.303$ | $39.175$ | **$796.39$** | Short Episode (2-6h) |
| **4** | `EP-06` | `2017-01-30 01:00` | `2017-01-30 18:00` | $18.0$ h | $8.436$ | $5.375$ | **$86.64$** | Persistent Episode (>6h) |
| **5** | `EP-16` | `2017-02-25 16:00` | `2017-02-25 18:00` | $3.0$ h | $6.791$ | $5.442$ | **$30.67$** | Short Episode (2-6h) |
| **6** | `EP-18` | `2017-02-26 09:00` | `2017-02-26 10:00` | $2.0$ h | $5.347$ | $5.242$ | **$20.57$** | Short Episode (2-6h) |
| **7** | `EP-21` | `2017-03-08 02:00` | `2017-03-08 07:00` | $6.0$ h | $3.957$ | $3.612$ | **$20.47$** | Short Episode (2-6h) |
| **8** | `EP-07` | `2017-01-30 20:00` | `2017-01-30 21:00` | $2.0$ h | $4.870$ | $4.801$ | **$17.76$** | Short Episode (2-6h) |
| **9** | `EP-08` | `2017-01-31 05:00` | `2017-01-31 07:00` | $3.0$ h | $3.572$ | $3.297$ | **$12.48$** | Short Episode (2-6h) |
| **10** | `EP-05` | `2017-01-29 22:00` | `2017-01-29 22:00` | $1.0$ h | $3.777$ | $3.777$ | **$8.51$** | Isolated Anomaly (1h) |
| **11** | `EP-22` | `2017-03-24 12:00` | `2017-03-24 12:00` | $1.0$ h | $3.774$ | $3.774$ | **$8.50$** | Isolated Anomaly (1h) |
| **12** | `EP-17` | `2017-02-25 21:00` | `2017-02-25 21:00` | $1.0$ h | $3.765$ | $3.765$ | **$8.47$** | Isolated Anomaly (1h) |
| **13** | `EP-19` | `2017-03-04 13:00` | `2017-03-04 13:00` | $1.0$ h | $3.649$ | $3.649$ | **$8.07$** | Isolated Anomaly (1h) |
| **14** | `EP-02` | `2017-01-09 14:00` | `2017-01-09 14:00` | $1.0$ h | $3.548$ | $3.548$ | **$7.73$** | Isolated Anomaly (1h) |
| **15** | `EP-15` | `2017-02-24 14:00` | `2017-02-24 14:00` | $1.0$ h | $3.502$ | $3.502$ | **$7.57$** | Isolated Anomaly (1h) |
| **16** | `EP-09` | `2017-01-31 11:00` | `2017-01-31 11:00` | $1.0$ h | $3.468$ | $3.468$ | **$7.46$** | Isolated Anomaly (1h) |
| **17** | `EP-10` | `2017-02-01 07:00` | `2017-02-01 07:00` | $1.0$ h | $3.422$ | $3.422$ | **$7.31$** | Isolated Anomaly (1h) |

---

## 4. Key Limitations & Numerical Analysis

> [!NOTE]
> The numerical values in this legacy summary predate the sensor-specific physical variance floors. Recalibrated held-out inference is recorded in `results/test_inference/corrected/test_inference_report.json`.

1. **Pump `PU3` Baseline Inactivity & Score Amplification**:
   - In normal baseline training (`BATADAL_dataset03.csv`), pump `PU3` is **OFF continuously** (`F_PU3 = 0.0`), producing near-zero normal calibration error variance $\mu_{	ext{sq, h}}[11] pprox 2.9 	imes 10^{-8}$.
   - When pump `PU3` turns **ON** in the test dataset (`F_PU3 = 86.49` L/s), dividing squared forecasting error ($14,294.26$) by $2.9 	imes 10^{-8}$ yields normalized residual $Z = 4.87 	imes 10^{11}$.
   - **Limitation**: Normalizing by near-zero normal baseline variance amplifies operational state transitions into multi-billion residual scores.

2. **Severity Formula Outlier Dominance**:
   - Because the severity formula scales quadratically with anomaly score ($	ext{Mean} 	imes 	ext{Peak}$), extreme single-sensor residual spikes in `EP-11` and `EP-12` ($\sim 10^{10}$) produce severity scores of $\sim 10^{19} - 10^{20}$, numerically dominating all other episodes.

3. **Absence of Ground-Truth Test Labels**:
   - Test set observations cannot be labeled as "confirmed cyberattacks" or "physical failures". They represent model-detected statistical anomalies.

---

## 5. System Integrity & Reproducibility Verification

- Candidate Checkpoint [`models/detection_gnn_h12_candidate.pt`](file:///c:/Users/newuser/Downloads/pro_X/X-Resilience-WDS/models/detection_gnn_h12_candidate.pt): **UNCHANGED** (SHA256: `8e85de2b871f...`)
- Production Checkpoint [`models/detection_gnn.pt`](file:///c:/Users/newuser/Downloads/pro_X/X-Resilience-WDS/models/detection_gnn.pt): **UNTOUCHED** (SHA256: `056284e983ba...`)
- Configuration [`configs/config.yaml`](file:///c:/Users/newuser/Downloads/pro_X/X-Resilience-WDS/configs/config.yaml): **UNCHANGED**
- Raw Data Files `data/raw/*`: **UNCHANGED & READ-ONLY**
- Unit Test Suite (`.venv\Scripts\python.exe -m pytest -v`): **$41 / 41$ tests PASSED**

---

## 6. Generated Publication Figures

All 6 figures are saved in [`results/final_figures/`](file:///c:/Users/newuser/Downloads/pro_X/X-Resilience-WDS/results/final_figures/):
1. **Dataset04 Validation Performance**: `fig1_dataset04_validation.png`
2. **H=1 vs H=12 Comparison**: `fig2_h1_vs_h12_horizon.png`
3. **Capacity-Controlled Comparison**: `fig3_capacity_controlled_comparison.png`
4. **Corrected Test Anomaly Timeline**: `fig4_test_anomaly_timeline.png`
5. **Top Sensor Attribution**: `fig5_sensor_attribution.png`
6. **High-Severity Episode Breakdown**: `fig6_high_severity_episode.png`

---

**Candidate model has NOT been promoted to production.**
**Awaiting your explicit approval before modifying `models/detection_gnn.pt`.**
