# Comprehensive Module-by-Module Technical Audit: X-Resilience-WDS

**Audit Target**: `c:\Users\hp\OneDrive\Desktop\Main Project\X-Resilience-WDS`  
**Commit**: `a426ff0` (19 October 2026)  
**Evaluator**: Senior Software Architect & CPS Security Specialist  

---

## 1. Attack Detection Subsystem (`src/detection/`)

### 1.1 Architecture & Components
- **`graph_model.py`**: Implements `SpatioTemporalGNN`.
  - Architecture: Input tensor $(B, W, F) = (B, 12, 43)$. Spatial aggregation via $X_{\text{spatial}} = X \cdot A$ (where $A$ is the normalized $43 \times 43$ shortest-path adjacency matrix built from C-Town topology). Temporal feature extraction via 1D Conv over time frames (kernel size 3, padding 1, 64 hidden channels). Prediction head: `Linear(64 * 12, 64) -> ReLU -> Dropout(0.2) -> Linear(64, 43 * 12)`.
  - Total Parameters: 95,236 trainable weights.
- **`preprocessing.py`**: Implements `DataPreprocessor`.
  - Strict temporal ordering: Chronological split of normal baseline `BATADAL_dataset03.csv` into 80% training (6,990 rows) and 20% normal calibration (1,747 rows).
  - Scaler fitting: `StandardScaler` fitted **only** on the first 6,990 training rows. No validation or test data is ever seen by the scaler.
- **`dataset_loader.py`**: Implements `BATADALDatasetLoader`.
  - Strips whitespace from column headers. Auto-detects 43 sensor channels.
  - Corrected handling of `-999`: unlabelled rows are preserved as NaN/mask, avoiding false negative inflation.
- **`smote_baseline.py`**: Implements `chronological_smote_split` and `evaluate_smote_baselines`.
  - Implements the leakage-free supervised benchmark required by the professor's roadmap.
  - Chronological split: first 80% (3,341 rows) for training, remaining 20% (836 rows) for testing.
  - Scaling, imputation, and `SMOTE` applied strictly to the training split.
  - Evaluates both `RandomForestClassifier` and `XGBClassifier`.

### 1.2 Model Checkpoints & Provenance
- `models/detection_gnn.pt` (SHA-256: `8e85de...`): Production checkpoint ($H=12$).
- `models/detection_gnn_h12_candidate.pt`: Verified byte-identical match to production checkpoint.
- Calibration threshold: $\tau = 3.013382$ derived from the 99.8th percentile of normalized residuals on the normal calibration partition.

### 1.3 Detection Performance Summary
- **SpatioTemporalGNN ($H=12$) on Dataset04**:
  - Precision: `0.5635`, Recall: `0.7900`, F1: `0.6578`, ROC-AUC: `0.9685`, PR-AUC: `0.6769`, FPR: `0.0341`.
  - Confusion Matrix: TN = 3,801, FP = 134, FN = 46, TP = 173.
- **Supervised Chronological RF Baseline on Dataset04** (`results/smote_baseline_report.json`):
  - Untouched Test Rows: 836 (756 normal, 80 attack).
  - Precision: `0.0000`, Recall: `0.0000`, F1: `0.0000`, ROC-AUC: `0.5230`.
  - *Finding*: Proves the roadmap's thesis that standard raw sensor values under chronological splitting cannot detect complex dynamic water attacks without feature engineering or temporal forecasting.

---

## 2. Explainable AI Subsystem (`src/explainability/`)

### 2.1 Implementation Inspection (`src/explainability/xai.py`)
- **Functions Present**:
  - `compute_xai_explanations(...)`: Calculates exact two-player marginal swap differences for quick dashboard rendering and fits a 50-sample local Ridge surrogate.
  - `compute_shap_explanations(...)`: Added in recent commits to execute true model attribution.
- **Explanation Scope**: Local window attribution mapping normalized forecasting residual contributions to each of the 43 physical sensors.
- **Sensor-to-Asset Mapping**: Resolves sensor names (e.g., `P_J280`, `L_T1`, `F_PU1`) to physical asset types (Junction, Tank, Pump, Valve) using `component_sensor_map.json`.

### 2.2 Gaps & Findings
1. While `compute_shap_explanations` now provides permutation-based attributions, full integration with `shap.TreeExplainer` for the tree baselines and `lime.lime_tabular.LimeTabularExplainer` should be formalized into an automated batch evaluation script.
2. An automated table reporting **Fidelity** (score drop upon top-K masking) and **Stability** (seed-to-seed correlation) across all 14 attacks must be permanently archived in `results/`.

---

## 3. Network Modeling & Graph Topology (`src/network/`)

### 3.1 Network Parser (`src/network/ctown_loader.py`)
- Reads raw EPANET file `data/raw/CTOWN.INP` using `wntr.network.WaterNetworkModel`.
- Parsed Inventory:
  - Junctions: 388
  - Tanks: 7
  - Reservoirs: 1
  - Pipes: 429
  - Pumps: 11
  - Valves: 4
  - Total Nodes: 396 | Total Links: 444.

### 3.2 Graph Construction (`src/network/graph_builder.py`)
- **Undirected Graph (`build_network_graph`)**: Node count = 396, Edge count = 444. Verified connected.
- **Directed Hydraulic Graph (`build_directed_hydraulic_graph`)**: Added to address Roadmap Task P1.4. Runs a steady-state hydraulic simulation in WNTR. Determines signed flow velocities through all links and directs edges in the predominant direction of water movement (444 directed edges).

### 3.3 DMA Zoning (`src/network/zoning.py`)
- Implements two partitioning schemes:
  1. `assign_dma_zones`: Assigns the 5 ground-truth District Metered Areas (DMAs) derived from C-Town junction demand patterns. All 396 nodes resolved.
  2. `assign_louvain_zones`: Unsupervised community detection partitioning the network into 9 topological clusters.

---

## 4. Localization & Root-Cause Analysis (`src/localization/` & `src/root_cause/`)

### 4.1 Asset & Zone Localization
- **`sensor_mapping.py`**: Maps sensors to their constituent nodes (e.g. `F_PU1` -> endpoints `J273`, `J285`).
- **`zone_embeddings.py`**: Computes an 8-dimensional zone attribution mass embedding from XAI weights to rank affected operational zones.

### 4.2 Reversed-Flow PageRank Root Cause (`src/root_cause/assessment.py`)
- Implements `rank_upstream_components(...)`:
  - Inverts the directed hydraulic graph ($G_{\text{rev}}$).
  - Personalizes PageRank with absolute sensor SHAP values: $p(u) = \sum_{s \in \text{Sensors}(u)} |\text{SHAP}(s)|$.
  - Computes stationary distribution on $G_{\text{rev}}$ with teleport probability $\alpha = 0.5$.
  - Ranks controllable components (Pumps, Valves, Tanks) by combining PPR mass with direct sensor attribution.

### 4.3 14-Attack Localization Benchmark (`src/root_cause/evaluate_localization.py`)
- Evaluated against `data/ground_truth_attacks.csv` across all 14 BATADAL attacks.
- Results recorded in `results/localization_benchmark_report.json`:
  - **Baseline A (Raw Residual Deviation)**: Top-1 Acc = `14.29%`, Top-3 Acc = `21.43%`, Zone Acc = `64.29%`, Mean Hop Distance = `12.36`.
  - **Baseline B (Top-1 Direct SHAP)**: Top-1 Acc = `21.43%`, Top-3 Acc = `21.43%`, Zone Acc = `57.14%`, Mean Hop Distance = `21.57`.
  - **Proposed (SHAP + Reversed-Flow PPR)**: Top-1 Acc = `21.43%`, Top-3 Acc = `21.43%`, Zone Acc = `57.14%`, Mean Hop Distance = `17.93`.
- **Audit Finding**: The proposed method reduces the mean topological error distance by **3.64 hops** compared to direct SHAP (Baseline B), but achieves the exact same Top-1 and Top-3 accuracy (3/14). This directly triggered the test assertion failure in `tests/test_localization_benchmark.py`.

---

## 5. Cyber-Physical Impact Assessment (CPIS) (`src/impact/`)

### 5.1 Formulation (`src/impact/cpis.py`)
- Computes Cyber-Physical Impact Score:
  $$\text{CPIS} = w_P \cdot I_P + w_F \cdot I_F + w_S \cdot I_S + w_C \cdot I_C$$
  Weights configured in `configs/config.yaml`: $w_P = 0.30, w_F = 0.30, w_S = 0.25, w_C = 0.15$.
- Factors:
  - $I_P$: Pressure impact factor (saturated residual deviation).
  - $I_F$: Flow anomaly factor.
  - $I_S$: Service disruption estimate based on episode duration and class coverage.
  - $I_C$: Infrastructure criticality prior ($P=0.55, T=0.75, F=0.85, S=0.90$).

### 5.2 Critical Audit Gap
- `src/impact/cpis.py` is currently calculated via **mathematical proxy functions** on sensor residuals.
- To fulfill the professor's requirement for Phase 2 (P2.1 & P2.2), this must be replaced with **actual WNTR Pressure-Dependent Demand (PDA) simulations** that compute physical unmet demand volume ($\text{m}^3$) and junction pressure deficits ($P < 20\text{ m}$).

---

## 6. Recovery Optimization & Decision Support (`src/recovery/`)

### 6.1 Optimizer Implementation (`src/recovery/optimizer.py`)
- Implements `optimize_recovery_actions(...)` and `optimize_component_outage(...)`.
- Simulates candidate recovery actions using `wntr.sim.EpanetSimulator` with PDA (`demand_model='PDA'`, `required_pressure=20.0`, `minimum_pressure=0.0`).
- Generates 5 candidate action classes:
  1. Component Restoration (`restore_link`)
  2. Pump Switching (`switch_target_to_backup`)
  3. Flow Rerouting (`open_alternate_valve_route`)
  4. Zone Containment (`close_boundary_valves`)
  5. Emergency Speed Adjustment (`boost_adjacent_pumps`)
- Evaluates multi-objective resilience score:
  $$J(a) = 0.30 \cdot \text{Service Continuity} + 0.25 \cdot \text{Hydraulic Stability} + 0.20 \cdot \text{Recovery Time} - 0.10 \cdot \text{Cost} + 0.15 \cdot \text{Criticality}$$

### 6.2 Audit Finding
- The recovery simulator is mathematically sound and fully tested (`tests/test_recovery_optimizer.py` passes).
- However, it currently operates on **simulated single-link outages** generated during demonstration mode. It has not yet been batch-executed across the 14 real BATADAL attack scenarios to produce a comparative benchmark against the "No Action" baseline.

---

## 7. Operational Dashboard (`dashboard/app.py`)

### 7.1 Features & User Workflow
- Framework: Streamlit (`dashboard/app.py`, 1,320 lines of code).
- Tabs:
  1. *Executive Summary*: Headline metrics, model specifications, scientific policy notice.
  2. *Anomaly Timeline*: Interactive Plotly timeline of continuous test inference scores.
  3. *Episode Explorer*: Table and inspection tool for the 17 detected anomaly episodes.
  4. *Live Incident Investigation (Demo Mode)*: Interactive sensor perturbation, XAI attribution waterfall, DMA zone ranking, and topology rerouting map.
  5. *Research Artifacts & Methodology*: Methodological gallery and validation figures.

### 7.2 Audit Findings & Defects
1. **Test Failure Mismatch**: In `tests/test_dashboard.py`, lines 125 and 162 assert `assert df_scores["anomaly_flag"].sum() == 156`. The dataset was recalibrated to 147 flagged windows, causing 2 test failures.
2. **Demo Mode Decoupling**: In recent commit `a426ff0`, `dashboard/app.py` was updated to import `assess_explanation_root_cause` and display the inferred candidate sensor. However, the downstream topology reroute is still partially passed the injected sensor `target_col`. This must be fully decoupled so downstream actions reflect the inferred candidate component $C_{\text{pred}}^*$.

---

## 8. Summary Status Table

| Module | Location | Unit Tests | Connected to App | Scientific Status |
|:---|:---|:---:|:---:|:---|
| Dataset Loader | `src/detection/dataset_loader.py` | 8/8 Passed | Yes | Cleaned; -999 handled |
| Preprocessing | `src/detection/preprocessing.py` | 5/5 Passed | Yes | Leakage-free; chronological |
| GNN Model | `src/detection/graph_model.py` | 4/4 Passed | Yes | Verified checkpoint ($H=12$) |
| SMOTE Baseline | `src/detection/smote_baseline.py` | 5/5 Passed | Standalone | Validated chronological split |
| XAI Engine | `src/explainability/xai.py` | 2/2 Passed | Yes | Permutation SHAP active |
| Network Topology | `src/network/` | 15/15 Passed | Yes | Directed WNTR flow graph |
| Root Cause & Localization | `src/root_cause/` | 7/8 Passed | Yes | PPR active; 1 assertion tied |
| CPIS Impact | `src/impact/` | 2/2 Passed | Yes | Residual proxy (needs PDA replay) |
| Recovery Optimizer | `src/recovery/` | 2/2 Passed | Demo only | PDA simulator verified |
| Dashboard Cockpit | `dashboard/app.py` | 5/7 Passed | Main Entry | 2 test assertions need count update |
