# Automated Test Suite & Execution Audit Report

**Date of Execution**: 19 October 2026  
**Environment**: Windows 11, Python 3.13.0, Pytest 9.1.1  
**Working Directory**: `c:\Users\hp\OneDrive\Desktop\Main Project\X-Resilience-WDS`  
**Execution Command**: `.\.venv\Scripts\pytest.exe tests/ -v`  

---

## 1. Overall Test Results Summary

- **Total Tests Collected**: 83
- **Passed**: 80 (96.4%)
- **Failed**: 3 (3.6%)
- **Warnings**: 8 (WNTR EPANET unused curve warnings)
- **Execution Duration**: 18.10 seconds
- **Exit Code**: 1 (Non-zero due to 3 failed assertions)

---

## 2. Detailed Breakdown of Test Failures

### Failure 1: `test_corrected_test_scores_monotonicity_and_counts`
- **File & Line**: `tests/test_dashboard.py:125`
- **Test Class**: `TestDashboardDataIntegrity`
- **Failure Traceback**:
  ```python
  def test_corrected_test_scores_monotonicity_and_counts(self):
      df_scores = pd.read_csv(CORRECTED_DIR / "test_anomaly_scores.csv")
      df_scores["timestamp"] = pd.to_datetime(df_scores["timestamp"])
      ...
  >   assert df_scores["anomaly_flag"].sum() == 156
  E   assert np.int64(147) == 156
  ```
- **Root Cause Analysis**: In earlier iterations before implementing instrument-based variance floors, pump PU3 noise spikes caused 156 windows to exceed threshold $\tau$. When `calibrated_sensor_variance` introduced the physical noise floor ($\sigma^2_{\min} = 0.05^2$), false alarms dropped, producing 147 legitimate anomaly windows. The test retained the obsolete hard-coded value `156`.
- **Recommended Remediation**: Update `assert df_scores["anomaly_flag"].sum() == 147` in `tests/test_dashboard.py`.

---

### Failure 2: `test_dashboard_loads_exactly_17_episodes`
- **File & Line**: `tests/test_dashboard.py:162`
- **Test Class**: `TestDashboardDataIntegrity`
- **Failure Traceback**:
  ```python
  def test_dashboard_loads_exactly_17_episodes(self):
      from dashboard.app import load_dashboard_data
      df_scores, df_episodes, df_attribution, summary_data, component_map = load_dashboard_data()
      assert len(df_episodes) == 17
      assert len(df_scores) == 2066
  >   assert df_scores["anomaly_flag"].sum() == 156
  E   assert np.int64(147) == 156
  ```
- **Root Cause Analysis**: Identical to Failure 1. `load_dashboard_data()` successfully loads the 17 authoritative episodes and 2,066 windows, but the internal assertion checks for 156 flagged windows rather than the recalibrated count of 147.
- **Recommended Remediation**: Update assertion to `assert df_scores["anomaly_flag"].sum() == 147`.

---

### Failure 3: `test_proposed_localization_beats_both_baselines_on_top1_and_top3`
- **File & Line**: `tests/test_localization_benchmark.py:24`
- **Test Function**: `test_proposed_localization_beats_both_baselines_on_top1_and_top3`
- **Failure Traceback**:
  ```python
  for baseline_name in BASELINES:
      baseline = metrics[baseline_name]
      assert baseline["attacks"] == 14
      assert proposed["top1_component_accuracy_pct"] > baseline["top1_component_accuracy_pct"]
  >   assert proposed["top3_component_accuracy_pct"] > baseline["top3_component_accuracy_pct"]
  E   assert 21.428571428571427 > 21.428571428571427
  ```
- **Root Cause Analysis**: In `results/localization_benchmark_report.json`:
  - Baseline B (Direct Top-1 SHAP): Top-1 = 21.43%, Top-3 = 21.43%
  - Proposed Method (SHAP + Reversed PPR): Top-1 = 21.43%, Top-3 = 21.43%
  The proposed method tied Baseline B on exact component identification while reducing topological hop distance from 21.57 down to 17.93 hops. The test strictly asserted `>` (strictly greater than), which evaluates to False when tied.
- **Recommended Remediation**:
  1. Refine the fusion weights in `src/root_cause/assessment.py` between direct SHAP attribution and PageRank mass to improve top-3 coverage, OR
  2. Modify the test assertion to check `proposed["top3_component_accuracy_pct"] >= baseline["top3_component_accuracy_pct"]` while asserting `proposed["mean_topological_hop_distance"] < baseline["mean_topological_hop_distance"]`.

---

## 3. Inventory of Passing Tests (80/83)

| Test Module | Tests Passed | Subsystem Covered |
|:---|:---:|:---|
| `tests/test_phase1_network.py` | 15 / 15 | `CTOWN.INP` parsing, node/edge counts, graph connectivity, 5 DMA zones, Louvain zoning |
| `tests/test_phase1b_detection.py` | 17 / 17 | Data loader, sliding window creation, scaler fitting on training only, GNN forward pass, threshold calibration |
| `tests/test_ground_truth.py` | 5 / 5 | Ground truth schema, 14 attacks verification, datetime monotonicity, C-Town component resolution |
| `tests/test_phase2_services.py` | 9 / 9 | Sensor prefix mapping, zone embeddings mass, topology impact footprint, alternate path discovery, reversed-flow PPR |
| `tests/test_recovery_optimizer.py` | 1 / 1 | WNTR PDA simulation of candidate recovery actions, input model immutability, resilience scoring |
| `tests/test_reporting_integrity.py` | 1 / 1 | Verification that no fabricated Table 5 RF claims exist in documentation |
| `tests/test_smote_baseline.py` | 5 / 5 | Chronological SMOTE split, zero test leakage, trailing temporal feature extraction, tree SHAP branches |
| `tests/test_localization_benchmark.py` | 1 / 2 | Verified attack type breakdowns and propagation paths present across all 14 attacks |
| `tests/test_dashboard.py` | 26 / 28 | Terminology policy compliance, episode formatting, demo mode pipeline, threshold locking, topology rendering |

---

## 4. Verification Checkpoint Status

| Check | Target | Observed Outcome | Status |
|:---|:---|:---|:---:|
| Clean Syntax Compilation | Zero syntax errors | All files compile cleanly under Python 3.13 | **PASS** |
| Dataset Integrity Hash Check | All 4 raw files match SHA-256 | `verify_datasets.py` outputs 4 MATCH records | **PASS** |
| Data Leakage Invariant | Scaler/SMOTE fitted on train only | Verified in `test_scaler_fitted_only_on_training_data` and `test_chronological_split...` | **PASS** |
| Zero Table 5 Fraud Claims | Excluded from docs | `test_reporting_integrity.py` passes | **PASS** |
| Full Test Suite Execution | 83 tests passing | 80 passed, 3 failed | **FAIL** |
