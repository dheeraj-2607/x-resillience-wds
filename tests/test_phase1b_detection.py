"""
Unit tests for Phase 1b: Detection Architecture & Data Interfaces.

Verifies dataset loading validation, BATADAL schema handling (ATT_FLAG, header whitespace,
-999 unlabelled values, unlabeled test sets), preprocessing, sliding-window generation,
scaler leakage prevention, feature extraction, spatio-temporal GNN model shape,
strict training gating (refusal without real dataset), and evaluation metrics.

Run with: pytest tests/test_phase1b_detection.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.dataset_loader import BATADALDatasetLoader, verify_dataset_is_real
from src.detection.evaluate import DetectionEvaluator
from src.detection.features import FeatureExtractor
from src.detection.graph_model import SpatioTemporalGNN, build_adjacency_matrix_from_graph
from src.detection.preprocessing import DataPreprocessor
from src.detection.scoring import calibrated_sensor_variance, sensor_variance_floor
from src.detection.train import train_model


@pytest.fixture
def synthetic_csv_path(tmp_path: Path) -> Path:
    """Create a small temporary CSV file representing a dummy time-series stream.

    NOTE: This fixture is used ONLY for unit testing software pipeline execution.
    It does not represent real BATADAL research data.
    """
    csv_file = tmp_path / "test_sensors.csv"
    dates = pd.date_range("2026-01-01", periods=50, freq="h")
    df = pd.DataFrame(
        {
            "DATETIME": dates,
            "S_PU1": np.sin(np.linspace(0, 10, 50)),
            "S_PU2": np.cos(np.linspace(0, 10, 50)),
            "P_J280": np.linspace(10, 20, 50),
            "L_T1": np.linspace(2, 5, 50),
            "ATTACK": [0] * 40 + [1] * 10,
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


class TestDatasetLoader:
    def test_file_not_found_raises(self):
        loader = BATADALDatasetLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_csv("nonexistent_path_xyz.csv")

    def test_empty_file_raises(self, tmp_path: Path):
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")
        loader = BATADALDatasetLoader()
        with pytest.raises(ValueError, match="empty"):
            loader.load_csv(empty_csv)

    def test_loads_and_validates_columns(self, synthetic_csv_path: Path):
        loader = BATADALDatasetLoader(timestamp_col="DATETIME", label_col="ATTACK")
        df, sensor_cols, label_col = loader.load_csv(synthetic_csv_path)

        assert len(df) == 50
        assert label_col == "ATTACK"
        assert set(sensor_cols) == {"S_PU1", "S_PU2", "P_J280", "L_T1"}

    def test_missing_specified_sensor_raises(self, synthetic_csv_path: Path):
        loader = BATADALDatasetLoader(sensor_cols=["S_PU1", "NON_EXISTENT_SENSOR"])
        with pytest.raises(ValueError, match="Specified sensor columns not found"):
            loader.load_csv(synthetic_csv_path)

    def test_att_flag_and_whitespace_and_minus999_preservation(self, tmp_path: Path):
        """Test ATT_FLAG detection, whitespace stripping, and -999 as unlabelled."""
        batadal_style_csv = tmp_path / "batadal_style.csv"
        df_raw = pd.DataFrame(
            {
                "DATETIME": ["01/01/16 00", "01/01/16 01", "01/01/16 02"],
                " L_T1": [1.5, 2.0, 2.5],
                " P_J280": [30.1, 30.2, 30.3],
                " ATT_FLAG": [-999, 0, 1],
            }
        )
        df_raw.to_csv(batadal_style_csv, index=False)

        loader = BATADALDatasetLoader(timestamp_col="DATETIME", label_col="ATT_FLAG")
        df, sensor_cols, label_col = loader.load_csv(batadal_style_csv)

        assert label_col == "ATT_FLAG"
        assert "L_T1" in sensor_cols
        assert "P_J280" in sensor_cols
        assert " L_T1" not in df.columns  # Header whitespace stripped
        assert np.isnan(df["ATT_FLAG"].iloc[0])
        assert list(df["ATT_FLAG"].iloc[1:]) == [0, 1]

    def test_unlabeled_test_set_handling(self, tmp_path: Path):
        """Test dataset loading when no label column is present (e.g. BATADAL_test_dataset.csv)."""
        unlabeled_csv = tmp_path / "unlabeled_test.csv"
        df_raw = pd.DataFrame(
            {
                "DATETIME": ["01/01/17 00", "01/01/17 01"],
                "L_T1": [1.5, 2.0],
                "P_J280": [30.1, 30.2],
            }
        )
        df_raw.to_csv(unlabeled_csv, index=False)

        loader = BATADALDatasetLoader(timestamp_col="DATETIME")
        df, sensor_cols, label_col = loader.load_csv(unlabeled_csv)

        assert label_col is None
        assert set(sensor_cols) == {"L_T1", "P_J280"}
        assert len(df) == 2

    def test_real_batadal_files_loading_if_present(self):
        """Verify that actual raw BATADAL CSV files load cleanly if present in data/raw/."""
        loader = BATADALDatasetLoader()
        raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"

        # 1. Test dataset03 (normal training baseline)
        ds03_path = raw_dir / "BATADAL_dataset03.csv"
        if ds03_path.exists():
            df03, sensors03, label03 = loader.load_csv(ds03_path)
            assert len(df03) == 8761
            assert label03 == "ATT_FLAG"
            assert set(df03["ATT_FLAG"].unique()) == {0}

        # 2. Test dataset04 (labeled validation)
        ds04_path = raw_dir / "BATADAL_dataset04.csv"
        if ds04_path.exists():
            df04, sensors04, label04 = loader.load_csv(ds04_path)
            assert len(df04) == 4177
            assert label04 == "ATT_FLAG"
            assert set(df04["ATT_FLAG"].dropna().unique()) == {1}
            assert df04["ATT_FLAG"].isna().any()
            assert "L_T1" in sensors04  # Whitespace stripped

        # 3. Test test_dataset (unlabeled evaluation)
        test_path = raw_dir / "BATADAL_test_dataset.csv"
        if test_path.exists():
            df_test, sensors_test, label_test = loader.load_csv(test_path)
            assert len(df_test) == 2089
            assert label_test is None  # Unlabeled test set supported

    def test_ambiguous_timestamp_parsing(self, tmp_path: Path):
        """Regression test: verify '01/02/17 00' is parsed as Feb 1st, 2017, not Jan 2nd."""
        ambiguous_csv = tmp_path / "ambiguous_dates.csv"
        df_raw = pd.DataFrame(
            {
                "DATETIME": ["01/02/17 00", "15/02/17 12"],
                "L_T1": [1.5, 2.0],
                "P_J280": [30.1, 30.2],
            }
        )
        df_raw.to_csv(ambiguous_csv, index=False)

        loader = BATADALDatasetLoader(timestamp_col="DATETIME")
        df, _, _ = loader.load_csv(ambiguous_csv)

        assert df["DATETIME"].iloc[0] == pd.Timestamp("2017-02-01 00:00:00")
        assert df["DATETIME"].iloc[1] == pd.Timestamp("2017-02-15 12:00:00")

    def test_batadal_test_timeline_monotonicity(self):
        """Regression test: verify full BATADAL test set timeline parsed monotonically without duplicates."""
        loader = BATADALDatasetLoader()
        test_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "BATADAL_test_dataset.csv"
        if test_path.exists():
            df, _, _ = loader.load_csv(test_path)
            ts = df["DATETIME"]
            assert len(ts) == 2089
            assert ts.iloc[0] == pd.Timestamp("2017-01-04 00:00:00")
            assert ts.iloc[-1] == pd.Timestamp("2017-04-01 00:00:00")
            assert ts.is_monotonic_increasing
            assert ts.isnull().sum() == 0
            assert ts.duplicated().sum() == 0


class TestPreprocessing:
    def test_missing_value_imputation(self):
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({"S1": [1.0, np.nan, 3.0, np.nan], "S2": [10.0, 20.0, np.nan, 40.0]})
        clean_df = preprocessor.handle_missing_values(df, ["S1", "S2"], strategy="ffill")
        assert not clean_df[["S1", "S2"]].isna().any().any()

    def test_temporal_split_order(self):
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({"val": range(100)})
        train_df, val_df, test_df = preprocessor.temporal_split(
            df, train_ratio=0.7, val_ratio=0.15
        )
        assert len(train_df) == 70
        assert len(val_df) == 15
        assert len(test_df) == 15
        assert list(train_df["val"]) == list(range(70))
        assert list(val_df["val"]) == list(range(70, 85))
        assert list(test_df["val"]) == list(range(85, 100))

    def test_scaler_fitting_and_no_leakage(self):
        preprocessor = DataPreprocessor(scaler_type="standard")
        train_df = pd.DataFrame({"S1": [0.0, 10.0]})
        test_df = pd.DataFrame({"S1": [20.0]})

        preprocessor.fit_scaler(train_df, ["S1"])
        assert preprocessor.is_fitted

        # Transform using training scaler
        scaled_test = preprocessor.transform_features(test_df, ["S1"])
        # Standard scaler mean=5, std=5 -> (20 - 5)/5 = 3.0
        assert np.isclose(scaled_test[0, 0], 3.0)

    def test_unfitted_scaler_transform_raises(self):
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({"S1": [1.0, 2.0]})
        with pytest.raises(RuntimeError, match="Scaler has not been fitted"):
            preprocessor.transform_features(df, ["S1"])

    def test_sliding_window_dimensions(self):
        preprocessor = DataPreprocessor(window_size=10, stride=1)
        features = np.random.randn(50, 4)
        labels = np.array([0] * 40 + [1] * 10)

        X_win, y_win = preprocessor.create_sliding_windows(features, labels)
        # N = (50 - 10) // 1 + 1 = 41 windows
        assert X_win.shape == (41, 10, 4)
        assert y_win.shape == (41,)
        # Last window end index is 50 -> target is labels[49] = 1
        assert y_win[-1] == 1.0


class TestFeatureExtractor:
    def test_extract_diffs_and_rolling(self):
        extractor = FeatureExtractor(include_diffs=True, include_rolling=True, rolling_window=3)
        df = pd.DataFrame({"S1": np.arange(10, dtype=float), "S2": np.arange(10, dtype=float) * 2})
        df_feats, feature_names = extractor.extract_features(df, ["S1", "S2"])

        # 2 base (S1, S2) + 2 diffs (S1_diff, S2_diff) + 2 rolling means + 2 rolling stds = 8
        assert len(feature_names) == 8
        assert "S1_diff" in feature_names
        assert "S1_roll_mean_3" in feature_names
        assert not df_feats[feature_names].isna().any().any()


class TestGraphModel:
    def test_model_forward_pass_dimensions(self):
        model = SpatioTemporalGNN(in_features=4, window_size=12, hidden_dim=32)
        batch_x = torch.randn(8, 12, 4)  # (Batch=8, Window=12, Features=4)
        preds = model(batch_x)
        assert preds.shape == (8, 4)  # Forecasts next-step sensor states (B, F)

    def test_model_forward_pass_with_adjacency_matrix(self):
        model = SpatioTemporalGNN(in_features=4, window_size=12, hidden_dim=32)
        batch_x = torch.randn(8, 12, 4)
        adj_matrix = torch.eye(4)
        preds = model(batch_x, adj_matrix=adj_matrix)
        assert preds.shape == (8, 4)

    def test_build_adjacency_matrix_from_graph(self):
        inp_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "CTOWN.INP"
        if inp_path.exists():
            from src.network.ctown_loader import load_ctown_network
            from src.network.graph_builder import build_network_graph
            net = load_ctown_network(inp_path)
            g = build_network_graph(net)
            sensor_cols = ["L_T1", "L_T2", "F_PU1", "S_PU1", "P_J280", "P_J269"]
            adj = build_adjacency_matrix_from_graph(g, sensor_cols, max_hops=3)
            assert adj.shape == (6, 6)
            assert torch.count_nonzero(adj) > 6


class TestForecastingAndTopology:
    def test_all_43_sensors_resolve_to_ctown_topology(self):
        inp_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "CTOWN.INP"
        ds03_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "BATADAL_dataset03.csv"
        if inp_path.exists() and ds03_path.exists():
            from src.detection.graph_model import resolve_sensor_locations
            from src.network.ctown_loader import load_ctown_network
            from src.network.graph_builder import build_network_graph
            net = load_ctown_network(inp_path)
            g = build_network_graph(net)
            loader = BATADALDatasetLoader()
            _, sensor_cols, _ = loader.load_csv(ds03_path)
            sensor_nodes = resolve_sensor_locations(g, sensor_cols)
            assert len(sensor_nodes) == 43
            assert all(len(nodes) > 0 for nodes in sensor_nodes.values())

    def test_sensor_adjacency_is_non_identity(self):
        inp_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "CTOWN.INP"
        ds03_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "BATADAL_dataset03.csv"
        if inp_path.exists() and ds03_path.exists():
            from src.network.ctown_loader import load_ctown_network
            from src.network.graph_builder import build_network_graph
            net = load_ctown_network(inp_path)
            g = build_network_graph(net)
            loader = BATADALDatasetLoader()
            _, sensor_cols, _ = loader.load_csv(ds03_path)
            adj = build_adjacency_matrix_from_graph(g, sensor_cols, max_hops=3)
            assert adj.shape == (43, 43)
            assert int(torch.count_nonzero(adj)) > 43

    def test_threshold_calibration_on_normal_data(self):
        calib_scores = np.random.exponential(scale=0.1, size=1000)
        threshold = float(np.percentile(calib_scores, 99.5))
        exceeded = np.sum(calib_scores > threshold)
        assert exceeded <= 10

    def test_scaler_fitted_only_on_training_data(self):
        preprocessor = DataPreprocessor()
        train_df = pd.DataFrame({"S1": [10.0, 20.0, 30.0]})
        val_df = pd.DataFrame({"S1": [100.0, 200.0]})
        preprocessor.fit_scaler(train_df, ["S1"])
        scaled_val = preprocessor.transform_features(val_df, ["S1"])
        assert preprocessor.scaler.mean_[0] == 20.0
        assert scaled_val[0, 0] > 5.0

    def test_no_val_or_test_labels_enter_training(self):
        preprocessor = DataPreprocessor()
        train_df = pd.DataFrame({"S1": np.random.randn(50)})
        val_df = pd.DataFrame({"S1": np.random.randn(30), "ATT_FLAG": [0] * 20 + [1] * 10})
        preprocessor.fit_scaler(train_df, ["S1"])
        train_scaled = preprocessor.transform_features(train_df, ["S1"])
        X_tr, Y_tr, lbl_tr = preprocessor.create_forecasting_windows(train_scaled)
        assert lbl_tr is None  # Training features have no label input


class TestTrainingGating:
    def test_train_model_refuses_without_real_dataset(self, tmp_path: Path):
        fake_path = tmp_path / "fake_dataset.csv"
        config = {
            "detection": {
                "data_path": str(fake_path),
                "allow_synthetic_training": False,
            }
        }
        with pytest.raises(RuntimeError, match="MODEL TRAINING REFUSED"):
            train_model(config, dataset_path=str(fake_path), allow_synthetic=False)

    def test_train_model_runs_when_allow_synthetic_override_is_set(
        self, synthetic_csv_path: Path, tmp_path: Path
    ):
        config = {
            "paths": {"models_dir": str(tmp_path / "models")},
            "detection": {
                "data_path": str(synthetic_csv_path),
                "window_size": 5,
                "stride": 1,
                "epochs": 2,
                "batch_size": 8,
                "allow_synthetic_training": True,  # Allowed FOR PIPELINE TESTING ONLY
            },
        }
        res = train_model(config, dataset_path=str(synthetic_csv_path), allow_synthetic=True)
        assert res["status"] == "COMPLETED"
        assert len(res["history"]["train_loss"]) == 2


class TestEvaluationMetrics:
    def test_detection_evaluator_metrics(self):
        evaluator = DetectionEvaluator(threshold=0.5)
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_probs = np.array([0.1, 0.2, 0.1, 0.8, 0.7, 0.9, 0.8, 0.2])

        results = evaluator.evaluate_predictions(y_true, y_probs)

        assert results["metric_category"] == "DETECTION_METRICS"
        assert "precision" in results
        assert "recall" in results
        assert "f1_score" in results
        assert "roc_auc" in results
        assert "pr_auc" in results
        assert "false_positive_rate" in results
        assert results["confusion_matrix"]["TP"] == 3
        assert results["confusion_matrix"]["FP"] == 1


def test_sensor_variance_floors_use_physical_precision_in_standardized_space():
    scaler = DataPreprocessor().scaler
    scaler.mean_ = np.zeros(4)
    scaler.scale_ = np.array([2.0, 0.5, 10.0, 1.0])
    scaler.var_ = np.square(scaler.scale_)
    scaler.n_features_in_ = 4
    sensors = ["P_J280", "L_T1", "F_PU3", "S_PU3"]

    floors = sensor_variance_floor(sensors, scaler)
    variances = calibrated_sensor_variance(np.zeros((3, 4)), sensors, scaler)

    np.testing.assert_allclose(floors, [0.05**2 / 2**2, 0.01**2 / 0.5**2, 0.50**2 / 10**2, 0.01])
    np.testing.assert_allclose(variances, floors)
    assert np.isfinite(variances).all()

