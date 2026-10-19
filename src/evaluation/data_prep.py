"""One place that builds every data split, so all experiments use identical inputs.

Precision fix: BATADAL_dataset03.csv stores sensor values with 3+ decimals, while
BATADAL_dataset04.csv and the test file are rounded to 2 decimals. P_J280 has a
training standard deviation of only 0.0065, so 2-decimal rounding alone moves it by
~1.5 standard deviations in dataset04/test. The model reads this as a real input
change and P_J280 then dominates the SHAP explanations. `harmonise_precision` rounds
dataset03 to the precision of the evaluation files before anything is fitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.detection.dataset_loader import BATADALDatasetLoader
from src.detection.preprocessing import DataPreprocessor
from src.evaluation.labels import HORIZON, WINDOW, interval_labels, official_test_labels, target_timestamps

DATASET_FILES = {
    "Dataset04": "BATADAL_dataset04.csv",
    "Test Dataset": "BATADAL_test_dataset.csv",
}


def harmonise_precision(frame: pd.DataFrame, sensors: list[str], decimals: int = 2) -> pd.DataFrame:
    out = frame.copy()
    out[sensors] = out[sensors].astype(float).round(decimals)
    return out


@dataclass
class Splits:
    sensors: list[str]
    preprocessor: DataPreprocessor
    train: tuple[np.ndarray, np.ndarray]
    calibration: tuple[np.ndarray, np.ndarray]
    # name -> (windows, targets, target_timestamps)
    evaluation: dict[str, tuple[np.ndarray, np.ndarray, pd.Series]] = field(default_factory=dict)
    labels: dict[str, np.ndarray] = field(default_factory=dict)


def build_splits(root: str | Path, attacks: pd.DataFrame, harmonise: bool = True) -> Splits:
    """Dataset03: first 80% train, last 20% calibration (normal only).
    Dataset04 labels: ground-truth intervals. Test labels: official ATTACK_FLAG."""
    root = Path(root)
    loader = BATADALDatasetLoader()
    d03, sensors03, _ = loader.load_csv(root / "data/raw/BATADAL_dataset03.csv")
    frames = {name: loader.load_csv(root / "data/raw" / f)[0] for name, f in DATASET_FILES.items()}
    sensors = [s for s in sensors03 if all(s in f.columns for f in frames.values())]
    if harmonise:
        d03 = harmonise_precision(d03, sensors)
    split = int(0.8 * len(d03))
    pp = DataPreprocessor(window_size=WINDOW)
    train_frame = pp.handle_missing_values(d03.iloc[:split], sensors)
    calibration_frame = pp.handle_missing_values(d03.iloc[split:], sensors)
    pp.fit_scaler(train_frame, sensors)

    def windows(frame):
        x, y, _ = pp.create_forecasting_windows(pp.transform_features(frame, sensors), forecast_horizon=HORIZON)
        return x.astype(np.float32), y.astype(np.float32)

    splits = Splits(sensors, pp, windows(train_frame), windows(calibration_frame))
    for name, frame in frames.items():
        frame = pp.handle_missing_values(frame, sensors)
        x, y = windows(frame)
        ts = target_timestamps(frame)
        if len(ts) != len(x):
            raise RuntimeError(f"{name}: timestamps and windows are misaligned.")
        splits.evaluation[name] = (x, y, ts)
    splits.labels["Dataset04"] = interval_labels(splits.evaluation["Dataset04"][2], attacks[attacks.dataset == "Dataset04"])
    test_path = root / "data/raw" / DATASET_FILES["Test Dataset"]
    if "ATTACK_FLAG" in {str(column).strip() for column in frames["Test Dataset"].columns}:
        splits.labels["Test Dataset"] = official_test_labels(test_path)
    else:
        splits.labels["Test Dataset"] = interval_labels(
            splits.evaluation["Test Dataset"][2],
            attacks[attacks.dataset == "Test Dataset"],
        )
    return splits
