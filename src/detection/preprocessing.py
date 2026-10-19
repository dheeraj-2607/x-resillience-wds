"""
preprocessing.py

Data preprocessing, temporal train/val/test splitting, scaler fitting without
data leakage, missing value imputation, and sliding-window tensor construction
for multivariate WDS time-series detection models.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles time-series cleaning, temporal splitting, scaling, and sliding windows.

    Attributes:
        scaler_type: 'standard' or 'minmax'
        window_size: Number of time steps per sliding window frame (W)
        stride: Step size between successive sliding windows
    """

    def __init__(
        self,
        scaler_type: str = "standard",
        window_size: int = 12,
        stride: int = 1,
    ):
        self.scaler_type = scaler_type
        self.window_size = window_size
        self.stride = stride
        self.scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        self.is_fitted = False

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        strategy: str = "ffill",
    ) -> pd.DataFrame:
        """Impute missing values in sensor feature columns.

        Args:
            df: DataFrame containing sensor columns.
            sensor_cols: List of column names to impute.
            strategy: 'ffill' (forward fill then backward fill) or 'zero'.

        Returns:
            DataFrame with imputed sensor columns.
        """
        df_clean = df.copy()
        if strategy == "ffill":
            df_clean[sensor_cols] = df_clean[sensor_cols].ffill().bfill().fillna(0.0)
        elif strategy == "zero":
            df_clean[sensor_cols] = df_clean[sensor_cols].fillna(0.0)
        else:
            raise ValueError(f"Unknown missing value imputation strategy '{strategy}'")
        return df_clean

    def temporal_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split DataFrame chronologically into train, validation, and test sets.

        Maintains temporal order to prevent data leakage.

        Args:
            df: Full chronologically-ordered DataFrame.
            train_ratio: Proportion of rows for training.
            val_ratio: Proportion of rows for validation.

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        logger.info(
            "Temporal split: Train=%d, Val=%d, Test=%d (Total=%d)",
            len(train_df),
            len(val_df),
            len(test_df),
            n,
        )
        return train_df, val_df, test_df

    def fit_scaler(self, train_df: pd.DataFrame, sensor_cols: List[str]) -> None:
        """Fit the scaler strictly on the training set features to prevent leakage."""
        train_features = train_df[sensor_cols].values
        self.scaler.fit(train_features)
        self.is_fitted = True
        logger.info("Fitted %s scaler on training set shape %s", self.scaler_type, train_features.shape)

    def transform_features(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
    ) -> np.ndarray:
        """Transform features using the fitted training scaler.

        Raises:
            RuntimeError: If fit_scaler has not been called first.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted. Call fit_scaler on training set first.")
        features = df[sensor_cols].values
        return self.scaler.transform(features)

    def create_sliding_windows(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Construct sliding window tensors from 2D time-series feature array.

        Args:
            features: 2D array of shape (T, F), where T=timesteps, F=features.
            labels: Optional 1D array of shape (T,) containing binary attack labels.

        Returns:
            Tuple of:
              - X_windows: 3D array of shape (N, W, F), where N=number of windows,
                W=window_size, F=features.
              - y_windows: 1D array of shape (N,) containing binary window labels
                (label of the last timestamp in each window), or None if labels is None.

        Raises:
            ValueError: If input length T is less than window_size W.
        """
        T, F = features.shape
        W = self.window_size
        S = self.stride

        if T < W:
            raise ValueError(f"Sequence length T={T} is smaller than window_size W={W}")

        n_windows = (T - W) // S + 1
        X_windows = np.zeros((n_windows, W, F), dtype=np.float32)
        y_windows = np.zeros(n_windows, dtype=np.float32) if labels is not None else None

        for i in range(n_windows):
            start_idx = i * S
            end_idx = start_idx + W
            X_windows[i] = features[start_idx:end_idx]
            if labels is not None:
                # Use the target label of the last timestamp in the window
                y_windows[i] = labels[end_idx - 1]

        logger.debug(
            "Created %d sliding windows of shape (W=%d, F=%d) from T=%d sequence",
            n_windows,
            W,
            F,
            T,
        )
        return X_windows, y_windows

    def create_forecasting_windows(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        forecast_horizon: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Construct sliding windows and multi-step forecasting targets.

        Given sequence features of shape (T, F), constructs:
          - X_windows: (N, W, F) historical sliding window [i : i+W]
          - Y_targets: (N, F) if H=1 or (N, H, F) if H>1, target timesteps [i+W : i+W+H]
          - y_labels: (N,) binary attack ground-truth label at target timestep [i+W] (if labels provided)

        Args:
            features: 2D array of shape (T, F).
            labels: Optional 1D array of shape (T,) containing binary attack labels.
            forecast_horizon: Number of future time steps to forecast (default 1).

        Returns:
            Tuple of (X_windows, Y_targets, y_labels)
        """
        T, F = features.shape
        W = self.window_size
        S = self.stride

        max_available_horizon = max(1, T - W)
        H = min(int(forecast_horizon), max_available_horizon)

        if T < W + H:
            raise ValueError(f"Sequence length T={T} must be >= W+H={W+H} for forecasting horizon H={H}.")

        n_windows = (T - W - H + 1) // S
        X_windows = np.zeros((n_windows, W, F), dtype=np.float32)
        if H == 1:
            Y_targets = np.zeros((n_windows, F), dtype=np.float32)
        else:
            Y_targets = np.zeros((n_windows, H, F), dtype=np.float32)
        y_labels = np.zeros(n_windows, dtype=np.float32) if labels is not None else None

        for i in range(n_windows):
            start_idx = i * S
            end_idx = start_idx + W
            X_windows[i] = features[start_idx:end_idx]
            if H == 1:
                Y_targets[i] = features[end_idx]
            else:
                Y_targets[i] = features[end_idx : end_idx + H]
            if labels is not None:
                y_labels[i] = labels[end_idx]

        return X_windows, Y_targets, y_labels

