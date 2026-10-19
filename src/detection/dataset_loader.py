"""
dataset_loader.py

Data loading interface for BATADAL-style time-series sensor datasets.
Validates file presence, timestamp parsing, whitespace-stripped headers,
sensor feature columns, label column detection ('ATT_FLAG', 'ATTACK'),
preserves unlabelled -999 labels as missing values, and supports unlabeled test sets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

CANDIDATE_LABEL_COLS = ["ATT_FLAG", "ATTACK", "label", "target"]


class BATADALDatasetLoader:
    """Interface for loading and validating BATADAL multivariate WDS time-series data.

    Attributes:
        timestamp_col: Name of the timestamp column (e.g. 'DATETIME').
        label_col: Name of the attack ground-truth label column (defaults to 'ATT_FLAG').
        sensor_cols: Optional list of sensor column names. If None, auto-detected.
    """

    def __init__(
        self,
        timestamp_col: str = "DATETIME",
        label_col: str = "ATT_FLAG",
        sensor_cols: Optional[List[str]] = None,
    ):
        self.timestamp_col = timestamp_col
        self.label_col = label_col
        self.sensor_cols = sensor_cols

    def load_csv(
        self,
        filepath: Union[str, Path],
        parse_dates: bool = True,
    ) -> Tuple[pd.DataFrame, List[str], Optional[str]]:
        """Load a CSV time-series file, strip header whitespace, normalize labels, and validate columns.

        Args:
            filepath: Path to the dataset CSV file.
            parse_dates: Whether to parse the timestamp column to datetime.

        Returns:
            Tuple of:
              - df: Cleaned/validated DataFrame
              - sensor_cols: List of validated sensor feature column names
              - label_col: Name of validated label column, or None if absent (unlabeled test set)

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the file is empty or missing required sensor columns.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path.resolve()}")

        if path.stat().st_size == 0:
            raise ValueError(f"Dataset file is empty (0 bytes): {path}")

        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV file {path}: {str(e)}") from e

        if df.empty:
            raise ValueError(f"Dataset CSV contains no rows: {path}")

        # REQUIREMENT 2: Strip leading/trailing whitespace from all CSV column headers
        df.columns = df.columns.astype(str).str.strip()

        # Validate timestamp column
        if self.timestamp_col in df.columns:
            if parse_dates:
                try:
                    df[self.timestamp_col] = pd.to_datetime(
                        df[self.timestamp_col], format="%d/%m/%y %H"
                    )
                except Exception as e:
                    try:
                        df[self.timestamp_col] = pd.to_datetime(
                            df[self.timestamp_col], dayfirst=True
                        )
                    except Exception as inner_e:
                        logger.warning(
                            "Could not parse timestamp column '%s' as datetime: %s",
                            self.timestamp_col,
                            str(inner_e),
                        )
        else:
            logger.info("Timestamp column '%s' not present in %s", self.timestamp_col, path)

        # REQUIREMENT 1 & 4: Detect attack label column (ATT_FLAG, ATTACK, etc.) or None if unlabeled test set
        candidates = [self.label_col] + [c for c in CANDIDATE_LABEL_COLS if c != self.label_col]
        found_label_col = None
        for cand in candidates:
            if cand in df.columns:
                found_label_col = cand
                break

        # Preserve BATADAL's -999 unlabelled sentinel instead of treating it as normal.
        if found_label_col is not None:
            df[found_label_col] = pd.to_numeric(df[found_label_col], errors="raise").replace(
                -999, float("nan")
            )
            logger.info(
                "Normalized label column '%s' in %s (Value counts: %s)",
                found_label_col,
                path,
                df[found_label_col].value_counts().to_dict(),
            )
        else:
            logger.info("No attack label column found in %s (Unlabeled test dataset)", path)

        # Auto-detect or validate sensor feature columns
        if self.sensor_cols is not None:
            # Strip whitespace from requested sensor_cols as well
            req_sensors = [s.strip() for s in self.sensor_cols]
            missing_sensors = [col for col in req_sensors if col not in df.columns]
            if missing_sensors:
                raise ValueError(
                    f"Specified sensor columns not found in dataset: {missing_sensors}"
                )
            detected_sensors = list(req_sensors)
        else:
            # Auto-detect all numeric feature columns except timestamp and label columns
            excluded = {self.timestamp_col}
            if found_label_col:
                excluded.add(found_label_col)
            for c in CANDIDATE_LABEL_COLS:
                excluded.add(c)

            detected_sensors = [
                col
                for col in df.columns
                if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
            ]

        if not detected_sensors:
            raise ValueError(
                f"No numeric sensor feature columns found in dataset {path}. Columns present: {list(df.columns)}"
            )

        logger.info(
            "Loaded dataset %s: %d rows, %d sensor features, label column='%s'",
            path,
            len(df),
            len(detected_sensors),
            found_label_col,
        )

        return df, detected_sensors, found_label_col


def verify_dataset_is_real(filepath: Optional[Union[str, Path]]) -> bool:
    """Helper to check if a real dataset file exists at the given path.

    Returns False if path is None, does not exist, or is empty.
    """
    if filepath is None:
        return False
    p = Path(filepath)
    return p.exists() and p.is_file() and p.stat().st_size > 0
