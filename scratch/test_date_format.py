from pathlib import Path

import pandas as pd
import pytest

TEST_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "BATADAL_test_dataset.csv"


@pytest.mark.skipif(
    not TEST_DATA_PATH.exists(),
    reason="BATADAL_test_dataset.csv is not present in this workspace; date-format validation is skipped until the raw dataset is available.",
)
def test_batadal_test_dataset_uses_explicit_ddmmyy_hh_format():
    df = pd.read_csv(TEST_DATA_PATH)
    raw_dates = df["DATETIME"].astype(str).str.strip()

    dt_explicit = pd.to_datetime(raw_dates, format="%d/%m/%y %H")

    assert dt_explicit.is_monotonic_increasing
    assert dt_explicit.isna().sum() == 0
    assert isinstance(dt_explicit.iloc[0], pd.Timestamp)
