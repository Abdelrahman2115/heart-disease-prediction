"""Loading, validating and cleaning the raw dataset.

Design rule: the raw CSV on disk is never modified. Every cleaning decision
happens here, in code, so it is visible, testable and reproducible.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from heart import config

logger = logging.getLogger(__name__)


@dataclass
class CleaningReport:
    """What clean_data changed, so it can be logged and documented."""

    rows_in: int
    duplicates_removed: int
    rows_out: int
    invalid_codes_set_missing: dict[str, int]


def load_raw(path: Path | str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and check that it has exactly the expected columns."""
    df = pd.read_csv(path)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Fail loudly if columns are missing or unexpected.

    Never assume the schema: a silently renamed column would otherwise turn
    into a feature full of NaNs further down the pipeline.
    """
    expected, actual = set(config.RAW_COLUMNS), set(df.columns)
    missing, extra = expected - actual, actual - expected
    if missing or extra:
        raise ValueError(f"Schema mismatch. Missing: {sorted(missing)}; unexpected: {sorted(extra)}")
    non_numeric = [c for c in config.RAW_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Expected numeric columns, got non-numeric: {non_numeric}")


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Remove exact duplicate rows and mark invalid category codes as missing.

    * Duplicates: the Kaggle file has 1,025 rows but only 302 unique ones.
      Keeping copies would put the same patient in both train and test sets
      (data leakage) and inflate every test metric.
    * Invalid codes (e.g. ca == 4, thal == 0) are not real categories in the
      source data; they most likely encode "not recorded". We set them to NaN
      and let the pipeline impute them, rather than deleting whole patients.
    """
    rows_in = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = rows_in - len(out)

    invalid_counts: dict[str, int] = {}
    out = out.astype({c: "float64" for c in config.VALID_CODES})
    for col, allowed in config.VALID_CODES.items():
        bad = ~out[col].isin(allowed) & out[col].notna()
        if bad.any():
            invalid_counts[col] = int(bad.sum())
            out.loc[bad, col] = np.nan

    report = CleaningReport(rows_in, duplicates_removed, len(out), invalid_counts)
    logger.info("Cleaning: %s", report)
    return out, report


def make_label(df: pd.DataFrame, invert: bool = config.INVERT_TARGET) -> pd.Series:
    """Create the modelling label where 1 always means 'disease'."""
    y = df[config.RAW_TARGET].astype(int)
    y = (1 - y) if invert else y
    return y.rename(config.LABEL)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
):
    """Stratified hold-out split.

    Stratifying keeps the disease/no-disease ratio the same in both sets.
    The test set returned here must not be used for any modelling decision.
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def load_dataset():
    """Convenience: raw -> cleaned features X and label y."""
    df, report = clean_data(load_raw())
    return df[config.FEATURES], make_label(df), report
