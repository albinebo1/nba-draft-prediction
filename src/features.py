"""
Feature engineering for the NBA Draft Bust/Star prediction task.

Public API:
  * ``FEATURE_COLUMNS``           - the canonical input feature list
  * ``CATEGORICAL_FEATURES``      - subset that needs one-hot encoding
  * ``NUMERIC_FEATURES``          - subset that needs imputation + scaling
  * ``TARGET_COLUMN``             - 'career_label'
  * ``CLASS_ORDER``               - ['Bust', 'Solid', 'Star']
  * ``make_preprocessor(scale)``  - sklearn ColumnTransformer
  * ``load_xy()``                 - return X (DataFrame), y (Series) ready to use
  * ``train_test_split_stratified(X, y)`` - reproducible 80/20 stratified split

Design notes:
  - Imputation (median) and StandardScaler are applied INSIDE a Pipeline so
    that they fit on training folds only. This avoids the classic
    leakage where imputation statistics are computed over the full dataset.
  - ``position`` has missing values for ~54% of rows (pre-2000 picks +
    no-combine prospects); we treat NaN as its own category ``"Unknown"`` via
    a SimpleImputer with fill_value strategy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "nba_draft_processed.csv"

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "draft_pick",
    "draft_round",
    "height_no_shoes",
    "weight",
    "wingspan",
    "standing_reach",
    "hand_length",
    "hand_width",
    "standing_vertical",
    "vertical_leap_max",
    "lane_agility",
    "shuttle_run",
    "sprint",
    "bench_press",
]

CATEGORICAL_FEATURES = ["position"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "career_label"
CLASS_ORDER = ["Bust", "Solid", "Star"]


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def make_preprocessor(scale: bool = False) -> ColumnTransformer:
    """Build the sklearn ColumnTransformer.

    Args:
        scale: include StandardScaler on numeric features. Needed for SVM,
            harmless for trees but adds compute, so we opt out by default.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    # OneHotEncoder: handle_unknown so unseen positions at predict time don't
    # blow up. Position NaN → SimpleImputer fills with 'Unknown' then OHE.
    # sklearn >=1.2 prefers ``sparse_output``; older versions use ``sparse``.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ohe", ohe),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_xy(processed_csv: Path | str = PROCESSED_CSV) -> tuple[pd.DataFrame, pd.Series]:
    """Load the processed CSV and return X (feature DataFrame) and y (Series)."""
    df = pd.read_csv(processed_csv)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"processed CSV is missing required columns: {missing}")
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def load_full(processed_csv: Path | str = PROCESSED_CSV) -> pd.DataFrame:
    """Load the full processed DataFrame including player name etc., useful
    for error analysis."""
    return pd.read_csv(processed_csv)


def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
):
    """80/20 stratified split. Returns the same shape as sklearn's helper."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


if __name__ == "__main__":
    X, y = load_xy()
    print(f"X: {X.shape}, y: {y.shape}")
    print(f"feature dtypes:\n{X.dtypes}")
    print(f"\ntarget distribution:\n{y.value_counts()}")

    pre = make_preprocessor(scale=True)
    Xt = pre.fit_transform(X, y)
    print(f"\ntransformed shape: {Xt.shape}")
