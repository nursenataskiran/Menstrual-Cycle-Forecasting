# -*- coding: utf-8 -*-
"""
src/daily_dataset_builder.py
=============================
Builds a daily-level dataset from raw cycle-level menstrual data.

What this module does:
  - Loads raw cycle-level menstrual dataset
  - Normalises column names
  - Parses and validates date columns
  - Checks and enforces per-user chronological ordering
  - Produces leakage-safe historical context features
  - Expands cycle-level rows to daily-level rows
  - Produces days_until_next_cycle (regression target)
  - Produces phase_label (classification target)
  - Applies diet/symptoms one-hot and exercise ordinal encoding

Outputs:
  data/processed/daily_data.csv
  data/processed/feature_info.json

What this module does NOT do (reserved for later steps):
  - Scaling / normalization
  - Train / validation / test split
  - Sliding window sequence generation
  - Model training
"""

import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Required raw column names (after normalisation)
REQUIRED_COLUMNS = [
    "user_id",
    "age",
    "bmi",
    "stress_level",
    "exercise_frequency",
    "sleep_hours",
    "diet",
    "cycle_start_date",
    "cycle_length",
    "period_length",
    "next_cycle_start_date",
    "symptoms",
]

# Population-level prior values (Bull et al., 2019 -- Nature Digital Medicine)
COLD_START_PRIORS = {
    "hist_mean_cycle": 29.3,
    "hist_mean_period": 4.0,
}

# Phase integer labels
PHASE_MAP = {
    "menstruation": 0,
    "follicular": 1,
    "ovulation": 2,
    "luteal": 3,
}

# ---------------------------------------------------------------------------
# Encoding constants
# ---------------------------------------------------------------------------

# Ordinal mapping for exercise_frequency
EXERCISE_ORDINAL_MAP = {"Low": 0, "Moderate": 1, "High": 2}

# Expected diet categories (title-case as they appear in the raw data)
DIET_CATEGORIES = ["Balanced", "High Sugar", "Low Carb", "Vegetarian"]

# Corresponding one-hot column names
DIET_OHE_COLS = ["diet_balanced", "diet_high_sugar", "diet_low_carb", "diet_vegetarian"]

# Expected symptom categories
SYMPTOM_CATEGORIES = ["Bloating", "Cramps", "Fatigue", "Headache", "Mood Swings"]

# Corresponding one-hot column names
SYMPTOM_OHE_COLS = [
    "symptoms_bloating",
    "symptoms_cramps",
    "symptoms_fatigue",
    "symptoms_headache",
    "symptoms_mood_swings",
]

# Columns that must NEVER appear in input_features (leakage guard)
_LEAKAGE_COLS = {
    "user_id", "cycle_start_date", "next_cycle_start_date", "sample_date",
    "cycle_length", "period_length",
    "exercise_frequency", "diet", "symptoms",
    "days_until_next_cycle", "phase_label", "phase_name",
}

# Columns kept in daily_data.csv (in order)
DAILY_COLUMNS_ORDERED = [
    # Metadata
    "user_id",
    "cycle_start_date",
    "next_cycle_start_date",
    "sample_date",
    # Current cycle info
    "cycle_length",
    "period_length",
    # Daily target / label columns
    "day_in_cycle",
    "days_until_next_cycle",
    "phase_label",
    "phase_name",
    # Original context (raw -- kept for audit)
    "age",
    "bmi",
    "stress_level",
    "exercise_frequency",
    "sleep_hours",
    "diet",
    "symptoms",
    # Historical context
    "hist_mean_cycle",
    "hist_mean_period",
    "is_historical_data_missing",
    # Encoded features
    "exercise_frequency_encoded",
    "diet_balanced",
    "diet_high_sugar",
    "diet_low_carb",
    "diet_vegetarian",
    "symptoms_bloating",
    "symptoms_cramps",
    "symptoms_fatigue",
    "symptoms_headache",
    "symptoms_mood_swings",
]

# feature_info.json content
FEATURE_INFO = {
    "not_encoded_yet": False,
    "not_scaled_yet": True,
    "not_split_yet": True,
    "not_windowed_yet": True,
    "daily_level_file": "daily_data.csv",
    "raw_categorical_columns": [
        "exercise_frequency",
        "diet",
        "symptoms",
    ],
    "numeric_context_columns": [
        "age",
        "bmi",
        "stress_level",
        "sleep_hours",
        "hist_mean_cycle",
        "hist_mean_period",
        "is_historical_data_missing",
        "exercise_frequency_encoded",
    ],
    "encoded_categorical_columns": [
        "diet_balanced",
        "diet_high_sugar",
        "diet_low_carb",
        "diet_vegetarian",
        "symptoms_bloating",
        "symptoms_cramps",
        "symptoms_fatigue",
        "symptoms_headache",
        "symptoms_mood_swings",
    ],
    "daily_columns": [
        "day_in_cycle",
    ],
    "targets": {
        "regression": "days_until_next_cycle",
        "classification": "phase_label",
    },
    "input_features": [
        "day_in_cycle",
        "age",
        "bmi",
        "stress_level",
        "sleep_hours",
        "hist_mean_cycle",
        "hist_mean_period",
        "is_historical_data_missing",
        "exercise_frequency_encoded",
        "diet_balanced",
        "diet_high_sugar",
        "diet_low_carb",
        "diet_vegetarian",
        "symptoms_bloating",
        "symptoms_cramps",
        "symptoms_fatigue",
        "symptoms_headache",
        "symptoms_mood_swings",
    ],
    "excluded_from_input": [
        "user_id",
        "cycle_start_date",
        "next_cycle_start_date",
        "sample_date",
        "cycle_length",
        "period_length",
        "exercise_frequency",
        "diet",
        "symptoms",
        "phase_name",
        "days_until_next_cycle",
        "phase_label",
    ],
    "phase_mapping": {
        "menstruation": 0,
        "follicular": 1,
        "ovulation": 2,
        "luteal": 3,
    },
    "categorical_encoding": {
        "exercise_frequency": {
            "method": "ordinal",
            "mapping": {"Low": 0, "Moderate": 1, "High": 2},
            "encoded_column": "exercise_frequency_encoded",
            "rationale": (
                "Exercise frequency has a natural ordinal order (Low < Moderate < High), "
                "so ordinal encoding preserves the intensity direction. "
                "Diet and symptoms have no natural ordering and are one-hot encoded."
            ),
        },
        "diet": {
            "method": "one_hot",
            "columns": [
                "diet_balanced",
                "diet_high_sugar",
                "diet_low_carb",
                "diet_vegetarian",
            ],
        },
        "symptoms": {
            "method": "one_hot",
            "columns": [
                "symptoms_bloating",
                "symptoms_cramps",
                "symptoms_fatigue",
                "symptoms_headache",
                "symptoms_mood_swings",
            ],
            "note": (
                "Symptoms is treated as single-label categorical in this synthetic dataset. "
                "In a real multi-symptom setting, multi-hot encoding would be required."
            ),
        },
    },
    "cold_start_priors": {
        "hist_mean_cycle": 29.3,
        "hist_mean_period": 4.0,
        "source": "Bull et al. (2019), Nature Digital Medicine",
    },
    "leakage_note": (
        "cycle_length and period_length are kept in daily_data for audit and "
        "label/target construction, but should not be used as direct model input features."
    ),
}


# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------

def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV dataset from *path* and return a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    df = pd.read_csv(path)
    print(f"[load_raw_data] Loaded {df.shape[0]} rows x {df.shape[1]} columns from '{path}'.")
    return df


# ---------------------------------------------------------------------------
# 2. Normalise column names
# ---------------------------------------------------------------------------

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all column names to lowercase snake_case, stripping special chars."""
    def _normalize(name: str) -> str:
        name = name.strip().lower()
        name = re.sub(r"[^\w\s]", "", name)   # remove special chars
        name = re.sub(r"\s+", "_", name)       # spaces -> underscore
        name = re.sub(r"_+", "_", name)        # collapse multiple underscores
        return name

    df = df.copy()
    df.columns = [_normalize(c) for c in df.columns]
    print(f"[normalize_column_names] Normalised columns: {df.columns.tolist()}")

    # Verify all required columns are present
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing after normalisation: {missing}")

    return df


# ---------------------------------------------------------------------------
# 3. Parse date columns
# ---------------------------------------------------------------------------

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse cycle_start_date and next_cycle_start_date to datetime (date only)."""
    df = df.copy()
    for col in ("cycle_start_date", "next_cycle_start_date"):
        before = df[col].copy()
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
        n_failed = df[col].isna().sum() - before.isna().sum()
        if n_failed > 0:
            warnings.warn(
                f"[parse_dates] {n_failed} value(s) in '{col}' could not be parsed "
                f"and were set to NaT.",
                UserWarning,
                stacklevel=2,
            )
            print(f"  WARNING: {n_failed} unparseable date(s) in '{col}' -> NaT.")
        else:
            print(f"[parse_dates] '{col}' parsed successfully. "
                  f"Range: {df[col].min().date()} to {df[col].max().date()}")

    # Drop rows with NaT dates (cannot be used)
    n_before = len(df)
    df = df.dropna(subset=["cycle_start_date", "next_cycle_start_date"])
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[parse_dates] Dropped {n_dropped} rows with unparseable dates.")

    return df


# ---------------------------------------------------------------------------
# 4. Chronological-order check (pre-sort)
# ---------------------------------------------------------------------------

def check_chronological_order(df: pd.DataFrame) -> int:
    """
    Check per-user monotonic ordering of cycle_start_date.
    Returns the number of users whose cycles are NOT in ascending order.
    """
    def _is_monotonic(series: pd.Series) -> bool:
        return series.is_monotonic_increasing

    not_sorted_users = (
        df.groupby("user_id")["cycle_start_date"]
        .apply(_is_monotonic)
        .pipe(lambda s: (~s).sum())
    )
    print(f"[check_chronological_order] Users with non-chronological cycles: {not_sorted_users}")
    return int(not_sorted_users)


# ---------------------------------------------------------------------------
# 5. Sort by user and date
# ---------------------------------------------------------------------------

def sort_by_user_and_date(df: pd.DataFrame) -> pd.DataFrame:
    """Sort DataFrame by (user_id, cycle_start_date) ascending."""
    df = df.sort_values(["user_id", "cycle_start_date"], ascending=True).reset_index(drop=True)
    print("[sort_by_user_and_date] DataFrame sorted by user_id + cycle_start_date.")

    # Post-sort monotonic check -- must pass after sorting
    still_bad = check_chronological_order(df)
    if still_bad > 0:
        raise ValueError(
            f"[sort_by_user_and_date] {still_bad} user(s) still have non-monotonic "
            "cycle_start_date after sorting. Investigate duplicate or malformed dates."
        )
    return df


# ---------------------------------------------------------------------------
# 6. Duplicate cycle check
# ---------------------------------------------------------------------------

def check_duplicate_cycles(df: pd.DataFrame) -> int:
    """
    Check for duplicate (user_id, cycle_start_date) pairs.
    Prints a warning if any are found. Returns duplicate count.
    """
    dup_mask = df.duplicated(subset=["user_id", "cycle_start_date"], keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        warnings.warn(
            f"[check_duplicate_cycles] {n_dup} rows share the same "
            "(user_id, cycle_start_date). These duplicates may bias historical "
            "aggregation. Consider deduplication.",
            UserWarning,
            stacklevel=2,
        )
        print(f"  WARNING: {n_dup} duplicate (user_id, cycle_start_date) rows found.")
    else:
        print("[check_duplicate_cycles] No duplicate (user_id, cycle_start_date) found.")
    return n_dup


# ---------------------------------------------------------------------------
# 7. Historical aggregation features
# ---------------------------------------------------------------------------

def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Expanding mean shifted by 1 -- never includes the current row
    df["hist_mean_cycle"] = (
        df.groupby("user_id")["cycle_length"]
        .transform(lambda s: s.expanding().mean().shift(1))
    )
    df["hist_mean_period"] = (
        df.groupby("user_id")["period_length"]
        .transform(lambda s: s.expanding().mean().shift(1))
    )

    # Flag rows that will receive cold-start imputation
    hist_missing_mask = df["hist_mean_cycle"].isna() | df["hist_mean_period"].isna()
    df["is_historical_data_missing"] = hist_missing_mask.astype(int)

    # Fill NaN with domain priors (float -- no rounding)
    df["hist_mean_cycle"] = df["hist_mean_cycle"].fillna(COLD_START_PRIORS["hist_mean_cycle"])
    df["hist_mean_period"] = df["hist_mean_period"].fillna(COLD_START_PRIORS["hist_mean_period"])

    # Ensure float dtype
    df["hist_mean_cycle"] = df["hist_mean_cycle"].astype(float)
    df["hist_mean_period"] = df["hist_mean_period"].astype(float)
    
    # Keep historical averages readable while preserving continuous values
    df["hist_mean_cycle"] = df["hist_mean_cycle"].round(2)
    df["hist_mean_period"] = df["hist_mean_period"].round(2)

    print("[add_historical_features] hist_mean_cycle and hist_mean_period added.")
    print(f"  Rows with cold-start prior (is_historical_data_missing=1): "
          f"{df['is_historical_data_missing'].sum()}")

    # --- Assertions ---
    assert df["hist_mean_cycle"].isna().sum() == 0, "hist_mean_cycle still contains NaN!"
    assert df["hist_mean_period"].isna().sum() == 0, "hist_mean_period still contains NaN!"

    # Each user's FIRST cycle must have is_historical_data_missing == 1
    first_cycle_flags = df.groupby("user_id")["is_historical_data_missing"].first()
    assert (first_cycle_flags == 1).all(), (
        "Some users have is_historical_data_missing=0 for their first cycle!"
    )

    # Subsequent cycles (not first) must have is_historical_data_missing == 0
    # (Only check users that have >1 cycle)
    multi_cycle_users = df.groupby("user_id").filter(lambda g: len(g) > 1)
    if not multi_cycle_users.empty:
        # For each such user, drop the first row, then check flags
        subsequent_flags = (
            multi_cycle_users
            .groupby("user_id")["is_historical_data_missing"]
            .apply(lambda s: s.iloc[1:])
            .reset_index(drop=True)
        )
        assert (subsequent_flags == 0).all(), (
            "Some subsequent cycles (non-first) have is_historical_data_missing=1!"
        )

    return df


# ---------------------------------------------------------------------------
# 8. Phase assignment (per row)
# ---------------------------------------------------------------------------

def assign_phase(day_in_cycle: int, cycle_length: int, period_length: int) -> tuple[str, int]:

    ovulation_day = cycle_length - 14

    if day_in_cycle < period_length:
        return "menstruation", PHASE_MAP["menstruation"]
    elif day_in_cycle < ovulation_day:
        return "follicular", PHASE_MAP["follicular"]
    elif day_in_cycle == ovulation_day:
        return "ovulation", PHASE_MAP["ovulation"]
    else:
        return "luteal", PHASE_MAP["luteal"]


# ---------------------------------------------------------------------------
# 9. Phase rule validation (cycle-level)
# ---------------------------------------------------------------------------

def validate_phase_rule(df: pd.DataFrame) -> int:

    ovulation_day = df["cycle_length"] - 14
    viol_lower = (ovulation_day <= df["period_length"]).sum()
    viol_upper = (ovulation_day >= df["cycle_length"]).sum()
    total_violations = int(viol_lower + viol_upper)

    if viol_lower:
        print(f"  WARNING [validate_phase_rule] {viol_lower} cycle(s) where "
              f"ovulation_day <= period_length  (phase boundary overlap).")
    if viol_upper:
        print(f"  WARNING [validate_phase_rule] {viol_upper} cycle(s) where "
              f"ovulation_day >= cycle_length  (ovulation outside cycle).")
    if total_violations == 0:
        print("[validate_phase_rule] No phase rule violations detected.")

    return total_violations


# ---------------------------------------------------------------------------
# 10. Expand to daily rows
# ---------------------------------------------------------------------------

def expand_to_daily_rows(df: pd.DataFrame) -> pd.DataFrame:

    daily_rows = []

    for _, row in df.iterrows():
        cycle_len = int(row["cycle_length"])
        period_len = int(row["period_length"])
        start_date = row["cycle_start_date"]
        next_start = row["next_cycle_start_date"]

        for day in range(cycle_len):
            sample_date = start_date + pd.Timedelta(days=day)
            days_until = (next_start - sample_date).days
            phase_name, phase_label = assign_phase(day, cycle_len, period_len)

            daily_rows.append({
                **row.to_dict(),
                "day_in_cycle": day,
                "sample_date": sample_date,
                "days_until_next_cycle": days_until,
                "phase_name": phase_name,
                "phase_label": phase_label,
            })

    daily_df = pd.DataFrame(daily_rows)
    print(f"[expand_to_daily_rows] Expanded to {len(daily_df)} daily rows.")
    return daily_df


# ---------------------------------------------------------------------------
# 11. Validate daily expansion
# ---------------------------------------------------------------------------

def validate_daily_expansion(raw_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:

    expected_rows = int(raw_df["cycle_length"].sum())
    actual_rows = len(daily_df)
    assert actual_rows == expected_rows, (
        f"Row count mismatch: expected {expected_rows}, got {actual_rows}."
    )
    print(f"[validate_daily_expansion] Row count check passed: {actual_rows} rows.")

    assert (daily_df["days_until_next_cycle"] >= 0).all(), (
        "days_until_next_cycle contains negative values!"
    )
    print("[validate_daily_expansion] days_until_next_cycle >= 0 -- OK.")

    valid_labels = {0, 1, 2, 3}
    assert set(daily_df["phase_label"].unique()).issubset(valid_labels), (
        f"phase_label contains unexpected values: {daily_df['phase_label'].unique()}"
    )
    print("[validate_daily_expansion] phase_label values {0,1,2,3} -- OK.")

    valid_names = {"menstruation", "follicular", "ovulation", "luteal"}
    assert set(daily_df["phase_name"].unique()).issubset(valid_names), (
        f"phase_name contains unexpected values: {daily_df['phase_name'].unique()}"
    )
    print("[validate_daily_expansion] phase_name values -- OK.")

    assert daily_df["hist_mean_cycle"].isna().sum() == 0, "hist_mean_cycle has NaN in daily_df!"
    assert daily_df["hist_mean_period"].isna().sum() == 0, "hist_mean_period has NaN in daily_df!"
    print("[validate_daily_expansion] hist_mean_cycle / hist_mean_period NaN check -- OK.")

    # Per-group checks
    group_errors = []
    for (uid, csd), grp in daily_df.groupby(["user_id", "cycle_start_date"]):
        expected_len = grp["cycle_length"].iloc[0]
        expected_days = list(range(int(expected_len)))
        actual_days = sorted(grp["day_in_cycle"].tolist())
        if actual_days != expected_days:
            group_errors.append(
                f"user_id={uid}, cycle_start_date={csd}: "
                f"day_in_cycle mismatch (expected 0..{expected_len-1})."
            )
        if not grp["sample_date"].is_monotonic_increasing:
            group_errors.append(
                f"user_id={uid}, cycle_start_date={csd}: "
                f"sample_date is not monotonically increasing."
            )

    if group_errors:
        for e in group_errors:
            print(f"  ERROR: {e}")
        raise AssertionError(
            f"{len(group_errors)} per-group validation errors. See above."
        )
    print("[validate_daily_expansion] Per-group day_in_cycle and sample_date checks -- OK.")


# ---------------------------------------------------------------------------
# 12. Basic input assertions (cycle-level)
# ---------------------------------------------------------------------------

def _assert_cycle_level_integrity(df: pd.DataFrame) -> None:
    """Run basic sanity checks on the raw (but normalised) cycle-level DataFrame."""
    assert (df["cycle_length"] > 0).all(), "cycle_length must be positive."
    assert (df["period_length"] > 0).all(), "period_length must be positive."
    assert (df["period_length"] < df["cycle_length"]).all(), (
        "period_length must be less than cycle_length."
    )
    assert (df["next_cycle_start_date"] > df["cycle_start_date"]).all(), (
        "next_cycle_start_date must be after cycle_start_date."
    )
    print("[_assert_cycle_level_integrity] All cycle-level integrity checks passed.")


# ---------------------------------------------------------------------------
# 13. Categorical encoding
# ---------------------------------------------------------------------------

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # -- 0. Presence check --
    for col in ("diet", "symptoms", "exercise_frequency"):
        if col not in df.columns:
            raise ValueError(f"[encode_categorical_features] Required column '{col}' not found.")

    # ------------------------------------------------------------------
    # A. diet -- one-hot
    # ------------------------------------------------------------------
    diet_vals = df["diet"].unique().tolist()
    unexpected_diet = [v for v in diet_vals if v not in DIET_CATEGORIES]
    if unexpected_diet:
        warnings.warn(
            f"[encode_categorical_features] Unexpected diet value(s): {unexpected_diet}. "
            "These rows will produce all-zero diet one-hot columns.",
            UserWarning,
            stacklevel=2,
        )
        print(f"  WARNING: Unexpected diet value(s) found: {unexpected_diet}")

    # Check for multi-label patterns (comma, semicolon, pipe)
    multi_label_diet = df["diet"].str.contains(r"[,;|]", na=False).sum()
    if multi_label_diet:
        warnings.warn(
            f"[encode_categorical_features] {multi_label_diet} diet value(s) look like "
            "multi-label entries (contain ',', ';', or '|'). "
            "One-hot encoding is applied as-is; consider multi-hot in a future step.",
            UserWarning,
            stacklevel=2,
        )
        print(f"  WARNING: {multi_label_diet} possible multi-label diet value(s) detected.")

    for cat, col_name in zip(DIET_CATEGORIES, DIET_OHE_COLS):
        df[col_name] = (df["diet"] == cat).astype(int)

    # ------------------------------------------------------------------
    # B. symptoms -- one-hot
    # ------------------------------------------------------------------
    symptom_vals = df["symptoms"].unique().tolist()
    unexpected_sym = [v for v in symptom_vals if v not in SYMPTOM_CATEGORIES]
    if unexpected_sym:
        warnings.warn(
            f"[encode_categorical_features] Unexpected symptom value(s): {unexpected_sym}. "
            "These rows will produce all-zero symptom one-hot columns.",
            UserWarning,
            stacklevel=2,
        )
        print(f"  WARNING: Unexpected symptom value(s) found: {unexpected_sym}")

    # Check for multi-label patterns
    multi_label_sym = df["symptoms"].str.contains(r"[,;|]", na=False).sum()
    if multi_label_sym:
        warnings.warn(
            f"[encode_categorical_features] {multi_label_sym} symptom value(s) look like "
            "multi-label entries (contain ',', ';', or '|'). "
            "One-hot encoding is applied as-is; consider multi-hot in a future step.",
            UserWarning,
            stacklevel=2,
        )
        print(f"  WARNING: {multi_label_sym} possible multi-label symptom value(s) detected.")

    for cat, col_name in zip(SYMPTOM_CATEGORIES, SYMPTOM_OHE_COLS):
        df[col_name] = (df["symptoms"] == cat).astype(int)

    # ------------------------------------------------------------------
    # C. exercise_frequency -- ordinal
    # ------------------------------------------------------------------
    ex_vals = df["exercise_frequency"].unique().tolist()
    unexpected_ex = [v for v in ex_vals if v not in EXERCISE_ORDINAL_MAP]
    if unexpected_ex:
        warnings.warn(
            f"[encode_categorical_features] Unexpected exercise_frequency value(s): "
            f"{unexpected_ex}. These rows will be mapped to NaN.",
            UserWarning,
            stacklevel=2,
        )
        print(f"  WARNING: Unexpected exercise_frequency value(s): {unexpected_ex}")

    df["exercise_frequency_encoded"] = df["exercise_frequency"].map(EXERCISE_ORDINAL_MAP)

    # ------------------------------------------------------------------
    # D. Post-encoding assertions
    # ------------------------------------------------------------------

    # (1) All diet one-hot cols must exist
    for col in DIET_OHE_COLS:
        assert col in df.columns, f"Missing diet one-hot column: {col}"

    # (2) All symptom one-hot cols must exist
    for col in SYMPTOM_OHE_COLS:
        assert col in df.columns, f"Missing symptom one-hot column: {col}"

    # (3) exercise_frequency_encoded must exist and have no NaN
    assert "exercise_frequency_encoded" in df.columns, \
        "exercise_frequency_encoded column not created."
    n_ex_nan = df["exercise_frequency_encoded"].isna().sum()
    assert n_ex_nan == 0, (
        f"exercise_frequency_encoded contains {n_ex_nan} NaN value(s). "
        "Check for unexpected exercise_frequency categories."
    )

    # (4) exercise_frequency_encoded must only contain {0, 1, 2}
    ex_unique = set(df["exercise_frequency_encoded"].unique())
    assert ex_unique.issubset({0, 1, 2}), (
        f"exercise_frequency_encoded contains unexpected values: {ex_unique}"
    )

    # (5) diet one-hot row-sum must equal 1 (single-label)
    diet_row_sum = df[DIET_OHE_COLS].sum(axis=1)
    diet_violations = int((diet_row_sum != 1).sum())
    if diet_violations:
        warnings.warn(
            f"[encode_categorical_features] {diet_violations} row(s) have diet "
            "one-hot sum != 1 (missing or unexpected category).",
            UserWarning,
            stacklevel=2,
        )
        print(f"  WARNING: {diet_violations} row(s) with diet one-hot row-sum != 1.")
    else:
        print("[encode_categorical_features] diet one-hot row-sum = 1 for all rows -- OK.")

    # (6) symptom one-hot row-sum must equal 1 (single-label)
    sym_row_sum = df[SYMPTOM_OHE_COLS].sum(axis=1)
    sym_violations = int((sym_row_sum != 1).sum())
    if sym_violations:
        warnings.warn(
            f"[encode_categorical_features] {sym_violations} row(s) have symptom "
            "one-hot sum != 1 (missing or unexpected category).",
            UserWarning,
            stacklevel=2,
        )
        print(f"  WARNING: {sym_violations} row(s) with symptom one-hot row-sum != 1.")
    else:
        print("[encode_categorical_features] symptom one-hot row-sum = 1 for all rows -- OK.")

    # (7) Leakage guard: input_features must not contain forbidden columns
    input_features = set(FEATURE_INFO["input_features"])
    leakage_found = input_features & _LEAKAGE_COLS
    assert not leakage_found, (
        f"Leakage detected: these columns are in input_features but should be excluded: "
        f"{leakage_found}"
    )

    print("[encode_categorical_features] All encoding assertions passed.")
    return df, diet_violations, sym_violations


# ---------------------------------------------------------------------------
# 14. Save outputs
# ---------------------------------------------------------------------------

def save_outputs(daily_df: pd.DataFrame, processed_dir: str) -> tuple[str, str]:
    """
    Save daily_data.csv and feature_info.json to *processed_dir*.
    Returns (csv_path, json_path).
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Keep only the defined columns (in order)
    cols_present = [c for c in DAILY_COLUMNS_ORDERED if c in daily_df.columns]
    daily_df = daily_df[cols_present]

    csv_path = processed_dir / "daily_data.csv"
    daily_df.to_csv(csv_path, index=False)
    print(f"[save_outputs] daily_data.csv saved -> {csv_path}")

    json_path = processed_dir / "feature_info.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_INFO, f, indent=2)
    print(f"[save_outputs] feature_info.json saved -> {json_path}")

    return str(csv_path), str(json_path)


# ---------------------------------------------------------------------------
# 15. Summary print
# ---------------------------------------------------------------------------

def _print_summary(
    raw_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    n_unsorted_users: int,
    n_duplicates: int,
    n_phase_violations: int,
    diet_ohe_violations: int,
    sym_ohe_violations: int,
    csv_path: str,
    json_path: str,
) -> None:
    """Print a comprehensive summary of the data preparation run."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  DAILY DATASET BUILDER SUMMARY")
    print(sep)
    print(f"  Raw DataFrame shape           : {raw_df.shape}")
    print(f"  Daily DataFrame shape         : {daily_df.shape}")
    print(f"  Total unique users            : {raw_df['user_id'].nunique()}")
    print(f"  Total cycle rows (raw)        : {len(raw_df)}")
    print(f"  cycle_length  min / max       : {raw_df['cycle_length'].min()} / {raw_df['cycle_length'].max()}")
    print(f"  period_length min / max       : {raw_df['period_length'].min()} / {raw_df['period_length'].max()}")
    print(f"  Non-chronological users       : {n_unsorted_users}")
    print(f"  Duplicate (uid, start) rows   : {n_duplicates}")
    print(f"  Phase rule violations         : {n_phase_violations}")
    print()
    print("  phase_label distribution:")
    lbl_dist = daily_df["phase_label"].value_counts().sort_index()
    for lbl, cnt in lbl_dist.items():
        print(f"    {lbl} : {cnt}")
    print()
    print("  phase_name distribution:")
    name_dist = daily_df["phase_name"].value_counts().sort_index()
    for name, cnt in name_dist.items():
        print(f"    {name} : {cnt}")
    print()
    print("  is_historical_data_missing distribution:")
    flag_dist = daily_df["is_historical_data_missing"].value_counts().sort_index()
    for flag, cnt in flag_dist.items():
        print(f"    {flag} : {cnt}")
    print()
    # --- Encoding summary ---
    print("  exercise_frequency_encoded distribution:")
    ex_dist = daily_df["exercise_frequency_encoded"].value_counts().sort_index()
    label_map = {0: "Low", 1: "Moderate", 2: "High"}
    for val, cnt in ex_dist.items():
        print(f"    {val} ({label_map.get(int(val), '?')}) : {cnt}")
    print()
    print("  diet one-hot column totals:")
    for col in DIET_OHE_COLS:
        print(f"    {col:<30}: {int(daily_df[col].sum())}")
    print(f"  diet one-hot row-sum violations : {diet_ohe_violations}")
    print()
    print("  symptoms one-hot column totals:")
    for col in SYMPTOM_OHE_COLS:
        print(f"    {col:<30}: {int(daily_df[col].sum())}")
    print(f"  symptom one-hot row-sum violations : {sym_ohe_violations}")
    print()
    print(f"  daily_data.csv saved to       : {csv_path}")
    print(f"  feature_info.json saved to    : {json_path}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# 16. Main pipeline
# ---------------------------------------------------------------------------

def run_daily_dataset_builder(raw_path: str, processed_dir: str) -> None:
    """
    End-to-end pipeline: raw cycle-level CSV -> daily-level dataset with
    historical features, phase labels, targets, and categorical encoding.

    Parameters
    ----------
    raw_path : str
        Path to the raw CSV file.
    processed_dir : str
        Directory where processed outputs will be saved.
    """
    print("\n" + "=" * 60)
    print("  STARTING DAILY DATASET BUILDER PIPELINE")
    print("=" * 60 + "\n")

    # --- Step 1: Load ---
    raw_df = load_raw_data(raw_path)

    # --- Step 2: Normalise column names ---
    df = normalize_column_names(raw_df)

    # --- Step 3: Parse dates ---
    df = parse_dates(df)

    # --- Step 4: Chronological order check (pre-sort) ---
    n_unsorted_users = check_chronological_order(df)

    # --- Step 5: Sort by user + date ---
    df = sort_by_user_and_date(df)

    # --- Step 6: Duplicate cycle check ---
    n_duplicates = check_duplicate_cycles(df)

    # --- Cycle-level integrity assertions ---
    _assert_cycle_level_integrity(df)

    # --- Step 7: Historical aggregation features ---
    # (Must be called on sorted df to avoid leakage)
    df = add_historical_features(df)

    # --- Step 8: Phase rule validation ---
    n_phase_violations = validate_phase_rule(df)

    # --- Step 9: Expand to daily rows ---
    daily_df = expand_to_daily_rows(df)

    # --- Step 10: Validate daily expansion ---
    validate_daily_expansion(df, daily_df)

    # --- Step 11: Encoding ---
    # Categorical encoding after daily expansion:
    # diet / symptoms categorical values are cycle-constant and already
    # replicated across daily rows, so encoding here is safe and correct.
    daily_df, diet_ohe_violations, sym_ohe_violations = encode_categorical_features(daily_df)

    # --- Step 12: Save outputs ---
    csv_path, json_path = save_outputs(daily_df, processed_dir)

    # --- Step 13: Print summary ---
    _print_summary(
        raw_df=df,
        daily_df=daily_df,
        n_unsorted_users=n_unsorted_users,
        n_duplicates=n_duplicates,
        n_phase_violations=n_phase_violations,
        diet_ohe_violations=diet_ohe_violations,
        sym_ohe_violations=sym_ohe_violations,
        csv_path=csv_path,
        json_path=json_path,
    )

    print("DAILY DATASET BUILD COMPLETE.\n")
