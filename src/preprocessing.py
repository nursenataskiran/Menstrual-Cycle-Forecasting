# -*- coding: utf-8 -*-
"""
src/preprocessing.py
====================
Step 2 -- User-Level Split and MinMax Normalization

What this module does:
  - Loads daily_data.csv produced by daily_dataset_builder
  - Splits data at the user level (no user appears in more than one split)
  - Fits a MinMaxScaler on the training split only
  - Transforms train / validation / test splits with that scaler
  - Reports out-of-range values in val/test (no clipping applied)
  - Saves split CSVs, the fitted scaler, split_users.json,
    preprocessing_info.json and updates feature_info.json

What this module does NOT do (reserved for later steps):
  - Sliding window / sequence generation
  - Target scaling
  - Model training
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns to scale with MinMaxScaler
SCALE_COLUMNS = [
    "day_in_cycle",
    "age",
    "bmi",
    "stress_level",
    "sleep_hours",
    "hist_mean_cycle",
    "hist_mean_period",
    "exercise_frequency_encoded",
]

# Columns that must NOT be scaled
NOT_SCALED_COLUMNS = [
    "is_historical_data_missing",
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

# Target columns (never scaled)
TARGET_COLUMNS = ["days_until_next_cycle", "phase_label"]

# One-hot + binary columns (must stay in {0, 1})
BINARY_COLS = NOT_SCALED_COLUMNS + ["is_historical_data_missing"]

# preprocessing_info template
PREPROCESSING_INFO = {
    "split_strategy": "user_level_split",
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_state": 42,
    "scaler": {
        "type": "MinMaxScaler",
        "feature_range": [0, 1],
        "fit_on": "train_df_only",
        "clip": False,
    },
    "scaled_columns": SCALE_COLUMNS,
    "not_scaled_columns": NOT_SCALED_COLUMNS,
    "targets_not_scaled": TARGET_COLUMNS,
    "leakage_note": (
        "Scaler was fitted only on the training users and then applied "
        "to validation and test users."
    ),
}


# ---------------------------------------------------------------------------
# 1. Load daily data
# ---------------------------------------------------------------------------

def load_daily_data(daily_data_path: str) -> pd.DataFrame:
    """Load the preprocessed daily_data.csv."""
    path = Path(daily_data_path)
    if not path.exists():
        raise FileNotFoundError(f"[load_daily_data] File not found: {path}")
    df = pd.read_csv(path, parse_dates=["cycle_start_date", "next_cycle_start_date", "sample_date"])
    print(f"[load_daily_data] Loaded {len(df)} rows x {len(df.columns)} cols from '{path}'.")
    return df


# ---------------------------------------------------------------------------
# 2. Load feature_info
# ---------------------------------------------------------------------------

def load_feature_info(feature_info_path: str) -> dict:
    """Load feature_info.json and return as dict."""
    path = Path(feature_info_path)
    if not path.exists():
        raise FileNotFoundError(f"[load_feature_info] File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        info = json.load(f)
    print(f"[load_feature_info] Loaded feature_info from '{path}'.")
    return info


# ---------------------------------------------------------------------------
# 3. User-level split
# ---------------------------------------------------------------------------

def user_level_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:

    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio:.4f}."
        )

    unique_users = sorted(df["user_id"].unique().tolist())
    n_users = len(unique_users)

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(unique_users).tolist()

    n_train = round(n_users * train_ratio)
    n_val   = round(n_users * val_ratio)
    # test gets the remainder to avoid rounding issues
    n_test  = n_users - n_train - n_val

    train_users = shuffled[:n_train]
    val_users   = shuffled[n_train : n_train + n_val]
    test_users  = shuffled[n_train + n_val :]

    print(f"[user_level_split] {n_users} total users -> "
          f"train={len(train_users)}, val={len(val_users)}, test={len(test_users)}")

    # --- Overlap assertions ---
    assert set(train_users).isdisjoint(val_users),  "train/val user overlap detected!"
    assert set(train_users).isdisjoint(test_users), "train/test user overlap detected!"
    assert set(val_users).isdisjoint(test_users),   "val/test user overlap detected!"
    assert len(train_users) + len(val_users) + len(test_users) == n_users, \
        "User count mismatch after split!"
    print("[user_level_split] No user overlap detected -- OK.")

    train_df = df[df["user_id"].isin(train_users)].copy()
    val_df   = df[df["user_id"].isin(val_users)].copy()
    test_df  = df[df["user_id"].isin(test_users)].copy()

    # --- DataFrame-level overlap assertions ---
    assert set(train_df["user_id"].unique()).isdisjoint(val_df["user_id"].unique()), \
        "DataFrame train/val user overlap!"
    assert set(train_df["user_id"].unique()).isdisjoint(test_df["user_id"].unique()), \
        "DataFrame train/test user overlap!"
    assert set(val_df["user_id"].unique()).isdisjoint(test_df["user_id"].unique()), \
        "DataFrame val/test user overlap!"

    # --- All users accounted for ---
    all_assigned = set(train_df["user_id"].unique()) | set(val_df["user_id"].unique()) | \
                   set(test_df["user_id"].unique())
    assert all_assigned == set(unique_users), "Some users were not assigned to any split!"
    print("[user_level_split] All users assigned to exactly one split -- OK.")

    split_users = {
        "train_users": sorted(train_users),
        "val_users":   sorted(val_users),
        "test_users":  sorted(test_users),
        "train_ratio": train_ratio,
        "val_ratio":   val_ratio,
        "test_ratio":  test_ratio,
        "random_state": random_state,
    }

    return train_df, val_df, test_df, split_users


# ---------------------------------------------------------------------------
# 4. Normalize features
# ---------------------------------------------------------------------------

def normalize_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scale_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:

    # Verify columns exist in all splits
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = [c for c in scale_columns if c not in split_df.columns]
        if missing:
            raise ValueError(f"[normalize_features] Columns missing in {split_name}_df: {missing}")

    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit ONLY on train
    scaler.fit(train_df[scale_columns])
    print("[normalize_features] MinMaxScaler fitted on train_df only.")

    # Transform all three splits
    train_df[scale_columns] = scaler.transform(train_df[scale_columns])
    val_df[scale_columns]   = scaler.transform(val_df[scale_columns])
    test_df[scale_columns]  = scaler.transform(test_df[scale_columns])

    print(f"[normalize_features] Scaled {len(scale_columns)} columns across train/val/test.")
    return train_df, val_df, test_df, scaler


# ---------------------------------------------------------------------------
# 5. Check scaled ranges
# ---------------------------------------------------------------------------

def check_scaled_ranges(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scale_columns: list[str],
) -> dict:

    tol = 1e-9  # floating-point tolerance for train check
    summary = {"train_oor": 0, "val_oor": 0, "test_oor": 0}

    # --- Train: must be [0, 1] ---
    train_oor = int(
        ((train_df[scale_columns] < -tol) | (train_df[scale_columns] > 1 + tol)).sum().sum()
    )
    if train_oor:
        warnings.warn(
            f"[check_scaled_ranges] Train set has {train_oor} out-of-range scaled value(s). "
            "This is unexpected since scaler was fit on train.",
            UserWarning, stacklevel=2,
        )
        print(f"  WARNING: Train out-of-range scaled values: {train_oor}")
    else:
        print("[check_scaled_ranges] Train scaled values all in [0, 1] -- OK.")
    summary["train_oor"] = train_oor

    # --- Val / Test: report out-of-range (expected for unseen users) ---
    for split_name, split_df, key in [("val", val_df, "val_oor"), ("test", test_df, "test_oor")]:
        oor = int(
            ((split_df[scale_columns] < 0) | (split_df[scale_columns] > 1)).sum().sum()
        )
        if oor:
            print(f"[check_scaled_ranges] {split_name} set: {oor} out-of-range scaled value(s) "
                  f"(unseen user range -- no clipping applied).")
        else:
            print(f"[check_scaled_ranges] {split_name} scaled values all in [0, 1] -- OK.")
        summary[key] = oor

    # --- NaN check ---
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        nan_count = split_df[scale_columns].isna().sum().sum()
        assert nan_count == 0, (
            f"[check_scaled_ranges] {split_name}_df has {nan_count} NaN(s) in scaled columns!"
        )
    print("[check_scaled_ranges] No NaN in scaled columns -- OK.")

    # --- Targets unchanged (check NaN only; values are unbounded) ---
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for tgt in TARGET_COLUMNS:
            if tgt in split_df.columns:
                assert split_df[tgt].isna().sum() == 0, (
                    f"[check_scaled_ranges] Target '{tgt}' has NaN in {split_name}_df!"
                )
    print("[check_scaled_ranges] Target columns NaN check -- OK.")

    # --- Binary / one-hot columns: must remain in {0, 1} ---
    binary_check_cols = [c for c in BINARY_COLS if c in train_df.columns]
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        bad = split_df[binary_check_cols].apply(
            lambda col: ~col.isin([0, 1])
        ).sum().sum()
        assert bad == 0, (
            f"[check_scaled_ranges] {split_name}_df has {bad} non-binary value(s) "
            "in binary/one-hot columns!"
        )
    print("[check_scaled_ranges] Binary/one-hot columns remain in {0, 1} -- OK.")

    return summary


# ---------------------------------------------------------------------------
# 6. Save preprocessing outputs
# ---------------------------------------------------------------------------

def save_preprocessing_outputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_users: dict,
    scaler: MinMaxScaler,
    preprocessing_info: dict,
    processed_dir: str,
) -> dict:
    """
    Save all preprocessing artifacts to *processed_dir*.

    Saved files
    -----------
    train_daily.csv
    val_daily.csv
    test_daily.csv
    split_users.json
    scaler.pkl
    preprocessing_info.json

    Returns a dict with all file paths.
    """
    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Split CSVs
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        p = out / f"{name}_daily.csv"
        split_df.to_csv(p, index=False)
        print(f"[save_preprocessing_outputs] {name}_daily.csv saved -> {p}")
        paths[f"{name}_csv"] = str(p)

    # split_users.json
    split_path = out / "split_users.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_users, f, indent=2)
    print(f"[save_preprocessing_outputs] split_users.json saved -> {split_path}")
    paths["split_users_json"] = str(split_path)

    # scaler.pkl
    scaler_path = out / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"[save_preprocessing_outputs] scaler.pkl saved -> {scaler_path}")
    paths["scaler_pkl"] = str(scaler_path)

    # preprocessing_info.json
    info_path = out / "preprocessing_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(preprocessing_info, f, indent=2)
    print(f"[save_preprocessing_outputs] preprocessing_info.json saved -> {info_path}")
    paths["preprocessing_info_json"] = str(info_path)

    return paths


# ---------------------------------------------------------------------------
# 7. Update feature_info.json
# ---------------------------------------------------------------------------

def update_feature_info(
    feature_info_path: str,
    scaling_info: dict,
    split_info: dict,
) -> None:

    path = Path(feature_info_path)
    with open(path, "r", encoding="utf-8") as f:
        info = json.load(f)

    info["not_scaled_yet"] = False
    info["not_split_yet"]  = False
    info["scaling"] = scaling_info
    info["split"]   = split_info

    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"[update_feature_info] feature_info.json updated -> {path}")


# ---------------------------------------------------------------------------
# 8. Summary print
# ---------------------------------------------------------------------------

def _print_summary(
    daily_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scale_columns: list[str],
    range_summary: dict,
    paths: dict,
) -> None:
    """Print a comprehensive summary of the preprocessing run."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  PREPROCESSING SUMMARY")
    print(sep)

    # --- Shape ---
    print(f"  daily_data shape    : {daily_df.shape}")
    print(f"  train_daily shape   : {train_df.shape}")
    print(f"  val_daily shape     : {val_df.shape}")
    print(f"  test_daily shape    : {test_df.shape}")
    print()

    # --- User counts ---
    print(f"  Train unique users  : {train_df['user_id'].nunique()}")
    print(f"  Val unique users    : {val_df['user_id'].nunique()}")
    print(f"  Test unique users   : {test_df['user_id'].nunique()}")
    overlap = (
        set(train_df["user_id"].unique()) &
        set(val_df["user_id"].unique()) |
        set(train_df["user_id"].unique()) &
        set(test_df["user_id"].unique()) |
        set(val_df["user_id"].unique()) &
        set(test_df["user_id"].unique())
    )
    print(f"  User overlap        : {len(overlap)} (expected 0)")
    print()

    # --- Scaled columns ---
    print(f"  Scaled columns ({len(scale_columns)}):")
    for col in scale_columns:
        t_min = train_df[col].min()
        t_max = train_df[col].max()
        print(f"    {col:<35} train [{t_min:.4f}, {t_max:.4f}]")
    print()

    # --- Out-of-range ---
    print(f"  Train OOR scaled values  : {range_summary['train_oor']}")
    print(f"  Val   OOR scaled values  : {range_summary['val_oor']}")
    print(f"  Test  OOR scaled values  : {range_summary['test_oor']}")
    print()

    # --- Target summary ---
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        tgt = "days_until_next_cycle"
        print(f"  {split_name} {tgt}: "
              f"min={split_df[tgt].min():.1f}, max={split_df[tgt].max():.1f}")
    print()

    # --- phase_label distribution per split ---
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split_df["phase_label"].value_counts().sort_index()
        dist_str = ", ".join(f"{k}:{v}" for k, v in dist.items())
        print(f"  {split_name} phase_label : {dist_str}")
    print()

    # --- Saved paths ---
    print("  Saved files:")
    for key, p in paths.items():
        print(f"    {key:<30}: {p}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# 9. Main pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(
    daily_data_path: str,
    feature_info_path: str,
    processed_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> None:

    print("\n" + "=" * 60)
    print("  STARTING PREPROCESSING PIPELINE")
    print("=" * 60 + "\n")

    # --- Step 1: Load ---
    daily_df = load_daily_data(daily_data_path)
    feature_info = load_feature_info(feature_info_path)

    # --- Guard: daily dataset must already be encoded ---
    if feature_info.get("not_encoded_yet", True):
        raise RuntimeError(
            "[run_preprocessing] daily_data.csv does not appear to be encoded yet. "
            "Run run_daily_dataset_builder first."
        )

    # --- Step 2: User-level split ---
    train_df, val_df, test_df, split_users = user_level_split(
        daily_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    # Confirm scale_columns exist in data
    missing_scale_cols = [c for c in SCALE_COLUMNS if c not in daily_df.columns]
    if missing_scale_cols:
        raise ValueError(
            f"[run_preprocessing] SCALE_COLUMNS not found in daily_data: {missing_scale_cols}"
        )

    # --- Step 3: Normalize ---
    train_df, val_df, test_df, scaler = normalize_features(
        train_df, val_df, test_df, SCALE_COLUMNS
    )

    # --- Step 4: Range / integrity checks ---
    range_summary = check_scaled_ranges(train_df, val_df, test_df, SCALE_COLUMNS)

    # --- Step 5: Build metadata dicts ---
    scaling_info = {
        "method": "MinMaxScaler",
        "feature_range": [0, 1],
        "fit_on": "train_df_only",
        "scaled_columns": SCALE_COLUMNS,
        "targets_scaled": False,
        "binary_and_one_hot_scaled": False,
    }

    split_info = {
        "method": "user_level",
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_state": random_state,
    }

    preprocessing_info = {
        **PREPROCESSING_INFO,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_state": random_state,
    }

    # --- Step 6: Save ---
    paths = save_preprocessing_outputs(
        train_df, val_df, test_df,
        split_users, scaler, preprocessing_info,
        processed_dir,
    )

    # --- Step 7: Update feature_info.json ---
    update_feature_info(feature_info_path, scaling_info, split_info)

    # --- Step 8: Print summary ---
    _print_summary(
        daily_df=daily_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        scale_columns=SCALE_COLUMNS,
        range_summary=range_summary,
        paths=paths,
    )

    print("PREPROCESSING COMPLETE.\n")
