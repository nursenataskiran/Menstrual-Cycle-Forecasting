"""
sequence_builder.py
===================
Sliding-window sequence builder for the menstrual cycle multi-output LSTM.

Reads:
    data/processed/train_daily.csv
    data/processed/val_daily.csv
    data/processed/test_daily.csv
    data/processed/feature_info.json

Produces per split (example: seq_length=14):
    data/sequences/X_train_seq14.npy          shape: (n_windows, 14, n_features)
    data/sequences/y_phase_train_seq14.npy    shape: (n_windows, 14)
    data/sequences/y_reg_train_seq14.npy      shape: (n_windows, 14, 1)
    data/sequences/sequence_info_seq14.json

Design contract
---------------
* Windows are generated **within** each (user_id, cycle_start_date) group.
  A window never spans two different cycles.
* Both targets are sequence-to-sequence: every timestep in a window receives
  a phase_label and a days_until_next_cycle value.
* Scaling, splitting, and encoding are NOT performed here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

#: Columns that must NOT appear in input_features (leakage / identifier risk).
LEAKAGE_COLUMNS: List[str] = [
    "user_id",
    "cycle_start_date",
    "next_cycle_start_date",
    "sample_date",
    "cycle_length",
    "period_length",
    "exercise_frequency",
    "diet",
    "symptoms",
    "days_until_next_cycle",
    "phase_label",
    "phase_name",
]

#: Valid integer phase labels expected in the classification target.
VALID_PHASE_LABELS: set = {0, 1, 2, 3}


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────


def load_split_data(
    train_path: str,
    val_path: str,
    test_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    print("[sequence_builder] Loading split CSVs …")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(
        f"  train_daily shape : {train_df.shape}\n"
        f"  val_daily   shape : {val_df.shape}\n"
        f"  test_daily  shape : {test_df.shape}"
    )
    return train_df, val_df, test_df


def load_feature_info(feature_info_path: str) -> dict:

    print(f"[sequence_builder] Loading feature_info from: {feature_info_path}")
    with open(feature_info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    return info


# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────────────


def validate_input_features(
    df_list: List[pd.DataFrame],
    feature_cols: List[str],
) -> None:
    """Validate that feature_cols are safe to use as model inputs.

    Checks
    ------
    1. No leakage column appears in feature_cols.
    2. Every feature column exists in each DataFrame.

    Parameters
    ----------
    df_list:
        List of DataFrames (train, val, test) to check column presence against.
    feature_cols:
        The input_features list read from feature_info.json.

    Raises
    ------
    ValueError
        If any leakage column is found or if a required column is missing.
    """
    print("[sequence_builder] Validating input features …")

    # --- leakage check -------------------------------------------------------
    leakage_found = [col for col in feature_cols if col in LEAKAGE_COLUMNS]
    if leakage_found:
        raise ValueError(
            f"[sequence_builder] ERROR: The following leakage columns were found "
            f"in input_features and must be removed: {leakage_found}"
        )

    # --- column presence check -----------------------------------------------
    split_names = ["train", "val", "test"]
    for split_name, df in zip(split_names, df_list):
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"[sequence_builder] ERROR: The following input_features are "
                f"missing from {split_name}_daily.csv: {missing}"
            )

    print(f"  OK – {len(feature_cols)} input features passed all checks.")


def validate_sequence_arrays(
    X: np.ndarray,
    y_phase: np.ndarray,
    y_reg: np.ndarray,
    seq_length: int,
    split_name: str,
) -> None:

    print(f"[sequence_builder] Validating arrays for split='{split_name}' …")

    # 1. Dimension checks
    assert X.ndim == 3, (
        f"[{split_name}] X must be 3-D (n_windows, seq_length, n_features), "
        f"got ndim={X.ndim}"
    )
    assert y_phase.ndim == 2, (
        f"[{split_name}] y_phase must be 2-D (n_windows, seq_length), "
        f"got ndim={y_phase.ndim}"
    )
    assert y_reg.ndim == 3, (
        f"[{split_name}] y_reg must be 3-D (n_windows, seq_length, 1), "
        f"got ndim={y_reg.ndim}"
    )

    # 2. Sequence-length axis
    assert X.shape[1] == seq_length, (
        f"[{split_name}] X.shape[1]={X.shape[1]} != seq_length={seq_length}"
    )
    assert y_phase.shape[1] == seq_length, (
        f"[{split_name}] y_phase.shape[1]={y_phase.shape[1]} != seq_length={seq_length}"
    )
    assert y_reg.shape[1] == seq_length, (
        f"[{split_name}] y_reg.shape[1]={y_reg.shape[1]} != seq_length={seq_length}"
    )

    # 3. Consistent window count
    assert X.shape[0] == y_phase.shape[0] == y_reg.shape[0], (
        f"[{split_name}] Inconsistent window counts: "
        f"X={X.shape[0]}, y_phase={y_phase.shape[0]}, y_reg={y_reg.shape[0]}"
    )

    # 4. Regression last dim must be 1
    assert y_reg.shape[2] == 1, (
        f"[{split_name}] y_reg.shape[2]={y_reg.shape[2]} – last dimension must be 1"
    )

    # 5. Phase label range
    unique_phases = set(y_phase.flatten().tolist())
    invalid_phases = unique_phases - VALID_PHASE_LABELS
    assert not invalid_phases, (
        f"[{split_name}] y_phase contains invalid label(s): {invalid_phases}. "
        f"Expected subset of {VALID_PHASE_LABELS}."
    )

    # 6. NaN / Inf checks
    assert not np.any(np.isnan(X)), f"[{split_name}] X contains NaN values."
    assert not np.any(np.isinf(X)), f"[{split_name}] X contains infinite values."
    assert not np.any(np.isnan(y_phase)), f"[{split_name}] y_phase contains NaN values."
    assert not np.any(np.isnan(y_reg)), f"[{split_name}] y_reg contains NaN values."
    assert not np.any(np.isinf(y_reg)), f"[{split_name}] y_reg contains infinite values."

    # 7. Non-empty check
    assert X.shape[0] > 0, (
        f"[{split_name}] No windows were generated. "
        "Check seq_length vs. available cycle lengths."
    )

    print(
        f"  OK – {split_name}: X{X.shape}, y_phase{y_phase.shape}, y_reg{y_reg.shape}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Window calculation
# ──────────────────────────────────────────────────────────────────────────────


def calculate_expected_windows(df: pd.DataFrame, seq_length: int) -> int:
    """Compute the expected total window count for a DataFrame.

    For each (user_id, cycle_start_date) group the contribution is:
        max(group_size - seq_length + 1, 0)

    Parameters
    ----------
    df:
        A processed daily DataFrame (train, val, or test).
    seq_length:
        Sliding-window length.

    Returns
    -------
    Total expected window count across all groups.
    """
    group_sizes = df.groupby(["user_id", "cycle_start_date"]).size()
    contributions = group_sizes.apply(
        lambda n: max(n - seq_length + 1, 0)
    )
    return int(contributions.sum())


# ──────────────────────────────────────────────────────────────────────────────
# Core sliding-window builder
# ──────────────────────────────────────────────────────────────────────────────


def create_time_windows(
    df: pd.DataFrame,
    seq_length: int,
    feature_cols: List[str],
    phase_target_col: str = "phase_label",
    reg_target_col: str = "days_until_next_cycle",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sliding-window sequences from a daily DataFrame.

    Windows are created **within** each (user_id, cycle_start_date) group,
    so they never cross cycle boundaries.

    Parameters
    ----------
    df:
        Processed daily DataFrame (one row per user-day).
    seq_length:
        Number of consecutive days in each input window.
    feature_cols:
        Ordered list of feature column names (X columns).
    phase_target_col:
        Column name for the classification target (default: "phase_label").
    reg_target_col:
        Column name for the regression target (default: "days_until_next_cycle").

    Returns
    -------
    X        : np.ndarray  shape (n_windows, seq_length, n_features)   float32
    y_phase  : np.ndarray  shape (n_windows, seq_length)                int32
    y_reg    : np.ndarray  shape (n_windows, seq_length, 1)             float32
    """
    X_list: List[np.ndarray] = []
    y_phase_list: List[np.ndarray] = []
    y_reg_list: List[np.ndarray] = []

    skipped_cycles = 0
    total_cycles = 0

    groups = df.groupby(["user_id", "cycle_start_date"], sort=False)

    for (user_id, cycle_start), group in groups:
        total_cycles += 1

        # Sort by day position within the cycle
        group = group.sort_values("day_in_cycle").reset_index(drop=True)
        cycle_length = len(group)

        if cycle_length < seq_length:
            skipped_cycles += 1
            continue

        feature_vals = group[feature_cols].values.astype(np.float32)      # (L, F)
        phase_vals = group[phase_target_col].values.astype(np.int32)       # (L,)
        reg_vals = group[reg_target_col].values.astype(np.float32)         # (L,)

        n_windows = cycle_length - seq_length + 1

        for start in range(n_windows):
            end = start + seq_length
            X_list.append(feature_vals[start:end])        # (seq_length, F)
            y_phase_list.append(phase_vals[start:end])    # (seq_length,)
            y_reg_list.append(reg_vals[start:end])        # (seq_length,)

    if skipped_cycles > 0:
        print(
            f"  [WARNING] {skipped_cycles}/{total_cycles} cycles skipped "
            f"(cycle_length < seq_length={seq_length})."
        )

    if not X_list:
        raise ValueError(
            "[sequence_builder] No windows were generated. "
            "All cycles are shorter than seq_length. "
            "Consider using a smaller seq_length."
        )

    X = np.stack(X_list, axis=0)           # (n_windows, seq_length, n_features)
    y_phase = np.stack(y_phase_list, axis=0)  # (n_windows, seq_length)
    y_reg = np.stack(y_reg_list, axis=0)   # (n_windows, seq_length)

    # Reshape regression target to (n_windows, seq_length, 1)
    y_reg = y_reg[..., np.newaxis]

    return X, y_phase, y_reg


# ──────────────────────────────────────────────────────────────────────────────
# Save helpers
# ──────────────────────────────────────────────────────────────────────────────


def save_sequence_arrays(
    X_train: np.ndarray,
    y_phase_train: np.ndarray,
    y_reg_train: np.ndarray,
    X_val: np.ndarray,
    y_phase_val: np.ndarray,
    y_reg_val: np.ndarray,
    X_test: np.ndarray,
    y_phase_test: np.ndarray,
    y_reg_test: np.ndarray,
    output_dir: str,
    seq_length: int,
) -> dict:

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"seq{seq_length}"

    arrays = {
        f"X_train_{suffix}": X_train,
        f"y_phase_train_{suffix}": y_phase_train,
        f"y_reg_train_{suffix}": y_reg_train,
        f"X_val_{suffix}": X_val,
        f"y_phase_val_{suffix}": y_phase_val,
        f"y_reg_val_{suffix}": y_reg_val,
        f"X_test_{suffix}": X_test,
        f"y_phase_test_{suffix}": y_phase_test,
        f"y_reg_test_{suffix}": y_reg_test,
    }

    saved_paths: dict = {}
    for name, arr in arrays.items():
        path = os.path.join(output_dir, f"{name}.npy")
        np.save(path, arr)
        saved_paths[name] = path
        print(f"  Saved {name}.npy  {arr.shape}  ({arr.dtype})")

    return saved_paths


def save_sequence_info(
    output_dir: str,
    seq_length: int,
    feature_cols: List[str],
    X_train: np.ndarray,
    y_phase_train: np.ndarray,
    y_reg_train: np.ndarray,
    X_val: np.ndarray,
    y_phase_val: np.ndarray,
    y_reg_val: np.ndarray,
    X_test: np.ndarray,
    y_phase_test: np.ndarray,
    y_reg_test: np.ndarray,
) -> str:
    """Write the ``sequence_info_seq<N>.json`` metadata file.

    Returns
    -------
    Path to the written JSON file.
    """
    info = {
        "sequence_length": seq_length,
        "input_features": feature_cols,
        "n_features": len(feature_cols),
        "target_classification": "phase_label",
        "target_regression": "days_until_next_cycle",
        "windowing_strategy": "sliding_window_within_user_cycle",
        "grouping_columns": ["user_id", "cycle_start_date"],
        "sorting_column": "day_in_cycle",
        "cross_cycle_windows_allowed": False,
        "sequence_to_sequence_targets": True,
        "multi_output_lstm_ready": True,
        "model_target_format": {
            "X": "(n_windows, seq_length, n_features)",
            "y_phase": "(n_windows, seq_length)",
            "y_reg": "(n_windows, seq_length, 1)",
        },
        "shapes": {
            "X_train": list(X_train.shape),
            "y_phase_train": list(y_phase_train.shape),
            "y_reg_train": list(y_reg_train.shape),
            "X_val": list(X_val.shape),
            "y_phase_val": list(y_phase_val.shape),
            "y_reg_val": list(y_reg_val.shape),
            "X_test": list(X_test.shape),
            "y_phase_test": list(y_phase_test.shape),
            "y_reg_test": list(y_reg_test.shape),
        },
        "notes": [
            "Windows are generated within each user-cycle group and never "
            "cross cycle boundaries.",
            "Targets are sequence-to-sequence: each timestep has both a "
            "phase label and a days-until-next-cycle value.",
            "Targets are saved separately for the two output heads of the "
            "future multi-output LSTM.",
            "Scaling is not performed in this module; it is expected to be "
            "completed in preprocessing.py.",
        ],
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"sequence_info_seq{seq_length}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"  Saved sequence_info_seq{seq_length}.json")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Summary printer
# ──────────────────────────────────────────────────────────────────────────────


def _print_summary(
    seq_length: int,
    feature_cols: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_train: np.ndarray,
    y_phase_train: np.ndarray,
    y_reg_train: np.ndarray,
    X_val: np.ndarray,
    y_phase_val: np.ndarray,
    y_reg_val: np.ndarray,
    X_test: np.ndarray,
    y_phase_test: np.ndarray,
    y_reg_test: np.ndarray,
    expected_train: int,
    expected_val: int,
    expected_test: int,
    saved_paths: dict,
) -> None:
    """Print a human-readable summary of the build run."""
    sep = "=" * 72

    print(f"\n{sep}")
    print("  SEQUENCE BUILDER – SUMMARY")
    print(sep)
    print(f"  seq_length      : {seq_length}")
    print(f"  n_features      : {len(feature_cols)}")
    print(f"  input_features  : {feature_cols}")

    print(f"\n  Daily-level shapes")
    print(f"    train_daily   : {train_df.shape}")
    print(f"    val_daily     : {val_df.shape}")
    print(f"    test_daily    : {test_df.shape}")

    print(f"\n  Window counts")
    for split, actual, expected in [
        ("train", X_train.shape[0], expected_train),
        ("val",   X_val.shape[0],   expected_val),
        ("test",  X_test.shape[0],  expected_test),
    ]:
        match = "[OK]" if actual == expected else "[!! MISMATCH]"
        print(
            f"    {split:5s}  actual={actual:6d}  expected={expected:6d}  {match}"
        )

    print(f"\n  Array shapes")
    for name, arr in [
        ("X_train",       X_train),
        ("y_phase_train", y_phase_train),
        ("y_reg_train",   y_reg_train),
        ("X_val",         X_val),
        ("y_phase_val",   y_phase_val),
        ("y_reg_val",     y_reg_val),
        ("X_test",        X_test),
        ("y_phase_test",  y_phase_test),
        ("y_reg_test",    y_reg_test),
    ]:
        print(f"    {name:20s}: {arr.shape}  dtype={arr.dtype}")

    print(f"\n  phase_label distribution")
    for split, y_ph in [("train", y_phase_train), ("val", y_phase_val), ("test", y_phase_test)]:
        flat = y_ph.flatten()
        total = flat.size
        dist = {
            int(lbl): f"{(flat == lbl).sum():,}  ({100.0 * (flat == lbl).mean():.1f}%)"
            for lbl in sorted(VALID_PHASE_LABELS)
        }
        print(f"    {split:5s}  total_timesteps={total:,}")
        for lbl, info in dist.items():
            phase_names = {0: "menstruation", 1: "follicular", 2: "ovulation", 3: "luteal"}
            print(f"           phase {lbl} ({phase_names[lbl]:13s}): {info}")

    print(f"\n  y_reg (days_until_next_cycle) statistics")
    for split, y_r in [("train", y_reg_train), ("val", y_reg_val), ("test", y_reg_test)]:
        flat = y_r.flatten()
        print(
            f"    {split:5s}  min={flat.min():.2f}  max={flat.max():.2f}  "
            f"mean={flat.mean():.2f}  std={flat.std():.2f}"
        )

    print(f"\n  Saved output paths")
    for name, path in saved_paths.items():
        print(f"    {name:30s}: {path}")

    print(sep + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────


def run_sequence_builder(
    train_path: str = "data/processed/train_daily.csv",
    val_path: str = "data/processed/val_daily.csv",
    test_path: str = "data/processed/test_daily.csv",
    feature_info_path: str = "data/processed/feature_info.json",
    output_dir: str = "data/sequences",
    seq_length: int = 14,
) -> None:
    """End-to-end sequence building pipeline.

    Steps
    -----
    1. Load processed CSV splits.
    2. Load feature_info.json and extract input_features.
    3. Validate feature columns (no leakage, all columns present).
    4. Build sliding-window sequences for each split.
    5. Validate generated arrays.
    6. Compare actual vs. expected window counts.
    7. Save .npy arrays and sequence_info JSON.
    8. Print summary.

    """
    print("\n" + "=" * 72)
    print("  SEQUENCE BUILDER – START")
    print("=" * 72)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    train_df, val_df, test_df = load_split_data(train_path, val_path, test_path)

    # ── 2. Load feature info ──────────────────────────────────────────────────
    feature_info = load_feature_info(feature_info_path)
    feature_cols: List[str] = feature_info["input_features"]
    print(f"  input_features ({len(feature_cols)}): {feature_cols}")

    # ── 3. Validate features ──────────────────────────────────────────────────
    validate_input_features([train_df, val_df, test_df], feature_cols)

    # ── 4. Build sequences ────────────────────────────────────────────────────
    print("\n[sequence_builder] Building train sequences …")
    X_train, y_phase_train, y_reg_train = create_time_windows(
        train_df, seq_length, feature_cols
    )

    print("[sequence_builder] Building val sequences …")
    X_val, y_phase_val, y_reg_val = create_time_windows(
        val_df, seq_length, feature_cols
    )

    print("[sequence_builder] Building test sequences …")
    X_test, y_phase_test, y_reg_test = create_time_windows(
        test_df, seq_length, feature_cols
    )

    # ── 5. Validate arrays ────────────────────────────────────────────────────
    print()
    validate_sequence_arrays(X_train, y_phase_train, y_reg_train, seq_length, "train")
    validate_sequence_arrays(X_val, y_phase_val, y_reg_val, seq_length, "val")
    validate_sequence_arrays(X_test, y_phase_test, y_reg_test, seq_length, "test")

    # ── 6. Expected-vs-actual window count check ──────────────────────────────
    print("\n[sequence_builder] Checking expected vs. actual window counts …")
    expected_train = calculate_expected_windows(train_df, seq_length)
    expected_val = calculate_expected_windows(val_df, seq_length)
    expected_test = calculate_expected_windows(test_df, seq_length)

    for split, actual, expected in [
        ("train", X_train.shape[0], expected_train),
        ("val",   X_val.shape[0],   expected_val),
        ("test",  X_test.shape[0],  expected_test),
    ]:
        if actual != expected:
            raise AssertionError(
                f"[sequence_builder] Window count mismatch for split='{split}': "
                f"actual={actual}, expected={expected}."
            )
        print(f"  {split:5s}: {actual:,} windows  [OK]")

    # ── 7. Save arrays ────────────────────────────────────────────────────────
    print(f"\n[sequence_builder] Saving arrays to: {output_dir}")
    saved_paths = save_sequence_arrays(
        X_train, y_phase_train, y_reg_train,
        X_val, y_phase_val, y_reg_val,
        X_test, y_phase_test, y_reg_test,
        output_dir=output_dir,
        seq_length=seq_length,
    )

    # ── 7b. Save metadata JSON ────────────────────────────────────────────────
    json_path = save_sequence_info(
        output_dir=output_dir,
        seq_length=seq_length,
        feature_cols=feature_cols,
        X_train=X_train,
        y_phase_train=y_phase_train,
        y_reg_train=y_reg_train,
        X_val=X_val,
        y_phase_val=y_phase_val,
        y_reg_val=y_reg_val,
        X_test=X_test,
        y_phase_test=y_phase_test,
        y_reg_test=y_reg_test,
    )
    saved_paths[f"sequence_info_seq{seq_length}"] = json_path

    # ── 8. Summary ────────────────────────────────────────────────────────────
    _print_summary(
        seq_length=seq_length,
        feature_cols=feature_cols,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        X_train=X_train,
        y_phase_train=y_phase_train,
        y_reg_train=y_reg_train,
        X_val=X_val,
        y_phase_val=y_phase_val,
        y_reg_val=y_reg_val,
        X_test=X_test,
        y_phase_test=y_phase_test,
        y_reg_test=y_reg_test,
        expected_train=expected_train,
        expected_val=expected_val,
        expected_test=expected_test,
        saved_paths=saved_paths,
    )

    print("[sequence_builder] Done.")
