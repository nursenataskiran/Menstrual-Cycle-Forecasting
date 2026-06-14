"""
baseline_lstm.py
================
Baseline multi-output LSTM for menstrual cycle forecasting.

Architecture
------------
  Input  --> LSTM(64, return_sequences=True)
         --> TimeDistributed(Dense(32, tanh))
         --> [phase_output]  TimeDistributed(Dense(4, softmax))   shape (B, T, 4)
         --> [regression_output] TimeDistributed(Dense(1, linear)) shape (B, T, 1)

Single input, two outputs (Functional API).
Dropout is optional and controlled by the configuration dictionary.
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

# ──────────────────────────────────────────────────────────────────────────────
# Deferred TF import so seed-setting happens before any TF graph allocation
# ──────────────────────────────────────────────────────────────────────────────

def _import_tf():
    import tensorflow as tf
    return tf


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int = 42) -> None:
    """Fix Python, NumPy and TensorFlow seeds for reproducibility.

    Parameters
    ----------
    seed:
        Integer seed value (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    tf = _import_tf()
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[baseline_lstm] Seeds fixed: seed={seed}")


# ──────────────────────────────────────────────────────────────────────────────
# Data loading & validation
# ──────────────────────────────────────────────────────────────────────────────

def load_sequence_arrays(
    sequence_dir: str = "data/sequences",
    seq_length: int = 14,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    dict,
]:
    """Load all nine .npy arrays and sequence_info JSON.

    Parameters
    ----------
    sequence_dir:
        Directory containing the sequence files.
    seq_length:
        Sequence length encoded in the filenames (e.g., 14 → ``*_seq14.npy``).

    Returns
    -------
    X_train, y_phase_train, y_reg_train,
    X_val,   y_phase_val,   y_reg_val,
    X_test,  y_phase_test,  y_reg_test,
    sequence_info
    """
    sfx = f"seq{seq_length}"
    print(f"[baseline_lstm] Loading sequence arrays from: {sequence_dir}  (suffix={sfx})")

    def _load(name: str) -> np.ndarray:
        path = os.path.join(sequence_dir, f"{name}_{sfx}.npy")
        arr = np.load(path)
        print(f"  {name}_{sfx}.npy  {arr.shape}  {arr.dtype}")
        return arr

    X_train       = _load("X_train")
    y_phase_train = _load("y_phase_train")
    y_reg_train   = _load("y_reg_train")
    X_val         = _load("X_val")
    y_phase_val   = _load("y_phase_val")
    y_reg_val     = _load("y_reg_val")
    X_test        = _load("X_test")
    y_phase_test  = _load("y_phase_test")
    y_reg_test    = _load("y_reg_test")

    info_path = os.path.join(sequence_dir, f"sequence_info_{sfx}.json")
    with open(info_path, "r", encoding="utf-8") as f:
        sequence_info = json.load(f)
    print(f"  sequence_info_{sfx}.json loaded.")

    # ── validation ────────────────────────────────────────────────────────────
    _validate_loaded_arrays(
        X_train, y_phase_train, y_reg_train,
        X_val,   y_phase_val,   y_reg_val,
        X_test,  y_phase_test,  y_reg_test,
        seq_length=seq_length,
        n_features=sequence_info["n_features"],
    )

    return (
        X_train, y_phase_train, y_reg_train,
        X_val,   y_phase_val,   y_reg_val,
        X_test,  y_phase_test,  y_reg_test,
        sequence_info,
    )


def _validate_loaded_arrays(
    X_train, y_phase_train, y_reg_train,
    X_val,   y_phase_val,   y_reg_val,
    X_test,  y_phase_test,  y_reg_test,
    seq_length: int,
    n_features: int,
) -> None:
    """Run shape / value sanity checks on loaded arrays.

    Raises
    ------
    AssertionError on any failed check.
    """
    valid_labels = {0, 1, 2, 3}

    for split, X, y_ph, y_r in [
        ("train", X_train, y_phase_train, y_reg_train),
        ("val",   X_val,   y_phase_val,   y_reg_val),
        ("test",  X_test,  y_phase_test,  y_reg_test),
    ]:
        assert X.ndim == 3,       f"[{split}] X must be 3-D, got {X.ndim}-D"
        assert y_ph.ndim == 2,    f"[{split}] y_phase must be 2-D, got {y_ph.ndim}-D"
        assert y_r.ndim == 3,     f"[{split}] y_reg must be 3-D, got {y_r.ndim}-D"
        assert X.shape[1] == seq_length,   f"[{split}] X.shape[1]={X.shape[1]} != seq_length={seq_length}"
        assert y_ph.shape[1] == seq_length, f"[{split}] y_phase.shape[1]={y_ph.shape[1]} != {seq_length}"
        assert y_r.shape[1] == seq_length,  f"[{split}] y_reg.shape[1]={y_r.shape[1]} != {seq_length}"
        assert X.shape[2] == n_features,   f"[{split}] X.shape[2]={X.shape[2]} != n_features={n_features}"
        bad = set(y_ph.flatten().tolist()) - valid_labels
        assert not bad, f"[{split}] Invalid phase labels found: {bad}"
        assert not np.any(np.isnan(X)),    f"[{split}] X contains NaN"
        assert not np.any(np.isinf(X)),    f"[{split}] X contains Inf"
        assert not np.any(np.isnan(y_ph)), f"[{split}] y_phase contains NaN"
        assert not np.any(np.isnan(y_r)),  f"[{split}] y_reg contains NaN"
        assert not np.any(np.isinf(y_r)),  f"[{split}] y_reg contains Inf"

    print("  [validate] All array checks passed.")


# ──────────────────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────────────────

def build_baseline_multi_output_lstm(
    seq_length: int,
    n_features: int,
    n_phase_classes: int = 4,
    lstm_units: int = 64,
    dense_units: int = 32,
    learning_rate: float = 0.001,
    dense_activation: str = "tanh",
    kernel_initializer: str = "glorot_uniform",
    recurrent_initializer: str = "orthogonal",
    optimizer_name: str = "Adam",
    loss_weights: Optional[dict] = None,
    dropout_rate: float = 0.0,
) -> "tf.keras.Model":
   
    tf = _import_tf()
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
    from tensorflow.keras.optimizers import Adam, SGD

    # ── Resolve loss_weights default ──────────────────────────────────────────
    if loss_weights is None:
        loss_weights = {"phase_output": 1.0, "regression_output": 0.1}

    # ── Resolve optimizer ─────────────────────────────────────────────────────
    if optimizer_name == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = SGD(learning_rate=learning_rate)
    else:
        raise ValueError(
            f"[build_baseline_multi_output_lstm] Unknown optimizer_name: "
            f"'{optimizer_name}'. Supported: 'Adam', 'SGD'."
        )

    # ── Input ─────────────────────────────────────────────────────────────────
    inputs = Input(shape=(seq_length, n_features), name="input")

    # ── Shared LSTM encoder ───────────────────────────────────────────────────
    x = LSTM(
        units=lstm_units,
        return_sequences=True,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        name="shared_lstm",
    )(inputs)

    # ── Optional Dropout (after LSTM, before shared Dense) ─────────────────────
    if dropout_rate and dropout_rate > 0.0:
        x = Dropout(dropout_rate, name="shared_dropout")(x)

    # ── Shared dense representation ───────────────────────────────────────────
    x = TimeDistributed(
        Dense(
            dense_units,
            activation=dense_activation,
            kernel_initializer=kernel_initializer,
        ),
        name="shared_dense",
    )(x)

    # ── Classification head: (B, T, n_phase_classes) ──────────────────────────
    phase_output = TimeDistributed(
        Dense(
            n_phase_classes,
            activation="softmax",
            kernel_initializer=kernel_initializer,
        ),
        name="phase_output",
    )(x)

    # ── Regression head: (B, T, 1) ───────────────────────────────────────────
    regression_output = TimeDistributed(
        Dense(
            1,
            activation="linear",
            kernel_initializer=kernel_initializer,
        ),
        name="regression_output",
    )(x)

    # ── Build model ───────────────────────────────────────────────────────────
    model = Model(
        inputs=inputs,
        outputs=[phase_output, regression_output],
        name="baseline_multi_output_lstm",
    )

    # ── Compile ───────────────────────────────────────────────────────────────
    model.compile(
        optimizer=optimizer,
        loss={
            "phase_output":      "sparse_categorical_crossentropy",
            "regression_output": "mae",
        },
        loss_weights=loss_weights,
        metrics={
            "phase_output":      ["accuracy"],
            "regression_output": ["mae"],
        },
    )

    return model




# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_baseline_model(
    X_train: np.ndarray,
    y_phase_train: np.ndarray,
    y_reg_train: np.ndarray,
    X_val: np.ndarray,
    y_phase_val: np.ndarray,
    y_reg_val: np.ndarray,
    seq_length: int,
    n_features: int,
    config: dict,
) -> Tuple["tf.keras.Model", "tf.keras.callbacks.History"]:
    """Build and train the baseline multi-output LSTM.

    Parameters
    ----------
    X_train, y_phase_train, y_reg_train:
        Training arrays.
    X_val, y_phase_val, y_reg_val:
        Validation arrays.
    seq_length, n_features:
        Used to construct the model architecture.
    config:
        Training configuration dict. Expected keys:
        ``lstm_units``, ``dense_units``, ``learning_rate``,
        ``batch_size``, ``epochs``, ``model_save_path``.

    Returns
    -------
    (model, history)
    """
    tf = _import_tf()
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    # ── Build ─────────────────────────────────────────────────────────────────
    # Resolve optimizer_name: prefer explicit key, fall back to legacy "optimizer"
    optimizer_name = config.get("optimizer_name", config.get("optimizer", "Adam"))
    # Resolve dropout_rate: prefer "dropout_rate", fall back to "dropout"
    dropout_rate = float(
        config.get("dropout_rate", config.get("dropout", None) or 0.0)
    )

    model = build_baseline_multi_output_lstm(
        seq_length=seq_length,
        n_features=n_features,
        n_phase_classes=4,
        lstm_units=config["lstm_units"],
        dense_units=config["dense_units"],
        learning_rate=config["learning_rate"],
        dense_activation=config.get("dense_activation", "tanh"),
        kernel_initializer=config.get("kernel_initializer", "glorot_uniform"),
        recurrent_initializer=config.get("recurrent_initializer", "orthogonal"),
        optimizer_name=optimizer_name,
        loss_weights=config.get("loss_weights", None),
        dropout_rate=dropout_rate,
    )

    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    model_save_path = config["model_save_path"]
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    # ── Fit ───────────────────────────────────────────────────────────────────
    print("\n[baseline_lstm] Starting training …")
    history = model.fit(
        X_train,
        {
            "phase_output":      y_phase_train,
            "regression_output": y_reg_train,
        },
        validation_data=(
            X_val,
            {
                "phase_output":      y_phase_val,
                "regression_output": y_reg_val,
            },
        ),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=[early_stopping, checkpoint],
        verbose=1,
    )

    print("[baseline_lstm] Training complete.")
    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

PHASE_NAMES = ["menstruation", "follicular", "ovulation", "luteal"]


def evaluate_model(
    model: "tf.keras.Model",
    X: np.ndarray,
    y_phase: np.ndarray,
    y_reg: np.ndarray,
    split_name: str,
    figures_dir: str,
    seq_length: int,
    experiment_name: Optional[str] = None,
) -> dict:
    """Evaluate the model on one split and return a metrics dictionary.

    Steps
    -----
    1. model.evaluate  – total loss and per-output metrics.
    2. model.predict   – raw predictions.
    3. Argmax → integer phase predictions.
    4. Flatten for sklearn metrics.
    5. Compute classification_report, confusion_matrix, macro/weighted F1.
    6. Compute regression MAE and RMSE (in days).
    7. Save confusion matrix PNG.

    Parameters
    ----------
    model:
        Trained Keras model.
    X, y_phase, y_reg:
        Arrays for the evaluation split.
    split_name:
        ``"val"`` or ``"test"`` – used in filenames and print headers.
    figures_dir:
        Directory for PNG outputs.
    seq_length:
        Used in the figure title (not in the filename).
    experiment_name:
        Identifier prepended to all output filenames.  If ``None``, falls
        back to ``"baseline_seq{seq_length}"`` (backward-compatible).

    Returns
    -------
    Dictionary with all evaluation metrics.
    """
    if experiment_name is None:
        experiment_name = f"baseline_seq{seq_length}"

    print(f"\n[baseline_lstm] Evaluating on split='{split_name}' …")

    # ── Keras evaluate ────────────────────────────────────────────────────────
    eval_results = model.evaluate(
        X,
        {"phase_output": y_phase, "regression_output": y_reg},
        verbose=0,
    )
    # Print raw Keras metrics
    metric_names = model.metrics_names
    print(f"  Keras evaluate results:")
    for mname, mval in zip(metric_names, eval_results):
        print(f"    {mname}: {mval:.5f}")

    # ── Predict ───────────────────────────────────────────────────────────────
    predictions = model.predict(X, verbose=0)

    # Handle both list and tuple returns
    if isinstance(predictions, (list, tuple)):
        phase_pred_probs = predictions[0]   # (n_windows, seq_length, 4)
        reg_pred         = predictions[1]   # (n_windows, seq_length, 1)
    else:
        raise TypeError(
            f"model.predict returned unexpected type: {type(predictions)}"
        )

    # ── Phase predictions → integer labels ───────────────────────────────────
    y_phase_pred = np.argmax(phase_pred_probs, axis=-1)  # (n_windows, seq_length)

    # ── Flatten for sklearn ───────────────────────────────────────────────────
    y_phase_flat      = y_phase.flatten()         # (n_windows * seq_length,)
    y_phase_pred_flat = y_phase_pred.flatten()    # (n_windows * seq_length,)

    y_reg_flat      = y_reg.flatten()             # (n_windows * seq_length,)
    y_reg_pred_flat = reg_pred.flatten()          # (n_windows * seq_length,)

    # ── Classification metrics ────────────────────────────────────────────────
    acc        = float(accuracy_score(y_phase_flat, y_phase_pred_flat))
    macro_f1   = float(f1_score(y_phase_flat, y_phase_pred_flat, average="macro",    zero_division=0))
    weighted_f1= float(f1_score(y_phase_flat, y_phase_pred_flat, average="weighted", zero_division=0))
    cls_report = classification_report(
        y_phase_flat, y_phase_pred_flat,
        target_names=PHASE_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_phase_flat, y_phase_pred_flat, labels=[0, 1, 2, 3])

    # ── Regression metrics ────────────────────────────────────────────────────
    mae_days  = float(mean_absolute_error(y_reg_flat, y_reg_pred_flat))
    rmse_days = float(math.sqrt(mean_squared_error(y_reg_flat, y_reg_pred_flat)))

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n  === {split_name.upper()} EVALUATION ===")
    print(f"  Classification accuracy  : {acc:.4f}")
    print(f"  Macro F1                 : {macro_f1:.4f}")
    print(f"  Weighted F1              : {weighted_f1:.4f}")
    print(f"  Regression MAE (days)    : {mae_days:.4f}")
    print(f"  Regression RMSE (days)   : {rmse_days:.4f}")
    print(f"\n  Classification report:")
    print(
        classification_report(
            y_phase_flat, y_phase_pred_flat,
            target_names=PHASE_NAMES,
            zero_division=0,
        )
    )
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    os.makedirs(figures_dir, exist_ok=True)
    _plot_confusion_matrix(
        cm=cm,
        class_names=PHASE_NAMES,
        split_name=split_name,
        seq_length=seq_length,
        figures_dir=figures_dir,
        experiment_name=experiment_name,
    )

    # ── Package results ───────────────────────────────────────────────────────
    result = {
        "classification_accuracy": acc,
        "macro_f1":                macro_f1,
        "weighted_f1":             weighted_f1,
        "regression_mae_days":     mae_days,
        "regression_rmse_days":    rmse_days,
        "classification_report":   cls_report,
        "confusion_matrix":        cm.tolist(),
    }
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    split_name: str,
    seq_length: int,
    figures_dir: str,
    experiment_name: Optional[str] = None,
) -> None:

    if experiment_name is None:
        experiment_name = f"baseline_seq{seq_length}"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    ax.set_title(
        f"Confusion Matrix – {split_name} (seq{seq_length})", fontsize=11
    )
    plt.tight_layout()

    out_path = os.path.join(
        figures_dir, f"{experiment_name}_confusion_matrix_{split_name}.png"
    )
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix: {out_path}")


def _resolve_history_key(history_dict: dict, candidates: List[str]) -> Optional[str]:
    """Return the first candidate key found in history_dict, else None."""
    for k in candidates:
        if k in history_dict:
            return k
    return None


def plot_training_curves(
    history: "tf.keras.callbacks.History",
    figures_dir: str,
    experiment_name: str,
    seq_length: int,
) -> None:

    os.makedirs(figures_dir, exist_ok=True)
    h = history.history
    print(f"\n[baseline_lstm] History keys: {list(h.keys())}")

    epochs_range = range(1, len(h["loss"]) + 1)

    # ── 1. Total loss ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs_range, h["loss"],     label="train total loss", color="royalblue")
    ax.plot(epochs_range, h["val_loss"], label="val total loss",   color="coral", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (weighted sum)")
    ax.set_title(f"Total Loss – {experiment_name} (seq{seq_length})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(figures_dir, f"{experiment_name}_total_loss.png")
    plt.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved: {p}")

    # ── 2. Phase accuracy ─────────────────────────────────────────────────────
    train_acc_key = _resolve_history_key(h, [
        "phase_output_accuracy",
        "phase_output_acc",
        "accuracy",
        "acc",
    ])
    val_acc_key = _resolve_history_key(h, [
        "val_phase_output_accuracy",
        "val_phase_output_acc",
        "val_accuracy",
        "val_acc",
    ])

    if train_acc_key and val_acc_key:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs_range, h[train_acc_key], label="train phase accuracy", color="royalblue")
        ax.plot(epochs_range, h[val_acc_key],   label="val phase accuracy",   color="coral", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Phase Accuracy – {experiment_name} (seq{seq_length})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(figures_dir, f"{experiment_name}_phase_accuracy.png")
        plt.savefig(p, dpi=150); plt.close(fig)
        print(f"  Saved: {p}")
    else:
        print(f"  [WARNING] Phase accuracy keys not found in history. "
              f"Available: {list(h.keys())}")

    # ── 3. Regression MAE ─────────────────────────────────────────────────────
    train_mae_key = _resolve_history_key(h, [
        "regression_output_mae",
        "regression_output_mean_absolute_error",
        "regression_mae",
        "mae",
    ])
    val_mae_key = _resolve_history_key(h, [
        "val_regression_output_mae",
        "val_regression_output_mean_absolute_error",
        "val_regression_mae",
        "val_mae",
    ])

    if train_mae_key and val_mae_key:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs_range, h[train_mae_key], label="train reg MAE (days)", color="royalblue")
        ax.plot(epochs_range, h[val_mae_key],   label="val reg MAE (days)",   color="coral", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE (days)")
        ax.set_title(f"Regression MAE – {experiment_name} (seq{seq_length})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(figures_dir, f"{experiment_name}_regression_mae.png")
        plt.savefig(p, dpi=150); plt.close(fig)
        print(f"  Saved: {p}")
    else:
        print(f"  [WARNING] Regression MAE keys not found in history. "
              f"Available: {list(h.keys())}")


# ──────────────────────────────────────────────────────────────────────────────
# Report saving
# ──────────────────────────────────────────────────────────────────────────────

def save_history(
    history: "tf.keras.callbacks.History",
    reports_dir: str,
    experiment_name: str,
) -> str:

    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, f"{experiment_name}_history.json")

    # Convert numpy floats to Python floats for JSON serialisation
    serialisable = {
        k: [float(v) for v in vals]
        for k, vals in history.history.items()
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  Saved training history: {out_path}")
    return out_path


def save_metrics(
    val_metrics: dict,
    test_metrics: dict,
    config: dict,
    reports_dir: str,
    experiment_name: str,
) -> str:

    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, f"{experiment_name}_metrics.json")

    payload = {
        "validation": val_metrics,
        "test":       test_metrics,
        # experiment_config is the new canonical key;
        # baseline_config is kept as an alias for backward compatibility.
        "experiment_config": {
            "experiment_name":      experiment_name,
            "seq_length":           config.get("seq_length"),
            "lstm_units":           config.get("lstm_units", 64),
            "dense_units":          config.get("dense_units", 32),
            "dropout":              config.get("dropout", None),
            "dropout_rate":         config.get("dropout_rate", 0.0),
            "dense_activation":     config.get("dense_activation", "tanh"),
            "kernel_initializer":   config.get("kernel_initializer", "glorot_uniform"),
            "recurrent_initializer": config.get("recurrent_initializer", "orthogonal"),
            "optimizer_name":       config.get("optimizer_name", config.get("optimizer", "Adam")),
            "learning_rate":        config.get("learning_rate", 0.001),
            "batch_size":           config.get("batch_size", 32),
            "epochs":               config.get("epochs", 30),
            "loss_weights":         config.get("loss_weights", {
                "phase_output": 1.0, "regression_output": 0.1
            }),
        },
    }
    # Alias for backward compatibility
    payload["baseline_config"] = payload["experiment_config"]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved metrics report: {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Final summary printer
# ──────────────────────────────────────────────────────────────────────────────

def print_final_summary(
    model: "tf.keras.Model",
    X_train: np.ndarray, y_phase_train: np.ndarray, y_reg_train: np.ndarray,
    X_val:   np.ndarray, y_phase_val:   np.ndarray, y_reg_val:   np.ndarray,
    X_test:  np.ndarray, y_phase_test:  np.ndarray, y_reg_test:  np.ndarray,
    history: "tf.keras.callbacks.History",
    val_metrics: dict,
    test_metrics: dict,
    saved_paths: dict,
) -> None:
    """Print a consolidated final summary to stdout."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("  BASELINE LSTM – FINAL SUMMARY")
    print(sep)

    # Model shapes
    print(f"\n  Model: {model.name}")
    print(f"    Input  shape : {model.input_shape}")
    print(f"    Output shapes: {[o.shape for o in model.outputs]}")

    # Array shapes
    print(f"\n  Array shapes")
    for lbl, arr in [
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
        print(f"    {lbl:20s}: {arr.shape}  {arr.dtype}")

    # Training history
    best_val_loss = min(history.history["val_loss"])
    print(f"\n  Training")
    print(f"    Epochs trained        : {len(history.history['loss'])}")
    print(f"    Best val_loss         : {best_val_loss:.5f}")

    # Validation metrics
    print(f"\n  Validation metrics")
    print(f"    Accuracy              : {val_metrics['classification_accuracy']:.4f}")
    print(f"    Macro F1              : {val_metrics['macro_f1']:.4f}")
    print(f"    Weighted F1           : {val_metrics['weighted_f1']:.4f}")
    print(f"    Reg MAE  (days)       : {val_metrics['regression_mae_days']:.4f}")
    print(f"    Reg RMSE (days)       : {val_metrics['regression_rmse_days']:.4f}")

    # Test metrics
    print(f"\n  Test metrics")
    print(f"    Accuracy              : {test_metrics['classification_accuracy']:.4f}")
    print(f"    Macro F1              : {test_metrics['macro_f1']:.4f}")
    print(f"    Weighted F1           : {test_metrics['weighted_f1']:.4f}")
    print(f"    Reg MAE  (days)       : {test_metrics['regression_mae_days']:.4f}")
    print(f"    Reg RMSE (days)       : {test_metrics['regression_rmse_days']:.4f}")

    # Saved paths
    print(f"\n  Saved files")
    for label, path in saved_paths.items():
        print(f"    {label:35s}: {path}")

    print(sep + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_baseline_lstm(
    sequence_dir: str = "data/sequences",
    models_dir: str = "models",
    reports_dir: str = "reports",
    figures_dir: str = "reports/figures",
    seq_length: int = 14,
    config: Optional[dict] = None,
) -> None:

    if config is None:
        config = {}

    # ── Resolve experiment_name FIRST (needed for model_save_path default) ────
    experiment_name: str = config.get(
        "experiment_name", f"baseline_seq{seq_length}"
    )
    config["experiment_name"] = experiment_name

    # ── Merge remaining defaults ──────────────────────────────────────────────
    defaults = {
        "seq_length":           seq_length,
        "lstm_units":           64,
        "dense_units":          32,
        "dropout":              None,
        "dropout_rate":         0.0,
        "dense_activation":     "tanh",
        "kernel_initializer":   "glorot_uniform",
        "recurrent_initializer": "orthogonal",
        "optimizer_name":       "Adam",
        "learning_rate":        0.001,
        "batch_size":           32,
        "epochs":               30,
        "loss_weights":         {"phase_output": 1.0, "regression_output": 0.1},
        "seed":                 42,
        # model_save_path derives from experiment_name if not explicitly given
        "model_save_path":      os.path.join(models_dir, f"{experiment_name}.keras"),
    }
    for k, v in defaults.items():
        config.setdefault(k, v)

    print("\n" + "=" * 72)
    print(f"  BASELINE LSTM – START  [{experiment_name}]")
    print("=" * 72)
    print(f"  experiment_name : {experiment_name}")
    print(f"  seq_length      : {seq_length}")
    print(f"  model_save_path : {config['model_save_path']}")

    # ── Seeds ─────────────────────────────────────────────────────────────────
    set_seeds(config["seed"])

    # ── Load data ─────────────────────────────────────────────────────────────
    (
        X_train, y_phase_train, y_reg_train,
        X_val,   y_phase_val,   y_reg_val,
        X_test,  y_phase_test,  y_reg_test,
        sequence_info,
    ) = load_sequence_arrays(sequence_dir=sequence_dir, seq_length=seq_length)

    n_features = sequence_info["n_features"]

    # ── Train ─────────────────────────────────────────────────────────────────
    model, history = train_baseline_model(
        X_train, y_phase_train, y_reg_train,
        X_val,   y_phase_val,   y_reg_val,
        seq_length=seq_length,
        n_features=n_features,
        config=config,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_metrics  = evaluate_model(
        model, X_val,  y_phase_val,  y_reg_val,  "val",
        figures_dir, seq_length, experiment_name=experiment_name,
    )
    test_metrics = evaluate_model(
        model, X_test, y_phase_test, y_reg_test, "test",
        figures_dir, seq_length, experiment_name=experiment_name,
    )

    # ── Save training curves ──────────────────────────────────────────────────
    plot_training_curves(history, figures_dir, experiment_name, seq_length)

    # ── Save reports ──────────────────────────────────────────────────────────
    history_path = save_history(history, reports_dir, experiment_name)
    metrics_path = save_metrics(val_metrics, test_metrics, config, reports_dir, experiment_name)

    # ── Collect saved paths ───────────────────────────────────────────────────
    saved_paths = {
        "best model (.keras)":  config["model_save_path"],
        "metrics JSON":         metrics_path,
        "history JSON":         history_path,
        "CM val PNG":           os.path.join(figures_dir, f"{experiment_name}_confusion_matrix_val.png"),
        "CM test PNG":          os.path.join(figures_dir, f"{experiment_name}_confusion_matrix_test.png"),
        "total loss PNG":       os.path.join(figures_dir, f"{experiment_name}_total_loss.png"),
        "phase accuracy PNG":   os.path.join(figures_dir, f"{experiment_name}_phase_accuracy.png"),
        "regression MAE PNG":   os.path.join(figures_dir, f"{experiment_name}_regression_mae.png"),
    }

    # ── Final summary ─────────────────────────────────────────────────────────
    print_final_summary(
        model=model,
        X_train=X_train, y_phase_train=y_phase_train, y_reg_train=y_reg_train,
        X_val=X_val,     y_phase_val=y_phase_val,     y_reg_val=y_reg_val,
        X_test=X_test,   y_phase_test=y_phase_test,   y_reg_test=y_reg_test,
        history=history,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        saved_paths=saved_paths,
    )

    print("[baseline_lstm] Done.")
