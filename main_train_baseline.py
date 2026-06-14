"""
main_train_baseline.py
======================
Training entry point for the baseline multi-output LSTM.

Usage
-----
    python main_train_baseline.py

This script is intentionally separate from main_data_prep.py.
Data preparation is not re-run here – it uses the already-generated
sequence arrays in data/sequences/.
"""

from src.baseline_lstm import run_baseline_lstm

if __name__ == "__main__":

    # ── Training configuration ─────────────────────────────────────────────────
    config = {
        # Experiment identifier  ← drives ALL output filenames
        "experiment_name": "seq21_tanh_glorot_adam_bs32_dropout02",

        # Architecture
        "seq_length":           21,
        "lstm_units":           64,
        "dense_units":          32,

        # Dropout  
        "dropout":              0.2,   # legacy key kept for compatibility
        "dropout_rate":         0.2,   # canonical key – Dropout(0.2) after LSTM

        # Activation & initializers  
        "dense_activation":     "tanh",
        "kernel_initializer":   "glorot_uniform",
        "recurrent_initializer": "orthogonal",

        # Optimisation
        "optimizer_name":       "Adam",
        "learning_rate":        0.001,
        "batch_size":           32,
        "epochs":               30,

        # Loss weighting  
        "loss_weights": {
            "phase_output":      1.0,
            "regression_output": 0.1,
        },

        # Reproducibility
        "seed": 42,

        # Checkpoint path – derived from experiment_name automatically if omitted.
        "model_save_path": "models/seq21_tanh_glorot_adam_bs32_dropout02.keras",
    }

    # ── Run pipeline ───────────────────────────────────────────────────────────
    run_baseline_lstm(
        sequence_dir="data/sequences",
        models_dir="models",
        reports_dir="reports",
        figures_dir="reports/figures",
        seq_length=config["seq_length"],
        config=config,
    )

