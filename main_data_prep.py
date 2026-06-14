"""
main_data_prep.py
=================
Orchestrator for the menstrual cycle forecasting data pipeline.

Usage:
    python main_data_prep.py
"""

from src.daily_dataset_builder import run_daily_dataset_builder
from src.preprocessing import run_preprocessing
from src.sequence_builder import run_sequence_builder

if __name__ == "__main__":
    # Step 1: Build daily-level dataset from raw cycle-level data
    run_daily_dataset_builder(
        raw_path="data/raw/menstrual_cycle_dataset.csv",
        processed_dir="data/processed",
    )

    # Step 2: User-level split and train-fitted MinMax normalization
    run_preprocessing(
        daily_data_path="data/processed/daily_data.csv",
        feature_info_path="data/processed/feature_info.json",
        processed_dir="data/processed",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    # Step 3: Build sliding-window LSTM sequences (seq14 and seq21)
    run_sequence_builder(
        train_path="data/processed/train_daily.csv",
        val_path="data/processed/val_daily.csv",
        test_path="data/processed/test_daily.csv",
        feature_info_path="data/processed/feature_info.json",
        output_dir="data/sequences",
        seq_length=14,
    )

    run_sequence_builder(
        train_path="data/processed/train_daily.csv",
        val_path="data/processed/val_daily.csv",
        test_path="data/processed/test_daily.csv",
        feature_info_path="data/processed/feature_info.json",
        output_dir="data/sequences",
        seq_length=21,
    )
