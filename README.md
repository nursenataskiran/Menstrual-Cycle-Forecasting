# Menstrual Cycle Forecasting

This repository contains the code, report, and experiment outputs for the HaloScape menstrual cycle forecasting case study.

## Project Overview

The goal of this project is to build a sequence-based forecasting pipeline for menstrual cycle data. The model uses historical cycle-level information expanded into daily records and transformed into fixed-length time-window sequences.

The final model is a multi-output LSTM with a shared sequence encoder and two task-specific heads:

- **Regression output:** predicts days until the next cycle
- **Classification output:** predicts the menstrual phase for each day in the sequence

The final selected configuration is:

```text
seq21_tanh_glorot_adam_bs32_dropout02
```

## Repository Structure

```text
.
├── data/
│   └── raw/
│       └── menstrual_cycle_dataset.csv
├── reports/
│   ├── rapor.pdf
│   ├── final_model/
│   │   ├── metrics, training history, and evaluation plots for the selected final model
│   └── experiments/
│       └── saved training histories and evaluation metrics for optimization experiments
├── src/
│   ├── baseline_lstm.py
│   ├── daily_dataset_builder.py
│   ├── preprocessing.py
│   ├── sequence_builder.py
│   └── sequence_error_analysis.py
├── eda_notebook.ipynb
├── main_data_prep.py
├── main_train_baseline.py
└── requirements.txt
```

## Report

The full case report is provided in:

```text
reports/rapor.pdf
```

The report includes the problem definition, EDA findings, preprocessing strategy, leakage prevention decisions, sequence construction, model architecture, experimental results, error analysis, limitations, and future work.

## Methodology

### 1. Daily Dataset Construction

The original dataset is cycle-level. Each cycle is expanded into daily records using the cycle start date, cycle length, period length, and next cycle start date.

The daily dataset includes:

- `day_in_cycle`
- `days_until_next_cycle`
- rule-based `phase_label`
- historical cycle features based only on previous cycles

To reduce leakage risk, current-cycle `Cycle Length` and `Period Length` are not used directly as model inputs.

### 2. Preprocessing

The preprocessing pipeline applies:

- user-level train/validation/test split
- categorical encoding
- train-fitted MinMax scaling
- historical feature construction
- leakage-safe feature selection

The split is performed at the user level to avoid having records from the same user in both training and evaluation sets.

### 3. Sequence Construction

Daily records are converted into sliding-window sequences within the same user and cycle. The final model uses a 21-day sequence length.

The generated inputs have the following structure:

```text
X:       (n_sequences, sequence_length, n_features)
y_phase: (n_sequences, sequence_length)
y_reg:   (n_sequences, sequence_length, 1)
```

### 4. Model

The model is a shared-encoder multi-task LSTM. A single LSTM layer learns temporal representations, followed by two task-specific output heads:

- softmax classification head for phase prediction
- linear regression head for days-until-next-cycle prediction

The final configuration uses:

```text
Sequence length: 21
LSTM units: 64
Dense units: 32
Dropout: 0.2
Dense activation: tanh
Kernel initializer: glorot_uniform
Optimizer: Adam
Batch size: 32
```

## How to Run

Install dependencies:

```bash
pip install joblib matplotlib numpy pandas scikit-learn seaborn tensorflow
```

Alternatively:

```bash
pip install -r requirements.txt
```

Run the data preparation pipeline:

```bash
python main_data_prep.py
```

This step builds the daily dataset, applies preprocessing, and generates both 14-day and 21-day sequence windows.

Train the final selected model:

```bash
python main_train_baseline.py
```

The training script uses the final selected configuration by default:

```text
seq21_tanh_glorot_adam_bs32_dropout02
```

## Running Alternative Experiments

The report compares multiple experimental configurations, including different sequence lengths, dense activations, optimizers, initializers, and batch sizes.

To reproduce or extend these experiments, the configuration values in `main_train_baseline.py` can be changed, for example:

- `seq_length`
- `dense_activation`
- `kernel_initializer`
- `optimizer_name`
- `batch_size`
- `dropout_rate`

After changing the configuration, run:

```bash
python main_train_baseline.py
```

The generated metrics and plots can then be compared with the final model outputs reported in `reports/final_model/`.

## Final Model Outputs

The final model metrics and plots are stored under:

```text
reports/final_model/
```

This folder includes:

- training history
- final validation/test metrics
- phase accuracy curve
- regression MAE curve
- total loss curve
- validation confusion matrix
- test confusion matrix

## Notes

This project uses a synthetic and clean menstrual cycle dataset. Therefore, real-world issues such as missed logging, irregular adherence behavior, hormone measurements, wearable sensor signals, and uncertain ovulation timing are not explicitly modeled.

The detailed discussion of model limitations and possible future improvements is included in the report.
