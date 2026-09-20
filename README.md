# FYP Smoothing Filter Experiment

This prototype performs the complete offline smoothing-filter experiment for the
CO2 absorption/desorption SCADA signals.

## Pipeline

1. Load one or more SCADA CSV files.
2. Remove obvious all-zero startup rows and regularize timestamps.
3. Generate a data-informed clean reference for temperature, pressure, pH and CO2
   using robust local-median anchors + PCHIP interpolation.
4. Estimate real-data noise characteristics from residuals.
5. Generate controlled low/normal/high artificial noise using reproducible seeds.
6. Tune EMA, causal Savitzky-Golay, causal weighted Savitzky-Golay and 1-D Kalman.
7. Calculate RMSE, input/output SNR, SNR improvement, transition RMSE and stable RMSE.
8. Select the best hyperparameters inside each filter family.
9. Re-test those optimized filters using held-out artificial-noise seeds.
10. Select the best overall filter independently for each variable.
11. Automatically generate Matplotlib/Seaborn research plots.

## Install

```bash
pip install -r requirements.txt
```

## Run using the data folder

```bash
python run_experiment.py --data data --output results
```

You can also point directly to one CSV:

```bash
python run_experiment.py --data desorption_data_30072026.csv --output results
```

For faster debugging without figures:

```bash
python run_experiment.py --data data --output results --no-plots
```

When more reactor cycles are collected, simply place the extra CSV files in the
same `data/` folder. No source-code changes are required.

## Result CSV files

- `results/clean_references/`
  - raw and automatically generated clean reference for each variable/run.
- `results/estimated_noise_profiles.csv`
  - estimated sigma, lag-1 correlation, spike statistics and quantization.
- `results/all_tuning_results.csv`
  - every tuning trial.
- `results/candidate_summary.csv`
  - mean/std performance of each hyperparameter configuration.
- `results/best_per_filter.csv`
  - optimized hyperparameters within EMA, SG, weighted SG and Kalman.
- `results/validation_results.csv`
  - held-out synthetic validation trials.
- `results/best_per_variable.csv`
  - final selected filter for temperature, pressure, pH and CO2.

## Automatically generated figures

The figures are stored under `results/plots/`.

### 01_clean_references

Measured SCADA signal, automatic PCHIP clean reference, and robust anchor points.

**Purpose:** audit whether the pseudo-ground-truth clean reference is reasonable
before trusting SNR/RMSE results.

### 02_noise_residuals

Histogram + KDE of:

```text
residual = measured SCADA - clean reference
```

**Purpose:** inspect the estimated measurement-noise distribution and large tails.

### 03_hyperparameter_search

- EMA: selection score vs span.
- Kalman: selection score vs Q/R on a log scale.
- Causal SG: heatmap of window length vs polynomial order.
- Weighted causal SG: one heatmap per polynomial order showing window length vs
  oldest-sample weight ratio.

**Purpose:** show that hyperparameters were selected systematically rather than by
visual trial and error.

### 04_filter_comparison

Held-out validation bar plots for:

- RMSE
- SNR improvement
- transition-region RMSE
- stable-region RMSE

Error bars are standard deviations across repeated validation trials.

**Purpose:** compare the optimized version of each filter family.

### 05_noise_robustness

Performance vs artificial-noise severity (0.5x, 1x and 2x by default).

**Purpose:** check whether a selected filter only performs well at one noise level.

### 06_selected_filter_examples

Overlay of:

- clean reference
- artificial noisy signal
- final selected filter output

The example uses a held-out validation seed and nominal 1x noise.

**Purpose:** visual sanity check after quantitative filter selection. This plot is
not used to choose the winner.

## Plotting implementation

All plotting code is in:

```text
smoothing/visualization.py
```

The module is deliberately separate from:

- filtering,
- synthetic-reference generation,
- noise generation,
- metric calculation,
- optimization.

This keeps the research visualization layer independent from the filter classes
that can later be integrated into the real-time PyQt/SCADA system.
