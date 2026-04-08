# Intraday Implied-Volatility Curve Forecasting for Options Mispricing Detection

Repository for the B.Comp. Final Year Project dissertation at the School of Computing, National University of Singapore.

GitHub repository: <https://github.com/shamesjen/fyp>

## What This Repository Contains

This repository contains:

- the final dissertation source: [thesis.tex](thesis.tex)
- the thesis figure/table asset pack: [thesis_report_assets/](thesis_report_assets/)
- the experiment code used to build datasets, train models, run walk-forward benchmarks, and generate reports
- the saved experiment artifacts and markdown reports under [artifacts/](artifacts/)

The project studies **intraday forecasting of the SPY implied-volatility smile** on a fixed seven-point moneyness grid in the `30 +/- 7 DTE` bucket. Forecasts are evaluated both statistically and economically under:

- overlap-safe walk-forward retraining
- standardized stitched out-of-sample comparison
- execution-aware backtesting
- threshold-swept strategy evaluation

## Final Thesis Result

The final thesis is centered on the **retuned 5-minute carry-forward benchmark**.

Main benchmark characteristics:

- underlying: `SPY`
- frequency: `5-minute`
- target: fixed seven-node IV curve
- maturity bucket: `30 +/- 7` calendar days to expiry
- lookback: `seq_len = 12`
- horizon: `1` bar ahead
- dataset size: `50,019` supervised samples
- validation design: `15` expanding walk-forward folds with overlap-safe frontier stitching

Main findings used in the thesis:

- **Best statistical model:** `ElasticNet`
  - RMSE: `0.010886`
- **Best tuned economic model:** `xLSTM b2/e256`
  - tuned net PnL: `0.121296`
  - tuned Sharpe: `3.876124`
- **Important control result:** the transformer encoder is statistically strong
  - RMSE: `0.013660`
  - tuned net PnL: `0.103707`
  - but it remains economically weaker than the xLSTM

The final interpretation is deliberately narrow:

- classical baselines remain very strong
- xLSTM leads economically on the completed benchmark
- the economic lead is **real but narrow**
- statistical accuracy and monetizable signal quality diverge materially on this problem

## Where To Start

If you only need the submission materials, start here:

- thesis source: [thesis.tex](thesis.tex)
- thesis assets: [thesis_report_assets/](thesis_report_assets/)
- final retuned carry benchmark report: [artifacts/reports/carry_model_family_retuned_oldbudget.md](artifacts/reports/carry_model_family_retuned_oldbudget.md)
- threshold-swept economic comparison: [artifacts/reports/carry_model_family_retuned_oldbudget_threshold_sweep/threshold_sweep_summary.md](artifacts/reports/carry_model_family_retuned_oldbudget_threshold_sweep/threshold_sweep_summary.md)
- best xLSTM vs all competitors appendix pack: [artifacts/reports/carry_model_family_retuned_oldbudget_best_xlstm_vs_all_baselines_thresholded.md](artifacts/reports/carry_model_family_retuned_oldbudget_best_xlstm_vs_all_baselines_thresholded.md)
- bootstrap significance report: [artifacts/reports/carry_model_family_retuned_oldbudget_bootstrap_significance.md](artifacts/reports/carry_model_family_retuned_oldbudget_bootstrap_significance.md)

The repo also contains earlier daily, hourly, and pre-retune 5-minute experiments. Those are preserved for traceability, but the thesis argument is built around the retuned carry benchmark above.

## Repository Layout

```text
README.md
requirements.txt
.env.example
src/                    Core data, model, training, and evaluation modules
scripts/                Executable experiment runners and reporting scripts
configs/                YAML configs for datasets, training runs, and reports
data/                   Raw, processed, and sample data artifacts
artifacts/              Saved experiment outputs and markdown reports
thesis_report_assets/   Figures, tables, and asset manifests for the dissertation
thesis.tex              Final dissertation source
```

Useful navigation files:

- [artifacts/README.md](artifacts/README.md)
- [configs/README.md](configs/README.md)
- [scripts/README.md](scripts/README.md)

## Environment Setup

Create the virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want to rebuild datasets from raw Alpaca/yfinance inputs, create `.env` first:

```bash
cp .env.example .env
```

and provide:

```bash
ALPACA_KEY=your_key_here
ALPACA_SECRET=your_secret_here
```

Most saved thesis artifacts can be inspected without credentials. Credentials are only needed for raw-market-data rebuilds.

## Thesis-Facing Reproducibility Path

The commands below reproduce the main thesis benchmark and its final robustness layers.

### 1. Run the retuned carry benchmark

```bash
.venv/bin/python scripts/run_multiseed_model_family_benchmark.py --config configs/neural_retune_carry_oldbudget.yaml --skip-existing
```

This runs:

- 10 baselines
- vanilla LSTM family
- attention-pooled LSTM
- transformer encoder control
- xLSTM family

on the carry-forward `5-minute seq12/h1` benchmark.

Main report output:

- [artifacts/reports/carry_model_family_retuned_oldbudget.md](artifacts/reports/carry_model_family_retuned_oldbudget.md)

### 2. Run the threshold sweep

```bash
.venv/bin/python scripts/run_model_family_threshold_sweep.py \
  --benchmark-root artifacts/experiments/carry_model_family_retuned_oldbudget \
  --backtest-config configs/backtest_execution_5min.yaml \
  --output-root artifacts/reports/carry_model_family_retuned_oldbudget_threshold_sweep \
  --report-title "Retuned Carry Threshold Sweep"
```

Key output:

- [artifacts/reports/carry_model_family_retuned_oldbudget_threshold_sweep/threshold_sweep_summary.md](artifacts/reports/carry_model_family_retuned_oldbudget_threshold_sweep/threshold_sweep_summary.md)

### 3. Run the final winner-vs-all appendix pack

```bash
.venv/bin/python scripts/run_best_model_vs_all_baselines_evaluations.py --config configs/best_model_vs_all_baselines_evaluations_carry_retuned_thresholded.yaml
```

This evaluates the final winner `xlstm_oldb_b2_e256` against:

- all classical baselines
- the transformer control
- the attention-pooled LSTM control

on the same standardized prediction path, using each model’s selected threshold from the completed threshold sweep.

Key output:

- [artifacts/reports/carry_model_family_retuned_oldbudget_best_xlstm_vs_all_baselines_thresholded.md](artifacts/reports/carry_model_family_retuned_oldbudget_best_xlstm_vs_all_baselines_thresholded.md)

### 4. Run bootstrap significance for the economic frontier

```bash
.venv/bin/python scripts/run_block_bootstrap_strategy_significance.py --config configs/carry_retuned_bootstrap_significance.yaml
```

Key output:

- [artifacts/reports/carry_model_family_retuned_oldbudget_bootstrap_significance.md](artifacts/reports/carry_model_family_retuned_oldbudget_bootstrap_significance.md)

### 5. Rebuild thesis assets

```bash
.venv/bin/python scripts/build_thesis_report_assets.py
```

This refreshes the thesis asset pack under [thesis_report_assets/](thesis_report_assets/).

## Supporting SVI Sensitivity

The dissertation also includes a **carry-plus-SVI sensitivity**. This is a data-layer sensitivity, not the main final benchmark.

To rebuild the SVI-backed dataset from local market artifacts:

```bash
.venv/bin/python scripts/build_svi_dataset_from_local_market_artifacts.py --config configs/data_spy_5min_walkforward_svi_carry.yaml
```

Relevant reports:

- [artifacts/reports/svi_full_experiment_suite_summary.md](artifacts/reports/svi_full_experiment_suite_summary.md)
- [artifacts/reports/carry_vs_svi_model_family_comparison.md](artifacts/reports/carry_vs_svi_model_family_comparison.md)

Main conclusion from the SVI sensitivity:

- SVI improves the structural fit layer
- it does **not** overturn the downstream model ranking on this dataset

## Key Files for the Dissertation

Primary thesis files:

- [thesis.tex](thesis.tex)
- [thesis_report_assets/figures/](thesis_report_assets/figures/)
- [thesis_report_assets/tables/](thesis_report_assets/tables/)
- [thesis_report_assets/summaries/README_asset_manifest.md](thesis_report_assets/summaries/README_asset_manifest.md)

Core reports cited by the thesis:

- [artifacts/reports/carry_model_family_retuned_oldbudget.md](artifacts/reports/carry_model_family_retuned_oldbudget.md)
- [artifacts/reports/carry_model_family_retuned_oldbudget_best_xlstm_vs_all_baselines_thresholded.md](artifacts/reports/carry_model_family_retuned_oldbudget_best_xlstm_vs_all_baselines_thresholded.md)
- [artifacts/reports/carry_model_family_retuned_oldbudget_bootstrap_significance.md](artifacts/reports/carry_model_family_retuned_oldbudget_bootstrap_significance.md)
- [artifacts/reports/carry_vs_svi_model_family_comparison.md](artifacts/reports/carry_vs_svi_model_family_comparison.md)

## Data Notes

The repo contains:

- processed datasets used by the thesis under [data/processed/](data/processed/)
- raw and intermediate local-market artifacts under [data/raw/](data/raw/) where available
- sample fallback data under [data/samples/](data/samples/)

The final thesis benchmark relies on the processed carry dataset:

- `data/processed/spy_5min_walkforward_h1_dataset_carry_local.npz`

and the SVI sensitivity relies on:

- `data/processed/spy_5min_walkforward_h1_dataset_svi_carry_local.npz`

## Model Families Included

Final thesis benchmark families:

- baselines:
  - persistence
  - AR(1) per grid
  - factor AR/VAR
  - GARCH-style ATM shift
  - ElasticNet
  - MLP
  - ExtraTrees
  - histogram gradient boosting
  - HAR factor
  - smile-coefficient baseline
- recurrent controls:
  - vanilla LSTM
  - attention-pooled LSTM
- non-recurrent neural control:
  - transformer encoder
- proposed family:
  - xLSTM

## Notes on Interpretation

This repository contains many experiment layers accumulated during the project. The thesis does **not** treat them all equally.

For grading and submission purposes:

- use [thesis.tex](thesis.tex) as the authoritative report source
- use the retuned carry benchmark and its attached robustness reports as the primary evidence base
- treat earlier daily/hourly/pre-retune experiments as supporting development history, not as the final empirical claim

## Limitations of the Repository

- The economic backtest is execution-aware, but it is still a stylized benchmark rather than a full contract-level order-book simulator.
- Intraday listed-option data remain sparse even after carry-forward.
- Some older reports and organized views remain in the repo for traceability and are not all thesis-final.
- Local LaTeX compilation is not configured in this repository; compile [thesis.tex](thesis.tex) in your TeX environment.

## License / Submission Note

This repository is included as part of the FYP submission package and is intended to document:

- code used in the dissertation
- saved experiment artifacts
- thesis figures and tables
- reproducibility entry points for the final benchmark
