# Implied-Volatility Curve Forecasting MVP

This repo is a practical first implementation of the FYP project described in the CA report: forecast the next-step implied-volatility curve on a fixed moneyness grid, compare forecast IV against current market IV, and translate that IV edge into a simple vega-style mispricing signal. The report targets intraday, execution-aware research; this MVP intentionally starts with a cleaner daily SPY setup so the full pipeline is runnable end to end before adding intraday data, richer microstructure features, and more advanced curve constraints.

## Why This MVP Looks Like This

- `SPY` is used first because it is highly liquid, operationally simpler than index options, and a practical starting point for data and model debugging.
- `Alpaca` is the default options provider in code because it is one of the more accessible free options APIs.
- `yfinance` is used only for underlying OHLCV data, not options data.
- The repo includes a prepared sample IV panel so the full demo still runs when Alpaca historical coverage is insufficient or credentials/network are unavailable.

## Research Objective

The project framing follows the CA report:

- Target: next-step IV curve on a fixed moneyness grid for one chosen maturity bucket.
- Compare forecast IV vs current IV to estimate mispricing.
- Convert IV edge into a simple price-edge signal with vega-style logic.
- Benchmark against persistence, AR(1), factor VAR, GARCH-style ATM shift, MLP, and a simple LSTM.

The report also motivates later upgrades that are only stubbed or left as TODOs here:

- Intraday snapshots instead of daily data
- SVI smoothing / no-arbitrage constraints
- Execution-aware filtering with depth, spreads, and latency
- Multi-maturity modeling
- Provider upgrades beyond Alpaca

## Architecture

The modeling code depends on a fixed IV panel interface, not on a vendor-specific raw schema.

```text
yfinance underlying data  ----\
                               -> daily panel builder -> feature engineering -> sequence dataset -> models/eval/backtest
Alpaca options history   ----/
prepared CSV/parquet panel --/
```

Provider abstraction:

- `src/data/yfinance_underlying.py`: underlying OHLCV provider
- `src/data/alpaca_options.py`: default options provider
- `src/data/csv_panel_loader.py`: prepared-panel fallback
- `src/data/daily_panel_builder.py`: chain-level rows to fixed-grid IV curves

The rest of the pipeline consumes the same processed dataset regardless of where the IV panel came from.

## Repo Navigation

The original experiment files are preserved in place, but the repo now also includes grouped navigation layers:

- [artifacts/README.md](artifacts/README.md): canonical experiment outputs grouped by research phase
- [configs/README.md](configs/README.md): configs grouped by stage
- [scripts/README.md](scripts/README.md): scripts grouped by purpose

If you only want the final thesis result set, start in:

- `artifacts/current_final/`

## Repo Layout

```text
README.md
requirements.txt
.env.example
configs/
data/
notebooks/
src/
scripts/
tests/
```

Outputs are written under `artifacts/`.

## Setup

1. Create the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create `.env` in the repo root:

```bash
cp .env.example .env
```

3. Put your Alpaca credentials in `.env` using these exact names:

```bash
ALPACA_KEY=your_key_here
ALPACA_SECRET=your_secret_here
```

`.env` is already ignored by git. The Alpaca client automatically reads `ALPACA_KEY` and `ALPACA_SECRET` through `python-dotenv`. If either variable is missing, the provider raises a clear error and the build script can fall back to the bundled sample panel if configured.

## End-to-End Demo

Offline-safe demo path:

```bash
python scripts/build_daily_dataset.py --config configs/data_spy_daily.yaml
python scripts/run_baselines.py --config configs/train_baselines.yaml
python scripts/run_lstm.py --config configs/train_lstm.yaml
python scripts/run_backtest_demo.py --config configs/backtest_demo.yaml
```

This works even without live Alpaca history because `configs/data_spy_daily.yaml` is set to:

- try `yfinance` first for underlying data
- try `Alpaca` first for options data
- fall back to `data/samples/spy_underlying_sample.csv`
- fall back to `data/samples/spy_iv_panel_sample.csv`

Real-data path:

```bash
python scripts/download_underlying_data.py --config configs/data_spy_daily.yaml
python scripts/download_alpaca_options.py --config configs/data_spy_daily.yaml
python scripts/build_daily_dataset.py --config configs/data_spy_daily.yaml
```

If Alpaca historical options coverage is not deep enough for the window you want, replace the fallback path with your own prepared CSV/parquet IV panel and keep the training configs unchanged.

Strict live-data path:

```bash
python scripts/build_daily_dataset.py --config configs/data_spy_daily_live.yaml
python scripts/run_baselines.py --config configs/train_baselines_live.yaml
python scripts/run_lstm.py --config configs/train_lstm_live.yaml
python scripts/run_backtest_demo.py --config configs/backtest_demo_live.yaml
python scripts/print_run_summary.py --preset live
```

The live config disables CSV fallback and uses Alpaca historical contracts/bars plus yfinance underlying data.

Hourly live path:

```bash
python scripts/build_daily_dataset.py --config configs/data_spy_hourly_live.yaml
python scripts/run_baselines.py --config configs/train_baselines_hourly_live.yaml
python scripts/run_lstm.py --config configs/train_lstm_hourly_live.yaml
python scripts/run_backtest_demo.py --config configs/backtest_demo_hourly_live.yaml
python scripts/print_run_summary.py --preset hourly-live
```

The hourly live config uses SPY `60m` underlying bars from yfinance and `1Hour` option bars from Alpaca on a shorter practical live window so the free-data path remains runnable.

Hourly horizon-1 path:

```bash
python scripts/build_daily_dataset.py --config configs/data_spy_hourly_h1_live.yaml
python scripts/run_baselines.py --config configs/train_baselines_hourly_h1_live.yaml
python scripts/run_lstm.py --config configs/train_lstm_hourly_h1_live.yaml
python scripts/run_backtest_demo.py --config configs/backtest_demo_hourly_h1_live.yaml
python scripts/print_run_summary.py --preset hourly-h1-live
```

This variant uses a rolling window of `10` hourly IV observations to predict the immediate next hourly IV curve.

Hourly next-day-close path:

```bash
python scripts/build_daily_dataset.py --config configs/data_spy_hourly_nextday_live.yaml
python scripts/run_baselines.py --config configs/train_baselines_hourly_nextday_live.yaml
python scripts/run_lstm.py --config configs/train_lstm_hourly_nextday_live.yaml
python scripts/run_backtest_demo.py --config configs/backtest_demo_hourly_nextday_live.yaml
python scripts/print_run_summary.py --preset hourly-nextday-live
```

This variant uses a rolling window of `10` hourly IV observations to predict the curve `7` hourly bars ahead, so samples slide one hour at a time instead of only targeting end-of-day points.

Prepared-panel mode:

- set `providers.options.type: csv` in [`configs/data_spy_daily.yaml`](/Users/james/Documents/NUS/FYP/code/configs/data_spy_daily.yaml)
- point `paths.prepared_panel_path` at your CSV/parquet file
- rerun `scripts/build_daily_dataset.py`

## Expected Prepared Panel Schema

The CSV/parquet fallback is expected to look like:

```text
date, underlying, dte_bucket, iv_mny_m0p15, ..., iv_mny_0p0, ..., iv_mny_0p15
```

Notes:

- `date` is the observation date
- `underlying` is the ticker, such as `SPY`
- `dte_bucket` is the maturity bucket in days
- each `iv_mny_*` column is the IV value on one fixed moneyness node

The model code does not change if this panel came from Alpaca, another vendor, or an offline file.

## Current Modeling Choices

Baselines:

- `persistence`: random-walk IV curve
- `ar1_per_grid`: separate AR(1) at each moneyness bucket
- `factor_ar_var`: PCA factors of the curve, then AR/VAR-style forecasting in factor space
- `garch_baseline`: GARCH-style ATM shift baseline that pushes the full curve with an ATM-centered forecast
- `mlp_baseline`: flattened-window MLP
- `lstm_curve`: simple 1-layer LSTM with a linear head

Current default sequence convention:

- `X`: `[batch, seq_len, features]`
- `y`: `[batch, grid_size]`

Features in the daily MVP:

- lagged IV curves via the sequence window itself
- lagged ATM IV
- lagged underlying returns
- rolling realized-vol proxy from underlying returns
- DTE bucket indicator

## Evaluation

Implemented:

- RMSE, MAE, R² overall
- metrics by moneyness bucket
- Diebold-Mariano utility for forecast comparison
- curve examples
- error-by-bucket plots
- LSTM training-curve plot
- toy economic backtest using forecast IV vs current IV, a vega proxy, thresholded signals, and transaction-cost placeholders

## Known Limitations

- The CA report is intraday-first; this repo is daily-first for a stable MVP.
- Alpaca free historical options coverage is limited, especially for deep history and historical IV reconstruction.
- Historical IV is reconstructed from option bars with a Black-Scholes-style approximation when direct IV is unavailable.
- No SVI smoothing is applied yet; the current builder uses interpolation on the fixed moneyness grid.
- The toy backtest is a signal demo, not an execution-quality simulator.
- No full no-arbitrage enforcement is implemented yet; only lightweight structural hooks exist in the loss code.

## What To Upgrade Next

- Swap the fixed daily builder for intraday event-time snapshots
- Add SVI or another arbitrage-aware smile fitter
- Replace the simplified GARCH-style baseline with richer volatility models if needed
- Add vega-weighted loss and stronger shape regularization in training
- Extend from one maturity bucket to multi-bucket or full surface forecasting
- Add a better options provider with deeper historical chain coverage

## Notebooks and Tests

Notebooks:

- `notebooks/01_data_inspection.ipynb`
- `notebooks/02_baseline_results.ipynb`
- `notebooks/03_lstm_results.ipynb`

Tests:

```bash
python -m unittest discover -s tests
```
# fyp
