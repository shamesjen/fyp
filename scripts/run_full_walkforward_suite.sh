#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN or create the virtualenv first." >&2
  exit 1
fi

run_step() {
  echo
  echo "==> $1"
  shift
  "$PYTHON_BIN" "$@"
}

if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
  run_step "Build two-year hourly walk-forward dataset" \
    scripts/build_daily_dataset.py \
    --config configs/data_spy_hourly_h1_walkforward_live.yaml
else
  echo
  echo "==> Skipping live data download/build because SKIP_DOWNLOAD=1"
fi

run_step "Build local dataset variant: seq_len=7 horizon=1" \
  scripts/build_dataset_from_local_raw.py \
  --seq-len 7 \
  --target-shift 1 \
  --tag spy_hourly_walkforward_seq7_h1

run_step "Build local dataset variant: seq_len=28 horizon=1" \
  scripts/build_dataset_from_local_raw.py \
  --seq-len 28 \
  --target-shift 1 \
  --tag spy_hourly_walkforward_seq28_h1

run_step "Build local dataset variant: seq_len=14 horizon=3" \
  scripts/build_dataset_from_local_raw.py \
  --seq-len 14 \
  --target-shift 3 \
  --tag spy_hourly_walkforward_seq14_h3

run_step "Build local dataset variant: seq_len=14 horizon=7" \
  scripts/build_dataset_from_local_raw.py \
  --seq-len 14 \
  --target-shift 7 \
  --tag spy_hourly_walkforward_seq14_h7

run_step "Run stitched walk-forward baselines for 1h horizon" \
  scripts/run_baselines_walkforward.py \
  --config configs/walkforward_baselines_hourly_h1_live.yaml \
  --dataset-path data/processed/spy_hourly_h1_walkforward_dataset_live.npz \
  --output-dir artifacts/walkforward_baselines/experiments \
  --tag h1_seq14 \
  --holding-period 1

run_step "Run stitched walk-forward baselines for 3h horizon" \
  scripts/run_baselines_walkforward.py \
  --config configs/walkforward_baselines_hourly_h1_live.yaml \
  --dataset-path data/processed/spy_hourly_walkforward_seq14_h3.npz \
  --output-dir artifacts/walkforward_baselines/experiments \
  --tag h3_seq14 \
  --holding-period 3

run_step "Run stitched walk-forward baselines for 7h horizon" \
  scripts/run_baselines_walkforward.py \
  --config configs/walkforward_baselines_hourly_h1_live.yaml \
  --dataset-path data/processed/spy_hourly_walkforward_seq14_h7.npz \
  --output-dir artifacts/walkforward_baselines/experiments \
  --tag h7_seq14 \
  --holding-period 7

run_step "Run base stitched walk-forward LSTM: seq14 h1" \
  scripts/run_lstm_walkforward.py \
  --config configs/walkforward_lstm_hourly_h1_live.yaml \
  --num-layers 2 \
  --hidden-size 128 \
  --tag l2_h128

run_step "Run seq-length ablation: seq7 h1" \
  scripts/run_lstm_walkforward.py \
  --config configs/walkforward_lstm_hourly_h1_live.yaml \
  --dataset-path data/processed/spy_hourly_walkforward_seq7_h1.npz \
  --output-dir artifacts/walkforward_experiments/seq_length \
  --num-layers 2 \
  --hidden-size 128 \
  --tag seq7_h1_l2_h128 \
  --holding-period 1

run_step "Run seq-length ablation: seq28 h1" \
  scripts/run_lstm_walkforward.py \
  --config configs/walkforward_lstm_hourly_h1_live.yaml \
  --dataset-path data/processed/spy_hourly_walkforward_seq28_h1.npz \
  --output-dir artifacts/walkforward_experiments/seq_length \
  --num-layers 2 \
  --hidden-size 128 \
  --tag seq28_h1_l2_h128 \
  --holding-period 1

run_step "Run horizon ablation: seq14 h3" \
  scripts/run_lstm_walkforward.py \
  --config configs/walkforward_lstm_hourly_h1_live.yaml \
  --dataset-path data/processed/spy_hourly_walkforward_seq14_h3.npz \
  --output-dir artifacts/walkforward_experiments/horizon \
  --num-layers 2 \
  --hidden-size 128 \
  --tag seq14_h3_l2_h128 \
  --holding-period 3

run_step "Run horizon ablation: seq14 h7" \
  scripts/run_lstm_walkforward.py \
  --config configs/walkforward_lstm_hourly_h1_live.yaml \
  --dataset-path data/processed/spy_hourly_walkforward_seq14_h7.npz \
  --output-dir artifacts/walkforward_experiments/horizon \
  --num-layers 2 \
  --hidden-size 128 \
  --tag seq14_h7_l2_h128 \
  --holding-period 7

run_step "Run research ablation: shape projection off" \
  scripts/run_lstm_walkforward.py \
  --config configs/walkforward_lstm_hourly_h1_live.yaml \
  --dataset-path data/processed/spy_hourly_h1_walkforward_dataset_live.npz \
  --output-dir artifacts/walkforward_experiments/research \
  --num-layers 2 \
  --hidden-size 128 \
  --tag seq14_h1_l2_h128_shapeoff \
  --holding-period 1 \
  --disable-shape-projection

run_step "Run research ablation: smoothness penalty off" \
  scripts/run_lstm_walkforward.py \
  --config configs/walkforward_lstm_hourly_h1_live.yaml \
  --dataset-path data/processed/spy_hourly_h1_walkforward_dataset_live.npz \
  --output-dir artifacts/walkforward_experiments/research \
  --num-layers 2 \
  --hidden-size 128 \
  --tag seq14_h1_l2_h128_smooth0 \
  --holding-period 1 \
  --smoothness-penalty 0.0

run_step "Generate aggregate report plots/tables" \
  - <<'PY'
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('artifacts')
report_dir = root / 'reports'
report_dir.mkdir(parents=True, exist_ok=True)

lstm_metrics_paths = {
    'h1_seq14_base': root/'walkforward/hourly_h1/l2_h128/walkforward_metrics.json',
    'h1_seq7': root/'walkforward_experiments/seq_length/seq7_h1_l2_h128/walkforward_metrics.json',
    'h1_seq28': root/'walkforward_experiments/seq_length/seq28_h1_l2_h128/walkforward_metrics.json',
    'h3_seq14': root/'walkforward_experiments/horizon/seq14_h3_l2_h128/walkforward_metrics.json',
    'h7_seq14': root/'walkforward_experiments/horizon/seq14_h7_l2_h128/walkforward_metrics.json',
    'h1_shapeoff': root/'walkforward_experiments/research/seq14_h1_l2_h128_shapeoff/walkforward_metrics.json',
    'h1_smooth0': root/'walkforward_experiments/research/seq14_h1_l2_h128_smooth0/walkforward_metrics.json',
}
rows = []
for name, path in lstm_metrics_paths.items():
    payload = json.loads(path.read_text())
    back = payload['backtest']
    test = payload['stitched_test']
    rows.append({
        'experiment': name,
        'rmse': test['rmse'],
        'mae': test['mae'],
        'r2': test['r2'],
        'dm_stat_vs_persistence': payload['dm_vs_persistence']['dm_stat'],
        'dm_p_value_vs_persistence': payload['dm_vs_persistence']['p_value'],
        'num_trades': back['num_trades'],
        'net_pnl': back['net_pnl'],
        'sharpe_annualized': back['sharpe_annualized'],
        'turnover': back['turnover'],
        'hit_rate': back['hit_rate'],
        'long_trades': back['long_trades'],
        'short_trades': back['short_trades'],
        'signal_realized_corr': back['signal_realized_corr'],
        'edge_sign_accuracy': back['edge_sign_accuracy'],
        'max_drawdown': back['max_drawdown'],
        'trade_pnl_skew': back['trade_pnl_skew'],
        'trade_pnl_kurtosis': back['trade_pnl_kurtosis'],
    })
lstm_df = pd.DataFrame(rows)
lstm_df.to_csv(report_dir/'lstm_ablation_summary.csv', index=False)

baseline_tags = {
    'h1_seq14': root/'walkforward_baselines/experiments/h1_seq14/baseline_walkforward_summary.csv',
    'h3_seq14': root/'walkforward_baselines/experiments/h3_seq14/baseline_walkforward_summary.csv',
    'h7_seq14': root/'walkforward_baselines/experiments/h7_seq14/baseline_walkforward_summary.csv',
}
all_baselines = []
best_rows = []
for tag, path in baseline_tags.items():
    frame = pd.read_csv(path)
    frame.insert(0, 'experiment', tag)
    all_baselines.append(frame)
    best_rows.append(frame.sort_values('test_rmse').iloc[0].to_dict())
all_baseline_df = pd.concat(all_baselines, ignore_index=True)
all_baseline_df.to_csv(report_dir/'baseline_walkforward_all.csv', index=False)
best_baseline_df = pd.DataFrame(best_rows)
best_baseline_df.to_csv(report_dir/'baseline_walkforward_best_by_horizon.csv', index=False)

plot_df = lstm_df.copy()
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].bar(plot_df['experiment'], plot_df['rmse'])
axes[0].set_title('LSTM RMSE')
axes[0].tick_params(axis='x', rotation=45)
axes[1].bar(plot_df['experiment'], plot_df['net_pnl'])
axes[1].set_title('LSTM Net PnL')
axes[1].tick_params(axis='x', rotation=45)
axes[2].bar(plot_df['experiment'], plot_df['sharpe_annualized'])
axes[2].set_title('LSTM Sharpe')
axes[2].tick_params(axis='x', rotation=45)
fig.tight_layout()
fig.savefig(report_dir/'lstm_ablation_metrics.png', dpi=150)
plt.close(fig)

compare_rows = []
for horizon in ['h1', 'h3', 'h7']:
    lstm_key = {'h1':'h1_seq14_base','h3':'h3_seq14','h7':'h7_seq14'}[horizon]
    lstm_row = lstm_df.loc[lstm_df['experiment'] == lstm_key].iloc[0]
    baseline_row = best_baseline_df.loc[best_baseline_df['experiment'] == f'{horizon}_seq14'].iloc[0]
    compare_rows.append({'horizon': horizon, 'strategy': 'lstm', 'rmse': lstm_row['rmse'], 'net_pnl': lstm_row['net_pnl'], 'sharpe': lstm_row['sharpe_annualized']})
    compare_rows.append({'horizon': horizon, 'strategy': str(baseline_row['model']), 'rmse': baseline_row['test_rmse'], 'net_pnl': baseline_row['net_pnl'], 'sharpe': baseline_row['sharpe_annualized']})
compare_df = pd.DataFrame(compare_rows)
compare_df.to_csv(report_dir/'best_lstm_vs_best_baseline_by_horizon.csv', index=False)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metric in zip(axes, ['rmse', 'net_pnl', 'sharpe']):
    pivot = compare_df.pivot(index='horizon', columns='strategy', values=metric)
    pivot.plot(kind='bar', ax=ax)
    ax.set_title(metric)
    ax.tick_params(axis='x', rotation=0)
fig.tight_layout()
fig.savefig(report_dir/'best_lstm_vs_best_baseline_by_horizon.png', dpi=150)
plt.close(fig)

selected_equities = {
    'h1_seq14_base': root/'walkforward/hourly_h1/l2_h128/backtest/backtest_trades.csv',
    'h1_seq7': root/'walkforward_experiments/seq_length/seq7_h1_l2_h128/backtest/backtest_trades.csv',
    'h1_seq28': root/'walkforward_experiments/seq_length/seq28_h1_l2_h128/backtest/backtest_trades.csv',
    'h3_seq14': root/'walkforward_experiments/horizon/seq14_h3_l2_h128/backtest/backtest_trades.csv',
    'h7_seq14': root/'walkforward_experiments/horizon/seq14_h7_l2_h128/backtest/backtest_trades.csv',
}
fig, ax = plt.subplots(figsize=(10, 5))
for name, path in selected_equities.items():
    frame = pd.read_csv(path, parse_dates=['date'])
    ax.plot(frame['date'], frame['cumulative_pnl'], label=name)
ax.set_title('LSTM Stitched Equity Curves')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative PnL')
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
fig.savefig(report_dir/'lstm_equity_curves.png', dpi=150)
plt.close(fig)

diagnostic_rows = []
lstm_map = {
    'h1_lstm': root/'walkforward/hourly_h1/l2_h128/walkforward_metrics.json',
    'h3_lstm': root/'walkforward_experiments/horizon/seq14_h3_l2_h128/walkforward_metrics.json',
    'h7_lstm': root/'walkforward_experiments/horizon/seq14_h7_l2_h128/walkforward_metrics.json',
}
baseline_map = {
    'h1_mlp_baseline': root/'walkforward_baselines/experiments/h1_seq14/mlp_baseline/backtest/backtest_summary.csv',
    'h3_mlp_baseline': root/'walkforward_baselines/experiments/h3_seq14/mlp_baseline/backtest/backtest_summary.csv',
    'h7_mlp_baseline': root/'walkforward_baselines/experiments/h7_seq14/mlp_baseline/backtest/backtest_summary.csv',
}
for label, path in lstm_map.items():
    payload = json.loads(path.read_text())
    diagnostic_rows.append({'label': label, 'strategy_type': 'lstm', **payload['backtest']})
for label, path in baseline_map.items():
    summary = pd.read_csv(path).iloc[0].to_dict()
    diagnostic_rows.append({'label': label, 'strategy_type': 'baseline', **summary})
diag_df = pd.DataFrame(diagnostic_rows)
diag_df.to_csv(report_dir/'strategy_diagnostics_summary.csv', index=False)

fig, axes = plt.subplots(3, 2, figsize=(14, 14))
for ax, col in zip(axes.flat, ['turnover', 'sharpe_annualized', 'hit_rate', 'trade_pnl_skew', 'trade_pnl_kurtosis', 'signal_realized_corr']):
    ax.bar(diag_df['label'], diag_df[col])
    ax.set_title(col)
    ax.tick_params(axis='x', rotation=45)
fig.tight_layout()
fig.savefig(report_dir/'strategy_diagnostics_metrics.png', dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(diag_df['label'], diag_df['long_trades'], label='long_trades')
ax.bar(diag_df['label'], diag_df['short_trades'], bottom=diag_df['long_trades'], label='short_trades')
ax.set_title('Long vs Short Trade Counts')
ax.tick_params(axis='x', rotation=45)
ax.legend()
fig.tight_layout()
fig.savefig(report_dir/'strategy_long_short_balance.png', dpi=150)
plt.close(fig)
PY

echo
echo "Suite complete."
echo "Reports:"
echo "  artifacts/reports/hourly_walkforward_research_ablation.md"
echo "  artifacts/reports/strategy_diagnostics.md"
