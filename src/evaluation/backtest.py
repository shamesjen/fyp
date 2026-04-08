from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import ensure_parent


def _annualization_factor(dates: pd.Series) -> float:
    ordered = pd.to_datetime(dates).sort_values()
    if len(ordered) < 2:
        return 0.0
    span_days = max((ordered.iloc[-1] - ordered.iloc[0]).total_seconds() / 86400.0, 1e-9)
    return max(len(ordered) / span_days * 365.0, 0.0)


def vega_proxy(
    current_curve: np.ndarray,
    moneyness_grid: list[float],
    maturity_bucket_days: int,
) -> np.ndarray:
    grid = np.asarray(moneyness_grid, dtype=float)
    time_scale = np.sqrt(max(maturity_bucket_days, 1) / 365.0)
    shape = np.exp(-5.0 * np.abs(grid))
    return (0.5 + current_curve) * time_scale * shape


def build_prediction_frame(
    dates: np.ndarray,
    current_curve: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    curve_columns: list[str],
) -> pd.DataFrame:
    frame = pd.DataFrame({"date": pd.to_datetime(dates)})
    for idx, column in enumerate(curve_columns):
        frame[f"current_{column}"] = current_curve[:, idx]
        frame[f"actual_{column}"] = y_true[:, idx]
        frame[f"pred_{column}"] = y_pred[:, idx]
    return frame


def _resolve_execution_config(execution: dict[str, Any] | None) -> dict[str, float | int | None]:
    cfg = execution or {}
    return {
        "per_trade_exposure": float(cfg.get("per_trade_exposure", 1.0)),
        "min_trade_exposure": float(cfg.get("min_trade_exposure", 0.0)),
        "max_concurrent_positions": None
        if cfg.get("max_concurrent_positions") is None
        else int(cfg.get("max_concurrent_positions")),
        "gross_exposure_cap": None
        if cfg.get("gross_exposure_cap") is None
        else float(cfg.get("gross_exposure_cap")),
        "net_exposure_cap": None
        if cfg.get("net_exposure_cap") is None
        else float(cfg.get("net_exposure_cap")),
        "commission_bps_per_side": float(cfg.get("commission_bps_per_side", 0.0)),
        "half_spread_bps_per_side": float(cfg.get("half_spread_bps_per_side", 0.0)),
        "slippage_bps_per_side": float(cfg.get("slippage_bps_per_side", 0.0)),
        "impact_bps_per_side": float(cfg.get("impact_bps_per_side", 0.0)),
    }


def _execution_round_trip_cost_bps(
    transaction_cost_bps: float,
    execution_cfg: dict[str, float | int | None],
) -> float:
    per_side = (
        float(execution_cfg["commission_bps_per_side"])
        + float(execution_cfg["half_spread_bps_per_side"])
        + float(execution_cfg["slippage_bps_per_side"])
        + float(execution_cfg["impact_bps_per_side"])
    )
    return float(transaction_cost_bps) + 2.0 * per_side


def _infer_entry_dates(exit_dates: pd.Series, holding_period_bars: int) -> pd.Series:
    entry_dates = exit_dates.shift(int(holding_period_bars))
    deltas = exit_dates.diff().dropna()
    median_delta = deltas.median() if not deltas.empty else pd.Timedelta(0)
    if median_delta <= pd.Timedelta(0):
        return entry_dates
    for idx in range(min(int(holding_period_bars), len(exit_dates))):
        steps = int(holding_period_bars) - idx
        entry_dates.iloc[idx] = exit_dates.iloc[idx] - steps * median_delta
    return entry_dates


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    max_duration = 0
    current_duration = 0
    for value in drawdown.to_numpy(dtype=float):
        if value < 0:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    return int(max_duration)


def run_backtest(
    prediction_frame: pd.DataFrame,
    curve_columns: list[str],
    moneyness_grid: list[float],
    maturity_bucket_days: int,
    signal_threshold: float = 0.0005,
    transaction_cost_bps: float = 1.0,
    holding_period_bars: int = 1,
    allow_overlapping_positions: bool = True,
    execution: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    current = prediction_frame[[f"current_{column}" for column in curve_columns]].to_numpy(dtype=float)
    actual = prediction_frame[[f"actual_{column}" for column in curve_columns]].to_numpy(dtype=float)
    predicted = prediction_frame[[f"pred_{column}" for column in curve_columns]].to_numpy(dtype=float)

    vega = vega_proxy(current, moneyness_grid, maturity_bucket_days)
    forecast_edge = (predicted - current) * vega
    realized_edge = (actual - current) * vega
    signal_score = forecast_edge.mean(axis=1)
    realized_score = realized_edge.mean(axis=1)
    raw_signal = np.where(signal_score > signal_threshold, 1, np.where(signal_score < -signal_threshold, -1, 0))

    hold = max(1, int(holding_period_bars))
    execution_cfg = _resolve_execution_config(execution)
    round_trip_cost_bps = _execution_round_trip_cost_bps(transaction_cost_bps, execution_cfg)

    requested_weight = raw_signal.astype(float) * float(execution_cfg["per_trade_exposure"])
    position_weight = np.zeros(len(raw_signal), dtype=float)
    blocked_by_overlap = np.zeros(len(raw_signal), dtype=bool)
    blocked_by_max_positions = np.zeros(len(raw_signal), dtype=bool)
    blocked_by_gross_cap = np.zeros(len(raw_signal), dtype=bool)
    blocked_by_net_cap = np.zeros(len(raw_signal), dtype=bool)
    active_positions_before = np.zeros(len(raw_signal), dtype=int)
    gross_exposure_before = np.zeros(len(raw_signal), dtype=float)
    net_exposure_before = np.zeros(len(raw_signal), dtype=float)

    for idx, raw in enumerate(raw_signal):
        window_start = max(0, idx - hold + 1)
        open_weights = position_weight[window_start:idx]
        open_count = int(np.count_nonzero(open_weights))
        gross_before = float(np.abs(open_weights).sum())
        net_before = float(open_weights.sum())

        active_positions_before[idx] = open_count
        gross_exposure_before[idx] = gross_before
        net_exposure_before[idx] = net_before

        if raw == 0:
            continue
        if not allow_overlapping_positions and open_count > 0:
            blocked_by_overlap[idx] = True
            continue
        max_positions = execution_cfg["max_concurrent_positions"]
        if max_positions is not None and open_count >= int(max_positions):
            blocked_by_overlap[idx] = False
            blocked_by_max_positions[idx] = True
            continue

        sign = float(np.sign(raw))
        allowed_abs = float(abs(requested_weight[idx]))

        gross_cap = execution_cfg["gross_exposure_cap"]
        if gross_cap is not None:
            gross_room = max(float(gross_cap) - gross_before, 0.0)
            if gross_room <= 0:
                blocked_by_overlap[idx] = False
                blocked_by_gross_cap[idx] = True
                continue
            allowed_abs = min(allowed_abs, gross_room)

        net_cap = execution_cfg["net_exposure_cap"]
        if net_cap is not None:
            if sign > 0:
                net_room = max(float(net_cap) - net_before, 0.0)
            else:
                net_room = max(float(net_cap) + net_before, 0.0)
            if net_room <= 0:
                blocked_by_overlap[idx] = False
                blocked_by_net_cap[idx] = True
                continue
            allowed_abs = min(allowed_abs, net_room)

        if allowed_abs < float(execution_cfg["min_trade_exposure"]):
            blocked_by_gross_cap[idx] = blocked_by_gross_cap[idx] or gross_cap is not None
            blocked_by_net_cap[idx] = blocked_by_net_cap[idx] or net_cap is not None
            continue
        position_weight[idx] = sign * allowed_abs

    signal = np.sign(position_weight).astype(int)
    gross_pnl = position_weight * realized_edge.mean(axis=1)
    transaction_cost = np.abs(position_weight) * round_trip_cost_bps * 1e-4
    net_pnl = gross_pnl - transaction_cost

    exit_dates = pd.to_datetime(prediction_frame["date"])
    entry_dates = _infer_entry_dates(exit_dates, holding_period_bars=hold)
    trades = prediction_frame[["date"]].copy()
    trades["entry_date"] = entry_dates
    trades["exit_date"] = exit_dates
    trades["signal_score"] = signal_score
    trades["raw_signal"] = raw_signal
    trades["requested_weight"] = requested_weight
    trades["position_weight"] = position_weight
    trades["signal"] = signal
    trades["blocked_by_holding_period"] = blocked_by_overlap
    trades["blocked_by_max_positions"] = blocked_by_max_positions
    trades["blocked_by_gross_cap"] = blocked_by_gross_cap
    trades["blocked_by_net_cap"] = blocked_by_net_cap
    trades["holding_period_bars"] = hold
    trades["allow_overlapping_positions"] = bool(allow_overlapping_positions)
    trades["active_positions_before"] = active_positions_before
    trades["gross_exposure_before"] = gross_exposure_before
    trades["net_exposure_before"] = net_exposure_before
    trades["gross_pnl"] = gross_pnl
    trades["transaction_cost"] = transaction_cost
    trades["net_pnl"] = net_pnl
    trades["cumulative_pnl"] = trades["net_pnl"].cumsum()
    trades["drawdown"] = trades["cumulative_pnl"] - trades["cumulative_pnl"].cummax()
    trades["realized_edge_score"] = realized_score

    traded_mask = position_weight != 0
    traded_signal_score = signal_score[traded_mask]
    traded_gross_pnl = gross_pnl[traded_mask]
    traded_net_pnl = net_pnl[traded_mask]
    annualization = _annualization_factor(trades["date"])
    net_std = float(np.std(net_pnl, ddof=1)) if len(net_pnl) > 1 else 0.0
    downside = np.minimum(net_pnl, 0.0)
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    sharpe = float(np.mean(net_pnl) / net_std * np.sqrt(annualization)) if net_std > 0 and annualization > 0 else 0.0
    sortino = (
        float(np.mean(net_pnl) / downside_std * np.sqrt(annualization))
        if downside_std > 0 and annualization > 0
        else 0.0
    )
    signal_realized_corr = (
        float(np.corrcoef(signal_score, realized_score)[0, 1])
        if len(signal_score) > 1 and np.std(signal_score) > 0 and np.std(realized_score) > 0
        else 0.0
    )
    edge_sign_accuracy = (
        float(np.mean(np.sign(traded_signal_score) == np.sign(realized_score[traded_mask])))
        if traded_mask.any()
        else 0.0
    )
    wins = traded_net_pnl[traded_net_pnl > 0]
    losses = traded_net_pnl[traded_net_pnl < 0]
    gross_positive = float(wins.sum()) if len(wins) else 0.0
    gross_negative = float(np.abs(losses.sum())) if len(losses) else 0.0
    drawdown = trades["drawdown"]
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    max_open_positions = int(active_positions_before.max()) if len(active_positions_before) else 0
    avg_open_positions = float(active_positions_before.mean()) if len(active_positions_before) else 0.0
    max_gross_exposure = float(gross_exposure_before.max()) if len(gross_exposure_before) else 0.0
    avg_gross_exposure = float(gross_exposure_before.mean()) if len(gross_exposure_before) else 0.0
    max_abs_net_exposure = float(np.abs(net_exposure_before).max()) if len(net_exposure_before) else 0.0
    avg_abs_net_exposure = float(np.abs(net_exposure_before).mean()) if len(net_exposure_before) else 0.0
    annualized_mean = float(np.mean(net_pnl) * annualization) if annualization > 0 else 0.0
    calmar = float(annualized_mean / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    bar_pnl_skew = float(pd.Series(net_pnl).skew()) if len(net_pnl) > 1 else 0.0
    bar_pnl_kurtosis = float(pd.Series(net_pnl).kurt()) if len(net_pnl) > 1 else 0.0
    var_5 = float(np.quantile(net_pnl, 0.05)) if len(net_pnl) else 0.0
    cvar_5 = float(np.mean(net_pnl[net_pnl <= var_5])) if len(net_pnl) and np.any(net_pnl <= var_5) else 0.0
    trade_expectancy = float(np.mean(traded_net_pnl)) if traded_mask.any() else 0.0
    median_trade_pnl = float(np.median(traded_net_pnl)) if traded_mask.any() else 0.0
    win_loss_ratio = float(abs(np.mean(wins) / np.mean(losses))) if len(wins) and len(losses) else 0.0
    summary = {
        "num_periods": int(len(prediction_frame)),
        "num_trades": int(np.count_nonzero(position_weight)),
        "holding_period_bars": hold,
        "allow_overlapping_positions": bool(allow_overlapping_positions),
        "per_trade_exposure": float(execution_cfg["per_trade_exposure"]),
        "min_trade_exposure": float(execution_cfg["min_trade_exposure"]),
        "max_concurrent_positions": execution_cfg["max_concurrent_positions"],
        "gross_exposure_cap": execution_cfg["gross_exposure_cap"],
        "net_exposure_cap": execution_cfg["net_exposure_cap"],
        "round_trip_cost_bps": float(round_trip_cost_bps),
        "gross_pnl": float(gross_pnl.sum()),
        "net_pnl": float(net_pnl.sum()),
        "mean_net_pnl": float(net_pnl.mean()),
        "annualized_mean_pnl": annualized_mean,
        "hit_rate": float(np.mean(np.sign(traded_gross_pnl) == np.sign(traded_signal_score))) if traded_mask.any() else 0.0,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration_bars": _max_drawdown_duration(drawdown),
        "calmar_ratio": calmar,
        "turnover": float(np.abs(position_weight).sum() / max(len(position_weight), 1)),
        "trade_frequency": float(np.count_nonzero(position_weight) / max(len(position_weight), 1)),
        "annualization_factor": float(annualization),
        "sharpe_annualized": sharpe,
        "sortino_annualized": sortino,
        "net_pnl_std": net_std,
        "bar_pnl_skew": bar_pnl_skew,
        "bar_pnl_kurtosis": bar_pnl_kurtosis,
        "trade_pnl_skew": float(pd.Series(traded_net_pnl).skew()) if traded_mask.any() else 0.0,
        "trade_pnl_kurtosis": float(pd.Series(traded_net_pnl).kurt()) if traded_mask.any() else 0.0,
        "value_at_risk_5pct": var_5,
        "conditional_var_5pct": cvar_5,
        "long_trades": int(np.count_nonzero(position_weight > 0)),
        "short_trades": int(np.count_nonzero(position_weight < 0)),
        "long_fraction": float(np.mean(position_weight[traded_mask] > 0)) if traded_mask.any() else 0.0,
        "short_fraction": float(np.mean(position_weight[traded_mask] < 0)) if traded_mask.any() else 0.0,
        "avg_signal_score": float(np.mean(signal_score)),
        "avg_abs_signal_score": float(np.mean(np.abs(signal_score))),
        "signal_realized_corr": signal_realized_corr,
        "edge_sign_accuracy": edge_sign_accuracy,
        "profit_factor": float(gross_positive / gross_negative) if gross_negative > 0 else 0.0,
        "avg_win": float(np.mean(wins)) if len(wins) else 0.0,
        "avg_loss": float(np.mean(losses)) if len(losses) else 0.0,
        "win_loss_ratio": win_loss_ratio,
        "trade_expectancy": trade_expectancy,
        "median_trade_pnl": median_trade_pnl,
        "best_trade": float(np.max(traded_net_pnl)) if traded_mask.any() else 0.0,
        "worst_trade": float(np.min(traded_net_pnl)) if traded_mask.any() else 0.0,
        "total_transaction_cost": float(transaction_cost.sum()),
        "avg_position_weight": float(np.mean(np.abs(position_weight[traded_mask]))) if traded_mask.any() else 0.0,
        "avg_open_positions": avg_open_positions,
        "max_open_positions": max_open_positions,
        "avg_gross_exposure": avg_gross_exposure,
        "max_gross_exposure": max_gross_exposure,
        "avg_abs_net_exposure": avg_abs_net_exposure,
        "max_abs_net_exposure": max_abs_net_exposure,
        "num_blocked_overlap": int(blocked_by_overlap.sum()),
        "num_blocked_max_positions": int(blocked_by_max_positions.sum()),
        "num_blocked_gross_cap": int(blocked_by_gross_cap.sum()),
        "num_blocked_net_cap": int(blocked_by_net_cap.sum()),
    }
    return trades, summary


def save_backtest_outputs(
    trades: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output / "backtest_trades.csv", index=False)
    pd.DataFrame([summary]).to_csv(output / "backtest_summary.csv", index=False)
