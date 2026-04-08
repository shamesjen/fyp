from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

cache_root = ROOT / ".cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from run_execution_backtest import run_from_config
from src.data.csv_panel_loader import curve_sort_key
from src.evaluation.statistical_tests import diebold_mariano_test
from src.training.metrics import compute_metrics
from src.utils.config import load_yaml_config, resolve_path


def frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.astype(object).itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def infer_curve_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [
            column.replace("current_", "")
            for column in frame.columns
            if column.startswith("current_iv_mny_") and f"actual_{column.replace('current_', '')}" in frame.columns
        ],
        key=curve_sort_key,
    )


def write_yaml(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def run_command(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def read_summary_row(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Expected non-empty summary at {path}")
    return frame.iloc[0].to_dict()


def aggregate_metric(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = frame.groupby("name")[metric].agg(["mean", "std", "min", "max"]).reset_index()
    grouped.columns = ["name", f"{metric}_mean", f"{metric}_std", f"{metric}_min", f"{metric}_max"]
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final seq12_h1 multi-seed benchmark.")
    parser.add_argument("--config", default="configs/multiseed_final_benchmark.yaml")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    dataset_path = str(cfg["paths"]["dataset_path"])
    output_root = resolve_path(cfg["paths"]["output_root"])
    report_path = resolve_path(cfg["paths"]["report_path"])
    output_root.mkdir(parents=True, exist_ok=True)

    base_lstm_cfg = load_yaml_config(cfg["paths"]["base_lstm_config"])
    base_xlstm_cfg = load_yaml_config(cfg["paths"]["base_xlstm_config"])
    base_baseline_cfg = load_yaml_config(cfg["paths"]["base_baseline_config"])
    backtest_cfg = load_yaml_config(cfg["paths"]["execution_backtest_config"])

    seeds = [int(seed) for seed in cfg["benchmark"]["seeds"]]
    signal_threshold = float(cfg["benchmark"]["signal_threshold"])
    holding_period = int(cfg["benchmark"]["holding_period_bars"])
    wf_cfg = base_lstm_cfg["walkforward"]

    generated_cfg_root = output_root / "generated_configs"
    raw_rows: list[dict[str, Any]] = []
    standardized_rows: list[dict[str, Any]] = []
    dm_rows: list[dict[str, Any]] = []
    equity_frames: list[pd.DataFrame] = []

    for seed in seeds:
        seed_label = f"seed_{seed}"
        print(f"\n=== Multi-seed benchmark: {seed_label} ===")

        lstm_cfg = copy.deepcopy(base_lstm_cfg)
        lstm_cfg["paths"]["dataset_path"] = dataset_path
        lstm_cfg["paths"]["output_dir"] = str(output_root / cfg["plain_lstm"]["output_subdir"])
        lstm_cfg["training"]["random_seed"] = seed
        lstm_cfg["model"]["architecture"] = cfg["plain_lstm"]["architecture"]
        lstm_cfg["model"]["num_layers"] = int(cfg["plain_lstm"]["num_layers"])
        lstm_cfg["model"]["hidden_size"] = int(cfg["plain_lstm"]["hidden_size"])
        lstm_cfg_path = generated_cfg_root / f"{seed_label}_plain_lstm.yaml"
        write_yaml(lstm_cfg, lstm_cfg_path)
        lstm_tag = seed_label
        lstm_out = output_root / cfg["plain_lstm"]["output_subdir"] / lstm_tag
        if not args.skip_existing or not (lstm_out / "stitched_test_predictions.csv").exists():
            run_command(
                [
                    sys.executable,
                    "scripts/run_lstm_walkforward.py",
                    "--config",
                    str(lstm_cfg_path),
                    "--tag",
                    lstm_tag,
                ]
            )

        xlstm_cfg = copy.deepcopy(base_xlstm_cfg)
        xlstm_cfg["paths"]["dataset_path"] = dataset_path
        xlstm_cfg["paths"]["output_dir"] = str(output_root / cfg["xlstm"]["output_subdir"])
        xlstm_cfg["training"]["random_seed"] = seed
        xlstm_cfg["model"]["architecture"] = cfg["xlstm"]["architecture"]
        xlstm_cfg["model"]["num_blocks"] = int(cfg["xlstm"]["num_blocks"])
        xlstm_cfg["model"]["embedding_dim"] = int(cfg["xlstm"]["embedding_dim"])
        xlstm_cfg_path = generated_cfg_root / f"{seed_label}_xlstm.yaml"
        write_yaml(xlstm_cfg, xlstm_cfg_path)
        xlstm_tag = seed_label
        xlstm_out = output_root / cfg["xlstm"]["output_subdir"] / xlstm_tag
        if not args.skip_existing or not (xlstm_out / "stitched_test_predictions.csv").exists():
            run_command(
                [
                    sys.executable,
                    "scripts/run_lstm_walkforward.py",
                    "--config",
                    str(xlstm_cfg_path),
                    "--tag",
                    xlstm_tag,
                ]
            )

        baseline_cfg = copy.deepcopy(base_baseline_cfg)
        baseline_cfg["paths"]["dataset_path"] = dataset_path
        baseline_cfg["paths"]["output_dir"] = str(output_root / cfg["baselines"]["output_subdir"])
        baseline_cfg["training"]["random_seed"] = seed
        baseline_cfg["models"] = {
            name: copy.deepcopy(base_baseline_cfg["models"][name])
            for name in cfg["baselines"]["include_models"]
        }
        baseline_cfg_path = generated_cfg_root / f"{seed_label}_baselines.yaml"
        write_yaml(baseline_cfg, baseline_cfg_path)
        baseline_tag = seed_label
        baseline_out = output_root / cfg["baselines"]["output_subdir"] / baseline_tag
        if not args.skip_existing or not (baseline_out / "baseline_walkforward_summary.csv").exists():
            run_command(
                [
                    sys.executable,
                    "scripts/run_baselines_walkforward.py",
                    "--config",
                    str(baseline_cfg_path),
                    "--tag",
                    baseline_tag,
                ]
            )

        lstm_summary = read_summary_row(lstm_out / "walkforward_summary.csv")
        lstm_summary.update({"seed": seed, "name": cfg["plain_lstm"]["name"], "family": "lstm"})
        raw_rows.append(lstm_summary)

        xlstm_summary = read_summary_row(xlstm_out / "walkforward_summary.csv")
        xlstm_summary.update({"seed": seed, "name": cfg["xlstm"]["name"], "family": "xlstm"})
        raw_rows.append(xlstm_summary)

        baseline_summary = pd.read_csv(baseline_out / "baseline_walkforward_summary.csv")
        for _, row in baseline_summary.iterrows():
            if row["model"] not in set(cfg["baselines"]["include_models"]):
                continue
            raw_rows.append(
                {
                    "seed": seed,
                    "name": str(row["model"]),
                    "family": "baseline",
                    "test_rmse": float(row["test_rmse"]),
                    "test_mae": float(row["test_mae"]),
                    "test_r2": float(row["test_r2"]),
                    "num_trades": int(row["num_trades"]),
                    "net_pnl": float(row["net_pnl"]),
                    "sharpe_annualized": float(row["sharpe_annualized"]),
                    "hit_rate": float(row["hit_rate"]),
                    "turnover": float(row["turnover"]),
                    "max_drawdown": float(row["max_drawdown"]),
                    "signal_realized_corr": float(row["signal_realized_corr"]),
                    "edge_sign_accuracy": float(row["edge_sign_accuracy"]),
                }
            )

        prediction_paths = {
            cfg["plain_lstm"]["name"]: lstm_out / "stitched_test_predictions.csv",
            cfg["xlstm"]["name"]: xlstm_out / "stitched_test_predictions.csv",
        }
        for model_name in cfg["baselines"]["include_models"]:
            prediction_paths[str(model_name)] = baseline_out / str(model_name) / "stitched_test_predictions.csv"

        frames: dict[str, pd.DataFrame] = {}
        common_dates: pd.Index | None = None
        for name, path in prediction_paths.items():
            frame = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
            frames[name] = frame
            common_dates = pd.Index(frame["date"]) if common_dates is None else common_dates.intersection(frame["date"])
        if common_dates is None or len(common_dates) == 0:
            raise ValueError(f"No common dates found for {seed_label}.")

        standardized_root = output_root / "standardized" / seed_label
        standardized_root.mkdir(parents=True, exist_ok=True)
        clipped_predictions: dict[str, pd.DataFrame] = {}

        for name, frame in frames.items():
            clipped = frame[frame["date"].isin(common_dates)].sort_values("date").reset_index(drop=True)
            clipped_predictions[name] = clipped
            clipped_path = standardized_root / f"{name}_standardized_predictions.csv"
            clipped.to_csv(clipped_path, index=False)

            curve_columns = infer_curve_columns(clipped)
            y_true = clipped[[f"actual_{column}" for column in curve_columns]].to_numpy()
            y_pred = clipped[[f"pred_{column}" for column in curve_columns]].to_numpy()
            metrics = compute_metrics(y_true, y_pred, curve_columns)

            bt_cfg = copy.deepcopy(backtest_cfg)
            bt_cfg["paths"]["predictions_path"] = str(clipped_path)
            bt_cfg["paths"]["output_dir"] = str(standardized_root / name)
            bt_cfg["backtest"]["signal_threshold"] = signal_threshold
            bt_cfg["backtest"]["holding_period_bars"] = holding_period
            trades, summary = run_from_config(bt_cfg)

            standardized_rows.append(
                {
                    "seed": seed,
                    "name": name,
                    "family": "lstm" if name == cfg["plain_lstm"]["name"] else ("xlstm" if name == cfg["xlstm"]["name"] else "baseline"),
                    "start_date": str(clipped["date"].min()),
                    "end_date": str(clipped["date"].max()),
                    "num_rows": int(len(clipped)),
                    "test_rmse": float(metrics["rmse"]),
                    "test_mae": float(metrics["mae"]),
                    "test_r2": float(metrics["r2"]),
                    **summary,
                }
            )
            equity_frames.append(
                trades[["exit_date", "cumulative_pnl"]].rename(
                    columns={"exit_date": "date", "cumulative_pnl": f"{seed_label}:{name}"}
                )
            )

        lstm_frame = clipped_predictions[cfg["plain_lstm"]["name"]]
        curve_columns = infer_curve_columns(lstm_frame)
        lstm_y_true = lstm_frame[[f"actual_{column}" for column in curve_columns]].to_numpy()
        lstm_y_pred = lstm_frame[[f"pred_{column}" for column in curve_columns]].to_numpy()
        lstm_losses = np.mean((lstm_y_true - lstm_y_pred) ** 2, axis=1)
        atm_column = "iv_mny_0p0"
        lstm_atm_losses = (lstm_frame[f"actual_{atm_column}"].to_numpy() - lstm_frame[f"pred_{atm_column}"].to_numpy()) ** 2

        for rival_name, rival_frame in clipped_predictions.items():
            if rival_name == cfg["plain_lstm"]["name"]:
                continue
            rival_y_pred = rival_frame[[f"pred_{column}" for column in curve_columns]].to_numpy()
            rival_losses = np.mean((lstm_y_true - rival_y_pred) ** 2, axis=1)
            overall_dm = diebold_mariano_test(lstm_losses, rival_losses, horizon=holding_period)
            rival_atm_losses = (rival_frame[f"actual_{atm_column}"].to_numpy() - rival_frame[f"pred_{atm_column}"].to_numpy()) ** 2
            atm_dm = diebold_mariano_test(lstm_atm_losses, rival_atm_losses, horizon=holding_period)
            dm_rows.append(
                {
                    "seed": seed,
                    "benchmark_model": cfg["plain_lstm"]["name"],
                    "comparison_model": rival_name,
                    "overall_dm_stat": float(overall_dm["dm_stat"]),
                    "overall_dm_p_value": float(overall_dm["p_value"]),
                    "atm_dm_stat": float(atm_dm["dm_stat"]),
                    "atm_dm_p_value": float(atm_dm["p_value"]),
                }
            )

    raw_frame = pd.DataFrame(raw_rows).sort_values(["seed", "name"]).reset_index(drop=True)
    standardized_frame = pd.DataFrame(standardized_rows).sort_values(["seed", "name"]).reset_index(drop=True)
    dm_frame = pd.DataFrame(dm_rows).sort_values(["seed", "comparison_model"]).reset_index(drop=True)

    raw_frame.to_csv(output_root / "raw_seed_summary.csv", index=False)
    standardized_frame.to_csv(output_root / "standardized_seed_summary.csv", index=False)
    dm_frame.to_csv(output_root / "pairwise_dm_by_seed.csv", index=False)

    agg_metrics = standardized_frame.groupby("name").agg(
        seeds=("seed", "nunique"),
        rmse_mean=("test_rmse", "mean"),
        rmse_std=("test_rmse", "std"),
        mae_mean=("test_mae", "mean"),
        r2_mean=("test_r2", "mean"),
        net_pnl_mean=("net_pnl", "mean"),
        net_pnl_std=("net_pnl", "std"),
        sharpe_mean=("sharpe_annualized", "mean"),
        sharpe_std=("sharpe_annualized", "std"),
        hit_rate_mean=("hit_rate", "mean"),
        max_drawdown_mean=("max_drawdown", "mean"),
        num_trades_mean=("num_trades", "mean"),
        signal_realized_corr_mean=("signal_realized_corr", "mean"),
        edge_sign_accuracy_mean=("edge_sign_accuracy", "mean"),
    ).reset_index()
    agg_metrics["rmse_rank"] = agg_metrics["rmse_mean"].rank(method="min")
    agg_metrics["net_pnl_rank"] = agg_metrics["net_pnl_mean"].rank(method="min", ascending=False)
    agg_metrics = agg_metrics.sort_values(["rmse_rank", "net_pnl_rank", "name"]).reset_index(drop=True)
    agg_metrics.to_csv(output_root / "aggregate_model_summary.csv", index=False)

    dm_agg = dm_frame.groupby("comparison_model").agg(
        seeds=("seed", "nunique"),
        overall_dm_mean=("overall_dm_stat", "mean"),
        overall_dm_std=("overall_dm_stat", "std"),
        overall_p_mean=("overall_dm_p_value", "mean"),
        overall_sig_count=("overall_dm_p_value", lambda s: int((s < 0.05).sum())),
        atm_dm_mean=("atm_dm_stat", "mean"),
        atm_dm_std=("atm_dm_stat", "std"),
        atm_p_mean=("atm_dm_p_value", "mean"),
        atm_sig_count=("atm_dm_p_value", lambda s: int((s < 0.05).sum())),
    ).reset_index().sort_values("overall_dm_mean")
    dm_agg.to_csv(output_root / "aggregate_pairwise_dm_summary.csv", index=False)

    rmse_winners = standardized_frame.loc[standardized_frame.groupby("seed")["test_rmse"].idxmin(), ["seed", "name"]]
    rmse_win_counts = rmse_winners["name"].value_counts().rename_axis("name").reset_index(name="rmse_seed_wins")
    pnl_winners = standardized_frame.loc[standardized_frame.groupby("seed")["net_pnl"].idxmax(), ["seed", "name"]]
    pnl_win_counts = pnl_winners["name"].value_counts().rename_axis("name").reset_index(name="net_pnl_seed_wins")
    agg_metrics = agg_metrics.merge(rmse_win_counts, on="name", how="left").merge(pnl_win_counts, on="name", how="left")
    agg_metrics = agg_metrics.fillna({"rmse_seed_wins": 0, "net_pnl_seed_wins": 0})
    agg_metrics.to_csv(output_root / "aggregate_model_summary.csv", index=False)

    merged_equity = None
    for frame in equity_frames:
        merged_equity = frame if merged_equity is None else merged_equity.merge(frame, on="date", how="outer")
    if merged_equity is not None:
        merged_equity = merged_equity.sort_values("date").ffill().fillna(0.0)
        merged_equity.to_csv(output_root / "seed_equity_curves.csv", index=False)

        fig, ax = plt.subplots(figsize=(11, 5))
        for column in merged_equity.columns:
            if column == "date":
                continue
            ax.plot(pd.to_datetime(merged_equity["date"]), merged_equity[column], alpha=0.35, linewidth=1.0)
        ax.set_title("Multi-Seed Final Benchmark Equity Curves")
        ax.set_xlabel("Exit Date")
        ax.set_ylabel("Cumulative PnL")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_root / "seed_equity_curves.png", dpi=180)
        plt.close(fig)

    report_lines = [
        "# Multi-Seed Final Benchmark",
        "",
        "This benchmark reruns the final matched `seq12_h1` setup across multiple random seeds for the two neural models and the four strongest non-neural comparators chosen for the final conference-style benchmark:",
        "",
        "- `plain_lstm_l2_h128`",
        "- `xlstm_b2_e128`",
        "- `elastic_net_baseline`",
        "- `smile_coefficient_baseline`",
        "- `hist_gradient_boosting_baseline`",
        "- `har_factor_baseline`",
        "",
        f"Seeds: `{seeds}`",
        f"Walk-forward: train `{int(wf_cfg['initial_train_size'])}`, val `{int(wf_cfg['val_size'])}`, test `{int(wf_cfg['test_size'])}`, step `{int(wf_cfg.get('step_size', wf_cfg['test_size']))}`",
        f"Stitch mode: `{str(wf_cfg.get('stitch_mode', 'full_test'))}`",
        f"Signal threshold: `{signal_threshold}`",
        f"Holding period: `{holding_period}` bar",
        "",
        "## Aggregate Standardized Summary",
        "",
        frame_to_markdown(agg_metrics.round(6)),
        "",
        "## Pairwise Diebold-Mariano Summary Against Plain LSTM",
        "",
        frame_to_markdown(dm_agg.round(6)),
        "",
        "## Per-Seed Standardized Results",
        "",
        frame_to_markdown(standardized_frame.round(6)),
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(agg_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
