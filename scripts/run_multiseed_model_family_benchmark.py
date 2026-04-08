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


def _model_output_path(output_root: Path, family_subdir: str, tag: str) -> Path:
    return output_root / family_subdir / tag


def _lstm_tag(candidate: dict[str, Any], seed: int) -> str:
    return f"seed_{seed}_{candidate['tag']}"


def _xlstm_tag(candidate: dict[str, Any], seed: int) -> str:
    return f"seed_{seed}_{candidate['tag']}"


def _standardized_family(name: str, candidate_lookup: dict[str, str]) -> str:
    return candidate_lookup.get(name, "baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-seed model-family benchmark over multiple LSTM/xLSTM candidates and baselines.")
    parser.add_argument("--config", default="configs/multiseed_model_family_benchmark_carry.yaml")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    dataset_path = str(cfg["paths"]["dataset_path"])
    output_root = resolve_path(cfg["paths"]["output_root"])
    report_path = resolve_path(cfg["paths"]["report_path"])
    output_root.mkdir(parents=True, exist_ok=True)

    base_lstm_cfg = load_yaml_config(cfg["paths"]["base_lstm_config"])
    base_xlstm_cfg = load_yaml_config(cfg["paths"]["base_xlstm_config"])
    base_transformer_cfg = load_yaml_config(cfg["paths"]["base_transformer_config"]) if "base_transformer_config" in cfg["paths"] else None
    base_baseline_cfg = load_yaml_config(cfg["paths"]["base_baseline_config"])
    backtest_cfg = load_yaml_config(cfg["paths"]["execution_backtest_config"])

    seeds = [int(seed) for seed in cfg["benchmark"]["seeds"]]
    signal_threshold = float(cfg["benchmark"]["signal_threshold"])
    holding_period = int(cfg["benchmark"]["holding_period_bars"])
    generated_cfg_root = output_root / "generated_configs"

    lstm_candidates = list(cfg.get("lstm_candidates", []))
    xlstm_candidates = list(cfg.get("xlstm_candidates", []))
    transformer_candidates = list(cfg.get("transformer_candidates", []))
    baseline_models = [str(name) for name in cfg["baselines"]["include_models"]]

    candidate_family_lookup: dict[str, str] = {}
    raw_rows: list[dict[str, Any]] = []
    standardized_rows: list[dict[str, Any]] = []
    equity_frames: list[pd.DataFrame] = []

    for seed in seeds:
        seed_label = f"seed_{seed}"
        print(f"\n=== Model family benchmark: {seed_label} ===")

        prediction_paths: dict[str, Path] = {}

        for candidate in lstm_candidates:
            run_cfg = copy.deepcopy(base_lstm_cfg)
            run_cfg["paths"]["dataset_path"] = dataset_path
            run_cfg["paths"]["output_dir"] = str(output_root / "lstm")
            run_cfg["training"]["random_seed"] = seed
            run_cfg["model"]["architecture"] = "lstm"
            run_cfg["model"]["num_layers"] = int(candidate["num_layers"])
            run_cfg["model"]["hidden_size"] = int(candidate["hidden_size"])
            if "dropout" in candidate:
                run_cfg["model"]["dropout"] = float(candidate["dropout"])
            if "pooling_mode" in candidate:
                run_cfg["model"]["pooling_mode"] = str(candidate["pooling_mode"])
            if "hooks" in candidate:
                run_cfg.setdefault("hooks", {})
                for key, value in candidate["hooks"].items():
                    run_cfg["hooks"][key] = copy.deepcopy(value)
            cfg_path = generated_cfg_root / f"{seed_label}_{candidate['name']}.yaml"
            write_yaml(run_cfg, cfg_path)
            tag = _lstm_tag(candidate, seed)
            out_dir = _model_output_path(output_root, "lstm", tag)
            if not args.skip_existing or not (out_dir / "stitched_test_predictions.csv").exists():
                run_command(
                    [
                        sys.executable,
                        "scripts/run_lstm_walkforward.py",
                        "--config",
                        str(cfg_path),
                        "--tag",
                        tag,
                    ]
                )
            summary = read_summary_row(out_dir / "walkforward_summary.csv")
            summary.update({"seed": seed, "name": str(candidate["name"]), "family": "lstm"})
            raw_rows.append(summary)
            prediction_paths[str(candidate["name"])] = out_dir / "stitched_test_predictions.csv"
            candidate_family_lookup[str(candidate["name"])] = "lstm"

        for candidate in xlstm_candidates:
            run_cfg = copy.deepcopy(base_xlstm_cfg)
            run_cfg["paths"]["dataset_path"] = dataset_path
            run_cfg["paths"]["output_dir"] = str(output_root / "xlstm")
            run_cfg["training"]["random_seed"] = seed
            run_cfg["model"]["architecture"] = "xlstm"
            run_cfg["model"]["num_blocks"] = int(candidate["num_blocks"])
            run_cfg["model"]["embedding_dim"] = int(candidate["embedding_dim"])
            for key in ["dropout", "num_heads", "proj_factor", "conv1d_kernel_size", "bias"]:
                if key in candidate:
                    run_cfg["model"][key] = copy.deepcopy(candidate[key])
            if "hooks" in candidate:
                run_cfg.setdefault("hooks", {})
                for key, value in candidate["hooks"].items():
                    run_cfg["hooks"][key] = copy.deepcopy(value)
            cfg_path = generated_cfg_root / f"{seed_label}_{candidate['name']}.yaml"
            write_yaml(run_cfg, cfg_path)
            tag = _xlstm_tag(candidate, seed)
            out_dir = _model_output_path(output_root, "xlstm", tag)
            if not args.skip_existing or not (out_dir / "stitched_test_predictions.csv").exists():
                run_command(
                    [
                        sys.executable,
                        "scripts/run_lstm_walkforward.py",
                        "--config",
                        str(cfg_path),
                        "--tag",
                        tag,
                    ]
                )
            summary = read_summary_row(out_dir / "walkforward_summary.csv")
            summary.update({"seed": seed, "name": str(candidate["name"]), "family": "xlstm"})
            raw_rows.append(summary)
            prediction_paths[str(candidate["name"])] = out_dir / "stitched_test_predictions.csv"
            candidate_family_lookup[str(candidate["name"])] = "xlstm"

        for candidate in transformer_candidates:
            if base_transformer_cfg is None:
                raise ValueError("Transformer candidates provided but base_transformer_config is missing.")
            run_cfg = copy.deepcopy(base_transformer_cfg)
            run_cfg["paths"]["dataset_path"] = dataset_path
            run_cfg["paths"]["output_dir"] = str(output_root / "transformer")
            run_cfg["training"]["random_seed"] = seed
            run_cfg["model"]["architecture"] = "transformer"
            run_cfg["model"]["num_layers"] = int(candidate["num_layers"])
            run_cfg["model"]["embedding_dim"] = int(candidate["embedding_dim"])
            for key in ["dropout", "num_heads", "ffn_dim", "pooling_mode"]:
                if key in candidate:
                    run_cfg["model"][key] = copy.deepcopy(candidate[key])
            if "hooks" in candidate:
                run_cfg.setdefault("hooks", {})
                for key, value in candidate["hooks"].items():
                    run_cfg["hooks"][key] = copy.deepcopy(value)
            cfg_path = generated_cfg_root / f"{seed_label}_{candidate['name']}.yaml"
            write_yaml(run_cfg, cfg_path)
            tag = f"seed_{seed}_{candidate['tag']}"
            out_dir = _model_output_path(output_root, "transformer", tag)
            if not args.skip_existing or not (out_dir / "stitched_test_predictions.csv").exists():
                run_command(
                    [
                        sys.executable,
                        "scripts/run_lstm_walkforward.py",
                        "--config",
                        str(cfg_path),
                        "--tag",
                        tag,
                    ]
                )
            summary = read_summary_row(out_dir / "walkforward_summary.csv")
            summary.update({"seed": seed, "name": str(candidate["name"]), "family": "transformer"})
            raw_rows.append(summary)
            prediction_paths[str(candidate["name"])] = out_dir / "stitched_test_predictions.csv"
            candidate_family_lookup[str(candidate["name"])] = "transformer"

        baseline_cfg = copy.deepcopy(base_baseline_cfg)
        baseline_cfg["paths"]["dataset_path"] = dataset_path
        baseline_cfg["paths"]["output_dir"] = str(output_root / "baselines")
        baseline_cfg["training"]["random_seed"] = seed
        baseline_cfg["models"] = {
            model_name: copy.deepcopy(base_baseline_cfg["models"][model_name])
            for model_name in baseline_models
        }
        baseline_cfg_path = generated_cfg_root / f"{seed_label}_baselines.yaml"
        write_yaml(baseline_cfg, baseline_cfg_path)
        baseline_tag = seed_label
        baseline_out = _model_output_path(output_root, "baselines", baseline_tag)
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

        baseline_summary = pd.read_csv(baseline_out / "baseline_walkforward_summary.csv")
        for _, row in baseline_summary.iterrows():
            model_name = str(row["model"])
            if model_name not in baseline_models:
                continue
            raw_rows.append(
                {
                    "seed": seed,
                    "name": model_name,
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
            prediction_paths[model_name] = baseline_out / model_name / "stitched_test_predictions.csv"

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

        for name, frame in frames.items():
            clipped = frame[frame["date"].isin(common_dates)].sort_values("date").reset_index(drop=True)
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
                    "family": _standardized_family(name, candidate_family_lookup),
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

    raw_frame = pd.DataFrame(raw_rows).sort_values(["seed", "family", "name"]).reset_index(drop=True)
    standardized_frame = pd.DataFrame(standardized_rows).sort_values(["seed", "family", "name"]).reset_index(drop=True)
    raw_frame.to_csv(output_root / "raw_seed_summary.csv", index=False)
    standardized_frame.to_csv(output_root / "standardized_seed_summary.csv", index=False)

    agg_metrics = standardized_frame.groupby(["name", "family"]).agg(
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
    agg_metrics["sharpe_rank"] = agg_metrics["sharpe_mean"].rank(method="min", ascending=False)

    rmse_winners = standardized_frame.loc[standardized_frame.groupby("seed")["test_rmse"].idxmin(), ["seed", "name"]]
    rmse_win_counts = rmse_winners["name"].value_counts().rename_axis("name").reset_index(name="rmse_seed_wins")
    pnl_winners = standardized_frame.loc[standardized_frame.groupby("seed")["net_pnl"].idxmax(), ["seed", "name"]]
    pnl_win_counts = pnl_winners["name"].value_counts().rename_axis("name").reset_index(name="net_pnl_seed_wins")
    sharpe_winners = standardized_frame.loc[standardized_frame.groupby("seed")["sharpe_annualized"].idxmax(), ["seed", "name"]]
    sharpe_win_counts = sharpe_winners["name"].value_counts().rename_axis("name").reset_index(name="sharpe_seed_wins")
    agg_metrics = (
        agg_metrics.merge(rmse_win_counts, on="name", how="left")
        .merge(pnl_win_counts, on="name", how="left")
        .merge(sharpe_win_counts, on="name", how="left")
        .fillna({"rmse_seed_wins": 0, "net_pnl_seed_wins": 0, "sharpe_seed_wins": 0})
        .sort_values(["rmse_rank", "net_pnl_rank", "sharpe_rank", "name"])
        .reset_index(drop=True)
    )
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
            ax.plot(pd.to_datetime(merged_equity["date"]), merged_equity[column], alpha=0.25, linewidth=0.9)
        ax.set_title("Multi-Seed Model Family Benchmark Equity Curves")
        ax.set_xlabel("Exit Date")
        ax.set_ylabel("Cumulative PnL")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_root / "seed_equity_curves.png", dpi=180)
        plt.close(fig)

    report_lines = [
        "# Multi-Seed Model Family Benchmark",
        "",
        f"Dataset: `{dataset_path}`",
        f"Seeds: `{seeds}`",
        f"Signal threshold: `{signal_threshold}`",
        f"Holding period: `{holding_period}` bar",
        "",
        "## LSTM Candidates",
        "",
        frame_to_markdown(pd.DataFrame(lstm_candidates)),
        "",
        "## xLSTM Candidates",
        "",
        frame_to_markdown(pd.DataFrame(xlstm_candidates)),
        "",
        "## Transformer Candidates",
        "",
        frame_to_markdown(pd.DataFrame(transformer_candidates)),
        "",
        "## Included Baselines",
        "",
        frame_to_markdown(pd.DataFrame({"baseline_model": baseline_models})),
        "",
        "## Aggregate Standardized Summary",
        "",
        frame_to_markdown(agg_metrics.round(6)),
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
