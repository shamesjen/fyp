from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

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

from src.data.splits import walkforward_expanding_splits
from src.evaluation.backtest import run_backtest, save_backtest_outputs
from src.evaluation.statistical_tests import diebold_mariano_test
from src.training.metrics import compute_metrics
from src.training.train_lstm import train_on_split
from src.utils.config import load_yaml_config
from src.utils.io import load_dataset_bundle, save_json


def frame_to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.astype(object).itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def best_epoch(history: dict[str, list[float]]) -> tuple[int, float, float] | None:
    val_loss = history.get("val_loss", [])
    train_loss = history.get("train_loss", [])
    if not val_loss:
        return None
    idx = min(range(len(val_loss)), key=val_loss.__getitem__)
    return idx + 1, train_loss[idx], val_loss[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stitched walk-forward LSTM evaluation.")
    parser.add_argument("--config", default="configs/walkforward_lstm_hourly_h1_live.yaml")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--holding-period", type=int, default=None)
    parser.add_argument("--signal-threshold", type=float, default=None)
    parser.add_argument("--smoothness-penalty", type=float, default=None)
    parser.add_argument("--disable-shape-projection", action="store_true")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--architecture", default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if args.dataset_path:
        config["paths"]["dataset_path"] = args.dataset_path
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    if args.holding_period is not None:
        config["backtest"]["holding_period_bars"] = int(args.holding_period)
    if args.signal_threshold is not None:
        config["backtest"]["signal_threshold"] = float(args.signal_threshold)
    if args.smoothness_penalty is not None:
        config.setdefault("hooks", {})
        config["hooks"]["smoothness_penalty"] = float(args.smoothness_penalty)
    if args.disable_shape_projection:
        config.setdefault("hooks", {})
        config.setdefault("hooks", {}).setdefault("shape_projection", {})
        config["hooks"]["shape_projection"]["enabled"] = False
    if args.num_layers is not None:
        config["model"]["num_layers"] = int(args.num_layers)
    if args.hidden_size is not None:
        config["model"]["hidden_size"] = int(args.hidden_size)
    if args.architecture is not None:
        config["model"]["architecture"] = str(args.architecture)
    if args.num_blocks is not None:
        config["model"]["num_blocks"] = int(args.num_blocks)
    if args.embedding_dim is not None:
        config["model"]["embedding_dim"] = int(args.embedding_dim)

    architecture = str(config["model"].get("architecture", "lstm")).lower()
    if architecture == "xlstm":
        default_tag = f"xlstm_b{int(config['model'].get('num_blocks', 2))}_e{int(config['model'].get('embedding_dim', 128))}"
    elif architecture == "transformer":
        default_tag = (
            f"transformer_l{int(config['model'].get('num_layers', 2))}"
            f"_e{int(config['model'].get('embedding_dim', 128))}"
            f"_h{int(config['model'].get('num_heads', 4))}"
        )
    else:
        default_tag = f"lstm_l{int(config['model']['num_layers'])}_h{int(config['model']['hidden_size'])}"
    tag = args.tag or default_tag
    output_root = ROOT / str(config["paths"]["output_dir"]) / tag
    output_root.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset_bundle(config["paths"]["dataset_path"])
    wf_cfg = config["walkforward"]
    splits = walkforward_expanding_splits(
        n_samples=len(bundle.X),
        initial_train_size=wf_cfg["initial_train_size"],
        val_size=wf_cfg["val_size"],
        test_size=wf_cfg["test_size"],
        step_size=wf_cfg.get("step_size"),
        max_splits=wf_cfg.get("max_splits"),
    )

    fold_rows: list[dict[str, float | int | str]] = []
    stitched_frames: list[pd.DataFrame] = []
    stitched_y_true: list[np.ndarray] = []
    stitched_y_pred: list[np.ndarray] = []
    stitched_persistence: list[np.ndarray] = []

    save_fold_artifacts = bool(wf_cfg.get("save_fold_artifacts", False))
    stitch_mode = str(wf_cfg.get("stitch_mode", "full_test")).lower()
    stitch_step = int(wf_cfg.get("step_size", wf_cfg.get("test_size")))
    for fold_idx, split in enumerate(splits, start=1):
        fold_dir = output_root / f"fold_{fold_idx:02d}"
        fold_config = copy.deepcopy(config)
        fold_config["paths"]["output_dir"] = str(fold_dir)
        result = train_on_split(
            bundle=bundle,
            split=split,
            config=fold_config,
            output_dir=fold_dir,
            save_artifacts=save_fold_artifacts,
        )

        frame = result["prediction_frame"].copy()
        if stitch_mode == "frontier":
            stitch_count = len(split.test_idx) if fold_idx == len(splits) else min(stitch_step, len(split.test_idx))
        else:
            stitch_count = len(split.test_idx)
        frame = frame.iloc[:stitch_count].copy()
        frame["fold"] = fold_idx
        stitched_frames.append(frame)
        stitched_y_true.append(result["y_true"][:stitch_count])
        stitched_y_pred.append(result["y_pred"][:stitch_count])
        stitched_persistence.append(result["persistence_pred"][:stitch_count])

        history = result["summary"]["history"]
        epoch_info = best_epoch(history)
        test_metrics = result["summary"]["test"]
        row = {
            "fold": fold_idx,
            "train_size": int(len(split.train_idx)),
            "val_size": int(len(split.val_idx)),
            "test_size": int(len(split.test_idx)),
            "stitched_test_size": int(stitch_count),
            "train_start": str(bundle.dates[split.train_idx[0]]),
            "train_end": str(bundle.dates[split.train_idx[-1]]),
            "val_start": str(bundle.dates[split.val_idx[0]]),
            "val_end": str(bundle.dates[split.val_idx[-1]]),
            "test_start": str(bundle.dates[split.test_idx[0]]),
            "test_end": str(bundle.dates[split.test_idx[-1]]),
            "stitched_test_start": str(bundle.dates[split.test_idx[0]]),
            "stitched_test_end": str(bundle.dates[split.test_idx[stitch_count - 1]]),
            "test_rmse": float(test_metrics["rmse"]),
            "test_mae": float(test_metrics["mae"]),
            "test_r2": float(test_metrics["r2"]),
            "dm_stat_vs_persistence": float(result["summary"]["dm_vs_persistence"]["dm_stat"]),
            "dm_p_value_vs_persistence": float(result["summary"]["dm_vs_persistence"]["p_value"]),
        }
        if epoch_info is not None:
            row["best_epoch"] = int(epoch_info[0])
            row["best_train_loss"] = float(epoch_info[1])
            row["best_val_loss"] = float(epoch_info[2])
        fold_rows.append(row)

    stitched_prediction_frame = pd.concat(stitched_frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    stitched_prediction_frame.to_csv(output_root / "stitched_test_predictions.csv", index=False)

    y_true = np.concatenate(stitched_y_true, axis=0)
    y_pred = np.concatenate(stitched_y_pred, axis=0)
    persistence_pred = np.concatenate(stitched_persistence, axis=0)
    overall_metrics = compute_metrics(y_true, y_pred, bundle.curve_columns)
    dm_result = diebold_mariano_test(
        np.mean((y_true - y_pred) ** 2, axis=1),
        np.mean((y_true - persistence_pred) ** 2, axis=1),
    )

    trades, backtest_summary = run_backtest(
        prediction_frame=stitched_prediction_frame,
        curve_columns=bundle.curve_columns,
        moneyness_grid=list(bundle.metadata["moneyness_grid"]),
        maturity_bucket_days=int(bundle.metadata["maturity_bucket_days"]),
        signal_threshold=float(config["backtest"]["signal_threshold"]),
        transaction_cost_bps=float(config["backtest"]["transaction_cost_bps"]),
        holding_period_bars=int(config["backtest"].get("holding_period_bars", 1)),
        allow_overlapping_positions=bool(config["backtest"].get("allow_overlapping_positions", True)),
        execution=config["backtest"].get("execution"),
    )
    save_backtest_outputs(trades, backtest_summary, output_root / "backtest")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trades["date"], trades["cumulative_pnl"])
    ax.set_title(f"Walk-Forward Equity Curve: {tag}")
    ax.set_xlabel("Date")
    ax.set_ylabel("PnL")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_root / "backtest" / "equity_curve.png", dpi=150)
    plt.close(fig)

    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(output_root / "fold_metrics.csv", index=False)

    summary_payload = {
        "model": {
            "architecture": architecture,
        },
        "walkforward": {
            "num_folds": len(splits),
            "initial_train_size": int(wf_cfg["initial_train_size"]),
            "val_size": int(wf_cfg["val_size"]),
            "test_size": int(wf_cfg["test_size"]),
            "step_size": int(wf_cfg.get("step_size", wf_cfg["test_size"])),
            "stitch_mode": stitch_mode,
        },
        "stitched_test": overall_metrics,
        "dm_vs_persistence": dm_result,
        "backtest": backtest_summary,
    }
    if architecture == "xlstm":
        summary_payload["model"]["num_blocks"] = int(config["model"].get("num_blocks", 2))
        summary_payload["model"]["embedding_dim"] = int(config["model"].get("embedding_dim", 128))
    elif architecture == "transformer":
        summary_payload["model"]["num_layers"] = int(config["model"].get("num_layers", 2))
        summary_payload["model"]["embedding_dim"] = int(config["model"].get("embedding_dim", 128))
        summary_payload["model"]["num_heads"] = int(config["model"].get("num_heads", 4))
        summary_payload["model"]["ffn_dim"] = int(config["model"].get("ffn_dim", 256))
    else:
        summary_payload["model"]["num_layers"] = int(config["model"]["num_layers"])
        summary_payload["model"]["hidden_size"] = int(config["model"]["hidden_size"])
    save_json(summary_payload, output_root / "walkforward_metrics.json")
    summary_row = {
        "architecture": architecture,
        "num_folds": len(splits),
        "test_rmse": float(overall_metrics["rmse"]),
        "test_mae": float(overall_metrics["mae"]),
        "test_r2": float(overall_metrics["r2"]),
        "dm_stat_vs_persistence": float(dm_result["dm_stat"]),
        "dm_p_value_vs_persistence": float(dm_result["p_value"]),
        "num_trades": int(backtest_summary["num_trades"]),
        "net_pnl": float(backtest_summary["net_pnl"]),
        "hit_rate": float(backtest_summary["hit_rate"]),
        "max_drawdown": float(backtest_summary["max_drawdown"]),
    }
    if architecture == "xlstm":
        summary_row["num_blocks"] = int(config["model"].get("num_blocks", 2))
        summary_row["embedding_dim"] = int(config["model"].get("embedding_dim", 128))
    elif architecture == "transformer":
        summary_row["num_layers"] = int(config["model"].get("num_layers", 2))
        summary_row["embedding_dim"] = int(config["model"].get("embedding_dim", 128))
        summary_row["num_heads"] = int(config["model"].get("num_heads", 4))
        summary_row["ffn_dim"] = int(config["model"].get("ffn_dim", 256))
    else:
        summary_row["num_layers"] = int(config["model"]["num_layers"])
        summary_row["hidden_size"] = int(config["model"]["hidden_size"])
    pd.DataFrame([summary_row]).to_csv(output_root / "walkforward_summary.csv", index=False)

    lines = [
        f"# Walk-Forward Summary: {tag}",
        "",
        f"- Architecture: `{architecture}`",
        f"- Folds: `{len(splits)}`",
        f"- Stitched test RMSE: `{float(overall_metrics['rmse']):.6f}`",
        f"- Stitched test R^2: `{float(overall_metrics['r2']):.6f}`",
        f"- Backtest net PnL: `{float(backtest_summary['net_pnl']):.6f}`",
        "",
        "## Fold Metrics",
        "",
        frame_to_markdown(fold_frame.round(6)),
        "",
    ]
    if architecture == "xlstm":
        lines.insert(3, f"- Embedding dim: `{int(config['model'].get('embedding_dim', 128))}`")
        lines.insert(4, f"- Blocks: `{int(config['model'].get('num_blocks', 2))}`")
    elif architecture == "transformer":
        lines.insert(3, f"- Layers: `{int(config['model'].get('num_layers', 2))}`")
        lines.insert(4, f"- Embedding dim: `{int(config['model'].get('embedding_dim', 128))}`")
        lines.insert(5, f"- Heads: `{int(config['model'].get('num_heads', 4))}`")
    else:
        lines.insert(3, f"- Layers: `{int(config['model']['num_layers'])}`")
        lines.insert(4, f"- Hidden size: `{int(config['model']['hidden_size'])}`")
    (output_root / "walkforward_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
