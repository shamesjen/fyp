from __future__ import annotations

import argparse
import copy
import json
import os
import re
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
import pandas as pd

from src.data.csv_panel_loader import load_panel, load_underlying_csv
from src.data.feature_engineering import build_sequence_dataset
from src.utils.config import load_yaml_config
from src.utils.io import save_dataset_bundle, save_json


DEFAULT_SEQ_LENGTHS = [6, 12, 48, 84, 168]
DEFAULT_HORIZONS = [1, 12, 24]
DEFAULT_ARCHITECTURES = ["2x128", "3x128", "2x256", "3x256"]

BASE_DATA_CONFIG = ROOT / "configs" / "data_spy_5min_walkforward_live.yaml"
BASE_LSTM_CONFIG = ROOT / "configs" / "walkforward_lstm_5min_live.yaml"
BASE_BASELINE_CONFIG = ROOT / "configs" / "walkforward_baselines_5min_live.yaml"

ABLATION_ROOT = ROOT / "artifacts" / "ablations" / "5min_walkforward"
REPORT_ROOT = ABLATION_ROOT / "reports"
DATASET_ROOT = ROOT / "data" / "processed"


def architecture_token(num_layers: int, hidden_size: int) -> str:
    return f"l{num_layers}_h{hidden_size}"


def parse_architecture(spec: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)x(\d+)", spec.strip())
    if match is None:
        raise ValueError(f"Invalid architecture spec '{spec}'. Use format <layers>x<hidden>, e.g. 2x128.")
    return int(match.group(1)), int(match.group(2))


def dataset_tag(seq_len: int, horizon: int) -> str:
    return f"spy_5min_walkforward_seq{seq_len}_h{horizon}"


def lstm_tag(seq_len: int, horizon: int, num_layers: int, hidden_size: int) -> str:
    return f"seq{seq_len}_h{horizon}_l{num_layers}_h{hidden_size}"


def baseline_tag(seq_len: int, horizon: int) -> str:
    return f"h{horizon}_seq{seq_len}"


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


def run_command(args: list[str], dry_run: bool) -> None:
    pretty = " ".join(args)
    print(f"[RUN] {pretty}")
    if dry_run:
        return
    subprocess.run(args, cwd=ROOT, check=True)


def ensure_live_data(rebuild_live_data: bool, dry_run: bool) -> None:
    base_config = load_yaml_config(BASE_DATA_CONFIG)
    raw_underlying = ROOT / base_config["paths"]["raw_underlying_path"]
    raw_panel = ROOT / base_config["paths"]["raw_options_panel_path"]
    dataset_path = ROOT / base_config["paths"]["dataset_path"]
    if rebuild_live_data or not raw_underlying.exists() or not raw_panel.exists() or not dataset_path.exists():
        run_command(
            [
                sys.executable,
                "scripts/build_daily_dataset.py",
                "--config",
                str(BASE_DATA_CONFIG.relative_to(ROOT)),
            ],
            dry_run=dry_run,
        )


def load_base_frames() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    base_config = load_yaml_config(BASE_DATA_CONFIG)
    panel = load_panel(ROOT / base_config["paths"]["raw_options_panel_path"])
    underlying = load_underlying_csv(ROOT / base_config["paths"]["raw_underlying_path"])
    return base_config, panel, underlying


def build_dataset_variant(
    *,
    base_config: dict[str, Any],
    panel: pd.DataFrame,
    underlying: pd.DataFrame,
    seq_len: int,
    horizon: int,
    force: bool,
    dry_run: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    tag = dataset_tag(seq_len=seq_len, horizon=horizon)
    dataset_path = DATASET_ROOT / f"{tag}.npz"
    metadata_path = DATASET_ROOT / f"{tag}.metadata.json"
    if dataset_path.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return dataset_path, metadata_path, metadata
    if dry_run:
        return dataset_path, metadata_path, {}

    config = copy.deepcopy(base_config)
    config["data"]["seq_len"] = int(seq_len)
    config.setdefault("supervision", {})
    config["supervision"]["mode"] = "fixed_horizon"
    config["supervision"]["target_shift"] = int(horizon)

    bundle = build_sequence_dataset(panel_df=panel, underlying_df=underlying, config=config)
    save_dataset_bundle(bundle, dataset_path)
    save_json(bundle.metadata, metadata_path)
    return dataset_path, metadata_path, bundle.metadata


def maybe_run_baselines(
    *,
    dataset_path: Path,
    seq_len: int,
    horizon: int,
    signal_threshold: float,
    force: bool,
    dry_run: bool,
) -> None:
    tag = baseline_tag(seq_len=seq_len, horizon=horizon)
    output_dir = ABLATION_ROOT / "baselines" / tag
    summary_path = output_dir / "baseline_walkforward_summary.csv"
    if summary_path.exists() and not force:
        print(f"[SKIP] Baselines already exist for {tag}")
        return

    run_command(
        [
            sys.executable,
            "scripts/run_baselines_walkforward.py",
            "--config",
            str(BASE_BASELINE_CONFIG.relative_to(ROOT)),
            "--dataset-path",
            str(dataset_path.relative_to(ROOT)),
            "--output-dir",
            str((ABLATION_ROOT / "baselines").relative_to(ROOT)),
            "--holding-period",
            str(horizon),
            "--signal-threshold",
            str(signal_threshold),
            "--tag",
            tag,
        ],
        dry_run=dry_run,
    )


def maybe_run_lstm(
    *,
    dataset_path: Path,
    seq_len: int,
    horizon: int,
    num_layers: int,
    hidden_size: int,
    signal_threshold: float,
    force: bool,
    dry_run: bool,
) -> None:
    tag = lstm_tag(seq_len=seq_len, horizon=horizon, num_layers=num_layers, hidden_size=hidden_size)
    output_dir = ABLATION_ROOT / "lstm" / tag
    summary_path = output_dir / "walkforward_metrics.json"
    if summary_path.exists() and not force:
        print(f"[SKIP] LSTM already exists for {tag}")
        return

    run_command(
        [
            sys.executable,
            "scripts/run_lstm_walkforward.py",
            "--config",
            str(BASE_LSTM_CONFIG.relative_to(ROOT)),
            "--dataset-path",
            str(dataset_path.relative_to(ROOT)),
            "--output-dir",
            str((ABLATION_ROOT / "lstm").relative_to(ROOT)),
            "--holding-period",
            str(horizon),
            "--signal-threshold",
            str(signal_threshold),
            "--num-layers",
            str(num_layers),
            "--hidden-size",
            str(hidden_size),
            "--tag",
            tag,
        ],
        dry_run=dry_run,
    )


def merge_equity_curves(series_map: dict[str, Path], output_csv: Path, output_png: Path, title: str) -> None:
    frames: list[pd.DataFrame] = []
    for label, path in series_map.items():
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        if "cumulative_pnl" not in frame.columns:
            continue
        frames.append(frame[["date", "cumulative_pnl"]].rename(columns={"cumulative_pnl": label}))

    if not frames:
        return

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")
    merged = merged.sort_values("date").ffill().fillna(0.0)
    merged.to_csv(output_csv, index=False)

    plot_columns = [column for column in merged.columns if column != "date"][:12]
    fig, ax = plt.subplots(figsize=(10, 5))
    for column in plot_columns:
        ax.plot(merged["date"], merged[column], label=column)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(alpha=0.2)
    if plot_columns:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def collect_results(
    *,
    seq_lengths: list[int],
    horizons: list[int],
    architectures: list[tuple[int, int]],
    signal_threshold: float,
) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    dataset_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    lstm_rows: list[dict[str, Any]] = []
    baseline_equity_paths: dict[str, Path] = {}
    lstm_equity_paths: dict[str, Path] = {}

    for seq_len in seq_lengths:
        for horizon in horizons:
            metadata_path = DATASET_ROOT / f"{dataset_tag(seq_len, horizon)}.metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                dataset_rows.append(
                    {
                        "seq_len": seq_len,
                        "horizon": horizon,
                        "num_samples": int(metadata["num_samples"]),
                        "grid_size": int(metadata["grid_size"]),
                        "feature_count": len(metadata["feature_columns"]),
                    }
                )

            baseline_root = ABLATION_ROOT / "baselines" / baseline_tag(seq_len=seq_len, horizon=horizon)
            baseline_summary_path = baseline_root / "baseline_walkforward_summary.csv"
            if baseline_summary_path.exists():
                frame = pd.read_csv(baseline_summary_path)
                frame["seq_len"] = seq_len
                frame["horizon"] = horizon
                baseline_rows.extend(frame.to_dict(orient="records"))
                for model_name in frame["model"].astype(str):
                    equity_path = baseline_root / model_name / "backtest" / "backtest_trades.csv"
                    baseline_equity_paths[f"seq{seq_len}_h{horizon}_{model_name}"] = equity_path

            for num_layers, hidden_size in architectures:
                tag = lstm_tag(seq_len=seq_len, horizon=horizon, num_layers=num_layers, hidden_size=hidden_size)
                metrics_path = ABLATION_ROOT / "lstm" / tag / "walkforward_metrics.json"
                if not metrics_path.exists():
                    continue
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                stitched = payload.get("stitched_test", {})
                backtest = payload.get("backtest", {})
                lstm_rows.append(
                    {
                        "tag": tag,
                        "seq_len": seq_len,
                        "horizon": horizon,
                        "num_layers": int(num_layers),
                        "hidden_size": int(hidden_size),
                        "signal_threshold": signal_threshold,
                        "num_folds": int(payload.get("walkforward", {}).get("num_folds", 0)),
                        "test_rmse": float(stitched.get("rmse", 0.0)),
                        "test_mae": float(stitched.get("mae", 0.0)),
                        "test_r2": float(stitched.get("r2", 0.0)),
                        "dm_stat_vs_persistence": float(payload.get("dm_vs_persistence", {}).get("dm_stat", 0.0)),
                        "dm_p_value_vs_persistence": float(payload.get("dm_vs_persistence", {}).get("p_value", 0.0)),
                        "num_trades": int(backtest.get("num_trades", 0)),
                        "net_pnl": float(backtest.get("net_pnl", 0.0)),
                        "sharpe_annualized": float(backtest.get("sharpe_annualized", 0.0)),
                        "sortino_annualized": float(backtest.get("sortino_annualized", 0.0)),
                        "hit_rate": float(backtest.get("hit_rate", 0.0)),
                        "turnover": float(backtest.get("turnover", 0.0)),
                        "max_drawdown": float(backtest.get("max_drawdown", 0.0)),
                        "long_trades": int(backtest.get("long_trades", 0)),
                        "short_trades": int(backtest.get("short_trades", 0)),
                        "signal_realized_corr": float(backtest.get("signal_realized_corr", 0.0)),
                        "edge_sign_accuracy": float(backtest.get("edge_sign_accuracy", 0.0)),
                    }
                )
                lstm_equity_paths[tag] = ABLATION_ROOT / "lstm" / tag / "backtest" / "backtest_trades.csv"

    dataset_frame = pd.DataFrame(dataset_rows)
    if not dataset_frame.empty:
        dataset_frame = dataset_frame.sort_values(["horizon", "seq_len"]).reset_index(drop=True)
    baseline_frame = pd.DataFrame(baseline_rows)
    lstm_frame = pd.DataFrame(lstm_rows)

    if not dataset_frame.empty:
        dataset_frame.to_csv(REPORT_ROOT / "dataset_manifest.csv", index=False)

    if not baseline_frame.empty:
        baseline_frame = baseline_frame.sort_values(["horizon", "seq_len", "test_rmse", "model"]).reset_index(drop=True)
        baseline_frame.to_csv(REPORT_ROOT / "baseline_grid_summary.csv", index=False)
        best_baseline = (
            baseline_frame.sort_values(["horizon", "seq_len", "test_rmse"], ascending=[True, True, True])
            .groupby(["horizon", "seq_len"], as_index=False)
            .first()
        )
        best_baseline.to_csv(REPORT_ROOT / "baseline_best_by_dataset.csv", index=False)
        merge_equity_curves(
            baseline_equity_paths,
            REPORT_ROOT / "baseline_equity_curves.csv",
            REPORT_ROOT / "baseline_equity_curves.png",
            title="5-Minute Walk-Forward Baseline Equity Curves",
        )
    else:
        best_baseline = pd.DataFrame()

    if not lstm_frame.empty:
        lstm_frame = lstm_frame.sort_values(["net_pnl", "test_rmse"], ascending=[False, True]).reset_index(drop=True)
        lstm_frame.to_csv(REPORT_ROOT / "lstm_grid_summary.csv", index=False)
        best_lstm_by_dataset = (
            lstm_frame.sort_values(["horizon", "seq_len", "net_pnl"], ascending=[True, True, False])
            .groupby(["horizon", "seq_len"], as_index=False)
            .first()
        )
        best_lstm_by_dataset.to_csv(REPORT_ROOT / "lstm_best_by_dataset.csv", index=False)
        top_lstm = lstm_frame.sort_values("net_pnl", ascending=False).head(15)
        top_lstm.to_csv(REPORT_ROOT / "lstm_top_runs.csv", index=False)
        merge_equity_curves(
            lstm_equity_paths,
            REPORT_ROOT / "lstm_equity_curves.csv",
            REPORT_ROOT / "lstm_equity_curves.png",
            title="5-Minute Walk-Forward LSTM Equity Curves",
        )
    else:
        best_lstm_by_dataset = pd.DataFrame()
        top_lstm = pd.DataFrame()

    lines = [
        "# 5-Minute Walk-Forward Ablation",
        "",
        "## Setup",
        "",
        f"- Seq lengths: `{seq_lengths}`",
        f"- Horizons (bars): `{horizons}`",
        f"- Architectures: `{[architecture_token(l, h) for l, h in architectures]}`",
        f"- Signal threshold: `{signal_threshold}`",
        f"- Base data config: `{BASE_DATA_CONFIG.relative_to(ROOT)}`",
        f"- Base LSTM config: `{BASE_LSTM_CONFIG.relative_to(ROOT)}`",
        f"- Base baseline config: `{BASE_BASELINE_CONFIG.relative_to(ROOT)}`",
        "",
    ]
    if not dataset_frame.empty:
        lines.extend(
            [
                "## Dataset Variants",
                "",
                frame_to_markdown(dataset_frame),
                "",
            ]
        )
    if not best_lstm_by_dataset.empty:
        lines.extend(
            [
                "## Best LSTM Per Dataset",
                "",
                frame_to_markdown(best_lstm_by_dataset.round(6)),
                "",
            ]
        )
    if not top_lstm.empty:
        lines.extend(
            [
                "## Top LSTM Runs By Net PnL",
                "",
                frame_to_markdown(top_lstm.head(10).round(6)),
                "",
            ]
        )
    if not best_baseline.empty:
        lines.extend(
            [
                "## Best Baseline Per Dataset",
                "",
                frame_to_markdown(best_baseline.round(6)),
                "",
            ]
        )
    lines.extend(
        [
            "## Output Files",
            "",
            "- `artifacts/ablations/5min_walkforward/reports/lstm_grid_summary.csv`",
            "- `artifacts/ablations/5min_walkforward/reports/baseline_grid_summary.csv`",
            "- `artifacts/ablations/5min_walkforward/reports/lstm_equity_curves.csv`",
            "- `artifacts/ablations/5min_walkforward/reports/baseline_equity_curves.csv`",
            "",
        ]
    )
    (REPORT_ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full 5-minute walk-forward ablation suite.")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=DEFAULT_SEQ_LENGTHS)
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--architectures", nargs="+", default=DEFAULT_ARCHITECTURES)
    parser.add_argument("--signal-threshold", type=float, default=0.0015)
    parser.add_argument("--rebuild-live-data", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    architectures = [parse_architecture(spec) for spec in args.architectures]

    planned_datasets = len(args.seq_lengths) * len(args.horizons)
    planned_lstm = planned_datasets * len(architectures) if not args.skip_lstm else 0
    planned_baselines = planned_datasets if not args.skip_baselines else 0
    print(
        f"[INFO] Planned dataset variants={planned_datasets}, baseline runs={planned_baselines}, "
        f"lstm runs={planned_lstm}"
    )

    if not args.summary_only:
        ensure_live_data(rebuild_live_data=args.rebuild_live_data, dry_run=args.dry_run)
        base_config, panel, underlying = load_base_frames()

        for seq_len in args.seq_lengths:
            for horizon in args.horizons:
                dataset_path, _, metadata = build_dataset_variant(
                    base_config=base_config,
                    panel=panel,
                    underlying=underlying,
                    seq_len=seq_len,
                    horizon=horizon,
                    force=args.force,
                    dry_run=args.dry_run,
                )
                print(
                    f"[DATASET] seq_len={seq_len} horizon={horizon} samples={metadata.get('num_samples', '?')} "
                    f"path={dataset_path.relative_to(ROOT)}"
                )
                if not args.skip_baselines:
                    maybe_run_baselines(
                        dataset_path=dataset_path,
                        seq_len=seq_len,
                        horizon=horizon,
                        signal_threshold=args.signal_threshold,
                        force=args.force,
                        dry_run=args.dry_run,
                    )
                if not args.skip_lstm:
                    for num_layers, hidden_size in architectures:
                        maybe_run_lstm(
                            dataset_path=dataset_path,
                            seq_len=seq_len,
                            horizon=horizon,
                            num_layers=num_layers,
                            hidden_size=hidden_size,
                            signal_threshold=args.signal_threshold,
                            force=args.force,
                            dry_run=args.dry_run,
                        )

    if args.dry_run:
        return

    collect_results(
        seq_lengths=args.seq_lengths,
        horizons=args.horizons,
        architectures=architectures,
        signal_threshold=args.signal_threshold,
    )
    print(f"[DONE] Reports written under {REPORT_ROOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
