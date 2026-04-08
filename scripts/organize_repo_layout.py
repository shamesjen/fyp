from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def safe_reset_dir(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def relative_target(link_parent: Path, target: Path) -> str:
    return str(target.relative_to(ROOT) if target.is_absolute() and target.is_relative_to(ROOT) else target)


def make_symlink(link_path: Path, target: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    rel = Path(shutil.os.path.relpath(target, start=link_path.parent))
    link_path.symlink_to(rel)


def relocate_with_alias(old_rel: str, new_rel: str) -> None:
    old_path = ROOT / old_rel
    new_path = ROOT / new_rel
    new_path.parent.mkdir(parents=True, exist_ok=True)

    if old_path.exists() and not old_path.is_symlink():
        if new_path.exists():
            if new_path.is_dir() and not any(new_path.iterdir()):
                shutil.rmtree(new_path)
            else:
                raise FileExistsError(f"Refusing to overwrite populated destination {new_path}")
        old_path.rename(new_path)

    if new_path.exists() and not old_path.exists():
        make_symlink(old_path, new_path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def flatten(rows: list[tuple[str, Path, Path, str]]) -> list[dict[str, str]]:
    return [
        {
            "group": group,
            "alias_path": str(alias.relative_to(ROOT)),
            "source_path": str(source.relative_to(ROOT)),
            "notes": notes,
        }
        for group, alias, source, notes in rows
    ]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_artifact_layout() -> None:
    by_stage = ROOT / "artifacts" / "by_stage"
    current_final = ROOT / "artifacts" / "current_final"
    safe_reset_dir(by_stage)
    safe_reset_dir(current_final)

    links: list[tuple[str, Path, Path, str]] = []

    def add(group: str, alias: str, source: str, notes: str) -> None:
        alias_path = ROOT / alias
        source_path = ROOT / source
        if not source_path.exists():
            return
        make_symlink(alias_path, source_path)
        links.append((group, alias_path, source_path, notes))

    # Daily
    add("daily", "artifacts/by_stage/01_daily_mvp/00_sample_demo/baselines", "artifacts/baselines", "Offline-safe sample baseline outputs")
    add("daily", "artifacts/by_stage/01_daily_mvp/00_sample_demo/lstm", "artifacts/lstm", "Offline-safe sample LSTM outputs")
    add("daily", "artifacts/by_stage/01_daily_mvp/00_sample_demo/backtest", "artifacts/backtest", "Offline-safe sample backtest outputs")
    add("daily", "artifacts/by_stage/01_daily_mvp/01_live_spy_daily/baselines", "artifacts/live_baselines", "Daily live SPY baselines")
    add("daily", "artifacts/by_stage/01_daily_mvp/01_live_spy_daily/lstm", "artifacts/live_lstm", "Daily live SPY LSTM")
    add("daily", "artifacts/by_stage/01_daily_mvp/01_live_spy_daily/backtest", "artifacts/live_backtest", "Daily live SPY backtest")

    # Hourly
    add("hourly", "artifacts/by_stage/02_hourly_research/00_early_hourly_h1_seq24/baselines", "artifacts/live_hourly_baselines", "Initial hourly H1 baselines")
    add("hourly", "artifacts/by_stage/02_hourly_research/00_early_hourly_h1_seq24/lstm", "artifacts/live_hourly_lstm", "Initial hourly H1 LSTM")
    add("hourly", "artifacts/by_stage/02_hourly_research/00_early_hourly_h1_seq24/backtest", "artifacts/live_hourly_backtest", "Initial hourly H1 backtest")
    add("hourly", "artifacts/by_stage/02_hourly_research/01_hourly_h1_seq10/baselines", "artifacts/live_hourly_h1_baselines", "Hourly H1 seq10 baselines")
    add("hourly", "artifacts/by_stage/02_hourly_research/01_hourly_h1_seq10/lstm", "artifacts/live_hourly_h1_lstm", "Hourly H1 seq10 LSTM")
    add("hourly", "artifacts/by_stage/02_hourly_research/01_hourly_h1_seq10/backtest", "artifacts/live_hourly_h1_backtest", "Hourly H1 seq10 backtest")
    add("hourly", "artifacts/by_stage/02_hourly_research/02_hourly_h7_rolling/baselines", "artifacts/live_hourly_nextday_baselines", "Hourly +7 rolling baselines")
    add("hourly", "artifacts/by_stage/02_hourly_research/02_hourly_h7_rolling/lstm", "artifacts/live_hourly_nextday_lstm", "Hourly +7 rolling LSTM")
    add("hourly", "artifacts/by_stage/02_hourly_research/02_hourly_h7_rolling/backtest", "artifacts/live_hourly_nextday_backtest", "Hourly +7 rolling backtest")
    add("hourly", "artifacts/by_stage/02_hourly_research/03_year_holdout/baselines", "artifacts/live_hourly_h1_year_baselines", "Year-long hourly baselines")
    add("hourly", "artifacts/by_stage/02_hourly_research/03_year_holdout/lstm_shuffle_on", "artifacts/live_hourly_h1_year_lstm", "Year-long hourly LSTM with train shuffle on")
    add("hourly", "artifacts/by_stage/02_hourly_research/03_year_holdout/lstm_shuffle_off", "artifacts/live_hourly_h1_year_lstm_shuffle_off", "Year-long hourly LSTM with train shuffle off")
    add("hourly", "artifacts/by_stage/02_hourly_research/03_year_holdout/backtest_shuffle_on", "artifacts/live_hourly_h1_year_backtest_shuffle_on", "Year-long hourly backtest, shuffle on")
    add("hourly", "artifacts/by_stage/02_hourly_research/03_year_holdout/backtest_shuffle_off", "artifacts/live_hourly_h1_year_backtest_shuffle_off", "Year-long hourly backtest, shuffle off")
    add("hourly", "artifacts/by_stage/02_hourly_research/04_capacity_sweep", "artifacts/experiments/year_long_capacity", "Hourly year-long capacity sweep")
    add("hourly", "artifacts/by_stage/02_hourly_research/05_hourly_walkforward_lstm", "artifacts/walkforward_experiments", "Hourly walk-forward LSTM seq/horizon/regularization studies")
    add("hourly", "artifacts/by_stage/02_hourly_research/06_hourly_walkforward_baselines", "artifacts/walkforward_baselines/experiments", "Hourly walk-forward baseline studies")
    add("hourly", "artifacts/by_stage/02_hourly_research/07_hourly_h1_ablation", "artifacts/ablations/hourly_h1", "Hourly H1 seq-length/threshold ablation")
    add("hourly", "artifacts/by_stage/02_hourly_research/08_hourly_master_walkforward", "artifacts/walkforward/hourly_h1/l2_h128", "Hourly walk-forward main LSTM run")

    # 5-minute
    add("5min", "artifacts/by_stage/03_5min_research/00_grid_ablation/lstm", "artifacts/ablations/5min_walkforward/lstm", "Full 5-minute LSTM grid")
    add("5min", "artifacts/by_stage/03_5min_research/00_grid_ablation/baselines", "artifacts/ablations/5min_walkforward/baselines", "5-minute baseline grids and refreshed finalist baselines")
    add("5min", "artifacts/by_stage/03_5min_research/00_grid_ablation/reports", "artifacts/ablations/5min_walkforward/reports", "5-minute ablation summaries")
    add("5min", "artifacts/by_stage/03_5min_research/01_finalists/threshold_sweep", "artifacts/ablations/5min_walkforward/threshold_sweep", "Threshold sweeps for frozen finalists")
    add("5min", "artifacts/by_stage/03_5min_research/01_finalists/execution_backtests", "artifacts/ablations/5min_walkforward/execution_backtests", "Execution-aware finalist backtests")
    add("5min", "artifacts/by_stage/03_5min_research/01_finalists/strategy_analysis", "artifacts/ablations/5min_walkforward/finalist_strategy_analysis", "Finalist tear-sheet metrics")
    add("5min", "artifacts/by_stage/03_5min_research/02_standardized/initial_matched", "artifacts/ablations/5min_walkforward/standardized", "Initial standardized comparisons")
    add("5min", "artifacts/by_stage/03_5min_research/02_standardized/refreshed_baselines", "artifacts/ablations/5min_walkforward/standardized_all_refreshed_baselines", "Standardized comparisons vs refreshed baselines")
    add("5min", "artifacts/by_stage/03_5min_research/03_robustness/final_additional_evaluations", "artifacts/reports/final_5min_additional_evaluations", "DM, bucket, regime, placebo, execution stress, vega, shape diagnostics")
    add("5min", "artifacts/by_stage/03_5min_research/03_robustness/best_model_vs_all_baselines", "artifacts/reports/best_model_vs_all_baselines_evaluations", "Best model vs all baseline appendix")
    add("5min", "artifacts/by_stage/03_5min_research/03_robustness/xlstm_vs_all_baselines", "artifacts/reports/xlstm_vs_all_baselines_evaluations", "xLSTM vs all baseline appendix")
    add("5min", "artifacts/by_stage/03_5min_research/04_xlstm_extension", "artifacts/experiments/xlstm_5min_best", "xLSTM extension runs")
    add("5min", "artifacts/by_stage/03_5min_research/05_multiseed_benchmark_4fold", "artifacts/experiments/multiseed_final_benchmark", "Original 4-fold multi-seed final benchmark")
    add("5min", "artifacts/by_stage/03_5min_research/06_multiseed_benchmark_overlap_7fold_final", "artifacts/experiments/multiseed_final_benchmark_overlap", "Definitive overlap-safe 7-fold final benchmark")

    # Writing
    add("writing", "artifacts/by_stage/04_written_outputs/reports", "artifacts/reports", "Master reports and derived markdown")
    add("writing", "artifacts/by_stage/04_written_outputs/thesis_assets", "thesis_report_assets", "Overleaf-ready thesis asset pack")

    # Current final pointers
    add("current_final", "artifacts/current_final/final_benchmark_overlap_7fold", "artifacts/experiments/multiseed_final_benchmark_overlap", "Definitive final benchmark")
    add("current_final", "artifacts/current_final/final_benchmark_report.md", "artifacts/reports/multiseed_final_benchmark_overlap.md", "Final benchmark markdown summary")
    add("current_final", "artifacts/current_final/master_report.md", "artifacts/reports/thesis_experiment_master_report.md", "Master experiment report")
    add("current_final", "artifacts/current_final/finalist_strategy_analysis", "artifacts/ablations/5min_walkforward/finalist_strategy_analysis", "Finalist tear sheet")
    add("current_final", "artifacts/current_final/final_robustness_suite", "artifacts/reports/final_5min_additional_evaluations", "Final 5-minute robustness suite")
    add("current_final", "artifacts/current_final/thesis_assets", "thesis_report_assets", "Thesis asset pack")

    artifact_readme = """
# Artifacts Layout

This directory now has a canonical navigation layer that groups experiment outputs by research phase without breaking the original saved paths.

Use these first:

- `artifacts/current_final/`
  Final benchmark, final robustness suite, master report, and thesis assets.
- `artifacts/by_stage/`
  Phase-based view of the full project from daily MVP through the final 5-minute benchmark.

Design choice:

- The original artifact paths are preserved so old reports, configs, and scripts keep working.
- The new grouped folders are symlink-based views on top of those original outputs.
- The definitive final benchmark is:
  `artifacts/by_stage/03_5min_research/06_multiseed_benchmark_overlap_7fold_final/`

Recommended navigation order:

1. `artifacts/current_final/`
2. `artifacts/by_stage/03_5min_research/`
3. `artifacts/by_stage/02_hourly_research/`
4. `artifacts/by_stage/01_daily_mvp/`
"""
    write_text(ROOT / "artifacts" / "README.md", artifact_readme)
    write_csv(
        ROOT / "artifacts" / "layout_index.csv",
        flatten(links),
        ["group", "alias_path", "source_path", "notes"],
    )


def canonicalize_live_artifacts() -> None:
    relocations = [
        ("artifacts/live_baselines", "artifacts/live/daily/live_spy_daily/baselines"),
        ("artifacts/live_lstm", "artifacts/live/daily/live_spy_daily/lstm"),
        ("artifacts/live_backtest", "artifacts/live/daily/live_spy_daily/backtest"),
        ("artifacts/live_hourly_baselines", "artifacts/live/hourly/00_early_h1_seq24/baselines"),
        ("artifacts/live_hourly_lstm", "artifacts/live/hourly/00_early_h1_seq24/lstm"),
        ("artifacts/live_hourly_backtest", "artifacts/live/hourly/00_early_h1_seq24/backtest"),
        ("artifacts/live_hourly_h1_baselines", "artifacts/live/hourly/01_h1_seq10/baselines"),
        ("artifacts/live_hourly_h1_lstm", "artifacts/live/hourly/01_h1_seq10/lstm"),
        ("artifacts/live_hourly_h1_backtest", "artifacts/live/hourly/01_h1_seq10/backtest"),
        ("artifacts/live_hourly_nextday_baselines", "artifacts/live/hourly/02_h7_rolling/baselines"),
        ("artifacts/live_hourly_nextday_lstm", "artifacts/live/hourly/02_h7_rolling/lstm"),
        ("artifacts/live_hourly_nextday_backtest", "artifacts/live/hourly/02_h7_rolling/backtest"),
        ("artifacts/live_hourly_h1_year_baselines", "artifacts/live/hourly/03_year_holdout/baselines"),
        ("artifacts/live_hourly_h1_year_lstm", "artifacts/live/hourly/03_year_holdout/lstm_shuffle_on"),
        ("artifacts/live_hourly_h1_year_lstm_shuffle_off", "artifacts/live/hourly/03_year_holdout/lstm_shuffle_off"),
        ("artifacts/live_hourly_h1_year_backtest_shuffle_on", "artifacts/live/hourly/03_year_holdout/backtest_shuffle_on"),
        ("artifacts/live_hourly_h1_year_backtest_shuffle_off", "artifacts/live/hourly/03_year_holdout/backtest_shuffle_off"),
        ("artifacts/live_hourly_h1_year_summary.md", "artifacts/live/hourly/03_year_holdout/summary.md"),
    ]
    for old_rel, new_rel in relocations:
        relocate_with_alias(old_rel, new_rel)

    live_readme = """
# Canonical Live Artifact Layout

This directory is the canonical home for all live-data experiment outputs.

- `daily/live_spy_daily/`
  Daily live SPY runs.
- `hourly/00_early_h1_seq24/`
  Initial hourly H1 study.
- `hourly/01_h1_seq10/`
  Hourly H1 seq10 study.
- `hourly/02_h7_rolling/`
  Rolling +7-bar hourly study.
- `hourly/03_year_holdout/`
  Year-long hourly holdout with shuffle-on/off variants.

The old `artifacts/live_*` names are preserved as symlink aliases for backward compatibility.
"""
    write_text(ROOT / "artifacts" / "live" / "README.md", live_readme)


def build_config_layout() -> None:
    by_stage = ROOT / "configs" / "by_stage"
    safe_reset_dir(by_stage)
    links: list[tuple[str, Path, Path, str]] = []

    def add(group: str, alias: str, source: str, notes: str) -> None:
        alias_path = ROOT / alias
        source_path = ROOT / source
        if not source_path.exists():
            return
        make_symlink(alias_path, source_path)
        links.append((group, alias_path, source_path, notes))

    # Daily
    add("daily", "configs/by_stage/01_daily/data_sample.yaml", "configs/data_spy_daily.yaml", "Offline-safe daily dataset build")
    add("daily", "configs/by_stage/01_daily/data_live.yaml", "configs/data_spy_daily_live.yaml", "Daily live dataset build")
    add("daily", "configs/by_stage/01_daily/train_baselines_sample.yaml", "configs/train_baselines.yaml", "Daily sample baselines")
    add("daily", "configs/by_stage/01_daily/train_baselines_live.yaml", "configs/train_baselines_live.yaml", "Daily live baselines")
    add("daily", "configs/by_stage/01_daily/train_lstm_sample.yaml", "configs/train_lstm.yaml", "Daily sample LSTM")
    add("daily", "configs/by_stage/01_daily/train_lstm_live.yaml", "configs/train_lstm_live.yaml", "Daily live LSTM")
    add("daily", "configs/by_stage/01_daily/backtest_sample.yaml", "configs/backtest_demo.yaml", "Daily sample backtest")
    add("daily", "configs/by_stage/01_daily/backtest_live.yaml", "configs/backtest_demo_live.yaml", "Daily live backtest")

    # Hourly
    add("hourly", "configs/by_stage/02_hourly/data_early_h1.yaml", "configs/data_spy_hourly_live.yaml", "Initial hourly H1 data")
    add("hourly", "configs/by_stage/02_hourly/data_h1_seq10.yaml", "configs/data_spy_hourly_h1_live.yaml", "Hourly H1 seq10 data")
    add("hourly", "configs/by_stage/02_hourly/data_h7.yaml", "configs/data_spy_hourly_nextday_live.yaml", "Hourly +7 rolling data")
    add("hourly", "configs/by_stage/02_hourly/data_year_holdout.yaml", "configs/data_spy_hourly_h1_year_live.yaml", "Year-long hourly holdout data")
    add("hourly", "configs/by_stage/02_hourly/train_h1_baselines.yaml", "configs/train_baselines_hourly_h1_live.yaml", "Hourly H1 baselines")
    add("hourly", "configs/by_stage/02_hourly/train_h1_lstm.yaml", "configs/train_lstm_hourly_h1_live.yaml", "Hourly H1 LSTM")
    add("hourly", "configs/by_stage/02_hourly/walkforward_lstm.yaml", "configs/walkforward_lstm_hourly_h1_live.yaml", "Hourly walk-forward LSTM")
    add("hourly", "configs/by_stage/02_hourly/walkforward_baselines.yaml", "configs/walkforward_baselines_hourly_h1_live.yaml", "Hourly walk-forward baselines")

    # 5-minute
    add("5min", "configs/by_stage/03_5min/data_live.yaml", "configs/data_spy_5min_walkforward_live.yaml", "Main 5-minute permissive panel")
    add("5min", "configs/by_stage/03_5min/data_strict.yaml", "configs/data_spy_5min_walkforward_strict.yaml", "Strict 5-minute panel variant")
    add("5min", "configs/by_stage/03_5min/walkforward_lstm.yaml", "configs/walkforward_lstm_5min_live.yaml", "Main 5-minute LSTM walk-forward")
    add("5min", "configs/by_stage/03_5min/walkforward_xlstm.yaml", "configs/walkforward_xlstm_5min_best.yaml", "Main 5-minute xLSTM walk-forward")
    add("5min", "configs/by_stage/03_5min/walkforward_baselines.yaml", "configs/walkforward_baselines_5min_live.yaml", "Main 5-minute baselines walk-forward")
    add("5min", "configs/by_stage/03_5min/finalists.yaml", "configs/finalists_5min.yaml", "Frozen 5-minute finalists")
    add("5min", "configs/by_stage/03_5min/threshold_sweep.yaml", "configs/threshold_sweep_5min.yaml", "Finalist threshold sweep")
    add("5min", "configs/by_stage/03_5min/standardized_all_refreshed_baselines.yaml", "configs/standardized_candidates_5min_all_refreshed_baselines.yaml", "Standardized finalist comparison vs refreshed baselines")
    add("5min", "configs/by_stage/03_5min/backtest_execution.yaml", "configs/backtest_execution_5min.yaml", "Execution-aware 5-minute backtest")

    # Final benchmarks
    add("final_benchmarks", "configs/by_stage/04_final_benchmarks/multiseed_4fold.yaml", "configs/multiseed_final_benchmark.yaml", "Original 4-fold multi-seed final benchmark")
    add("final_benchmarks", "configs/by_stage/04_final_benchmarks/multiseed_overlap_7fold.yaml", "configs/multiseed_final_benchmark_overlap.yaml", "Definitive overlap-safe 7-fold final benchmark")
    add("final_benchmarks", "configs/by_stage/04_final_benchmarks/best_model_vs_all_baselines.yaml", "configs/best_model_vs_all_baselines_evaluations.yaml", "Best model vs all baselines")
    add("final_benchmarks", "configs/by_stage/04_final_benchmarks/final_robustness_suite.yaml", "configs/final_5min_additional_evaluations.yaml", "Final 5-minute robustness suite")

    config_readme = """
# Config Layout

The original config filenames are preserved, but `configs/by_stage/` gives a cleaner grouped view:

- `01_daily/`
- `02_hourly/`
- `03_5min/`
- `04_final_benchmarks/`

Use `04_final_benchmarks/multiseed_overlap_7fold.yaml` as the definitive final benchmark config.
"""
    write_text(ROOT / "configs" / "README.md", config_readme)
    write_csv(
        ROOT / "configs" / "config_index.csv",
        flatten(links),
        ["group", "alias_path", "source_path", "notes"],
    )


def build_script_layout() -> None:
    by_purpose = ROOT / "scripts" / "by_purpose"
    safe_reset_dir(by_purpose)
    links: list[tuple[str, Path, Path, str]] = []

    def add(group: str, alias: str, source: str, notes: str) -> None:
        alias_path = ROOT / alias
        source_path = ROOT / source
        if not source_path.exists():
            return
        make_symlink(alias_path, source_path)
        links.append((group, alias_path, source_path, notes))

    add("data", "scripts/by_purpose/01_data/build_dataset.py", "scripts/build_daily_dataset.py", "Main dataset builder")
    add("data", "scripts/by_purpose/01_data/download_underlying.py", "scripts/download_underlying_data.py", "Underlying downloader")
    add("data", "scripts/by_purpose/01_data/download_options.py", "scripts/download_alpaca_options.py", "Alpaca options downloader")
    add("data", "scripts/by_purpose/01_data/build_from_local_raw.py", "scripts/build_dataset_from_local_raw.py", "Dataset builder from saved raw data")

    add("training", "scripts/by_purpose/02_training/run_baselines_single_split.py", "scripts/run_baselines.py", "Single-split baseline training")
    add("training", "scripts/by_purpose/02_training/run_lstm_single_split.py", "scripts/run_lstm.py", "Single-split LSTM training")
    add("training", "scripts/by_purpose/02_training/run_baselines_walkforward.py", "scripts/run_baselines_walkforward.py", "Walk-forward baseline training")
    add("training", "scripts/by_purpose/02_training/run_lstm_walkforward.py", "scripts/run_lstm_walkforward.py", "Walk-forward LSTM/xLSTM training")

    add("benchmarks", "scripts/by_purpose/03_benchmarks/run_5min_grid_ablation.py", "scripts/run_5min_walkforward_ablation.py", "5-minute grid ablation")
    add("benchmarks", "scripts/by_purpose/03_benchmarks/run_hourly_ablation.py", "scripts/run_hourly_h1_ablation.py", "Hourly H1 ablation")
    add("benchmarks", "scripts/by_purpose/03_benchmarks/run_year_capacity_sweep.py", "scripts/run_year_long_lstm_capacity_sweep.py", "Year-long hourly capacity sweep")
    add("benchmarks", "scripts/by_purpose/03_benchmarks/run_standardized_comparison.py", "scripts/run_standardized_5min_comparison.py", "Standardized common-window comparison")
    add("benchmarks", "scripts/by_purpose/03_benchmarks/run_threshold_sweep.py", "scripts/run_finalist_threshold_sweep.py", "Finalist threshold sweep")
    add("benchmarks", "scripts/by_purpose/03_benchmarks/run_finalist_baseline_refresh.py", "scripts/run_finalist_baselines_refresh.py", "Refresh baselines for finalist datasets")
    add("benchmarks", "scripts/by_purpose/03_benchmarks/run_multiseed_benchmark.py", "scripts/run_multiseed_final_benchmark.py", "Multi-seed final benchmark")
    add("benchmarks", "scripts/by_purpose/03_benchmarks/run_best_model_vs_all_baselines.py", "scripts/run_best_model_vs_all_baselines_evaluations.py", "Best model vs all baselines appendix")
    add("benchmarks", "scripts/by_purpose/03_benchmarks/run_final_robustness_suite.py", "scripts/run_final_5min_additional_evaluations.py", "Final robustness suite")

    add("backtests", "scripts/by_purpose/04_backtests/run_demo_backtest.py", "scripts/run_backtest_demo.py", "Simple demo backtest")
    add("backtests", "scripts/by_purpose/04_backtests/run_execution_backtest.py", "scripts/run_execution_backtest.py", "Execution-aware backtest")
    add("backtests", "scripts/by_purpose/04_backtests/run_finalist_execution_backtests.py", "scripts/run_finalist_execution_backtests.py", "Batch execution backtests for finalists")
    add("backtests", "scripts/by_purpose/04_backtests/run_finalist_strategy_analysis.py", "scripts/run_finalist_strategy_analysis.py", "Finalist tear-sheet analysis")

    add("reporting", "scripts/by_purpose/05_reporting/build_thesis_assets.py", "scripts/build_thesis_report_assets.py", "Build thesis asset pack")
    add("reporting", "scripts/by_purpose/05_reporting/print_run_summary.py", "scripts/print_run_summary.py", "Quick summary printer")
    add("reporting", "scripts/by_purpose/05_reporting/fix_thesis_tex.py", "scripts/fix_final_report_with_assets.py", "Patch thesis tex body")

    scripts_readme = """
# Scripts Layout

The executable scripts remain in the repo root `scripts/`, but `scripts/by_purpose/` now groups them by use:

- `01_data/`
- `02_training/`
- `03_benchmarks/`
- `04_backtests/`
- `05_reporting/`

If you are rerunning the final thesis benchmark, start with:

- `scripts/by_purpose/03_benchmarks/run_multiseed_benchmark.py`
"""
    write_text(ROOT / "scripts" / "README.md", scripts_readme)
    write_csv(
        ROOT / "scripts" / "script_index.csv",
        flatten(links),
        ["group", "alias_path", "source_path", "notes"],
    )


def update_root_readme() -> None:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    marker = "## Repo Layout\n"
    insert = """
## Repo Navigation

The original experiment files are preserved in place, but the repo now also includes grouped navigation layers:

- [artifacts/README.md](artifacts/README.md): canonical experiment outputs grouped by research phase
- [configs/README.md](configs/README.md): configs grouped by stage
- [scripts/README.md](scripts/README.md): scripts grouped by purpose

If you only want the final thesis result set, start in:

- `artifacts/current_final/`
"""
    if "## Repo Navigation" not in text and marker in text:
        text = text.replace(marker, insert.strip() + "\n\n" + marker, 1)
        readme.write_text(text, encoding="utf-8")


def main() -> None:
    canonicalize_live_artifacts()
    build_artifact_layout()
    build_config_layout()
    build_script_layout()
    update_root_readme()
    print("Organized artifacts, configs, and scripts into grouped navigation views.")


if __name__ == "__main__":
    main()
