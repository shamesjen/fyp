from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.utils.config import load_yaml_config, resolve_path


def threshold_to_tag(value: float) -> str:
    return f"threshold_{value:.4f}".replace(".", "p")


def annualized_sharpe(net_pnl: np.ndarray, annualization_factor: float) -> float:
    if len(net_pnl) <= 1 or annualization_factor <= 0:
        return 0.0
    std = float(np.std(net_pnl, ddof=1))
    if std <= 0:
        return 0.0
    return float(np.mean(net_pnl) / std * math.sqrt(annualization_factor))


def circular_block_indices(num_rows: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if num_rows <= 0:
        return np.array([], dtype=int)
    if block_length <= 1:
        return rng.integers(0, num_rows, size=num_rows, endpoint=False)
    needed_blocks = int(math.ceil(num_rows / block_length))
    starts = rng.integers(0, num_rows, size=needed_blocks, endpoint=False)
    pieces = [((start + np.arange(block_length)) % num_rows) for start in starts]
    return np.concatenate(pieces)[:num_rows]


def quantile_bounds(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lower = float(np.quantile(values, alpha / 2.0))
    upper = float(np.quantile(values, 1.0 - alpha / 2.0))
    return lower, upper


def load_model_entries(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    summary_csv = resolve_path(cfg["paths"]["model_summary_csv"])
    threshold_root = resolve_path(cfg["paths"]["threshold_backtest_root"])
    summary = pd.read_csv(summary_csv)

    include_families = {str(value) for value in cfg.get("selection", {}).get("include_families", [])}
    include_names = {str(value) for value in cfg.get("selection", {}).get("include_names", [])}
    exclude_names = {str(value) for value in cfg.get("selection", {}).get("exclude_names", [])}
    seed = int(cfg["selection"].get("seed", 7))

    entries: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        name = str(row["name"])
        family = str(row["family"])
        if include_families and family not in include_families:
            continue
        if include_names and name not in include_names:
            continue
        if name in exclude_names:
            continue
        threshold = float(row["threshold"])
        backtest_dir = threshold_root / f"seed_{seed}" / name / threshold_to_tag(threshold)
        if not backtest_dir.is_dir():
            raise FileNotFoundError(f"Missing backtest directory for {name}: {backtest_dir}")
        entries.append(
            {
                "name": name,
                "family": family,
                "threshold": threshold,
                "backtest_dir": backtest_dir,
            }
        )
    if not entries:
        raise ValueError("No model entries selected for bootstrap significance analysis.")
    return entries


def bootstrap_model_series(
    entries: list[dict[str, Any]],
    replications: int,
    block_length: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    rng = np.random.default_rng(seed)

    model_state: dict[str, dict[str, Any]] = {}
    for entry in entries:
        trades = pd.read_csv(entry["backtest_dir"] / "backtest_trades.csv")
        summary = pd.read_csv(entry["backtest_dir"] / "backtest_summary.csv").iloc[0]
        series = trades["net_pnl"].to_numpy(dtype=float)
        annualization = float(summary["annualization_factor"])
        model_state[entry["name"]] = {
            "family": entry["family"],
            "threshold": float(entry["threshold"]),
            "series": series,
            "annualization_factor": annualization,
            "observed_net_pnl": float(series.sum()),
            "observed_sharpe": annualized_sharpe(series, annualization),
            "observed_num_rows": int(len(series)),
            "observed_num_trades": int(summary["num_trades"]),
        }

    bootstrap_draws: dict[str, dict[str, np.ndarray]] = {
        name: {
            "net_pnl": np.empty(replications, dtype=float),
            "sharpe": np.empty(replications, dtype=float),
        }
        for name in model_state
    }

    num_rows = next(iter(model_state.values()))["observed_num_rows"]
    for name, state in model_state.items():
        if state["observed_num_rows"] != num_rows:
            raise ValueError("All selected models must share the same standardized evaluation length.")

    for idx in range(replications):
        sample_idx = circular_block_indices(num_rows, block_length, rng)
        for name, state in model_state.items():
            sampled = state["series"][sample_idx]
            bootstrap_draws[name]["net_pnl"][idx] = float(sampled.sum())
            bootstrap_draws[name]["sharpe"][idx] = annualized_sharpe(sampled, state["annualization_factor"])

    rows: list[dict[str, Any]] = []
    for name, state in model_state.items():
        pnl_draws = bootstrap_draws[name]["net_pnl"]
        sharpe_draws = bootstrap_draws[name]["sharpe"]
        pnl_lo, pnl_hi = quantile_bounds(pnl_draws)
        sharpe_lo, sharpe_hi = quantile_bounds(sharpe_draws)
        rows.append(
            {
                "name": name,
                "family": state["family"],
                "threshold": state["threshold"],
                "num_rows": state["observed_num_rows"],
                "num_trades": state["observed_num_trades"],
                "observed_net_pnl": state["observed_net_pnl"],
                "observed_sharpe": state["observed_sharpe"],
                "bootstrap_net_pnl_mean": float(np.mean(pnl_draws)),
                "bootstrap_net_pnl_std": float(np.std(pnl_draws, ddof=1)),
                "bootstrap_net_pnl_ci_lower": pnl_lo,
                "bootstrap_net_pnl_ci_upper": pnl_hi,
                "bootstrap_prob_net_pnl_gt_zero": float(np.mean(pnl_draws > 0.0)),
                "bootstrap_sharpe_mean": float(np.mean(sharpe_draws)),
                "bootstrap_sharpe_std": float(np.std(sharpe_draws, ddof=1)),
                "bootstrap_sharpe_ci_lower": sharpe_lo,
                "bootstrap_sharpe_ci_upper": sharpe_hi,
                "bootstrap_prob_sharpe_gt_zero": float(np.mean(sharpe_draws > 0.0)),
            }
        )

    frame = pd.DataFrame(rows).sort_values(["observed_net_pnl", "observed_sharpe"], ascending=[False, False]).reset_index(drop=True)
    return frame, bootstrap_draws


def bootstrap_pairwise_deltas(
    winner_name: str,
    summary_frame: pd.DataFrame,
    bootstrap_draws: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    if winner_name not in bootstrap_draws:
        raise ValueError(f"Winner {winner_name} not present in bootstrap draws.")

    winner_row = summary_frame[summary_frame["name"] == winner_name].iloc[0]
    rows: list[dict[str, Any]] = []
    winner_pnl = bootstrap_draws[winner_name]["net_pnl"]
    winner_sharpe = bootstrap_draws[winner_name]["sharpe"]

    for _, row in summary_frame.iterrows():
        challenger = str(row["name"])
        if challenger == winner_name:
            continue
        pnl_delta = winner_pnl - bootstrap_draws[challenger]["net_pnl"]
        sharpe_delta = winner_sharpe - bootstrap_draws[challenger]["sharpe"]
        pnl_lo, pnl_hi = quantile_bounds(pnl_delta)
        sharpe_lo, sharpe_hi = quantile_bounds(sharpe_delta)
        rows.append(
            {
                "winner": winner_name,
                "challenger": challenger,
                "challenger_family": row["family"],
                "observed_net_pnl_delta": float(winner_row["observed_net_pnl"] - row["observed_net_pnl"]),
                "observed_sharpe_delta": float(winner_row["observed_sharpe"] - row["observed_sharpe"]),
                "bootstrap_net_pnl_delta_mean": float(np.mean(pnl_delta)),
                "bootstrap_net_pnl_delta_ci_lower": pnl_lo,
                "bootstrap_net_pnl_delta_ci_upper": pnl_hi,
                "bootstrap_prob_winner_net_pnl_gt_challenger": float(np.mean(pnl_delta > 0.0)),
                "bootstrap_sharpe_delta_mean": float(np.mean(sharpe_delta)),
                "bootstrap_sharpe_delta_ci_lower": sharpe_lo,
                "bootstrap_sharpe_delta_ci_upper": sharpe_hi,
                "bootstrap_prob_winner_sharpe_gt_challenger": float(np.mean(sharpe_delta > 0.0)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["bootstrap_prob_winner_net_pnl_gt_challenger", "observed_net_pnl_delta"],
        ascending=[False, False],
    ).reset_index(drop=True)


def frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, record in frame.iterrows():
        values = []
        for column in columns:
            value = record[column]
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_report(
    report_path: Path,
    winner_name: str,
    replications: int,
    block_length: int,
    summary_frame: pd.DataFrame,
    delta_frame: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    winner_row = summary_frame[summary_frame["name"] == winner_name].iloc[0]
    lines = [
        "# Block Bootstrap Significance for the Final Retuned Carry Benchmark",
        "",
        f"- Winner under tuned net PnL: `{winner_name}`",
        f"- Bootstrap replications: `{replications}`",
        f"- Circular block length: `{block_length}` bars",
        "- Resampling unit: bar-level net PnL from the saved threshold-swept backtests on the standardized carry-retuned window.",
        "",
        "## Winner Summary",
        "",
        (
            f"`{winner_name}` has observed `net_pnl={winner_row['observed_net_pnl']:.6f}` and "
            f"`Sharpe={winner_row['observed_sharpe']:.4f}`. "
            f"The bootstrap estimates `P(net_pnl>0)={winner_row['bootstrap_prob_net_pnl_gt_zero']:.3f}` and "
            f"`P(Sharpe>0)={winner_row['bootstrap_prob_sharpe_gt_zero']:.3f}`."
        ),
        "",
        "## Model-Level Bootstrap Summary",
        "",
        frame_to_markdown(summary_frame),
        "",
        "## Winner-Vs-Challenger Bootstrap Deltas",
        "",
        "Positive delta probabilities favor the winner.",
        "",
        frame_to_markdown(delta_frame),
        "",
        "## Interpretation",
        "",
        "- This is a block-bootstrap confidence assessment, not a formal structural proof of generalization.",
        "- If a confidence interval for net PnL or Sharpe straddles zero, the realized point estimate should be treated cautiously even when it is the sample winner.",
        "- Pairwise delta probabilities are especially useful for deciding whether the winner's economic lead over the strongest baselines is robust or only marginal.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run block-bootstrap significance checks on saved backtest outputs.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    output_root = resolve_path(cfg["paths"]["output_root"])
    report_path = resolve_path(cfg["paths"]["report_path"])
    replications = int(cfg["bootstrap"].get("replications", 1000))
    block_length = int(cfg["bootstrap"].get("block_length", 78))
    seed = int(cfg["bootstrap"].get("seed", 20260407))
    winner_name = str(cfg["selection"]["winner_name"])

    entries = load_model_entries(cfg)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_frame, bootstrap_draws = bootstrap_model_series(
        entries=entries,
        replications=replications,
        block_length=block_length,
        seed=seed,
    )
    delta_frame = bootstrap_pairwise_deltas(
        winner_name=winner_name,
        summary_frame=summary_frame,
        bootstrap_draws=bootstrap_draws,
    )

    summary_frame.to_csv(output_root / "bootstrap_model_summary.csv", index=False)
    delta_frame.to_csv(output_root / "bootstrap_winner_deltas.csv", index=False)
    write_report(
        report_path=report_path,
        winner_name=winner_name,
        replications=replications,
        block_length=block_length,
        summary_frame=summary_frame,
        delta_frame=delta_frame,
    )
    print(f"Wrote bootstrap report to {report_path}")


if __name__ == "__main__":
    main()
