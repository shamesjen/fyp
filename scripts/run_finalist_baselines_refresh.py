from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_yaml_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the expanded walk-forward baseline suite on the five frozen 5-minute finalist datasets."
    )
    parser.add_argument("--config", default="configs/finalist_baseline_refresh_5min.yaml")
    parser.add_argument("--force", action="store_true", help="Rerun even if the target summary already exists.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    base_config = resolve_path(cfg["base_config"])
    output_root = resolve_path(cfg["paths"]["output_root"])

    for run in cfg["runs"]:
        tag = str(run["tag"])
        summary_path = output_root / tag / "baseline_walkforward_summary.csv"
        if summary_path.exists() and not args.force:
            print(f"[SKIP] Baselines already exist for {tag}: {summary_path}")
            continue

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_baselines_walkforward.py"),
            "--config",
            str(base_config),
            "--dataset-path",
            str(resolve_path(run["dataset_path"])),
            "--holding-period",
            str(int(run["holding_period_bars"])),
            "--signal-threshold",
            str(float(run["signal_threshold"])),
            "--tag",
            tag,
        ]
        print(f"[RUN] {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()
