#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-configs/final_5min_additional_evaluations.yaml}"
REPORT_PATH="${REPORT_PATH:-artifacts/reports/final_5min_additional_evaluations.md}"
LOG_PATH="${LOG_PATH:-artifacts/reports/final_5min_additional_evaluations.run.log}"

mkdir -p "$(dirname "$REPORT_PATH")"
mkdir -p "$(dirname "$LOG_PATH")"

echo "Running final 5-minute additional evaluations..."
echo "Config: $CONFIG_PATH"
echo "Log: $LOG_PATH"
echo "Report: $REPORT_PATH"

"$PYTHON_BIN" scripts/run_final_5min_additional_evaluations.py --config "$CONFIG_PATH" "$@" 2>&1 | tee "$LOG_PATH"

echo
echo "Completed."
echo "Markdown report: $REPORT_PATH"
echo "Run log: $LOG_PATH"
