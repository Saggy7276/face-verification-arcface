#!/usr/bin/env bash
# Runs the full pipeline end to end (Linux / macOS / Git Bash).
set -euo pipefail
cd "$(dirname "$0")"
python dataset_preparation.py --overwrite
python train.py
python generate_pairs.py
python evaluate_pairs.py
python roc_analysis.py
python gallery_probe.py
python make_report.py
