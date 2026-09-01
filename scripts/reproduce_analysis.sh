#!/usr/bin/env bash
# One-command reproduction of all ANALYSIS outputs (scored pairs, predictor
# report, final figures) from the raw data already collected this week
# (data/results/*.jsonl, data/experiences/*.jsonl). Deterministic given the
# same input files, configs, and seeds -- does NOT re-run the agent against
# the live API.
#
# Data collection (agent runs against GLM-4.7-Flash) is intentionally NOT
# part of this one-command runner: it required live judgment calls and
# explicit API-budget gating at each step throughout the week (pilot before
# scale, checkpoints after each stage -- see docs/experiment_log.md and
# docs/findings_summary.md). Reproducing it means re-running the individual
# scripts/*.py entry points in the order documented in findings_summary.md,
# with the same configs/*.yaml and .env (GLM_API_KEY) -- not a single script,
# by design.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "1/5: scoring Phase 1 (E,T) pairs -> results/processed/transfer_pairs.jsonl"
.venv/bin/python scripts/run_analysis.py

echo "2/5: freezing predictor split v1 (skips if already frozen)"
.venv/bin/python scripts/freeze_predictor_split.py || true

echo "3/5: extending predictor split with distractor_tool oversample -> v2 (skips if already frozen)"
.venv/bin/python scripts/extend_predictor_split.py || true

echo "4/5: training predictor v1/v2, both targets, vs both baselines"
.venv/bin/python scripts/train_predictor.py

echo "5/5: building final figures -> results/figures/"
.venv/bin/python scripts/build_final_visualizations.py

echo ""
echo "Done. Outputs: results/figures/, results/tables/, results/processed/"
