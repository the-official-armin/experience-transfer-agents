"""Compute Delta(E,T) and Delta_eff(E,T) for every scored pair in a
conditions results file, against the Step 2 baseline. Writes to results/
(derived output, per CLAUDE.md's data/results/ vs results/ distinction --
this is analysis output, not raw harness output).

Usage: python scripts/run_analysis.py [conditions_results_path]
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.transfer import EFFORT_FIELDS, build_pairs

DEFAULT_CONDITIONS_PATH = "data/results/conditions_full.jsonl"  # Phase 1's real 203-pair dataset,
# NOT the 24-pair Wednesday pilot -- this default drifted stale once
# conditions_full.jsonl became the actual Phase 1 corpus and silently
# regenerated the wrong transfer_pairs.jsonl when called with no argument.
BASELINE_PATH = "data/results/baseline_full.jsonl"
OUTPUT_PATH = "results/processed/transfer_pairs.jsonl"


def load_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    conditions_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONDITIONS_PATH

    baseline_by_task = {r["task_id"]: r for r in load_jsonl(BASELINE_PATH)}
    condition_records = load_jsonl(conditions_path)

    pairs = build_pairs(baseline_by_task, condition_records)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with Path(OUTPUT_PATH).open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"{len(pairs)} pairs scored, written to {OUTPUT_PATH}\n")

    for condition in ("oracle", "retrieved"):
        cond_pairs = [p for p in pairs if p["condition"] == condition]
        positive = sum(1 for p in cond_pairs if p["delta"] == 1)
        negative = sum(1 for p in cond_pairs if p["delta"] == -1)
        zero = sum(1 for p in cond_pairs if p["delta"] == 0)
        ambiguous = sum(1 for p in cond_pairs if p["ambiguous"])
        eff_pairs = [p for p in cond_pairs if p["delta_eff"] is not None]

        print(f"{condition}: {len(cond_pairs)} pairs -- "
              f"+{positive} positive / {negative} negative / {zero} zero (ambiguous={ambiguous})")
        if negative == 0:
            print(f"  (no negative transfer observed in this sample -- not yet evidence it doesn't occur)")
        print(f"  Delta_eff computable on {len(eff_pairs)}/{len(cond_pairs)} pairs (matched success); "
              f"agent-controlled effort only -- completion_tokens/steps/tool_call_count, retry_count excluded")
        if eff_pairs:
            for field in EFFORT_FIELDS:
                values = [p["delta_eff"][field] for p in eff_pairs]
                mean = sum(values) / len(values)
                median = statistics.median(values)
                print(f"    Delta_eff[{field}]: mean={mean:+.2f}, median={median:+.2f} "
                      f"(positive = experience was more efficient; median resists single-outlier skew)")

        retry_values = [p["retry_count_delta"] for p in cond_pairs]
        print(f"  retry_count_delta: mean={sum(retry_values)/len(retry_values):+.2f} "
              f"(endpoint-health monitoring only, not an effort metric -- shared infra noise, "
              f"not agent-controlled)")
        print()

    print("by shift_category (delta, both conditions pooled):")
    for category in sorted({p["shift_category"] for p in pairs}):
        cat_pairs = [p for p in pairs if p["shift_category"] == category]
        positive = sum(1 for p in cat_pairs if p["delta"] == 1)
        negative = sum(1 for p in cat_pairs if p["delta"] == -1)
        caution = " [noise_caution category]" if cat_pairs and cat_pairs[0]["noise_caution"] else ""
        print(f"  {category}: +{positive}/-{negative} out of {len(cat_pairs)}{caution}")


if __name__ == "__main__":
    main()
