"""Step 6: first results from the scored (E,T) pairs.

Generates:
  - results/tables/transfer_map.csv    -- every pair, not aggregated by task family
  - results/figures/transfer_by_category.png -- first transfer plot
  - oracle-vs-retrieved gap analysis (printed): how much of any transfer
    failure is retrieval quality (oracle succeeded, retrieved didn't) vs.
    genuine non-transfer (neither succeeded, even with privileged selection)

Usage: python scripts/build_transfer_map.py [transfer_pairs_path]
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_PAIRS_PATH = "results/processed/transfer_pairs.jsonl"
MAP_PATH = "results/tables/transfer_map.csv"
PLOT_PATH = "results/figures/transfer_by_category.png"


def load_pairs(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_transfer_map(pairs, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = ["task_id", "experience_id", "condition", "shift_category", "hard_easy",
              "baseline_success", "condition_success", "delta", "ambiguous", "noise_caution",
              "delta_eff_completion_tokens", "delta_eff_tool_call_count", "retry_count_delta"]
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in pairs:
            eff = p["delta_eff"] or {}
            writer.writerow({
                "task_id": p["task_id"], "experience_id": p["experience_id"],
                "condition": p["condition"], "shift_category": p["shift_category"],
                "hard_easy": p["hard_easy"], "baseline_success": p["baseline_success"],
                "condition_success": p["condition_success"], "delta": p["delta"],
                "ambiguous": p["ambiguous"], "noise_caution": p["noise_caution"],
                "delta_eff_completion_tokens": eff.get("completion_tokens", ""),
                "delta_eff_tool_call_count": eff.get("tool_call_count", ""),
                "retry_count_delta": p["retry_count_delta"],
            })


def plot_transfer_by_category(pairs, path):
    categories = sorted({p["shift_category"] for p in pairs})
    conditions = ("oracle", "retrieved")

    fig, axes = plt.subplots(1, len(conditions), figsize=(11, 4.5), sharey=True)
    for ax, condition in zip(axes, conditions):
        cond_pairs = [p for p in pairs if p["condition"] == condition]
        positive = [sum(1 for p in cond_pairs if p["shift_category"] == c and p["delta"] == 1) for c in categories]
        negative = [-sum(1 for p in cond_pairs if p["shift_category"] == c and p["delta"] == -1) for c in categories]

        x = range(len(categories))
        ax.bar(x, positive, color="#4C72B0", label="positive (+1)")
        ax.bar(x, negative, color="#C44E52", label="negative (-1)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(categories, rotation=30, ha="right")
        ax.set_title(f"{condition}")
        ax.set_ylabel("Delta(E,T) flip count")

    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Transfer gain by shift_category (n=%d pairs)" % len(pairs))
    fig.tight_layout()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def oracle_vs_retrieved_gap(pairs):
    """Restricted to `hard` (baseline-fail) tasks: Delta == +1 is structurally
    impossible for `easy` (baseline-success) tasks, so including them would
    inflate 'neither positive' with cases that could never show positive
    transfer regardless of experience quality -- not evidence of anything."""
    by_task = {}
    for p in pairs:
        if p["hard_easy"] != "hard":
            continue
        by_task.setdefault(p["task_id"], {})[p["condition"]] = p

    both_positive, retrieval_gap, retrieved_only, neither_positive = [], [], [], []
    for task_id, conds in by_task.items():
        if "oracle" not in conds or "retrieved" not in conds:
            continue
        oracle_pos = conds["oracle"]["delta"] == 1
        retrieved_pos = conds["retrieved"]["delta"] == 1
        if oracle_pos and retrieved_pos:
            both_positive.append(task_id)
        elif oracle_pos and not retrieved_pos:
            retrieval_gap.append(task_id)
        elif retrieved_pos and not oracle_pos:
            retrieved_only.append(task_id)
        else:
            neither_positive.append(task_id)

    return both_positive, retrieval_gap, retrieved_only, neither_positive


def main():
    pairs_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PAIRS_PATH
    pairs = load_pairs(pairs_path)

    write_transfer_map(pairs, MAP_PATH)
    print(f"Transfer map: {len(pairs)} rows -> {MAP_PATH}")

    plot_transfer_by_category(pairs, PLOT_PATH)
    print(f"Plot: {PLOT_PATH}")

    both, gap, retrieved_only, neither = oracle_vs_retrieved_gap(pairs)
    n_paired_tasks = len(both) + len(gap) + len(retrieved_only) + len(neither)
    print(f"\nOracle-vs-retrieved gap ({n_paired_tasks} HARD/baseline-fail tasks with both conditions run "
          f"-- Delta=+1 is structurally impossible on easy tasks, so they're excluded here):")
    print(f"  both positive (transfer achieved regardless of retrieval): {len(both)}")
    print(f"  oracle positive, retrieved not (RETRIEVAL QUALITY GAP):    {len(gap)}  -- {gap}")
    print(f"  retrieved positive, oracle not (retrieval got lucky):      {len(retrieved_only)}  -- {retrieved_only}")
    print(f"  neither positive (no transfer even with privileged pick):  {len(neither)}")
    if n_paired_tasks:
        print(f"\n  Of all {n_paired_tasks} paired tasks, {len(gap)} "
              f"({100*len(gap)/n_paired_tasks:.0f}%) show transfer that oracle could reach but "
              f"retrieved couldn't -- an upper bound on how much of retrieved's shortfall vs. oracle "
              f"is retrieval-quality-limited rather than the experience bank lacking anything useful.")


if __name__ == "__main__":
    main()
