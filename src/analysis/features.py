"""The predictor's feature set (CLAUDE.md: How the Sub-Questions Connect,
Property Variance -- Closed Investigation).

PRIMARY set: shift_category, representation, and the 5 reliable properties
(length, num_steps, success_failure, task_structure, tool_dependence).

CLUSTERED set adds the 4 properties already found to cluster (abstraction,
specificity, composability, information_context) -- included ONLY in the
secondary ablation model, to report their near-zero contribution
explicitly, per CLAUDE.md: "do not include them silently."

Delta_eff target scoping: Delta_eff has 3 dimensions (completion_tokens,
steps, tool_call_count) per src/analysis/transfer.py, but `steps` is
constant (always 1, single-shot agent loop) and contributes no variance.
The predictor's Delta_eff regression target is delta_eff.completion_tokens
specifically -- the dimension CLAUDE.md's own Delta_eff discussion centers
on (the mean/median disagreement note). Stated here plainly as a scope
decision, not something to re-derive per use site.
"""

import json
from pathlib import Path

import pandas as pd

PRIMARY_PROPERTY_FEATURES = ("length", "num_steps", "success_failure", "task_structure", "tool_dependence")
CLUSTERED_PROPERTY_FEATURES = ("abstraction", "specificity", "composability", "information_context")
CATEGORICAL_FEATURES = ("shift_category", "representation", "task_structure")


def _load_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _normalize_pair(record):
    """Both source files use slightly different field names/shapes --
    normalize to one common row schema before joining with experience
    properties."""
    experience_id = record.get("experience_id") or record.get("experience_id_or_null")
    delta_eff = record.get("delta_eff")
    delta_eff_completion_tokens = delta_eff["completion_tokens"] if delta_eff else None

    return {
        "task_id": record["task_id"],
        "condition": record["condition"],
        "experience_id": experience_id,
        "shift_category": record["shift_category"],
        "hard_easy": record["hard_easy"],
        "representation": record.get("representation") or "raw",
        "delta": record["delta"],
        "delta_eff_completion_tokens": delta_eff_completion_tokens,
    }


def build_dataset(pair_paths, bank_path):
    """Join scored pairs (from one or more results files) with their
    matched experience's properties. Rows with experience_id=None
    (baseline-condition records, if any slip in) are dropped -- the
    predictor's feature set requires an experience to describe.

    Adds a `pair_id` column, disambiguated when the same (task_id,
    condition, representation) triple appears more than once -- e.g. the
    Step 1 representation-crossing "raw" runs deliberately re-tested some
    tasks already present in Phase 1's oracle/raw results, as independent
    live observations, not duplicates to be merged."""
    bank = {e["experience_id"]: e for e in _load_jsonl(bank_path)}

    rows = []
    seen_key_counts = {}
    for path in pair_paths:
        for record in _load_jsonl(path):
            pair = _normalize_pair(record)
            if pair["experience_id"] is None or pair["experience_id"] not in bank:
                continue

            key = f"{pair['task_id']}|{pair['condition']}|{pair['representation']}"
            occurrence = seen_key_counts.get(key, 0)
            seen_key_counts[key] = occurrence + 1

            props = bank[pair["experience_id"]]["properties"]
            row = dict(pair)
            row["pair_id"] = f"{key}|{occurrence}"
            for key_name in PRIMARY_PROPERTY_FEATURES:
                row[key_name] = props[key_name]
            for key_name in CLUSTERED_PROPERTY_FEATURES:
                row[key_name] = props[key_name]
            rows.append(row)

    return pd.DataFrame(rows)
