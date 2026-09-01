"""Day 3, Step 2: extend the frozen predictor split with the new
distractor_tool oversample pairs, WITHOUT touching the original v1
assignments -- every pair_id that was train/held-out in v1 stays exactly
that; only the 72 new rows get a fresh 80/20 split (stratified by
shift_category, though the new batch is entirely distractor_tool so this
degenerates to a plain random split for this round). This is disciplined
enrichment, not cherry-picking new negative examples into held-out to
inflate the finding -- the new rows are split honestly, same methodology
as v1.

Usage: python scripts/extend_predictor_split.py
"""

import json
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.features import build_dataset

PAIR_PATHS = [
    "results/processed/transfer_pairs.jsonl",
    "data/results/representation_crossing.jsonl",
    "data/results/distractor_tool_oversample.jsonl",
]
BANK_PATH = "data/experiences/experiences.jsonl"
V1_PATH = Path("results/processed/predictor_split.json")
OUTPUT_PATH = Path("results/processed/predictor_split_v2.json")
HELD_OUT_FRACTION = 0.2
SEED = 42


def main():
    if OUTPUT_PATH.exists():
        raise SystemExit(f"{OUTPUT_PATH} already exists and is frozen -- refusing to overwrite.")

    v1 = json.loads(V1_PATH.read_text())
    v1_ids = set(v1["train"]) | set(v1["held_out"])

    df = build_dataset(PAIR_PATHS, BANK_PATH)
    new_rows = df[~df["pair_id"].isin(v1_ids)]

    print(f"v1 had {len(v1_ids)} pairs. Full enriched dataset has {len(df)}. "
          f"{len(new_rows)} new rows to split.\n")

    new_train_ids, new_held_out_ids = train_test_split(
        new_rows["pair_id"], test_size=HELD_OUT_FRACTION,
        stratify=new_rows["shift_category"], random_state=SEED,
    )

    manifest = {
        "seed": SEED,
        "held_out_fraction": HELD_OUT_FRACTION,
        "source_pair_paths": PAIR_PATHS,
        "extends": str(V1_PATH),
        "note": "v1 assignments preserved exactly; only new (distractor_tool oversample) rows freshly split.",
        "train": sorted(v1["train"] + new_train_ids.tolist()),
        "held_out": sorted(v1["held_out"] + new_held_out_ids.tolist()),
    }

    OUTPUT_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Frozen: {len(manifest['train'])} train / {len(manifest['held_out'])} held-out -> {OUTPUT_PATH}\n")

    print("by shift_category:")
    for category in sorted(df["shift_category"].unique()):
        cat_df = df[df["shift_category"] == category]
        train_count = cat_df["pair_id"].isin(manifest["train"]).sum()
        held_out_count = cat_df["pair_id"].isin(manifest["held_out"]).sum()
        print(f"  {category}: {train_count} train / {held_out_count} held-out (of {len(cat_df)} total)")


if __name__ == "__main__":
    main()
