"""Day 2, Step 4: freeze a predictor-specific held-out split.

Separate from the task-level experience-generation/held-out split frozen
earlier in the week (tasks/split.json) -- this splits the scored (E,T)
pairs THEMSELVES into predictor-train and predictor-held-out sets,
stratified by shift_category so all four are represented in both. 80/20
split. Refuses to overwrite an existing frozen split -- freeze once, don't
touch it again after training starts.

Usage: python scripts/freeze_predictor_split.py
"""

import json
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.features import build_dataset

PAIR_PATHS = ["results/processed/transfer_pairs.jsonl", "data/results/representation_crossing.jsonl"]
BANK_PATH = "data/experiences/experiences.jsonl"
OUTPUT_PATH = Path("results/processed/predictor_split.json")
HELD_OUT_FRACTION = 0.2
SEED = 42


def main():
    if OUTPUT_PATH.exists():
        raise SystemExit(f"{OUTPUT_PATH} already exists and is frozen -- refusing to overwrite.")

    df = build_dataset(PAIR_PATHS, BANK_PATH)

    train_ids, held_out_ids = train_test_split(
        df["pair_id"], test_size=HELD_OUT_FRACTION, stratify=df["shift_category"], random_state=SEED,
    )

    manifest = {
        "seed": SEED,
        "held_out_fraction": HELD_OUT_FRACTION,
        "source_pair_paths": PAIR_PATHS,
        "train": sorted(train_ids.tolist()),
        "held_out": sorted(held_out_ids.tolist()),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(manifest, indent=2))

    print(f"Frozen: {len(manifest['train'])} train / {len(manifest['held_out'])} held-out "
          f"-> {OUTPUT_PATH}\n")

    print("by shift_category:")
    for category in sorted(df["shift_category"].unique()):
        cat_df = df[df["shift_category"] == category]
        train_count = cat_df["pair_id"].isin(manifest["train"]).sum()
        held_out_count = cat_df["pair_id"].isin(manifest["held_out"]).sum()
        print(f"  {category}: {train_count} train / {held_out_count} held-out (of {len(cat_df)} total)")


if __name__ == "__main__":
    main()
