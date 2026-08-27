"""Freeze the experience-generation / held-out test split (Step 2).

Reads the full adapted BFCL pool (tasks/task_definitions/bfcl_tasks.jsonl),
partitions it per shift_category into experience_generation / test / unused,
and writes the frozen manifest to tasks/split.json. Also materializes full
task records into tasks/experience_tasks/ and tasks/test_tasks/ for
convenience.

Sizing target: 150-200+ scored (E,T) pairs for the predictor's held-out
evaluation by Thursday, even though only ~20-50 experiences get generated
today (Step 6). The test pool (280 tasks, 60-80 per category) is sized well
above the pair floor on its own, so even a modest experiences-per-task
pairing ratio later clears 150-200+ pairs. Experience-generation and test
pools never overlap. `ood` is not populated (per instruction) and is
excluded entirely -- only familiar/novel_tool/novel_composition/distractor_tool
are split.

Usage: python scripts/split_tasks.py
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environment.tasks import load_tasks, save_tasks

SEED = 42
SOURCE_PATH = Path("tasks/task_definitions/bfcl_tasks.jsonl")
SPLIT_PATH = Path("tasks/split.json")
EXPERIENCE_TASKS_PATH = Path("tasks/experience_tasks/tasks.jsonl")
TEST_TASKS_PATH = Path("tasks/test_tasks/tasks.jsonl")

# Per-category counts for experience_generation and test (test == exp-gen
# here, keeping the two pools symmetric across all four available shift
# categories; remainder per category stays "unused" reserve for later scale-up).
ALLOCATION = {
    "familiar":           {"experience_generation": 80, "test": 80},
    "novel_tool":         {"experience_generation": 60, "test": 60},
    "novel_composition":  {"experience_generation": 80, "test": 80},
    "distractor_tool":    {"experience_generation": 60, "test": 60},
}


def main():
    if SPLIT_PATH.exists():
        raise SystemExit(f"{SPLIT_PATH} already exists and is frozen -- refusing to overwrite.")

    all_tasks = load_tasks(SOURCE_PATH)

    by_category = defaultdict(list)
    for t in all_tasks:
        by_category[t["shift_category"]].append(t)

    unknown_categories = set(by_category) - set(ALLOCATION)
    if unknown_categories:
        raise SystemExit(f"Unhandled shift_category in pool: {unknown_categories}")

    rng = random.Random(SEED)
    split_manifest = {"experience_generation": {}, "test": {}, "unused": {}}
    experience_tasks, test_tasks = [], []

    for category, alloc in ALLOCATION.items():
        pool = list(by_category[category])
        rng.shuffle(pool)

        n_exp, n_test = alloc["experience_generation"], alloc["test"]
        if n_exp + n_test > len(pool):
            raise SystemExit(
                f"{category}: requested {n_exp + n_test} but only {len(pool)} available"
            )

        exp_slice = pool[:n_exp]
        test_slice = pool[n_exp:n_exp + n_test]
        unused_slice = pool[n_exp + n_test:]

        split_manifest["experience_generation"][category] = [t["task_id"] for t in exp_slice]
        split_manifest["test"][category] = [t["task_id"] for t in test_slice]
        split_manifest["unused"][category] = [t["task_id"] for t in unused_slice]

        for t in exp_slice:
            t["split"] = "experience_generation"
            experience_tasks.append(t)
        for t in test_slice:
            t["split"] = "test"
            test_tasks.append(t)

    counts = {
        split_name: {cat: len(ids) for cat, ids in cats.items()}
        for split_name, cats in split_manifest.items()
    }
    counts["experience_generation"]["total"] = sum(counts["experience_generation"].values())
    counts["test"]["total"] = sum(counts["test"].values())
    counts["unused"]["total"] = sum(counts["unused"].values())

    manifest = {
        "seed": SEED,
        "source": str(SOURCE_PATH),
        "target_pairs": "150-200+ scored (E,T) pairs by Thursday",
        "notes": (
            "ood excluded (not populated by the adapter, not invented here). "
            "experience_generation and test pools are disjoint and category-symmetric; "
            "remainder per category is 'unused' reserve, not touched today."
        ),
        "counts": counts,
        "splits": split_manifest,
    }

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(manifest, indent=2))

    save_tasks(experience_tasks, EXPERIENCE_TASKS_PATH)
    save_tasks(test_tasks, TEST_TASKS_PATH)

    print(f"Wrote frozen split to {SPLIT_PATH}")
    print(f"experience_generation: {counts['experience_generation']}")
    print(f"test:                  {counts['test']}")
    print(f"unused:                {counts['unused']}")


if __name__ == "__main__":
    main()
