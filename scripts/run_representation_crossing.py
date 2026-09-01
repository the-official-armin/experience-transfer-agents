"""Day 2, Step 1: representation crossing on a meaningful subset.

Same (E,T) pair (oracle-selected experience, 110-experience bank), same
task, three representations (raw / reflection / procedure) -- does
Delta(E,T)/Delta_eff(E,T) change by encoding? Scope decision, stated
plainly: ~16 base (task, experience) pairs x 3 representations = ~48 runs,
spanning all four shift_categories and both hard/easy. This is NOT the full
matrix (every experience x every representation x every task) -- that isn't
affordable this week. This is a deliberate subset for a first read on
whether representation matters at all, not an exhaustive claim.

Usage: python scripts/run_representation_crossing.py
"""

import json
import random
import sys
import time
from pathlib import Path

from openai import APIError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.agent import Agent
from src.agent.config import load_agent_config, load_model_config
from src.environment.tasks import load_tasks
from src.experience.storage import load_experiences
from src.experiments.conditions import select_oracle_experience
from src.experiments.logging import append_result, start_fresh
from src.experiments.runner import REPRESENTATIONS, run_task

HARD_PER_CATEGORY = 2
EASY_PER_CATEGORY = 2
SEED = 42
BASELINE_PATH = "data/results/baseline_full.jsonl"
BANK_PATH = "data/experiences/experiences.jsonl"
EXPERIENCE_TASK_SOURCE = "tasks/experience_tasks/tasks.jsonl"
TEST_TASK_SOURCE = "tasks/test_tasks/tasks.jsonl"
HARD25_PATH = "data/results/day1_hard25_task_ids.json"
OUTPUT_PATH = "data/results/representation_crossing.jsonl"
INTER_TASK_DELAY = 3


def load_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def pick_base_tasks():
    all_test_tasks = {t["task_id"]: t for t in load_tasks(TEST_TASK_SOURCE)}
    baseline_by_task = {r["task_id"]: r for r in load_jsonl(BASELINE_PATH)}
    hard25 = set(json.loads(Path(HARD25_PATH).read_text()))

    by_category = {}
    for task_id, result in baseline_by_task.items():
        bucket = by_category.setdefault(result["shift_category"], {"hard": [], "easy": []})
        bucket["hard" if not result["success"] else "easy"].append(task_id)

    rng = random.Random(SEED)
    selected = []
    for category in sorted(by_category):
        hard_pool = sorted(tid for tid in by_category[category]["hard"] if tid in hard25)
        selected.extend(hard_pool[:HARD_PER_CATEGORY])

        easy_pool = list(by_category[category]["easy"])
        rng.shuffle(easy_pool)
        selected.extend(easy_pool[:EASY_PER_CATEGORY])

    return [all_test_tasks[tid] for tid in selected], baseline_by_task


def main():
    model_config = load_model_config()
    agent_config = load_agent_config()
    agent = Agent(model_config, agent_config)

    base_tasks, baseline_by_task = pick_base_tasks()
    experiences = load_experiences(BANK_PATH)
    source_tasks_by_id = {t["task_id"]: t for t in load_tasks(EXPERIENCE_TASK_SOURCE)}

    print(f"{len(base_tasks)} base pairs x {len(REPRESENTATIONS)} representations = "
          f"{len(base_tasks) * len(REPRESENTATIONS)} runs\n")

    start_fresh(OUTPUT_PATH)
    results = []
    for task in base_tasks:
        experience = select_oracle_experience(task, experiences, source_tasks_by_id)
        baseline = baseline_by_task[task["task_id"]]
        hard_easy = "hard" if not baseline["success"] else "easy"

        for representation in REPRESENTATIONS:
            try:
                record = run_task(agent, task, condition="oracle", experience=experience,
                                   representation=representation)
            except APIError as e:
                print(f"{task['task_id']} [{representation}]: ERROR ({e.__class__.__name__}), skipping")
                time.sleep(INTER_TASK_DELAY)
                continue

            delta = int(record["success"]) - int(baseline["success"])
            delta_eff = None
            if baseline["success"] and record["success"]:
                delta_eff = {
                    "completion_tokens": baseline["completion_tokens"] - record["completion_tokens"],
                    "steps": baseline["steps"] - record["steps"],
                    "tool_call_count": baseline["tool_call_count"] - record["tool_call_count"],
                }
            record["hard_easy"] = hard_easy
            record["delta"] = delta
            record["delta_eff"] = delta_eff
            append_result(record, OUTPUT_PATH)
            results.append(record)
            print(f"{task['task_id']} [{task['shift_category']}/{hard_easy}] {representation}: "
                  f"{'PASS' if record['success'] else 'FAIL'} (delta={delta:+d})")
            time.sleep(INTER_TASK_DELAY)

    print(f"\n{len(results)} runs written to {OUTPUT_PATH}")

    by_task = {}
    for r in results:
        by_task.setdefault(r["task_id"], {})[r["representation"]] = r

    changed = [tid for tid, reps in by_task.items()
               if len({reps[r]["success"] for r in reps if r in reps}) > 1]
    print(f"\nPairs where success/fail CHANGED across representations: {len(changed)}/{len(by_task)} -- {changed}")


if __name__ == "__main__":
    main()
