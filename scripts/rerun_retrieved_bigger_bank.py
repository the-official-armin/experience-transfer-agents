"""Day 2, Step 3: re-run retrieved (embedding-based) selection on the
identical 25 hard tasks, now against the 110-experience bank (up from 30) --
same check as Step 2 but for the retrieved condition, secondary but cheap
once the bank exists. Retrieval method itself is unchanged (fixed before
evaluation, per CLAUDE.md) -- only the candidate pool size changes.

Usage: python scripts/rerun_retrieved_bigger_bank.py
"""

import json
import sys
import time
from pathlib import Path

from openai import APIError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.agent import Agent
from src.agent.config import load_agent_config, load_model_config
from src.environment.tasks import load_tasks
from src.experience.storage import load_experiences
from src.experiments.conditions import select_retrieved_experience
from src.experiments.logging import append_result, start_fresh
from src.experiments.runner import run_task

TASK_IDS_PATH = "data/results/day1_hard25_task_ids.json"
BASELINE_PATH = "data/results/baseline_full.jsonl"
BANK_PATH = "data/experiences/experiences.jsonl"
TEST_TASK_SOURCE = "tasks/test_tasks/tasks.jsonl"
OUTPUT_PATH = "data/results/retrieved_bigbank_25.jsonl"
INTER_TASK_DELAY = 3


def load_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    model_config = load_model_config()
    agent_config = load_agent_config()
    agent = Agent(model_config, agent_config)

    task_ids = json.loads(Path(TASK_IDS_PATH).read_text())
    all_test_tasks = {t["task_id"]: t for t in load_tasks(TEST_TASK_SOURCE)}
    tasks = [all_test_tasks[tid] for tid in task_ids]

    baseline_by_task = {r["task_id"]: r for r in load_jsonl(BASELINE_PATH)}
    experiences = load_experiences(BANK_PATH)

    print(f"Bank size: {len(experiences)} (was 30 on Day 1)")
    print(f"Re-running retrieved on {len(tasks)} hard tasks\n")

    start_fresh(OUTPUT_PATH)
    results = []
    for task in tasks:
        experience = select_retrieved_experience(task, experiences)
        try:
            record = run_task(agent, task, condition="retrieved", experience=experience)
        except APIError as e:
            print(f"{task['task_id']}: ERROR ({e.__class__.__name__}), skipping")
            time.sleep(INTER_TASK_DELAY)
            continue

        baseline = baseline_by_task[task["task_id"]]
        delta = int(record["success"]) - int(baseline["success"])
        record["bank_size"] = len(experiences)
        record["delta"] = delta
        append_result(record, OUTPUT_PATH)
        results.append(record)
        print(f"{task['task_id']} [{task['shift_category']}] exp={experience['experience_id']}: "
              f"{'PASS' if record['success'] else 'FAIL'} (delta={delta:+d})")
        time.sleep(INTER_TASK_DELAY)

    n = len(results)
    positive = sum(1 for r in results if r["delta"] == 1)
    no_transfer = n - positive

    print(f"\n{n}/{len(tasks)} tasks completed with bank_size=110")
    print(f"positive: {positive}/{n}, no-transfer: {no_transfer}/{n} = {100*no_transfer/n:.0f}%")
    print(f"\nOriginal retrieved (bank_size=30): 8/25 positive, 17/25 (68%) no-transfer")
    print(f"New      retrieved (bank_size=110): {positive}/{n} positive, {no_transfer}/{n} "
          f"({100*no_transfer/n:.0f}%) no-transfer")


if __name__ == "__main__":
    main()
