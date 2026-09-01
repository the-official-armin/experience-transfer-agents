"""Day 3, Step 2: oversample distractor_tool -- run oracle+retrieved on the
remaining unused portion of that category's test pool (36 of 60 tasks never
tested under either condition; the other 24 already have Phase 1 results).
Exhausts the full 60-task category, so there's no ambiguity about whether
sampling more would have helped further.

Serves two purposes: (a) tightens the Fisher's exact CI on the distractor_tool
negative-transfer finding, which currently doesn't survive Bonferroni
correction; (b) directly fixes Delta(E,T)'s thin held-out negative-class
support (1 example in 51 rows).

Resumable: skips (task_id, condition) pairs already present in the output
file.

Usage: python scripts/oversample_distractor_tool.py
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
from src.experiments.conditions import select_oracle_experience, select_retrieved_experience
from src.experiments.logging import append_result
from src.experiments.runner import run_task

BASELINE_PATH = "data/results/baseline_full.jsonl"
CONDITIONS_FULL_PATH = "data/results/conditions_full.jsonl"
BANK_PATH = "data/experiences/experiences.jsonl"
EXPERIENCE_TASK_SOURCE = "tasks/experience_tasks/tasks.jsonl"
TEST_TASK_SOURCE = "tasks/test_tasks/tasks.jsonl"
OUTPUT_PATH = "data/results/distractor_tool_oversample.jsonl"
INTER_TASK_DELAY = 3


def load_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def already_done_pairs(path):
    if not Path(path).exists():
        return set()
    return {(r["task_id"], r["condition"]) for r in load_jsonl(path)}


def main():
    model_config = load_model_config()
    agent_config = load_agent_config()
    agent = Agent(model_config, agent_config)

    all_test_tasks = {t["task_id"]: t for t in load_tasks(TEST_TASK_SOURCE)}
    all_distractor_ids = {tid for tid, t in all_test_tasks.items() if t["shift_category"] == "distractor_tool"}

    already_covered = {r["task_id"] for r in load_jsonl(CONDITIONS_FULL_PATH)
                        if r["shift_category"] == "distractor_tool"}
    unused_ids = sorted(all_distractor_ids - already_covered)
    tasks = [all_test_tasks[tid] for tid in unused_ids]

    baseline_by_task = {r["task_id"]: r for r in load_jsonl(BASELINE_PATH)}
    experiences = load_experiences(BANK_PATH)
    source_tasks_by_id = {t["task_id"]: t for t in load_tasks(EXPERIENCE_TASK_SOURCE)}

    done_pairs = already_done_pairs(OUTPUT_PATH)
    print(f"{len(tasks)} unused distractor_tool tasks x 2 conditions = {len(tasks) * 2} target pairs. "
          f"{len(done_pairs)} already done.\n")

    errors = []
    for task in tasks:
        baseline = baseline_by_task[task["task_id"]]
        hard_easy = "hard" if not baseline["success"] else "easy"

        oracle_experience = select_oracle_experience(task, experiences, source_tasks_by_id)
        retrieved_experience = select_retrieved_experience(task, experiences)

        for condition, experience in (("oracle", oracle_experience), ("retrieved", retrieved_experience)):
            if (task["task_id"], condition) in done_pairs:
                continue
            try:
                record = run_task(agent, task, condition=condition, experience=experience)
            except APIError as e:
                print(f"{task['task_id']} [{condition}]: ERROR ({e.__class__.__name__}), skipping")
                errors.append((task["task_id"], condition))
                time.sleep(INTER_TASK_DELAY)
                continue

            record["hard_easy"] = hard_easy
            record["baseline_success"] = baseline["success"]
            delta = int(record["success"]) - int(baseline["success"])
            record["delta"] = delta
            delta_eff = None
            if baseline["success"] and record["success"]:
                delta_eff = {
                    "completion_tokens": baseline["completion_tokens"] - record["completion_tokens"],
                    "steps": baseline["steps"] - record["steps"],
                    "tool_call_count": baseline["tool_call_count"] - record["tool_call_count"],
                }
            record["delta_eff"] = delta_eff
            append_result(record, OUTPUT_PATH)
            print(f"{task['task_id']} [{hard_easy}] {condition} exp={experience['experience_id']}: "
                  f"{'PASS' if record['success'] else 'FAIL'} (delta={delta:+d})")
            time.sleep(INTER_TASK_DELAY)

    final = load_jsonl(OUTPUT_PATH)
    print(f"\n{len(final)} total pairs in {OUTPUT_PATH}")
    if errors:
        print(f"{len(errors)} pair(s) errored out: {errors}")


if __name__ == "__main__":
    main()
