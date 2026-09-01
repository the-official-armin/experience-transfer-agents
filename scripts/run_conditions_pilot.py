"""Step 3 pilot: oracle and retrieved conditions on a small sample before
scaling (~20-30 pairs), the same way Tuesday piloted before scaling. Both
conditions use the `raw` representation only (reflection/procedure crossing
is Thursday's work). Confirms the new oracle/retrieved harness produces
sane, complete Result records before spending budget on the full run.

Pilot sample: 2 baseline-fail + 1 baseline-success task per shift_category
(12 tasks x 2 conditions = 24 pairs) -- deliberately mixed hard/easy so the
pilot validates both halves (Delta(E,T) needs baseline-fail tasks;
Delta_eff needs matched-success tasks), not just a random sample that might
land almost entirely on the easy side given Step 2's ceiling finding.

Usage: python scripts/run_conditions_pilot.py
"""

import json
import random
import sys
import time
from pathlib import Path

from openai import APIError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.agent import Agent
from src.agent.config import load_agent_config, load_experiment_config, load_model_config
from src.environment.tasks import load_tasks
from src.experience.storage import load_experiences, start_fresh
from src.experiments.conditions import select_oracle_experience, select_retrieved_experience
from src.experiments.logging import append_result
from src.experiments.runner import run_task

PER_CATEGORY_FAIL = 2
PER_CATEGORY_SUCCESS = 1

REQUIRED_FIELDS = {
    "task_id", "shift_category", "condition", "experience_id_or_null", "success",
    "prompt_tokens", "completion_tokens", "tool_call_count", "retry_count", "steps",
    "trajectory", "hard_easy", "baseline_success",
}


def load_baseline_by_task(path):
    with Path(path).open() as f:
        return {(r := json.loads(line))["task_id"]: r for line in f if line.strip()}


def pick_pilot_tasks(config, baseline_by_task):
    all_tasks = {t["task_id"]: t for t in load_tasks(config["task_source"])}

    by_category = {}
    for task_id, result in baseline_by_task.items():
        bucket = by_category.setdefault(result["shift_category"], {"fail": [], "success": []})
        bucket["fail" if not result["success"] else "success"].append(task_id)

    rng = random.Random(config["seed"])
    picked_ids = []
    for category in sorted(by_category):
        fails, successes = list(by_category[category]["fail"]), list(by_category[category]["success"])
        rng.shuffle(fails)
        rng.shuffle(successes)
        picked_ids.extend(fails[:PER_CATEGORY_FAIL])
        picked_ids.extend(successes[:PER_CATEGORY_SUCCESS])

    return [all_tasks[tid] for tid in picked_ids]


def main():
    config = load_experiment_config("conditions_pilot")
    model_config = load_model_config()
    agent_config = load_agent_config()
    agent = Agent(model_config, agent_config)

    baseline_by_task = load_baseline_by_task(config["baseline_results_path"])
    experiences = load_experiences(config["experience_bank_path"])
    source_tasks_by_id = {t["task_id"]: t for t in load_tasks(config["experience_task_source"])}
    pilot_tasks = pick_pilot_tasks(config, baseline_by_task)

    print(f"Pilot: {len(pilot_tasks)} tasks x 2 conditions (oracle, retrieved) = "
          f"{len(pilot_tasks) * 2} pairs\n")

    output_path = config["output_path"]
    start_fresh(output_path)
    results = []

    for task in pilot_tasks:
        baseline = baseline_by_task[task["task_id"]]
        hard_easy = "hard" if not baseline["success"] else "easy"

        oracle_experience = select_oracle_experience(task, experiences, source_tasks_by_id)
        retrieved_experience = select_retrieved_experience(task, experiences)

        for condition, experience in (("oracle", oracle_experience), ("retrieved", retrieved_experience)):
            try:
                record = run_task(agent, task, condition=condition, experience=experience)
            except APIError as e:
                print(f"{task['task_id']} [{condition}]: ERROR ({e.__class__.__name__}), skipping")
                time.sleep(config["inter_task_delay_seconds"])
                continue

            record["hard_easy"] = hard_easy
            record["baseline_success"] = baseline["success"]
            append_result(record, output_path)
            results.append(record)
            print(f"{task['task_id']} [{task['shift_category']}/{hard_easy}] {condition} "
                  f"(exp={experience['experience_id']}): {'PASS' if record['success'] else 'FAIL'}")
            time.sleep(config["inter_task_delay_seconds"])

    print(f"\n{len(results)}/{len(pilot_tasks) * 2} Result records written to {output_path}")
    for condition in ("oracle", "retrieved"):
        cond_results = [r for r in results if r["condition"] == condition]
        n_success = sum(r["success"] for r in cond_results)
        print(f"  {condition}: {n_success}/{len(cond_results)} succeeded")

    incomplete = [r["task_id"] for r in results if not REQUIRED_FIELDS.issubset(r.keys())]
    print(f"\nSchema check: {'OK, all fields present' if not incomplete else f'MISSING FIELDS on {incomplete}'}")


if __name__ == "__main__":
    main()
