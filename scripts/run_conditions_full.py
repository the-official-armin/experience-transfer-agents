"""Step 5: scale oracle/retrieved to target volume (150-200+ scored pairs,
CLAUDE.md's data scale target). Seeds from Step 3's already-validated
24-pair pilot rather than recomputing it, then adds enough additional
held-out tasks to reach ~100 unique tasks x 2 conditions ~= 200 pairs --
ALL 26 baseline-failure ("hard") tasks are included (the entire pool where
positive success-transfer is observable), topped up with a balanced sample
of baseline-success ("easy") tasks for Delta_eff volume and general
predictor training data.

Resumable: skips (task_id, condition) pairs already present in the output
file, so a crash under Z.ai's rate-limiting picks back up.

Usage: python scripts/run_conditions_full.py
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
from src.experience.storage import load_experiences
from src.experiments.conditions import select_oracle_experience, select_retrieved_experience
from src.experiments.logging import append_result
from src.experiments.runner import run_task


def load_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def pick_target_tasks(config, baseline_by_task, done_task_ids):
    all_tasks = {t["task_id"]: t for t in load_tasks(config["task_source"])}

    by_category = {}
    for task_id, result in baseline_by_task.items():
        bucket = by_category.setdefault(result["shift_category"], {"hard": [], "easy": []})
        bucket["hard" if not result["success"] else "easy"].append(task_id)

    hard_ids = [tid for cat in by_category.values() for tid in cat["hard"]]
    easy_target_total = config["target_total_tasks"] - len(hard_ids)

    rng = random.Random(config["seed"])
    categories = sorted(by_category)
    per_category_target = easy_target_total // len(categories)

    easy_picked = []
    for category in categories:
        pool = list(by_category[category]["easy"])
        rng.shuffle(pool)
        easy_picked.extend(pool[:per_category_target])

    remaining_pool = [tid for cat in by_category.values() for tid in cat["easy"] if tid not in easy_picked]
    rng.shuffle(remaining_pool)
    while len(easy_picked) < easy_target_total and remaining_pool:
        easy_picked.append(remaining_pool.pop())

    target_ids = set(hard_ids) | set(easy_picked)
    new_ids = target_ids - done_task_ids
    return [all_tasks[tid] for tid in new_ids], len(target_ids)


def already_done_pairs(path):
    if not Path(path).exists():
        return set()
    return {(r["task_id"], r["condition"]) for r in load_jsonl(path)}


def main():
    config = load_experiment_config("conditions_full")
    model_config = load_model_config()
    agent_config = load_agent_config()
    agent = Agent(model_config, agent_config)

    output_path = Path(config["output_path"])
    if not output_path.exists():
        pilot_records = load_jsonl(config["pilot_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            for record in pilot_records:
                f.write(json.dumps(record) + "\n")
        print(f"Seeded {len(pilot_records)} pairs from the Step 3 pilot into {output_path}\n")

    baseline_by_task = load_jsonl(config["baseline_results_path"])
    baseline_by_task = {r["task_id"]: r for r in baseline_by_task}
    experiences = load_experiences(config["experience_bank_path"])
    source_tasks_by_id = {t["task_id"]: t for t in load_tasks(config["experience_task_source"])}

    done_pairs = already_done_pairs(output_path)
    done_task_ids = {tid for tid, _cond in done_pairs}

    new_tasks, target_total = pick_target_tasks(config, baseline_by_task, done_task_ids)
    print(f"Target: {target_total} unique tasks (~{target_total * 2} pairs). "
          f"{len(done_task_ids)} tasks already done, {len(new_tasks)} to add.\n")

    errors = []
    for i, task in enumerate(new_tasks):
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
                time.sleep(config["inter_task_delay_seconds"])
                continue

            record["hard_easy"] = hard_easy
            record["baseline_success"] = baseline["success"]
            append_result(record, output_path)
            time.sleep(config["inter_task_delay_seconds"])

        completed_tasks = len(done_task_ids) + i + 1
        if completed_tasks % 10 == 0 or (i + 1) == len(new_tasks):
            current_pairs = len(already_done_pairs(output_path))
            print(f"[{completed_tasks}/{target_total} tasks, {current_pairs} pairs] "
                  f"{task['task_id']} [{task['shift_category']}/{hard_easy}]")

    final_pairs = load_jsonl(output_path)
    print(f"\n{len(final_pairs)} total pairs in {output_path}")
    if errors:
        print(f"{len(errors)} pair(s) errored out and were skipped: {errors}")


if __name__ == "__main__":
    main()
