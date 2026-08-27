"""Noise-floor measurement.

How often does the SAME task, same condition (baseline), same config, flip
its pass/fail outcome across repeated runs at temperature=0? GLM-4.7-Flash
via this endpoint has already been observed to be non-deterministic at
temp=0 (bfcl_multiple_25 in the baseline pilot flipped across 2 of 3 runs).
That was one task's incidental behavior across repeated calibration runs --
this is the dedicated, systematic version: a fixed sample of tasks (one per
shift_category) repeated several times each, so we have an actual noise
floor to interpret Wednesday's Delta(E,T) values against, rather than
guessing from one anecdote.

Usage: python scripts/measure_noise_floor.py
"""

import random
import sys
import time
from collections import Counter
from pathlib import Path

from openai import APIError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.agent import Agent
from src.agent.config import load_agent_config, load_experiment_config, load_model_config
from src.environment.tasks import load_tasks
from src.experiments.logging import append_result, start_fresh
from src.experiments.runner import run_task


def pick_sample_tasks(config):
    tasks = load_tasks(config["task_source"])
    by_category = {}
    for t in tasks:
        by_category.setdefault(t["shift_category"], []).append(t)

    rng = random.Random(config["seed"])
    sample = []
    for category in sorted(by_category):
        pool = list(by_category[category])
        rng.shuffle(pool)
        sample.append(pool[0])
    return sample


def main():
    config = load_experiment_config("noise_floor")
    model_config = load_model_config()
    agent_config = load_agent_config()
    agent = Agent(model_config, agent_config)

    output_path = config["output_path"]
    start_fresh(output_path)

    sample_tasks = pick_sample_tasks(config)
    print(f"Sampled {len(sample_tasks)} tasks (1/shift_category), {config['repeats']} repeats each "
          f"= {len(sample_tasks) * config['repeats']} trials\n")

    outcomes_by_task = {t["task_id"]: [] for t in sample_tasks}

    for task in sample_tasks:
        for trial in range(config["repeats"]):
            try:
                record = run_task(agent, task, condition=config["condition"])
            except APIError as e:
                print(f"{task['task_id']} trial {trial}: ERROR ({e.__class__.__name__}), skipping")
                continue
            finally:
                time.sleep(config["inter_task_delay_seconds"])

            record["trial"] = trial
            append_result(record, output_path)
            outcomes_by_task[task["task_id"]].append(record["success"])
            print(f"{task['task_id']} [{task['shift_category']}] trial {trial}: "
                  f"{'PASS' if record['success'] else 'FAIL'}")

    print("\n--- noise floor ---")
    total_trials, total_flips = 0, 0
    for task_id, outcomes in outcomes_by_task.items():
        if not outcomes:
            continue
        modal = Counter(outcomes).most_common(1)[0][0]
        flips = sum(1 for o in outcomes if o != modal)
        total_trials += len(outcomes)
        total_flips += flips
        print(f"{task_id}: {outcomes} -- {flips}/{len(outcomes)} disagree with modal outcome ({modal})")

    rate = total_flips / total_trials if total_trials else 0.0
    print(f"\nnoise floor: {total_flips}/{total_trials} trials disagreed with their task's modal "
          f"outcome ({100 * rate:.0f}%) -- treat Delta(E,T) magnitudes below this as within noise.")


if __name__ == "__main__":
    main()
