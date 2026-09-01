"""Step 6: does Tuesday's noise-floor concentration hypothesis hold up now
that we have real novel_tool volume, or was it specific to one task?

Tuesday's noise-floor measurement (measure_noise_floor.py) had n=1 novel_tool
task (bfcl_multiple_25, 3 repeats, 1/3 flipped) -- too thin to say whether
instability concentrates on that one swap-similar-API decision or is a
broader novel_tool property. Now that Step 5 has 24 distinct novel_tool
tasks, this repeats the SAME baseline-condition, temp=0, repeated-trial
methodology across 5 distinct novel_tool tasks (the original suspect +
the 2 tasks that showed real signal in Step 5's single-shot data +
2 more random ones) to get real repeated-trial evidence at the category
level, not just one anecdote.

Resumable: counts already-saved trials per task_id in the output file and
only runs the shortfall, rather than wiping progress and starting over.

Usage: python scripts/measure_novel_tool_noise.py
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.agent import Agent
from src.agent.config import load_agent_config, load_model_config
from src.environment.tasks import load_tasks
from src.experiments.logging import append_result
from src.experiments.runner import run_task

TASK_IDS = ["bfcl_multiple_25", "bfcl_multiple_76", "bfcl_multiple_196", "bfcl_multiple_79", "bfcl_multiple_144"]
REPEATS = 3
INTER_TRIAL_DELAY = 3
OUTPUT_PATH = "data/results/novel_tool_noise.jsonl"


def load_existing_outcomes(path):
    if not Path(path).exists():
        return {}
    with Path(path).open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    outcomes_by_task = {}
    for r in records:
        outcomes_by_task.setdefault(r["task_id"], []).append(r["success"])
    return outcomes_by_task


def main():
    model_config = load_model_config()
    agent_config = load_agent_config()
    agent = Agent(model_config, agent_config)

    all_tasks = {t["task_id"]: t for t in load_tasks("tasks/test_tasks/tasks.jsonl")}
    outcomes_by_task = load_existing_outcomes(OUTPUT_PATH)
    for tid in TASK_IDS:
        outcomes_by_task.setdefault(tid, [])

    already_done = sum(len(v) for v in outcomes_by_task.values())
    print(f"Resuming: {already_done}/{len(TASK_IDS) * REPEATS} trials already done\n")

    for task_id in TASK_IDS:
        task = all_tasks[task_id]
        missing = REPEATS - len(outcomes_by_task[task_id])
        for _ in range(missing):
            record = run_task(agent, task, condition="baseline")
            record["trial"] = len(outcomes_by_task[task_id])
            append_result(record, OUTPUT_PATH)
            outcomes_by_task[task_id].append(record["success"])
            print(f"{task_id} trial {record['trial']}: {'PASS' if record['success'] else 'FAIL'}")
            time.sleep(INTER_TRIAL_DELAY)

    print("\n--- novel_tool noise floor (5 tasks x 3 repeats) ---")
    total_trials, total_flips, flip_tasks = 0, 0, []
    for task_id in TASK_IDS:
        outcomes = outcomes_by_task[task_id]
        modal = Counter(outcomes).most_common(1)[0][0]
        flips = sum(1 for o in outcomes if o != modal)
        total_trials += len(outcomes)
        total_flips += flips
        if flips:
            flip_tasks.append(task_id)
        print(f"{task_id}: {outcomes} -- {flips}/{len(outcomes)} disagree with modal ({modal})")

    rate = total_flips / total_trials if total_trials else 0.0
    print(f"\nnovel_tool noise floor: {total_flips}/{total_trials} trials disagreed with their task's "
          f"modal outcome ({100 * rate:.0f}%)")
    print(f"tasks with ANY flip: {len(flip_tasks)}/{len(TASK_IDS)} -- {flip_tasks}")
    if len(flip_tasks) <= 1:
        print("CONCENTRATED: instability still isolated to (at most) one task -- Tuesday's hypothesis holds.")
    else:
        print("SPREAD: instability appears across multiple distinct novel_tool tasks, not just one -- "
              "Tuesday's concentration hypothesis does NOT hold at this larger sample.")


if __name__ == "__main__":
    main()
