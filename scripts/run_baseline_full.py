"""Step 2 (Wednesday): baseline condition at full scale (all 280 held-out
test tasks), broken out by shift_category -- the ceiling check CLAUDE.md
has been waiting on since Step 3's calibration (95% aggregate on 20 tasks,
per-category breakdown unknown at real sample size).

Resumable: skips task_ids already present in the output file on a re-run,
so a crash under Z.ai's rate-limiting picks back up instead of repeating
completed work (results are appended one at a time as they complete, never
buffered and written only at the end).

Usage: python scripts/run_baseline_full.py
"""

import json
import sys
import time
from pathlib import Path

from openai import APIError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.agent import Agent
from src.agent.config import load_agent_config, load_experiment_config, load_model_config
from src.environment.tasks import load_tasks
from src.experiments.logging import append_result
from src.experiments.runner import run_task


def already_done_ids(path):
    if not Path(path).exists():
        return set()
    with Path(path).open() as f:
        return {json.loads(line)["task_id"] for line in f if line.strip()}


def main():
    config = load_experiment_config("baseline_full")
    model_config = load_model_config()
    agent_config = load_agent_config()
    agent = Agent(model_config, agent_config)

    output_path = config["output_path"]
    all_tasks = load_tasks(config["task_source"])
    done_ids = already_done_ids(output_path)
    remaining = [t for t in all_tasks if t["task_id"] not in done_ids]

    print(f"{len(done_ids)}/{len(all_tasks)} already done, {len(remaining)} remaining\n")

    model_verified = bool(done_ids)
    errors = []

    for i, task in enumerate(remaining):
        try:
            record = run_task(agent, task, condition=config["condition"])
        except APIError as e:
            print(f"{task['task_id']}: ERROR ({e.__class__.__name__}), skipping")
            errors.append(task["task_id"])
            time.sleep(config["inter_task_delay_seconds"])
            continue

        if not model_verified:
            expected, served = model_config["model"], record["served_model"]
            if served and served != expected:
                raise SystemExit(
                    f"Model mismatch: configs/models.yaml says {expected!r} but the API served "
                    f"{served!r}. Stopping before trusting any more results."
                )
            print(f"Model check OK: configured={expected!r}, served={served!r}\n")
            model_verified = True

        append_result(record, output_path)
        completed = len(done_ids) + i + 1
        if completed % 20 == 0 or (i + 1) == len(remaining):
            print(f"[{completed}/{len(all_tasks)}] {task['task_id']} [{task['shift_category']}]: "
                  f"{'PASS' if record['success'] else 'FAIL'}")
        time.sleep(config["inter_task_delay_seconds"])

    with Path(output_path).open() as f:
        results = [json.loads(line) for line in f]

    n_success = sum(r["success"] for r in results)
    print(f"\n{n_success}/{len(results)} succeeded ({100 * n_success / len(results):.0f}%)")
    print("by shift_category:")

    ceiling_flag = config["ceiling_flag"]
    flagged = []
    for category in sorted({r["shift_category"] for r in results}):
        cat_results = [r for r in results if r["shift_category"] == category]
        cat_success = sum(r["success"] for r in cat_results)
        rate = cat_success / len(cat_results)
        flag = ""
        if rate >= ceiling_flag and category != "familiar":
            flag = "  <-- CEILING FLAG"
            flagged.append((category, rate, len(cat_results)))
        print(f"  {category}: {cat_success}/{len(cat_results)} ({100 * rate:.0f}%){flag}")

    if errors:
        print(f"\n{len(errors)} task(s) errored out and were skipped: {errors}")

    if flagged:
        print(f"\nFLAG: {len(flagged)} non-familiar categor{'y is' if len(flagged)==1 else 'ies are'} "
              f"near-ceiling (>= {100*ceiling_flag:.0f}%) -- thin positive-transfer signal, check before "
              f"continuing to Step 3: {flagged}")


if __name__ == "__main__":
    main()
