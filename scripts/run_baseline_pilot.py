"""Step 3/4: baseline calibration pilot, using the instrumented runner.

Run the agent with no experience (condition=baseline) on a sample of held-out
test tasks and report the success rate. Flags to the console if it's near 0%
or near 100%. All run parameters live in configs/experiments.yaml (kept
separate from configs/agent.yaml and configs/models.yaml per CLAUDE.md).

Per CLAUDE.md's Models section: verifies the model actually serving requests
matches the configured model before trusting any result (model choice has
drifted silently before).

Usage: python scripts/run_baseline_pilot.py
"""

import random
import sys
import time
from pathlib import Path

from openai import APIError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.agent import Agent
from src.agent.config import load_agent_config, load_experiment_config, load_model_config
from src.environment.tasks import load_tasks
from src.experiments.logging import append_result, start_fresh
from src.experiments.runner import run_task


def pick_pilot_tasks(config):
    tasks = load_tasks(config["task_source"])
    by_category = {}
    for t in tasks:
        by_category.setdefault(t["shift_category"], []).append(t)

    rng = random.Random(config["seed"])
    pilot = []
    for category in sorted(by_category):
        pool = list(by_category[category])
        rng.shuffle(pool)
        pilot.extend(pool[:config["tasks_per_category"]])
    return pilot


def main():
    config = load_experiment_config("baseline_pilot")
    model_config = load_model_config()
    agent_config = load_agent_config()
    agent = Agent(model_config, agent_config)

    output_path = config["output_path"]
    start_fresh(output_path)

    pilot_tasks = pick_pilot_tasks(config)
    results, errors = [], []
    model_verified = False

    for task in pilot_tasks:
        try:
            record = run_task(agent, task, condition=config["condition"])
        except APIError as e:
            print(f"{task['task_id']} [{task['shift_category']}]: ERROR ({e.__class__.__name__}), skipping")
            errors.append({"task_id": task["task_id"], "shift_category": task["shift_category"],
                            "error": str(e)})
            continue
        finally:
            time.sleep(config["inter_task_delay_seconds"])

        if not model_verified:
            expected, served = model_config["model"], record["served_model"]
            if served and served != expected:
                raise SystemExit(
                    f"Model mismatch: configs/models.yaml says {expected!r} but the API "
                    f"served {served!r}. Stopping before trusting any more results."
                )
            print(f"Model check OK: configured={expected!r}, served={served!r}\n")
            model_verified = True

        append_result(record, output_path)
        results.append(record)
        flags = "".join([
            " [TRUNCATED]" if record["truncated"] else "",
            f" [retried x{record['retry_count']}]" if record["retry_count"] else "",
        ])
        print(f"{task['task_id']} [{task['shift_category']}]: {'PASS' if record['success'] else 'FAIL'}{flags}")

    n = len(results)
    n_success = sum(r["success"] for r in results)
    rate = n_success / n if n else 0.0
    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_completion = sum(r["completion_tokens"] for r in results)
    total_reasoning = sum(r["reasoning_tokens"] or 0 for r in results)
    total_tool_calls = sum(r["tool_call_count"] for r in results)
    total_retries = sum(r["retry_count"] for r in results)

    print(f"\n{n_success}/{n} succeeded ({100 * rate:.0f}%)")
    print("by shift_category:")
    for category in sorted({r["shift_category"] for r in results}):
        cat_results = [r for r in results if r["shift_category"] == category]
        cat_success = sum(r["success"] for r in cat_results)
        print(f"  {category}: {cat_success}/{len(cat_results)}")
    print(f"tokens: {total_prompt} prompt + {total_completion} completion "
          f"= {total_prompt + total_completion} total (of which {total_reasoning} reasoning)")
    print(f"tool calls: {total_tool_calls} total, {total_tool_calls / n:.1f} avg/task" if n else "")
    print(f"retries: {total_retries} total across {sum(1 for r in results if r['retry_count'])} task(s)")

    flag = config["near_floor_ceiling_flag"]
    if rate <= flag or rate >= 1 - flag:
        print(f"\nFLAG: success rate is {100 * rate:.0f}% -- near-floor or near-ceiling, "
              f"check before continuing.")

    if errors:
        max_retries = agent_config.get("api_max_retries", 5)
        print(f"\n{len(errors)} task(s) errored out after {max_retries} attempts (configs/agent.yaml) "
              f"and were skipped (not counted in the rate above): {[e['task_id'] for e in errors]}")

    truncated = [r for r in results if r["truncated"]]
    if truncated:
        print(f"\nWARNING: {len(truncated)} response(s) hit max_tokens before finishing "
              f"(reasoning likely ate the budget) -- their success/fail grade may not reflect "
              f"real capability: {[r['task_id'] for r in truncated]}")


if __name__ == "__main__":
    main()
