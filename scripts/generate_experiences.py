"""Step 6: end-to-end experience generation.

Run the agent (condition=baseline, no experience) on a sample of
experience-generation-split tasks, auto-generate raw/reflection/procedure
representations and all 9 properties for each, and store them. Experiences
are generated regardless of whether the source task succeeded or failed --
success_failure is itself one of the 9 logged properties, not a filter.

4 LLM calls per experience: solve the task, reflection, procedure, one
bundled property-rating call (abstraction/specificity/composability/
information_context together, to minimize cost -- see
src/experience/generator.py for the full rule-vs-LLM property breakdown).

Usage: python scripts/generate_experiences.py
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
from src.experience.generator import build_experience
from src.experience.storage import append_experience, load_experiences, start_fresh
from src.experiments.runner import run_task


def pick_source_tasks(config):
    tasks = load_tasks(config["task_source"])
    by_category = {}
    for t in tasks:
        by_category.setdefault(t["shift_category"], []).append(t)

    rng = random.Random(config["seed"])
    sample = []
    for category in sorted(by_category):
        pool = list(by_category[category])
        rng.shuffle(pool)
        sample.extend(pool[:config["tasks_per_category"]])
    return sample


def main():
    config = load_experiment_config("generate_experiences")
    model_config = load_model_config()
    agent_config = load_agent_config()
    agent = Agent(model_config, agent_config)

    output_path = config["output_path"]
    start_fresh(output_path)

    source_tasks = pick_source_tasks(config)
    print(f"Generating experiences from {len(source_tasks)} source tasks "
          f"(condition={config['condition']})\n")

    generated, errors = [], []

    for task in source_tasks:
        try:
            result_record = run_task(agent, task, condition=config["condition"])
            experience = build_experience(
                agent, task, result_record, experience_id=f"exp_{task['task_id']}"
            )
        except APIError as e:
            print(f"{task['task_id']}: ERROR ({e.__class__.__name__}), skipping")
            errors.append({"task_id": task["task_id"], "error": str(e)})
            continue
        finally:
            time.sleep(config["inter_task_delay_seconds"])

        append_experience(experience, output_path)
        generated.append(experience)
        outcome = "success" if result_record["success"] else "failure"
        print(f"{task['task_id']} [{task['shift_category']}]: generated ({outcome})")

    print(f"\n{len(generated)} experience(s) generated and stored to {output_path}")
    if errors:
        print(f"{len(errors)} task(s) errored out and were skipped: {[e['task_id'] for e in errors]}")

    reloaded = load_experiences(output_path)
    ok = len(reloaded) == len(generated)
    print(f"\nReload check: {len(reloaded)}/{len(generated)} records read back "
          f"{'OK' if ok else '-- MISMATCH'}")


if __name__ == "__main__":
    main()
