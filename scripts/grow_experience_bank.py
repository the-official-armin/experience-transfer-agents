"""Day 2, Step 1: grow the experience bank additively.

The 25-hard-task oracle-vs-retrieved gap analysis (Day 1) found 64% (16/25)
show no transfer even under oracle selection -- but that came from a
30-experience bank (~7-8 per shift_category), so "oracle" was picking the
least-bad of a handful of candidates, not the best of a rich pool. This
confounds "experience genuinely doesn't transfer" with "30 experiences
don't cover the task space." This script generates ~80 more experiences
from the UNUSED portion of the 280-task experience-generation pool (only 30
used so far) to actually test that, rather than let it ship as a finding
untested.

Purely additive: appends to the existing data/experiences/experiences.jsonl
bank rather than replacing it. The existing 30 experiences and the 203
Phase 1 scored pairs stay valid as-is (Phase 1 results, based on the
30-experience bank -- document them as such, don't discard or re-score).

Uses the exact pipeline already fixed this week: the anchored property-
rating rubric (src/experience/generator.py's current PROPERTY_RATING_PROMPT,
not the earlier biased/neutral-pole versions), all three representations,
retry_count already excluded from Delta_eff-related properties.

Resumable: the target new-task list is computed once and persisted to
data/experiences/growth_batch_targets.json, so a resume reads the same list
back (rather than recomputing against a bank that's now grown, which would
shrink the remaining target) and just fills in what's missing.

Usage: python scripts/grow_experience_bank.py
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
from src.experience.generator import build_experience
from src.experience.storage import append_experience, load_experiences
from src.experiments.runner import run_task

TARGET_PER_CATEGORY = 20
TARGETS_PATH = Path("data/experiences/growth_batch_targets.json")


def compute_or_load_targets(config):
    if TARGETS_PATH.exists():
        return json.loads(TARGETS_PATH.read_text())

    all_tasks = load_tasks(config["task_source"])
    bank = load_experiences(config["output_path"])
    used_ids = {e["source_task_id"] for e in bank}

    by_category = {}
    for t in all_tasks:
        if t["task_id"] not in used_ids:
            by_category.setdefault(t["shift_category"], []).append(t["task_id"])

    rng = random.Random(config["seed"])
    targets = []
    for category in sorted(by_category):
        pool = list(by_category[category])
        rng.shuffle(pool)
        targets.extend(pool[:TARGET_PER_CATEGORY])

    TARGETS_PATH.write_text(json.dumps(targets, indent=2))
    return targets


def main():
    config = load_experiment_config("generate_experiences")
    model_config = load_model_config()
    agent_config = load_agent_config()
    agent = Agent(model_config, agent_config)

    all_tasks = {t["task_id"]: t for t in load_tasks(config["task_source"])}
    target_ids = compute_or_load_targets(config)

    bank = load_experiences(config["output_path"])
    already_done_source_ids = {e["source_task_id"] for e in bank}
    remaining_ids = [tid for tid in target_ids if tid not in already_done_source_ids]

    print(f"Target: {len(target_ids)} new experiences ({TARGET_PER_CATEGORY}/category). "
          f"{len(target_ids) - len(remaining_ids)} already done, {len(remaining_ids)} remaining.\n")

    errors = []
    for task_id in remaining_ids:
        task = all_tasks[task_id]
        try:
            result_record = run_task(agent, task, condition="baseline")
            experience = build_experience(agent, task, result_record, experience_id=f"exp_{task_id}")
        except APIError as e:
            print(f"{task_id}: ERROR ({e.__class__.__name__}), skipping")
            errors.append(task_id)
            time.sleep(config["inter_task_delay_seconds"])
            continue

        append_experience(experience, config["output_path"])
        outcome = "success" if result_record["success"] else "failure"
        print(f"{task_id} [{task['shift_category']}]: generated ({outcome})")
        time.sleep(config["inter_task_delay_seconds"])

    final_bank = load_experiences(config["output_path"])
    print(f"\nBank size now: {len(final_bank)}")
    if errors:
        print(f"{len(errors)} task(s) errored out and were skipped: {errors}")


if __name__ == "__main__":
    main()
