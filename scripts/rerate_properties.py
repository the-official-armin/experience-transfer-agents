"""Re-run only the property-rating call on already-stored experiences,
reusing their existing reflection/procedure text unchanged. Cheaper and more
targeted than regenerate_representations.py -- 1 call per experience instead
of 3 -- used when the diagnosis is a rating-rubric problem (the text already
varies meaningfully; the rating call isn't discriminating it), not a
generation problem. See the anchored PROPERTY_RATING_PROMPT in
src/experience/generator.py.

Writes to a separate progress file incrementally and resumes from it on a
second run (skipping already-rated experience_ids) -- the endpoint has shown
sustained rate-limiting that can exhaust even a 6-attempt/135s backoff
budget, so a crash losing zero completed work matters more than usual here.
The canonical output path is only overwritten once ALL experiences are done.

Usage: python scripts/rerate_properties.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.agent import Agent
from src.agent.config import load_agent_config, load_experiment_config, load_model_config
from src.experience.generator import compute_llm_based_properties
from src.experience.schema import Experience
from src.experience.storage import append_experience, load_experiences, start_fresh

RULE_BASED_PROPERTY_KEYS = ("length", "num_steps", "success_failure", "task_structure", "tool_dependence")


def main():
    config = load_experiment_config("generate_experiences")
    model_config = load_model_config()
    agent_config = load_agent_config()
    agent = Agent(model_config, agent_config)

    output_path = Path(config["output_path"])
    progress_path = output_path.with_name(output_path.stem + ".rerating.jsonl")

    source = load_experiences(output_path)
    done_ids = set()
    if progress_path.exists():
        done_ids = {e["experience_id"] for e in load_experiences(progress_path)}
        print(f"Resuming: {len(done_ids)}/{len(source)} already re-rated in {progress_path}")

    remaining = [r for r in source if r["experience_id"] not in done_ids]
    print(f"Re-rating {len(remaining)} experience(s) (1 call each)\n")

    for record in remaining:
        llm_properties = compute_llm_based_properties(agent, record["representations"])
        properties = {
            **{k: record["properties"][k] for k in RULE_BASED_PROPERTY_KEYS},
            **llm_properties,
        }
        experience = Experience(
            experience_id=record["experience_id"],
            source_task_id=record["source_task_id"],
            trajectory=record["trajectory"],
            representations=record["representations"],
            properties=properties,
        )
        append_experience(experience, progress_path)
        print(f"{record['experience_id']}: {llm_properties}")
        time.sleep(config["inter_task_delay_seconds"])

    final = load_experiences(progress_path)
    start_fresh(output_path)
    for record in final:
        append_experience(record, output_path)
    progress_path.unlink()

    print(f"\n{len(final)} experience(s) re-rated and stored to {output_path}")


if __name__ == "__main__":
    main()
