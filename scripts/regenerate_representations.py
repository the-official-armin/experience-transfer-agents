"""Regenerate reflection/procedure/LLM-rated properties for already-stored
experiences, reusing the existing raw representation and rule-based
properties (both derived from the trajectory, which hasn't changed) --
skips re-running the agent on the task entirely. This is also more correct
than a full re-run: CLAUDE.md requires all three representations come from
the SAME trajectory, and re-running the agent could non-deterministically
produce a different one (already observed on this endpoint).

Used to fix a prompt bias that pushed abstraction/specificity/composability
to ceiling -- see the "Deliberately NOT instructed to..." note in
src/experience/representations.py.

Usage: python scripts/regenerate_representations.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.agent import Agent
from src.agent.config import load_agent_config, load_experiment_config, load_model_config
from src.experience.generator import compute_llm_based_properties
from src.experience.representations import generate_procedure, generate_reflection
from src.experience.schema import Experience
from src.experience.storage import append_experience, load_experiences, start_fresh

RULE_BASED_PROPERTY_KEYS = ("length", "num_steps", "success_failure", "task_structure", "tool_dependence")


def main():
    config = load_experiment_config("generate_experiences")
    model_config = load_model_config()
    agent_config = load_agent_config()
    agent = Agent(model_config, agent_config)
    delay = agent_config.get("inter_call_delay_seconds", 0)

    existing = load_experiences(config["output_path"])
    print(f"Regenerating reflection/procedure/properties for {len(existing)} existing experiences "
          f"(3 calls each, vs 4 for a full regenerate)\n")

    regenerated = []
    for record in existing:
        raw = record["representations"]["raw"]

        reflection = generate_reflection(agent, raw)
        time.sleep(delay)
        procedure = generate_procedure(agent, raw)
        time.sleep(delay)

        representations = {"raw": raw, "reflection": reflection, "procedure": procedure}
        llm_properties = compute_llm_based_properties(agent, representations)

        properties = {
            **{k: record["properties"][k] for k in RULE_BASED_PROPERTY_KEYS},
            **llm_properties,
        }

        regenerated.append(Experience(
            experience_id=record["experience_id"],
            source_task_id=record["source_task_id"],
            trajectory=record["trajectory"],
            representations=representations,
            properties=properties,
        ))
        print(f"{record['experience_id']}: regenerated")
        time.sleep(config["inter_task_delay_seconds"])

    start_fresh(config["output_path"])
    for experience in regenerated:
        append_experience(experience, config["output_path"])

    print(f"\n{len(regenerated)} experience(s) regenerated and stored to {config['output_path']}")


if __name__ == "__main__":
    main()
