"""Generate the three experience representations from ONE source trajectory
(raw, reflection, procedure) -- all derived from the SAME trajectory per
CLAUDE.md ("never independently collected, or the representation comparison
is confounded with content"), using the SAME model that ran the agent
(single-model policy: never a stronger model for representation generation,
or "does representation help" gets confounded with "is the generator model
smarter").

raw needs no LLM call (it's a formatted view of the already-logged
trajectory); reflection and procedure are each one LLM call.
"""

import json
import time


def build_raw_representation(task, result_record):
    lines = [
        f"Goal: {task['goal']}",
        f"Tools available: {', '.join(t['name'] for t in task['tools_available'])}",
    ]
    for step in result_record["trajectory"]:
        lines.append(f"Step {step['step']} response: {step['agent_response']}")
        for action in step["actions"]:
            lines.append(f"  called {action['name']}({json.dumps(action['arguments'])})")
    lines.append(f"Outcome: {'success' if result_record['success'] else 'failure'}")
    return "\n".join(lines)


# Deliberately NOT instructed to "be general" or "reusable" or "not just
# restate what happened" -- those instructions were pushing every generated
# representation toward maximal abstraction/specificity/composability
# regardless of the underlying task, which then showed up as near-zero
# variance when rated (see the ceiling-clustering finding). Left neutral so
# the model's own natural framing -- concrete or general -- is what gets
# rated, not a framing we asked it to perform.
REFLECTION_PROMPT = """Below is a record of an agent's attempt at a tool-use task.

{raw}

Write a short reflection (2-4 sentences) on this attempt: what happened,
whether it worked, and why. Respond with only the reflection text, no
preamble."""

PROCEDURE_PROMPT = """Below is a record of an agent's attempt at a tool-use task.

{raw}

Write a step-by-step procedure (3-6 numbered steps) describing the approach
taken in this attempt. Respond with only the numbered procedure, no
preamble."""


def generate_reflection(agent, raw_representation):
    messages = [{"role": "user", "content": REFLECTION_PROMPT.format(raw=raw_representation)}]
    content, usage, model, finish_reason, reasoning_tokens, retry_count = agent.ask_llm(messages)
    return content


def generate_procedure(agent, raw_representation):
    messages = [{"role": "user", "content": PROCEDURE_PROMPT.format(raw=raw_representation)}]
    content, usage, model, finish_reason, reasoning_tokens, retry_count = agent.ask_llm(messages)
    return content


def build_representations(agent, task, result_record):
    delay = agent.agent_config.get("inter_call_delay_seconds", 0)
    raw = build_raw_representation(task, result_record)

    reflection = generate_reflection(agent, raw)
    time.sleep(delay)
    procedure = generate_procedure(agent, raw)
    time.sleep(delay)

    return {"raw": raw, "reflection": reflection, "procedure": procedure}
