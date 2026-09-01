"""Run a single task through the agent under a given experimental condition
and package the outcome into a Result record (CLAUDE.md Data Schema
Contracts: task_id, experience_id_or_null, condition, success, steps,
trajectory, timestamp). Token usage is logged as separate prompt_tokens/
completion_tokens/reasoning_tokens rather than one combined number, since
reasoning effort is itself a plausible efficiency signal, not just action
count.

Retry-with-backoff for transient API errors lives inside Agent.ask_llm (so
it covers every call the agent makes, not just this one) -- this function
just reads the resulting retry_count back out.

`condition="baseline"` runs with no experience; `oracle`/`retrieved` require
an `experience` dict. `representation` selects which of the experience's
three encodings (raw/reflection/procedure) gets injected into the prompt --
defaults to "raw" (Wednesday's condition results all used raw only;
crossing representations is Day 2 work). Generation-side budget (max_tokens,
temperature, max_steps) is identical across all three conditions and all
three representations by construction, since they all go through the same
Agent/model_config -- only the prompt content differs, which is the point.
"""

from src.environment.evaluation import exact_match

CONDITIONS = ("baseline", "oracle", "retrieved")
REPRESENTATIONS = ("raw", "reflection", "procedure")


def run_task(agent, task, condition="baseline", experience=None, representation="raw"):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}, expected one of {CONDITIONS}")
    if condition == "baseline" and experience is not None:
        raise ValueError("baseline condition must not receive an experience")
    if condition != "baseline" and experience is None:
        raise ValueError(f"{condition!r} condition requires an experience")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation {representation!r}, expected one of {REPRESENTATIONS}")

    outcome = agent.run(task, experience=experience, representation=representation)
    success = exact_match(outcome["predicted_calls"], task["gold_call"], task["tools_available"])
    declined = outcome["trajectory"][-1]["declined"] if outcome["trajectory"] else False
    tool_call_count = sum(len(step["actions"]) for step in outcome["trajectory"])

    reasoning_values = [
        step["reasoning_tokens"] for step in outcome["trajectory"] if step["reasoning_tokens"] is not None
    ]
    reasoning_tokens = sum(reasoning_values) if reasoning_values else None

    return {
        "task_id": task["task_id"],
        "shift_category": task["shift_category"],
        "condition": condition,
        "experience_id_or_null": experience["experience_id"] if experience else None,
        "representation": representation if experience else None,
        "success": success,
        "declined": declined,
        "truncated": outcome["truncated"],
        "prompt_tokens": outcome["prompt_tokens"],
        "completion_tokens": outcome["completion_tokens"],
        "reasoning_tokens": reasoning_tokens,
        "tool_call_count": tool_call_count,
        "retry_count": outcome["retry_count"],
        "steps": outcome["steps"],
        "trajectory": outcome["trajectory"],
        "context": outcome["context"],
        "predicted_calls": outcome["predicted_calls"],
        "gold_call": task["gold_call"],
        "served_model": outcome["served_model"],
    }
