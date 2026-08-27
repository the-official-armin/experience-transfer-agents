"""Build a full Experience record (schema.py) from a completed baseline run
on an experience-generation-split task: raw/reflection/procedure
representations (representations.py) plus the 9 CLAUDE.md properties.

Property extraction, by method (for the paper's methodology section):
  RULE / heuristic, free, computed from already-logged data:
    - length            (chars across goal + agent responses)
    - num_steps         (Result record's `steps`)
    - success_failure   (Result record's `success` -- exact-match, not an
                          LLM judge, consistent with CLAUDE.md's evaluation
                          policy)
    - task_structure    (derived from the task's shift_subtype: does solving
                          it require one call, choosing among several tools,
                          or composing multiple calls)
    - tool_dependence    (1 / distinct tool names called -- a structural
                          concentration proxy, not a semantic judgment)
  LLM call, ONE combined call per experience (bundled to minimize cost --
  four separate calls would be 4x the spend for the same information):
    - abstraction
    - specificity
    - composability
    - information_context
"""

import json
import re
import time

from src.experience.representations import build_representations
from src.experience.schema import Experience

_TASK_STRUCTURE_BY_SUBTYPE = {
    "same_api_diff_params": "single_call",
    "swap_similar_api": "single_call_choice",
    "compose_multiple_tools": "multi_call_composition",
    "distractor_tools": "no_call",
}

PROPERTY_RATING_PROMPT = """You will rate an agent's experience from a past task on 4 properties, each on a 1-5 integer scale. Base your ratings only on the reflection and procedure text below.

Reflection:
{reflection}

Procedure:
{procedure}

These 4 properties are INDEPENDENT axes, not one bundled "how concrete does
this feel" judgment -- a lesson can be BOTH abstract AND specific at once
(e.g. "always validate every required parameter has a real value before
calling a tool" is a general principle stated precisely, not vaguely).
Judge each axis on its own, using the anchors below.

- abstraction (does the guidance state a PRINCIPLE vs. describe THIS instance):
  1: "Call get_winner with team_name='Lakers' and date='2024-03-15'."
  3: "When multiple tools could apply, pick the one whose parameters match the entities named in the request."
  5: "Before calling a tool, verify every required parameter has a real value; if one is missing, don't guess."

- specificity (how PRECISE/actionable the guidance is, regardless of abstract vs. concrete):
  1: "Try to use the right tool."
  3: "Check whether the tool's required parameters are all present before calling it."
  5: "Set progression=['C','F','G'], measures=4, instrument='Piano' when calling compose_melody."

- composability (could this combine with guidance from OTHER, DIFFERENT experiences to solve a harder task):
  1: "This exact 3-tool sequence (games.update.find, games.price.find, games.reviews.find) solves this exact multi-part query."
  3: "Decompose a multi-part request into one tool call per distinct sub-question."
  5: "Always validate parameter types before calling any tool, regardless of which tool or domain."

- information_context (how much it assumes/embeds task-specific entities that would NOT carry over to a different task):
  1: "Verify all required parameters are present before calling a tool."
  3: "When a request names multiple entities (teams, products, locations), extract each as a separate parameter."
  5: "The request asked about 'Call of Duty' on 'Xbox' and 'FIFA 21' in the 'American' region -- these were mapped directly to games.price.find and games.reviews.find."

Respond with ONLY a JSON object, no other text: {{"abstraction": <int>, "specificity": <int>, "composability": <int>, "information_context": <int>}}"""


def task_structure(task):
    return _TASK_STRUCTURE_BY_SUBTYPE.get(task.get("shift_subtype"), "unknown")


def tool_dependence(result_record):
    tool_names = {c["name"] for c in result_record["predicted_calls"]}
    if not tool_names:
        return 0.0
    return round(1.0 / len(tool_names), 3)  # 1.0 = concentrated on one tool, lower = spread across more


def compute_rule_based_properties(task, result_record):
    goal_length = len(task["goal"])
    response_length = sum(len(step["agent_response"]) for step in result_record["trajectory"])
    return {
        "length": goal_length + response_length,
        "num_steps": result_record["steps"],
        "success_failure": result_record["success"],
        "task_structure": task_structure(task),
        "tool_dependence": tool_dependence(result_record),
    }


_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

# Property-rating is a short JSON-output task, but has been observed to burn
# far more reasoning than any other call the agent makes (up to ~1000 tokens)
# -- occasionally still empty/unparseable even with a generous max_tokens.
# Retry a few times before falling back to null ratings.
PROPERTY_RATING_RETRIES = 3


def _extract_json(content):
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content.removeprefix("json").strip()
    match = _JSON_OBJECT_PATTERN.search(content)
    return json.loads(match.group(0) if match else content)


def compute_llm_based_properties(agent, representations):
    prompt = PROPERTY_RATING_PROMPT.format(
        reflection=representations["reflection"], procedure=representations["procedure"]
    )
    content = ""
    for attempt in range(PROPERTY_RATING_RETRIES):
        content, usage, model, finish_reason, reasoning_tokens, retry_count = agent.ask_llm([
            {"role": "user", "content": prompt}
        ])
        try:
            return _extract_json(content)
        except (json.JSONDecodeError, AttributeError):
            time.sleep(agent.agent_config.get("inter_call_delay_seconds", 0))
            continue

    return {
        "abstraction": None, "specificity": None,
        "composability": None, "information_context": None,
        "_unparsed": content,
    }


def build_experience(agent, task, result_record, experience_id):
    time.sleep(agent.agent_config.get("inter_call_delay_seconds", 0))
    representations = build_representations(agent, task, result_record)
    properties = {
        **compute_rule_based_properties(task, result_record),
        **compute_llm_based_properties(agent, representations),
    }
    return Experience(
        experience_id=experience_id,
        source_task_id=task["task_id"],
        trajectory=result_record["trajectory"],
        representations=representations,
        properties=properties,
    )
