"""Wrapped, instrumented agent loop.

Generalizes prototype/agent.py (single hardcoded calculator tool, local HF
model.generate()) to: an arbitrary set of task-supplied tools, a GLM-4.7-Flash
backend via the OpenAI-compatible client, and JSON-argument tool calls. The
TOOL:/ARGUMENT:/FINAL: prompting convention and step-loop shape are kept from
the prototype.

BFCL's non-executable categories (simple/multiple/parallel/parallel_multiple/
irrelevance) are graded by exact match against the gold call, not by running
a real tool and observing a result -- so max_steps is 1 by default (configs/
agent.yaml): the agent proposes its call(s) once and we grade what it
proposed, rather than looping tool-call -> observation -> reasoning. Each
trajectory step still carries tool_responses/observations fields (kept null)
so the schema doesn't silently change shape if a later, executable benchmark
category needs them.
"""

import json
import re
import time

from openai import APIError, OpenAI

from src.agent.prompts import build_system_prompt

_CALL_PATTERN = re.compile(
    r"TOOL:\s*(?P<name>\S+)\s*\nARGUMENT:\s*(?P<rest>.*?)(?=\nTOOL:|\nFINAL:|\Z)",
    re.DOTALL,
)
_FINAL_PATTERN = re.compile(r"FINAL:\s*(.*)", re.DOTALL)


def parse_response(text, no_call_marker):
    calls = []
    for match in _CALL_PATTERN.finditer(text):
        name = match.group("name").strip()
        raw_argument = match.group("rest").strip()
        try:
            arguments = json.loads(raw_argument)
        except json.JSONDecodeError:
            arguments = {"_unparsed": raw_argument}
        calls.append({"name": name, "arguments": arguments})

    final_match = _FINAL_PATTERN.search(text)
    final_answer = final_match.group(1).strip() if final_match else None
    declined = final_answer == no_call_marker

    return calls, final_answer, declined


class Agent:

    def __init__(self, model_config, agent_config):
        self.client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"],
            max_retries=2,  # ask_llm() below adds its own outer exponential-backoff retry on top
            timeout=120,
        )
        self.model_config = model_config
        self.agent_config = agent_config

    def ask_llm(self, messages):
        """Every LLM call the agent makes goes through here, so retry
        protection applies uniformly -- not just to the tool-use loop, but
        to reflection/procedure/property-rating calls too (those don't have
        their own retry wrapper, since this is where it lives)."""
        max_retries = self.agent_config.get("api_max_retries", 5)
        base_backoff = self.agent_config.get("api_retry_backoff_seconds", 5)
        max_backoff = self.agent_config.get("api_retry_backoff_cap_seconds", 60)

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_config["model"],
                    messages=messages,
                    temperature=self.model_config.get("temperature", 0),
                    max_tokens=self.model_config.get("max_tokens", 512),
                )
                break
            except APIError as e:
                if attempt == max_retries:
                    raise
                backoff = min(base_backoff * (2 ** (attempt - 1)), max_backoff)
                print(f"  ask_llm: {e.__class__.__name__} on attempt {attempt}, retrying in {backoff}s...")
                time.sleep(backoff)

        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        reasoning_tokens = getattr(
            getattr(response.usage, "completion_tokens_details", None), "reasoning_tokens", None
        )
        return content, response.usage, response.model, choice.finish_reason, reasoning_tokens, attempt - 1

    def run(self, task):
        no_call_marker = self.agent_config.get("no_call_marker", "NO_CALL_NEEDED")
        max_steps = self.agent_config.get("max_steps", 1)
        system_prompt = build_system_prompt(task["tools_available"], no_call_marker)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task["goal"]},
        ]

        trajectory = []
        prompt_tokens = 0
        completion_tokens = 0
        served_model = None
        retry_count = 0
        calls, final_answer = [], None

        for step in range(max_steps):
            response_text, usage, served_model, finish_reason, reasoning_tokens, step_retries = self.ask_llm(messages)
            retry_count += step_retries
            if usage:
                prompt_tokens += usage.prompt_tokens
                completion_tokens += usage.completion_tokens

            calls, final_answer, declined = parse_response(response_text, no_call_marker)

            trajectory.append({
                "step": step,
                "agent_response": response_text,
                "actions": calls,
                "tool_responses": None,  # no real execution for our AST-graded categories
                "observations": None,
                "final_answer": final_answer,
                "declined": declined,
                "finish_reason": finish_reason,
                "reasoning_tokens": reasoning_tokens,
            })

            if calls or final_answer is not None:
                break

        truncated = any(step["finish_reason"] == "length" for step in trajectory)

        return {
            "task_id": task["task_id"],
            "context": {"system_prompt": system_prompt, "goal": task["goal"]},
            "trajectory": trajectory,
            "predicted_calls": calls,
            "final_answer": final_answer,
            "steps": len(trajectory),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "served_model": served_model,
            "truncated": truncated,
            "retry_count": retry_count,
        }
