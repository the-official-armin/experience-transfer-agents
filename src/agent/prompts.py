"""System prompt construction.

Generalizes the prototype's TOOL: / ARGUMENT: / FINAL: convention
(prototype/agent.py) from a single hardcoded calculator tool to an arbitrary
set of task-supplied tool schemas with JSON arguments, and adds support for
proposing multiple calls in one response (BFCL parallel/parallel_multiple)
and declining to call anything (BFCL irrelevance). The step-by-step loop
shape and the TOOL:/FINAL: format itself are kept from the prototype.
"""

import json


def _format_tool(tool):
    return (
        f"{tool['name']}({', '.join(tool.get('parameters', {}).get('properties', {}))})\n"
        f"  {tool['description']}\n"
        f"  parameters: {json.dumps(tool.get('parameters', {}))}"
    )


def build_system_prompt(tools, no_call_marker="NO_CALL_NEEDED"):
    tool_list = "\n\n".join(_format_tool(t) for t in tools)

    return f"""
You are a tool-using agent. You have access to the following tool(s):

{tool_list}

You must follow this format exactly.

If a tool call is needed, output one block per call:

TOOL: <tool name>
ARGUMENT: <JSON object of arguments, matching the parameter names above exactly>

You may output more than one TOOL/ARGUMENT block if the request needs more
than one call. If none of the tools are relevant to the request, output
exactly:

FINAL: {no_call_marker}

Otherwise, after any tool calls, end with:

FINAL: <answer>

Do not write anything before TOOL or FINAL. Do not include words such as
"assistant", "user", or "system".
""".strip()
