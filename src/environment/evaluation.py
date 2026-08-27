"""Exact-match grading against BFCL gold calls (not an LLM judge).

Ports BFCL's own AST checker (ShishirPatil/gorilla, Apache 2.0,
berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py)
rather than inventing new matching rules, since CLAUDE.md requires exact
match on BFCL's own ground truth. Scoped to the Python-only, single-turn AST
categories we use (simple/multiple/parallel/parallel_multiple/irrelevance);
Java/JS type conversion and multi-turn variable-substitution logic are
dropped as out of scope.

Two bugs this fixes over the first-pass grader, both by replicating the
official logic instead of inventing new normalization:
  - String comparison: BFCL doesn't do symbolic/math equivalence for
    "^" vs "**" -- it strips " ,./-_*^" and lowercases before comparing
    (standardize_string), so notation differences fall out for free.
  - List/dict-valued parameters (e.g. dietary_requirements: ["vegan"]) need
    per-element / per-key matching against each candidate, not a flat
    str()-of-the-whole-value comparison.
"""

import re

PYTHON_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}

_STANDARDIZE_PATTERN = re.compile(r"[ \,\.\/\-\_\*\^]")


def standardize_string(value):
    """Verbatim port of BFCL's standardize_string: strip ' ,./-_*^', lowercase, ' -> \"."""
    return _STANDARDIZE_PATTERN.sub("", value).lower().replace("'", '"')


def _string_matches(value, acceptable_values):
    standardized_value = standardize_string(value)
    standardized_acceptable = [standardize_string(a) for a in acceptable_values if isinstance(a, str)]
    return standardized_value in standardized_acceptable


def _list_matches(value, acceptable_values):
    standardized_value = [standardize_string(v) if isinstance(v, str) else v for v in value]
    for acceptable in acceptable_values:
        if not isinstance(acceptable, list):
            continue
        standardized_acceptable = [standardize_string(v) if isinstance(v, str) else v for v in acceptable]
        if standardized_value == standardized_acceptable:
            return True
    return False


def _dict_matches(value, acceptable_dicts):
    for acceptable in acceptable_dicts:
        if acceptable == "" or not isinstance(acceptable, dict):
            continue

        ok = True
        for key, sub_value in value.items():
            if key not in acceptable:
                ok = False
                break
            standardized_value = standardize_string(sub_value) if isinstance(sub_value, str) else sub_value
            standardized_acceptable = [
                standardize_string(v) if isinstance(v, str) else v for v in acceptable[key]
            ]
            if standardized_value not in standardized_acceptable:
                ok = False
                break

        if ok:
            for key, acceptable_sub in acceptable.items():
                if key not in value and "" not in acceptable_sub:
                    ok = False
                    break

        if ok:
            return True
    return False


def _param_matches(value, acceptable_values, expected_type):
    python_type = PYTHON_TYPE_MAPPING.get(expected_type, str)

    # BFCL allows Python's int -> float auto-conversion when the schema says "float".
    if expected_type == "float" and type(value) is int:
        value = float(value)

    if type(value) is not python_type:
        return False

    if python_type is dict:
        return _dict_matches(value, acceptable_values)
    if python_type is str:
        return _string_matches(value, acceptable_values)
    if python_type is list:
        return _list_matches(value, acceptable_values)
    return value in acceptable_values


def _function_schema(tools_available, func_name):
    for tool in tools_available:
        if tool["name"] == func_name:
            params = tool.get("parameters", {})
            return params.get("properties", {}), params.get("required", [])
    return {}, []


def _call_matches_gold(predicted_call, gold_func_name, gold_params, param_schema, required_params):
    if predicted_call["name"] != gold_func_name:
        return False

    arguments = predicted_call.get("arguments", {})
    if not isinstance(arguments, dict):
        return False

    for param in required_params:
        if param not in arguments:
            return False

    for param in arguments:
        if param not in param_schema or param not in gold_params:
            return False  # unexpected parameter

    for param, acceptable_values in gold_params.items():
        if param not in arguments:
            if "" not in acceptable_values:
                return False
            continue
        expected_type = param_schema.get(param, {}).get("type", "string")
        if not _param_matches(arguments[param], acceptable_values, expected_type):
            return False

    return True


def exact_match(predicted_calls, gold_call, tools_available):
    """gold_call is None (BFCL irrelevance: no call expected) or a list of
    {func_name: {param: [acceptable...]}} dicts, one per expected call,
    matched order-independently (BFCL's parallel_function_checker_no_order)."""
    if gold_call is None:
        return len(predicted_calls) == 0

    if len(predicted_calls) != len(gold_call):
        return False

    remaining = list(predicted_calls)
    for gold_entry in gold_call:
        (gold_func_name, gold_params), = gold_entry.items()
        param_schema, required_params = _function_schema(tools_available, gold_func_name)
        match_index = next(
            (
                i for i, p in enumerate(remaining)
                if _call_matches_gold(p, gold_func_name, gold_params, param_schema, required_params)
            ),
            None,
        )
        if match_index is None:
            return False
        remaining.pop(match_index)

    return True
