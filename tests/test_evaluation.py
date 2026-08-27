"""Unit tests for src/environment/evaluation.py -- the ported BFCL AST checker.

Written before scaling up API spend (CLAUDE.md engineering practice), and
directly motivated by two real bugs the first-pass grader had during Step 3:
notation differences ("^" vs "**") and list/dict-valued parameters.
"""

from src.environment.evaluation import exact_match, standardize_string

TRIANGLE_TOOL = [{
    "name": "calculate_triangle_area",
    "parameters": {
        "type": "dict",
        "properties": {
            "base": {"type": "integer"},
            "height": {"type": "integer"},
            "unit": {"type": "string"},
        },
        "required": ["base", "height"],
    },
}]

DERIVATIVE_TOOL = [{
    "name": "calculus.derivative",
    "parameters": {
        "type": "dict",
        "properties": {
            "function": {"type": "string"},
            "value": {"type": "integer"},
            "function_variable": {"type": "string"},
        },
        "required": ["function", "value"],
    },
}]

RESTAURANT_TOOL = [{
    "name": "find_restaurants",
    "parameters": {
        "type": "dict",
        "properties": {
            "location": {"type": "string"},
            "dietary_requirements": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["location"],
    },
}]


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def test_standardize_string_strips_notation_punctuation():
    assert standardize_string("3x^2 + 2x - 1") == standardize_string("3x**2 + 2x - 1")


def test_simple_scalar_match():
    gold = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
    predicted = [call("calculate_triangle_area", base=10, height=5)]
    assert exact_match(predicted, gold, TRIANGLE_TOOL) is True


def test_optional_param_with_empty_string_sentinel_can_be_omitted_or_provided():
    gold = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
    omitted = [call("calculate_triangle_area", base=10, height=5)]
    provided = [call("calculate_triangle_area", base=10, height=5, unit="units")]
    assert exact_match(omitted, gold, TRIANGLE_TOOL) is True
    assert exact_match(provided, gold, TRIANGLE_TOOL) is True


def test_required_param_missing_fails():
    gold = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
    predicted = [call("calculate_triangle_area", base=10)]
    assert exact_match(predicted, gold, TRIANGLE_TOOL) is False


def test_unexpected_param_fails():
    gold = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
    predicted = [call("calculate_triangle_area", base=10, height=5, extra="nope")]
    assert exact_match(predicted, gold, TRIANGLE_TOOL) is False


def test_string_notation_equivalence_via_standardize_string():
    gold = [{"calculus.derivative": {
        "function": ["3x**2 + 2x - 1"], "value": [5], "function_variable": ["x", ""],
    }}]
    predicted = [call("calculus.derivative", function="3x^2 + 2x - 1", value=5, function_variable="x")]
    assert exact_match(predicted, gold, DERIVATIVE_TOOL) is True


def test_list_valued_param_matches_by_element():
    gold = [{"find_restaurants": {
        "location": ["San Francisco", "SF"], "dietary_requirements": [["vegan"]],
    }}]
    predicted = [call("find_restaurants", location="SF", dietary_requirements=["vegan"])]
    assert exact_match(predicted, gold, RESTAURANT_TOOL) is True


def test_list_valued_param_wrong_elements_fails():
    gold = [{"find_restaurants": {
        "location": ["San Francisco", "SF"], "dietary_requirements": [["vegan"]],
    }}]
    predicted = [call("find_restaurants", location="SF", dietary_requirements=["vegetarian"])]
    assert exact_match(predicted, gold, RESTAURANT_TOOL) is False


def test_parallel_calls_match_regardless_of_order():
    gold = [
        {"calculus.derivative": {"function": ["3x**2"], "value": [5], "function_variable": ["x", ""]}},
        {"calculus.derivative": {"function": ["4y**3"], "value": [3], "function_variable": ["y", ""]}},
    ]
    predicted = [
        call("calculus.derivative", function="4y**3", value=3, function_variable="y"),
        call("calculus.derivative", function="3x**2", value=5, function_variable="x"),
    ]
    assert exact_match(predicted, gold, DERIVATIVE_TOOL) is True


def test_wrong_call_count_fails():
    gold = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
    predicted = [
        call("calculate_triangle_area", base=10, height=5),
        call("calculate_triangle_area", base=1, height=1),
    ]
    assert exact_match(predicted, gold, TRIANGLE_TOOL) is False


def test_irrelevance_no_gold_call_expects_no_predicted_calls():
    assert exact_match([], None, TRIANGLE_TOOL) is True
    assert exact_match([call("calculate_triangle_area", base=1, height=1)], None, TRIANGLE_TOOL) is False


def test_wrong_function_name_fails():
    gold = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
    predicted = [call("wrong_function", base=10, height=5)]
    assert exact_match(predicted, gold, TRIANGLE_TOOL) is False


def test_int_to_float_autoconversion():
    tool = [{
        "name": "circle_area",
        "parameters": {"type": "dict", "properties": {"radius": {"type": "float"}}, "required": ["radius"]},
    }]
    gold = [{"circle_area": {"radius": [5.0]}}]
    predicted = [call("circle_area", radius=5)]
    assert exact_match(predicted, gold, tool) is True
