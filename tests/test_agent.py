"""Unit tests for src/agent/agent.py's parse_response (no API calls)."""

from src.agent.agent import parse_response

NO_CALL_MARKER = "NO_CALL_NEEDED"


def test_parses_single_tool_call():
    text = 'TOOL: calculate_triangle_area\nARGUMENT: {"base": 10, "height": 5}\nFINAL: 25'
    calls, final_answer, declined = parse_response(text, NO_CALL_MARKER)
    assert calls == [{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}]
    assert final_answer == "25"
    assert declined is False


def test_parses_multiple_tool_calls_in_one_response():
    text = (
        "TOOL: calculus.derivative\n"
        'ARGUMENT: {"function": "3x**2", "value": 5}\n'
        "TOOL: calculus.derivative\n"
        'ARGUMENT: {"function": "4y**3", "value": 3}\n'
        "FINAL: done"
    )
    calls, final_answer, declined = parse_response(text, NO_CALL_MARKER)
    assert len(calls) == 2
    assert calls[0]["name"] == "calculus.derivative"
    assert calls[0]["arguments"]["value"] == 5
    assert calls[1]["arguments"]["value"] == 3
    assert final_answer == "done"


def test_declined_when_final_is_no_call_marker():
    text = f"FINAL: {NO_CALL_MARKER}"
    calls, final_answer, declined = parse_response(text, NO_CALL_MARKER)
    assert calls == []
    assert declined is True


def test_no_final_no_calls_returns_none():
    calls, final_answer, declined = parse_response("I'm thinking about it.", NO_CALL_MARKER)
    assert calls == []
    assert final_answer is None
    assert declined is False


def test_malformed_json_argument_falls_back_to_unparsed():
    text = "TOOL: calculate_triangle_area\nARGUMENT: {not valid json\nFINAL: oops"
    calls, final_answer, declined = parse_response(text, NO_CALL_MARKER)
    assert len(calls) == 1
    assert "_unparsed" in calls[0]["arguments"]
