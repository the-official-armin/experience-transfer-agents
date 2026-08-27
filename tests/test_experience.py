"""Unit tests for src/experience/{schema,storage,generator}.py.

Covers the deterministic parts (schema round-trip, storage, rule-based
property extraction) with no API calls. The 4 LLM-rated properties
(compute_llm_based_properties) are validated via a live smoke test instead,
same pattern as the agent harness in Step 4.
"""

import json

from src.experience.generator import _extract_json, compute_rule_based_properties, task_structure, tool_dependence
from src.experience.schema import Experience
from src.experience.storage import append_experience, load_experiences, start_fresh

TASK = {
    "task_id": "bfcl_multiple_25",
    "goal": "Find the store price of Assassins Creed Valhalla on PlayStation.",
    "shift_subtype": "swap_similar_api",
    "tools_available": [{"name": "video_games.store_price", "parameters": {}}],
}

RESULT_RECORD = {
    "success": True,
    "steps": 1,
    "predicted_calls": [{"name": "video_games.store_price", "arguments": {"game_title": "x"}}],
    "trajectory": [{"step": 0, "agent_response": "TOOL: video_games.store_price\nARGUMENT: {}"}],
}


def test_task_structure_maps_known_subtypes():
    assert task_structure({"shift_subtype": "same_api_diff_params"}) == "single_call"
    assert task_structure({"shift_subtype": "swap_similar_api"}) == "single_call_choice"
    assert task_structure({"shift_subtype": "compose_multiple_tools"}) == "multi_call_composition"
    assert task_structure({"shift_subtype": "distractor_tools"}) == "no_call"


def test_task_structure_unknown_subtype_falls_back():
    assert task_structure({"shift_subtype": "something_new"}) == "unknown"


def test_tool_dependence_single_tool_is_fully_concentrated():
    result = {"predicted_calls": [{"name": "a", "arguments": {}}, {"name": "a", "arguments": {}}]}
    assert tool_dependence(result) == 1.0


def test_tool_dependence_multiple_tools_is_spread():
    result = {"predicted_calls": [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]}
    assert tool_dependence(result) == 0.5


def test_tool_dependence_no_calls_is_zero():
    assert tool_dependence({"predicted_calls": []}) == 0.0


def test_compute_rule_based_properties_shape():
    properties = compute_rule_based_properties(TASK, RESULT_RECORD)
    assert properties["success_failure"] is True
    assert properties["num_steps"] == 1
    assert properties["task_structure"] == "single_call_choice"
    assert properties["tool_dependence"] == 1.0
    assert properties["length"] > 0


def test_experience_to_dict_round_trips_through_json():
    exp = Experience(
        experience_id="exp_1",
        source_task_id="bfcl_multiple_25",
        trajectory=RESULT_RECORD["trajectory"],
        representations={"raw": "r", "reflection": "f", "procedure": "p"},
        properties=compute_rule_based_properties(TASK, RESULT_RECORD),
    )
    round_tripped = json.loads(json.dumps(exp.to_dict()))
    assert round_tripped["experience_id"] == "exp_1"
    assert round_tripped["representations"]["reflection"] == "f"


def test_storage_append_and_load_round_trip(tmp_path):
    path = tmp_path / "experiences.jsonl"
    start_fresh(path)

    exp = Experience(
        experience_id="exp_1",
        source_task_id="bfcl_multiple_25",
        trajectory=RESULT_RECORD["trajectory"],
        representations={"raw": "r", "reflection": "f", "procedure": "p"},
        properties=compute_rule_based_properties(TASK, RESULT_RECORD),
    )
    append_experience(exp, path)

    loaded = load_experiences(path)
    assert len(loaded) == 1
    assert loaded[0]["experience_id"] == "exp_1"
    assert loaded[0]["properties"]["task_structure"] == "single_call_choice"


def test_extract_json_parses_plain_json():
    assert _extract_json('{"abstraction": 5, "specificity": 4}') == {"abstraction": 5, "specificity": 4}


def test_extract_json_strips_markdown_code_fence():
    content = '```json\n{\n  "abstraction": 5,\n  "specificity": 3\n}\n```'
    assert _extract_json(content) == {"abstraction": 5, "specificity": 3}


def test_extract_json_ignores_surrounding_prose():
    content = 'Sure, here are the ratings:\n{"abstraction": 5, "specificity": 3}\nHope that helps!'
    assert _extract_json(content) == {"abstraction": 5, "specificity": 3}


def test_start_fresh_clears_stale_data(tmp_path):
    path = tmp_path / "experiences.jsonl"
    path.write_text('{"stale": true}\n')
    start_fresh(path)
    assert load_experiences(path) == []
