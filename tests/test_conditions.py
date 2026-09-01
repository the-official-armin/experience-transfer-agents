"""Unit tests for src/experiments/conditions.py's oracle selection rule."""

from src.experiments.conditions import select_oracle_experience


def tool(name):
    return {"name": name, "parameters": {}}


def task(task_id, tool_names, shift_subtype="same_api_diff_params"):
    return {"task_id": task_id, "tools_available": [tool(n) for n in tool_names], "shift_subtype": shift_subtype}


def experience(experience_id, source_task_id, success=True):
    return {"experience_id": experience_id, "source_task_id": source_task_id,
            "properties": {"success_failure": success}}


def test_exact_tool_overlap_wins():
    query = task("t1", ["video_games.store_price"])
    sources = {
        "sA": task("sA", ["video_games.store_price"]),
        "sB": task("sB", ["video_games.on_sale"]),  # family match only
    }
    experiences = [experience("expA", "sA"), experience("expB", "sB")]
    best = select_oracle_experience(query, experiences, sources)
    assert best["experience_id"] == "expA"


def test_family_overlap_wins_over_no_overlap():
    query = task("t1", ["video_games.store_price"])
    sources = {
        "sA": task("sA", ["video_games.on_sale"]),       # family match
        "sB": task("sB", ["calculus.derivative"]),        # no overlap at all
    }
    experiences = [experience("expA", "sA"), experience("expB", "sB")]
    best = select_oracle_experience(query, experiences, sources)
    assert best["experience_id"] == "expA"


def test_shift_subtype_match_wins_when_no_tool_or_family_overlap():
    query = task("t1", ["video_games.store_price"], shift_subtype="swap_similar_api")
    sources = {
        "sA": task("sA", ["calculus.derivative"], shift_subtype="swap_similar_api"),  # subtype match
        "sB": task("sB", ["restaurant.find_nearby"], shift_subtype="compose_multiple_tools"),
    }
    experiences = [experience("expA", "sA"), experience("expB", "sB")]
    best = select_oracle_experience(query, experiences, sources)
    assert best["experience_id"] == "expA"


def test_prefers_successful_source_task_on_tie():
    query = task("t1", ["video_games.store_price"])
    sources = {
        "sA": task("sA", ["calculus.derivative"]),
        "sB": task("sB", ["restaurant.find_nearby"]),
    }
    experiences = [
        experience("expA", "sA", success=False),
        experience("expB", "sB", success=True),
    ]
    best = select_oracle_experience(query, experiences, sources)
    assert best["experience_id"] == "expB"


def test_deterministic_tie_break_on_experience_id():
    query = task("t1", ["video_games.store_price"])
    sources = {"sA": task("sA", ["video_games.store_price"]), "sB": task("sB", ["video_games.store_price"])}
    experiences = [experience("exp_z", "sA"), experience("exp_a", "sB")]
    best = select_oracle_experience(query, experiences, sources)
    assert best["experience_id"] == "exp_a"
