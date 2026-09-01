"""Unit tests for src/analysis/features.py."""

import json

from src.analysis.features import (
    CLUSTERED_PROPERTY_FEATURES,
    PRIMARY_PROPERTY_FEATURES,
    build_dataset,
)


def write_jsonl(path, records):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def make_experience(experience_id):
    return {
        "experience_id": experience_id,
        "properties": {
            "length": 100, "num_steps": 1, "success_failure": True,
            "task_structure": "single_call", "tool_dependence": 1.0,
            "abstraction": 5, "specificity": 5, "composability": 5, "information_context": 1,
        },
    }


def test_build_dataset_joins_pairs_with_experience_properties(tmp_path):
    bank_path = tmp_path / "bank.jsonl"
    write_jsonl(bank_path, [make_experience("exp1")])

    pairs_path = tmp_path / "pairs.jsonl"
    write_jsonl(pairs_path, [{
        "task_id": "t1", "condition": "oracle", "experience_id": "exp1",
        "shift_category": "familiar", "hard_easy": "hard", "delta": 1, "delta_eff": None,
    }])

    df = build_dataset([str(pairs_path)], str(bank_path))
    assert len(df) == 1
    for field in PRIMARY_PROPERTY_FEATURES:
        assert field in df.columns
    for field in CLUSTERED_PROPERTY_FEATURES:
        assert field in df.columns
    assert df.iloc[0]["length"] == 100
    assert df.iloc[0]["representation"] == "raw"  # defaulted when absent


def test_build_dataset_drops_rows_without_experience(tmp_path):
    bank_path = tmp_path / "bank.jsonl"
    write_jsonl(bank_path, [make_experience("exp1")])

    pairs_path = tmp_path / "pairs.jsonl"
    write_jsonl(pairs_path, [
        {"task_id": "t1", "condition": "baseline", "experience_id": None,
         "shift_category": "familiar", "hard_easy": "hard", "delta": 0, "delta_eff": None},
        {"task_id": "t2", "condition": "oracle", "experience_id": "exp1",
         "shift_category": "familiar", "hard_easy": "hard", "delta": 1, "delta_eff": None},
    ])

    df = build_dataset([str(pairs_path)], str(bank_path))
    assert len(df) == 1
    assert df.iloc[0]["task_id"] == "t2"


def test_build_dataset_flattens_delta_eff_completion_tokens(tmp_path):
    bank_path = tmp_path / "bank.jsonl"
    write_jsonl(bank_path, [make_experience("exp1")])

    pairs_path = tmp_path / "pairs.jsonl"
    write_jsonl(pairs_path, [{
        "task_id": "t1", "condition": "oracle", "experience_id": "exp1",
        "shift_category": "familiar", "hard_easy": "easy", "delta": 0,
        "delta_eff": {"completion_tokens": 42, "steps": 0, "tool_call_count": 1},
    }])

    df = build_dataset([str(pairs_path)], str(bank_path))
    assert df.iloc[0]["delta_eff_completion_tokens"] == 42


def test_build_dataset_preserves_representation_field(tmp_path):
    bank_path = tmp_path / "bank.jsonl"
    write_jsonl(bank_path, [make_experience("exp1")])

    pairs_path = tmp_path / "pairs.jsonl"
    write_jsonl(pairs_path, [{
        "task_id": "t1", "condition": "oracle", "experience_id_or_null": "exp1",
        "shift_category": "familiar", "hard_easy": "hard", "representation": "procedure",
        "delta": 1, "delta_eff": None,
    }])

    df = build_dataset([str(pairs_path)], str(bank_path))
    assert df.iloc[0]["representation"] == "procedure"
