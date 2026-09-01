"""Unit tests for src/analysis/transfer.py."""

from src.analysis.transfer import build_pair, compute_delta, compute_delta_eff, compute_retry_delta


def result(success, completion_tokens=100, steps=1, tool_call_count=1, retry_count=0,
           task_id="t1", condition="baseline", experience_id=None, shift_category="familiar"):
    return {
        "success": success, "completion_tokens": completion_tokens, "steps": steps,
        "tool_call_count": tool_call_count, "retry_count": retry_count,
        "task_id": task_id, "condition": condition, "experience_id_or_null": experience_id,
        "shift_category": shift_category,
    }


def test_compute_delta_positive_when_flips_to_success():
    assert compute_delta(baseline_success=False, condition_success=True) == 1


def test_compute_delta_negative_when_flips_to_failure():
    assert compute_delta(baseline_success=True, condition_success=False) == -1


def test_compute_delta_zero_when_unchanged():
    assert compute_delta(baseline_success=True, condition_success=True) == 0
    assert compute_delta(baseline_success=False, condition_success=False) == 0


def test_delta_eff_none_when_baseline_failed():
    baseline = result(success=False)
    condition = result(success=True)
    assert compute_delta_eff(baseline, condition) is None


def test_delta_eff_none_when_condition_failed():
    baseline = result(success=True)
    condition = result(success=False)
    assert compute_delta_eff(baseline, condition) is None


def test_delta_eff_computed_when_both_succeed():
    baseline = result(success=True, completion_tokens=200, steps=1, tool_call_count=2, retry_count=1)
    condition = result(success=True, completion_tokens=150, steps=1, tool_call_count=1, retry_count=0)
    delta_eff = compute_delta_eff(baseline, condition)
    assert delta_eff == {"completion_tokens": 50, "steps": 0, "tool_call_count": 1}
    assert "retry_count" not in delta_eff


def test_retry_delta_computed_even_when_a_run_failed():
    baseline = result(success=False, retry_count=3)
    condition = result(success=True, retry_count=1)
    assert compute_retry_delta(baseline, condition) == 2


def test_build_pair_includes_retry_count_delta_separately_from_delta_eff():
    baseline = result(success=True, retry_count=2)
    condition = result(success=True, retry_count=0)
    pair = build_pair(baseline, condition)
    assert pair["retry_count_delta"] == 2
    assert "retry_count" not in pair["delta_eff"]


def test_build_pair_ambiguous_flag_on_zero_delta():
    baseline = result(success=True, task_id="t1")
    condition = result(success=True, task_id="t1", condition="oracle", experience_id="exp1")
    pair = build_pair(baseline, condition)
    assert pair["delta"] == 0
    assert pair["ambiguous"] is True


def test_build_pair_not_ambiguous_on_real_flip():
    baseline = result(success=False, task_id="t1")
    condition = result(success=True, task_id="t1", condition="oracle", experience_id="exp1")
    pair = build_pair(baseline, condition)
    assert pair["delta"] == 1
    assert pair["ambiguous"] is False


def test_build_pair_hard_easy_label_from_baseline():
    hard_pair = build_pair(result(success=False), result(success=True))
    easy_pair = build_pair(result(success=True), result(success=True))
    assert hard_pair["hard_easy"] == "hard"
    assert easy_pair["hard_easy"] == "easy"


def test_build_pair_noise_caution_only_on_novel_tool():
    novel_tool_pair = build_pair(
        result(success=True, shift_category="novel_tool"),
        result(success=True, shift_category="novel_tool"),
    )
    familiar_pair = build_pair(
        result(success=True, shift_category="familiar"),
        result(success=True, shift_category="familiar"),
    )
    assert novel_tool_pair["noise_caution"] is True
    assert familiar_pair["noise_caution"] is False
