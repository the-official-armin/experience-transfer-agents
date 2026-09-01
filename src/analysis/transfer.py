"""Transfer gain computation (CLAUDE.md Core Definitions):
Delta(E,T) = P(T|E) - P(T|None), computed per (experience, task) pair --
success-based, single-trial per pair (so Delta in {-1, 0, +1} today; we
don't have repeated Monte Carlo trials per pair), never pooled across a
task family or memory bank.

Delta_eff(E,T): a second outcome variable defined only on execution-side
effort the AGENT ITSELF controls -- completion_tokens, steps, tool_call_count.
retry_count is deliberately excluded from this aggregate (though still
logged, separately, per pair): retries reflect API infra flakiness
(rate-limiting), not the model needing more attempts to solve the task, and
mixing it into Delta_eff would quietly drag the aggregate toward network
noise once averaged across hundreds of pairs rather than 24. Never defined
on raw/input tokens either, which trivially rise from injecting experience
regardless of whether it helps. Computed only on pairs where baseline AND
the experience condition both succeeded, since efficiency is only cleanly
interpretable when the outcome is held constant -- None otherwise.
retry_count_delta, by contrast, is computed unconditionally (not gated on
matched success) since its purpose is endpoint-health monitoring, not a
performance signal.

Noise floor (measured Tuesday, Step 4): 8% single-trial flip rate. Applied
here per CLAUDE.md's "treat single-trial Delta magnitudes below ~8% as
ambiguous": since Delta only takes magnitude 0 or 1 (single-trial, binary),
this means exactly Delta == 0 gets flagged `ambiguous` (an actual flip,
magnitude 1 = 100%, is far above the floor and is NOT flagged). novel_tool
pairs get an additional standing `noise_caution` flag regardless of Delta,
since that's specifically where the noise-floor measurement concentrated
(n=1 task then; treat as a hypothesis, not a settled category property).
"""

NOISE_FLOOR = 0.08

EFFORT_FIELDS = ("completion_tokens", "steps", "tool_call_count")


def compute_delta(baseline_success, condition_success):
    return int(condition_success) - int(baseline_success)


def compute_delta_eff(baseline_record, condition_record):
    """None if either run failed -- efficiency isn't cleanly interpretable
    unless the outcome (success) is held constant across both conditions."""
    if not (baseline_record["success"] and condition_record["success"]):
        return None

    return {
        field: baseline_record[field] - condition_record[field]
        for field in EFFORT_FIELDS
    }


def compute_retry_delta(baseline_record, condition_record):
    """Logged for endpoint-health monitoring, not as an effort metric --
    computed regardless of success, unlike compute_delta_eff."""
    return baseline_record["retry_count"] - condition_record["retry_count"]


def build_pair(baseline_record, condition_record):
    """One scored (E,T) pair. condition_record is a Result record from an
    oracle/retrieved run; baseline_record is the matching Step 2 baseline
    Result record for the same task_id."""
    delta = compute_delta(baseline_record["success"], condition_record["success"])
    shift_category = condition_record["shift_category"]

    return {
        "task_id": condition_record["task_id"],
        "condition": condition_record["condition"],
        "experience_id": condition_record["experience_id_or_null"],
        "shift_category": shift_category,
        "hard_easy": "hard" if not baseline_record["success"] else "easy",
        "baseline_success": baseline_record["success"],
        "condition_success": condition_record["success"],
        "delta": delta,
        "delta_eff": compute_delta_eff(baseline_record, condition_record),
        "retry_count_delta": compute_retry_delta(baseline_record, condition_record),
        "ambiguous": delta == 0,
        "noise_caution": shift_category == "novel_tool",
    }


def build_pairs(baseline_by_task, condition_records):
    """condition_records: Result records from oracle/retrieved runs.
    baseline_by_task: {task_id: Result record} from the Step 2 baseline."""
    return [build_pair(baseline_by_task[r["task_id"]], r) for r in condition_records]
