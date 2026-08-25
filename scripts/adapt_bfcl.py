"""Adapt BFCL v4 (Berkeley Function-Calling Leaderboard, Apache 2.0, ShishirPatil/gorilla)
task files into our Task schema (CLAUDE.md), tagging each task with a shift_category.

Usage: python scripts/adapt_bfcl.py
"""

import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environment.task_families import BFCL_CATEGORY_TO_SUBTYPE, SUBTYPE_TO_CATEGORY
from src.environment.tasks import Task, save_tasks

BASE_URL = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)
RAW_CACHE_DIR = Path("data/bfcl_raw")
OUTPUT_PATH = Path("tasks/task_definitions/bfcl_tasks.jsonl")

CATEGORIES = list(BFCL_CATEGORY_TO_SUBTYPE.keys())


def _fetch(relative_path):
    cache_path = RAW_CACHE_DIR / relative_path
    if cache_path.exists():
        return cache_path.read_text()

    response = requests.get(f"{BASE_URL}/{relative_path}", timeout=30)
    response.raise_for_status()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(response.text)
    return response.text


def _load_jsonl_text(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _extract_goal(question_turns):
    # BFCL nests a single-turn conversation as [[turn, turn, ...]].
    turns = question_turns[0]
    user_parts = [t["content"] for t in turns if t["role"] == "user"]
    system_parts = [t["content"] for t in turns if t["role"] == "system"]
    goal = " ".join(user_parts).strip()
    if system_parts:
        goal = f"[context: {' '.join(system_parts)}] {goal}"
    return goal


def adapt_category(category):
    questions = _load_jsonl_text(_fetch(f"BFCL_v4_{category}.json"))

    answers_by_id = {}
    try:
        answers = _load_jsonl_text(_fetch(f"possible_answer/BFCL_v4_{category}.json"))
        answers_by_id = {a["id"]: a["ground_truth"] for a in answers}
    except requests.HTTPError:
        pass  # irrelevance has no possible_answer file: no call is the gold answer

    subtype = BFCL_CATEGORY_TO_SUBTYPE[category]
    shift_category = SUBTYPE_TO_CATEGORY[subtype]

    tasks = []
    for q in questions:
        tasks.append(Task(
            task_id=f"bfcl_{q['id']}",
            tools_available=q["function"],
            goal=_extract_goal(q["question"]),
            gold_call=answers_by_id.get(q["id"]),
            shift_category=shift_category,
            shift_subtype=subtype,
            source="bfcl",
            source_id=q["id"],
        ))
    return tasks


def main():
    all_tasks = []
    for category in CATEGORIES:
        category_tasks = adapt_category(category)
        subtype = BFCL_CATEGORY_TO_SUBTYPE[category]
        print(f"{category}: {len(category_tasks)} tasks "
              f"-> shift_subtype={subtype}, shift_category={SUBTYPE_TO_CATEGORY[subtype]}")
        all_tasks.extend(category_tasks)

    save_tasks(all_tasks, OUTPUT_PATH)
    print(f"\nWrote {len(all_tasks)} tasks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
