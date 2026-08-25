"""Task record schema (CLAUDE.md data schema contract) and JSONL load/save."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Task:
    task_id: str
    tools_available: list
    goal: str
    gold_call: list | None
    shift_category: str
    split: str | None = None
    shift_subtype: str | None = None
    source: str = "bfcl"
    source_id: str | None = None

    def to_dict(self):
        return asdict(self)


def save_tasks(tasks, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for task in tasks:
            record = task.to_dict() if isinstance(task, Task) else task
            f.write(json.dumps(record) + "\n")


def load_tasks(path):
    tasks = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks
