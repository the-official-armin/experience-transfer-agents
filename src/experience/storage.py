"""Append-only JSONL storage for Experience records (schema.py), mirroring
src/experiments/logging.py's crash-safe incremental-append pattern.
"""

import json
from pathlib import Path


def start_fresh(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def append_experience(experience, path):
    record = experience.to_dict() if hasattr(experience, "to_dict") else experience
    with Path(path).open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def load_experiences(path):
    experiences = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                experiences.append(json.loads(line))
    return experiences
