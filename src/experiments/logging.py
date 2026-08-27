"""Append-only writer for raw per-(task,condition) Result records.

data/results/ is machine-written harness output only (CLAUDE.md) -- never
hand-edited, and never the place exploratory analysis writes to. Writes are
appended one record at a time so a mid-run crash (a stalled API call, a
timeout) keeps whatever already completed instead of losing the whole batch.
"""

import json
from datetime import datetime, timezone
from pathlib import Path


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def start_fresh(path):
    """Truncate path (or create its parent dir) so a re-run doesn't append to stale data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def append_result(record, path):
    record = dict(record, timestamp=record.get("timestamp") or timestamp())
    with Path(path).open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record
