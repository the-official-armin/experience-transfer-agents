"""Experience record schema (CLAUDE.md Data Schema Contracts):
{experience_id, source_task_id, trajectory, representations: {raw, reflection,
procedure}, properties: {...9 fields...}}
"""

from dataclasses import asdict, dataclass, field

PROPERTY_NAMES = (
    "abstraction", "specificity", "task_structure", "tool_dependence",
    "length", "num_steps", "success_failure", "composability", "information_context",
)

REPRESENTATION_NAMES = ("raw", "reflection", "procedure")


@dataclass
class Experience:
    experience_id: str
    source_task_id: str
    trajectory: list
    representations: dict = field(default_factory=dict)  # raw / reflection / procedure
    properties: dict = field(default_factory=dict)        # the 9 PROPERTY_NAMES

    def to_dict(self):
        return asdict(self)
