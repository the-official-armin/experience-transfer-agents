"""Experience selection for the `oracle` and `retrieved` conditions
(CLAUDE.md Experimental Conditions). Both select from the experience bank
using the `raw` representation only today; reflection/procedure crossing is
Thursday's work.

Oracle rule (confirmed before implementation): score each candidate
experience E (traced to its source task S) against the held-out task T on a
priority-ordered tuple, maximized in order --
  1. exact tool-name overlap between T's and S's tools_available
  2. API-family overlap (namespace before the first "." in tool names, e.g.
     "video_games.store_price" / "video_games.on_sale" share "video_games")
  3. shift_subtype match between T and S (same underlying task shape)
  4. whether S's own attempt succeeded (prefer a trustworthy exemplar)
  5. deterministic tie-break: lowest experience_id alphabetically
This is fully programmatic -- "oracle" here means privileged structural
information the automatic retriever doesn't use, not literal manual
clicking. A task sharing nothing with any experience still gets a
deterministic pick via tiers 3-5, which is the correct (not broken)
behavior for measuring negative transfer on truly unrelated injection.

Retrieved uses src/experience/retrieval.py's fixed cosine-similarity method
-- fixed before evaluation begins, per CLAUDE.md's hard constraint, not
tuned after seeing results.
"""

from src.experience.retrieval import retrieve_best_experience


def _tool_names(tools_available):
    return {t["name"] for t in tools_available}


def _family(name):
    return name.split(".", 1)[0]


def _families(tool_names):
    return {_family(n) for n in tool_names}


def select_oracle_experience(task, experiences, source_tasks_by_id):
    task_tools = _tool_names(task["tools_available"])
    task_families = _families(task_tools)

    def sort_key(experience):
        source_task = source_tasks_by_id[experience["source_task_id"]]
        source_tools = _tool_names(source_task["tools_available"])
        source_families = _families(source_tools)

        tool_overlap = len(task_tools & source_tools)
        family_overlap = len(task_families & source_families)
        subtype_match = int(source_task.get("shift_subtype") == task.get("shift_subtype"))
        source_success = int(bool(experience["properties"]["success_failure"]))

        # negate the "maximize" criteria so min() picks the best match;
        # experience_id sorts ascending naturally for the tie-break
        return (-tool_overlap, -family_overlap, -subtype_match, -source_success, experience["experience_id"])

    return min(experiences, key=sort_key)


def select_retrieved_experience(task, experiences):
    experience, _score = retrieve_best_experience(task, experiences, representation="raw")
    return experience
