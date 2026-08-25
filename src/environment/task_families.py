"""Task shift taxonomy.

CLAUDE.md defines `shift_category` as an ordered novelty scale:
familiar -> novel_composition -> novel_tool -> distractor_tool -> ood

The kickoff brief separately named a finer-grained *mechanism* taxonomy
(same-API-diff-params, swap-similar-API, distractor-tools, compose-multiple-tools,
novel-API) for the adapter. The two lists don't line up 1:1, so we store the
kickoff list as `shift_subtype` and roll each subtype up into CLAUDE.md's
canonical `shift_category` via SUBTYPE_TO_CATEGORY. `ood` is not populated by
the BFCL adapter; nothing in the categories we use maps to it naturally.
"""

SHIFT_CATEGORIES = ["familiar", "novel_composition", "novel_tool", "distractor_tool", "ood"]

SHIFT_SUBTYPES = [
    "same_api_diff_params",
    "swap_similar_api",
    "distractor_tools",
    "compose_multiple_tools",
    "novel_api",
]

SUBTYPE_TO_CATEGORY = {
    "same_api_diff_params": "familiar",
    "swap_similar_api": "novel_tool",
    "distractor_tools": "distractor_tool",
    "compose_multiple_tools": "novel_composition",
    "novel_api": "novel_tool",
}

# BFCL v4 category name -> shift_subtype
BFCL_CATEGORY_TO_SUBTYPE = {
    "simple_python": "same_api_diff_params",
    "multiple": "swap_similar_api",
    "parallel": "compose_multiple_tools",
    "parallel_multiple": "compose_multiple_tools",
    "irrelevance": "distractor_tools",
}
