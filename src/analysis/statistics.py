"""Significance checks for transfer-pair findings.

Kept separate from confirmatory analysis output (transfer.py, build_transfer_
map.py) per CLAUDE.md's engineering practice of not letting exploratory
analysis silently become the reported result -- this module exists so a
finding either earns a "confirmed" label with a stated p-value, or gets
explicitly softened to "suggestive," never presented as settled by default.
"""

from scipy.stats import fisher_exact


def category_vs_rest_fisher(category_negative, category_total, rest_negative, rest_total):
    """Two-sided Fisher's exact test on a 2x2 contingency table: does one
    category's negative-flip rate differ from the pooled rate of the rest?
    Returns (odds_ratio, p_value)."""
    table = [
        [category_negative, category_total - category_negative],
        [rest_negative, rest_total - rest_negative],
    ]
    odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
    return odds_ratio, p_value
