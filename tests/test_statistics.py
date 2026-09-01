"""Unit tests for src/analysis/statistics.py."""

from src.analysis.statistics import category_vs_rest_fisher


def test_identical_rates_give_high_p_value():
    # same rate in both groups -> no significant difference
    _, p = category_vs_rest_fisher(5, 50, 5, 50)
    assert p > 0.5


def test_extreme_difference_gives_low_p_value():
    # category all-negative vs. rest all-positive -> highly significant
    _, p = category_vs_rest_fisher(10, 10, 0, 10)
    assert p < 0.01


def test_odds_ratio_above_one_when_category_rate_higher():
    odds_ratio, _ = category_vs_rest_fisher(4, 47, 2, 156)
    assert odds_ratio > 1
