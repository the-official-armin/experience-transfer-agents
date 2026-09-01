"""Unit tests for src/analysis/predictor.py's baselines."""

import numpy as np
import pandas as pd

from src.analysis.predictor import (
    BreakItDownBaseline,
    MajorityClassBaseline,
    MeanDeltaBaseline,
    check_utility_score_variance,
    utility_score,
)


def test_utility_score_is_product_of_specificity_and_abstraction():
    df = pd.DataFrame({"specificity": [5, 3], "abstraction": [2, 4]})
    scores = utility_score(df)
    assert scores.tolist() == [10, 12]


def test_check_utility_score_variance_detects_degenerate_case():
    df = pd.DataFrame({"specificity": [5] * 10, "abstraction": [1] * 10})
    stats = check_utility_score_variance(df)
    assert stats["unique_values"] == 1
    assert stats["std"] == 0.0
    assert stats["modal_fraction"] == 1.0


def test_check_utility_score_variance_detects_varied_case():
    df = pd.DataFrame({"specificity": [1, 2, 3, 4, 5], "abstraction": [5, 4, 3, 2, 1]})
    stats = check_utility_score_variance(df)
    assert stats["unique_values"] > 1


def test_majority_class_baseline_predicts_most_common_label():
    baseline = MajorityClassBaseline().fit(np.array([0, 0, 1, 0, -1]))
    predictions = baseline.predict(5)
    assert (predictions == 0).all()


def test_mean_delta_baseline_predicts_training_mean():
    baseline = MeanDeltaBaseline().fit(np.array([10, 20, 30]))
    predictions = baseline.predict(3)
    assert np.allclose(predictions, 20.0)


def test_break_it_down_baseline_fits_and_predicts():
    train_df = pd.DataFrame({"specificity": [1, 3, 5], "abstraction": [1, 3, 5]})
    y_train = np.array([2.0, 18.0, 50.0])  # roughly linear in utility_score
    baseline = BreakItDownBaseline().fit(train_df, y_train)

    test_df = pd.DataFrame({"specificity": [4], "abstraction": [4]})
    prediction = baseline.predict(test_df)
    assert 15 < prediction[0] < 40  # between the fitted extremes, not exact


def test_break_it_down_baseline_classification_clips_to_valid_range():
    train_df = pd.DataFrame({"specificity": [1, 5], "abstraction": [1, 5]})
    y_train = np.array([-1, 1])
    baseline = BreakItDownBaseline(classification=True).fit(train_df, y_train)

    test_df = pd.DataFrame({"specificity": [5], "abstraction": [5]})
    prediction = baseline.predict(test_df)
    assert prediction[0] in (-1, 0, 1)
