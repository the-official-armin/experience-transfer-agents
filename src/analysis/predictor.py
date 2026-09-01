"""Hero-result predictor + both baselines (CLAUDE.md Research Question:
evaluated against (a) majority-class/mean-Delta and (b) a re-implementation
of Break It Down's specificity x abstractness utility score).

Break-It-Down baseline caveat (checked before trusting it, per instruction):
utility_score = specificity * abstraction takes only 5 unique values across
the 251-row dataset, and 58% of rows share the identical value (5) --
specificity and abstraction are two of the four properties already found to
cluster (docs/experiment_log.md). This is NOT fully degenerate (nonzero
variance, 5 distinct buckets), but it IS severely discretized: for the
majority of pairs, this baseline has no power to distinguish between them
at all. Beating it here is a lower bar than in Break It Down's own setting,
where specificity/abstractness presumably vary more -- state this plainly
in the writeup, don't present a win over it as equally meaningful.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, median_absolute_error

from src.analysis.features import CATEGORICAL_FEATURES, CLUSTERED_PROPERTY_FEATURES, PRIMARY_PROPERTY_FEATURES


def utility_score(df):
    """Break It Down's specificity x abstractness utility score, literal
    product of the two properties as named in CLAUDE.md."""
    return df["specificity"] * df["abstraction"]


def check_utility_score_variance(df):
    """Report plainly whether the utility score is usably varied or
    effectively constant across the dataset -- call before trusting the
    baseline built from it."""
    scores = utility_score(df)
    value_counts = scores.value_counts()
    return {
        "n": len(scores),
        "unique_values": scores.nunique(),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "modal_value": value_counts.index[0],
        "modal_fraction": float(value_counts.iloc[0] / len(scores)),
    }


class MajorityClassBaseline:
    """Baseline 1 (classification target, e.g. Delta(E,T) in {-1,0,1}):
    always predicts the most frequent label seen in training."""

    def fit(self, y_train):
        values, counts = np.unique(y_train, return_counts=True)
        self.majority_value_ = values[np.argmax(counts)]
        return self

    def predict(self, n):
        return np.full(n, self.majority_value_)


class MeanDeltaBaseline:
    """Baseline 1 (regression target, e.g. Delta_eff): always predicts the
    training mean."""

    def fit(self, y_train):
        self.mean_value_ = float(np.mean(y_train))
        return self

    def predict(self, n):
        return np.full(n, self.mean_value_)


class BreakItDownBaseline:
    """Baseline 2: single-feature linear regression from utility_score to
    the target, fit on training data only. Works for both classification
    (Delta, treated as a numeric target then rounded/clipped to {-1,0,1})
    and regression (Delta_eff) targets -- CLAUDE.md specifies this as ONE
    baseline, not two separate re-implementations."""

    def __init__(self, classification=False):
        self.classification = classification
        self.model = LinearRegression()

    def fit(self, train_df, y_train):
        x = utility_score(train_df).to_numpy().reshape(-1, 1)
        self.model.fit(x, y_train)
        return self

    def predict(self, df):
        x = utility_score(df).to_numpy().reshape(-1, 1)
        predictions = self.model.predict(x)
        if self.classification:
            predictions = np.clip(np.round(predictions), -1, 1).astype(int)
        return predictions


def encode_features(train_df, other_df, include_clustered=False):
    """One-hot encode categorical features (fit dummy columns on train,
    apply the same columns to other_df, filling missing categories with 0
    so train/held-out feature matrices always align in shape)."""
    numeric_features = [f for f in PRIMARY_PROPERTY_FEATURES if f not in CATEGORICAL_FEATURES]
    if include_clustered:
        numeric_features = numeric_features + list(CLUSTERED_PROPERTY_FEATURES)

    def prep(df):
        base = df[list(CATEGORICAL_FEATURES) + numeric_features].copy()
        base["success_failure"] = base["success_failure"].astype(int)
        return pd.get_dummies(base, columns=list(CATEGORICAL_FEATURES))

    train_encoded = prep(train_df)
    other_encoded = prep(other_df).reindex(columns=train_encoded.columns, fill_value=0)
    return train_encoded, other_encoded


def train_delta_eff_predictor(train_df, held_out_df, include_clustered=False, seed=42):
    """Primary target: Delta_eff (completion_tokens), matched-success pairs
    only. Well-powered relative to Delta -- ~169 of 251 rows usable."""
    train_matched = train_df[train_df["delta_eff_completion_tokens"].notna()]
    held_out_matched = held_out_df[held_out_df["delta_eff_completion_tokens"].notna()]

    x_train, x_held_out = encode_features(train_matched, held_out_matched, include_clustered)
    y_train = train_matched["delta_eff_completion_tokens"].to_numpy()
    y_held_out = held_out_matched["delta_eff_completion_tokens"].to_numpy()

    model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=seed)
    model.fit(x_train, y_train)
    predictions = model.predict(x_held_out)

    mean_baseline = MeanDeltaBaseline().fit(y_train)
    mean_predictions = mean_baseline.predict(len(y_held_out))

    bid_baseline = BreakItDownBaseline(classification=False).fit(train_matched, y_train)
    bid_predictions = bid_baseline.predict(held_out_matched)

    def metrics(y_true, y_pred):
        return {"mae": float(mean_absolute_error(y_true, y_pred)),
                "median_ae": float(median_absolute_error(y_true, y_pred))}

    return {
        "n_train": len(y_train), "n_held_out": len(y_held_out),
        "predictor": metrics(y_held_out, predictions),
        "mean_baseline": metrics(y_held_out, mean_predictions),
        "bid_baseline": metrics(y_held_out, bid_predictions),
        "feature_importances": dict(zip(x_train.columns.tolist(), model.feature_importances_.tolist())),
    }


def train_delta_predictor(train_df, held_out_df, include_clustered=False, seed=42):
    """Secondary target: Delta(E,T) success-flip, all rows (small-n,
    severely imbalanced: ~85% of pairs are Delta=0 -- report accuracy AND
    macro-F1, since accuracy alone rewards a trivial always-predict-0
    baseline for reasons that have nothing to do with being a useful
    predictor."""
    x_train, x_held_out = encode_features(train_df, held_out_df, include_clustered)
    y_train = train_df["delta"].to_numpy()
    y_held_out = held_out_df["delta"].to_numpy()

    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=seed,
                                    class_weight="balanced")
    model.fit(x_train, y_train)
    predictions = model.predict(x_held_out)

    majority_baseline = MajorityClassBaseline().fit(y_train)
    majority_predictions = majority_baseline.predict(len(y_held_out))

    bid_baseline = BreakItDownBaseline(classification=True).fit(train_df, y_train)
    bid_predictions = bid_baseline.predict(held_out_df)

    def metrics(y_true, y_pred):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=[-1, 0, 1], average="macro", zero_division=0)),
        }

    held_out_support = dict(zip(*np.unique(y_held_out, return_counts=True)))

    return {
        "n_train": len(y_train), "n_held_out": len(y_held_out),
        "held_out_class_support": {int(k): int(v) for k, v in held_out_support.items()},
        "predictor": metrics(y_held_out, predictions),
        "majority_baseline": metrics(y_held_out, majority_predictions),
        "bid_baseline": metrics(y_held_out, bid_predictions),
        "feature_importances": dict(zip(x_train.columns.tolist(), model.feature_importances_.tolist())),
    }


def train_delta_eff_sign_predictor(train_df, held_out_df, include_clustered=False, seed=42):
    """Day 3, Step 1: Delta_eff recast as sign classification (effort down
    vs. up vs. unchanged) rather than magnitude regression. Motivation
    already in hand from the Step 4 pilot finding: oracle's mean and median
    completion-token deltas disagreed in SIGN at n=4, direct evidence the
    underlying distribution is fat-tailed/outlier-dominated -- a bad fit for
    magnitude regression (MAE dominated by a few extreme values), possibly a
    fine fit for direction, which large outliers can't distort the same way.
    Same matched-success-only pairs as the regression version."""
    train_matched = train_df[train_df["delta_eff_completion_tokens"].notna()].copy()
    held_out_matched = held_out_df[held_out_df["delta_eff_completion_tokens"].notna()].copy()

    train_matched["delta_eff_sign"] = np.sign(train_matched["delta_eff_completion_tokens"]).astype(int)
    held_out_matched["delta_eff_sign"] = np.sign(held_out_matched["delta_eff_completion_tokens"]).astype(int)

    x_train, x_held_out = encode_features(train_matched, held_out_matched, include_clustered)
    y_train = train_matched["delta_eff_sign"].to_numpy()
    y_held_out = held_out_matched["delta_eff_sign"].to_numpy()

    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=seed,
                                    class_weight="balanced")
    model.fit(x_train, y_train)
    predictions = model.predict(x_held_out)

    majority_baseline = MajorityClassBaseline().fit(y_train)
    majority_predictions = majority_baseline.predict(len(y_held_out))

    # mean-sign baseline: sign of the training mean, applied uniformly (a
    # continuous-target-style baseline adapted to the classification framing)
    mean_sign = int(np.sign(y_train.mean())) if y_train.mean() != 0 else 0
    mean_sign_predictions = np.full(len(y_held_out), mean_sign)

    def metrics(y_true, y_pred):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=[-1, 0, 1], average="macro", zero_division=0)),
        }

    held_out_support = dict(zip(*np.unique(y_held_out, return_counts=True)))

    return {
        "n_train": len(y_train), "n_held_out": len(y_held_out),
        "held_out_class_support": {int(k): int(v) for k, v in held_out_support.items()},
        "predictor": metrics(y_held_out, predictions),
        "majority_baseline": metrics(y_held_out, majority_predictions),
        "mean_sign_baseline": metrics(y_held_out, mean_sign_predictions),
        "feature_importances": dict(zip(x_train.columns.tolist(), model.feature_importances_.tolist())),
    }
