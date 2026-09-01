"""Day 2, Step 5: train predictor v1, both targets, against both baselines,
on the frozen predictor-specific held-out split.

Primary target: Delta_eff (completion_tokens) -- well-powered (~169/251
matched-success pairs), reported as the primary quantitative claim.
Secondary target: Delta(E,T) -- small-n (37/251 non-zero), severely
imbalanced (~85% zero), reported with explicit hedging -- NOT a robust
held-out claim the way Delta_eff's is.

Results for the two targets are reported separately throughout, never
averaged into one number.

Usage: python scripts/train_predictor.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.features import build_dataset
from src.analysis.predictor import (
    check_utility_score_variance,
    train_delta_eff_predictor,
    train_delta_predictor,
)

PAIR_PATHS = ["results/processed/transfer_pairs.jsonl", "data/results/representation_crossing.jsonl"]
BANK_PATH = "data/experiences/experiences.jsonl"
SPLIT_PATH = "results/processed/predictor_split.json"


def print_delta_eff_report(result, label):
    print(f"--- Delta_eff (completion_tokens) -- {label} ---")
    print(f"n_train={result['n_train']}, n_held_out={result['n_held_out']}")
    for name in ("predictor", "mean_baseline", "bid_baseline"):
        m = result[name]
        print(f"  {name:15s}: MAE={m['mae']:.2f}, median_AE={m['median_ae']:.2f}")
    beats_mean = result["predictor"]["mae"] < result["mean_baseline"]["mae"]
    beats_bid = result["predictor"]["mae"] < result["bid_baseline"]["mae"]
    print(f"  beats mean-Delta baseline (MAE): {beats_mean}")
    print(f"  beats Break-It-Down baseline (MAE): {beats_bid}")
    print()


def print_delta_report(result, label):
    print(f"--- Delta(E,T) success-flip -- {label} (SMALL-N, HEDGED) ---")
    print(f"n_train={result['n_train']}, n_held_out={result['n_held_out']}")
    print(f"held_out class support: {result['held_out_class_support']}")
    for name in ("predictor", "majority_baseline", "bid_baseline"):
        m = result[name]
        print(f"  {name:18s}: accuracy={m['accuracy']:.3f}, macro_F1={m['macro_f1']:.3f}")
    beats_majority_acc = result["predictor"]["accuracy"] > result["majority_baseline"]["accuracy"]
    beats_majority_f1 = result["predictor"]["macro_f1"] > result["majority_baseline"]["macro_f1"]
    beats_bid_f1 = result["predictor"]["macro_f1"] > result["bid_baseline"]["macro_f1"]
    print(f"  beats majority-class on accuracy: {beats_majority_acc} "
          f"(caution: majority-class gets ~85% accuracy for free from class imbalance alone)")
    print(f"  beats majority-class on macro-F1: {beats_majority_f1}")
    print(f"  beats Break-It-Down baseline on macro-F1: {beats_bid_f1}")
    print()


def main():
    df = build_dataset(PAIR_PATHS, BANK_PATH)
    split = json.loads(Path(SPLIT_PATH).read_text())
    train_df = df[df["pair_id"].isin(split["train"])]
    held_out_df = df[df["pair_id"].isin(split["held_out"])]

    variance = check_utility_score_variance(df)
    print(f"Break-It-Down utility_score check: {variance['unique_values']} unique values, "
          f"{100*variance['modal_fraction']:.0f}% share the modal value -- "
          f"beating this baseline is a lower bar than in a domain with more natural variance.\n")

    print("=" * 60)
    print("PRIMARY TARGET: Delta_eff(E,T)")
    print("=" * 60)
    primary_eff = train_delta_eff_predictor(train_df, held_out_df, include_clustered=False)
    print_delta_eff_report(primary_eff, "primary feature set (5 reliable properties)")

    ablation_eff = train_delta_eff_predictor(train_df, held_out_df, include_clustered=True)
    print_delta_eff_report(ablation_eff, "ABLATION (+ 4 clustered properties)")

    clustered_importance = {
        k: v for k, v in ablation_eff["feature_importances"].items()
        if k in ("abstraction", "specificity", "composability", "information_context")
    }
    print(f"Clustered-property feature importances (ablation model): {clustered_importance}")
    print(f"(sum = {sum(clustered_importance.values()):.4f} out of 1.0 total across all features)\n")

    print("=" * 60)
    print("SECONDARY TARGET: Delta(E,T) success-flip")
    print("=" * 60)
    primary_delta = train_delta_predictor(train_df, held_out_df, include_clustered=False)
    print_delta_report(primary_delta, "primary feature set")


if __name__ == "__main__":
    main()
