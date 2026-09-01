"""Day 3, Step 5: final visualizations for the paper.

All figures built from the complete, current dataset (all 3 result sources
+ the distractor_tool oversample, predictor_split_v2) for consistency with
today's final numbers -- confirms/regenerates transfer-by-category (stale
after the oversample), adds calibration, property-correlation, and
oracle-vs-retrieved gap figures that didn't exist yet.

Usage: python scripts/build_final_visualizations.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.features import CLUSTERED_PROPERTY_FEATURES, build_dataset
from src.analysis.predictor import (
    train_delta_eff_predictor,
    train_delta_eff_sign_predictor,
    encode_features,
)
from sklearn.ensemble import RandomForestRegressor

PAIR_PATHS = [
    "results/processed/transfer_pairs.jsonl",
    "data/results/representation_crossing.jsonl",
    "data/results/distractor_tool_oversample.jsonl",
]
BANK_PATH = "data/experiences/experiences.jsonl"
SPLIT_PATH = "results/processed/predictor_split_v2.json"
FIGURES_DIR = Path("results/figures")

RELIABLE_NUMERIC = ("length", "tool_dependence", "success_failure")


def load_data():
    df = build_dataset(PAIR_PATHS, BANK_PATH)
    split = json.loads(Path(SPLIT_PATH).read_text())
    train_df = df[df["pair_id"].isin(split["train"])]
    held_out_df = df[df["pair_id"].isin(split["held_out"])]
    return df, train_df, held_out_df


def fig_transfer_by_category(df, path):
    categories = sorted(df["shift_category"].unique())
    conditions = ("oracle", "retrieved")

    fig, axes = plt.subplots(1, len(conditions), figsize=(11, 4.5), sharey=True)
    for ax, condition in zip(axes, conditions):
        cond_df = df[df["condition"] == condition]
        positive = [((cond_df["shift_category"] == c) & (cond_df["delta"] == 1)).sum() for c in categories]
        negative = [-((cond_df["shift_category"] == c) & (cond_df["delta"] == -1)).sum() for c in categories]

        x = range(len(categories))
        ax.bar(x, positive, color="#4C72B0", label="positive (+1)")
        ax.bar(x, negative, color="#C44E52", label="negative (-1)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(categories, rotation=30, ha="right")
        ax.set_title(condition)
        ax.set_ylabel("Delta(E,T) flip count")

    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Transfer gain by shift_category (n={len(df)} pairs, final enriched dataset)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_delta_eff_calibration(train_df, held_out_df, path):
    train_matched = train_df[train_df["delta_eff_completion_tokens"].notna()]
    held_out_matched = held_out_df[held_out_df["delta_eff_completion_tokens"].notna()]
    x_train, x_held_out = encode_features(train_matched, held_out_matched, include_clustered=False)
    y_train = train_matched["delta_eff_completion_tokens"].to_numpy()
    y_held_out = held_out_matched["delta_eff_completion_tokens"].to_numpy()

    model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_held_out)

    sign_result = train_delta_eff_sign_predictor(train_df, held_out_df, include_clustered=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    lim = max(abs(y_held_out).max(), abs(y_pred).max()) * 1.1
    ax.scatter(y_held_out, y_pred, alpha=0.6, color="#4C72B0")
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, label="perfect prediction")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("actual Delta_eff (completion tokens)")
    ax.set_ylabel("predicted Delta_eff (completion tokens)")
    ax.set_title(f"Magnitude regression (v2, n={len(y_held_out)})\nMAE={np.mean(np.abs(y_held_out-y_pred)):.1f} -- loses to baselines")
    ax.legend(fontsize=8)

    ax = axes[1]
    from sklearn.metrics import confusion_matrix
    train_matched2 = train_df[train_df["delta_eff_completion_tokens"].notna()].copy()
    held_out_matched2 = held_out_df[held_out_df["delta_eff_completion_tokens"].notna()].copy()
    y_true_sign = np.sign(held_out_matched2["delta_eff_completion_tokens"]).astype(int)
    x_train2, x_held_out2 = encode_features(train_matched2, held_out_matched2, include_clustered=True)
    y_train_sign = np.sign(train_matched2["delta_eff_completion_tokens"]).astype(int)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, class_weight="balanced")
    clf.fit(x_train2, y_train_sign)
    y_pred_sign = clf.predict(x_held_out2)
    labels = sorted(set(y_true_sign) | set(y_pred_sign))
    cm = confusion_matrix(y_true_sign, y_pred_sign, labels=labels)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted sign")
    ax.set_ylabel("actual sign")
    ax.set_title(f"Sign classification (ablation, n={len(y_true_sign)})\nmacro-F1={sign_result['predictor']['macro_f1']:.3f} -- beats baselines")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_property_correlation(df, path):
    numeric_df = df.copy()
    numeric_df["success_failure"] = numeric_df["success_failure"].astype(int)
    target = numeric_df["delta_eff_completion_tokens"]

    reliable_corr = {p: numeric_df[p].corr(target) for p in RELIABLE_NUMERIC}
    clustered_corr = {p: numeric_df[p].corr(target) for p in CLUSTERED_PROPERTY_FEATURES}

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(reliable_corr.keys()) + list(clustered_corr.keys())
    values = list(reliable_corr.values()) + list(clustered_corr.values())
    colors = ["#4C72B0"] * len(reliable_corr) + ["#C44E52"] * len(clustered_corr)

    y = range(len(labels))
    ax.barh(y, values, color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with Delta_eff (completion tokens)")
    ax.set_title("Property correlation with Delta_eff: reliable (blue) vs. clustered/ablation (red)")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4C72B0", label="primary/reliable"),
                        Patch(color="#C44E52", label="clustered (ablation only)")],
               loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_oracle_vs_retrieved_gap(df, path):
    """Scoped strictly to the frozen canonical 25-task set
    (day1_hard25_task_ids.json) and representation='raw' -- the full
    enriched df has since re-run some of these same 25 tasks via the
    representation-crossing raw condition, and given real measured
    non-determinism, some produced a DIFFERENT outcome than the original
    Phase 1 result. Without this filter, a naive (task_id, condition) dict
    silently keeps whichever row iterates last, which can quietly change
    Day 1's reported 8/1/0/16 breakdown. Deduplicate to the first
    occurrence (Phase 1's original result) for continuity with what's
    already been reported."""
    canonical_25 = set(json.loads(Path("data/results/day1_hard25_task_ids.json").read_text()))
    hard = df[(df["task_id"].isin(canonical_25)) & (df["representation"] == "raw")]

    by_task = {}
    for _, row in hard.iterrows():
        by_task.setdefault(row["task_id"], {}).setdefault(row["condition"], row["delta"])

    both, oracle_only, retrieved_only, neither = 0, 0, 0, 0
    for task_id, conds in by_task.items():
        if "oracle" not in conds or "retrieved" not in conds:
            continue
        o, r = conds["oracle"] == 1, conds["retrieved"] == 1
        if o and r:
            both += 1
        elif o and not r:
            oracle_only += 1
        elif r and not o:
            retrieved_only += 1
        else:
            neither += 1

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    labels = ["both\npositive", "oracle only\n(retrieval gap)", "retrieved only", "neither\n(no transfer)"]
    counts = [both, oracle_only, retrieved_only, neither]
    colors = ["#55A868", "#DD8452", "#8172B2", "#C44E52"]
    ax.bar(labels, counts, color=colors)
    for i, c in enumerate(counts):
        ax.text(i, c + 0.3, str(c), ha="center")
    ax.set_ylabel("hard-task count")
    ax.set_title(f"Oracle-vs-retrieved gap (n={both+oracle_only+retrieved_only+neither} hard tasks)")

    ax = axes[1]
    oracle_eff = df[(df["condition"] == "oracle") & (df["delta_eff_completion_tokens"].notna())]["delta_eff_completion_tokens"]
    retrieved_eff = df[(df["condition"] == "retrieved") & (df["delta_eff_completion_tokens"].notna())]["delta_eff_completion_tokens"]
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, [oracle_eff.mean(), retrieved_eff.mean()], width, label="mean", color="#4C72B0")
    ax.bar(x + width/2, [oracle_eff.median(), retrieved_eff.median()], width, label="median", color="#DD8452")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"oracle\n(n={len(oracle_eff)})", f"retrieved\n(n={len(retrieved_eff)})"])
    ax.set_ylabel("Delta_eff completion-token delta")
    ax.set_title("Efficiency: oracle vs. retrieved (matched-success)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df, train_df, held_out_df = load_data()

    fig_transfer_by_category(df, FIGURES_DIR / "transfer_by_category.png")
    print(f"1/4 transfer_by_category.png ({len(df)} pairs, regenerated with full enriched data)")

    fig_delta_eff_calibration(train_df, held_out_df, FIGURES_DIR / "delta_eff_calibration.png")
    print("2/4 delta_eff_calibration.png (magnitude regression + sign classification)")

    fig_property_correlation(df, FIGURES_DIR / "property_correlation.png")
    print("3/4 property_correlation.png (5 reliable vs. 4 clustered)")

    fig_oracle_vs_retrieved_gap(df, FIGURES_DIR / "oracle_vs_retrieved_gap.png")
    print("4/4 oracle_vs_retrieved_gap.png (hard-task breakdown + efficiency)")


if __name__ == "__main__":
    main()
