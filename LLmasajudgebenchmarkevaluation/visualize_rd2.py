"""
Visualization functions for Research Direction 2 (LLM-as-a-Judge for
benchmark-quality assessment) results.

Input: a folder containing, for EACH judge backbone you ran, the pipeline's
three output files, sharing one filename tag per judge:

  audit_results_<tag>.xlsx  - one row per (task_id, dataset): the five
                               rubric-dimension scores plus
                               benchmark_quality_score. 164 rows = 82
                               matched tasks x 2 datasets (HumanEval,
                               HumanEvalNext).
  comparison_<tag>.xlsx     - one row per matched task_id: HumanEval score,
                               HumanEvalNext score, and their difference.
                               82 rows.
  summary_<tag>.xlsx        - dataset-level means (not loaded directly here
                               - fig1/fig2/fig6 recompute the same means
                               from audit_results so everything derives
                               from one source of truth).

The judge panel is NOT hardcoded. The script scans the input folder for every
audit_results_<tag>.xlsx file, derives <tag> automatically, and pairs it with
its summary_/comparison_ counterparts -- so it works unchanged whether your
folder holds 3 judges or 6, and regardless of whether filenames use hyphens
("Qwen2.5-Coder-14B-Instruct"), no hyphens ("Qwen2.5Coder14BInstruct"), or
carry a Claude-upload hash prefix ("3127d820-summary_...xlsx"). If you rename
a judge's files or add a new judge backbone, just drop the three files in --
no code edits needed.

Produces six figures:

  fig1_dimension_heatmap.png      -> mean score per rubric dimension per
                                      judge, split HumanEval | HumanEvalNext
                                      (sequential heatmap)
  fig2_humanevalnext_diff.png     -> HumanEvalNext - HumanEval difference
                                      per judge per dimension (diverging
                                      heatmap)
  fig3_paired_per_task.png        -> paired per-task comparison, HumanEval
                                      vs. HumanEvalNext, one scatter panel
                                      per judge
  fig4_difference_distribution.png-> distribution of the per-task
                                      difference, one box per judge
  fig5_cross_judge_agreement.png  -> pairwise agreement between judges'
                                      benchmark_quality_score, one scatter
                                      panel per judge pair (grid layout)
  fig6_dimension_bars.png         -> grouped bar chart, dimension on the
                                      x-axis, one bar per judge

Run: python3 visualize_rd2.py <folder_with_rd2_xlsx> <output_folder>
"""

import sys
import glob
import os
import re
import math
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Categorical colors, fixed assignment order (Okabe-Ito colorblind-safe
# palette). Judges are assigned a color in the order they are discovered on
# disk, so the same judge keeps the same color across every figure in one
# run. Extend this list if you ever audit more than 8 judge backbones at
# once (it will cycle with a warning rather than crash).
CATEGORICAL_POOL = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#CC79A7",  # purple
    "#F0E442",  # yellow
    "#000000",  # black
]

SEQUENTIAL_CMAP = "Blues"
DIVERGING_CMAP = "RdBu_r"

DIMENSIONS = [
    "correct_tests_score", "coverage_score", "solution_score",
    "clarity_score", "representativeness_score",
]
DIMENSION_LABELS = [
    "Correctness of\nTests", "Coverage", "Solution\nQuality",
    "Problem\nClarity", "Representativeness",
]

mpl.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


UPLOAD_PREFIX_RE = re.compile(r"^[0-9a-fA-F]{6,10}-")


def _clean_name(path):
    return UPLOAD_PREFIX_RE.sub("", os.path.basename(path))


def _tag_to_label(tag):
    """Turn a filename tag like 'allenai_OLMo-2-1124-13B-Instruct' or
    'Qwen_Qwen2.5-Coder-14B-Instruct' into a readable display label. Most
    tags are '<org>_<model-name>', and the model name alone is enough to
    identify the judge, so the (often redundant) org prefix is dropped when
    there's an unambiguous split point."""
    if "_" in tag:
        org, rest = tag.split("_", 1)
        label = rest if rest else tag
    else:
        label = tag
    if label[:1].islower():
        label = label[0].upper() + label[1:]
    return label


def discover_judges(folder):
    """Scan folder for every audit_results_<tag>.xlsx file and pair it with
    its summary_<tag>.xlsx / comparison_<tag>.xlsx counterparts. Returns an
    ordered list of (tag, label) tuples, in the order found on disk (glob's
    sorted order, so results are deterministic run to run)."""
    pattern = re.compile(r"^audit_results_(.+)\.xlsx$", re.IGNORECASE)
    tags = []
    for path in sorted(glob.glob(os.path.join(folder, "*.xlsx"))):
        m = pattern.match(_clean_name(path))
        if m and m.group(1) not in tags:
            tags.append(m.group(1))

    judges = []
    for tag in tags:
        has_summary = _find_exact(folder, f"summary_{tag}.xlsx") is not None
        has_comparison = _find_exact(folder, f"comparison_{tag}.xlsx") is not None
        if has_summary and has_comparison:
            judges.append((tag, _tag_to_label(tag)))
        else:
            missing = [k for k, ok in [("summary", has_summary), ("comparison", has_comparison)] if not ok]
            print(f"WARNING: found audit_results_{tag}.xlsx but no matching {'/'.join(missing)} file(s) "
                  f"for it in {folder} -- skipping this judge.")
    return judges


def _find_exact(folder, exact_name):
    """Find a file in folder whose name, after stripping any upload-hash
    prefix, equals exact_name exactly."""
    for path in glob.glob(os.path.join(folder, "*.xlsx")):
        if _clean_name(path) == exact_name:
            return path
    return None


def load_all(folder):
    """Discover every judge in folder and load/concatenate their
    audit_results and comparison files. Returns (audit_df, comp_df, judges)
    where judges is the ordered [(tag, label), ...] list discovery found,
    each tagged with a judge_label column (categorical, ordered to match)."""
    judges = discover_judges(folder)
    if not judges:
        raise FileNotFoundError(
            f"No complete (audit_results + summary + comparison) judge file sets found in {folder}. "
            "Expected files named audit_results_<tag>.xlsx / summary_<tag>.xlsx / comparison_<tag>.xlsx."
        )

    judge_tags = [tag for tag, _ in judges]
    audit_frames, comp_frames = [], []
    for tag, _label in judges:
        audit_path = _find_exact(folder, f"audit_results_{tag}.xlsx")
        adf = pd.read_excel(audit_path)[["dataset", "task_id"] + DIMENSIONS + ["benchmark_quality_score"]].copy()
        adf["judge_label"] = tag
        audit_frames.append(adf)

        comp_path = _find_exact(folder, f"comparison_{tag}.xlsx")
        cdf = pd.read_excel(comp_path)[["task_id", "HumanEval", "HumanEvalNext", "difference"]].copy()
        cdf["judge_label"] = tag
        comp_frames.append(cdf)

    audit_df = pd.concat(audit_frames, ignore_index=True)
    audit_df["judge_label"] = pd.Categorical(audit_df["judge_label"], categories=judge_tags, ordered=True)

    comp_df = pd.concat(comp_frames, ignore_index=True)
    comp_df["judge_label"] = pd.Categorical(comp_df["judge_label"], categories=judge_tags, ordered=True)

    return audit_df, comp_df, judges


def _palette(judges):
    """tag -> color, assigned in discovery order. Cycles (with a warning)
    past 8 judges rather than crashing."""
    if len(judges) > len(CATEGORICAL_POOL):
        print(f"WARNING: {len(judges)} judges discovered but only {len(CATEGORICAL_POOL)} "
              "distinct colorblind-safe colors defined -- colors will repeat.")
    return {tag: CATEGORICAL_POOL[i % len(CATEGORICAL_POOL)] for i, (tag, _label) in enumerate(judges)}


# ---------------------------------------------------------------------------
# Mean score per dimension per judge, SPLIT BY DATASET. Form: sequential
# heatmap - the job is magnitude across two categorical axes (judge x
# dimension). One hue, light -> dark, SHARED across both dataset blocks so a
# value is comparable across judges and across datasets, not only averaged
# away.
# ---------------------------------------------------------------------------
def fig1_dimension_heatmap(audit_df, judges, out_path):
    judge_tags = [t for t, _ in judges]
    judge_labels = {t: l for t, l in judges}
    cols = DIMENSIONS + ["benchmark_quality_score"]
    labels = DIMENSION_LABELS + ["Overall"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 1.2 + 0.5 * len(judges)), gridspec_kw={"wspace": 0.55})
    im = None
    for ax, dataset in zip(axes, ["HumanEval", "HumanEvalNext"]):
        sub = audit_df[audit_df["dataset"] == dataset]
        agg = sub.groupby("judge_label", observed=True)[cols].mean().loc[judge_tags]
        mat = agg[cols].values
        im = ax.imshow(mat, cmap=SEQUENTIAL_CMAP, vmin=1, vmax=5, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_yticks(range(len(judge_tags)))
        ax.set_yticklabels([judge_labels[t] for t in judge_tags])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                color = "white" if val > 3.3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.5, color=color)
        ax.set_title(dataset, fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Mean score (1-5)")
    fig.suptitle("Mean benchmark-quality score per dimension, by judge backbone and dataset", y=1.05)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# HumanEvalNext - HumanEval difference per judge per dimension. Form:
# diverging heatmap - the job is POLARITY, centered at 0.
# ---------------------------------------------------------------------------
def fig2_humanevalnext_diff(audit_df, judges, out_path):
    judge_tags = [t for t, _ in judges]
    judge_labels = {t: l for t, l in judges}
    pivot = (
        audit_df.groupby(["judge_label", "dataset"], observed=True)[DIMENSIONS + ["benchmark_quality_score"]]
        .mean()
        .reset_index()
    )
    he = pivot[pivot["dataset"] == "HumanEval"].set_index("judge_label")
    hen = pivot[pivot["dataset"] == "HumanEvalNext"].set_index("judge_label")
    cols = DIMENSIONS + ["benchmark_quality_score"]
    diff = (hen[cols] - he[cols]).loc[judge_tags]

    labels = DIMENSION_LABELS + ["Overall"]
    mat = diff[cols].values
    vmax = max(np.abs(mat).max(), 0.05)

    fig, ax = plt.subplots(figsize=(9, 1.0 + 0.45 * len(judges)))
    im = ax.imshow(mat, cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_yticks(range(len(judge_tags)))
    ax.set_yticklabels([judge_labels[t] for t in judge_tags])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8.5, color=color)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label="HumanEvalNext - HumanEval")
    ax.set_title("Detected benchmark-quality difference, HumanEvalNext vs. HumanEval\n(positive = judge scored HumanEvalNext higher)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return diff


# ---------------------------------------------------------------------------
# Paired per-task comparison. Form: small-multiples scatter, one panel per
# judge, x=HumanEval score, y=HumanEvalNext score, with a y=x reference line
# - the job is showing where a matched pair diverges, which a paired
# scatter shows more directly than an aggregate.
# ---------------------------------------------------------------------------
def fig3_paired_per_task(comp_df, judges, out_path):
    colors = _palette(judges)
    n = len(judges)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.5), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (tag, label) in zip(axes, judges):
        sub = comp_df[comp_df["judge_label"] == tag]
        ax.plot([1, 5], [1, 5], color="#999999", linewidth=1, linestyle="--", zorder=1)
        ax.scatter(
            sub["HumanEval"], sub["HumanEvalNext"],
            s=20, alpha=0.55, color=colors[tag], edgecolors="none", zorder=2,
        )
        pct_above = (sub["difference"] > 0).mean() * 100
        r = sub["HumanEval"].corr(sub["HumanEvalNext"])
        ax.set_title(f"{label}\n{pct_above:.0f}% scored higher on HumanEvalNext (r = {r:.2f})", fontsize=9)
        ax.set_xlim(0.8, 5.2)
        ax.set_ylim(0.8, 5.2)
        ax.set_xlabel("HumanEval score")
        ax.set_aspect("equal")
    axes[0].set_ylabel("HumanEvalNext score")
    fig.suptitle("Matched-task benchmark-quality score: HumanEval vs. HumanEvalNext, per judge", y=1.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Distribution of the per-task difference, per judge. Form: box plot - the
# job is comparing a distribution's center/spread/skew across a categorical
# axis (judge).
# ---------------------------------------------------------------------------
def fig4_difference_distribution(comp_df, judges, out_path):
    colors = _palette(judges)
    judge_tags = [t for t, _ in judges]
    judge_labels = [l for _, l in judges]

    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(judges)), 4.5))
    data = [comp_df.loc[comp_df["judge_label"] == t, "difference"].dropna().values for t in judge_tags]
    ax.axhline(0, color="#999999", linewidth=1, linestyle="--", zorder=1)
    bp = ax.boxplot(
        data, tick_labels=judge_labels, patch_artist=True, showfliers=True,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markersize=3, alpha=0.4, markeredgecolor="none"),
        zorder=2,
    )
    for patch, tag in zip(bp["boxes"], judge_tags):
        patch.set_facecolor(colors[tag])
        patch.set_alpha(0.65)
    ax.set_ylabel("Per-task difference (HumanEvalNext - HumanEval)")
    ax.set_xticklabels(judge_labels, rotation=12, ha="right", fontsize=8.5)
    ax.set_title("Distribution of the per-task benchmark-quality difference, by judge")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-judge agreement, SPLIT BY DATASET. Form: small-multiples scatter,
# one panel per judge PAIR, benchmark_quality_score of judge A vs. judge B
# at the (task_id, dataset) level - the job is showing how well two judges'
# absolute scores agree, point by point. HumanEval vs. HumanEvalNext points
# are a distinct category, encoded by a fixed two-color scheme (not the
# judge palette, which is already used for panel/axis identity). Panels are
# laid out in a grid (<=3 per row) so the figure stays legible as the judge
# panel grows past 3 (and the pair count past C(3,2)=3).
# ---------------------------------------------------------------------------
DATASET_COLOR = {"HumanEval": "#0072B2", "HumanEvalNext": "#D55E00"}


def fig5_cross_judge_agreement(audit_df, judges, out_path):
    judge_tags = [t for t, _ in judges]
    judge_labels = {t: l for t, l in judges}
    wide = audit_df.pivot_table(
        index=["task_id", "dataset"], columns="judge_label",
        values="benchmark_quality_score", observed=True,
    ).reset_index()

    pairs = list(itertools.combinations(judge_tags, 2))
    if not pairs:
        print("Only one judge discovered -- skipping cross-judge agreement figure (needs >=2 judges).")
        return pd.DataFrame(columns=["judge_a", "judge_b", "pearson_r_overall", "mean_abs_diff"])

    ncols = min(len(pairs), 3)
    nrows = math.ceil(len(pairs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 5.2 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    agreement_rows = []
    for ax, (ja, jb) in zip(axes, pairs):
        ax.plot([1, 5], [1, 5], color="#999999", linewidth=1, linestyle="--", zorder=1)
        r_parts = []
        for dataset in ["HumanEval", "HumanEvalNext"]:
            sub = wide[wide["dataset"] == dataset]
            x, y = sub[ja], sub[jb]
            ax.scatter(x, y, s=18, alpha=0.5, color=DATASET_COLOR[dataset], edgecolors="none", zorder=2)
            r_parts.append(f"r_{'HE' if dataset == 'HumanEval' else 'HEN'}={x.corr(y):.2f}")
        x_all, y_all = wide[ja], wide[jb]
        r_all = x_all.corr(y_all)
        mae = (x_all - y_all).abs().mean()
        agreement_rows.append({
            "judge_a": judge_labels[ja], "judge_b": judge_labels[jb],
            "pearson_r_overall": r_all, "mean_abs_diff": mae,
        })
        ax.set_title(f"{judge_labels[ja]}\nvs.\n{judge_labels[jb]}\n({', '.join(r_parts)}, MAE={mae:.2f})", fontsize=8.5)
        ax.set_xlabel(f"{judge_labels[ja]} score")
        ax.set_xlim(0.8, 5.2)
        ax.set_ylim(0.8, 5.2)
        ax.set_aspect("equal")

    for ax in axes[len(pairs):]:
        ax.axis("off")
    for row_start in range(0, len(pairs), ncols):
        axes[row_start].set_ylabel("Score")

    # Reserve a fixed-HEIGHT header band (in inches) for the legend+title,
    # regardless of nrows -- not a fraction that grows with the figure, which
    # is what previously left a large blank gap above the grid as more
    # judges (and so more rows) were discovered.
    fig_height = 5.2 * nrows
    header_inches = 0.9
    rect_top = max(0.80, 1 - header_inches / fig_height)
    fig.tight_layout(rect=[0, 0, 1, rect_top])

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", color=DATASET_COLOR["HumanEval"], label="HumanEval"),
        plt.Line2D([0], [0], marker="o", linestyle="none", color=DATASET_COLOR["HumanEvalNext"], label="HumanEvalNext"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, rect_top + 0.005), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Cross-judge agreement on benchmark_quality_score, by dataset\n(one point per matched task)", y=rect_top + 0.055)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(agreement_rows)


# ---------------------------------------------------------------------------
# Mean score per dimension, grouped by judge, SPLIT BY DATASET. Form:
# grouped bar chart, two side-by-side subplots (HumanEval | HumanEvalNext)
# on a shared y-axis.
# ---------------------------------------------------------------------------
def fig6_dimension_bars(audit_df, judges, out_path):
    colors = _palette(judges)
    judge_tags = [t for t, _ in judges]
    judge_labels = {t: l for t, l in judges}

    fig, subaxes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
    n_judges = len(judge_tags)
    n_dims = len(DIMENSIONS)
    group_width = 0.75
    bar_width = group_width / n_judges
    x = np.arange(n_dims)

    for ax, dataset in zip(subaxes, ["HumanEval", "HumanEvalNext"]):
        sub = audit_df[audit_df["dataset"] == dataset]
        agg = sub.groupby("judge_label", observed=True)[DIMENSIONS].mean().loc[judge_tags]
        for i, tag in enumerate(judge_tags):
            offset = (i - (n_judges - 1) / 2) * bar_width
            vals = agg.loc[tag, DIMENSIONS].values
            ax.bar(x + offset, vals, width=bar_width * 0.9, color=colors[tag], label=judge_labels[tag])
        ax.set_xticks(x)
        ax.set_xticklabels(DIMENSION_LABELS)
        ax.set_ylim(0, 5.6)
        ax.set_title(dataset, fontsize=13, fontweight="bold")

    subaxes[0].set_ylabel("Mean judged score (1-5)")
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[t], label=judge_labels[t]) for t in judge_tags]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=min(n_judges, 4), frameon=False, fontsize=9)
    fig.suptitle("Mean benchmark-quality score per dimension, by judge backbone and dataset", y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    out_folder = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(out_folder, exist_ok=True)

    audit_df, comp_df, judges = load_all(folder)
    print(f"Discovered {len(judges)} judge(s): " + ", ".join(label for _tag, label in judges))

    fig1_dimension_heatmap(audit_df, judges, os.path.join(out_folder, "fig1_dimension_heatmap.png"))
    diff = fig2_humanevalnext_diff(audit_df, judges, os.path.join(out_folder, "fig2_humanevalnext_diff.png"))
    fig3_paired_per_task(comp_df, judges, os.path.join(out_folder, "fig3_paired_per_task.png"))
    fig4_difference_distribution(comp_df, judges, os.path.join(out_folder, "fig4_difference_distribution.png"))
    agreement = fig5_cross_judge_agreement(audit_df, judges, os.path.join(out_folder, "fig5_cross_judge_agreement.png"))
    fig6_dimension_bars(audit_df, judges, os.path.join(out_folder, "fig6_dimension_bars.png"))

    print("Saved 6 figures to", out_folder)

    print("\nHumanEvalNext - HumanEval difference, per judge per dimension:")
    print(diff.round(3).to_string())

    print("\nProportion of matched tasks scored higher on HumanEvalNext, per judge:")
    print(comp_df.groupby("judge_label", observed=True)["difference"].apply(lambda s: (s > 0).mean()).round(3).to_string())

    print("\nCross-judge agreement on benchmark_quality_score:")
    print(agreement.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
