"""Exploratory data analysis: figures + a markdown data profile.

Run:  python -m heart.eda
Outputs go to reports/figures/ and reports/data_profile.md.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from heart import config
from heart.data import clean_data, load_raw, make_label

logger = logging.getLogger(__name__)
PALETTE = {config.NEGATIVE_CLASS_NAME: "#4C72B0", config.POSITIVE_CLASS_NAME: "#DD5555"}


def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label"] = make_label(df).map({0: config.NEGATIVE_CLASS_NAME, 1: config.POSITIVE_CLASS_NAME})
    return out


def iqr_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Count values outside 1.5*IQR for each continuous feature (flagged, not removed)."""
    rows = []
    for col in config.NUMERIC_FEATURES:
        q1, q3 = df[col].quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        mask = (df[col] < lo) | (df[col] > hi)
        rows.append({"feature": col, "lower_fence": lo, "upper_fence": hi,
                     "n_outliers": int(mask.sum()),
                     "outlier_values": sorted(df.loc[mask, col].tolist())})
    return pd.DataFrame(rows)


def plot_class_balance(d, path):
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    counts = d["label"].value_counts().reindex([config.NEGATIVE_CLASS_NAME, config.POSITIVE_CLASS_NAME])
    ax.bar(counts.index, counts.values, color=[PALETTE[k] for k in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, f"{v} ({v / counts.sum():.0%})", ha="center", fontsize=9)
    ax.set_ylabel("Patients (unique rows)")
    ax.set_title("Class balance")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_numeric_distributions(d, path):
    fig, axes = plt.subplots(1, len(config.NUMERIC_FEATURES), figsize=(17, 3.4))
    for ax, col in zip(axes, config.NUMERIC_FEATURES):
        sns.kdeplot(data=d, x=col, hue="label", fill=True, common_norm=False,
                    palette=PALETTE, ax=ax, alpha=0.35, warn_singular=False)
        ax.set_title(config.FEATURE_DESCRIPTIONS[col], fontsize=8.5)
        ax.set_ylabel("")
    fig.suptitle("Continuous features by class (density)", y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_boxplots(d, path):
    fig, axes = plt.subplots(1, len(config.NUMERIC_FEATURES), figsize=(17, 3.4))
    for ax, col in zip(axes, config.NUMERIC_FEATURES):
        sns.boxplot(data=d, x="label", y=col, hue="label", palette=PALETTE, ax=ax, legend=False)
        ax.set_xlabel("")
    fig.suptitle("Outlier check: boxplots by class", y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_categorical_rates(d, path):
    cats = config.BINARY_FEATURES + config.NOMINAL_FEATURES + config.ORDINAL_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, col in zip(axes.ravel(), cats):
        tab = d.groupby(col)["y"].agg(["mean", "size"]).reset_index()
        ax.bar(tab[col].astype(int).astype(str), tab["mean"], color="#DD5555", alpha=0.85)
        for i, (m, n) in enumerate(zip(tab["mean"], tab["size"])):
            ax.text(i, m + 0.02, f"n={n}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(config.FEATURE_DESCRIPTIONS[col], fontsize=8.5)
        ax.set_ylabel("Disease rate")
    fig.suptitle("Disease rate within each category", y=1.0)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_correlation(d, path):
    cols = config.FEATURES + ["y"]
    corr = d[cols].rename(columns={"y": config.LABEL}).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
    ax.set_title("Spearman correlation (rank-based; robust to skew and coded categories)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return corr


def plot_label_direction(raw_unique, path):
    """Evidence for the label inversion: clinical markers by RAW target value."""
    markers = {"thalach": "Max heart rate", "oldpeak": "ST depression",
               "exang": "Exercise angina rate", "ca": "Vessels (0-3)"}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (col, name) in zip(axes, markers.items()):
        means = raw_unique.groupby(config.RAW_TARGET)[col].mean()
        ax.bar(["raw target = 0", "raw target = 1"], means.values, color=["#888", "#bbb"])
        ax.set_title(name, fontsize=9)
    fig.suptitle("Mean clinical markers by RAW target value: target = 0 shows the disease-like profile", y=1.03)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def write_profile(raw, clean, report, corr, outliers, path):
    d = clean.copy()
    lines = ["# Data profile", "",
             "Generated by `python -m heart.eda`. All statistics use the de-duplicated data.", "",
             "## Shape and cleaning", "",
             f"- Raw rows: **{report.rows_in}**, columns: **{raw.shape[1]}**",
             f"- Exact duplicate rows removed: **{report.duplicates_removed}**",
             f"- Unique patients kept: **{report.rows_out}**",
             f"- Missing values in raw file: **{int(raw.isna().sum().sum())}**",
             f"- Invalid category codes set to missing: `{report.invalid_codes_set_missing}`",
             "", "## Features", "", "| Column | Meaning | Treated as | Range / values |", "|---|---|---|---|"]
    group = {**{c: "numeric" for c in config.NUMERIC_FEATURES},
             **{c: "binary" for c in config.BINARY_FEATURES},
             **{c: "nominal (one-hot)" for c in config.NOMINAL_FEATURES},
             **{c: "ordinal" for c in config.ORDINAL_FEATURES}}
    for c in config.FEATURES:
        if c in config.NUMERIC_FEATURES:
            rng = f"{d[c].min():g} to {d[c].max():g} (median {d[c].median():g})"
        else:
            rng = ", ".join(f"{int(v)}" for v in sorted(d[c].dropna().unique()))
        lines.append(f"| `{c}` | {config.FEATURE_DESCRIPTIONS[c]} | {group[c]} | {rng} |")
    y = make_label(d)
    lines += ["", "## Class balance (modelling label)", "",
              f"- disease (1): **{int(y.sum())}** ({y.mean():.1%})",
              f"- no disease (0): **{int((1 - y).sum())}** ({1 - y.mean():.1%})",
              "", "## Correlation with the label (Spearman)", "", "| Feature | rho |", "|---|---|"]
    for f, v in corr[config.LABEL].drop(config.LABEL).sort_values(key=abs, ascending=False).items():
        lines.append(f"| `{f}` | {v:+.2f} |")
    lines += ["", "## Potential outliers (1.5 x IQR rule; flagged, not removed)", "",
              "| Feature | Fences | Count | Values |", "|---|---|---|---|"]
    for _, r in outliers.iterrows():
        lines.append(f"| `{r.feature}` | {r.lower_fence:g} to {r.upper_fence:g} | {r.n_outliers} | {r.outlier_values} |")
    lines += ["", "All flagged values are physiologically possible (e.g. cholesterol 564 mg/dl is very high "
              "but real), so they are kept. Removing them would hide exactly the extreme patients a "
              "clinical model must handle. Median imputation and tree models are robust to them."]
    sex_tab = pd.crosstab(d["sex"].map({0: "female", 1: "male"}), y.map({0: "no disease", 1: "disease"}))
    lines += ["", "## Representation by sex", "", sex_tab.to_markdown(), "",
              f"Males are {d['sex'].mean():.0%} of patients, so estimates for women rest on far fewer "
              "examples and are less reliable."]
    path.write_text("\n".join(lines) + "\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    clean, report = clean_data(raw)
    d = _labelled(clean)
    d["y"] = make_label(clean)

    plot_class_balance(d, config.FIGURES_DIR / "01_class_balance.png")
    plot_numeric_distributions(d, config.FIGURES_DIR / "02_numeric_distributions.png")
    plot_boxplots(d, config.FIGURES_DIR / "03_boxplots.png")
    plot_categorical_rates(d, config.FIGURES_DIR / "04_categorical_disease_rates.png")
    corr = plot_correlation(d, config.FIGURES_DIR / "05_correlation.png")
    plot_label_direction(raw.drop_duplicates(), config.FIGURES_DIR / "00_label_direction.png")
    write_profile(raw, clean, report, corr, iqr_outliers(clean), config.REPORTS_DIR / "data_profile.md")
    logger.info("EDA complete: %s", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
