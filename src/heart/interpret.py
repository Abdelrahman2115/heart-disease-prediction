"""Model interpretation.

Run (after training):  python -m heart.interpret

Three complementary views:
  1. Permutation importance (model-agnostic): how much does test ROC-AUC drop
     when one feature's values are shuffled? Large drop = model relies on it.
  2. Logistic-regression odds ratios (global, model-specific): direction and
     size of each feature's effect, holding the others fixed.
  3. Per-patient contributions: for a linear model, each feature adds
     coefficient x (scaled value) to the log-odds, so a single prediction can
     be explained term by term.
"""
from __future__ import annotations

import logging

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from heart import config

logger = logging.getLogger(__name__)


def load_artifacts():
    bundle = joblib.load(config.MODEL_PATH)
    split = joblib.load(config.MODELS_DIR / "split.joblib")
    return bundle["pipeline"], bundle["metadata"], split


def permutation_table(pipe, X, y, n_repeats: int = 50) -> pd.DataFrame:
    r = permutation_importance(pipe, X, y, scoring="roc_auc", n_repeats=n_repeats,
                               random_state=config.RANDOM_STATE, n_jobs=-1)
    return (pd.DataFrame({"feature": X.columns, "importance_mean": r.importances_mean,
                          "importance_std": r.importances_std})
            .sort_values("importance_mean", ascending=False).reset_index(drop=True))


def plot_permutation(table: pd.DataFrame, path):
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(t.feature, t.importance_mean, xerr=t.importance_std, color="#4C72B0", alpha=0.85)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled")
    ax.set_title("Permutation importance (selected model, test set)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def linear_coefficients(pipe) -> pd.DataFrame | None:
    """Odds ratios per (transformed) feature, if the model is linear."""
    model = pipe.named_steps["model"]
    if not hasattr(model, "coef_"):
        return None
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    coef = model.coef_.ravel()
    return (pd.DataFrame({"feature": names, "coefficient": coef, "odds_ratio": np.exp(coef)})
            .sort_values("coefficient", key=np.abs, ascending=False).reset_index(drop=True))


def plot_coefficients(table: pd.DataFrame, path):
    t = table[table.coefficient != 0].iloc[::-1]
    colors = ["#DD5555" if c > 0 else "#4C72B0" for c in t.coefficient]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(t.feature, t.coefficient, color=colors)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Coefficient (log-odds). Red pushes towards 'disease', blue away")
    ax.set_title("Logistic regression coefficients (features shrunk to 0 by L1 omitted)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def explain_one(pipe, row: pd.DataFrame, top: int = 6) -> pd.DataFrame:
    """Per-feature contribution to the log-odds for one patient (linear models)."""
    model = pipe.named_steps["model"]
    pre = pipe.named_steps["preprocess"]
    z = pre.transform(row)[0]
    contrib = model.coef_.ravel() * z
    out = pd.DataFrame({"feature": pre.get_feature_names_out(), "contribution": contrib})
    out = out[out.contribution != 0]
    return out.reindex(out.contribution.abs().sort_values(ascending=False).index).head(top)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    pipe, meta, split = load_artifacts()
    X_test, y_test = split["X_test"], split["y_test"]

    perm = permutation_table(pipe, X_test, y_test)
    perm.to_csv(config.REPORTS_DIR / "permutation_importance.csv", index=False)
    plot_permutation(perm, config.FIGURES_DIR / "10_permutation_importance.png")

    lines = ["# Model interpretation", "", f"Selected model: **{meta['model_name']}**", "",
             "## Permutation importance (test set, ROC-AUC drop, 50 repeats)", "",
             "| Feature | Mean drop | Std |", "|---|---|---|"]
    lines += [f"| `{r.feature}` | {r.importance_mean:.3f} | {r.importance_std:.3f} |" for r in perm.itertuples()]

    coefs = linear_coefficients(pipe)
    if coefs is not None:
        coefs.to_csv(config.REPORTS_DIR / "coefficients.csv", index=False)
        plot_coefficients(coefs, config.FIGURES_DIR / "11_coefficients.png")
        lines += ["", "## Logistic regression odds ratios", "",
                  "Numeric features are standardised, so their odds ratio is per **one standard deviation**. "
                  "One-hot features compare that category against the others. An odds ratio of 1.0 "
                  "means the L1 penalty removed the feature.", "",
                  "| Transformed feature | Coefficient | Odds ratio |", "|---|---|---|"]
        lines += [f"| `{r.feature}` | {r.coefficient:+.3f} | {r.odds_ratio:.2f} |" for r in coefs.itertuples()]

        proba = pipe.predict_proba(X_test)[:, 1]
        for label, idx in [("highest-risk", int(np.argmax(proba))), ("lowest-risk", int(np.argmin(proba)))]:
            row = X_test.iloc[[idx]]
            ex = explain_one(pipe, row)
            lines += ["", f"## Example explanation: {label} test patient", "",
                      f"Predicted probability of the *disease* label: **{proba[idx]:.2f}** "
                      f"(dataset label: {'disease' if y_test.iloc[idx] == 1 else 'no disease'})", "",
                      "| Feature | Contribution to log-odds |", "|---|---|"]
            lines += [f"| `{r.feature}` | {r.contribution:+.2f} |" for r in ex.itertuples()]

    lines += ["", "## Caveats", "",
              "- Importance shows what **this model** relies on to reproduce **this dataset's labels**; "
              "it is association, not causation.",
              "- Correlated features share credit: shuffling one can understate its importance when a "
              "correlated partner carries the same information (e.g. `slope` and `oldpeak`).",
              "- With 61 test patients, importance estimates are noisy; see the standard deviations."]
    (config.REPORTS_DIR / "interpretation.md").write_text("\n".join(lines) + "\n")
    logger.info("Interpretation written.")


if __name__ == "__main__":
    main()
