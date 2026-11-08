"""End-to-end training: split -> CV -> tuning -> threshold -> final test -> save.

Run:  python -m heart.train

The workflow that protects the test set:
  1. Split once into train (80%) and test (20%), stratified.
  2. Every decision (feature set, model family, hyperparameters, threshold)
     is made with cross-validation on the TRAIN set only.
  3. The selected model is evaluated on the test set exactly once, at the end.
  All other models' test scores are reported for transparency only and are
  never used to choose anything.
"""
from __future__ import annotations

import json
import logging
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from heart import __version__, config
from heart.data import load_dataset, split_data
from heart.evaluate import (
    choose_threshold,
    compute_metrics,
    plot_confusion,
    plot_roc_curves,
    plot_threshold_tradeoff,
)
from heart.features import build_pipeline
from heart.models import get_baseline, get_model_specs

logger = logging.getLogger(__name__)

SCORING = {
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "accuracy": "accuracy",
}


def repeated_cv():
    return RepeatedStratifiedKFold(n_splits=config.CV_SPLITS, n_repeats=config.CV_REPEATS,
                                   random_state=config.RANDOM_STATE)


def cv_summary(estimator, X, y) -> dict:
    """Mean and std of each metric over repeated stratified K-fold CV."""
    res = cross_validate(estimator, X, y, cv=repeated_cv(), scoring=SCORING, n_jobs=-1)
    out = {}
    for m in SCORING:
        out[f"cv_{m}"] = float(np.mean(res[f"test_{m}"]))
        out[f"cv_{m}_std"] = float(np.std(res[f"test_{m}"]))
    return out


def compare_feature_sets(X, y) -> pd.DataFrame:
    """Does the engineered feature help? Checked with CV on train only."""
    rows = []
    for spec in get_model_specs():
        for engineered in (False, True):
            s = cv_summary(build_pipeline(clone(spec.estimator), engineered), X, y)
            rows.append({"model": spec.name, "engineered_feature": engineered,
                         "cv_roc_auc": s["cv_roc_auc"], "cv_roc_auc_std": s["cv_roc_auc_std"]})
    return pd.DataFrame(rows)


def run(save: bool = True) -> dict:
    t0 = time.time()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data --------------------------------------------------------------
    X, y, cleaning = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    logger.info("Train %s (disease rate %.2f) | Test %s (disease rate %.2f)",
                X_train.shape, y_train.mean(), X_test.shape, y_test.mean())

    # 2. Feature engineering check -------------------------------------------
    fe = compare_feature_sets(X_train, y_train)
    fe.to_csv(config.REPORTS_DIR / "feature_engineering_cv.csv", index=False)
    gain = (fe[fe.engineered_feature].cv_roc_auc.values - fe[~fe.engineered_feature].cv_roc_auc.values).mean()
    use_engineered = bool(gain > 0.005)  # only keep it if it clearly helps
    logger.info("Engineered feature mean CV AUC change: %+.4f -> use=%s", gain, use_engineered)

    # 3. Baseline + untuned + tuned models (CV on train) ---------------------
    rows, tuned_pipes = [], {}
    base = get_baseline()
    rows.append({"model": base.name, "version": "baseline",
                 **cv_summary(build_pipeline(base.estimator, use_engineered), X_train, y_train)})

    tune_cv = StratifiedKFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    best_params = {}
    for spec in get_model_specs():
        untuned = build_pipeline(clone(spec.estimator), use_engineered)
        rows.append({"model": spec.name, "version": "untuned", **cv_summary(untuned, X_train, y_train)})

        search = GridSearchCV(build_pipeline(clone(spec.estimator), use_engineered),
                              spec.param_grid, scoring="roc_auc", cv=tune_cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        tuned = search.best_estimator_
        best_params[spec.name] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        # Re-score the tuned config with the SAME repeated CV as the untuned one,
        # so the comparison is apples-to-apples. (This is still slightly
        # optimistic, since the grid was chosen on overlapping folds; that is
        # why the final word comes from the untouched test set.)
        rows.append({"model": spec.name, "version": "tuned",
                     **cv_summary(clone(tuned), X_train, y_train)})
        tuned_pipes[spec.name] = tuned
        logger.info("%-20s tuned CV AUC %.3f  params=%s", spec.name, rows[-1]["cv_roc_auc"], best_params[spec.name])

    cv_table = pd.DataFrame(rows)
    cv_table.to_csv(config.REPORTS_DIR / "cv_results.csv", index=False)

    # 4. Model selection (train CV only) ---------------------------------------
    # Primary: ROC-AUC (threshold-independent ranking quality).
    # Within 1 standard error of the best, prefer the simpler / more
    # interpretable model -- a common, defensible tie-break with small data.
    tuned_rows = cv_table[cv_table.version == "tuned"].copy()
    best = tuned_rows.loc[tuned_rows.cv_roc_auc.idxmax()]
    se = best.cv_roc_auc_std / np.sqrt(config.CV_SPLITS * config.CV_REPEATS)
    simplicity_order = ["Logistic Regression", "K-Nearest Neighbors", "SVM (RBF)", "Decision Tree",
                        "Random Forest", "Gradient Boosting"]
    contenders = tuned_rows[tuned_rows.cv_roc_auc >= best.cv_roc_auc - se]
    selected_name = min(contenders.model, key=simplicity_order.index)
    logger.info("Best CV AUC: %s (%.3f, 1-SE band %.3f). Selected: %s",
                best.model, best.cv_roc_auc, se, selected_name)
    final_pipe = tuned_pipes[selected_name]

    # 5. Threshold from out-of-fold TRAIN predictions --------------------------
    oof = cross_val_predict(clone(final_pipe), X_train, y_train, cv=tune_cv, method="predict_proba")[:, 1]
    threshold = choose_threshold(y_train, oof, config.TARGET_SENSITIVITY)
    plot_threshold_tradeoff(y_train.values, oof, threshold, config.FIGURES_DIR / "07_threshold_tradeoff.png")
    oof_metrics = {"default_0.5": compute_metrics(y_train, oof, 0.5),
                   "chosen": compute_metrics(y_train, oof, threshold)}
    logger.info("Chosen threshold %.2f (OOF sensitivity %.3f, specificity %.3f)", threshold,
                oof_metrics["chosen"]["recall_sensitivity"], oof_metrics["chosen"]["specificity"])

    # 6. Final, one-time test evaluation ---------------------------------------
    test_proba = final_pipe.predict_proba(X_test)[:, 1]
    test_default = compute_metrics(y_test, test_proba, 0.5)
    test_chosen = compute_metrics(y_test, test_proba, threshold)
    plot_confusion(y_test, test_proba, 0.5, f"{selected_name} - test, threshold 0.50",
                   config.FIGURES_DIR / "08_confusion_default.png")
    plot_confusion(y_test, test_proba, threshold, f"{selected_name} - test, threshold {threshold:.2f}",
                   config.FIGURES_DIR / "09_confusion_chosen.png")

    # Transparency only: every tuned model on the test set at 0.5.
    # NOT used for selection (selection already happened above).
    test_all = []
    curves = {}
    for name, pipe in tuned_pipes.items():
        p = pipe.predict_proba(X_test)[:, 1]
        curves[name] = (y_test.values, p)
        test_all.append({"model": name, **compute_metrics(y_test, p, 0.5)})
    pd.DataFrame(test_all).to_csv(config.REPORTS_DIR / "test_results_all_models.csv", index=False)
    plot_roc_curves(curves, config.FIGURES_DIR / "06_roc_curves_test.png", "ROC curves on the test set (tuned models)")

    # 7. Refit the chosen pipeline on ALL training data and save --------------
    # (GridSearchCV refit already did this; we keep test data out of training
    # so the reported test metrics stay honest for the saved artefact.)
    metadata = {
        "project_version": __version__,
        "model_name": selected_name,
        "best_params": best_params[selected_name],
        "threshold": threshold,
        "target_sensitivity": config.TARGET_SENSITIVITY,
        "use_engineered_feature": use_engineered,
        "features": config.FEATURES,
        "label": config.LABEL,
        "label_inverted_from_raw_target": config.INVERT_TARGET,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "cleaning": {"rows_in": cleaning.rows_in, "duplicates_removed": cleaning.duplicates_removed,
                     "rows_out": cleaning.rows_out, "invalid_codes_set_missing": cleaning.invalid_codes_set_missing},
        "test_metrics_threshold_0.5": test_default,
        "test_metrics_chosen_threshold": test_chosen,
        "oof_train_metrics": oof_metrics,
        "all_best_params": best_params,
        "selection_rule": "highest tuned CV ROC-AUC; within 1 SE prefer simpler model",
        "best_cv_model": best.model,
    }
    if save:
        joblib.dump({"pipeline": final_pipe, "metadata": metadata}, config.MODEL_PATH)
        (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2, default=float))
        # save the split ids so interpretation uses identical data
        joblib.dump({"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test},
                    config.MODELS_DIR / "split.joblib")
    logger.info("Done in %.0fs. Test @%.2f: %s", time.time() - t0, threshold,
                {k: round(v, 3) for k, v in test_chosen.items() if isinstance(v, float)})
    return metadata


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=FutureWarning)
    # The majority-class baseline never predicts "disease", so precision is undefined (0/0) for it.
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    run()


if __name__ == "__main__":
    main()
