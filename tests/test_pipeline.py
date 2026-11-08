"""Tests for the data and modelling code.  Run: pytest"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heart import config
from heart.data import clean_data, load_dataset, load_raw, make_label, split_data, validate_schema
from heart.evaluate import choose_threshold, compute_metrics, specificity_score
from heart.features import PercentMaxHeartRate, build_pipeline, to_frame


@pytest.fixture(scope="module")
def raw():
    return load_raw()


def test_schema_matches_expected_columns(raw):
    assert list(raw.columns) == config.RAW_COLUMNS


def test_schema_validation_rejects_missing_column(raw):
    with pytest.raises(ValueError, match="Missing"):
        validate_schema(raw.drop(columns=["chol"]))


def test_cleaning_removes_all_duplicates(raw):
    clean, report = clean_data(raw)
    assert not clean.duplicated().any()
    assert report.rows_out == len(raw.drop_duplicates())


def test_invalid_codes_become_missing(raw):
    clean, _ = clean_data(raw)
    for col, allowed in config.VALID_CODES.items():
        assert clean[col].dropna().isin(allowed).all(), col


def test_label_inversion():
    df = pd.DataFrame({config.RAW_TARGET: [0, 1, 1]})
    assert make_label(df, invert=True).tolist() == [1, 0, 0]
    assert make_label(df, invert=False).tolist() == [0, 1, 1]


def test_disease_label_has_clinically_expected_direction():
    """Guards the label decision: disease patients should reach a LOWER max heart rate."""
    X, y, _ = load_dataset()
    assert X.loc[y == 1, "thalach"].mean() < X.loc[y == 0, "thalach"].mean()
    assert X.loc[y == 1, "oldpeak"].mean() > X.loc[y == 0, "oldpeak"].mean()


def test_no_patient_in_both_train_and_test():
    X, y, _ = load_dataset()
    X_tr, X_te, _, _ = split_data(X, y)
    merged = X_tr.merge(X_te, how="inner", on=list(X.columns))
    assert merged.empty, "data leakage: identical rows in train and test"


def test_split_is_stratified():
    X, y, _ = load_dataset()
    _, _, y_tr, y_te = split_data(X, y)
    assert abs(y_tr.mean() - y_te.mean()) < 0.03


def test_pipeline_handles_missing_and_unseen_categories():
    X, y, _ = load_dataset()
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    # int columns cannot hold NaN in pandas 3, so cast first (as real missing data would be)
    row = X.iloc[[0]].astype("float64")
    row.loc[:, "chol"] = np.nan
    row.loc[:, "cp"] = 9  # category never seen in training -> ignored by OneHotEncoder
    p = pipe.predict_proba(row)[0, 1]
    assert 0.0 <= p <= 1.0


def test_engineered_feature():
    out = PercentMaxHeartRate().transform(pd.DataFrame({"age": [40], "thalach": [180]}))
    assert out[config.ENGINEERED_FEATURE].iloc[0] == pytest.approx(1.0)


def test_metrics_and_specificity():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    m = compute_metrics(y, p, 0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert specificity_score(y, (p >= 0.5).astype(int)) == 0.5


def test_threshold_reaches_target_sensitivity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    t = choose_threshold(y, p, 0.9)
    assert compute_metrics(y, p, t)["recall_sensitivity"] >= 0.9


def test_to_frame_requires_all_features():
    with pytest.raises(ValueError):
        to_frame({"age": 50})


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="model not trained yet")
def test_saved_model_predicts():
    from heart.predict import predict_one
    rec = {"age": 58, "sex": 1, "cp": 0, "trestbps": 140, "chol": 260, "fbs": 0, "restecg": 1,
           "thalach": 130, "exang": 1, "oldpeak": 2.0, "slope": 1, "ca": 2, "thal": 3}
    pred = predict_one(rec)
    assert 0 <= pred.probability <= 1
    assert pred.flagged == (pred.probability >= pred.threshold)
