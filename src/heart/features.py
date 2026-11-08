"""Preprocessing and feature engineering as scikit-learn transformers.

Everything that *learns* from data (medians, means, standard deviations,
category lists) lives inside a Pipeline. The pipeline is fitted only on
training folds, so no statistics leak in from validation or test data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart import config


class PercentMaxHeartRate(BaseEstimator, TransformerMixin):
    """Add pct_max_hr = thalach / (220 - age).

    Raw max heart rate is hard to compare across ages, because the heart rate
    a person can reach falls with age. Dividing by the common age-predicted
    maximum (220 - age) gives "what share of their expected maximum did this
    person reach?", a more comparable measure of exercise capacity.

    The transformer is stateless (it learns nothing), so it cannot leak data,
    but writing it as a transformer means it runs identically at training
    time and inside the deployed app.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[config.ENGINEERED_FEATURE] = X["thalach"] / (220 - X["age"])
        return X

    def get_feature_names_out(self, input_features=None):
        names = list(input_features) if input_features is not None else []
        return np.asarray(names + [config.ENGINEERED_FEATURE], dtype=object)


def build_preprocessor(add_engineered: bool = False) -> Pipeline:
    """Column-wise preprocessing.

    * numeric  -> median imputation (robust to skew/outliers) + standardisation
    * binary   -> most-frequent imputation; already 0/1, no scaling needed
    * nominal  -> most-frequent imputation + one-hot encoding, because the
                  integer codes are labels, not quantities
    * ordinal  -> most-frequent imputation + standardisation (order matters)

    Scaling matters for distance- and margin-based models (KNN, SVM) and for
    regularised logistic regression. It does not affect tree models, so one
    shared preprocessor is used for every model to keep comparisons fair.
    """
    numeric = list(config.NUMERIC_FEATURES)
    if add_engineered:
        numeric.append(config.ENGINEERED_FEATURE)

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("bin", SimpleImputer(strategy="most_frequent"), config.BINARY_FEATURES),
            ("nom", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), config.NOMINAL_FEATURES),
            ("ord", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("scale", StandardScaler()),
            ]), config.ORDINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    steps = []
    if add_engineered:
        steps.append(("engineer", PercentMaxHeartRate()))
    steps.append(("columns", column_transformer))
    return Pipeline(steps)


def build_pipeline(model, add_engineered: bool = False) -> Pipeline:
    """Preprocessing + model as a single estimator (what gets saved and deployed)."""
    return Pipeline([
        ("preprocess", build_preprocessor(add_engineered=add_engineered)),
        ("model", model),
    ])


def to_frame(record: dict) -> pd.DataFrame:
    """Turn one input record (e.g. from the app form) into a one-row DataFrame."""
    missing = [f for f in config.FEATURES if f not in record]
    if missing:
        raise ValueError(f"Missing input features: {missing}")
    return pd.DataFrame([{f: record[f] for f in config.FEATURES}])
