"""Candidate models and their hyperparameter search spaces.

Each entry pairs an *untuned* estimator (sensible defaults) with a grid that
GridSearchCV explores. Grids are kept small on purpose: with ~240 training
rows, large grids mostly find settings that fit the noise in the CV folds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from heart.config import RANDOM_STATE


@dataclass
class ModelSpec:
    name: str
    estimator: object
    param_grid: dict = field(default_factory=dict)
    summary: str = ""


def get_baseline() -> ModelSpec:
    """Always predicts the majority class. Any real model must beat this."""
    return ModelSpec(
        name="Baseline (majority class)",
        estimator=DummyClassifier(strategy="prior"),
        summary="Ignores the features; predicts the most common class.",
    )


def get_model_specs() -> list[ModelSpec]:
    rs = RANDOM_STATE
    return [
        ModelSpec(
            "Logistic Regression",
            LogisticRegression(max_iter=5000, random_state=rs),
            {
                "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                "model__l1_ratio": [0.0, 1.0],  # 0 = L2 (ridge), 1 = L1 (lasso)
                "model__solver": ["saga"],
            },
            "Weighted sum of features passed through a sigmoid; coefficients are interpretable as log-odds.",
        ),
        ModelSpec(
            "K-Nearest Neighbors",
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            "Classifies a patient by the labels of the k most similar patients; needs scaled features.",
        ),
        ModelSpec(
            "Decision Tree",
            DecisionTreeClassifier(random_state=rs),
            {
                "model__max_depth": [2, 3, 4, 5, 6, None],
                "model__min_samples_leaf": [1, 5, 10, 20],
                "model__criterion": ["gini", "entropy"],
            },
            "Learns a sequence of if/else splits; easy to read, but a single tree overfits easily.",
        ),
        ModelSpec(
            "Random Forest",
            RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=-1),
            {
                "model__max_depth": [3, 5, 8, None],
                "model__min_samples_leaf": [1, 3, 5, 10],
                "model__max_features": ["sqrt", 0.5],
            },
            "Averages many decorrelated trees trained on bootstrap samples; reduces a single tree's variance.",
        ),
        ModelSpec(
            "SVM (RBF)",
            SVC(kernel="rbf", probability=True, random_state=rs),
            {
                "model__C": [0.1, 0.3, 1, 3, 10],
                "model__gamma": ["scale", 0.01, 0.03, 0.1],
            },
            "Finds the widest-margin boundary; the RBF kernel allows curved boundaries.",
        ),
        ModelSpec(
            "Gradient Boosting",
            GradientBoostingClassifier(random_state=rs),
            {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [1, 2, 3],
                "model__subsample": [0.8, 1.0],
            },
            "Adds shallow trees one at a time, each correcting the previous ensemble's errors.",
        ),
    ]
