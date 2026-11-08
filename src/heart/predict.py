"""Load the saved model and score new records.

Used by the Streamlit app and usable from the command line:
    python -m heart.predict
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib

from heart import config
from heart.features import to_frame


@dataclass
class Prediction:
    probability: float      # model's estimated probability of the "disease" label
    threshold: float        # decision threshold chosen during training
    flagged: bool           # probability >= threshold
    model_name: str

    @property
    def label(self) -> str:
        return config.POSITIVE_CLASS_NAME if self.flagged else config.NEGATIVE_CLASS_NAME


@lru_cache(maxsize=1)
def load_model(path: Path | str = config.MODEL_PATH):
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle["metadata"]


def predict_one(record: dict, path: Path | str = config.MODEL_PATH) -> Prediction:
    pipe, meta = load_model(path)
    proba = float(pipe.predict_proba(to_frame(record))[0, 1])
    thr = float(meta["threshold"])
    return Prediction(proba, thr, proba >= thr, meta["model_name"])


if __name__ == "__main__":
    example = {"age": 58, "sex": 1, "cp": 0, "trestbps": 140, "chol": 260, "fbs": 0, "restecg": 1,
               "thalach": 130, "exang": 1, "oldpeak": 2.0, "slope": 1, "ca": 2, "thal": 3}
    print(predict_one(example))
