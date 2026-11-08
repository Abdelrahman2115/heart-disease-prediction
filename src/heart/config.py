"""Central configuration: paths, feature groups, and modelling constants.

Keeping every "magic value" here means the rest of the code never hard-codes
column names or file paths, which makes the project easier to change and test.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_PATH = MODELS_DIR / "heart_disease_model.joblib"

# --------------------------------------------------------------------------
# Reproducibility and splitting
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5
CV_REPEATS = 5  # repeated CV -> more stable estimates on a ~240-row train set

# --------------------------------------------------------------------------
# Raw schema (verified against the actual CSV, see reports/data_profile.md)
# --------------------------------------------------------------------------
RAW_TARGET = "target"
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", RAW_TARGET,
]

# Allowed values for coded columns. Anything outside these sets is treated as
# missing rather than silently trusted (see data.clean_data).
VALID_CODES = {
    "sex": {0, 1},
    "cp": {0, 1, 2, 3},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {0, 1, 2},
    "ca": {0, 1, 2, 3},      # number of major vessels (0-3); 4 is not a valid count
    "thal": {1, 2, 3},       # 3 categories in the source data; 0 = unrecorded
}

# --------------------------------------------------------------------------
# Label definition
# --------------------------------------------------------------------------
# In this Kaggle copy of the UCI data, rows with target == 1 show the
# *healthier* clinical profile (higher max heart rate, less ST depression,
# less exercise-induced angina, fewer diseased vessels). The evidence is
# documented in reports/label_direction.md. We therefore define the modelling
# label as has_disease = 1 - target, so that the *positive* class (the one
# recall/sensitivity is measured on) really is "disease".
# Set this to False to model the raw column as-is.
INVERT_TARGET = True
LABEL = "has_disease"
POSITIVE_CLASS_NAME = "disease"
NEGATIVE_CLASS_NAME = "no disease"

# --------------------------------------------------------------------------
# Feature groups -- how each column is treated in preprocessing
# --------------------------------------------------------------------------
# Continuous measurements: impute median, then standardise.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# Binary flags: already 0/1, only need imputation.
BINARY_FEATURES = ["sex", "fbs", "exang"]
# Nominal categories stored as integers: the numbers are labels, not amounts,
# so they must be one-hot encoded.
NOMINAL_FEATURES = ["cp", "restecg", "slope", "thal"]
# Ordinal count (0-3 vessels): the order is meaningful, so keep it numeric.
ORDINAL_FEATURES = ["ca"]

FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES

# Engineered feature (optional, evaluated by cross-validation in train.py)
ENGINEERED_FEATURE = "pct_max_hr"  # thalach / (220 - age)

# Human-readable meaning of every column (used in docs and the app)
FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (4 categories, coded 0-3)",
    "trestbps": "Resting blood pressure on admission (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true)",
    "restecg": "Resting ECG result (coded 0-2)",
    "thalach": "Maximum heart rate achieved during exercise test",
    "exang": "Exercise-induced angina (1 = yes)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of the peak exercise ST segment (coded 0-2)",
    "ca": "Number of major vessels coloured by fluoroscopy (0-3)",
    "thal": "Thallium stress test result (coded 1-3)",
}

# Selection policy: the decision threshold is chosen on out-of-fold training
# predictions so that sensitivity (recall on disease) reaches this target,
# while keeping as much specificity as possible.
TARGET_SENSITIVITY = 0.90
