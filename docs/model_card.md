# Model card: heart disease label classifier

## Model details
- **Type:** L1-regularised logistic regression (`C = 0.3`, `saga` solver) in a scikit-learn
  `Pipeline` with median/mode imputation, standard scaling and one-hot encoding.
- **Decision threshold:** 0.31 on the predicted probability of the *disease* label.
- **Version:** 1.0.0. Artefact: `models/heart_disease_model.joblib` (pipeline + threshold + metadata).

## Intended use
- Education and portfolio demonstration of a complete, leakage-safe ML workflow.
- **Out of scope:** any clinical, screening, triage or insurance use, or any decision about a real person.

## Training data
- Kaggle "Heart Disease Dataset" (a copy of UCI Cleveland): 1,025 rows, of which **302 are unique**.
- The modelling label is `has_disease = 1 − target`, because the raw `target` is inverted relative
  to its documentation (see README, finding 2).
- 138 disease (45.7%) / 164 no disease. 68% male. Ages 29–77.
- Split: stratified 80/20 → 241 train / 61 test, `random_state = 42`.

## Evaluation

| Metric | Out-of-fold (train, threshold 0.31) | Test (threshold 0.31) | Test (threshold 0.50) |
|---|---|---|---|
| Sensitivity (recall) | 0.90 | 0.93 | 0.75 |
| Specificity | 0.71 | 0.73 | 0.85 |
| Precision | 0.72 | 0.74 | 0.81 |
| F1 | 0.80 | 0.83 | 0.78 |
| Accuracy | 0.80 | 0.82 | 0.80 |
| ROC-AUC | 0.91 | 0.90 | 0.90 |
| Missed disease cases (FN) | 11 / 110 | 2 / 28 | 7 / 28 |
| False alarms (FP) | 38 / 131 | 9 / 33 | 5 / 33 |

The threshold was chosen on out-of-fold training predictions, never on the test set.
Test results come from a single evaluation of 61 patients and carry roughly ±0.1 uncertainty.

## Why this threshold
In a screening setting, a missed case (false negative) is usually more harmful than a false alarm,
which is resolved by further testing. The threshold is the highest one that still reaches 90%
sensitivity on training data, so it keeps as much specificity as possible at that target. The
right operating point is a clinical and ethical decision, not a purely technical one.

## Ethical considerations and limitations
- **Not a diagnosis.** The probability means "records like this were usually labelled disease in
  this dataset", not "this person's chance of disease".
- **Under-represented groups:** only 96 women (24 with disease). Subgroup performance cannot be
  estimated reliably and may be worse.
- **Historical, single-source, referral data:** patients were already suspected of heart problems,
  so the disease rate (46%) is far higher than in the general population. Probabilities would be
  badly miscalibrated for a general screening population.
- **Feature availability:** `ca` and `thal` require fluoroscopy and a thallium stress test,
  invasive or specialised tests that would rarely be available at a screening stage. The features
  the model relies on most are not available exactly where screening would happen.
- **Ambiguous category codes** (`cp`, `thal`, `slope`, `restecg`) across sources limit how far
  the coefficients can be interpreted clinically.
