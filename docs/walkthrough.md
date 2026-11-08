# Project walkthrough: every stage, explained

This document explains *why* each decision was made, in the order of the ML lifecycle. Each
stage ends with interview-style questions. Try answering them before reading the code.

---

## 1. Problem understanding

**Task.** Supervised binary classification: learn a mapping from 13 recorded clinical features to
a label (disease / no disease) using examples where the answer is known.

**Predicting a label ≠ diagnosing a patient.** The model learns what *this dataset's labels* look
like. A predicted probability of 0.8 means "records like this were usually labelled disease here",
not "this person has an 80% chance of heart disease". A clinician uses far more information, is
accountable for the decision, and can order more tests.

**Asymmetric errors.**
- *False negative:* a disease case predicted as healthy. In screening, a patient at risk may go without follow-up.
- *False positive:* a healthy person flagged. This costs anxiety and extra tests, but is usually recoverable.

This is why we optimise **sensitivity** and choose a threshold, instead of reporting accuracy alone.

> **Q:** Why is accuracy alone a poor metric here, even though the classes are fairly balanced?
> **Q:** What does a calibrated probability of 0.8 mean, and what does it not mean?

---

## 2. Data understanding

| Column | Meaning | Type here |
|---|---|---|
| `age` | Age (years) | numeric |
| `sex` | 1 = male, 0 = female | binary |
| `cp` | Chest pain type, coded 0–3 | nominal → one-hot |
| `trestbps` | Resting blood pressure (mm Hg) | numeric |
| `chol` | Serum cholesterol (mg/dl) | numeric |
| `fbs` | Fasting blood sugar > 120 mg/dl | binary |
| `restecg` | Resting ECG result, coded 0–2 | nominal → one-hot |
| `thalach` | Max heart rate in exercise test | numeric |
| `exang` | Exercise-induced angina | binary |
| `oldpeak` | ST depression, exercise vs rest | numeric |
| `slope` | Slope of peak exercise ST segment, coded 0–2 | nominal → one-hot |
| `ca` | Major vessels coloured by fluoroscopy (0–3) | ordinal |
| `thal` | Thallium stress test result, coded 1–3 | nominal → one-hot |

**Numbers that are not quantities.** `cp = 3` is not "three times" `cp = 1`; the integers are
labels. Feeding them as numbers to a linear model or KNN would impose a false order and spacing,
so we **one-hot encode** them. `ca` is a genuine count (0–3 vessels), so its order is meaningful
and it stays numeric.

**What the codes are associated with in this data** (from `04_categorical_disease_rates.png`):
`cp = 0` has by far the highest disease rate, consistent with "asymptomatic" chest pain in the
original UCI coding, where silent ischaemia found on testing is a strong sign. `thal = 3` is
associated with disease and `thal = 2` with no disease (consistent with reversible defect and
normal, respectively). Because sources document these codes inconsistently, the project treats
them as codes and states its interpretation as an association.

> **Q:** Which columns would you one-hot encode, and why not `ca`?

---

## 3. Exploratory data analysis

Run `python -m heart.eda`. Key observations (`reports/data_profile.md`, `reports/figures/`):

- **Class balance:** 46% disease / 54% no disease. Mild imbalance; we stratify splits anyway.
- **Strongest associations (Spearman ρ):** `ca` +0.49, `cp` −0.46, `exang` +0.44,
  `thalach` −0.43, `oldpeak` +0.42, `thal` +0.41. `fbs` is almost unrelated (+0.03).
- **Weak features are not useless features:** `chol` and `trestbps` correlate weakly *on their
  own*, but could still help in combination. That's why we don't drop features based on
  correlation alone.
- **Spearman, not Pearson:** it uses ranks, so it is robust to skewed variables and to the coded
  categoricals, and it captures monotonic (not just linear) relationships.
- **Outliers:** flagged with the 1.5×IQR rule (e.g. cholesterol 564, ST depression 6.2). All are
  physiologically possible, so they are **kept**. Deleting extreme-but-real patients removes
  exactly the cases a clinical model must handle.

> **Q:** You find a cholesterol value of 564. What would you check before deciding whether to remove it?

---

## 4. Data cleaning: the three big findings

1. **Duplicates (723 of 1,025 rows).** If duplicates are split across train and test, the model is
   tested on patients it has already seen. This is **data leakage**, and it inflates every metric.
   We keep one copy of each row.
2. **The label is inverted.** Disease is clinically associated with lower max heart rate, more ST
   depression, exercise angina and more blocked vessels. In this file, `target = 0` has all of
   those. We therefore model `has_disease = 1 − target` (`00_label_direction.png`). The
   cross-check: after the flip there are 138 disease cases, the same as the original UCI data.
   A unit test asserts this direction so it can't silently regress.
3. **Invalid codes.** `ca = 4` and `thal = 0` are not valid values; they most likely mean
   "not recorded". They are set to `NaN` and imputed. That is better than inventing a category or
   deleting the patient.

> **Q:** Why does leakage from duplicates make test scores *optimistic* rather than just noisy?
> **Q:** What would have happened to "sensitivity" if we had not fixed the label direction?

---

## 5. Preprocessing and leakage prevention

`src/heart/features.py` builds a `ColumnTransformer`:

| Group | Steps | Why |
|---|---|---|
| numeric | median imputer → `StandardScaler` | median is robust to outliers; scaling puts features on comparable scales for distance/margin/regularised models |
| binary | most-frequent imputer | already 0/1 |
| nominal | most-frequent imputer → `OneHotEncoder(handle_unknown="ignore")` | codes are labels; unknown codes at prediction time are ignored rather than crashing |
| ordinal | most-frequent imputer → `StandardScaler` | keeps the order |

**Why a Pipeline matters.** Imputers and scalers *learn* statistics (medians, means, standard
deviations). If you fit them on the full dataset before splitting, information from the test set
leaks into training. Inside a `Pipeline`, `cross_validate` and `GridSearchCV` re-fit the whole
pipeline on each training fold, so validation data never influences preprocessing. The same
fitted pipeline is saved and deployed, so the app applies *identical* transformations.

> **Q:** Give a concrete example of leakage caused by scaling before splitting.

---

## 6. Feature engineering and selection

**Engineered feature:** `pct_max_hr = thalach / (220 − age)`, the share of the age-predicted
maximum heart rate a patient reached. It's clinically motivated, since raw max heart rate falls
with age.

**Result:** it didn't help (mean CV ROC-AUC change −0.006, `reports/feature_engineering_cv.csv`).
Models can already combine `age` and `thalach`, so the ratio adds little. It stays in the code as a
documented experiment and is switched off.

**Selection:** we removed no features by hand. L1 regularisation in logistic regression
performs selection *inside* cross-validation: it shrank `age`, `fbs` and `restecg` to exactly zero.
Hand-removing features using the full dataset would itself be a form of leakage.

> **Q:** Why did L1 set `age` to zero even though age correlates with disease? (Hint: correlated features.)

---

## 7. Model development

| Model | How it works | Why try it here |
|---|---|---|
| **Baseline** (majority/prior) | Ignores features | The bar every model must clear (ROC-AUC 0.5) |
| **Logistic regression** | Weighted sum → sigmoid → probability; fitted by maximising likelihood with a penalty | Strong on small tabular data, well-calibrated, coefficients are interpretable as log-odds |
| **KNN** | Label = vote of the *k* most similar training patients | Simple, non-parametric; shows why scaling matters |
| **Decision tree** | Learns if/else splits that best separate classes | Very readable; prone to overfitting |
| **Random forest** | Many trees on bootstrap samples with random feature subsets, averaged | Reduces a single tree's variance; handles interactions |
| **SVM (RBF)** | Maximum-margin boundary; RBF kernel allows curved boundaries | Strong on small, scaled datasets |
| **Gradient boosting** | Adds shallow trees sequentially, each fixing the current errors | Often the strongest on tabular data, but more hyperparameters to overfit |

> **Q:** Why does KNN need scaling but a random forest doesn't?

---

## 8. Evaluation metrics

With TP/FP/TN/FN from the confusion matrix (positive = disease):

- **Accuracy** = (TP+TN)/all. Misleading when errors have different costs.
- **Precision** = TP/(TP+FP). Of those flagged, how many have the disease label?
- **Recall / sensitivity** = TP/(TP+FN). Of disease cases, how many did we catch?
- **Specificity** = TN/(TN+FP). Of healthy cases, how many did we correctly clear?
- **F1** = harmonic mean of precision and recall.
- **ROC-AUC** = probability that a random disease case gets a higher score than a random healthy
  one. It is threshold-independent, so we use it to *rank models*, then choose the threshold separately.

**The trade-off.** Lowering the threshold flags more people: sensitivity ↑, specificity ↓
(`07_threshold_tradeoff.png`). Our model on the test set:

| Threshold | Sensitivity | Specificity | Missed cases | False alarms |
|---|---|---|---|---|
| 0.50 | 0.75 | 0.85 | 7 | 5 |
| **0.31** | **0.93** | **0.73** | **2** | 9 |

Five fewer missed cases cost four more false alarms. The threshold was picked on **out-of-fold
training predictions** (the highest threshold reaching 90% sensitivity), never on the test set.

> **Q:** A colleague tunes the threshold to maximise test-set F1. What's wrong with that?

---

## 9. Cross-validation and tuning

- **Stratified 80/20 hold-out split.** The test set is used once.
- **Repeated stratified 5-fold CV (5 repeats = 25 fits)** on the training set. With ~240 rows, a
  single 5-fold split is noisy; repeating with different shuffles averages that noise out and gives
  a standard deviation.
- **GridSearchCV** with small grids, scored by ROC-AUC. Large grids on small data mostly find
  settings that fit the noise in the folds.
- **Tuned vs untuned:** tuning helped most where defaults overfit (decision tree 0.75 → 0.86,
  KNN 0.88 → 0.91) and barely changed logistic regression (0.909 → 0.911), which was already
  well-regularised.
- **Caveat:** re-scoring a tuned model with CV on the same data is slightly optimistic. The
  rigorous fix is *nested* CV, which is a good extension.

> **Q:** Why can't you use the test set to pick hyperparameters, even "just once"?

---

## 10. Interpretation

- **Permutation importance** (model-agnostic): shuffle one feature in the test set and measure the
  ROC-AUC drop. Top features: `ca`, `cp`, `thal`, `sex`, `oldpeak`.
- **Odds ratios** (logistic regression): `exp(coefficient)`. For example, `exang` has an odds ratio
  of 1.83: exercise-induced angina multiplies the odds of the disease label by about 1.8, holding
  the other features fixed. Numeric features are standardised, so their odds ratio is per one
  standard deviation.
- **Per-patient explanations:** for a linear model, each feature contributes
  `coefficient × scaled value` to the log-odds, so any single prediction can be broken into
  additive parts (`reports/interpretation.md`).
- **Limits:** importance is association, not causation. Correlated features share credit.

> **Q:** Explain an odds ratio of 2.8 for `ca` to a non-technical stakeholder.

---

## 11. Final model selection

Rule: highest tuned CV ROC-AUC, but **within one standard error of the best, prefer the simpler
model** (the "one-standard-error rule"). Random Forest scored 0.918 and logistic regression 0.911,
a gap of 0.007, which is within noise. Logistic regression wins on interpretability, calibration
and simplicity.

Note that SVM scored best on the *test* set. Choosing it *because* of that would be using the test
set for model selection, and its 0.906 would then no longer be an unbiased estimate.

**Strengths:** transparent, stable, fast, probabilities are meaningful.
**Weaknesses:** linear in the transformed features (no automatic interactions); relies on `ca`
and `thal`, which come from specialised tests.

---

## 12. Deployment

- `models/heart_disease_model.joblib` stores the *whole* pipeline (preprocessing + model), the
  threshold and metadata, so the app can't apply different preprocessing from training.
- `app/streamlit_app.py` collects inputs, calls `heart.predict.predict_one`, and shows the
  probability with clear non-diagnostic wording.
- `tests/` covers schema, cleaning, leakage, label direction, metrics and prediction.

**Next steps if you extend it:** nested CV; bootstrap confidence intervals for test metrics;
calibration curves; subgroup metrics by sex; validation on the other UCI sites
(Hungary, Switzerland, VA Long Beach); a GitHub Actions workflow that runs `pytest`.
