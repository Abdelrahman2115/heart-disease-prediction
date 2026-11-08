# Model interpretation

Selected model: **Logistic Regression**

## Permutation importance (test set, ROC-AUC drop, 50 repeats)

| Feature | Mean drop | Std |
|---|---|---|
| `ca` | 0.056 | 0.034 |
| `cp` | 0.043 | 0.019 |
| `thal` | 0.036 | 0.017 |
| `sex` | 0.019 | 0.008 |
| `oldpeak` | 0.013 | 0.010 |
| `slope` | 0.011 | 0.009 |
| `thalach` | 0.009 | 0.005 |
| `exang` | 0.008 | 0.008 |
| `trestbps` | 0.004 | 0.003 |
| `chol` | 0.000 | 0.001 |
| `age` | 0.000 | 0.000 |
| `fbs` | 0.000 | 0.000 |
| `restecg` | 0.000 | 0.000 |

## Logistic regression odds ratios

Numeric features are standardised, so their odds ratio is per **one standard deviation**. One-hot features compare that category against the others. An odds ratio of 1.0 means the L1 penalty removed the feature.

| Transformed feature | Coefficient | Odds ratio |
|---|---|---|
| `nom__cp_0.0` | +1.228 | 3.42 |
| `ord__ca` | +1.024 | 2.79 |
| `nom__thal_3.0` | +0.916 | 2.50 |
| `bin__exang` | +0.604 | 1.83 |
| `bin__sex` | +0.500 | 1.65 |
| `num__oldpeak` | +0.469 | 1.60 |
| `nom__slope_1.0` | +0.383 | 1.47 |
| `nom__thal_2.0` | -0.228 | 0.80 |
| `num__thalach` | -0.203 | 0.82 |
| `nom__slope_2.0` | -0.114 | 0.89 |
| `num__trestbps` | +0.094 | 1.10 |
| `num__chol` | +0.024 | 1.02 |
| `num__age` | +0.000 | 1.00 |
| `bin__fbs` | +0.000 | 1.00 |
| `nom__restecg_1.0` | +0.000 | 1.00 |
| `nom__restecg_0.0` | +0.000 | 1.00 |
| `nom__cp_3.0` | +0.000 | 1.00 |
| `nom__cp_2.0` | +0.000 | 1.00 |
| `nom__cp_1.0` | +0.000 | 1.00 |
| `nom__slope_0.0` | +0.000 | 1.00 |
| `nom__restecg_2.0` | +0.000 | 1.00 |
| `nom__thal_1.0` | +0.000 | 1.00 |

## Example explanation: highest-risk test patient

Predicted probability of the *disease* label: **0.99** (dataset label: disease)

| Feature | Contribution to log-odds |
|---|---|
| `ord__ca` | +2.61 |
| `nom__cp_0.0` | +1.23 |
| `num__oldpeak` | +1.17 |
| `nom__thal_3.0` | +0.92 |
| `nom__slope_1.0` | +0.38 |
| `num__trestbps` | +0.10 |

## Example explanation: lowest-risk test patient

Predicted probability of the *disease* label: **0.04** (dataset label: no disease)

| Feature | Contribution to log-odds |
|---|---|
| `ord__ca` | -0.72 |
| `num__oldpeak` | -0.42 |
| `nom__thal_2.0` | -0.23 |
| `num__thalach` | -0.20 |
| `nom__slope_2.0` | -0.11 |
| `num__trestbps` | -0.06 |

## Caveats

- Importance shows what **this model** relies on to reproduce **this dataset's labels**; it is association, not causation.
- Correlated features share credit: shuffling one can understate its importance when a correlated partner carries the same information (e.g. `slope` and `oldpeak`).
- With 61 test patients, importance estimates are noisy; see the standard deviations.
