# Insurance Claim Prediction

Binary classification model predicting whether a motor insurance customer will make a claim, built on synthetic NZ insurance data.

---

## Business Context

Identifying high-risk customers at the policy level helps insurers:
- Price premiums more precisely
- Manage claims exposure proactively
- Allocate underwriting resources effectively

---

## Business Constraints

- **Interpretability:** Model decisions must be explainable to underwriters → SHAP values used
- **Threshold tuning:** Default 0.5 threshold optimises accuracy, not business cost. In insurance, missing a high-risk customer (false negative) is typically more costly than a false alarm → lower threshold considered
- **Data limitations:** 300-row synthetic dataset - real deployment would require significantly more data

---

## Dataset

Three tables across `data/`:

| Table | Rows | Description |
|---|---|---|
| `customers.csv` | 300 | Customer demographics and risk score |
| `policies.csv` | ~392 | Policy details per customer |
| `claims.csv` | ~277 | Claims made against policies |

**Schema**
```
customers:  customer_id, name, age, region, years_licensed, risk_score
policies:   policy_id, customer_id, product_type, start_date, end_date, annual_premium, status
claims:     claim_id, policy_id, claim_date, claim_type, claimed_amount, approved_amount, status
```

---

## Pipeline

### Step 1 - Load & Explore
Check shape, dtypes, descriptive stats, and missing values across all three tables.

Key findings:
- `approved_amount` has ~121 nulls - intentional, pending/rejected claims have no payout → filled with 0
- Target distribution: ~55% have a claim, ~45% do not → reasonably balanced

### Step 2 - Feature Engineering
Aggregate policy and claim data to customer level, then create binary target.

```python
claim_features = (
    policies
    .merge(claims, on='policy_id', how='left')
    .groupby('customer_id')
    .agg(
        total_policies=('policy_id',      'nunique'),
        total_claimed =('claimed_amount', 'sum'),
        total_approved=('approved_amount','sum'),
        n_claims      =('claim_id',       'count'),
        avg_premium   =('annual_premium', 'mean')
    )
    .reset_index()
)

df['has_claim'] = (df['n_claims'] > 0).astype(int)
```

### Step 3 - Preprocessing & Time-Based Split

One-hot encode `region`, then split by policy `start_date`:

```
Cutoff: 2023-01-01
Train: 229 samples (57% positive)
Test:  105 samples (52% positive)
→ Stable class distribution across time periods
```

> **Data leakage:** `n_claims`, `total_claimed`, and `total_approved` are derived directly from the target - including them inflates scores to a perfect 1.0. Always remove leaky features before training.

### Step 4 - Cross-Validation

5-fold stratified CV on the full dataset for a stable performance baseline.

```
CV AUC: ~0.62 ± 0.03
```

### Step 5 - Hyperparameter Tuning

GridSearchCV over Random Forest parameters, scored by AUC.

```python
param_grid = {
    'n_estimators':    [100, 200],
    'max_depth':       [3, 5, None],
    'min_samples_leaf':[1, 5, 10]
}
```

### Step 6 - Model Evaluation

| Model | Accuracy | AUC |
|---|---|---|
| Random Forest (tuned) | ~0.60 | ~0.66 |
| XGBoost | ~0.55 | ~0.60 |

Random Forest outperformed XGBoost on this dataset. With only ~300 rows, Random Forest is more stable. Model selection should always be driven by experimentation, not assumption.

### Step 7 - Threshold Tuning

Default 0.5 threshold optimises accuracy. In insurance, false negatives (missing a high-risk customer) are more costly than false positives, so we lower the threshold to increase recall.

```python
# Find threshold that maximises recall while keeping precision ≥ 0.5
valid = precision >= 0.5
best_threshold = thresholds[valid][np.argmax(recall[valid])]
```

### Step 8 - Feature Importance

| Feature | Importance |
|---|---|
| risk_score | ~0.29 |
| age | ~0.20 |
| avg_premium | ~0.19 |
| years_licensed | ~0.18 |
| total_policies | ~0.04 |
| region_* | ~0.02 each |

`risk_score` is the strongest predictor - consistent with domain knowledge. Region features have minimal predictive power and could be dropped or replaced with richer geographic data.

### Step 9 - SHAP Explainability

SHAP values explain both global feature importance and individual predictions - essential for regulatory auditability in insurance.

```python
explainer   = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test)
sv = shap_values[:, :, 1]  # class 1 (has claim)

shap.summary_plot(sv, X_test)
shap.force_plot(explainer.expected_value[1], sv[0], X_test.iloc[0])
```

---

## Limitations & Next Steps

- **Dataset size:** 300 rows is too small for production - metrics are unstable
- **Synthetic data:** Real insurance data would include vehicle type, telematics, cross-insurer claims history, etc.
- **Feature engineering:** More signals possible - claim frequency per policy year, seasonal patterns, claim type mix
- **Model monitoring:** Production models need drift detection over time
- **Fairness:** Ensure model does not proxy-discriminate via age or region in a regulated context

---

## Dependencies

```
pandas
scikit-learn
xgboost
shap
matplotlib
```
