# Insurance Claim Prediction - ML Walkthrough

End-to-end data science exercise using a synthetic NZ motor insurance dataset.  
**Goal:** Predict whether a customer will make a claim (`has_claim`: 0 or 1).

---

## Dataset

Three tables:

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

## Step 1 - Load & Explore

```python
import pandas as pd
import matplotlib.pyplot as plt

customers = pd.read_csv('data/customers.csv')
policies  = pd.read_csv('data/policies.csv')
claims    = pd.read_csv('data/claims.csv')

# Basic structure
print(customers.shape)       # (300, 6)
print(customers.dtypes)
print(customers.describe())

# Missing values
print(customers.isnull().sum())
print(policies.isnull().sum())
print(claims.isnull().sum())
# approved_amount has ~121 nulls - expected, as pending/rejected claims have no approved amount
```

**Key findings:**
- `approved_amount` has ~121 nulls - intentional, not missing data. Pending/rejected claims have no approved amount yet.
- Target variable distribution: ~55% have a claim, ~45% do not → reasonably balanced, no need for resampling.

---

## Step 2 - Feature Engineering

Aggregate policy and claim data to the customer level so each row = one customer.

```python
# Fix nulls in approved_amount - fill with 0 (no payout)
claims['approved_amount'] = claims['approved_amount'].fillna(0)

# Aggregate to customer level
claim_features = (
    policies
    .merge(claims, on='policy_id', how='left')
    .groupby('customer_id')
    .agg(
        total_policies=('policy_id',       'nunique'),
        total_claimed =('claimed_amount',  'sum'),
        total_approved=('approved_amount', 'sum'),
        n_claims      =('claim_id',        'count'),
        avg_premium   =('annual_premium',  'mean')
    )
    .reset_index()
)

# Merge with customer info
df = customers.merge(claim_features, on='customer_id', how='left')

# Target variable: 1 if customer has made at least one claim
df['has_claim'] = (df['n_claims'] > 0).astype(int)

print(df['has_claim'].value_counts())
# 1 → 166,  0 → 134
```

---

## Step 3 - Preprocessing

```python
# Encode categorical: region → dummy variables
df = pd.get_dummies(df, columns=['region'])

# Drop columns that would cause data leakage or are not useful as features
# n_claims / total_claimed / total_approved directly encode the target → leakage!
drop_cols = ['customer_id', 'name', 'has_claim', 'n_claims', 'total_claimed', 'total_approved']

X = df.drop(columns=drop_cols)
y = df['has_claim']

print(X.shape)  # (300, 10)
print(X.columns.tolist())
```

> **Data leakage warning:** `n_claims`, `total_claimed`, and `total_approved` are derived directly from the target. Including them would give the model the answer - resulting in suspiciously perfect scores (Accuracy 1.0, AUC 1.0). Always remove leaky features before training.

---

## Step 4 - Train / Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)  # (240, 10)
print(X_test.shape)   # (60, 10)
```

---

## Step 5 - Model Training & Evaluation

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")   # 0.600
print(f"AUC:      {roc_auc_score(y_test, y_prob_rf):.3f}")    # 0.583
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
```

### XGBoost

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.3f}")  # 0.533
print(f"AUC:      {roc_auc_score(y_test, y_prob_xgb):.3f}")   # 0.508
print(confusion_matrix(y_test, y_pred_xgb))
```

### Model Comparison

| Model | Accuracy | AUC |
|---|---|---|
| Random Forest | 0.600 | 0.583 |
| XGBoost | 0.533 | 0.508 |

**Random Forest outperformed XGBoost here.** XGBoost generally performs better with larger datasets and careful hyperparameter tuning. With only 300 rows, Random Forest is more stable.

---

## Step 6 - Feature Importance

```python
import pandas as pd
import matplotlib.pyplot as plt

feat_imp = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

feat_imp.plot(kind='bar', figsize=(10, 4))
plt.title('Feature importance - Random Forest')
plt.tight_layout()
plt.show()

print(feat_imp)
```

**Results:**

| Feature | Importance |
|---|---|
| risk_score | 0.287 |
| age | 0.195 |
| avg_premium | 0.195 |
| years_licensed | 0.183 |
| total_policies | 0.038 |
| region_* | ~0.02 each |

**Interpretation:**
- `risk_score` is the strongest predictor - aligns with domain knowledge (higher risk = more likely to claim)
- `age`, `avg_premium`, `years_licensed` contribute similarly
- `region` has minimal predictive power with this dataset - could be dropped or replaced with richer geographic data

---

## Key Takeaways

| Concept | What we did |
|---|---|
| EDA | Checked shape, dtypes, nulls, target distribution |
| Feature engineering | Aggregated claim data to customer level |
| Data leakage | Identified and removed target-derived features |
| Encoding | One-hot encoded `region` |
| Model selection | Compared Random Forest vs XGBoost |
| Evaluation | Used Accuracy, AUC, confusion matrix |
| Interpretation | Feature importance to explain model decisions |

---

## Dependencies

```
pandas
scikit-learn
xgboost
matplotlib
```
