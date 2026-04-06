# Insurance Risk Framework

**Clustering → ML → Monte Carlo → Pricing**

A full pricing pipeline for motor insurance using the freMTPL2 dataset (678,013 French policies).

---

## High-Level Summary

### What this does

Takes a portfolio of motor insurance policies and produces a risk-adjusted premium for each customer.

The pipeline answers four business questions in sequence:

1. Who are your customers? (Clustering)
2. Which ones are likely to claim? (ML)
3. If they all behave badly at once, how bad does it get? (Monte Carlo)
4. What price covers that risk while staying competitive? (Pricing)

### Results

| Metric | Value |
|--------|-------|
| Portfolio expected loss | $138M |
| Reserve requirement (95th pct) | $156M |
| Capital buffer needed | $18M |
| Loading factor applied | 1.129x |
| Cost reduction from threshold tuning | 48.6% |

**Pricing by risk segment (Final premium):**

| Segment | Actual claim rate | Final premium |
|---------|------------------|---------------|
| 0 | 6% | $265 |
| 1 | 5% | $300 |
| 2 | 6% | $534 |
| 3 | 4% | $277 |

### Limitations (business)

**This is a research framework, not a production pricing system.** Specific gaps:

- Premiums are not validated against actual market rates. The $265-534 range may be uncompetitive or below cost in specific regions.
- The framework does not account for regulatory floors, coverage type, or policy tenure, including regulatory floors, coverage type, and policy tenure.
- Segment 1 prices higher than Segment 0 despite nearly identical claim rates. This is an artefact of the ML adjustment layer and would require manual review before use.
- New customers have no BonusMalus history, which is the strongest predictor in the model. Pricing new business with this framework would require a separate cold-start approach.
- Monte Carlo assumes correlated claims follow a lognormal shock structure. Real catastrophe events (floods, hail storms) have heavier tails than this model produces.

---

## Technical Summary

### Architecture

```
Raw policy data
    → Clustering (KMeans, 4 segments, fit on train only)
    → GLM baseline (Poisson frequency × Gamma severity)
    → ML layer (GradientBoosting, exposure + class-imbalance weights)
    → Threshold tuning (business cost, val set only)
    → SHAP interpretability
    → Monte Carlo (1,000 simulations, correlated market shock)
    → Pricing (GLM base × ML ratio × MC loading)
```

### Pricing structure

GLM is the anchor. ML provides a ratio adjustment on top, clipped to [0.5, 2.0].

```python
ML_ratio      = (ML_PurePremium / GLM_PurePremium).clip(0.5, 2.0)
BlendPremium  = GLM_PurePremium * ML_ratio
FinalPremium  = BlendPremium * loading_factor   # loading_factor = pct95 / avg_loss
```

GLM stays as the regulatory-friendly base. ML captures non-linear interactions (age × driving history × BonusMalus) that GLM misses.

### Key decisions and why

**Threshold = 0.64, not 0.5**
Missing a high-risk customer costs 5x more than a false alarm in this cost structure. Tuned on val set, evaluated on held-out test. Cost reduction: 48.6%.

**Severity cap at 99th percentile ($18,822)**
Without a cap, ML severity predictions on extreme outliers inflate premiums by an order of magnitude when multiplied by claim probability. Cap applied to train distribution only.

**ML ratio clipped to [0.5, 2.0]**
Without clipping, ML-only premiums reached $2,694 vs GLM at $245 for the same segment. That is a 10x gap. The clip keeps ML as an adjustment, not a replacement.

**Monte Carlo with systematic market shock**
Independent Bernoulli draws underestimate tail risk because they miss correlated events (weather, economic shocks). A shared lognormal shock factor is applied per simulation run.

### Model performance

| Metric | Value |
|--------|-------|
| CV AUC mean | 0.6521 |
| CV AUC std | 0.0016 |
| Test AUC (held-out) | 0.6511 |
| Optimal threshold | 0.64 |

AUC of 0.65 is expected for claim prediction on imbalanced data (5% claim rate). Stability across folds (±0.0016) matters more than the absolute number here.

Top predictors (SHAP): BonusMalus (0.297), DrivAge (0.181), VehAge (0.166)

### Version history

**v1** pipeline existed but had multiple validity issues: KMeans fit on full dataset (leakage), threshold found on test set (evaluation contamination), no GLM feature scaling, no class imbalance handling, no SHAP, Monte Carlo without correlated shock.

**v2** fixed all methodology issues. Correct train/val/test discipline throughout. Added SHAP, CV, systematic MC shock. Remaining problem: pricing was 100% ML-based, producing premiums 10x higher than GLM with no regulatory justification.

**v3** fixed pricing structure. GLM is now the base. ML provides a bounded ratio adjustment. Severity outliers capped. Final premiums are in a defensible range.

### Limitations (technical) and how to fix them

**1. Segment pricing order is inconsistent**
Segment 1 ($300) prices higher than Segment 0 ($265) despite similar claim rates. The ML ratio adjustment is distorting relative pricing across segments.
Fix: apply isotonic regression or a monotonicity constraint on the final premium relative to predicted claim rate.

**2. GLM and ML severity are not calibrated to the same scale**
GLM Gamma severity and ML GBM severity are trained independently. Their outputs are not on the same scale, which is why raw ML premiums are 5-7x higher than GLM even after capping.
Fix: calibrate ML severity predictions against GLM outputs on the validation set before computing the ratio.

**3. Monte Carlo tail is likely underestimated**
The lognormal shock (std=0.08) is a reasonable approximation but not fit to historical loss data. Real catastrophe loss distributions have heavier tails.
Fix: fit the shock distribution to historical industry loss data or use a Pareto/EVT tail model.

**4. BonusMalus dominates, which creates cold-start risk**
New customers have no BonusMalus score. Any prediction for them is unreliable.
Fix: build a separate GLM-only model for new customers using only observable features (vehicle, area, demographics).

**5. No cross-validation on the full pipeline**
CV is applied to the ML classifier only. Clustering and GLM are fit once on train. Segment assignments could shift under different train splits.
Fix: wrap the full pipeline in a cross-validation loop to get stable estimates of end-to-end performance.
