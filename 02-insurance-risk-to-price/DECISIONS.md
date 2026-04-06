# Decisions: v1 to v3

This file documents what changed across three versions of this project and why. Not a tutorial, just an honest record of decisions, mistakes, and fixes.

---

## v1 to v2: Getting the methodology right

### Leakage was everywhere

KMeans was fit on the full dataset, so val and test segment assignments were contaminated by information they should not have seen. The fix is obvious in hindsight: fit on train only, transform val and test separately.

Same problem with threshold tuning. Finding the optimal threshold on the test set and then reporting test cost as a result is circular. Threshold goes on val. Test gets opened once at the end.

These are easy mistakes to make when you are focused on making the pipeline run. They are also the kind of mistakes that make results look better than they are.

### GLM needs feature scaling

Without scaling, the Poisson and Gamma solvers struggle to differentiate between segments and converge to near-identical predictions across all groups. Adding StandardScaler before GLM fitting fixed convergence and produced segment-differentiated premiums.

### Class imbalance is not just a modelling problem

With a 5% claim rate, an unweighted model learns to predict no claim for almost everyone and still gets 95% accuracy. Sample weights combining exposure and class frequency are necessary to make the minority class (claimants) actually count.

### Monte Carlo without correlated shocks underestimates tail risk

Independent Bernoulli draws per policy miss the point of stress testing. A bad weather event or economic shock hits many policies at once. Adding a shared lognormal market shock factor per simulation run widens the tail and produces a more conservative reserve estimate.

### AUC stability matters more than AUC value

A single train/test split AUC is sensitive to the random seed. 5-fold stratified CV with std of 0.0016 across folds is more useful than a single number, because it tells you the result is not lucky.

---

## v2 to v3: Making pricing defensible

### ML-only pricing is not defensible

v2 produced a technically correct model but a practically useless pricing output. ML premiums were 5 to 10x higher than GLM premiums for the same segments. The reason: ML severity predictions are not calibrated to the same scale as GLM, and class imbalance upweighting pushes probabilities high.

In real insurance pricing, GLM is the regulatory anchor. ML captures non-linear interactions that GLM misses, but it should adjust the GLM price, not replace it.

### The fix: GLM base, ML ratio adjustment

```python
ML_ratio     = (ML_PurePremium / GLM_PurePremium).clip(0.5, 2.0)
FinalPremium = GLM_PurePremium * ML_ratio * loading_factor
```

Clipping to [0.5, 2.0] means ML can move the price up to 2x in either direction but cannot produce extreme outputs. This keeps the model auditable and the premiums in a realistic range.

### Severity outliers inflate premiums multiplicatively

ML severity predictions without an upper bound interact badly with claim probabilities. A policy with a 20% claim probability multiplied by an outlier severity of $50,000 produces a $10,000 premium that bears no relationship to the actual risk distribution.

Capping severity at the 99th percentile of training claims ($18,822) before computing premiums removes this behaviour.

### What the numbers looked like before and after

| Segment | v2 Final premium | v3 Final premium |
|---------|-----------------|-----------------|
| 0 | $2,192 | $265 |
| 1 | $1,117 | $300 |
| 2 | $3,057 | $534 |
| 3 | $967 | $277 |

v3 premiums are in a range that could plausibly reflect real motor insurance pricing. v2 premiums could not.

---

## Limitations that remain

These are not fixed in v3. Documented here because ignoring them would be worse than acknowledging them.

**Segment pricing order is inconsistent.** Segment 1 prices higher than Segment 0 despite nearly identical claim rates. The ML ratio is distorting relative pricing across segments. A monotonicity constraint or isotonic regression on the final premium would fix this.

**GLM and ML severity are not calibrated to each other.** They are trained independently and their outputs are not on the same scale. The clip is a workaround, not a solution. Proper calibration of ML severity against GLM on the validation set would be the right fix.

**BonusMalus dominates, which creates a cold-start problem.** New customers have no BonusMalus history. The model has no reliable way to price them. A separate GLM-only model using only observable features would be needed for new business.

**Monte Carlo tail is likely underestimated.** The lognormal shock (std=0.08) is not fit to historical loss data. Real catastrophe distributions have heavier tails. Fitting to industry loss data or using an EVT tail model would improve reserve accuracy.

**No cross-validation on the full pipeline.** CV covers the ML classifier only. Clustering and GLM are fit once. Wrapping the full pipeline in CV would give more stable end-to-end performance estimates.

---

## The thing that kept coming up

Threshold, severity cap, ML ratio clip, Monte Carlo loading. None of these are model parameters in the traditional sense. They are all business decisions expressed as numbers.

The model tells you the probability. The business decides what to do with it. Getting that boundary right is most of the work.
