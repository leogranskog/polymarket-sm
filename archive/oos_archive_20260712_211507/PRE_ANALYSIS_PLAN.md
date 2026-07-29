# Pre-Analysis Plan — TRUE Out-of-Sample Confirmatory Test
Date frozen: 2026-07-12
FROZEN_HASH: 6b01b2d1fa12493e70a8884714d71f4f2af00eee45b69ba6b1cc44f5a49aa52d

## Held-out window
2025-H2 (features@2025-06-30, labels through 2025-12-31). Not accessed
for any tuning, threshold selection, or model comparison prior to this
document.

## Confirmatory hypotheses and PASS criteria

**H1 — No wallet-level CLV persistence.**
  Estimate: Spearman rho, H1-2025 fwd CLV -> H2-2025 fwd CLV.
  PASS (replicates null): |rho| < 0.05 and 95% CI contains 0.
  FAIL (persistence emerges): CI excludes 0 and |rho| >= 0.05.

**H2 — Past CLV reverses (negative coefficient).**
  Estimate: OLS coefficient on past_clv_vw, fwd_clv_vw ~ ... , TRUE OOS.
  PASS: coefficient is negative AND p < 0.05 (Holm-corrected).
  FAIL: coefficient is positive, or negative but not significant.

**H3 — Category specialization predicts lower forward-CLV loss.**
  Estimate: OLS coefficient on `specialist`, same spec as frozen run.
  PASS: coefficient is positive AND p < 0.05 (Holm-corrected).
  FAIL: coefficient is non-positive, or positive but not significant.

**H4 — Cross-sectional behavioral ML prediction of skill (frozen LightGBM*).**
  Estimate: AUC on TRUE-OOS wallets, frozen model from ml_pipeline_v2.
  PASS (replicates null): 95% CI contains 0.5.
  FAIL (signal emerges): CI excludes 0.5 with AUC > 0.5.

**H5 — Market-level informed late order flow.**
  Estimate: AUC of late_imbalance -> actual resolution outcome,
  event-clustered bootstrap CI, TRUE-OOS window (2025-H2 predictions
  resolving in this window only).
  PASS (replicates positive finding): CI excludes 0.5 with AUC > 0.5.
  FAIL: CI contains 0.5.

## Multiple-testing correction
Holm-Bonferroni applied across the p-values for H1, H2, H3, H5 (H4's
criterion is CI-based, not a p-value, and is evaluated separately).

## Effect-size reporting
AUC-based findings (H4, H5) additionally reported as a probit-link
Cohen's-d-equivalent: d = sqrt(2) * Phi^-1(AUC).

## Commitment
No further changes to feature engineering, labels, model architecture,
or hyperparameters will be made after this document is saved and hashed.
Whatever true_oos_final.py reports is what is reported in the paper,
including any hypothesis that fails.