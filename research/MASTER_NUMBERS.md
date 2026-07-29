======================================================================
  MASTER NUMBERS — single source of truth for the paper
  Generated: 2026-07-27T19:17:01.300824
  Script hash: 3696fee436f6e78c2206732ef6e99a8b14b408a218da73152783244045ce7d87
  Optuna trials per model: 40, seed: 42
======================================================================

======================================================================
  SECTION 1: MODEL COMPARISON (Optuna-tuned, identical procedure for every model)
======================================================================

  train n=1,278  val n=2,661  test n=79,329  features=31  Optuna trials per model=40

  Past-CLV benchmark: AUC=0.5600 [0.5553,0.5647]

  Tuning Logistic regression (40 trials on validation)...
    Logistic regression: AUC=0.4801 [0.4756,0.4845]  DeLong p vs benchmark=0.0000
    Best params: {'C': 0.00010507943638161338}

  Tuning Random forest (40 trials on validation)...
    Random forest: AUC=0.5507 [0.5461,0.5552]  DeLong p vs benchmark=0.0043
    Best params: {'n_estimators': 121, 'max_depth': 12, 'min_samples_leaf': 6}

  Tuning XGBoost (40 trials on validation)...
    XGBoost: AUC=0.5237 [0.5189,0.5284]  DeLong p vs benchmark=0.0000
    Best params: {'n_estimators': 198, 'max_depth': 3, 'learning_rate': 0.01759201053387172, 'subsample': 0.5489324244033761, 'colsample_bytree': 0.9616955700797583}

  Tuning LightGBM (40 trials on validation)...
    LightGBM: AUC=0.5405 [0.5358,0.5448]  DeLong p vs benchmark=0.0000
    Best params: {'n_estimators': 137, 'max_depth': 7, 'learning_rate': 0.022631863448175697, 'subsample': 0.7012000322254772, 'colsample_bytree': 0.6412568987644841, 'min_child_samples': 48}

  ✓ THE frozen model saved -> [project root]\research\models_v2\MASTER_frozen_model.pkl
  (this exact object is reused, unmodified, in Section 4)
  Saved -> [project root]\research\tables_v2\MASTER_table_model_comparison.csv

======================================================================
  SECTION 2: PERSISTENCE (full population, fixed cohort, survivor cohort)
======================================================================

  Fixed cohort (earliest-period wallets): 1,911
  Survivor cohort (present in ALL periods): 180

  H2-2023->H1-2024:
    Full:      n=759   rho=+0.0551
    Cohort:    n=759   rho=+0.0551
    Survivors: n=180   rho=+0.0242

  H1-2024->H2-2024:
    Full:      n=7,817   rho=+0.0695
    Cohort:    n=557   rho=+0.1786
    Survivors: n=180   rho=+0.1468

  H2-2024->H1-2025:
    Full:      n=130,935   rho=+0.0912
    Cohort:    n=427   rho=+0.2195
    Survivors: n=180   rho=+0.3847

  Confirmatory 1 (2025-H2):
    Full:      n=138,321   rho=+0.1720
    Cohort:    n=348   rho=+0.2404
    Survivors: n=180   rho=+0.2743

  Confirmatory 2 (2026-Q1):
    Full:      n=174,566   rho=+0.2967
    Cohort:    n=312   rho=+0.2976
    Survivors: n=180   rho=+0.4662

  Saved -> [project root]\research\tables_v2\MASTER_table_persistence.csv

======================================================================
  SECTION 3: DECILE PORTFOLIO SORT
======================================================================

  Panel: features@2024-12-31, forward through 2025-06-30
    D 1: mean fwd CLV = -0.0081  (n=7933)
    D 2: mean fwd CLV = -0.0017  (n=7933)
    D 3: mean fwd CLV = -0.0034  (n=7933)
    D 4: mean fwd CLV = -0.0040  (n=7933)
    D 5: mean fwd CLV = -0.0054  (n=7933)
    D 6: mean fwd CLV = -0.0032  (n=7932)
    D 7: mean fwd CLV = -0.0025  (n=7933)
    D 8: mean fwd CLV = -0.0011  (n=7933)
    D 9: mean fwd CLV = +0.0011  (n=7933)
    D10: mean fwd CLV = +0.0256  (n=7933)

  D10-D1 spread: +0.0337
  Saved -> [project root]\research\tables_v2\MASTER_table_deciles.csv

======================================================================
  SECTION 4: STALENESS (frozen model from Section 1, refit with identical tuning procedure and ALL genuinely available pre-cutoff data, placebo)
======================================================================

  Loaded frozen model (trained 2026-07-27T19:23:24.864138)

  Confirmatory 1 (2025-H2)
    Refit training windows: [('2023-06-30', '2023-12-31'), ('2023-09-30', '2024-03-31'), ('2023-12-31', '2024-06-30'), ('2024-06-30', '2024-12-31'), ('2024-12-31', '2025-06-30')]
    Frozen AUC:  0.6432 [0.6394,0.6472]
    Refit AUC:   0.6902 [0.6864,0.6940]  (n_train=83,268, params: {'n_estimators': 319, 'max_depth': 7, 'learning_rate': 0.07269377550630607, 'subsample': 0.936655836464787, 'colsample_bytree': 0.5922705404092732, 'min_child_samples': 31})
    Placebo AUC: 0.5011  (clean)

  Confirmatory 2 (2026-Q1)
    Refit training windows: [('2023-06-30', '2023-12-31'), ('2023-09-30', '2024-03-31'), ('2023-12-31', '2024-06-30'), ('2024-06-30', '2024-12-31'), ('2024-12-31', '2025-06-30'), ('2025-06-30', '2025-12-31')]
    Frozen AUC:  0.7015 [0.6981,0.7049]
    Refit AUC:   0.7864 [0.7834,0.7896]  (n_train=200,077, params: {'n_estimators': 498, 'max_depth': 7, 'learning_rate': 0.06465089433352318, 'subsample': 0.9345096954319009, 'colsample_bytree': 0.8265084079758891, 'min_child_samples': 50})
    Placebo AUC: 0.4881  (clean)

  Saved -> [project root]\research\tables_v2\MASTER_table_staleness.csv

======================================================================
  ALL SECTIONS COMPLETE. This file is the only source for
  every number cited in the paper. If any number changes,
  it is because this script was rerun; check the timestamp
  and script hash above against what is cited in the paper.
======================================================================


======================================================================
  ADDENDUM (Section 5) — appended run
  Generated: 2026-07-27T19:39:29.056666
  Script hash of master_numbers.py: 3696fee436f6e78c2206732ef6e99a8b14b408a218da73152783244045ce7d87
======================================================================

======================================================================
  SECTION 5a: MATCHED-HORIZON H3 (re-run on FINAL tiered-CLV panel)
======================================================================
  Building 3-month matched labels (2025-06-30, 2025-09-30]...
  Matched 3-month panel: n=59,065

  Matched 3-month horizon (final panel): coef=+0.0065  p=4.57e-19  n=39,379
  Original 6-month (Confirmatory 1):      coef=+0.0060
  Confirmatory 2 (3-month, actual):        coef=-0.0045

  Matched estimate remains closer to the original 6-month result: reversal is NOT primarily a horizon artifact.

======================================================================
  SECTION 5b: LEAKAGE EXHIBIT, point-in-time side (re-confirmed on final panel)
======================================================================

  Point-in-time LightGBM, final panel: AUC=0.5405 [0.5358,0.5448]
  (Terminal-snapshot side unchanged, definitionally the naive, pre-point-in-time pipeline established at project start: AUC 0.62-0.68)

======================================================================
  ADDENDUM COMPLETE. MASTER_NUMBERS.md now covers every
  number cited anywhere in the paper.
======================================================================
