VERIFIED NUMBERS FOR ICAIF DRAFT
All values computed fresh from the corrected tiered-CLV panel.
No hardcoded or estimated values below.

======================================================================
  SECTION 1: FOUR-MODEL COMPARISON (corrected tiered-CLV panel)
======================================================================
  train n=1,278  val n=2,661  test n=79,329  features=31

  Past-CLV benchmark: AUC=0.5600 [0.5553,0.5647]
  Logistic regression: AUC=0.4605 [0.4560,0.4647]  DeLong p vs benchmark=0.0000
  Random forest: AUC=0.5377 [0.5330,0.5423]  DeLong p vs benchmark=0.0000
  XGBoost: AUC=0.5195 [0.5150,0.5237]  DeLong p vs benchmark=0.0000
  LightGBM (tuned): AUC=0.5258 [0.5212,0.5302]  DeLong p vs benchmark=0.0000
    (saved as lgbm_primary.pkl for downstream use)

  Saved -> C:\Users\leogr\polymarket-sm\research\tables_v2\icaif_table2_model_comparison.csv

======================================================================
  SECTION 2: PERSISTENCE (full population + fixed cohort)
======================================================================

  Fixed cohort size: 1,911

  H2-2023->H1-2024: full n=759 rho=+0.0551   cohort n=759 rho=+0.0551

  H1-2024->H2-2024: full n=7,817 rho=+0.0695   cohort n=557 rho=+0.1786

  H2-2024->H1-2025: full n=130,935 rho=+0.0912   cohort n=427 rho=+0.2195

  Confirmatory 1 (2025-H2): full n=138,321 rho=+0.1720   cohort n=348 rho=+0.2404

  Confirmatory 2 (2026-Q1): full n=174,566 rho=+0.2967   cohort n=312 rho=+0.2976

  Saved -> C:\Users\leogr\polymarket-sm\research\tables_v2\icaif_table_persistence.csv

  Decile D10-D1 spread (forward CLV): +0.0337

======================================================================
  SECTION 3: MODEL STALENESS (frozen vs refit vs placebo)
======================================================================

  Confirmatory 1 (2025-H2)
    Frozen AUC:  0.6101
    Refit AUC:   0.7064
    Placebo AUC: 0.5022

  Confirmatory 2 (2026-Q1)
    Frozen AUC:  0.6847
    Refit AUC:   0.7858
    Placebo AUC: 0.5033

  Saved -> C:\Users\leogr\polymarket-sm\research\tables_v2\icaif_table3_staleness.csv

======================================================================
  ALL SECTIONS COMPLETE. Cross-check every number above
  against the draft before submitting.
======================================================================
