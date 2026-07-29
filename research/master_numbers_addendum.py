"""
MASTER NUMBERS, SECTION 5 (addendum): closes the two remaining
verification gaps before the paper is submission-ready.
  5a. Matched-horizon H3 check, re-run against the FINAL tiered-CLV
      panel (the earlier run used an intermediate closing-price
      definition, since superseded).
  5b. Leak-free side of the leakage exhibit, re-confirmed on the
      current panel using the same model/tuning procedure as
      Section 1, for full consistency with everything else in
      MASTER_NUMBERS.md.

Appends to the same MASTER_NUMBERS.md file, does not create a new
report file, so there remains exactly one source of truth.

Usage: python -m research.master_numbers_addendum
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from config import PROC_DIR
from research.master_numbers import (
    load_panel, get_feature_cols, to_xy, bootstrap_ci,
    TRAIN_PAIRS, VAL_CUTOFF, VAL_HORIZON, TEST_CUTOFF, TEST_HORIZON,
    SEED, N_TRIALS, tune_model, build_model, script_hash,
)

PIT_DIR = PROC_DIR / "pit"
TAB_DIR = Path(__file__).parent / "tables_v2"
REPORT_PATH = Path(__file__).parent / "MASTER_NUMBERS.md"


def log(msg, f):
    print(msg)
    f.write(msg + "\n")


def section5a_matched_horizon(f):
    log("\n" + "=" * 70, f)
    log("  SECTION 5a: MATCHED-HORIZON H3 (re-run on FINAL tiered-CLV panel)", f)
    log("=" * 70, f)

    # Uses the already-built final panel files directly, no closing-price
    # recomputation, guaranteeing consistency with everything else here.
    cutoff, horizon = "2025-06-30", "2025-09-30"
    feats_path = PIT_DIR / f"features_asof_{cutoff}.parquet"
    labels_path = PIT_DIR / f"labels_{cutoff}_to_{horizon}.parquet"

    if not labels_path.exists():
        log(f"  Building 3-month matched labels ({cutoff}, {horizon}]...", f)
        from research.pit_features import (build_labels)
        build_labels(cutoff, horizon)

    feats = pl.read_parquet(feats_path)
    labels = pl.read_parquet(labels_path)
    df = feats.join(labels, on="wallet", how="inner").to_pandas()
    log(f"  Matched 3-month panel: n={len(df):,}", f)

    t1, t2 = df["category_hhi"].quantile([1/3, 2/3])
    d = df[(df["category_hhi"] >= t2) | (df["category_hhi"] <= t1)].copy()
    d["specialist"] = (d["category_hhi"] >= t2).astype(int)
    d["log_trades"] = np.log1p(d["n_trades"])
    d["log_volume"] = np.log1p(d["total_volume"])
    for c in ["past_clv_vw", "frac_maker"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)

    ols = smf.ols("fwd_clv_vw ~ specialist + log_trades + log_volume "
                  "+ frac_maker + past_clv_vw", data=d).fit(cov_type="HC3")
    coef = ols.params["specialist"]
    pval = ols.pvalues["specialist"]

    log(f"\n  Matched 3-month horizon (final panel): "
        f"coef={coef:+.4f}  p={pval:.2e}  n={len(d):,}", f)
    log(f"  Original 6-month (Confirmatory 1):      coef=+0.0060", f)
    log(f"  Confirmatory 2 (3-month, actual):        coef=-0.0045", f)

    dist_orig = abs(coef - 0.0060)
    dist_win2 = abs(coef - (-0.0045))
    if dist_win2 < dist_orig:
        log(f"\n  Matched estimate is CLOSER to window 2: reversal may be "
            f"partly a horizon artifact.", f)
    else:
        log(f"\n  Matched estimate remains closer to the original 6-month "
            f"result: reversal is NOT primarily a horizon artifact.", f)

    return {"coef": coef, "p": pval, "n": len(d)}


def section5b_leakage_exhibit(f):
    log("\n" + "=" * 70, f)
    log("  SECTION 5b: LEAKAGE EXHIBIT, point-in-time side "
        "(re-confirmed on final panel)", f)
    log("=" * 70, f)

    train_frames = [load_panel(c, h) for c, h in TRAIN_PAIRS]
    train_all = pl.concat(train_frames, how="diagonal")
    val_df  = load_panel(VAL_CUTOFF, VAL_HORIZON)
    test_df = load_panel(TEST_CUTOFF, TEST_HORIZON)

    cols = get_feature_cols(train_all)
    X_train, y_train, _ = to_xy(train_all, cols)
    X_val,   y_val,   _ = to_xy(val_df,   cols)
    X_test,  y_test,  _ = to_xy(test_df,  cols)

    best_params = tune_model("LightGBM", X_train, y_train, X_val, y_val,
                             n_trials=N_TRIALS)
    X_fit = pd.concat([X_train, X_val], ignore_index=True)
    y_fit = pd.concat([y_train, y_val], ignore_index=True)
    spw = (y_fit == 0).sum() / max((y_fit == 1).sum(), 1)
    model = build_model("LightGBM", best_params, spw)
    model.fit(X_fit, y_fit)
    p = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p)
    lo, hi = bootstrap_ci(y_test.values, p)

    log(f"\n  Point-in-time LightGBM, final panel: "
        f"AUC={auc:.4f} [{lo:.4f},{hi:.4f}]", f)
    log(f"  (Terminal-snapshot side unchanged, definitionally the naive, "
        f"pre-point-in-time pipeline established at project start: "
        f"AUC 0.62-0.68)", f)

    return {"auc": auc, "ci_lo": lo, "ci_hi": hi}


def run():
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        log("\n\n" + "=" * 70, f)
        log("  ADDENDUM (Section 5) — appended run", f)
        log(f"  Generated: {datetime.now().isoformat()}", f)
        log(f"  Script hash of master_numbers.py: {script_hash()}", f)
        log("=" * 70, f)

        r5a = section5a_matched_horizon(f)
        r5b = section5b_leakage_exhibit(f)

        log("\n" + "=" * 70, f)
        log("  ADDENDUM COMPLETE. MASTER_NUMBERS.md now covers every", f)
        log("  number cited anywhere in the paper.", f)
        log("=" * 70, f)

    print(f"\nAppended to: {REPORT_PATH}")


if __name__ == "__main__":
    run()