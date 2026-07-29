"""
Robustness check for Extension 6: is the informed-flow effect driven
entirely by the extreme D1 (heavy late-sell-imbalance) tail, or does it
hold as a graded effect across the distribution?

Refits the outcome logit (event-clustered SEs) on:
  (a) full sample (replicates market_insider_flow_v2, sanity check)
  (b) excluding decile 1
  (c) excluding deciles 1 and 10 (both extreme tails)
  (d) rank-based: late_imbalance replaced by its within-sample percentile
      rank, which is robust to the tails dominating a raw-value fit

Usage: python -m research.insider_robustness
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from sklearn.metrics import roc_auc_score
from research.market_insider_flow_v2 import build_combined, cluster_bootstrap_ci

TAB_DIR = Path(__file__).parent / "tables_v2"
TAB_DIR.mkdir(exist_ok=True)


def fit_logit(df, label):
    X = sm.add_constant(df["late_imbalance"])
    y = df["win"]
    if y.nunique() < 2:
        print(f"  {label}: degenerate outcome, skipped")
        return None
    logit = sm.Logit(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": df["event_id"]}, disp=0
    )
    auc = roc_auc_score(y, df["late_imbalance"])
    lo, hi, n_valid = cluster_bootstrap_ci(
        df, lambda d: roc_auc_score(d["win"], d["late_imbalance"]),
        n_boot=300,
    )
    coef = logit.params["late_imbalance"]
    pval = logit.pvalues["late_imbalance"]
    print(f"  {label:<28} n={len(df):>7,}  AUC={auc:.4f} "
          f"[{lo:.4f},{hi:.4f}]  coef={coef:+.4f}  p={pval:.2e}")
    return {"config": label, "n": len(df), "auc": auc,
            "ci_lo": lo, "ci_hi": hi, "coef": coef, "p_clustered": pval}


def run():
    print("=" * 65)
    print("  ROBUSTNESS — is the informed-flow effect tail-driven?")
    print("=" * 65)

    print("\n  Rebuilding combined dataset (streaming)...")
    combined = build_combined()
    combined["decile"] = pd.qcut(
        combined["late_imbalance"].rank(method="first"), 10, labels=False
    ) + 1
    # rank-transform, 0-1 scale, robust to outlier magnitudes
    combined["imbalance_rank"] = (
        combined["late_imbalance"].rank(pct=True)
    )

    rows = []

    print("\n  (a) Full sample:")
    r = fit_logit(combined, "Full sample")
    if r: rows.append(r)

    print("\n  (b) Excluding decile 1 (heavy late-sell tail):")
    ex_d1 = combined[combined["decile"] != 1]
    r = fit_logit(ex_d1, "Excl. D1")
    if r: rows.append(r)

    print("\n  (c) Excluding deciles 1 and 10 (both tails):")
    ex_d1_d10 = combined[~combined["decile"].isin([1, 10])]
    r = fit_logit(ex_d1_d10, "Excl. D1+D10")
    if r: rows.append(r)

    print("\n  (d) Rank-transformed late_imbalance (full sample):")
    rank_df = combined.copy()
    rank_df["late_imbalance"] = rank_df["imbalance_rank"]
    r = fit_logit(rank_df, "Rank-transformed")
    if r: rows.append(r)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(TAB_DIR / "t19_insider_robustness.csv", index=False)
    with open(TAB_DIR / "t19_insider_robustness.tex", "w") as f:
        f.write(df_out.to_latex(index=False, float_format="%.4f",
                caption="Robustness: informed-flow effect excluding "
                        "distributional tails",
                label="tab:insider_robust"))

    print(f"\n  Interpretation: if (b)/(c) retain a similar sign and a "
          f"clustered CI still excluding 0.5, the effect is graded across "
          f"the distribution, not an artifact of a few extreme "
          f"observations. If AUC collapses toward 0.5 once D1 is dropped, "
          f"the effect is concentrated in the sell-imbalance tail only -- "
          f"report it as such, it is still a real but narrower finding.")
    print(f"\n  ✓ -> {TAB_DIR / 't19_insider_robustness.csv'}")


if __name__ == "__main__":
    run()
    