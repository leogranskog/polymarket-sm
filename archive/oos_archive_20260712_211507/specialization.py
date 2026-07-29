"""
Result 4 — Specialization: do category specialists have more skill?

(a) Specialist (top-tercile category HHI) vs generalist forward CLV
(b) OLS with controls + HC3 robust SEs:
    fwd_clv ~ specialist + log(n_trades) + log(volume) + frac_maker + past_clv
(c) Within dominant-category forward CLV: specialists in THEIR category
    vs generalists in the same category.

Usage: python -m research.specialization
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from config import PROC_DIR
from research.pit_features import (wallet_trades_lazy, add_clv,
                                   build_closing_prices)

PIT_DIR = PROC_DIR / "pit"
OUT_TAB = Path(__file__).parent / "tables_v2"
OUT_FIG = Path(__file__).parent / "figures_v2"

# test window (keep OOS locked)
CUTOFF, HORIZON = "2024-12-31", "2025-06-30"


def dominant_category(cutoff: str) -> pl.DataFrame:
    """Each wallet's most-traded category as of the cutoff."""
    return (
        wallet_trades_lazy(None, cutoff + " 23:59:59")
        .group_by(["wallet", "category"]).agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .group_by("wallet")
        .agg(pl.col("category").first().alias("dominant_cat"))
        .collect(engine="streaming")
    )


def forward_clv_by_category(cutoff: str, horizon: str) -> pl.DataFrame:
    closing = build_closing_prices()
    return (
        add_clv(wallet_trades_lazy(cutoff + " 23:59:59",
                                   horizon + " 23:59:59"), closing)
        .group_by(["wallet", "category"])
        .agg([
            pl.len().alias("n"),
            ((pl.col("clv") * pl.col("usdc")).sum() / pl.col("usdc").sum())
                .alias("fwd_clv_cat"),
        ])
        .filter(pl.col("n") >= 5)
        .collect(engine="streaming")
    )


def run():
    print("=" * 60)
    print("  RESULT 4 — SPECIALIZATION AND SKILL")
    print("=" * 60)

    feats  = pl.read_parquet(PIT_DIR / f"features_asof_{CUTOFF}.parquet")
    labels = pl.read_parquet(PIT_DIR / f"labels_{CUTOFF}_to_{HORIZON}.parquet")
    df = feats.join(labels, on="wallet", how="inner").to_pandas()
    print(f"  panel: {len(df):,} wallets")

    # (a) specialist vs generalist
    t1, t2 = df["category_hhi"].quantile([1/3, 2/3])
    df["group"] = np.where(df["category_hhi"] >= t2, "specialist",
                  np.where(df["category_hhi"] <= t1, "generalist", "mid"))
    g = (df[df["group"] != "mid"]
         .groupby("group")["fwd_clv_vw"]
         .agg(["mean", "median", "count", "std"]))
    from scipy.stats import mannwhitneyu, ttest_ind
    spec = df.loc[df["group"] == "specialist", "fwd_clv_vw"]
    gen  = df.loc[df["group"] == "generalist", "fwd_clv_vw"]
    t_p  = ttest_ind(spec, gen, equal_var=False).pvalue
    u_p  = mannwhitneyu(spec, gen).pvalue
    print("\n  (a) Forward CLV, specialist vs generalist:")
    print(g)
    print(f"      Welch t-test p={t_p:.2e}   Mann-Whitney p={u_p:.2e}")

    # (b) OLS with controls, HC3 robust SEs
    import statsmodels.formula.api as smf
    d = df[df["group"] != "mid"].copy()
    d["specialist"] = (d["group"] == "specialist").astype(int)
    d["log_trades"] = np.log1p(d["n_trades"])
    d["log_volume"] = np.log1p(d["total_volume"])
    for c in ["past_clv_vw", "frac_maker"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    ols = smf.ols("fwd_clv_vw ~ specialist + log_trades + log_volume "
                  "+ frac_maker + past_clv_vw", data=d).fit(cov_type="HC3")
    print("\n  (b) OLS with controls (HC3 robust SEs):")
    print(ols.summary().tables[1])
    with open(OUT_TAB / "t7_specialization_ols.tex", "w") as f:
        f.write(ols.summary().as_latex())

    # (c) within dominant category
    print("\n  (c) Within-category test (streaming scans)...")
    dom = dominant_category(CUTOFF).to_pandas()
    fwd = forward_clv_by_category(CUTOFF, HORIZON).to_pandas()
    m = (d[["wallet", "specialist"]]
         .merge(dom, on="wallet")
         .merge(fwd, left_on=["wallet", "dominant_cat"],
                right_on=["wallet", "category"]))
    wc = m.groupby("specialist")["fwd_clv_cat"].agg(["mean", "count"])
    from scipy.stats import ttest_ind as tt
    p_wc = tt(m.loc[m.specialist == 1, "fwd_clv_cat"],
              m.loc[m.specialist == 0, "fwd_clv_cat"],
              equal_var=False).pvalue
    print(f"      In own dominant category — "
          f"specialists: {wc.loc[1,'mean']:+.4f} (n={int(wc.loc[1,'count'])}) "
          f"vs generalists: {wc.loc[0,'mean']:+.4f} "
          f"(n={int(wc.loc[0,'count'])}), p={p_wc:.2e}")

    summary = pd.DataFrame([{
        "spec_fwd_clv": spec.mean(), "gen_fwd_clv": gen.mean(),
        "welch_p": t_p, "mannwhitney_p": u_p,
        "ols_specialist_coef": ols.params["specialist"],
        "ols_specialist_p": ols.pvalues["specialist"],
        "within_cat_spec": wc.loc[1, "mean"],
        "within_cat_gen": wc.loc[0, "mean"], "within_cat_p": p_wc,
    }])
    summary.to_csv(OUT_TAB / "t7_specialization.csv", index=False)
    print(f"\n  ✓ tables -> {OUT_TAB}")


if __name__ == "__main__":
    run()