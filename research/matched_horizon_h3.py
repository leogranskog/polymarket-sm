"""
Referee-mandated diagnostic #2: is the H3 sign reversal a genuine
temporal instability, or an artifact of window 1 having a 6-month
label horizon while window 2 has only 3 months?

Re-runs H3 on window 1's cutoff (2025-06-30) but restricts the label
window to the FIRST 3 months only (2025-07 to 2025-09), matching
window 2's horizon exactly. If this matched-horizon coefficient is
close to window 2's sign/magnitude rather than the original 6-month
result, the "instability" finding is (at least partly) a horizon
artifact, not evidence of a genuinely unstable market effect.

Usage: python -m research.matched_horizon_h3
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import polars as pl
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
from config import PROC_DIR
from research.pit_features import wallet_trades_lazy, add_clv, build_closing_prices

PIT_DIR = PROC_DIR / "pit"
TAB_DIR = Path(__file__).parent / "tables_v2"
TAB_DIR.mkdir(exist_ok=True)

MIN_TRADES_FORWARD = 10


def build_matched_labels(cutoff="2025-06-30", horizon_end="2025-09-30"):
    """3-month label window, matching window 2's horizon exactly,
    using the SAME cutoff as window 1 (2025-06-30)."""
    print(f"  Building matched-horizon labels: ({cutoff}, {horizon_end}]...")
    closing = build_closing_prices()
    wt = (
        add_clv(
            wallet_trades_lazy(cutoff + " 23:59:59", horizon_end + " 23:59:59"),
            closing,
        )
        .with_columns((pl.col("price") * pl.col("quantity")).alias("usdc"))
        .group_by("wallet")
        .agg([
            pl.len().alias("fwd_n_trades"),
            (pl.col("clv") * pl.col("usdc")).sum().alias("_clv_usdc_sum"),
            pl.col("usdc").sum().alias("_usdc_sum"),
        ])
        .filter(pl.col("_usdc_sum") > 0)
        .with_columns(
            (pl.col("_clv_usdc_sum") / pl.col("_usdc_sum")).alias("fwd_clv_vw")
        )
        .filter(pl.col("fwd_n_trades") >= MIN_TRADES_FORWARD)
        .filter(pl.col("fwd_clv_vw").is_finite())
        .select(["wallet", "fwd_n_trades", "fwd_clv_vw"])
        .collect(engine="streaming")
    )
    q75 = wt["fwd_clv_vw"].quantile(0.75)
    wt = wt.with_columns(
        (pl.col("fwd_clv_vw") >= q75).cast(pl.Int32).alias("label_skilled")
    )
    print(f"  ✓ {len(wt):,} labeled wallets, top-quartile threshold={q75:.4f}")
    return wt


def run():
    print("=" * 65)
    print("  MATCHED-HORIZON H3 DIAGNOSTIC")
    print("  (is the sign reversal a genuine effect or a horizon artifact?)")
    print("=" * 65)

    matched_labels = build_matched_labels()
    feats_path = PIT_DIR / "features_asof_2025-06-30.parquet"
    if not feats_path.exists():
        print(f"  ⚠ {feats_path} not found. Run the panel build first.")
        return

    feats = pl.read_parquet(feats_path)
    df = feats.join(matched_labels, on="wallet", how="inner").to_pandas()
    print(f"  Matched panel: {len(df):,} wallets")

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

    print(f"\n  Matched 3-month horizon (2025-07 to 2025-09), n={len(d):,}:")
    print(f"    specialist coef = {coef:+.4f}   p = {pval:.2e}")

    print(f"\n  ── Comparison table ──────────────────────────────────")
    print(f"    Original window 1 (6-month, 2025-H2): coef=+0.0060, p=3.6e-13")
    print(f"    Matched window 1 (3-month, THIS RUN): coef={coef:+.4f}, p={pval:.2e}")
    print(f"    Window 2 (3-month, 2026-Q1):          coef=-0.0045, p<1e-6")

    # Interpretation
    orig = 0.0060
    win2 = -0.0045
    dist_to_orig = abs(coef - orig)
    dist_to_win2 = abs(coef - win2)

    print(f"\n  Distance from matched estimate to original 6-month: "
          f"{dist_to_orig:.4f}")
    print(f"  Distance from matched estimate to window 2:          "
          f"{dist_to_win2:.4f}")

    if dist_to_win2 < dist_to_orig:
        print(f"\n  *** RESULT: matched-horizon estimate is CLOSER to "
              f"window 2 than to the original 6-month result. ***")
        print(f"  This suggests the H3 'reversal' is (at least partly) "
              f"a HORIZON ARTIFACT: shortening window 1's label period "
              f"alone moves the coefficient toward window 2's value, "
              f"without any change in calendar time. Sections 5.9, 6.1, "
              f"and 6.2 need to be rewritten to attribute the instability "
              f"primarily to horizon-length sensitivity rather than "
              f"temporal/market evolution.")
    else:
        print(f"\n  *** RESULT: matched-horizon estimate remains closer "
              f"to the original 6-month result than to window 2. ***")
        print(f"  This suggests the reversal is NOT primarily a horizon "
              f"artifact -- the original interpretation (genuine "
              f"temporal instability) is more defensible, though the "
              f"horizon difference should still be disclosed as a "
              f"co-occurring design limitation.")

    out = pd.DataFrame([{
        "specification": "matched_3mo_window1",
        "coef": coef, "p_value": pval, "n": len(d),
        "dist_to_original_6mo": dist_to_orig,
        "dist_to_window2": dist_to_win2,
    }])
    out.to_csv(TAB_DIR / "t23_matched_horizon_h3.csv", index=False)
    print(f"\n  ✓ saved -> {TAB_DIR / 't23_matched_horizon_h3.csv'}")


if __name__ == "__main__":
    run()