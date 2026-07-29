"""
Extension 5 — Category-level persistence.

Rationale: pooled persistence mixes a sports specialist with a politics
tourist who showed up once for the election. If skill is category-
specific, pooling across categories could mask it. Test: within each
category, does a wallet's category-specific forward CLV predict their
NEXT-period category-specific forward CLV?

Excludes TRUE OOS (2025-H2).

Usage: python -m research.specialization_by_category
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from config import PROC_DIR
from research.pit_features import wallet_trades_lazy, add_clv, build_closing_prices

PIT_DIR = PROC_DIR / "pit"
TAB_DIR = Path(__file__).parent / "tables_v2"
TAB_DIR.mkdir(exist_ok=True)

WINDOWS = [
    ("H2-2023", "2023-06-30", "2023-12-31"),
    ("H1-2024", "2023-12-31", "2024-06-30"),
    ("H2-2024", "2024-06-30", "2024-12-31"),
    ("H1-2025", "2024-12-31", "2025-06-30"),
]
MIN_TRADES_PER_CAT = 5


def clv_by_category(cutoff, horizon):
    closing = build_closing_prices()
    return (
        add_clv(wallet_trades_lazy(cutoff + " 23:59:59",
                                   horizon + " 23:59:59"), closing)
        .group_by(["wallet", "category"])
        .agg([
            pl.len().alias("n"),
            ((pl.col("clv") * pl.col("usdc")).sum() / pl.col("usdc").sum())
                .alias("clv_cat"),
        ])
        .filter(pl.col("n") >= MIN_TRADES_PER_CAT)
        .collect(engine="streaming")
    )


def run():
    print("=" * 65)
    print("  EXTENSION 5 — Category-level persistence")
    print("=" * 65)

    print("  Computing per-category CLV for each window "
          "(streaming, may take a few minutes)...")
    per_window = {}
    for name, cutoff, horizon in WINDOWS:
        print(f"    {name} ({cutoff} -> {horizon})...")
        per_window[name] = clv_by_category(cutoff, horizon).to_pandas()

    all_rows = []
    for (n1, _, _), (n2, _, _) in zip(WINDOWS[:-1], WINDOWS[1:]):
        a, b = per_window[n1], per_window[n2]
        j = a.merge(b, on=["wallet", "category"], suffixes=("", "_next"))
        print(f"\n  {n1} -> {n2}: n={len(j):,} (wallet,category) pairs")

        for cat in sorted(j["category"].dropna().unique()):
            sub = j[j["category"] == cat]
            if len(sub) < 30:
                continue
            rho, pval = spearmanr(sub["clv_cat"], sub["clv_cat_next"])
            all_rows.append({"transition": f"{n1}->{n2}", "category": cat,
                             "n": len(sub), "rho": rho, "p_value": pval})
            print(f"    {cat:<12} n={len(sub):>6,}  rho={rho:+.4f}  "
                  f"p={pval:.3f}")

        # pooled (for comparison, replicating persistence.py at this pairing)
        rho_p, p_p = spearmanr(j["clv_cat"], j["clv_cat_next"])
        all_rows.append({"transition": f"{n1}->{n2}", "category": "ALL_POOLED",
                         "n": len(j), "rho": rho_p, "p_value": p_p})
        print(f"    {'ALL_POOLED':<12} n={len(j):>6,}  rho={rho_p:+.4f}  "
              f"p={p_p:.3f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(TAB_DIR / "t16_category_persistence.csv", index=False)
    with open(TAB_DIR / "t16_category_persistence.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4f",
                caption="CLV persistence within category vs pooled",
                label="tab:cat_persistence"))
    print(f"\n  ✓ -> {TAB_DIR / 't16_category_persistence.csv'}")


if __name__ == "__main__":
    run()