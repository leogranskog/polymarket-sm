"""
Referee-mandated diagnostic #1 (CORRECTED): how correlated is CLV
with realized P&L against the ACTUAL resolution outcome, not against
another price proxy?

Realized P&L per share = direction * (actual_outcome - entry_price),
where actual_outcome in {0, 1} is the true resolved payoff -- NOT the
closing price. This is the correct test of whether CLV (measured
against last-trade price) collapses into being indistinguishable from
realized P&L.

Usage: python -m research.clv_diagnostic
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import polars as pl
import numpy as np
from scipy.stats import pearsonr, spearmanr
from research.pit_features import wallet_trades_lazy, add_clv, build_closing_prices
from config import RAW_DIR

TRADES_GLOB = str(RAW_DIR / "trades" / "**" / "*.parquet")


def run(horizon_hours: int = None):
    label = "UNCORRECTED (last trade)" if horizon_hours is None else \
            f"CORRECTED (VWAP, {horizon_hours}h-{horizon_hours*3}h window)"
    print(f"  Testing CLV definition: {label}")
    print(f"  (against ACTUAL resolution outcome, not another price proxy)")

    if horizon_hours is None:
        closing = build_closing_prices()
    else:
        closing = build_closing_prices(horizon_hours=horizon_hours)

    # Pull the actual resolution outcome (winner: True/False) per token,
    # completely independent of the closing-price computation.
    outcomes = (
        pl.scan_parquet(TRADES_GLOB)
        .filter(
            pl.col("timestamp") <= pl.lit("2025-09-30 23:59:59")
                .str.to_datetime(time_zone="UTC")
        )
        .filter(
            pl.col("timestamp") > pl.lit("2025-06-30 23:59:59")
                .str.to_datetime(time_zone="UTC")
        )
        .group_by("prediction_id")
        .agg(pl.col("winner").drop_nulls().first().alias("actual_outcome"))
        .filter(pl.col("actual_outcome").is_not_null())
        .with_columns(pl.col("actual_outcome").cast(pl.Float64))
        .collect(engine="streaming")
    )
    print(f"  {len(outcomes):,} tokens with a known resolution outcome")

    wt = (
        add_clv(
            wallet_trades_lazy("2025-06-30 23:59:59", "2025-09-30 23:59:59"),
            closing,
        )
        .join(outcomes.lazy(), on="prediction_id", how="inner")
        .with_columns(
            (pl.col("direction") *
             (pl.col("actual_outcome") - pl.col("price"))
            ).alias("realized_pnl_per_share")
        )
        .collect(engine="streaming")
        .to_pandas()
    )

    if len(wt) == 0:
        print("  ⚠ No trades with both CLV and known outcome -- check data")
        return

    valid = wt["clv"].notna() & wt["realized_pnl_per_share"].notna()

    r_pearson, p1 = pearsonr(wt.loc[valid, "clv"],
                              wt.loc[valid, "realized_pnl_per_share"])
    r_spear, p2 = spearmanr(wt.loc[valid, "clv"],
                             wt.loc[valid, "realized_pnl_per_share"])

    print(f"\n  n = {valid.sum():,} trades")
    print(f"  CLV (vs {label.split('(')[1].rstrip(')')} price) vs "
          f"REALIZED P&L (vs actual outcome):")
    print(f"    Pearson r  = {r_pearson:.4f}  (p={p1:.2e})")
    print(f"    Spearman r = {r_spear:.4f}  (p={p2:.2e})")

    print(f"\n  Interpretation guide:")
    print(f"    r > 0.85  -> CLV is essentially P&L; framing collapses,")
    print(f"                 must be reported plainly in the paper")
    print(f"    r 0.5-0.85 -> partial overlap; discuss explicitly")
    print(f"    r < 0.5   -> reasonable separation; CLV framing defensible")

    return {"r_pearson": r_pearson, "r_spearman": r_spear, "n": int(valid.sum())}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--horizon-hours", type=int, default=None)
    args = p.parse_args()
    run(horizon_hours=args.horizon_hours)