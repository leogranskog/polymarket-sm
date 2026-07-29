"""
Extension 6 — Market-level insider flow (Dan's core suggestion).

Rationale: instead of asking "is this WALLET informed," ask "does THIS
MARKET show a late imbalance in large trades that predicts the direction
the market moves before resolution." This is the classic insider-trading
signature: unusual size, late, one-directional, ahead of a move.

For each resolved market (prediction_id) with sufficient trades:
  - split trades into "late window" (last 48h before close) and earlier
  - compute late order-flow imbalance = (late buy volume - late sell
    volume) / late total volume
  - test whether late imbalance predicts the eventual price move from
    the point right before the late window to the close

Excludes TRUE OOS (2025-H2 predictions are dropped by the same cutoff
logic as the wallet-level panel).

Usage: python -m research.market_insider_flow
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pointbiserialr
from config import RAW_DIR, PROC_DIR

TRADES_GLOB = str(RAW_DIR / "trades" / "**" / "*.parquet")
TAB_DIR = Path(__file__).parent / "tables_v2"
FIG_DIR = Path(__file__).parent / "figures_v2"
TAB_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)

# Keep TRUE OOS locked: only use predictions whose last trade is
# before the OOS boundary.
OOS_BOUNDARY = "2025-06-30 23:59:59"
LATE_WINDOW_HOURS = 48
MIN_TRADES = 20


def run():
    print("=" * 65)
    print("  EXTENSION 6 — Market-level insider order-flow (Dan's question)")
    print("=" * 65)
    print(f"  Late window = last {LATE_WINDOW_HOURS}h before resolution")
    print("  Excludes predictions closing after TRUE OOS boundary")

    print("\n  Scanning trades (streaming)...")
    lf = (
        pl.scan_parquet(TRADES_GLOB)
        .filter(pl.col("timestamp") <= pl.lit(OOS_BOUNDARY)
                .str.to_datetime(time_zone="UTC"))
        .select(["prediction_id", "timestamp", "price", "quantity",
                 "taker_bought"])
    )

    # last trade time and price (proxy for "closing" behavior) per token
    close_info = (
        lf.group_by("prediction_id")
        .agg([
            pl.col("timestamp").max().alias("last_ts"),
            pl.len().alias("n_trades"),
        ])
        .filter(pl.col("n_trades") >= MIN_TRADES)
        .collect(engine="streaming")
    )
    print(f"  {len(close_info):,} predictions with >= {MIN_TRADES} trades")

    # join back to get time-to-close per trade, then split late/early
    trades = (
        lf.join(close_info.lazy(), on="prediction_id", how="inner")
        .with_columns([
            (pl.col("last_ts") - pl.col("timestamp")).dt.total_hours()
                .alias("hours_to_close"),
            (pl.col("price") * pl.col("quantity")).alias("usdc"),
        ])
        .with_columns([
            pl.when(pl.col("taker_bought")).then(pl.col("usdc"))
              .otherwise(-pl.col("usdc")).alias("signed_usdc"),
        ])
    )

    late = (
        trades.filter(pl.col("hours_to_close") <= LATE_WINDOW_HOURS)
        .group_by("prediction_id")
        .agg([
            pl.col("signed_usdc").sum().alias("late_net_flow"),
            pl.col("usdc").sum().alias("late_total_flow"),
            pl.len().alias("late_n"),
            pl.col("price").last().alias("late_last_price"),  # ~ closing
        ])
    )
    pre = (
        trades.filter(pl.col("hours_to_close") > LATE_WINDOW_HOURS)
        .group_by("prediction_id")
        .agg([
            pl.col("price").last().alias("pre_late_price"),  # price entering late window
            pl.len().alias("pre_n"),
        ])
    )

    combined = (
        late.join(pre, on="prediction_id", how="inner")
        .filter((pl.col("late_n") >= 5) & (pl.col("pre_n") >= 5))
        .with_columns([
            (pl.col("late_net_flow") / pl.col("late_total_flow"))
                .alias("late_imbalance"),
            (pl.col("late_last_price") - pl.col("pre_late_price"))
                .alias("late_price_move"),
        ])
        .collect(engine="streaming")
        .to_pandas()
    )

    print(f"\n  {len(combined):,} predictions with both pre- and late-window "
          f"trades")

    if len(combined) < 100:
        print("  Not enough data for a robust test.")
        return

    rho, pval = spearmanr(combined["late_imbalance"],
                          combined["late_price_move"])
    print(f"\n  Late order-flow imbalance vs subsequent price move:")
    print(f"    Spearman rho = {rho:+.4f}  (p = {pval:.2e}, n={len(combined):,})")
    print(f"    Interpretation: positive rho => markets with heavy late "
          f"buy imbalance tend to move UP into resolution (consistent "
          f"with informed late flow). Near-zero => no detectable "
          f"insider-flow signature at the market level.")

    # decile analysis: does late imbalance predict direction of move?
    combined["decile"] = pd.qcut(
        combined["late_imbalance"].rank(method="first"), 10, labels=False
    ) + 1
    g = combined.groupby("decile")["late_price_move"].agg(
        ["mean", "count"]).reset_index()
    print(f"\n  Price move by late-imbalance decile:")
    print(g.to_string(index=False))

    out = pd.DataFrame([{
        "n_predictions": len(combined),
        "spearman_rho": rho, "p_value": pval,
        "late_window_hours": LATE_WINDOW_HOURS,
    }])
    out.to_csv(TAB_DIR / "t17_market_insider_flow.csv", index=False)
    g.to_csv(TAB_DIR / "t17b_insider_flow_deciles.csv", index=False)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(g)))
    ax.bar(g["decile"], g["mean"], color=colors, edgecolor="k", lw=0.4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(f"Late ({LATE_WINDOW_HOURS}h) order-flow imbalance decile")
    ax.set_ylabel("Subsequent price move into resolution")
    ax.set_title("Market-level insider-flow signature")
    plt.tight_layout()
    for ext in (".pdf", ".png"):
        plt.savefig(FIG_DIR / f"f9_insider_flow{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close()

    print(f"\n  ✓ -> {TAB_DIR / 't17_market_insider_flow.csv'}, "
          f"{FIG_DIR / 'f9_insider_flow.pdf'}")


if __name__ == "__main__":
    run()