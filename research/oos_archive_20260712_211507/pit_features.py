"""
Point-in-Time (PIT) Feature Builder + CLV Engine.  LEAK-FREE.

Every feature is computed from raw trades using ONLY data <= cutoff.
Labels are computed from trades strictly AFTER the cutoff.

Panel design (expanding window, quarterly train cutoffs):
  train:      features@2023-06-30 -> labels (2023-07..2023-12)
  train:      features@2023-09-30 -> labels (2023-10..2024-03)
  train:      features@2023-12-31 -> labels (2024-01..2024-06)
  validation: features@2024-06-30 -> labels (2024-07..2024-12)
  test:       features@2024-12-31 -> labels (2025-01..2025-06)
  TRUE OOS:   features@2025-06-30 -> labels (2025-07..2025-12)  [touch ONCE]

Usage:
    python run.py pit
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import polars as pl
from pathlib import Path
from config import RAW_DIR, PROC_DIR

TRADES_GLOB = str(RAW_DIR / "trades" / "**" / "*.parquet")
PIT_DIR = PROC_DIR / "pit"
PIT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TRADES_FEATURES = 30   # min historical trades to be scored
MIN_TRADES_FORWARD  = 10   # min forward trades to be labeled

PANEL = [
    # (features_cutoff, label_window_end, role)
    # No train label window may overlap the validation label window
    # (2024-07..2024-12): last train cutoff is 2023-12-31 (labels to 2024-06-30).
    ("2023-06-30", "2023-12-31", "train"),
    ("2023-09-30", "2024-03-31", "train"),
    ("2023-12-31", "2024-06-30", "train"),
    ("2024-06-30", "2024-12-31", "validation"),
    ("2024-12-31", "2025-06-30", "test"),
    ("2025-06-30", "2025-12-31", "TRUE_OOS"),
]


# ── 1. Closing prices per prediction token (memory-safe, per-year) ───────────

def build_closing_prices(force: bool = False) -> pl.DataFrame:
    out = PIT_DIR / "closing_prices.parquet"
    if out.exists() and not force:
        return pl.read_parquet(out)

    print("  Building closing-price table (memory-safe, per-year)...")
    years = sorted((RAW_DIR / "trades").glob("year=*"))

    partials = []
    for ydir in years:
        print(f"    scanning {ydir.name}...")
        part = (
            pl.scan_parquet(str(ydir / "**" / "*.parquet"))
            .select(["prediction_id", "timestamp", "price"])
            .group_by("prediction_id")
            .agg([
                pl.col("price").sort_by("timestamp").last()
                    .alias("closing_price"),
                pl.col("timestamp").max().alias("last_trade_ts"),
            ])
            .collect(engine="streaming")
        )
        partials.append(part)
        print(f"      {len(part):,} tokens")

    combined = pl.concat(partials)
    closing = (
        combined
        .sort("last_trade_ts")
        .group_by("prediction_id")
        .agg([
            pl.col("closing_price").last().alias("closing_price"),
            pl.col("last_trade_ts").last().alias("last_trade_ts"),
        ])
    )
    closing.write_parquet(out)
    print(f"  ✓ {len(closing):,} tokens -> {out}")
    return closing


# ── 2. Per-wallet trade rows with direction, counterparty, CLV ───────────────

def wallet_trades_lazy(start: str | None, end: str) -> pl.LazyFrame:
    base = (
        pl.scan_parquet(TRADES_GLOB)
        .filter(pl.col("timestamp") <= pl.lit(end).str.to_datetime(time_zone="UTC"))
    )
    if start:
        base = base.filter(
            pl.col("timestamp") > pl.lit(start).str.to_datetime(time_zone="UTC")
        )

    cols = ["trade_id", "timestamp", "market_id", "prediction_id",
            "category", "price", "quantity"]

    taker = base.select(cols + [
        pl.col("taker_address").alias("wallet"),
        pl.col("maker_address").alias("counterparty"),
        pl.when(pl.col("taker_bought")).then(1).otherwise(-1).alias("direction"),
        pl.lit(False).alias("is_maker"),
    ])
    maker = base.select(cols + [
        pl.col("maker_address").alias("wallet"),
        pl.col("taker_address").alias("counterparty"),
        pl.when(pl.col("taker_bought")).then(-1).otherwise(1).alias("direction"),
        pl.lit(True).alias("is_maker"),
    ])
    return pl.concat([taker, maker]).filter(pl.col("wallet").is_not_null())


def add_clv(lf: pl.LazyFrame, closing: pl.DataFrame) -> pl.LazyFrame:
    return (
        lf.join(closing.lazy(), on="prediction_id", how="left")
        .with_columns([
            (pl.col("direction") * (pl.col("closing_price") - pl.col("price")))
                .alias("clv"),
            (pl.col("last_trade_ts") - pl.col("timestamp"))
                .dt.total_days().alias("days_to_close"),
            (pl.col("price") * pl.col("quantity")).alias("usdc"),
        ])
    )


# ── 3. PIT features as of a cutoff ────────────────────────────────────────────

def build_pit_features(cutoff: str, force: bool = False) -> Path:
    out = PIT_DIR / f"features_asof_{cutoff}.parquet"
    if out.exists() and not force:
        print(f"  ✓ features exist for {cutoff}")
        return out

    print(f"\n  PIT features as of {cutoff}...")
    closing = build_closing_prices()
    wt = add_clv(wallet_trades_lazy(None, cutoff + " 23:59:59"), closing)

    wt = wt.with_columns([
        ((pl.col("quantity") % 100 == 0) | (pl.col("quantity") % 50 == 0))
            .alias("is_round_size"),
        (pl.col("quantity") <= 5).alias("is_min_size"),
        (pl.col("direction") == 1).alias("is_buy"),
        (pl.col("price") <= 0.15).alias("is_longshot"),
        (pl.col("price") >= 0.85).alias("is_sureshot"),
        ((pl.col("price") > 0.35) & (pl.col("price") < 0.65)).alias("is_midrange"),
        (pl.col("days_to_close") <= 2).alias("is_late_entry"),
        (pl.col("days_to_close") >= 14).alias("is_early_entry"),
    ])

    feats = (
        wt.group_by("wallet")
        .agg([
            pl.len().alias("n_trades"),
            pl.col("market_id").n_unique().alias("n_markets"),
            pl.col("category").n_unique().alias("n_categories"),
            pl.col("usdc").sum().alias("total_volume"),
            pl.col("usdc").mean().alias("avg_trade_usdc"),
            pl.col("usdc").std().alias("usdc_std"),
            pl.col("usdc").max().alias("max_trade_usdc"),
            pl.col("timestamp").min().alias("first_trade"),
            pl.col("timestamp").max().alias("last_trade"),
            pl.col("price").mean().alias("avg_price"),
            pl.col("price").std().alias("price_std"),
            pl.col("is_longshot").mean().alias("frac_longshot"),
            pl.col("is_sureshot").mean().alias("frac_sureshot"),
            pl.col("is_midrange").mean().alias("frac_midrange"),
            # microstructure
            pl.col("is_round_size").mean().alias("frac_round_size"),
            pl.col("is_min_size").mean().alias("frac_min_size"),
            pl.col("is_buy").mean().alias("inflow_ratio"),
            pl.col("is_maker").mean().alias("frac_maker"),
            # timing
            pl.col("is_late_entry").mean().alias("frac_late_entry"),
            pl.col("is_early_entry").mean().alias("frac_early_entry"),
            pl.col("days_to_close").mean().alias("avg_days_to_close"),
            # past CLV (feature = track record; used for ablation)
            ((pl.col("clv") * pl.col("usdc")).sum() / pl.col("usdc").sum())
                .alias("past_clv_vw"),
            pl.col("clv").mean().alias("past_clv_mean"),
            (pl.col("clv") > 0).mean().alias("past_clv_hitrate"),
            pl.col("clv").std().alias("past_clv_std"),
            # insider vs skill decomposition
            pl.col("clv").filter(pl.col("is_late_entry")).mean()
                .alias("clv_when_late"),
            pl.col("clv").filter(pl.col("is_early_entry")).mean()
                .alias("clv_when_early"),
        ])
        .filter(pl.col("n_trades") >= MIN_TRADES_FEATURES)
        .collect(engine="streaming")
    )

    # Second pass: category HHI and counterparty HHI (wash-trade proxy)
    print("  Computing HHI features...")
    cat_hhi = (
        wallet_trades_lazy(None, cutoff + " 23:59:59")
        .group_by(["wallet", "category"]).agg(pl.len().alias("n"))
        .with_columns((pl.col("n") / pl.col("n").sum().over("wallet")).alias("s"))
        .group_by("wallet").agg((pl.col("s") ** 2).sum().alias("category_hhi"))
        .collect(engine="streaming")
    )
    cp_hhi = (
        wallet_trades_lazy(None, cutoff + " 23:59:59")
        .group_by(["wallet", "counterparty"]).agg(pl.len().alias("n"))
        .with_columns((pl.col("n") / pl.col("n").sum().over("wallet")).alias("s"))
        .group_by("wallet").agg((pl.col("s") ** 2).sum().alias("counterparty_hhi"))
        .collect(engine="streaming")
    )
    feats = feats.join(cat_hhi, on="wallet", how="left") \
                 .join(cp_hhi,  on="wallet", how="left")

    feats = feats.with_columns([
        ((pl.col("last_trade") - pl.col("first_trade")).dt.total_days() + 1)
            .alias("span_days"),
    ]).with_columns([
        (pl.col("n_trades") / pl.col("span_days").clip(lower_bound=1))
            .alias("trades_per_day"),
        (pl.col("frac_maker") * pl.col("category_hhi")).alias("specialist_maker"),
        (pl.col("clv_when_late") - pl.col("clv_when_early")).alias("insider_gap"),
        (pl.col("counterparty_hhi") > 0.5).alias("wash_flag"),
        pl.lit(cutoff).alias("asof"),
    ])

    feats.write_parquet(out)
    print(f"  ✓ {len(feats):,} wallets -> {out}")
    return out


# ── 4. Forward CLV labels ─────────────────────────────────────────────────────

def build_labels(cutoff: str, horizon_end: str, force: bool = False) -> Path:
    out = PIT_DIR / f"labels_{cutoff}_to_{horizon_end}.parquet"
    if out.exists() and not force:
        print(f"  ✓ labels exist for {cutoff} -> {horizon_end}")
        return out

    print(f"  CLV labels ({cutoff}, {horizon_end}]...")
    closing = build_closing_prices()
    wt = add_clv(
        wallet_trades_lazy(cutoff + " 23:59:59", horizon_end + " 23:59:59"),
        closing,
    )
    labels = (
        wt.group_by("wallet")
        .agg([
            pl.len().alias("fwd_n_trades"),
            ((pl.col("clv") * pl.col("usdc")).sum() / pl.col("usdc").sum())
                .alias("fwd_clv_vw"),
            (pl.col("clv") > 0).mean().alias("fwd_clv_hitrate"),
        ])
        .filter(pl.col("fwd_n_trades") >= MIN_TRADES_FORWARD)
        .collect(engine="streaming")
    )
    q75 = labels["fwd_clv_vw"].quantile(0.75)
    labels = labels.with_columns(
        (pl.col("fwd_clv_vw") >= q75).cast(pl.Int32).alias("label_skilled")
    )
    labels.write_parquet(out)
    print(f"  ✓ {len(labels):,} labeled wallets "
          f"(top-quartile fwd CLV >= {q75:.4f}) -> {out}")
    return out


def run_all():
    print("=" * 65)
    print("  PIT FEATURE + CLV LABEL PANEL (leak-free)")
    print("=" * 65)
    build_closing_prices()
    for cutoff, horizon, role in PANEL:
        print(f"\n── {role}: features@{cutoff} -> labels through {horizon} ──")
        build_pit_features(cutoff)
        build_labels(cutoff, horizon)
    print("\n✅ Panel complete:", PIT_DIR)


if __name__ == "__main__":
    run_all()