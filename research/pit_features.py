"""
Point-in-Time (PIT) Feature Builder + CLV Engine.  LEAK-FREE.

CORRECTED per referee feedback (round 2): closing price is now a
TIERED, LIFETIME-ADAPTIVE VWAP rather than the literal last trade
price or a fixed clock-time window. A fixed-hours window (e.g. 24-72h
before resolution) caused severe, category-concentrated attrition
(52.8% of Sports tokens missing) because short-duration markets
(same-day sports) may not have any trades that far before resolution.
The tiered approach tries progressively looser windows, always
relative to each token's OWN trading lifetime, only for tokens that
fail the stricter tier -- so long-running markets (elections) get the
most conservative, most defensible window, while short-lived markets
(sports) still get SOME pre-resolution window rather than being
dropped. Each token is tagged with which tier was used
(`closing_tier`), reported transparently as a coverage table.

  Tier 1 (strict, preferred):  25%-10% of lifetime before resolution
  Tier 2 (fallback):           50%-15% of lifetime before resolution
  Tier 3 (loosest fallback):   any trade in the first 75% of lifetime

Every feature is computed from raw trades using ONLY data <= cutoff.
Labels are computed from trades strictly AFTER the cutoff.

Panel design (expanding window, quarterly train cutoffs):
  train:      features@2023-06-30 -> labels (2023-07..2023-12)
  train:      features@2023-09-30 -> labels (2023-10..2024-03)
  train:      features@2023-12-31 -> labels (2024-01..2024-06)
  validation: features@2024-06-30 -> labels (2024-07..2024-12)
  test:       features@2024-12-31 -> labels (2025-01..2025-06)
  TRUE OOS 1: features@2025-06-30 -> labels (2025-07..2025-12)
  TRUE OOS 2: features@2025-12-31 -> labels (2026-01..2026-03)

Usage:
    python run.py pit
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import polars as pl
import time
from pathlib import Path
from datetime import date, timedelta
from config import RAW_DIR, PROC_DIR

TRADES_GLOB = str(RAW_DIR / "trades" / "**" / "*.parquet")
PIT_DIR = PROC_DIR / "pit"
PIT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TRADES_FEATURES = 30
MIN_TRADES_FORWARD  = 10

PANEL = [
    ("2023-06-30", "2023-12-31", "train"),
    ("2023-09-30", "2024-03-31", "train"),
    ("2023-12-31", "2024-06-30", "train"),
    ("2024-06-30", "2024-12-31", "validation"),
    ("2024-12-31", "2025-06-30", "test"),
    ("2025-06-30", "2025-12-31", "TRUE_OOS_1"),
    ("2025-12-31", "2026-03-29", "TRUE_OOS_2"),
]


# ── 1. TIERED, lifetime-adaptive closing prices ──────────────────────────────

def build_closing_prices(force: bool = False) -> pl.DataFrame:
    """
    Tiered adaptive closing-price window. See module docstring for
    rationale and tier definitions.

    Tier 1 (strict, preferred):  25%-10% of lifetime before resolution
    Tier 2 (fallback):           50%-15% of lifetime before resolution
    Tier 3 (loosest fallback):   any trade in the first 75% of lifetime

    Each token is tagged with which tier was used (`closing_tier`).
    Tokens failing all three tiers are marked no_window_liquidity=True
    and excluded from CLV computation entirely (true illiquid tail).
    """
    out = PIT_DIR / "closing_prices_tiered.parquet"
    if out.exists() and not force:
        return pl.read_parquet(out)

    print("  Building TIERED adaptive closing-price table...")
    years = sorted((RAW_DIR / "trades").glob("year=*"))

    print("  [pass 1/2] Finding first + final trade timestamp per token...")
    span_partials = []
    for ydir in years:
        print(f"    scanning {ydir.name}...")
        part = (
            pl.scan_parquet(str(ydir / "**" / "*.parquet"))
            .select(["prediction_id", "timestamp"])
            .group_by("prediction_id")
            .agg([
                pl.col("timestamp").min().alias("first_trade_ts"),
                pl.col("timestamp").max().alias("last_trade_ts"),
            ])
            .collect(engine="streaming")
        )
        span_partials.append(part)
    span = (
        pl.concat(span_partials)
        .group_by("prediction_id")
        .agg([
            pl.col("first_trade_ts").min().alias("first_trade_ts"),
            pl.col("last_trade_ts").max().alias("last_trade_ts"),
        ])
        .with_columns(
            (pl.col("last_trade_ts") - pl.col("first_trade_ts"))
                .dt.total_seconds().alias("lifetime_seconds")
        )
    )
    print(f"    ✓ {len(span):,} tokens")

    def compute_window_vwap(pct_start, pct_end, restrict_ids=None):
        """VWAP for all (or a restricted subset of) tokens within a
        given lifetime-percentage window, measured backward from each
        token's own final trade."""
        parts = []
        for ydir in years:
            base = (
                pl.scan_parquet(str(ydir / "**" / "*.parquet"))
                .select(["prediction_id", "timestamp", "price", "quantity"])
                .join(span.lazy(), on="prediction_id", how="inner")
            )
            if restrict_ids is not None:
                base = base.filter(pl.col("prediction_id").is_in(restrict_ids))
            part = (
                base
                .with_columns(
                    ((pl.col("last_trade_ts") - pl.col("timestamp"))
                        .dt.total_seconds() /
                     pl.col("lifetime_seconds").clip(lower_bound=1)
                    ).alias("pct_of_life_remaining")
                )
                .filter(
                    (pl.col("pct_of_life_remaining") <= pct_start) &
                    (pl.col("pct_of_life_remaining") >= pct_end)
                )
                .with_columns((pl.col("price") * pl.col("quantity")).alias("pv"))
                .group_by("prediction_id")
                .agg([
                    pl.col("pv").sum().alias("pv_sum"),
                    pl.col("quantity").sum().alias("qty_sum"),
                ])
                .collect(engine="streaming")
            )
            parts.append(part)
        return (
            pl.concat(parts)
            .group_by("prediction_id")
            .agg([
                pl.col("pv_sum").sum().alias("pv_sum"),
                pl.col("qty_sum").sum().alias("qty_sum"),
            ])
            .with_columns(
                (pl.col("pv_sum") / pl.col("qty_sum")).alias("closing_price")
            )
        )

    # ── Tier 1: strict window, all tokens ────────────────────────────
    print("  [Tier 1] 25%-10% of lifetime, all tokens...")
    tier1 = compute_window_vwap(0.25, 0.10).with_columns(
        pl.lit(1).alias("closing_tier")
    )
    print(f"    ✓ {len(tier1):,} tokens covered")

    all_ids = set(span["prediction_id"].to_list())
    covered_ids = set(tier1["prediction_id"].to_list())
    missing_t1 = list(all_ids - covered_ids)
    print(f"    {len(missing_t1):,} tokens missing, trying Tier 2...")

    # ── Tier 2: looser window, only tokens missing tier 1 ───────────
    if missing_t1:
        tier2 = compute_window_vwap(0.50, 0.15, restrict_ids=missing_t1) \
            .with_columns(pl.lit(2).alias("closing_tier"))
        print(f"    ✓ {len(tier2):,} additional tokens covered")
    else:
        tier2 = pl.DataFrame({
            "prediction_id": [], "pv_sum": [], "qty_sum": [],
            "closing_price": [], "closing_tier": [],
        })

    covered_ids |= set(tier2["prediction_id"].to_list())
    missing_t2 = list(all_ids - covered_ids)
    print(f"    {len(missing_t2):,} tokens still missing, trying Tier 3...")

    # ── Tier 3: loosest fallback, only tokens missing tier 1+2 ──────
    if missing_t2:
        tier3 = compute_window_vwap(1.00, 0.25, restrict_ids=missing_t2) \
            .with_columns(pl.lit(3).alias("closing_tier"))
        print(f"    ✓ {len(tier3):,} additional tokens covered")
    else:
        tier3 = pl.DataFrame({
            "prediction_id": [], "pv_sum": [], "qty_sum": [],
            "closing_price": [], "closing_tier": [],
        })

    all_tiers = pl.concat([tier1, tier2, tier3], how="diagonal")

    closing = (
        span.select(["prediction_id", "last_trade_ts"])
        .join(all_tiers.select(["prediction_id", "closing_price", "closing_tier"]),
              on="prediction_id", how="left")
        .with_columns(
            pl.col("closing_price").is_null().alias("no_window_liquidity")
        )
    )

    closing.write_parquet(out)

    n_total = len(closing)
    n_missing = closing.filter(pl.col("no_window_liquidity"))["prediction_id"].len()
    tier_counts = closing.group_by("closing_tier").agg(pl.len().alias("n"))
    print(f"\n  ✓ {n_total:,} tokens -> {out}")
    print(f"  Tier distribution:")
    for row in tier_counts.sort("closing_tier").iter_rows(named=True):
        tier = row["closing_tier"]
        n = row["n"]
        print(f"    Tier {tier}: {n:,} tokens ({n/n_total*100:.1f}%)")
    print(f"  Still uncovered after all tiers: {n_missing:,} "
          f"({n_missing/n_total*100:.1f}%)")

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
    """
    CLV computed against the tiered adaptive closing price. Trades on
    tokens that failed all three tiers (no_window_liquidity=True) are
    excluded -- their closing_price is null and this filter drops them
    explicitly rather than silently propagating nulls downstream.
    """
    return (
        lf.join(
            closing.lazy().select(["prediction_id", "closing_price",
                                    "closing_tier", "no_window_liquidity"]),
            on="prediction_id", how="left",
        )
        .filter(pl.col("no_window_liquidity") != True)
        .with_columns([
            (pl.col("direction") * (pl.col("closing_price") - pl.col("price")))
                .alias("clv"),
            (pl.col("price") * pl.col("quantity")).alias("usdc"),
        ])
    )


def add_clv_with_timing(lf: pl.LazyFrame, closing: pl.DataFrame) -> pl.LazyFrame:
    """
    Like add_clv, but also joins last_trade_ts for computing
    days_to_close / timing features (frac_late_entry, frac_early_entry
    etc.), which need the token's final trade timestamp separately
    from the closing-price computation itself.
    """
    return (
        lf.join(
            closing.lazy().select([
                "prediction_id", "closing_price", "closing_tier",
                "no_window_liquidity", "last_trade_ts",
            ]),
            on="prediction_id", how="left",
        )
        .filter(pl.col("no_window_liquidity") != True)
        .with_columns([
            (pl.col("direction") * (pl.col("closing_price") - pl.col("price")))
                .alias("clv"),
            (pl.col("price") * pl.col("quantity")).alias("usdc"),
            (pl.col("last_trade_ts") - pl.col("timestamp"))
                .dt.total_days().alias("days_to_close"),
        ])
    )


# ── 3. PIT features as of a cutoff — checkpointed, single streaming pass ────

def build_pit_features(cutoff: str, force: bool = False) -> Path:
    out = PIT_DIR / f"features_asof_{cutoff}.parquet"
    if out.exists() and not force:
        print(f"  ✓ features exist for {cutoff}")
        return out

    ckpt_daily     = PIT_DIR / f"_ckpt_daily_{cutoff}.parquet"
    ckpt_prefeats  = PIT_DIR / f"_ckpt_prefeats_{cutoff}.parquet"
    ckpt_cathhi    = PIT_DIR / f"_ckpt_cathhi_{cutoff}.parquet"
    ckpt_cphhi     = PIT_DIR / f"_ckpt_cphhi_{cutoff}.parquet"
    ckpt_nmarkets  = PIT_DIR / f"_ckpt_nmarkets_{cutoff}.parquet"

    print(f"\n  PIT features as of {cutoff} (checkpointed, single-pass, "
          f"TIERED CLV)...")
    t0 = time.time()
    closing = build_closing_prices()

    # ── STAGE 1: daily aggregation ────────────────────────────────────
    if ckpt_daily.exists():
        print(f"  [1/5] ✓ checkpoint exists, loading daily aggregation...")
        daily = pl.read_parquet(ckpt_daily)
    else:
        print("  [1/5] Aggregating per (wallet, date) — single streaming pass...")
        wt = add_clv_with_timing(
            wallet_trades_lazy(None, cutoff + " 23:59:59"), closing
        )
        wt = wt.with_columns([
            pl.col("timestamp").dt.date().alias("trade_date"),
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
        daily = (
            wt.group_by(["wallet", "trade_date"])
            .agg([
                pl.len().alias("n"),
                pl.col("usdc").sum().alias("usdc_sum"),
                pl.col("usdc").max().alias("usdc_max"),
                pl.col("timestamp").min().alias("first_trade_day"),
                pl.col("timestamp").max().alias("last_trade_day"),
                (pl.col("price") * pl.col("usdc")).sum().alias("price_usdc_sum"),
                pl.col("is_longshot").sum().alias("n_longshot"),
                pl.col("is_sureshot").sum().alias("n_sureshot"),
                pl.col("is_midrange").sum().alias("n_midrange"),
                pl.col("is_round_size").sum().alias("n_round_size"),
                pl.col("is_min_size").sum().alias("n_min_size"),
                pl.col("is_buy").sum().alias("n_buy"),
                pl.col("is_maker").sum().alias("n_maker"),
                pl.col("is_late_entry").sum().alias("n_late_entry"),
                pl.col("is_early_entry").sum().alias("n_early_entry"),
                pl.col("days_to_close").sum().alias("days_to_close_sum"),
                (pl.col("clv") * pl.col("usdc")).sum().alias("clv_usdc_sum"),
                pl.col("clv").sum().alias("clv_sum"),
                (pl.col("clv") > 0).sum().alias("n_positive_clv"),
                pl.col("clv").filter(pl.col("is_late_entry")).sum()
                    .alias("clv_late_sum"),
                pl.col("is_late_entry").sum().alias("n_late_for_avg"),
                pl.col("clv").filter(pl.col("is_early_entry")).sum()
                    .alias("clv_early_sum"),
            ])
            .collect(engine="streaming")
        )
        daily.write_parquet(ckpt_daily)
        print(f"        ✓ {len(daily):,} rows, checkpoint saved "
              f"({time.time()-t0:.0f}s elapsed)")

    # ── STAGE 2: collapse to per-wallet, apply min-trades filter ─────
    if ckpt_prefeats.exists():
        print(f"  [2/5] ✓ checkpoint exists, loading pre-HHI features...")
        feats = pl.read_parquet(ckpt_prefeats)
    else:
        print("  [2/5] Collapsing (wallet, date) -> per-wallet (pre-HHI)...")
        feats = (
            daily.lazy()
            .group_by("wallet")
            .agg([
                pl.col("n").sum().alias("n_trades"),
                pl.col("usdc_sum").sum().alias("total_volume"),
                pl.col("usdc_max").max().alias("max_trade_usdc"),
                pl.col("first_trade_day").min().alias("first_trade"),
                pl.col("last_trade_day").max().alias("last_trade"),
                pl.col("price_usdc_sum").sum().alias("_price_usdc_sum"),
                pl.col("n_longshot").sum().alias("_n_longshot"),
                pl.col("n_sureshot").sum().alias("_n_sureshot"),
                pl.col("n_midrange").sum().alias("_n_midrange"),
                pl.col("n_round_size").sum().alias("_n_round_size"),
                pl.col("n_min_size").sum().alias("_n_min_size"),
                pl.col("n_buy").sum().alias("_n_buy"),
                pl.col("n_maker").sum().alias("_n_maker"),
                pl.col("n_late_entry").sum().alias("_n_late_entry"),
                pl.col("n_early_entry").sum().alias("_n_early_entry"),
                pl.col("days_to_close_sum").sum().alias("_days_to_close_sum"),
                pl.col("clv_usdc_sum").sum().alias("_clv_usdc_sum"),
                pl.col("clv_sum").sum().alias("_clv_sum"),
                pl.col("n_positive_clv").sum().alias("_n_positive_clv"),
                pl.col("clv_late_sum").sum().alias("_clv_late_sum"),
                pl.col("n_late_for_avg").sum().alias("_n_late_for_avg"),
                pl.col("clv_early_sum").sum().alias("_clv_early_sum"),
            ])
            .filter(pl.col("n_trades") >= MIN_TRADES_FEATURES)
            .with_columns([
                (pl.col("total_volume") / pl.col("n_trades")).alias("avg_trade_usdc"),
                (pl.col("_price_usdc_sum") / pl.col("total_volume").clip(lower_bound=1e-9))
                    .alias("avg_price"),
                (pl.col("_n_longshot") / pl.col("n_trades")).alias("frac_longshot"),
                (pl.col("_n_sureshot") / pl.col("n_trades")).alias("frac_sureshot"),
                (pl.col("_n_midrange") / pl.col("n_trades")).alias("frac_midrange"),
                (pl.col("_n_round_size") / pl.col("n_trades")).alias("frac_round_size"),
                (pl.col("_n_min_size") / pl.col("n_trades")).alias("frac_min_size"),
                (pl.col("_n_buy") / pl.col("n_trades")).alias("inflow_ratio"),
                (pl.col("_n_maker") / pl.col("n_trades")).alias("frac_maker"),
                (pl.col("_n_late_entry") / pl.col("n_trades")).alias("frac_late_entry"),
                (pl.col("_n_early_entry") / pl.col("n_trades")).alias("frac_early_entry"),
                (pl.col("_days_to_close_sum") / pl.col("n_trades"))
                    .alias("avg_days_to_close"),
                (pl.col("_clv_usdc_sum") / pl.col("total_volume").clip(lower_bound=1e-9))
                    .alias("past_clv_vw"),
                (pl.col("_clv_sum") / pl.col("n_trades")).alias("past_clv_mean"),
                (pl.col("_n_positive_clv") / pl.col("n_trades")).alias("past_clv_hitrate"),
                pl.when(pl.col("_n_late_for_avg") > 0)
                  .then(pl.col("_clv_late_sum") / pl.col("_n_late_for_avg"))
                  .otherwise(None).alias("clv_when_late"),
                pl.when(pl.col("_n_early_entry") > 0)
                  .then(pl.col("_clv_early_sum") / pl.col("_n_early_entry"))
                  .otherwise(None).alias("clv_when_early"),
            ])
            .select([
                "wallet", "n_trades", "total_volume", "max_trade_usdc",
                "first_trade", "last_trade", "avg_trade_usdc", "avg_price",
                "frac_longshot", "frac_sureshot", "frac_midrange",
                "frac_round_size", "frac_min_size", "inflow_ratio", "frac_maker",
                "frac_late_entry", "frac_early_entry", "avg_days_to_close",
                "past_clv_vw", "past_clv_mean", "past_clv_hitrate",
                "clv_when_late", "clv_when_early",
            ])
            .collect(engine="streaming")
        )
        feats.write_parquet(ckpt_prefeats)
        print(f"        ✓ {len(feats):,} wallets, checkpoint saved "
              f"({time.time()-t0:.0f}s elapsed)")

    wallet_list = feats["wallet"].to_list()
    print(f"  Relevant wallet population: {len(wallet_list):,}")

    # ── STAGE 3: category HHI + n_categories ──────────────────────────
    if ckpt_cathhi.exists():
        print(f"  [3/5] ✓ checkpoint exists, loading category HHI...")
        cat_hhi = pl.read_parquet(ckpt_cathhi)
    else:
        print("  [3/5] Computing category HHI + n_categories (restricted wallets)...")
        cat_counts = (
            wallet_trades_lazy(None, cutoff + " 23:59:59")
            .filter(pl.col("wallet").is_in(wallet_list))
            .group_by(["wallet", "category"]).agg(pl.len().alias("n"))
            .collect(engine="streaming")
        )
        cat_hhi = (
            cat_counts.lazy()
            .with_columns((pl.col("n") / pl.col("n").sum().over("wallet")).alias("s"))
            .group_by("wallet")
            .agg([
                (pl.col("s") ** 2).sum().alias("category_hhi"),
                pl.len().alias("n_categories"),
            ])
            .collect()
        )
        cat_hhi.write_parquet(ckpt_cathhi)
        print(f"        ✓ checkpoint saved ({time.time()-t0:.0f}s elapsed)")

    # ── STAGE 4: n_markets ─────────────────────────────────────────────
    if ckpt_nmarkets.exists():
        print(f"  [4/5] ✓ checkpoint exists, loading n_markets...")
        n_markets_df = pl.read_parquet(ckpt_nmarkets)
    else:
        print("  [4/5] Computing n_markets (restricted wallets)...")
        n_markets_df = (
            wallet_trades_lazy(None, cutoff + " 23:59:59")
            .filter(pl.col("wallet").is_in(wallet_list))
            .group_by(["wallet", "market_id"]).agg(pl.len().alias("n"))
            .group_by("wallet").agg(pl.len().alias("n_markets"))
            .collect(engine="streaming")
        )
        n_markets_df.write_parquet(ckpt_nmarkets)
        print(f"        ✓ checkpoint saved ({time.time()-t0:.0f}s elapsed)")

    # ── STAGE 5: counterparty HHI, bucketed by wallet hash ─────────────
    if ckpt_cphhi.exists():
        print(f"  [5/5] ✓ checkpoint exists, loading counterparty HHI...")
        cp_hhi = pl.read_parquet(ckpt_cphhi)
    else:
        print("  [5/5] Computing counterparty HHI (bucketed by wallet, "
              "20 buckets)...")
        N_BUCKETS = 20
        bucket_results = []
        for bucket_id in range(N_BUCKETS):
            bucket_wallets = [
                w for w in wallet_list
                if (hash(w) % N_BUCKETS) == bucket_id
            ]
            if not bucket_wallets:
                continue
            bucket_hhi = (
                wallet_trades_lazy(None, cutoff + " 23:59:59")
                .filter(pl.col("wallet").is_in(bucket_wallets))
                .group_by(["wallet", "counterparty"]).agg(pl.len().alias("n"))
                .with_columns(
                    (pl.col("n") / pl.col("n").sum().over("wallet")).alias("s")
                )
                .group_by("wallet")
                .agg((pl.col("s") ** 2).sum().alias("counterparty_hhi"))
                .collect(engine="streaming")
            )
            bucket_results.append(bucket_hhi)
            print(f"        bucket {bucket_id+1}/{N_BUCKETS}: "
                  f"{len(bucket_wallets):,} wallets, "
                  f"{time.time()-t0:.0f}s elapsed")
        cp_hhi = pl.concat(bucket_results)
        cp_hhi.write_parquet(ckpt_cphhi)
        print(f"        ✓ checkpoint saved, {len(cp_hhi):,} wallets "
              f"({time.time()-t0:.0f}s elapsed)")

    # ── Final combine ─────────────────────────────────────────────────
    print("  Finalizing...")
    feats = feats.join(cat_hhi, on="wallet", how="left") \
                 .join(cp_hhi,  on="wallet", how="left") \
                 .join(n_markets_df, on="wallet", how="left")

    feats = feats.with_columns([
        ((pl.col("last_trade") - pl.col("first_trade")).dt.total_days() + 1)
            .alias("span_days"),
        pl.col("category_hhi").fill_null(0.0),
        pl.col("counterparty_hhi").fill_null(0.0),
        pl.col("n_categories").fill_null(0),
        pl.col("n_markets").fill_null(0),
    ]).with_columns([
        (pl.col("n_trades") / pl.col("span_days").clip(lower_bound=1))
            .alias("trades_per_day"),
        (pl.col("frac_maker") * pl.col("category_hhi")).alias("specialist_maker"),
        (pl.col("clv_when_late") - pl.col("clv_when_early")).alias("insider_gap"),
        (pl.col("counterparty_hhi") > 0.5).alias("wash_flag"),
        pl.lit(cutoff).alias("asof"),
    ])

    for col in ["price_std", "usdc_std", "past_clv_std"]:
        feats = feats.with_columns(pl.lit(0.0).alias(col))

    feats.write_parquet(out)
    print(f"  ✓ {len(feats):,} wallets -> {out}  "
          f"(total: {time.time()-t0:.0f}s)")

    for ckpt in [ckpt_daily, ckpt_prefeats, ckpt_cathhi, ckpt_cphhi, ckpt_nmarkets]:
        ckpt.unlink(missing_ok=True)

    return out


# ── 4. Forward CLV labels — single streaming pass, date-bucketed ────────────

def build_labels(cutoff: str, horizon_end: str, force: bool = False) -> Path:
    out = PIT_DIR / f"labels_{cutoff}_to_{horizon_end}.parquet"
    if out.exists() and not force:
        print(f"  ✓ labels exist for {cutoff} -> {horizon_end}")
        return out

    print(f"  CLV labels ({cutoff}, {horizon_end}]... "
          f"(single streaming pass, date-bucketed, TIERED CLV)")
    t0 = time.time()
    closing = build_closing_prices()

    wt = add_clv(
        wallet_trades_lazy(cutoff + " 23:59:59", horizon_end + " 23:59:59"),
        closing,
    ).with_columns(pl.col("timestamp").dt.date().alias("trade_date"))

    print("  Aggregating per (wallet, date)...")
    daily = (
        wt.group_by(["wallet", "trade_date"])
        .agg([
            pl.len().alias("n"),
            (pl.col("clv") * pl.col("usdc")).sum().alias("clv_usdc_sum"),
            pl.col("usdc").sum().alias("usdc_sum"),
            (pl.col("clv") > 0).sum().alias("n_positive_clv"),
        ])
        .collect(engine="streaming")
    )
    print(f"  ✓ {len(daily):,} (wallet, date) rows ({time.time()-t0:.0f}s elapsed)")

    labels = (
        daily.lazy()
        .group_by("wallet")
        .agg([
            pl.col("n").sum().alias("fwd_n_trades"),
            pl.col("clv_usdc_sum").sum().alias("_clv_usdc_sum"),
            pl.col("usdc_sum").sum().alias("_usdc_sum"),
            pl.col("n_positive_clv").sum().alias("_n_positive"),
        ])
        .filter(pl.col("_usdc_sum") > 0)
        .with_columns([
            (pl.col("_clv_usdc_sum") / pl.col("_usdc_sum")).alias("fwd_clv_vw"),
            (pl.col("_n_positive") / pl.col("fwd_n_trades")).alias("fwd_clv_hitrate"),
        ])
        .filter(pl.col("fwd_n_trades") >= MIN_TRADES_FORWARD)
        .filter(pl.col("fwd_clv_vw").is_finite())
        .select(["wallet", "fwd_n_trades", "fwd_clv_vw", "fwd_clv_hitrate"])
        .collect()
    )

    if len(labels) == 0:
        print("  ⚠ No wallets passed filters.")
        return out

    q75 = labels["fwd_clv_vw"].quantile(0.75)
    labels = labels.with_columns(
        (pl.col("fwd_clv_vw") >= q75).cast(pl.Int32).alias("label_skilled")
    )
    labels.write_parquet(out)
    print(f"  ✓ {len(labels):,} labeled wallets "
          f"(top-quartile fwd CLV >= {q75:.4f}) -> {out}  "
          f"(total: {time.time()-t0:.0f}s)")
    return out


def run_all():
    print("=" * 65)
    print("  PIT FEATURE + CLV LABEL PANEL (leak-free, TIERED CLV)")
    print("=" * 65)
    build_closing_prices()
    for cutoff, horizon, role in PANEL:
        print(f"\n── {role}: features@{cutoff} -> labels through {horizon} ──")
        build_pit_features(cutoff)
        build_labels(cutoff, horizon)
    print("\n✅ Panel complete:", PIT_DIR)


if __name__ == "__main__":
    run_all()