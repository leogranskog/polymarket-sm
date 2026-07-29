import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import polars as pl
import json
from datetime import date, timedelta
from dataclasses import dataclass
from typing import Optional
from config import (
    SENTIMENT_WEIGHTS, MOMENTUM_WINDOW,
    ENTRY_COMPOSITE_MIN, ENTRY_CONVICTION_MIN, ENTRY_MIN_WALLETS,
    EXIT_COMPOSITE_MAX, EXIT_MOMENTUM_DROP, EXIT_CONVICTION_MIN,
)

CATEGORIES = ["sports", "crypto", "finance", "politics", "tech", "culture", "weather"]


@dataclass
class SentimentScore:
    market_id:           str
    date:                date
    category:            str   = "unknown"
    direction:           float = 0.0
    conviction:          float = 0.0
    momentum:            float = 0.0
    timing:              float = 0.0
    composite:           float = 0.0
    sm_wallets_active:   int   = 0
    sm_net_flow:         float = 0.0
    sm_volume_24h:       float = 0.0
    sm_avg_entry_yes:    float = 0.0
    sm_avg_entry_no:     float = 0.0
    market_price_yes:    float = 0.0
    category_expert_count: int = 0      # SM wallets that are specialists in this category
    category_expert_agreement: float = 0.0  # how much experts agree vs generalists
    entry_signal:        bool  = False
    exit_signal:         bool  = False
    signal_strength:     str   = "none"

    def summary(self) -> str:
        bars = lambda v: "█" * int(abs(v) * 10) + "░" * (10 - int(abs(v) * 10))
        sign = lambda v: "+" if v >= 0 else ""
        return (
            f"\n  Market:   {self.market_id[:40]}"
            f"\n  Category: {self.category.upper()}   "
            f"Date: {self.date}   Price Yes: {self.market_price_yes:.2f}"
            f"\n"
            f"\n  Direction:        {bars(self.direction)}  {sign(self.direction)}{self.direction:.2f}"
            f"\n  Conviction:       {bars(self.conviction)}  {self.conviction:.2f}"
            f"\n  Momentum:         {bars(self.momentum)}  {sign(self.momentum)}{self.momentum:.2f}"
            f"\n  Timing:           {bars(self.timing)}  {self.timing:.2f}"
            f"\n  ──────────────────────────────────────────"
            f"\n  Composite:        {sign(self.composite)}{self.composite:.2f}"
            f"   ({self.signal_strength.upper()})"
            f"\n"
            f"\n  SM Wallets:       {self.sm_wallets_active} total"
            f"\n  Category experts: {self.category_expert_count}"
            f"  Expert agreement: {self.category_expert_agreement:.2f}"
            f"\n  Net Flow:         ${self.sm_net_flow:+,.0f}"
            f"\n  Entry signal:     {'✅ YES' if self.entry_signal else '❌ no'}"
            f"\n  Exit signal:      {'🚨 YES' if self.exit_signal else '✅ no'}"
        )


class SentimentEngine:
    """
    Category-aware SM sentiment engine.

    For each market, sentiment is computed in two layers:
      1. Category experts  — wallets whose top_category matches the market
      2. All SM wallets    — fallback if not enough experts

    Expert signals are weighted more heavily than generalist signals.
    """

    def __init__(self, trades_df: pl.DataFrame, wallets_df: pl.DataFrame):
        # Prepare trades
        self.trades = (
            trades_df
            .filter(pl.col("is_smart_money") == True)
            .with_columns([
                pl.col("timestamp").cast(pl.Date).alias("date"),
                pl.when(pl.col("side") == "YES")
                  .then(pl.col("quantity"))
                  .otherwise(-pl.col("quantity"))
                  .alias("signed_qty"),
                pl.when(pl.col("side") == "YES")
                  .then(pl.col("price") * pl.col("quantity"))
                  .otherwise(-(pl.col("price") * pl.col("quantity")))
                  .alias("usdc_flow"),
            ])
        )

        # Build wallet lookup: address -> {sm_score, top_category, cat_scores}
        self.wallet_meta: dict = {}
        for row in wallets_df.iter_rows(named=True):
            addr = row.get("address") or row.get("user_address", "")
            cat_scores = {}
            if "category_scores" in row and row["category_scores"]:
                try:
                    cat_scores = json.loads(row["category_scores"])
                except Exception:
                    pass
            self.wallet_meta[addr] = {
                "sm_score":     row.get("sm_score", 0.5),
                "top_category": row.get("top_category", "unknown"),
                "cat_scores":   cat_scores,
            }

        self._cache: dict = {}

    def _category_weight(self, wallet_address: str, market_category: str) -> float:
        """
        Returns a weight multiplier for this wallet on this market category.
        Expert in category → weight up to 2.0
        Generalist         → weight 1.0
        Weak in category   → weight 0.5
        """
        meta = self.wallet_meta.get(wallet_address, {})
        cat_scores = meta.get("cat_scores", {})
        top_cat    = meta.get("top_category", "unknown")

        if not market_category or market_category == "unknown":
            return 1.0

        cat_score = cat_scores.get(market_category, 0.0)

        if top_cat == market_category:
            # This is their specialist category
            return 1.0 + cat_score   # 1.0 to 2.0
        elif cat_score >= 0.5:
            # Decent in this category
            return 0.75 + cat_score * 0.5   # 0.75 to 1.25
        else:
            # Weak in this category — downweight
            return max(0.3, cat_score)

    def _get_sm_trades(self, market_id, up_to_date, since_date=None):
        q = (
            self.trades
            .filter(pl.col("market_id") == str(market_id))
            .filter(pl.col("date") <= up_to_date)
        )
        if since_date:
            q = q.filter(pl.col("date") >= since_date)
        return q

    def _compute_direction(self, sm_trades, market_category="unknown"):
        """
        Skill-weighted + category-weighted net direction.
        Expert wallets count more than generalists.
        """
        if len(sm_trades) == 0:
            return 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0

        positions = (
            sm_trades
            .group_by("wallet_address")
            .agg([
                pl.sum("signed_qty").alias("net_position"),
                pl.sum("usdc_flow").alias("usdc_flow"),
                pl.first("sm_score").alias("sm_score"),
            ])
        )

        if len(positions) == 0:
            return 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0

        # Add category weight per wallet
        cat_weights = []
        expert_flags = []
        for addr in positions["wallet_address"].to_list():
            cw = self._category_weight(addr, market_category)
            cat_weights.append(cw)
            meta = self.wallet_meta.get(addr, {})
            expert_flags.append(1 if meta.get("top_category") == market_category else 0)

        positions = positions.with_columns([
            pl.Series("cat_weight",   cat_weights),
            pl.Series("is_expert",    expert_flags),
            pl.col("net_position").sign().alias("direction"),
        ])

        # Combined weight = sm_score × category_weight
        positions = positions.with_columns([
            (pl.col("sm_score") * pl.col("cat_weight")).alias("combined_weight")
        ])

        total_weight = positions["combined_weight"].sum()
        if total_weight == 0:
            return 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0

        weighted_dir = float(
            (positions["direction"] * positions["combined_weight"]).sum()
            / total_weight
        )

        net_flow      = float(positions["usdc_flow"].sum())
        n_wallets     = len(positions)
        n_experts     = int(positions["is_expert"].sum())

        # Expert agreement: do specialists agree with the overall direction?
        experts = positions.filter(pl.col("is_expert") == 1)
        if len(experts) > 0:
            expert_dir = float(
                (experts["direction"] * experts["combined_weight"]).sum()
                / experts["combined_weight"].sum()
            )
            expert_agreement = 1.0 - abs(weighted_dir - expert_dir) / 2
        else:
            expert_agreement = 0.5   # no experts, neutral

        yes_trades = sm_trades.filter(pl.col("side") == "YES")
        no_trades  = sm_trades.filter(pl.col("side") == "NO")
        avg_yes = float(yes_trades["price"].mean()) if len(yes_trades) > 0 else 0.0
        avg_no  = float(no_trades["price"].mean())  if len(no_trades)  > 0 else 0.0

        return weighted_dir, n_wallets, n_experts, net_flow, avg_yes, avg_no, expert_agreement

    def _compute_conviction(self, sm_trades, market_category="unknown"):
        if len(sm_trades) == 0:
            return 0.0

        positions = (
            sm_trades
            .group_by("wallet_address")
            .agg([
                pl.sum("signed_qty").alias("net_position"),
                pl.first("sm_score").alias("sm_score"),
            ])
        )

        if len(positions) < 2:
            return 1.0

        # Build category weights as a new column
        addresses   = positions["wallet_address"].to_list()
        cat_weights = [self._category_weight(a, market_category) for a in addresses]

        positions = positions.with_columns(
            pl.Series("cat_weight", cat_weights)
        ).with_columns(
            (pl.col("sm_score") * pl.col("cat_weight")).alias("combined_weight")
        )

        long_weight  = float(
            positions.filter(pl.col("net_position") > 0)["combined_weight"].sum()
        )
        total_weight = float(positions["combined_weight"].sum())

        if total_weight == 0:
            return 0.0

        pct_long   = long_weight / total_weight
        conviction = abs(pct_long - 0.5) * 2
        return max(0.0, conviction)

    def _compute_momentum(self, market_id, current_date, current_direction, market_category):
        prior_date = current_date - timedelta(days=MOMENTUM_WINDOW)
        cache_key  = (market_id, prior_date, market_category)
        if cache_key in self._cache:
            prior_direction = self._cache[cache_key].direction
        else:
            prior_trades = self._get_sm_trades(market_id, prior_date)
            prior_direction, *_ = self._compute_direction(prior_trades, market_category)
        return current_direction - prior_direction

    def _compute_timing(self, sm_avg_entry_yes, current_price_yes, direction):
        if direction > 0 and sm_avg_entry_yes > 0 and current_price_yes > 0:
            gap = current_price_yes - sm_avg_entry_yes
            return min(1.0, max(0.0, gap / current_price_yes))
        elif direction < 0 and sm_avg_entry_yes > 0 and current_price_yes > 0:
            gap = sm_avg_entry_yes - current_price_yes
            return min(1.0, max(0.0, gap / (1 - current_price_yes + 1e-6)))
        return 0.0

    def _signal_strength(self, composite, conviction, n_wallets, n_experts):
        """
        Stronger signal when category experts are involved.
        """
        if abs(composite) >= 0.70 and conviction >= 0.75 and n_experts >= 3:
            return "strong"
        elif abs(composite) >= 0.70 and conviction >= 0.75 and n_wallets >= 5:
            return "strong"
        elif abs(composite) >= 0.50 and conviction >= 0.55 and n_wallets >= 3:
            return "moderate"
        elif abs(composite) >= 0.30:
            return "weak"
        return "none"

    def compute(
        self,
        market_id:         str,
        snapshot_date:     date,
        current_price_yes: float = 0.5,
        lookback_days:     int   = 30,
        market_category:   str   = "unknown",
    ) -> SentimentScore:

        cache_key = (market_id, snapshot_date, market_category)
        if cache_key in self._cache:
            return self._cache[cache_key]

        since     = snapshot_date - timedelta(days=lookback_days)
        sm_trades = self._get_sm_trades(market_id, snapshot_date, since)

        since_24h     = snapshot_date - timedelta(days=1)
        sm_trades_24h = self._get_sm_trades(market_id, snapshot_date, since_24h)
        sm_vol_24h    = float(sm_trades_24h["quantity"].sum()) \
                        if len(sm_trades_24h) > 0 else 0.0

        (direction, n_wallets, n_experts,
         net_flow, avg_yes, avg_no,
         expert_agreement) = self._compute_direction(sm_trades, market_category)

        conviction = self._compute_conviction(sm_trades, market_category)
        momentum   = self._compute_momentum(
            market_id, snapshot_date, direction, market_category
        )
        timing     = self._compute_timing(avg_yes, current_price_yes, direction)

        # Boost composite when category experts are involved
        expert_boost = 0.1 if n_experts >= 3 else 0.0

        w = SENTIMENT_WEIGHTS
        composite = (
            w["direction"]  * direction +
            w["conviction"] * conviction * (1 if direction >= 0 else -1) +
            w["momentum"]   * momentum +
            w["timing"]     * timing    * (1 if direction >= 0 else -1) +
            expert_boost    * expert_agreement * (1 if direction >= 0 else -1)
        )
        composite = max(-1.0, min(1.0, composite))

        # Entry: stricter when no experts present
        min_wallets = ENTRY_MIN_WALLETS if n_experts >= 2 else ENTRY_MIN_WALLETS + 2

        entry_signal = (
            abs(composite) >= ENTRY_COMPOSITE_MIN and
            conviction     >= ENTRY_CONVICTION_MIN and
            n_wallets      >= min_wallets          and
            momentum       >= 0
        )
        exit_signal = (
            abs(composite) < EXIT_COMPOSITE_MAX or
            momentum       < EXIT_MOMENTUM_DROP  or
            (conviction    < EXIT_CONVICTION_MIN and n_wallets >= 3)
        )
        strength = self._signal_strength(composite, conviction, n_wallets, n_experts)

        score = SentimentScore(
            market_id              = market_id,
            date                   = snapshot_date,
            category               = market_category,
            direction              = round(direction,  4),
            conviction             = round(conviction, 4),
            momentum               = round(momentum,   4),
            timing                 = round(timing,     4),
            composite              = round(composite,  4),
            sm_wallets_active      = n_wallets,
            sm_net_flow            = round(net_flow,   2),
            sm_volume_24h          = round(sm_vol_24h, 2),
            sm_avg_entry_yes       = round(avg_yes,    4),
            sm_avg_entry_no        = round(avg_no,     4),
            market_price_yes       = current_price_yes,
            category_expert_count  = n_experts,
            category_expert_agreement = round(expert_agreement, 4),
            entry_signal           = entry_signal,
            exit_signal            = exit_signal,
            signal_strength        = strength,
        )

        self._cache[cache_key] = score
        return score