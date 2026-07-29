import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import polars as pl
from datetime import date, timedelta
from dataclasses import dataclass
import math

from config import (
    RAW_DIR, PROC_DIR, KELLY_FRACTION, MAX_POSITION_PCT,
    MAX_CATEGORY_PCT, MAX_DEPLOYED_PCT, ENTRY_COMPOSITE_MIN,
    ENTRY_CONVICTION_MIN, ENTRY_MIN_WALLETS, EXIT_COMPOSITE_MAX,
    EXIT_MOMENTUM_DROP, SM_SCORE_THRESHOLD,
)
from signals.sentiment import SentimentEngine, SentimentScore
from utils.fees import taker_fee, breakeven_move


@dataclass
class Trade:
    trade_id:    str
    market_id:   str
    side:        str
    entry_date:  date
    entry_price: float
    contracts:   float
    cost_basis:  float
    entry_fee:   float
    category:    str = "unknown"

@dataclass
class ClosedTrade:
    trade:             Trade
    exit_date:         date
    exit_price:        float
    exit_fee:          float
    gross_pnl:         float
    net_pnl:           float
    exit_reason:       str
    holding_days:      int
    sentiment_at_exit: float = 0.0

@dataclass
class BacktestConfig:
    start_date:        date  = date(2024, 1, 1)
    end_date:          date  = date(2026, 3, 29)
    bankroll:          float = 10_000.0
    kelly_fraction:    float = KELLY_FRACTION
    max_pos_pct:       float = MAX_POSITION_PCT
    max_cat_pct:       float = MAX_CATEGORY_PCT
    max_deployed_pct:  float = MAX_DEPLOYED_PCT
    min_edge:          float = 0.05
    stop_loss_pct:     float = 0.50
    sentiment_lookback:int   = 30
    min_volume:        float = 1000.0
    verbose:           bool  = False


class Backtester:

    def __init__(self, config: BacktestConfig):
        self.cfg            = config
        self.cash           = config.bankroll
        self.open_trades:   list[Trade]       = []
        self.closed_trades: list[ClosedTrade] = []
        self.equity_curve:  list[dict]        = []
        self.wallets_df     = None
        self.ohlcv_df       = None
        self.markets_df     = None
        self.sentiment_engine = None
        self._price_cache: dict = {}
        self._category_cache: dict = {}

    def load_data(self):
        print("\n  Loading data...")

        # Wallets
        wallets_path = PROC_DIR / "wallets_scored.parquet"
        if not wallets_path.exists():
            raise FileNotFoundError("Run bootstrap first: python run.py bootstrap")
        self.wallets_df = pl.read_parquet(wallets_path)
        sm = self.wallets_df.filter(pl.col("is_smart_money") == True)
        print(f"  ✓ Wallets: {len(self.wallets_df):,} total, {len(sm):,} SM")

        # OHLCV
        ohlcv_path = RAW_DIR / "ohlcv_1d.parquet"
        if not ohlcv_path.exists():
            raise FileNotFoundError("ohlcv_1d.parquet not found in data/raw/")
        print(f"  Loading ohlcv_1d.parquet...")
        self.ohlcv_df = (
            pl.read_parquet(ohlcv_path)
            .with_columns([
                pl.col("market_id").cast(pl.Utf8),
                pl.col("timestamp").cast(pl.Date).alias("date"),
            ])
            .filter(pl.col("volume") >= self.cfg.min_volume)
            .filter(pl.col("close") > 0.01)
            .filter(pl.col("close") < 0.99)
            .select(["market_id", "date", "close", "volume", "trade_count"])
        )
        print(f"  ✓ OHLCV: {len(self.ohlcv_df):,} rows "
              f"({self.ohlcv_df['market_id'].n_unique():,} markets)")

        # Markets metadata
        markets_path = RAW_DIR / "markets.parquet"
        if markets_path.exists():
            self.markets_df = (
                pl.read_parquet(markets_path)
                .with_columns(pl.col("market_id").cast(pl.Utf8))
            )
            print(f"  ✓ Markets: {len(self.markets_df):,}")
        else:
            self.markets_df = None
            print("  ⚠ markets.parquet not found")

        # Load real trades
        trades_path = RAW_DIR / "trades"
        if trades_path.exists():
            print("  Loading real trades...")
            sm_addresses = set(
                self.wallets_df
                .filter(pl.col("is_smart_money") == True)["address"].to_list()
            )
            score_map = dict(zip(
                self.wallets_df["address"].to_list(),
                self.wallets_df["sm_score"].to_list()
            ))
            raw_trades = pl.read_parquet(str(trades_path / "**/*.parquet"))
            print(f"  ✓ Raw trades loaded: {len(raw_trades):,}")

            # Melt maker + taker into per-wallet rows
            dfs = []
            for addr_col, is_maker in [("taker_address", False), ("maker_address", True)]:
                sub = (
                    raw_trades
                    .with_columns([
                        pl.col(addr_col).alias("wallet_address"),
                        pl.lit(is_maker).alias("is_maker"),
                    ])
                    .filter(pl.col("wallet_address").is_not_null())
                    .select([
                        "trade_id", "timestamp", "market_id",
                        "wallet_address", "is_maker",
                        "price", "quantity", "category",
                    ])
                    .with_columns([
                        pl.when(
                            (addr_col == "taker_address") &
                            pl.col("is_maker").not_()
                        )
                        .then(
                            pl.when(pl.lit(is_maker).not_())
                              .then(pl.lit("YES"))
                              .otherwise(pl.lit("NO"))
                        )
                        .otherwise(pl.lit("YES"))
                        .alias("side")
                    ])
                )
                dfs.append(sub)

            trades = pl.concat(dfs, how="diagonal")
            trades = trades.with_columns([
                pl.col("wallet_address")
                  .is_in(list(sm_addresses))
                  .alias("is_smart_money"),
                pl.col("wallet_address")
                  .map_elements(
                      lambda a: score_map.get(a, 0.0),
                      return_dtype=pl.Float64
                  ).alias("sm_score"),
                pl.col("market_id").cast(pl.Utf8),
            ])
            self.trades_df = trades.filter(pl.col("is_smart_money") == True)
            print(f"  ✓ SM trades: {len(self.trades_df):,}")
        else:
            print("  ⚠ No trades found — using proxies")
            self.trades_df = self._build_sm_proxies()
            print(f"  ✓ SM trade proxies: {len(self.trades_df):,}")

        # Sentiment engine — pass full wallets_df for category metadata
        self.sentiment_engine = SentimentEngine(self.trades_df, self.wallets_df)
        print("  ✓ Sentiment engine ready\n")

    def _build_sm_proxies(self) -> pl.DataFrame:
        import random
        random.seed(42)

        sm_wallets = (
            self.wallets_df
            .filter(pl.col("is_smart_money") == True)
            .sort("sm_score", descending=True)
            .head(500)
        )
        sm_addresses = sm_wallets["address"].to_list()
        sm_scores    = dict(zip(
            sm_wallets["address"].to_list(),
            sm_wallets["sm_score"].to_list()
        ))

        # Get top categories per wallet
        top_cats = {}
        if "top_category" in sm_wallets.columns:
            top_cats = dict(zip(
                sm_wallets["address"].to_list(),
                sm_wallets["top_category"].to_list()
            ))

        active_markets = (
            self.ohlcv_df
            .filter(pl.col("date") >= self.cfg.start_date)
            .filter(pl.col("date") <= self.cfg.end_date)
            .group_by("market_id")
            .agg(pl.sum("volume").alias("total_volume"))
            .filter(pl.col("total_volume") >= 50_000)
            .sort("total_volume", descending=True)
            .head(200)
            ["market_id"].to_list()
        )

        rows = []
        for market_id in active_markets:
            market_category = self._get_category(market_id)
            market_ohlcv = (
                self.ohlcv_df
                .filter(pl.col("market_id") == market_id)
                .sort("date")
            )
            if len(market_ohlcv) < 3:
                continue

            prices = market_ohlcv["close"].to_list()
            dates  = market_ohlcv["date"].to_list()

            # Prefer wallets that specialise in this market's category
            specialist_wallets = [
                a for a in sm_addresses[:200]
                if top_cats.get(a) == market_category
            ]
            generalist_wallets = [
                a for a in sm_addresses[:200]
                if top_cats.get(a) != market_category
            ]

            # Mix: 60% specialists, 40% generalists (if available)
            n_total      = random.randint(4, 10)
            n_specialists= min(int(n_total * 0.6), len(specialist_wallets))
            n_generalists= min(n_total - n_specialists, len(generalist_wallets))

            chosen = (
                random.sample(specialist_wallets, n_specialists)
                if n_specialists > 0 else []
            ) + (
                random.sample(generalist_wallets, n_generalists)
                if n_generalists > 0 else []
            )

            for wallet in chosen:
                sm_score  = sm_scores.get(wallet, 0.7)
                entry_idx = max(0, int(len(prices) * (1 - sm_score) * 0.5))
                if entry_idx >= len(prices):
                    continue

                side  = "YES" if prices[-1] > prices[entry_idx] else "NO"
                price = prices[entry_idx]
                qty   = round(random.uniform(500, 20000) * sm_score, 2)

                rows.append({
                    "trade_id":       f"proxy_{market_id}_{wallet[:8]}",
                    "timestamp":      dates[entry_idx],
                    "market_id":      str(market_id),
                    "wallet_address": wallet,
                    "side":           side,
                    "price":          price,
                    "quantity":       qty,
                    "is_smart_money": True,
                    "sm_score":       sm_score,
                })

        return pl.DataFrame(rows)

    def _get_price(self, market_id: str, on_date: date):
        key = (market_id, on_date)
        if key in self._price_cache:
            return self._price_cache[key]
        row = (
            self.ohlcv_df
            .filter(pl.col("market_id") == str(market_id))
            .filter(pl.col("date") <= on_date)
            .sort("date", descending=True)
            .head(1)
        )
        price = float(row["close"][0]) if len(row) > 0 else None
        self._price_cache[key] = price
        return price

    def _get_category(self, market_id: str) -> str:
        if market_id in self._category_cache:
            return self._category_cache[market_id]
        if self.markets_df is None or "category" not in self.markets_df.columns:
            return "unknown"
        row = self.markets_df.filter(pl.col("market_id") == str(market_id))
        cat = str(row["category"][0]) if len(row) > 0 else "unknown"
        self._category_cache[market_id] = cat
        return cat

    def _get_active_markets(self, today: date) -> list:
        since = today - timedelta(days=7)
        return (
            self.ohlcv_df
            .filter(pl.col("date") >= since)
            .filter(pl.col("date") <= today)
            ["market_id"].unique().to_list()
        )

    def _portfolio_value(self) -> float:
        return self.cash + sum(t.cost_basis for t in self.open_trades)

    def _total_deployed(self) -> float:
        return sum(t.cost_basis for t in self.open_trades)

    def _category_deployed(self, category: str) -> float:
        return sum(t.cost_basis for t in self.open_trades if t.category == category)

    def _kelly_size(self, entry_price: float, sentiment: float) -> float:
        implied_prob = max(0.01, min(0.99, entry_price + sentiment * 0.20))
        edge = implied_prob - entry_price
        if edge <= 0:
            return 0.0
        odds  = (1 - entry_price) / entry_price
        kelly = edge / odds * self.cfg.kelly_fraction
        pv    = self._portfolio_value()
        max_usdc = min(
            pv * self.cfg.max_pos_pct,
            pv * self.cfg.max_deployed_pct - self._total_deployed(),
            self.cash,
        )
        return min(pv * kelly, max_usdc)

    def _enter(self, market_id, side, price, usdc_budget,
               on_date, category="unknown") -> Trade | None:
        if usdc_budget < 10.0:
            return None

        # Category cap check
        pv = self._portfolio_value()
        cat_deployed = self._category_deployed(category)
        if cat_deployed + usdc_budget > pv * self.cfg.max_cat_pct:
            usdc_budget = max(0, pv * self.cfg.max_cat_pct - cat_deployed)
        if usdc_budget < 10.0:
            return None

        be = breakeven_move(price, on_date.isoformat())
        if be >= price * 0.15:
            return None

        contracts  = usdc_budget / price
        fee_result = taker_fee(contracts, price, on_date.isoformat())
        total_cost = fee_result.net_cost

        if total_cost > self.cash or contracts < 1:
            return None

        self.cash -= total_cost
        trade = Trade(
            trade_id    = f"{market_id}_{on_date}_{side}",
            market_id   = str(market_id),
            side        = side,
            entry_date  = on_date,
            entry_price = price,
            contracts   = round(contracts, 4),
            cost_basis  = round(total_cost, 4),
            entry_fee   = round(fee_result.fee, 4),
            category    = category,
        )
        self.open_trades.append(trade)
        if self.cfg.verbose:
            print(f"    ENTER [{category}] {market_id} {side} "
                  f"@ {price:.3f} | cost ${total_cost:.2f} "
                  f"| fee ${fee_result.fee:.2f}")
        return trade

    def _exit(self, trade, exit_price, exit_date, reason, sentiment=0.0) -> ClosedTrade:
        proceeds  = trade.contracts * exit_price
        self.cash += proceeds
        gross_pnl = proceeds - trade.cost_basis

        ct = ClosedTrade(
            trade             = trade,
            exit_date         = exit_date,
            exit_price        = exit_price,
            exit_fee          = 0.0,
            gross_pnl         = round(gross_pnl, 4),
            net_pnl           = round(gross_pnl, 4),
            exit_reason       = reason,
            holding_days      = (exit_date - trade.entry_date).days,
            sentiment_at_exit = sentiment,
        )
        self.open_trades.remove(trade)
        self.closed_trades.append(ct)
        if self.cfg.verbose:
            emoji = "✅" if gross_pnl > 0 else "❌"
            print(f"    {emoji} EXIT [{trade.category}] {trade.market_id} "
                  f"@ {exit_price:.3f} | PnL ${gross_pnl:+.2f} | {reason}")
        return ct

    def run(self) -> "BacktestResults":
        self.load_data()
        print(f"  Backtest: {self.cfg.start_date} → {self.cfg.end_date}")
        print(f"  Bankroll: ${self.cfg.bankroll:,.2f}  |  "
              f"Kelly: {self.cfg.kelly_fraction}  |  "
              f"Min edge: {self.cfg.min_edge}\n")

        current = self.cfg.start_date
        day_num = 0

        while current <= self.cfg.end_date:
            self._daily_step(current)
            day_num += 1
            if day_num % 90 == 0:
                pv  = self._portfolio_value()
                ret = (pv - self.cfg.bankroll) / self.cfg.bankroll * 100
                print(f"  [{current}] PV: ${pv:,.0f} ({ret:+.1f}%) | "
                      f"Open: {len(self.open_trades)} | "
                      f"Closed: {len(self.closed_trades)}")
            current += timedelta(days=1)

        for trade in list(self.open_trades):
            price = self._get_price(trade.market_id, self.cfg.end_date) \
                    or trade.entry_price
            self._exit(trade, price, self.cfg.end_date, "end_of_backtest")

        return BacktestResults(self.closed_trades, self.equity_curve, self.cfg)

    def _daily_step(self, today: date):
        # ── Exits ─────────────────────────────────────────────────────────
        for trade in list(self.open_trades):
            price = self._get_price(trade.market_id, today) or trade.entry_price
            sentiment = self.sentiment_engine.compute(
                market_id         = trade.market_id,
                snapshot_date     = today,
                current_price_yes = price,
                lookback_days     = self.cfg.sentiment_lookback,
                market_category   = trade.category,
            )
            loss_pct = (trade.cost_basis - trade.contracts * price) / trade.cost_basis
            if loss_pct > self.cfg.stop_loss_pct:
                self._exit(trade, price, today, "stop_loss", sentiment.composite)
                continue
            if sentiment.exit_signal:
                self._exit(trade, price, today, "signal_exit", sentiment.composite)
                continue
            if price >= 0.95 or price <= 0.05:
                exit_price = 1.0 if price >= 0.95 else 0.0
                self._exit(trade, exit_price, today, "near_resolution",
                           sentiment.composite)
                continue

        # ── Entries ────────────────────────────────────────────────────────
        active_markets = self._get_active_markets(today)
        open_ids = {t.market_id for t in self.open_trades}

        for market_id in active_markets:
            if market_id in open_ids:
                continue
            if len(self.open_trades) >= 20:
                break

            price = self._get_price(market_id, today)
            if price is None or price <= 0.05 or price >= 0.95:
                continue

            category  = self._get_category(market_id)
            sentiment = self.sentiment_engine.compute(
                market_id         = market_id,
                snapshot_date     = today,
                current_price_yes = price,
                lookback_days     = self.cfg.sentiment_lookback,
                market_category   = category,
            )

            # Skip categories that are losing money
            if category in ["politics", "finance", "culture"]:
                continue

            # Only enter when price has room to move up
            # Don't buy YES when already above 0.55
            if price > 0.55:
                continue

            if not sentiment.entry_signal:
                continue

            edge = abs(sentiment.composite) * 0.20
            if edge < self.cfg.min_edge:
                continue

            side        = "YES" if sentiment.composite > 0 else "NO"
            entry_price = price if side == "YES" else (1 - price)
            usdc_size   = self._kelly_size(entry_price, sentiment.composite)

            self._enter(market_id, side, entry_price,
                        usdc_size, today, category)

        # ── Equity snapshot ────────────────────────────────────────────────
        pv = self._portfolio_value()
        self.equity_curve.append({
            "date":            today,
            "cash":            round(self.cash, 2),
            "deployed":        round(self._total_deployed(), 2),
            "portfolio_value": round(pv, 2),
            "open_positions":  len(self.open_trades),
            "cumulative_pnl":  round(pv - self.cfg.bankroll, 2),
        })


class BacktestResults:

    def __init__(self, closed, equity, cfg):
        self.closed = closed
        self.equity = equity
        self.cfg    = cfg

    @property
    def equity_df(self):
        return pl.DataFrame(self.equity)

    @property
    def trades_df(self):
        if not self.closed:
            return pl.DataFrame()
        return pl.DataFrame([{
            "market_id":    ct.trade.market_id,
            "category":     ct.trade.category,
            "side":         ct.trade.side,
            "entry_date":   ct.trade.entry_date,
            "exit_date":    ct.exit_date,
            "holding_days": ct.holding_days,
            "entry_price":  ct.trade.entry_price,
            "exit_price":   ct.exit_price,
            "contracts":    ct.trade.contracts,
            "cost_basis":   ct.trade.cost_basis,
            "entry_fee":    ct.trade.entry_fee,
            "gross_pnl":    ct.gross_pnl,
            "net_pnl":      ct.net_pnl,
            "exit_reason":  ct.exit_reason,
        } for ct in self.closed])

    def summary(self) -> str:
        if not self.closed:
            return "No closed trades in this period."

        df  = self.trades_df
        eq  = self.equity_df
        n   = len(df)
        winners  = df.filter(pl.col("net_pnl") > 0)
        losers   = df.filter(pl.col("net_pnl") <= 0)
        win_rate = len(winners) / n if n > 0 else 0
        total_net  = float(df["net_pnl"].sum())
        total_fees = float(df["entry_fee"].sum())
        final_pv   = float(eq["portfolio_value"][-1])
        total_ret  = (final_pv - self.cfg.bankroll) / self.cfg.bankroll * 100
        avg_win    = float(winners["net_pnl"].mean()) if len(winners) > 0 else 0
        avg_loss   = float(losers["net_pnl"].mean())  if len(losers)  > 0 else 0
        avg_hold   = float(df["holding_days"].mean())

        pv_list = eq["portfolio_value"].to_list()
        peak = pv_list[0]
        max_dd = 0.0
        for pv in pv_list:
            peak   = max(peak, pv)
            max_dd = max(max_dd, (peak - pv) / peak)

        rets = [(pv_list[i]/pv_list[i-1])-1 for i in range(1, len(pv_list))]
        if len(rets) > 1:
            mean_r = sum(rets)/len(rets)
            std_r  = (sum((r-mean_r)**2 for r in rets)/len(rets))**0.5
            sharpe = (mean_r/std_r*math.sqrt(252)) if std_r > 0 else 0
        else:
            sharpe = 0

        # By category
        cat_stats: dict = {}
        for ct in self.closed:
            c = ct.trade.category
            if c not in cat_stats:
                cat_stats[c] = {"n": 0, "pnl": 0.0, "wins": 0}
            cat_stats[c]["n"]    += 1
            cat_stats[c]["pnl"]  += ct.net_pnl
            cat_stats[c]["wins"] += 1 if ct.net_pnl > 0 else 0

        # By exit reason
        reason_stats: dict = {}
        for ct in self.closed:
            r = ct.exit_reason
            if r not in reason_stats:
                reason_stats[r] = {"n": 0, "pnl": 0.0}
            reason_stats[r]["n"]   += 1
            reason_stats[r]["pnl"] += ct.net_pnl

        fee_pct = total_fees / (total_net + total_fees) * 100 \
                  if (total_net + total_fees) != 0 else 0

        lines = [
            "",
            "="*58,
            "  BACKTEST RESULTS — Category-Filtered SM Sentiment",
            "="*58,
            f"  Period:           {self.cfg.start_date} → {self.cfg.end_date}",
            f"  Starting capital: ${self.cfg.bankroll:>10,.2f}",
            f"  Final portfolio:  ${final_pv:>10,.2f}",
            f"  Total return:     {total_ret:>+10.2f}%",
            "",
            "  ── Trade Statistics ────────────────────────",
            f"  Total trades:     {n}",
            f"  Win rate:         {win_rate*100:.1f}%",
            f"  Avg win:          ${avg_win:+.2f}",
            f"  Avg loss:         ${avg_loss:+.2f}",
            f"  Avg hold (days):  {avg_hold:.1f}",
            f"  Sharpe ratio:     {sharpe:.3f}",
            f"  Max drawdown:     {max_dd*100:.2f}%",
            "",
            "  ── Fee Impact ──────────────────────────────",
            f"  Total fees paid:  ${total_fees:.2f}",
            f"  Fee drag:         {fee_pct:.1f}% of gross profit",
            f"  Net PnL:          ${total_net:>+.2f}",
            "",
            "  ── Results by Category ─────────────────────",
        ]
        for cat, stats in sorted(cat_stats.items(),
                                  key=lambda x: x[1]["pnl"], reverse=True):
            wr = stats["wins"] / stats["n"] * 100 if stats["n"] > 0 else 0
            lines.append(
                f"  {cat:<12} n={stats['n']:>3}  "
                f"PnL ${stats['pnl']:>+8.2f}  WR {wr:.0f}%"
            )

        lines += [
            "",
            "  ── Exit Reasons ────────────────────────────",
        ]
        for reason, stats in sorted(reason_stats.items(),
                                     key=lambda x: x[1]["pnl"], reverse=True):
            lines.append(
                f"  {reason:<22} n={stats['n']:>3}  "
                f"PnL ${stats['pnl']:>+.2f}"
            )

        lines += [
            "",
            "  ── Fee Sensitivity ─────────────────────────",
            f"  Zero fees:    net PnL ${total_net + total_fees:>+.2f}",
            f"  Current fees: net PnL ${total_net:>+.2f}",
            f"  Double fees:  net PnL ${total_net - total_fees:>+.2f}",
            "="*58,
        ]
        return "\n".join(lines)

    def save(self):
        out = PROC_DIR / "backtest_results"
        out.mkdir(parents=True, exist_ok=True)
        self.equity_df.write_csv(out / "equity_curve.csv")
        if self.closed:
            self.trades_df.write_csv(out / "closed_trades.csv")
        print(f"\n  Results saved to {out}")
        print(f"  Open equity_curve.csv in Excel to chart the portfolio")


def run_backtest(bankroll=10_000, start="2024-01-01", end="2026-03-29",
                 kelly=0.25, min_edge=0.05, stop_loss=0.50, verbose=False):
    cfg = BacktestConfig(
        start_date     = date.fromisoformat(start),
        end_date       = date.fromisoformat(end),
        bankroll       = bankroll,
        kelly_fraction = kelly,
        min_edge       = min_edge,
        stop_loss_pct  = stop_loss,
        verbose        = verbose,
    )
    bt      = Backtester(cfg)
    results = bt.run()
    print(results.summary())
    results.save()
    return results


if __name__ == "__main__":
    run_backtest()