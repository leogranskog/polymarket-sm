"""
Deep analysis of backtest results.
Run after backtest completes.
Usage: python -m backtest.analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import polars as pl
import polars.selectors as cs
from pathlib import Path
from config import PROC_DIR


def load_results():
    trades_path = PROC_DIR / "backtest_results" / "closed_trades.csv"
    equity_path = PROC_DIR / "backtest_results" / "equity_curve.csv"

    if not trades_path.exists():
        print("❌ No backtest results found. Run backtest first.")
        return None, None

    trades = pl.read_csv(trades_path)
    equity = pl.read_csv(equity_path)
    print(f"  Loaded {len(trades):,} trades, {len(equity):,} equity snapshots")
    return trades, equity


def monthly_breakdown(trades: pl.DataFrame, equity: pl.DataFrame):
    print("\n── Monthly Performance ─────────────────────────────────")

    equity = equity.with_columns(
        pl.col("date").str.slice(0, 7).alias("month")
    )

    monthly = (
        equity
        .group_by("month")
        .agg([
            pl.last("portfolio_value").alias("end_pv"),
            pl.first("portfolio_value").alias("start_pv"),
        ])
        .with_columns(
            ((pl.col("end_pv") - pl.col("start_pv")) /
              pl.col("start_pv") * 100).alias("month_return_pct")
        )
        .sort("month")
    )

    print(f"  {'Month':<10} {'Return':>8}  {'End PV':>10}  Bar")
    print(f"  {'─'*10} {'─'*8}  {'─'*10}  {'─'*20}")
    for row in monthly.iter_rows(named=True):
        ret  = row["month_return_pct"]
        bar  = ("█" * int(abs(ret))) if abs(ret) < 20 else "█" * 20
        sign = "+" if ret >= 0 else ""
        color_open  = ""
        color_close = ""
        print(f"  {row['month']:<10} {sign}{ret:>7.2f}%  "
              f"${row['end_pv']:>9,.0f}  {bar}")

    best  = monthly.sort("month_return_pct", descending=True).head(1)
    worst = monthly.sort("month_return_pct").head(1)
    print(f"\n  Best month:  {best['month'][0]}  "
          f"{best['month_return_pct'][0]:+.2f}%")
    print(f"  Worst month: {worst['month'][0]}  "
          f"{worst['month_return_pct'][0]:+.2f}%")


def category_deep_dive(trades: pl.DataFrame):
    print("\n── Category Deep Dive ──────────────────────────────────")

    cats = (
        trades
        .group_by("category")
        .agg([
            pl.len().alias("n_trades"),
            pl.sum("net_pnl").alias("total_pnl"),
            pl.mean("net_pnl").alias("avg_pnl"),
            (pl.col("net_pnl") > 0).sum().alias("wins"),
            pl.mean("entry_price").alias("avg_entry"),
            pl.mean("exit_price").alias("avg_exit"),
            pl.mean("holding_days").alias("avg_hold"),
            pl.sum("entry_fee").alias("total_fees"),
        ])
        .with_columns([
            (pl.col("wins") / pl.col("n_trades") * 100).alias("win_rate"),
            (pl.col("avg_exit") - pl.col("avg_entry")).alias("avg_price_move"),
        ])
        .sort("total_pnl", descending=True)
    )

    print(f"\n  {'Category':<12} {'N':>4} {'WR':>6} {'Avg Hold':>9} "
          f"{'Avg Entry':>10} {'Avg Exit':>9} {'Total PnL':>10}")
    print(f"  {'─'*12} {'─'*4} {'─'*6} {'─'*9} {'─'*10} {'─'*9} {'─'*10}")

    for row in cats.iter_rows(named=True):
        print(f"  {row['category']:<12} "
              f"{row['n_trades']:>4} "
              f"{row['win_rate']:>5.1f}% "
              f"{row['avg_hold']:>8.1f}d "
              f"{row['avg_entry']:>10.3f} "
              f"{row['avg_exit']:>9.3f} "
              f"${row['total_pnl']:>+9.2f}")


def exit_reason_analysis(trades: pl.DataFrame):
    print("\n── Exit Reason Analysis ────────────────────────────────")

    reasons = (
        trades
        .group_by("exit_reason")
        .agg([
            pl.len().alias("n"),
            pl.sum("net_pnl").alias("total_pnl"),
            pl.mean("net_pnl").alias("avg_pnl"),
            pl.mean("holding_days").alias("avg_hold"),
            (pl.col("net_pnl") > 0).sum().alias("wins"),
        ])
        .with_columns(
            (pl.col("wins") / pl.col("n") * 100).alias("win_rate")
        )
        .sort("total_pnl", descending=True)
    )

    print(f"\n  {'Exit Reason':<22} {'N':>4} {'WR':>6} "
          f"{'Avg Hold':>9} {'Avg PnL':>9} {'Total PnL':>10}")
    print(f"  {'─'*22} {'─'*4} {'─'*6} {'─'*9} {'─'*9} {'─'*10}")

    for row in reasons.iter_rows(named=True):
        print(f"  {row['exit_reason']:<22} "
              f"{row['n']:>4} "
              f"{row['win_rate']:>5.1f}% "
              f"{row['avg_hold']:>8.1f}d "
              f"${row['avg_pnl']:>+8.2f} "
              f"${row['total_pnl']:>+9.2f}")

    # Key insight
    stops = trades.filter(pl.col("exit_reason") == "stop_loss")
    if len(stops) > 0:
        print(f"\n  ⚠ Stop loss analysis:")
        print(f"    Count:      {len(stops)}")
        print(f"    Avg loss:   ${float(stops['net_pnl'].mean()):+.2f}")
        print(f"    Avg entry:  {float(stops['entry_price'].mean()):.3f}")
        print(f"    Avg exit:   {float(stops['exit_price'].mean()):.3f}")
        print(f"    Avg days held before stop: "
              f"{float(stops['holding_days'].mean()):.1f}")
        print(f"    → Positions dropped "
              f"{(1 - float(stops['exit_price'].mean()) / float(stops['entry_price'].mean()))*100:.1f}%"
              f" before stop fired")


def signal_quality(trades: pl.DataFrame):
    print("\n── Signal Quality Analysis ─────────────────────────────")

    # Did YES bets go up? Did NO bets go down?
    yes_trades = trades.filter(pl.col("side") == "YES")
    no_trades  = trades.filter(pl.col("side") == "NO")

    if len(yes_trades) > 0:
        yes_correct = yes_trades.filter(
            pl.col("exit_price") > pl.col("entry_price")
        )
        print(f"\n  YES bets: {len(yes_trades)} total")
        print(f"    Price moved up:   {len(yes_correct)} "
              f"({len(yes_correct)/len(yes_trades)*100:.1f}%)")
        print(f"    Avg entry: {float(yes_trades['entry_price'].mean()):.3f}")
        print(f"    Avg exit:  {float(yes_trades['exit_price'].mean()):.3f}")

    if len(no_trades) > 0:
        no_correct = no_trades.filter(
            pl.col("exit_price") < pl.col("entry_price")
        )
        print(f"\n  NO bets: {len(no_trades)} total")
        print(f"    Price moved down: {len(no_correct)} "
              f"({len(no_correct)/len(no_trades)*100:.1f}%)")


def position_sizing_analysis(trades: pl.DataFrame):
    print("\n── Position Sizing Analysis ────────────────────────────")

    print(f"\n  Cost basis distribution:")
    stats = trades["cost_basis"].describe()
    print(f"    Min:    ${float(trades['cost_basis'].min()):>8.2f}")
    print(f"    Median: ${float(trades['cost_basis'].median()):>8.2f}")
    print(f"    Mean:   ${float(trades['cost_basis'].mean()):>8.2f}")
    print(f"    Max:    ${float(trades['cost_basis'].max()):>8.2f}")

    # Are bigger bets more profitable?
    median_size = float(trades["cost_basis"].median())
    big   = trades.filter(pl.col("cost_basis") >  median_size)
    small = trades.filter(pl.col("cost_basis") <= median_size)

    print(f"\n  Large bets (>${median_size:.0f}):")
    print(f"    N: {len(big)}  "
          f"WR: {(big['net_pnl']>0).mean()*100:.1f}%  "
          f"Avg PnL: ${float(big['net_pnl'].mean()):+.2f}")
    print(f"  Small bets (<=${median_size:.0f}):")
    print(f"    N: {len(small)}  "
          f"WR: {(small['net_pnl']>0).mean()*100:.1f}%  "
          f"Avg PnL: ${float(small['net_pnl'].mean()):+.2f}")


def risk_analysis(equity: pl.DataFrame):
    print("\n── Risk Analysis ───────────────────────────────────────")

    pv = equity["portfolio_value"].to_list()

    # Drawdown periods
    peak = pv[0]
    peak_idx = 0
    max_dd = 0.0
    max_dd_start = 0
    max_dd_end   = 0
    in_drawdown  = False
    dd_start     = 0

    for i, v in enumerate(pv):
        if v > peak:
            peak = v
            peak_idx = i
            in_drawdown = False
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
            max_dd_start = peak_idx
            max_dd_end   = i

    dates = equity["date"].to_list()

    print(f"\n  Max drawdown:    {max_dd*100:.2f}%")
    print(f"  DD period:       {dates[max_dd_start]} → {dates[max_dd_end]}")
    dd_days = max_dd_end - max_dd_start
    print(f"  DD duration:     {dd_days} days")

    # Consecutive losing days
    daily_rets = [(pv[i] - pv[i-1]) / pv[i-1] for i in range(1, len(pv))]
    max_consec_loss = 0
    cur_consec = 0
    for r in daily_rets:
        if r < 0:
            cur_consec += 1
            max_consec_loss = max(max_consec_loss, cur_consec)
        else:
            cur_consec = 0
    print(f"  Max consecutive losing days: {max_consec_loss}")

    # Volatility
    import math
    mean_r = sum(daily_rets) / len(daily_rets)
    std_r  = (sum((r-mean_r)**2 for r in daily_rets)/len(daily_rets))**0.5
    sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0
    print(f"  Daily volatility: {std_r*100:.3f}%")
    print(f"  Annualised Sharpe: {sharpe:.3f}")

    # Up vs down months
    pos_months = sum(1 for r in daily_rets if r > 0)
    neg_months = sum(1 for r in daily_rets if r < 0)
    print(f"  Positive days: {pos_months} | Negative days: {neg_months}")


def known_issues(trades: pl.DataFrame, equity: pl.DataFrame):
    print("\n── Known Issues & Fixes Needed ────────────────────────")

    issues = []

    # Check stop loss rate
    stops = trades.filter(pl.col("exit_reason") == "stop_loss")
    stop_rate = len(stops) / len(trades) if len(trades) > 0 else 0
    if stop_rate > 0.30:
        issues.append(
            f"⚠ High stop loss rate: {stop_rate*100:.1f}% of trades "
            f"→ signals entering too early or wrong direction"
        )

    # Check win rate
    wr = (trades["net_pnl"] > 0).mean()
    if wr < 0.35:
        issues.append(
            f"⚠ Low win rate: {wr*100:.1f}% "
            f"→ signal quality needs improvement"
        )

    # Check category concentration
    cat_counts = trades.group_by("category").len().sort("len", descending=True)
    top_cat    = cat_counts[0]["category"][0]
    top_pct    = cat_counts[0]["len"][0] / len(trades) * 100
    if top_pct > 60:
        issues.append(
            f"⚠ Over-concentrated in {top_cat}: {top_pct:.1f}% of trades "
            f"→ need better diversification"
        )

    # Check if fees are being calculated
    total_fees = float(trades["entry_fee"].sum())
    if total_fees == 0:
        issues.append(
            "⚠ Fees show $0 — fee model not triggering "
            "(expected for pre-2026 trades, but verify)"
        )

    # Check holding period
    avg_hold = float(trades["holding_days"].mean())
    if avg_hold > 60:
        issues.append(
            f"⚠ Long avg holding period: {avg_hold:.0f} days "
            f"→ capital tied up, consider shorter-duration markets"
        )

    # Check lookahead
    issues.append(
        "⚠ Lookahead bias: ML model trained on 2023 data only ✅ "
        "but SM universe still uses full-period PnL for category scoring"
    )

    issues.append(
        "⚠ Proxy trades: SM sentiment uses OHLCV-derived proxies, "
        "not actual wallet trades per market"
    )

    issues.append(
        "⚠ No rolling wallet scores: SM wallets scored once, "
        "not updated monthly as new data arrives"
    )

    print()
    for issue in issues:
        print(f"  {issue}")

    print(f"\n  Priority fixes:")
    print(f"  1. Use real trades per market for sentiment (not proxies)")
    print(f"  2. Build rolling monthly wallet rescoring")
    print(f"  3. Add category-specific signal thresholds")
    print(f"  4. Validate that 2025 results match 2024 pattern")


def run():
    print("\n" + "="*58)
    print("  DEEP BACKTEST ANALYSIS")
    print("="*58)

    trades, equity = load_results()
    if trades is None:
        return

    print(f"\n  Total trades:    {len(trades):,}")
    print(f"  Date range:      "
          f"{trades['entry_date'].min()} → {trades['exit_date'].max()}")
    print(f"  Starting capital: $10,000")

    monthly_breakdown(trades, equity)
    category_deep_dive(trades)
    exit_reason_analysis(trades)
    signal_quality(trades)
    position_sizing_analysis(trades)
    risk_analysis(equity)
    known_issues(trades, equity)

    print("\n" + "="*58)
    print("  Analysis complete.")
    print("  Fix priority issues before going live.")
    print("="*58)


if __name__ == "__main__":
    run()