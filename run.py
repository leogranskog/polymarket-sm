"""
Polymarket Smart Money System — main entry point.

Research pipeline (Paper 1, leak-free):
  python run.py bootstrap       Download base dataset files
  python run.py pit             Build point-in-time features + CLV labels
  python run.py research2       Leak-free ML pipeline (Results 2+3)
  python run.py persistence     Result 1: CLV persistence
  python run.py specialization  Result 4: specialists vs generalists
  python run.py insider         Insider/skill/luck decomposition

Paper 2 / legacy:
  python run.py backtest        Trading strategy backtest
  python run.py sentiment       Sentiment engine demo
  python run.py research        v1 pipeline (ARCHIVED — lookahead bias)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


def cmd_bootstrap():
    print("\n🚀 Downloading dataset\n")
    from bootstrap.download_dataset import download_all
    download_all()
    print("\n✅ Done. Next: python run.py pit")


def cmd_preview():
    from bootstrap.download_dataset import preview_subset
    from config import HF_SUBSETS
    for subset in HF_SUBSETS:
        try:
            preview_subset(subset)
        except Exception as e:
            print(f"  ⚠ {subset}: {e}")


def cmd_pit():
    from research.pit_features import run_all
    run_all()


def cmd_research2(argv):
    import argparse
    p = argparse.ArgumentParser(prog="run.py research2")
    p.add_argument("--trials",   type=int, default=50,
                   help="Optuna trials for LightGBM tuning")
    p.add_argument("--true-oos", action="store_true",
                   help="Evaluate on 2025-H2. USE ONCE, at the very end.")
    args = p.parse_args(argv)
    from research.ml_pipeline_v2 import run as v2_run
    v2_run(n_trials=args.trials, true_oos=args.true_oos)


def cmd_backtest(argv):
    import argparse
    p = argparse.ArgumentParser(prog="run.py backtest")
    p.add_argument("--bankroll",  type=float, default=10_000)
    p.add_argument("--start",     type=str,   default="2024-01-01")
    p.add_argument("--end",       type=str,   default="2024-12-31")
    p.add_argument("--kelly",     type=float, default=0.25)
    p.add_argument("--min-edge",  type=float, default=0.05)
    p.add_argument("--stop-loss", type=float, default=0.50)
    p.add_argument("--verbose",   action="store_true")
    args = p.parse_args(argv)
    from backtest.backtester import run_backtest
    run_backtest(
        bankroll  = args.bankroll,
        start     = args.start,
        end       = args.end,
        kelly     = args.kelly,
        min_edge  = args.min_edge,
        stop_loss = args.stop_loss,
        verbose   = args.verbose,
    )


def cmd_sentiment_demo():
    from datetime import date, timedelta
    from pathlib import Path
    import polars as pl
    import random

    from signals.sentiment import SentimentEngine

    proc_dir = Path(__file__).parent / "data/processed"
    wallets_path = proc_dir / "wallets_scored.parquet"
    if not wallets_path.exists():
        print("❌ No wallets_scored.parquet found.")
        return

    random.seed(42)
    n_demo = 500
    wallets = pl.read_parquet(wallets_path)
    sm_wallets = wallets.filter(pl.col("is_smart_money") == True)
    print(f"\n  SM wallet universe: {len(sm_wallets):,} wallets")
    sm_addresses = sm_wallets["address"].to_list()[:50]
    demo_markets = [f"market_{i:03d}" for i in range(5)]
    demo_trades = pl.DataFrame({
        "trade_id":       [f"t{i}" for i in range(n_demo)],
        "timestamp":      [date(2025, 1, 1) + timedelta(days=random.randint(0, 60))
                           for _ in range(n_demo)],
        "market_id":      [random.choice(demo_markets) for _ in range(n_demo)],
        "wallet_address": [random.choice(sm_addresses) for _ in range(n_demo)],
        "side":           [random.choice(["YES", "NO"]) for _ in range(n_demo)],
        "price":          [round(random.uniform(0.2, 0.8), 3) for _ in range(n_demo)],
        "quantity":       [round(random.uniform(100, 5000), 2) for _ in range(n_demo)],
        "is_smart_money": [True] * n_demo,
        "sm_score":       [round(random.uniform(0.6, 1.0), 3) for _ in range(n_demo)],
    })
    wallets_mini = sm_wallets.select(["address", "sm_score", "is_smart_money"])
    engine = SentimentEngine(demo_trades, wallets_mini)
    print("\n" + "=" * 55)
    print("  SM Sentiment Demo")
    print("=" * 55)
    for market_id in demo_markets[:3]:
        score = engine.compute(
            market_id=market_id,
            snapshot_date=date(2025, 2, 15),
            current_price_yes=round(random.uniform(0.3, 0.7), 2),
        )
        print(score.summary())


def cmd_research_v1(argv):
    """Archived v1 pipeline — kept only for the leak-comparison exhibit."""
    print("⚠  WARNING: v1 uses terminal-snapshot features with lookahead")
    print("   contamination. For real results use: python run.py research2\n")
    import argparse
    p = argparse.ArgumentParser(prog="run.py research")
    p.add_argument("--trials",   type=int, default=50)
    p.add_argument("--true-oos", action="store_true")
    args = p.parse_args(argv)
    from research.ml_pipeline import run as research_run
    research_run(optimize=True, n_trials=args.trials, true_oos=args.true_oos)


HELP = """
Polymarket Smart Money System
==============================

RESEARCH PIPELINE (Paper 1 — leak-free):
  python run.py bootstrap        Download base dataset files
  python run.py pit              Build PIT features + CLV labels (one-time)
  python run.py research2        Main ML pipeline -> figures_v2/ tables_v2/
  python run.py persistence      Result 1: CLV persistence
  python run.py specialization   Result 4: specialists vs generalists
  python run.py insider          Insider/skill/luck decomposition

Research options:
  python run.py research2 --trials 50        Standard run
  python run.py research2 --trials 100       Final run (more tuning)
  python run.py research2 --true-oos         2025-H2. ONE LOOK ONLY.

PAPER 2 / LEGACY:
  python run.py backtest         Trading strategy backtest
  python run.py sentiment        Sentiment engine demo
  python run.py research         v1 pipeline (ARCHIVED — lookahead bias)
  python run.py preview          Preview downloaded data

Order of operations:
  1. pit
  2. research2
  3. persistence / specialization / insider
  4. research2 --true-oos   (once, when everything is frozen)
"""


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "bootstrap":
        cmd_bootstrap()
    elif cmd == "preview":
        cmd_preview()
    elif cmd == "pit":
        cmd_pit()
    elif cmd == "research2":
        cmd_research2(sys.argv[2:])
    elif cmd == "persistence":
        from research.persistence import run as r
        r()
    elif cmd == "specialization":
        from research.specialization import run as r
        r()
    elif cmd == "insider":
        from research.insider_decomposition import run as r
        r()
    elif cmd == "backtest":
        cmd_backtest(sys.argv[2:])
    elif cmd == "sentiment":
        cmd_sentiment_demo()
    elif cmd == "research":
        cmd_research_v1(sys.argv[2:])
    elif cmd == "stability":
        import research.spec_stability
    elif cmd == "continuous":
        import argparse
        p = argparse.ArgumentParser(prog="run.py continuous")
        p.add_argument("--trials", type=int, default=50)
        args = p.parse_args(sys.argv[2:])
        from research.ml_pipeline_v3_continuous import run as r
        r(n_trials=args.trials)
    elif cmd == "adjacent":
        import argparse
        p = argparse.ArgumentParser(prog="run.py adjacent")
        p.add_argument("--trials", type=int, default=50)
        args = p.parse_args(sys.argv[2:])
        from research.ml_pipeline_v3_adjacent import run as r
        r(n_trials=args.trials)
    elif cmd == "persistence2":
        from research.persistence_v2 import run as r
        r()
    elif cmd == "cat_persistence":
        from research.specialization_by_category import run as r
        r()
    elif cmd == "insider_market":
        from research.market_insider_flow import run as r
        r()    
    elif cmd == "insider_market2":
        from research.market_insider_flow_v2 import run as r
        r()
    elif cmd == "insider_robust":
        from research.insider_robustness import run as r
        r()
    elif cmd == "descriptives":
        from research.descriptives import run as r
        r()
    elif cmd == "true-oos-final":
        from research.true_oos_final import run as r
        r()
    elif cmd == "true-oos-second":
        from research.true_oos_second_window import run as r
        r()
    else:
        print(HELP)


if __name__ == "__main__":
    main()