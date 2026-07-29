"""
Polymarket Fee Model — exact formula from docs.polymarket.us/fees
Effective exchange-wide from April 3, 2026.

Formula:  Fee = Θ × C × p × (1 - p)
  Θ (taker)        = 0.05
  Θ (maker rebate) = -0.0125   (maker RECEIVES this)
  C = number of contracts
  p = trade price (0.01 – 0.99)

Historical periods (the dataset spans 2022–2026):
  - Pre Q4 2024:   zero fees on all markets
  - Q4 2024:       fees introduced on crypto 15-min markets only
  - Mar 30 2026:   category-based fees rolled out exchange-wide
  - Apr 3  2026:   unified Θ=0.05 taker / Θ=-0.0125 maker rebate

The dataset already provides pnl_daily_no_fee and pnl_daily variants
so we replicate both worlds in the backtest.
"""

from dataclasses import dataclass
from enum import Enum


class OrderType(Enum):
    TAKER = "taker"
    MAKER = "maker"


# Fee periods — used to apply correct fee based on trade date
FEE_PERIODS = [
    # (start_date_str, taker_theta, maker_theta, description)
    ("2022-11-11", 0.0,    0.0,     "Zero-fee era"),
    ("2024-10-01", 0.0,    0.0,     "Fees only on 15-min crypto (we treat as ~0 for standard markets)"),
    ("2026-03-30", 0.05,  -0.0125,  "Category-based fees (unified April 3 formula)"),
]

# Current (live) thetas
TAKER_THETA   =  0.05
MAKER_THETA   = -0.0125   # negative = rebate paid TO maker


@dataclass
class FeeResult:
    gross_cost:    float   # USDC spent buying contracts
    fee:           float   # USDC paid in fees (taker) or received (maker, negative)
    net_cost:      float   # gross_cost + fee
    fee_pct:       float   # fee as % of gross_cost
    order_type:    str


def taker_fee(contracts: float, price: float, date_str: str = "2026-04-03") -> FeeResult:
    """
    Compute taker fee for a buy order.

    Args:
        contracts:  number of shares bought
        price:      price per share (0.01 – 0.99)
        date_str:   trade date string YYYY-MM-DD (selects correct fee period)
    """
    theta = _get_theta(date_str, OrderType.TAKER)
    price = max(0.01, min(0.99, price))   # clamp to valid range

    gross   = contracts * price
    fee     = theta * contracts * price * (1 - price)
    net     = gross + fee

    return FeeResult(
        gross_cost  = round(gross, 4),
        fee         = round(fee,   4),
        net_cost    = round(net,   4),
        fee_pct     = round(fee / gross * 100, 4) if gross > 0 else 0.0,
        order_type  = "taker",
    )


def maker_rebate(contracts: float, price: float, date_str: str = "2026-04-03") -> FeeResult:
    """
    Compute maker rebate for a limit order fill.
    Returns negative fee (= money received by maker).
    """
    theta = _get_theta(date_str, OrderType.MAKER)
    price = max(0.01, min(0.99, price))

    gross   = contracts * price
    rebate  = theta * contracts * price * (1 - price)   # negative value
    net     = gross + rebate                             # net < gross (cheaper)

    return FeeResult(
        gross_cost  = round(gross,  4),
        fee         = round(rebate, 4),
        net_cost    = round(net,    4),
        fee_pct     = round(rebate / gross * 100, 4) if gross > 0 else 0.0,
        order_type  = "maker",
    )


def round_trip_cost(
    contracts:   float,
    entry_price: float,
    exit_price:  float,
    entry_date:  str = "2026-04-03",
    exit_date:   str = "2026-04-03",
    entry_as_maker: bool = False,
    exit_as_maker:  bool = False,
) -> dict:
    """
    Full round-trip cost: entry + exit fees.
    Sell orders are NOT subject to taker fees (only buys are).
    Makers receive rebates on both sides.

    Returns dict with full breakdown.
    """
    # Entry
    if entry_as_maker:
        entry_fee_result = maker_rebate(contracts, entry_price, entry_date)
    else:
        entry_fee_result = taker_fee(contracts, entry_price, entry_date)

    # Exit (selling) — taker fee does NOT apply to sells;
    # but maker rebate DOES apply if selling as maker
    if exit_as_maker:
        exit_fee_result = maker_rebate(contracts, exit_price, exit_date)
    else:
        # Taker sell: no fee
        gross = contracts * exit_price
        exit_fee_result = FeeResult(
            gross_cost = round(gross, 4),
            fee        = 0.0,
            net_cost   = round(gross, 4),
            fee_pct    = 0.0,
            order_type = "taker_sell",
        )

    total_fees = entry_fee_result.fee + exit_fee_result.fee

    gross_pnl  = contracts * (exit_price - entry_price)
    net_pnl    = gross_pnl - total_fees

    return {
        "contracts":         contracts,
        "entry_price":       entry_price,
        "exit_price":        exit_price,
        "gross_pnl":         round(gross_pnl,   4),
        "entry_fee":         entry_fee_result.fee,
        "exit_fee":          exit_fee_result.fee,
        "total_fees":        round(total_fees,  4),
        "net_pnl":           round(net_pnl,     4),
        "entry_order_type":  entry_fee_result.order_type,
        "exit_order_type":   exit_fee_result.order_type,
        "fee_drag_pct":      round(total_fees / (contracts * entry_price) * 100, 3)
                             if entry_price > 0 else 0.0,
    }


def breakeven_move(entry_price: float, date_str: str = "2026-04-03") -> float:
    """
    Minimum price move (entry→exit) needed to break even after fees.
    Assumes taker entry, taker sell exit (worst case).
    """
    theta = _get_theta(date_str, OrderType.TAKER)
    # entry fee per contract = theta * entry_price * (1 - entry_price)
    # exit fee = 0 (taker sell is free)
    # need: (exit_price - entry_price) >= entry_fee_per_contract
    fee_per_contract = theta * entry_price * (1 - entry_price)
    return round(fee_per_contract, 5)


def _get_theta(date_str: str, order_type: OrderType) -> float:
    """Return correct theta for a given date and order type."""
    from datetime import date
    try:
        d = date.fromisoformat(date_str[:10])
    except Exception:
        d = date(2026, 4, 3)

    # Walk periods in reverse, return first match
    for start_str, t_taker, t_maker, _ in reversed(FEE_PERIODS):
        start = date.fromisoformat(start_str)
        if d >= start:
            return t_taker if order_type == OrderType.TAKER else t_maker

    return 0.0


# ── Quick sanity check ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Fee model sanity check")
    print("="*50)

    # Should match docs: 100 contracts at $0.50 → taker pays $1.25
    f = taker_fee(100, 0.50, "2026-04-03")
    print(f"\n100 contracts @ $0.50 (taker):")
    print(f"  Gross cost: ${f.gross_cost:.2f}")
    print(f"  Fee:        ${f.fee:.2f}  (docs say $1.25 ✓)" )
    print(f"  Net cost:   ${f.net_cost:.2f}")

    # Maker rebate
    m = maker_rebate(100, 0.50, "2026-04-03")
    print(f"\n100 contracts @ $0.50 (maker):")
    print(f"  Rebate:     ${m.fee:.2f}  (docs say -$0.31 ✓)")

    # Round trip example
    print(f"\nRound trip: buy 1000 @ 0.35, sell @ 0.65 (taker entry, free sell)")
    rt = round_trip_cost(1000, 0.35, 0.65, entry_as_maker=False, exit_as_maker=False)
    for k, v in rt.items():
        print(f"  {k}: {v}")

    # Break-even
    print(f"\nBreak-even move needed at various entry prices (taker):")
    for p in [0.20, 0.35, 0.50, 0.65, 0.80]:
        be = breakeven_move(p)
        print(f"  Entry {p:.2f}: need +{be:.4f} move ({be/p*100:.2f}% of price)")