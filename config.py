"""
Central config for the Polymarket Smart Money system.
Edit values here — everything imports from this file.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"

# ── Database ───────────────────────────────────────────────────────────────
DB_URL = "postgresql://postgres:postgres@localhost:5432/polymarket"

# ── HuggingFace dataset ────────────────────────────────────────────────────
HF_DATASET = "vgregoire/polymarket-users"

HF_SUBSETS = [
    "user_pnl_summary",
    "user_features",
    "markets",
    "events",
]

# ── Smart Money Scoring ────────────────────────────────────────────────────
SM_SCORE_WEIGHTS = {
    "profitability": 0.40,
    "skill":         0.35,
    "reliability":   0.25,
}

SM_SCORE_THRESHOLD  = 0.60
SM_MIN_TRADE_COUNT  = 50
SM_MIN_PNL          = 0.0

# ── SM Sentiment ───────────────────────────────────────────────────────────
SENTIMENT_WEIGHTS = {
    "direction":   0.40,
    "conviction":  0.25,
    "momentum":    0.20,
    "timing":      0.15,
}

ENTRY_COMPOSITE_MIN   = 0.65
ENTRY_CONVICTION_MIN  = 0.75
ENTRY_MIN_WALLETS     = 5
EXIT_COMPOSITE_MAX    = 0.20
EXIT_MOMENTUM_DROP    = -0.25
EXIT_CONVICTION_MIN   = 0.35

MOMENTUM_WINDOW  = 3
SENTIMENT_WINDOW = 7

# ── Position Sizing ────────────────────────────────────────────────────────
KELLY_FRACTION      = 0.25
MAX_POSITION_PCT    = 0.05
MAX_CATEGORY_PCT    = 0.20
MAX_DEPLOYED_PCT    = 0.40

# ── Live API ───────────────────────────────────────────────────────────────
POLYMARKET_API_BASE  = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_BASE = "https://clob.polymarket.com"
POLYMARKET_WS_URL    = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
API_POLL_INTERVAL    = 10