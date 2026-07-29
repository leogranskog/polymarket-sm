import psycopg2
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DB_URL

SCHEMA = """
CREATE TABLE IF NOT EXISTS wallets (
    address              TEXT PRIMARY KEY,
    pnl_total            FLOAT,
    pnl_resolved         FLOAT,
    pnl_resolved_no_fee  FLOAT,
    score_profitability  FLOAT,
    score_skill          FLOAT,
    score_reliability    FLOAT,
    sm_score             FLOAT,
    is_smart_money       BOOLEAN DEFAULT FALSE,
    top_category         TEXT,
    category_scores      JSONB,
    trade_count          INT,
    total_volume         FLOAT,
    taker_ratio          FLOAT,
    early_entry_ratio    FLOAT,
    avg_holding_days     FLOAT,
    first_trade_date     DATE,
    last_trade_date      DATE,
    last_updated         TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS markets (
    market_id            TEXT PRIMARY KEY,
    event_id             TEXT,
    question             TEXT,
    category             TEXT,
    resolution_date      DATE,
    resolved             BOOLEAN DEFAULT FALSE,
    outcome              TEXT,
    price_yes            FLOAT,
    price_no             FLOAT,
    volume_total         FLOAT,
    volume_24h           FLOAT,
    last_updated         TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id             TEXT PRIMARY KEY,
    timestamp            TIMESTAMPTZ NOT NULL,
    market_id            TEXT NOT NULL,
    wallet_address       TEXT NOT NULL,
    side                 TEXT NOT NULL,
    price                FLOAT,
    quantity             FLOAT,
    is_maker             BOOLEAN,
    is_smart_money       BOOLEAN DEFAULT FALSE,
    sm_score             FLOAT,
    source               TEXT DEFAULT 'live'
);

CREATE TABLE IF NOT EXISTS market_sm_sentiment (
    market_id              TEXT,
    date                   DATE,
    sentiment_direction    FLOAT,
    sentiment_conviction   FLOAT,
    sentiment_momentum     FLOAT,
    sentiment_timing       FLOAT,
    sentiment_composite    FLOAT,
    sm_wallets_active      INT,
    sm_volume_24h          FLOAT,
    sm_net_flow            FLOAT,
    sm_avg_entry_yes       FLOAT,
    sm_avg_entry_no        FLOAT,
    market_price_yes       FLOAT,
    entry_signal           BOOLEAN DEFAULT FALSE,
    exit_signal            BOOLEAN DEFAULT FALSE,
    signal_strength        TEXT,
    PRIMARY KEY (market_id, date)
);

CREATE TABLE IF NOT EXISTS positions (
    position_id          SERIAL PRIMARY KEY,
    market_id            TEXT NOT NULL,
    side                 TEXT NOT NULL,
    entry_price          FLOAT,
    entry_date           TIMESTAMPTZ,
    entry_signal_score   FLOAT,
    quantity             FLOAT,
    cost_basis           FLOAT,
    current_price        FLOAT,
    unrealized_pnl       FLOAT,
    exit_price           FLOAT,
    exit_date            TIMESTAMPTZ,
    realized_pnl         FLOAT,
    exit_reason          TEXT,
    status               TEXT DEFAULT 'open',
    last_updated         TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS mirror_actions (
    id                   SERIAL PRIMARY KEY,
    timestamp            TIMESTAMPTZ DEFAULT NOW(),
    market_id            TEXT,
    action               TEXT,
    trigger_wallet       TEXT,
    trigger_type         TEXT,
    sentiment_composite  FLOAT,
    side                 TEXT,
    price                FLOAT,
    quantity             FLOAT,
    executed             BOOLEAN DEFAULT FALSE,
    fail_reason          TEXT,
    notes                TEXT
);
"""

def setup_db():
    print(f"Connecting to database...")
    try:
        conn = psycopg2.connect(DB_URL)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(SCHEMA)
        print("✅ All tables created successfully.")
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = [r[0] for r in cur.fetchall()]
        print(f"   Tables: {', '.join(tables)}")
        conn.close()
    except Exception as e:
        print(f"❌ DB setup failed: {e}")
        print("   Make sure PostgreSQL is running and DB_URL in config.py is correct.")

if __name__ == "__main__":
    setup_db()