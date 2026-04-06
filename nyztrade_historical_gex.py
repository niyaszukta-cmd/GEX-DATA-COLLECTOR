"""
NYZTrade Historical GEX Collector — Dhan API
============================================
Fetches NIFTY options data (ATM ± 15 strikes) from Dhan API
and computes GEX, VANNA, DEX for each day.

WHAT IT DOES:
  1. Get NIFTY closing price from Dhan for each trading day
  2. Find ATM strike (round to nearest 50)
  3. Fetch daily OHLCV+OI for strikes ATM-15 to ATM+15 (CE + PE)
  4. Compute Black-Scholes IV, Gamma, Vanna, Delta
  5. Compute net GEX, VANNA, DEX per strike and daily aggregate
  6. Save to SQLite + export to CSV

USAGE:
  Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in Streamlit sidebar
  then click Run.

DHAN API ENDPOINTS USED:
  POST /v2/charts/historical  → OHLCV + OI per option contract
  POST /v2/charts/historical  → NIFTY index closing price
"""

import os, sys, io, time, sqlite3, json, logging, argparse
import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List, Optional, Tuple

# ── Safe logging (works on Streamlit Cloud) ────────────────────────────────────
_handlers = [logging.StreamHandler(sys.stdout)]
try:
    _handlers.insert(0, logging.FileHandler('gex_collector.log'))
except Exception:
    pass
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=_handlers)
log = logging.getLogger('NYZTrade')

# ── Paths (safe on Streamlit Cloud read-only filesystem) ──────────────────────
def _workdir():
    for base in [Path('.'), Path('/tmp/nyztrade_gex')]:
        try:
            base.mkdir(parents=True, exist_ok=True)
            t = base / '.test'; t.write_text('ok'); t.unlink()
            return base
        except Exception:
            continue
    return Path('/tmp')

WORK_DIR        = _workdir()
DB_PATH         = WORK_DIR / 'gex_dhan.db'
EXPORT_DIR      = WORK_DIR / 'export'
CHECKPOINT_FILE = WORK_DIR / 'checkpoint.json'
try:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
DHAN_BASE       = 'https://api.dhan.co'
NIFTY_SEC_ID    = '13'          # Dhan security ID for NIFTY 50 index
NIFTY_SEGMENT   = 'IDX_I'
NIFTY_FO_SEG    = 'NSE_FO'
STRIKE_INTERVAL = 50            # NIFTY strikes in multiples of 50
ATM_RANGE       = 15            # ATM ± 15 strikes = 31 strikes total
RISK_FREE       = 0.065         # 6.5% Indian T-bill
LOT_SIZE        = 75            # NIFTY lot size (post-Nov 2024)
UNIT_DIV        = 1e9           # Display in Billions

# ── Checkpoint ────────────────────────────────────────────────────────────────
def save_cp(key, val):
    cp = {}
    try:
        if CHECKPOINT_FILE.exists():
            cp = json.loads(CHECKPOINT_FILE.read_text())
    except Exception:
        pass
    cp[key] = val
    try:
        CHECKPOINT_FILE.write_text(json.dumps(cp, indent=2))
    except Exception:
        pass

def load_cp(key, default=None):
    try:
        if CHECKPOINT_FILE.exists():
            return json.loads(CHECKPOINT_FILE.read_text()).get(key, default)
    except Exception:
        pass
    return default


# ═════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═════════════════════════════════════════════════════════════════════════════
def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.executescript("""
    -- Daily NIFTY closing price (spot)
    CREATE TABLE IF NOT EXISTS nifty_ohlcv (
        trade_date  TEXT PRIMARY KEY,
        open        REAL, high REAL, low REAL, close REAL, volume REAL
    );

    -- Per-strike options data fetched from Dhan
    CREATE TABLE IF NOT EXISTS options_raw (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_date  TEXT NOT NULL,
        strike      REAL NOT NULL,
        option_type TEXT NOT NULL,   -- CE or PE
        open        REAL DEFAULT 0,
        high        REAL DEFAULT 0,
        low         REAL DEFAULT 0,
        close       REAL DEFAULT 0,  -- LTP / settlement
        volume      REAL DEFAULT 0,
        open_interest REAL DEFAULT 0,
        spot_price  REAL DEFAULT 0,
        UNIQUE(trade_date, strike, option_type)
    );

    -- Computed GEX per strike per day
    CREATE TABLE IF NOT EXISTS gex_per_strike (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_date  TEXT NOT NULL,
        strike      REAL NOT NULL,
        spot_price  REAL,
        tte_days    REAL,
        call_ltp    REAL, put_ltp  REAL,
        call_oi     REAL, put_oi   REAL,
        call_iv     REAL, put_iv   REAL,
        call_gamma  REAL, put_gamma REAL,
        call_vanna  REAL, put_vanna REAL,
        call_delta  REAL, put_delta REAL,
        net_gex     REAL, net_vanna REAL, net_dex REAL,
        UNIQUE(trade_date, strike)
    );

    -- Daily GEX aggregate (one row per day — the research table)
    CREATE TABLE IF NOT EXISTS gex_daily (
        trade_date          TEXT PRIMARY KEY,
        spot_price          REAL,
        total_net_gex       REAL,
        total_pos_gex       REAL,
        total_neg_gex       REAL,
        total_net_vanna     REAL,
        total_net_dex       REAL,
        gex_regime          TEXT,   -- POSITIVE / NEGATIVE / NEUTRAL
        dominant_call_wall  REAL,   -- strike with highest +GEX
        dominant_put_wall   REAL,   -- strike with highest -GEX
        gex_flip_zone       REAL,   -- nearest zero-crossing strike
        atm_call_iv         REAL,
        atm_put_iv          REAL,
        iv_skew             REAL,   -- put_iv - call_iv at ATM
        pcr                 REAL,   -- put/call OI ratio
        total_call_oi       REAL,
        total_put_oi        REAL,
        n_strikes           INTEGER
    );

    -- Download progress tracker
    CREATE TABLE IF NOT EXISTS fetch_log (
        trade_date  TEXT NOT NULL,
        strike      REAL NOT NULL,
        option_type TEXT NOT NULL,
        status      TEXT,           -- ok / error / no_data
        error_msg   TEXT,
        PRIMARY KEY(trade_date, strike, option_type)
    );
    """)
    conn.commit()
    return conn


# ═════════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES
# ═════════════════════════════════════════════════════════════════════════════
def _d1(S, K, T, r, sig):
    T = max(T, 1/365)
    return (np.log(S / max(K, 1e-6)) + (r + 0.5*sig**2)*T) / max(sig*np.sqrt(T), 1e-10)

def bs_gamma(S, K, T, r, sig):
    return norm.pdf(_d1(S, K, T, r, sig)) / max(S * sig * np.sqrt(max(T,1/365)), 1e-10)

def bs_vanna(S, K, T, r, sig):
    d1 = _d1(S, K, T, r, sig)
    d2 = d1 - sig * np.sqrt(max(T, 1/365))
    return -norm.pdf(d1) * d2 / max(sig, 1e-10)

def bs_delta(S, K, T, r, sig, otype='CE'):
    d1 = _d1(S, K, T, r, sig)
    return norm.cdf(d1) if otype == 'CE' else norm.cdf(d1) - 1.0

def bs_price(S, K, T, r, sig, otype='CE'):
    if T <= 0 or sig <= 0:
        return max(S-K, 0) if otype == 'CE' else max(K-S, 0)
    d1 = _d1(S, K, T, r, sig)
    d2 = d1 - sig * np.sqrt(T)
    if otype == 'CE':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def solve_iv(price, S, K, T, r, otype='CE') -> float:
    """Bisection IV solver — always converges."""
    if price <= 0 or T <= 0:
        return 0.0
    intrinsic = max(S-K, 0) if otype == 'CE' else max(K-S, 0)
    if price < intrinsic:
        return 0.0
    lo, hi = 0.001, 5.0
    for _ in range(100):
        mid = (lo + hi) / 2
        p   = bs_price(S, K, T, r, mid, otype)
        if abs(p - price) < 1e-5:
            return mid * 100
        if p < price:
            lo = mid
        else:
            hi = mid
    return mid * 100


# ═════════════════════════════════════════════════════════════════════════════
# DHAN API CLIENT
# ═════════════════════════════════════════════════════════════════════════════
class DhanClient:
    def __init__(self, client_id: str, access_token: str):
        self.headers = {
            'access-token': access_token,
            'client-id':    client_id,
            'Content-Type': 'application/json',
        }

    def _post(self, endpoint: str, payload: dict) -> Optional[dict]:
        try:
            r = requests.post(
                f'{DHAN_BASE}{endpoint}',
                headers=self.headers,
                json=payload,
                timeout=20)
            if r.status_code == 200:
                return r.json()
            log.debug(f"Dhan {endpoint}: HTTP {r.status_code} — {r.text[:200]}")
        except Exception as e:
            log.debug(f"Dhan {endpoint} error: {e}")
        return None

    def get_nifty_ohlcv(self, from_date: str, to_date: str) -> List[dict]:
        """
        Fetch NIFTY 50 daily OHLCV.
        Returns list of {date, open, high, low, close, volume}
        """
        data = self._post('/v2/charts/historical', {
            'securityId':      NIFTY_SEC_ID,
            'exchangeSegment': NIFTY_SEGMENT,
            'instrument':      'INDEX',
            'expiryCode':      0,
            'oi_flag':         '0',
            'fromDate':        from_date,
            'toDate':          to_date,
        })
        if not data:
            return []

        ts   = data.get('timestamp', [])
        opens  = data.get('open',   [])
        highs  = data.get('high',   [])
        lows   = data.get('low',    [])
        closes = data.get('close',  [])
        vols   = data.get('volume', [])

        rows = []
        for i, t in enumerate(ts):
            try:
                d = datetime.fromtimestamp(int(t)).strftime('%Y-%m-%d')
                rows.append({
                    'date':   d,
                    'open':   float(opens[i])  if i < len(opens)  else 0,
                    'high':   float(highs[i])  if i < len(highs)  else 0,
                    'low':    float(lows[i])   if i < len(lows)   else 0,
                    'close':  float(closes[i]) if i < len(closes) else 0,
                    'volume': float(vols[i])   if i < len(vols)   else 0,
                })
            except Exception:
                continue
        return rows

    def get_option_ohlcv(self, security_id: str,
                          from_date: str, to_date: str) -> List[dict]:
        """
        Fetch daily OHLCV + OI for a specific option contract.
        Dhan returns OI in the 'oi' field when oi_flag='1'.
        """
        data = self._post('/v2/charts/historical', {
            'securityId':      security_id,
            'exchangeSegment': NIFTY_FO_SEG,
            'instrument':      'OPTIDX',
            'expiryCode':      0,
            'oi_flag':         '1',     # request OI data
            'fromDate':        from_date,
            'toDate':          to_date,
        })
        if not data:
            return []

        ts      = data.get('timestamp', [])
        closes  = data.get('close',  [])
        volumes = data.get('volume', [])
        ois     = data.get('oi',     [])

        rows = []
        for i, t in enumerate(ts):
            try:
                d = datetime.fromtimestamp(int(t)).strftime('%Y-%m-%d')
                rows.append({
                    'date':  d,
                    'close': float(closes[i])  if i < len(closes)  else 0,
                    'vol':   float(volumes[i]) if i < len(volumes) else 0,
                    'oi':    float(ois[i])     if i < len(ois)     else 0,
                })
            except Exception:
                continue
        return rows

    def search_option_security_id(self, strike: float,
                                   expiry_date: str,
                                   option_type: str) -> Optional[str]:
        """
        Find the Dhan security_id for a specific NIFTY option contract.
        Uses Dhan's option chain search endpoint.
        expiry_date: 'YYYY-MM-DD'
        option_type: 'CE' or 'PE'
        """
        try:
            r = requests.get(
                f'{DHAN_BASE}/v2/optionchain',
                headers=self.headers,
                params={
                    'UnderlyingScrip': 'NIFTY',
                    'UnderlyingSeg':   'IDX_I',
                    'Expiry':          expiry_date,
                },
                timeout=15)
            if r.status_code != 200:
                return None
            chain = r.json().get('data', [])
            for item in chain:
                if abs(float(item.get('strikePrice', 0)) - strike) < 1:
                    side = item.get('callOption' if option_type == 'CE' else 'putOption', {})
                    if side:
                        return str(side.get('securityId', ''))
        except Exception as e:
            log.debug(f"Security ID search error: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# NIFTY EXPIRY HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def get_next_thursday(from_date: date) -> date:
    """Get the next/current Thursday (NSE weekly expiry day)."""
    d = from_date
    while d.weekday() != 3:  # 3 = Thursday
        d += timedelta(days=1)
    return d

def get_monthly_expiry(from_date: date) -> date:
    """Last Thursday of the month."""
    # Go to last day of month, then back to Thursday
    if from_date.month == 12:
        next_month = date(from_date.year+1, 1, 1)
    else:
        next_month = date(from_date.year, from_date.month+1, 1)
    last_day = next_month - timedelta(days=1)
    while last_day.weekday() != 3:
        last_day -= timedelta(days=1)
    return last_day

def atm_strike(spot: float) -> float:
    """Round spot to nearest NIFTY strike (multiple of 50)."""
    return round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL

def strike_range(atm: float) -> List[float]:
    """ATM-15 to ATM+15, step 50."""
    return [atm + i * STRIKE_INTERVAL for i in range(-ATM_RANGE, ATM_RANGE+1)]


# ═════════════════════════════════════════════════════════════════════════════
# MAIN COLLECTOR
# ═════════════════════════════════════════════════════════════════════════════
class GEXCollector:
    def __init__(self, conn: sqlite3.Connection, client: DhanClient):
        self.conn   = conn
        self.client = client

    # ── Step 1: Download NIFTY OHLCV ─────────────────────────────────────────
    def fetch_nifty_prices(self, start: date, end: date,
                            progress_cb=None) -> int:
        """Fetch NIFTY closing prices — needed for ATM calculation."""
        log.info(f"Fetching NIFTY OHLCV {start} → {end}...")

        # Chunk into 365-day windows (Dhan limit)
        total_rows = 0
        chunk_start = start
        chunk_num   = 0
        total_chunks = max(1, (end - start).days // 364 + 1)

        while chunk_start <= end:
            chunk_end = min(chunk_start + timedelta(days=364), end)
            chunk_num += 1
            if progress_cb:
                progress_cb(chunk_num / total_chunks,
                            f"NIFTY prices {chunk_start} → {chunk_end}")

            rows = self.client.get_nifty_ohlcv(
                chunk_start.strftime('%Y-%m-%d'),
                chunk_end.strftime('%Y-%m-%d'))

            if rows:
                self.conn.executemany("""
                    INSERT OR REPLACE INTO nifty_ohlcv
                    (trade_date, open, high, low, close, volume)
                    VALUES (:date, :open, :high, :low, :close, :volume)
                """, rows)
                self.conn.commit()
                total_rows += len(rows)
                log.info(f"  Chunk {chunk_num}: {len(rows)} days")

            chunk_start = chunk_end + timedelta(days=1)
            time.sleep(0.5)

        log.info(f"NIFTY OHLCV complete: {total_rows} trading days")
        return total_rows

    # ── Step 2: Fetch option data for all strikes ─────────────────────────────
    def fetch_options_data(self, start: date, end: date,
                            progress_cb=None):
        """
        For each trading day:
          1. Get NIFTY close → compute ATM
          2. Build strike list (ATM ± 15)
          3. Find nearest weekly expiry
          4. Fetch Dhan historical OHLCV+OI for each strike CE+PE
        """
        # Get all trading days where we have NIFTY prices
        trading_days = self.conn.execute("""
            SELECT trade_date, close FROM nifty_ohlcv
            WHERE trade_date >= ? AND trade_date <= ?
            ORDER BY trade_date
        """, (start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))).fetchall()

        total = len(trading_days)
        log.info(f"Fetching options for {total} trading days...")

        # Pre-build expiry→security_id cache to reduce API calls
        sec_id_cache = {}  # (strike, expiry, otype) → security_id

        for day_idx, (date_str, spot) in enumerate(trading_days):
            if progress_cb:
                progress_cb(day_idx / total,
                            f"Options {date_str} (ATM={atm_strike(spot):.0f})")

            atm   = atm_strike(spot)
            strikes = strike_range(atm)
            trade_date = date.fromisoformat(date_str)
            expiry = get_next_thursday(trade_date)

            # If expiry is today, use next week's expiry
            if expiry == trade_date:
                expiry = get_next_thursday(trade_date + timedelta(days=1))

            expiry_str = expiry.strftime('%Y-%m-%d')

            for strike in strikes:
                for otype in ['CE', 'PE']:
                    # Check if already fetched
                    existing = self.conn.execute("""
                        SELECT status FROM fetch_log
                        WHERE trade_date=? AND strike=? AND option_type=?
                    """, (date_str, strike, otype)).fetchone()
                    if existing and existing[0] == 'ok':
                        continue

                    # Get security ID
                    cache_key = (strike, expiry_str, otype)
                    sec_id = sec_id_cache.get(cache_key)
                    if not sec_id:
                        sec_id = self.client.search_option_security_id(
                            strike, expiry_str, otype)
                        if sec_id:
                            sec_id_cache[cache_key] = sec_id

                    if not sec_id:
                        self.conn.execute("""
                            INSERT OR REPLACE INTO fetch_log
                            (trade_date, strike, option_type, status, error_msg)
                            VALUES (?,?,?,'error','no_security_id')
                        """, (date_str, strike, otype))
                        self.conn.commit()
                        continue

                    # Fetch OHLCV+OI for this specific contract
                    # We only need the single day, but Dhan needs a range
                    rows = self.client.get_option_ohlcv(
                        sec_id,
                        date_str,
                        date_str)

                    # Find the row matching our trade date
                    day_row = next((r for r in rows if r['date'] == date_str), None)

                    if day_row:
                        self.conn.execute("""
                            INSERT OR REPLACE INTO options_raw
                            (trade_date, strike, option_type,
                             close, volume, open_interest, spot_price)
                            VALUES (?,?,?,?,?,?,?)
                        """, (date_str, strike, otype,
                              day_row['close'], day_row['vol'],
                              day_row['oi'], spot))
                        self.conn.execute("""
                            INSERT OR REPLACE INTO fetch_log
                            (trade_date, strike, option_type, status)
                            VALUES (?,?,?,'ok')
                        """, (date_str, strike, otype))
                    else:
                        self.conn.execute("""
                            INSERT OR REPLACE INTO fetch_log
                            (trade_date, strike, option_type, status)
                            VALUES (?,?,?,'no_data')
                        """, (date_str, strike, otype))

                    self.conn.commit()
                    time.sleep(0.1)  # gentle rate limit

            save_cp('last_options_date', date_str)
            log.info(f"  [{day_idx+1}/{total}] {date_str} ATM={atm:.0f}")

    # ── Step 3: Compute GEX ───────────────────────────────────────────────────
    def compute_gex(self, progress_cb=None):
        """
        For each day with options data, compute:
        IV (BS bisection) → Gamma, Vanna, Delta → GEX, VANNA, DEX
        Then aggregate to daily summary.
        """
        # Get all dates with options data not yet in gex_per_strike
        pending = self.conn.execute("""
            SELECT DISTINCT o.trade_date
            FROM options_raw o
            LEFT JOIN gex_daily g ON o.trade_date = g.trade_date
            WHERE g.trade_date IS NULL
            ORDER BY o.trade_date
        """).fetchall()

        total = len(pending)
        log.info(f"Computing GEX for {total} days...")

        for i, (date_str,) in enumerate(pending):
            if progress_cb:
                progress_cb(i / total, f"GEX compute: {date_str}")
            try:
                self._compute_day(date_str)
            except Exception as e:
                log.error(f"  GEX error {date_str}: {e}")
            if i % 10 == 0:
                save_cp('last_gex_date', date_str)

        log.info("GEX computation complete.")

    def _compute_day(self, date_str: str):
        """Compute GEX for one trading day."""
        # Get options data
        rows = self.conn.execute("""
            SELECT strike, option_type, close, open_interest, spot_price
            FROM options_raw
            WHERE trade_date = ? AND open_interest > 0
            ORDER BY strike
        """, (date_str,)).fetchall()

        if not rows:
            return

        df = pd.DataFrame(rows, columns=['strike','otype','ltp','oi','spot'])
        spot = float(df['spot'].iloc[0])
        if spot <= 0:
            # Try NIFTY OHLCV
            row = self.conn.execute(
                "SELECT close FROM nifty_ohlcv WHERE trade_date=?",
                (date_str,)).fetchone()
            if row:
                spot = float(row[0])
        if spot <= 0:
            return

        # Time to expiry — use nearest Thursday
        trade_date = date.fromisoformat(date_str)
        expiry     = get_next_thursday(trade_date)
        if expiry == trade_date:
            expiry = get_next_thursday(trade_date + timedelta(days=1))
        tte = max((expiry - trade_date).days / 365.0, 1/365)

        # Pivot to call/put per strike
        calls = df[df['otype']=='CE'].set_index('strike')
        puts  = df[df['otype']=='PE'].set_index('strike')
        strikes = sorted(set(calls.index) | set(puts.index))

        if not strikes:
            return

        strike_rows = []
        for K in strikes:
            c_ltp = float(calls.loc[K,'ltp']) if K in calls.index else 0
            p_ltp = float(puts.loc[K,'ltp'])  if K in puts.index  else 0
            c_oi  = float(calls.loc[K,'oi'])  * LOT_SIZE if K in calls.index else 0
            p_oi  = float(puts.loc[K,'oi'])   * LOT_SIZE if K in puts.index  else 0

            # Solve IV
            c_iv = solve_iv(c_ltp, spot, K, tte, RISK_FREE, 'CE') if c_ltp > 0 else 0
            p_iv = solve_iv(p_ltp, spot, K, tte, RISK_FREE, 'PE') if p_ltp > 0 else 0

            # Fill zero IVs with 25% default
            c_iv = c_iv or 25.0
            p_iv = p_iv or 25.0

            c_sig = np.clip(c_iv, 1, 500) / 100
            p_sig = np.clip(p_iv, 1, 500) / 100

            # BS Greeks
            c_g = bs_gamma(spot, K, tte, RISK_FREE, c_sig)
            p_g = bs_gamma(spot, K, tte, RISK_FREE, p_sig)
            c_v = bs_vanna(spot, K, tte, RISK_FREE, c_sig)
            p_v = bs_vanna(spot, K, tte, RISK_FREE, p_sig)
            c_d = bs_delta(spot, K, tte, RISK_FREE, c_sig, 'CE')
            p_d = bs_delta(spot, K, tte, RISK_FREE, p_sig, 'PE')

            # GEX, VANNA, DEX (same formula as live dashboard)
            net_gex   = (c_oi*c_g - p_oi*p_g) * spot**2 / UNIT_DIV
            net_vanna = (c_oi*c_v - p_oi*p_v) / UNIT_DIV
            net_dex   = (c_oi*c_d + p_oi*p_d) / UNIT_DIV

            strike_rows.append({
                'strike':      K,
                'spot':        spot,
                'tte_days':    round(tte * 365, 1),
                'call_ltp':    c_ltp,   'put_ltp':    p_ltp,
                'call_oi':     c_oi,    'put_oi':     p_oi,
                'call_iv':     round(c_iv,2), 'put_iv': round(p_iv,2),
                'call_gamma':  c_g,     'put_gamma':  p_g,
                'call_vanna':  c_v,     'put_vanna':  p_v,
                'call_delta':  c_d,     'put_delta':  p_d,
                'net_gex':     net_gex,
                'net_vanna':   net_vanna,
                'net_dex':     net_dex,
            })

        if not strike_rows:
            return

        # Store per-strike
        for r in strike_rows:
            self.conn.execute("""
                INSERT OR REPLACE INTO gex_per_strike
                (trade_date, strike, spot_price, tte_days,
                 call_ltp, put_ltp, call_oi, put_oi,
                 call_iv, put_iv, call_gamma, put_gamma,
                 call_vanna, put_vanna, call_delta, put_delta,
                 net_gex, net_vanna, net_dex)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (date_str, r['strike'], r['spot'], r['tte_days'],
                  r['call_ltp'], r['put_ltp'], r['call_oi'], r['put_oi'],
                  r['call_iv'], r['put_iv'], r['call_gamma'], r['put_gamma'],
                  r['call_vanna'], r['put_vanna'], r['call_delta'], r['put_delta'],
                  r['net_gex'], r['net_vanna'], r['net_dex']))

        df2 = pd.DataFrame(strike_rows)

        # Daily aggregate
        tot_gex   = df2['net_gex'].sum()
        tot_vanna = df2['net_vanna'].sum()
        tot_dex   = df2['net_dex'].sum()
        tot_coi   = df2['call_oi'].sum()
        tot_poi   = df2['put_oi'].sum()
        pcr       = tot_poi / max(tot_coi, 1)
        pos_gex   = df2[df2['net_gex']>0]['net_gex'].sum()
        neg_gex   = df2[df2['net_gex']<0]['net_gex'].sum()
        regime    = 'POSITIVE' if tot_gex>0 else ('NEGATIVE' if tot_gex<0 else 'NEUTRAL')

        # Dominant walls
        pm = df2['net_gex'] > 0
        nm = df2['net_gex'] < 0
        dcwall = float(df2[pm].loc[df2[pm]['net_gex'].idxmax(),'strike']) if pm.any() else 0
        dpwall = float(df2[nm].loc[df2[nm]['net_gex'].idxmin(),'strike']) if nm.any() else 0

        # Flip zone (nearest zero crossing)
        df_s = df2.sort_values('strike')
        flip = 0.0
        for j in range(len(df_s)-1):
            g1 = float(df_s['net_gex'].iloc[j])
            g2 = float(df_s['net_gex'].iloc[j+1])
            if g1 * g2 < 0:
                k1 = float(df_s['strike'].iloc[j])
                k2 = float(df_s['strike'].iloc[j+1])
                f  = k1 + (k2-k1)*abs(g1)/(abs(g1)+abs(g2))
                if abs(f-spot) < abs(flip-spot) or flip == 0:
                    flip = f

        # ATM IV
        atm = atm_strike(spot)
        atm_row = df2.iloc[(df2['strike']-spot).abs().argsort().iloc[0]]
        atm_civ = float(atm_row['call_iv'])
        atm_piv = float(atm_row['put_iv'])
        iv_skew = atm_piv - atm_civ

        self.conn.execute("""
            INSERT OR REPLACE INTO gex_daily
            (trade_date, spot_price, total_net_gex, total_pos_gex, total_neg_gex,
             total_net_vanna, total_net_dex, gex_regime,
             dominant_call_wall, dominant_put_wall, gex_flip_zone,
             atm_call_iv, atm_put_iv, iv_skew, pcr,
             total_call_oi, total_put_oi, n_strikes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (date_str, spot, tot_gex, pos_gex, neg_gex,
              tot_vanna, tot_dex, regime, dcwall, dpwall, flip,
              atm_civ, atm_piv, iv_skew, pcr,
              tot_coi, tot_poi, len(df2)))
        self.conn.commit()

    # ── Export ────────────────────────────────────────────────────────────────
    def export(self, progress_cb=None) -> dict:
        """Export all tables to CSV."""
        paths = {}
        tables = {
            'GEX_DAILY.csv':      'SELECT * FROM gex_daily ORDER BY trade_date',
            'GEX_PER_STRIKE.csv': 'SELECT * FROM gex_per_strike ORDER BY trade_date, strike',
            'NIFTY_OHLCV.csv':    'SELECT * FROM nifty_ohlcv ORDER BY trade_date',
            'OPTIONS_RAW.csv':    'SELECT * FROM options_raw ORDER BY trade_date, strike, option_type',
        }
        total = len(tables)
        for i, (fname, sql) in enumerate(tables.items()):
            if progress_cb:
                progress_cb(i/total, f"Exporting {fname}...")
            try:
                df = pd.read_sql(sql, self.conn)
                path = EXPORT_DIR / fname
                df.to_csv(path, index=False)
                paths[fname] = path
                log.info(f"  {fname}: {len(df):,} rows")
            except Exception as e:
                log.error(f"  Export {fname}: {e}")
        if progress_cb:
            progress_cb(1.0, "Export complete")
        return paths

    def summary(self) -> dict:
        """Return DB row counts for display."""
        s = {}
        for t in ['nifty_ohlcv','options_raw','gex_per_strike','gex_daily','fetch_log']:
            try:
                s[t] = self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except Exception:
                s[t] = 0
        try:
            r = self.conn.execute(
                "SELECT MIN(trade_date),MAX(trade_date) FROM gex_daily").fetchone()
            s['gex_from'] = r[0] or '—'
            s['gex_to']   = r[1] or '—'
        except Exception:
            s['gex_from'] = s['gex_to'] = '—'
        try:
            fl = self.conn.execute(
                "SELECT status,COUNT(*) FROM fetch_log GROUP BY status").fetchall()
            s['fetch_status'] = dict(fl)
        except Exception:
            s['fetch_status'] = {}
        return s


# ── CLI entry point ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--client-id',  required=True)
    ap.add_argument('--token',      required=True)
    ap.add_argument('--start',      default='2024-01-01')
    ap.add_argument('--end',        default=str(date.today()))
    ap.add_argument('--prices',     action='store_true')
    ap.add_argument('--options',    action='store_true')
    ap.add_argument('--gex',        action='store_true')
    ap.add_argument('--export',     action='store_true')
    ap.add_argument('--all',        action='store_true')
    args = ap.parse_args()

    conn   = init_db()
    client = DhanClient(args.client_id, args.token)
    gc     = GEXCollector(conn, client)
    start  = date.fromisoformat(args.start)
    end    = date.fromisoformat(args.end)

    if args.all or args.prices:
        gc.fetch_nifty_prices(start, end)
    if args.all or args.options:
        gc.fetch_options_data(start, end)
    if args.all or args.gex:
        gc.compute_gex()
    if args.all or args.export:
        gc.export()

    s = gc.summary()
    print("\n=== DB Summary ===")
    for k, v in s.items():
        print(f"  {k}: {v}")
    conn.close()


if __name__ == '__main__':
    main()
