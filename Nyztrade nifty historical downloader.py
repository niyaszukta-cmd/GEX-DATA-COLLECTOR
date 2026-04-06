# ============================================================================
# NYZTrade — NIFTY 5-Year Historical GEX Data Downloader v2
# Dhan API | Daily OHLCV + ALL Greeks + Cascade Math | Robust Checkpointing
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import sqlite3
import os
import pytz
import json
import warnings
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from scipy.stats import norm
from pathlib import Path
from typing import List, Dict, Optional
import io

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="NYZTrade NIFTY Downloader v2",
    page_icon="📥",
    layout="wide",
    initial_sidebar_state="expanded"
)

IST = pytz.timezone("Asia/Kolkata")

# ============================================================================
# CONSTANTS
# ============================================================================

DHAN_CLIENT_ID    = "1100480354"
NIFTY_SEC_ID      = 13
CONTRACT_SIZE     = 25
STRIKE_INTERVAL   = 50
SCALING_FACTOR    = 1e9
RISK_FREE_RATE    = 0.07
DB_PATH           = "nyztrade_nifty_historical_v2.db"
CKPT_PATH         = "nyztrade_download_checkpoint.json"

STRIKE_RANGE = (["ATM"] +
    [f"ATM+{i}" for i in range(1, 11)] +
    [f"ATM-{i}" for i in range(1, 11)])

NIFTY_CASCADE_PARAMS = {"pts_per_unit": 0.010, "strike_cap": 150}

# ============================================================================
# STYLES
# ============================================================================

st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@600;700&display=swap");
.main-header {
    background: linear-gradient(135deg,#1a0533 0%,#2d0a5e 60%,#3b0764 100%);
    border:1px solid rgba(168,85,247,0.4); border-radius:16px;
    padding:24px 32px; margin-bottom:20px;
}
.main-title {
    font-family:"Space Grotesk",sans-serif; font-size:1.8rem; font-weight:700;
    background:linear-gradient(135deg,#fff 0%,#c084fc 70%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0 0 6px 0;
}
.sub-title { font-family:"JetBrains Mono",monospace; font-size:0.8rem; color:rgba(255,255,255,0.5); margin:0; }
.stat-card { background:rgba(124,58,237,0.12); border:1px solid rgba(168,85,247,0.3); border-radius:10px; padding:14px 18px; text-align:center; }
.stat-val  { font-family:"Space Grotesk",sans-serif; font-size:1.4rem; font-weight:700; color:#c084fc; }
.stat-lbl  { font-family:"JetBrains Mono",monospace; font-size:0.65rem; color:rgba(255,255,255,0.4); margin-top:3px; }
.ckpt-box  { background:rgba(6,182,212,0.1); border:1px solid rgba(6,182,212,0.4); border-radius:8px; padding:10px 14px; font-family:"JetBrains Mono",monospace; font-size:0.78rem; color:#67e8f9; }
.log-box   { background:#0d0d1a; border:1px solid rgba(168,85,247,0.2); border-radius:8px; padding:12px 16px; font-family:"JetBrains Mono",monospace; font-size:0.73rem; color:#94a3b8; max-height:300px; overflow-y:auto; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# BLACK-SCHOLES CALCULATOR
# ============================================================================

class BS:
    @staticmethod
    def _d1(S,K,T,r,sig):
        if T<=0 or sig<=0: return 0.0
        return (np.log(S/K)+(r+0.5*sig**2)*T)/(sig*np.sqrt(T))
    @staticmethod
    def _d2(S,K,T,r,sig):
        return BS._d1(S,K,T,r,sig)-sig*np.sqrt(max(T,1e-8))
    @staticmethod
    def gamma(S,K,T,r,sig):
        if T<=0 or sig<=0 or S<=0 or K<=0: return 0.0
        try: return float(norm.pdf(BS._d1(S,K,T,r,sig))/(S*sig*np.sqrt(T)))
        except: return 0.0
    @staticmethod
    def call_delta(S,K,T,r,sig):
        if T<=0 or sig<=0 or S<=0 or K<=0: return 0.0
        try: return float(norm.cdf(BS._d1(S,K,T,r,sig)))
        except: return 0.0
    @staticmethod
    def put_delta(S,K,T,r,sig):
        if T<=0 or sig<=0 or S<=0 or K<=0: return 0.0
        try: return float(norm.cdf(BS._d1(S,K,T,r,sig))-1)
        except: return 0.0
    @staticmethod
    def vanna(S,K,T,r,sig):
        if T<=0 or sig<=0 or S<=0 or K<=0: return 0.0
        try:
            d1=BS._d1(S,K,T,r,sig); d2=BS._d2(S,K,T,r,sig)
            return float(-norm.pdf(d1)*d2/sig)
        except: return 0.0
    @staticmethod
    def charm(S,K,T,r,sig):
        if T<=0 or sig<=0 or S<=0 or K<=0: return 0.0
        try:
            d1=BS._d1(S,K,T,r,sig); d2=BS._d2(S,K,T,r,sig)
            return float(-norm.pdf(d1)*(2*r*T-d2*sig*np.sqrt(T))/(2*T*sig*np.sqrt(T)))
        except: return 0.0
    @staticmethod
    def theta(S,K,T,r,sig,opt="call"):
        if T<=0 or sig<=0 or S<=0 or K<=0: return 0.0
        try:
            d1=BS._d1(S,K,T,r,sig); d2=BS._d2(S,K,T,r,sig)
            t1=-(S*norm.pdf(d1)*sig)/(2*np.sqrt(T))
            if opt=="call": return float((t1-r*K*np.exp(-r*T)*norm.cdf(d2))/365)
            return float((t1+r*K*np.exp(-r*T)*norm.cdf(-d2))/365)
        except: return 0.0
    @staticmethod
    def vega(S,K,T,r,sig):
        if T<=0 or sig<=0 or S<=0 or K<=0: return 0.0
        try: return float(S*norm.pdf(BS._d1(S,K,T,r,sig))*np.sqrt(T)/100)
        except: return 0.0
    @staticmethod
    def speed(S,K,T,r,sig):
        if T<=0 or sig<=0 or S<=0 or K<=0: return 0.0
        try:
            d1=BS._d1(S,K,T,r,sig); g=BS.gamma(S,K,T,r,sig)
            return float(-g/S*(d1/(sig*np.sqrt(T))+1))
        except: return 0.0
    @staticmethod
    def zomma(S,K,T,r,sig):
        if T<=0 or sig<=0 or S<=0 or K<=0: return 0.0
        try:
            d1=BS._d1(S,K,T,r,sig); d2=BS._d2(S,K,T,r,sig)
            return float(BS.gamma(S,K,T,r,sig)*(d1*d2-1)/sig)
        except: return 0.0

bs = BS()

# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """
    Saves and restores download progress so interrupted downloads
    can resume from the exact date+expiry+strike that failed.
    """
    def __init__(self, path: str = CKPT_PATH):
        self.path = path
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    return json.load(f)
            except: pass
        return {
            "started_at":   None,
            "last_saved":   None,
            "total_dates":  0,
            "completed":    [],   # list of "date|expiry_flag|expiry_code" strings
            "failed":       [],   # same format
            "current_date": None,
            "current_exp":  None,
            "stats": {"ok":0,"err":0,"skipped":0,"rows":0}
        }

    def save(self):
        self._data["last_saved"] = datetime.now(IST).isoformat()
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def mark_started(self, total_dates: int):
        if not self._data["started_at"]:
            self._data["started_at"] = datetime.now(IST).isoformat()
        self._data["total_dates"] = total_dates
        self.save()

    def is_done(self, date_str: str, flag: str, code: int) -> bool:
        key = f"{date_str}|{flag}|{code}"
        return key in self._data["completed"]

    def mark_ok(self, date_str: str, flag: str, code: int, rows: int):
        key = f"{date_str}|{flag}|{code}"
        if key not in self._data["completed"]:
            self._data["completed"].append(key)
        self._data["current_date"] = date_str
        self._data["current_exp"]  = f"{flag}-{code}"
        self._data["stats"]["ok"]  += 1
        self._data["stats"]["rows"] += rows
        self.save()

    def mark_fail(self, date_str: str, flag: str, code: int):
        key = f"{date_str}|{flag}|{code}"
        if key not in self._data["failed"]:
            self._data["failed"].append(key)
        self._data["stats"]["err"] += 1
        self.save()

    def mark_skip(self):
        self._data["stats"]["skipped"] += 1

    def reset(self):
        self._data = {
            "started_at": None, "last_saved": None, "total_dates": 0,
            "completed": [], "failed": [],
            "current_date": None, "current_exp": None,
            "stats": {"ok":0,"err":0,"skipped":0,"rows":0}
        }
        self.save()

    @property
    def stats(self): return self._data["stats"]
    @property
    def completed_count(self): return len(self._data["completed"])
    @property
    def failed_dates(self): return self._data["failed"]
    @property
    def last_saved(self): return self._data.get("last_saved","—")
    @property
    def current_date(self): return self._data.get("current_date","—")
    @property
    def started_at(self): return self._data.get("started_at","—")
    @property
    def total_dates(self): return self._data.get("total_dates", 0)

ckpt = CheckpointManager()

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class HistDB:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        self._init()

    def _init(self):
        conn = sqlite3.connect(self.path)
        c = conn.cursor()

        # ── Main data table ───────────────────────────────────────────────
        c.execute("""
        CREATE TABLE IF NOT EXISTS nifty_daily (
            date TEXT, expiry_flag TEXT, expiry_code INTEGER,
            strike REAL, strike_type TEXT,
            -- spot
            spot_open REAL, spot_high REAL, spot_low REAL, spot_close REAL,
            -- call raw
            call_oi REAL, call_oi_open REAL, call_volume REAL,
            call_iv REAL, call_open REAL, call_high REAL, call_low REAL, call_close REAL,
            -- put raw
            put_oi REAL, put_oi_open REAL, put_volume REAL,
            put_iv REAL, put_open REAL, put_high REAL, put_low REAL, put_close REAL,
            -- oi changes
            call_oi_chg REAL, put_oi_chg REAL,
            -- tte
            tte REAL,
            -- greeks per contract
            call_gamma REAL, put_gamma REAL,
            call_delta REAL, put_delta REAL,
            call_vanna REAL, put_vanna REAL,
            call_charm REAL, put_charm REAL,
            call_theta REAL, put_theta REAL,
            call_vega  REAL, put_vega  REAL,
            call_speed REAL, put_speed REAL,
            call_zomma REAL, put_zomma REAL,
            -- exposures (Billions)
            call_gex REAL, put_gex REAL, net_gex REAL,
            call_dex REAL, put_dex REAL, net_dex REAL,
            call_vanna_exp REAL, put_vanna_exp REAL, net_vanna REAL,
            call_charm_exp REAL, put_charm_exp REAL, net_charm REAL,
            call_theta_exp REAL, put_theta_exp REAL, net_theta REAL,
            call_vega_exp  REAL, put_vega_exp  REAL, net_vega  REAL,
            call_speed_exp REAL, put_speed_exp REAL, net_speed REAL,
            call_zomma_exp REAL, put_zomma_exp REAL, net_zomma REAL,
            -- enhanced oi gex
            call_oi_gex REAL, put_oi_gex REAL, net_oi_gex REAL,
            -- cross-strike aggregates
            total_call_oi REAL, total_put_oi REAL,
            total_volume REAL, pcr_oi REAL,
            net_gex_all REAL, net_dex_all REAL, net_vanna_all REAL, net_charm_all REAL,
            fetched_at TEXT,
            PRIMARY KEY (date, expiry_flag, expiry_code, strike_type)
        )""")

        # ── Cascade data table ────────────────────────────────────────────
        c.execute("""
        CREATE TABLE IF NOT EXISTS nifty_cascade (
            date TEXT, expiry_flag TEXT, expiry_code INTEGER,
            gex_type TEXT,         -- "standard" or "enhanced_oi"
            cascade_direction TEXT,-- "BEAR" or "BULL"
            strike REAL,
            strike_type TEXT,
            gex_raw REAL,
            pts_raw REAL,
            vanna_adj REAL,
            vanna_adj_pct TEXT,
            vanna_note TEXT,
            pts_impact REAL,
            cumulative_pts REAL,
            role TEXT,
            -- cascade summary (same value for all rows of same date/exp/type/direction)
            bear_fuel_pts REAL,
            bear_brake_pts REAL,
            bear_net_pts REAL,
            bull_fuel_pts REAL,
            bull_brake_pts REAL,
            bull_net_pts REAL,
            -- iv context
            iv_regime TEXT,
            iv_skew REAL,
            spot_close REAL,
            -- vanna zones
            vanna_zones_json TEXT,
            fetched_at TEXT,
            PRIMARY KEY (date, expiry_flag, expiry_code, gex_type, cascade_direction, strike_type)
        )""")

        # ── Checkpoint backup table ───────────────────────────────────────
        c.execute("""
        CREATE TABLE IF NOT EXISTS checkpoint_log (
            saved_at TEXT,
            completed_count INTEGER,
            failed_count INTEGER,
            total_rows INTEGER,
            last_date TEXT
        )""")

        conn.commit()
        conn.close()

    def date_exists(self, date_str, flag, code) -> bool:
        conn = sqlite3.connect(self.path)
        c = conn.cursor()
        c.execute("SELECT 1 FROM nifty_daily WHERE date=? AND expiry_flag=? AND expiry_code=? LIMIT 1",
                  (date_str, flag, code))
        exists = c.fetchone() is not None
        conn.close()
        return exists

    def insert_rows(self, rows: list, date_str, flag, code) -> int:
        if not rows: return 0
        conn = sqlite3.connect(self.path)
        pd.DataFrame(rows).to_sql("nifty_daily", conn,
            if_exists="append", index=False, method="multi")
        conn.commit(); conn.close()
        return len(rows)

    def insert_cascade(self, cascade_rows: list) -> int:
        if not cascade_rows: return 0
        conn = sqlite3.connect(self.path)
        pd.DataFrame(cascade_rows).to_sql("nifty_cascade", conn,
            if_exists="append", index=False, method="multi")
        conn.commit(); conn.close()
        return len(cascade_rows)

    def save_checkpoint_log(self, ck: CheckpointManager):
        conn = sqlite3.connect(self.path)
        conn.execute("""INSERT INTO checkpoint_log
            (saved_at, completed_count, failed_count, total_rows, last_date)
            VALUES (?,?,?,?,?)""",
            (datetime.now(IST).isoformat(),
             ck.completed_count,
             len(ck.failed_dates),
             ck.stats["rows"],
             ck.current_date))
        conn.commit(); conn.close()

    def get_stats(self):
        conn = sqlite3.connect(self.path)
        tot  = pd.read_sql("SELECT COUNT(*) n FROM nifty_daily",   conn).iloc[0]["n"]
        days = pd.read_sql("SELECT COUNT(DISTINCT date) n FROM nifty_daily", conn).iloc[0]["n"]
        casc = pd.read_sql("SELECT COUNT(*) n FROM nifty_cascade",  conn).iloc[0]["n"]
        mb   = os.path.getsize(self.path)/1e6 if os.path.exists(self.path) else 0
        conn.close()
        return {"rows": int(tot), "days": int(days), "cascade_rows": int(casc), "mb": mb}

    def export_csv(self, table="nifty_daily", start=None, end=None) -> bytes:
        conn = sqlite3.connect(self.path)
        where = f"WHERE date >= '{start}' AND date <= '{end}'" if start and end else ""
        df = pd.read_sql(f"SELECT * FROM {table} {where} ORDER BY date, strike", conn)
        conn.close()
        buf = io.BytesIO(); df.to_csv(buf, index=False); return buf.getvalue()

db = HistDB()

# ============================================================================
# CASCADE MATHEMATICS ENGINE
# ============================================================================

def compute_cascade_for_day(rows: list, spot_close: float,
                             iv_regime: str = "FLAT", iv_skew: float = 0.0,
                             gex_type: str = "standard",
                             date_str: str = "", expiry_flag: str = "",
                             expiry_code: int = 1) -> list:
    """
    Runs cascade math on the daily strike rows.
    gex_type = "standard"    → uses net_gex
    gex_type = "enhanced_oi" → uses net_oi_gex
    Returns list of cascade rows ready to insert into nifty_cascade.
    """
    if not rows or spot_close == 0:
        return []

    df = pd.DataFrame(rows)
    gex_col = "net_gex" if gex_type == "standard" else "net_oi_gex"
    if gex_col not in df.columns:
        return []

    pts_per_unit = NIFTY_CASCADE_PARAMS["pts_per_unit"]
    strike_cap   = NIFTY_CASCADE_PARAMS["strike_cap"]

    # ── Identify VANNA flip zones ─────────────────────────────────────────
    df_s = df.sort_values("strike").reset_index(drop=True)
    vanna_zones = []
    for i in range(len(df_s) - 1):
        cv = df_s.iloc[i]["net_vanna"]
        nv = df_s.iloc[i+1]["net_vanna"]
        ck = df_s.iloc[i]["strike"]
        nk = df_s.iloc[i+1]["strike"]
        if (cv > 0 and nv < 0) or (cv < 0 and nv > 0):
            w     = abs(cv) / (abs(cv) + abs(nv) + 1e-12)
            flip  = ck + (nk - ck) * w
            mag   = (abs(cv) + abs(nv)) / 2
            ft    = "POS_TO_NEG" if cv > 0 else "NEG_TO_POS"
            above = flip > spot_close
            if above and ft == "POS_TO_NEG":   role = "RESISTANCE_CEILING"
            elif above and ft == "NEG_TO_POS": role = "VACUUM_ZONE"
            elif not above and ft == "POS_TO_NEG": role = "TRAP_DOOR"
            else:                              role = "SUPPORT_FLOOR"
            vanna_zones.append({"strike": flip, "role": role, "magnitude": mag})

    vanna_zones_json = json.dumps(vanna_zones)

    # ── Apply VANNA adjustments ────────────────────────────────────────────
    df_c = df_s.copy()
    df_c["gex_raw"]    = df_c[gex_col].fillna(0)
    df_c["vanna_adj"]  = 1.0
    df_c["vanna_note"] = ""

    actual_interval = float(df_c["strike"].diff().abs().dropna().median()) if len(df_c) > 1 else 50.0
    vanna_prox = actual_interval * 1.0

    for z in vanna_zones:
        zk   = z["strike"]
        role = z["role"]
        mag  = z["magnitude"]
        stre = min(1.5, 1.0 + mag / (mag + 10))
        mask = (df_c["strike"] - zk).abs() <= vanna_prox
        if not mask.any(): continue

        if role == "SUPPORT_FLOOR":
            if iv_regime == "COMPRESSING":   adj, note = 1-0.60*stre, f"Support@{zk:.0f}[COMPRESS-{60*stre:.0f}%]"
            elif iv_regime == "FLAT":        adj, note = 1-0.35*stre, f"Support@{zk:.0f}[FLAT-{35*stre:.0f}%]"
            else:                            adj, note = 1+0.20*stre, f"Support@{zk:.0f}[EXPAND+{20*stre:.0f}%]"
        elif role == "TRAP_DOOR":
            if iv_regime == "EXPANDING":     adj, note = 1+0.30*stre, f"Trap@{zk:.0f}[EXPAND+{30*stre:.0f}%]"
            elif iv_regime == "FLAT":        adj, note = 1+0.15*stre, f"Trap@{zk:.0f}[FLAT+{15*stre:.0f}%]"
            else:                            adj, note = 1-0.10*stre, f"Trap@{zk:.0f}[COMPRESS-{10*stre:.0f}%]"
        elif role == "VACUUM_ZONE":
            if iv_regime == "EXPANDING":     adj, note = 1-0.50*stre, f"Vacuum@{zk:.0f}[EXPAND-{50*stre:.0f}%]"
            else:                            adj, note = 1.0, ""
        elif role == "RESISTANCE_CEILING":
            if iv_regime == "EXPANDING":     adj, note = 1+0.20*stre, f"Resist@{zk:.0f}[EXPAND+{20*stre:.0f}%]"
            else:                            adj, note = 1-0.15*stre, f"Resist@{zk:.0f}[COMPRESS-{15*stre:.0f}%]"
        else:
            adj, note = 1.0, ""

        if note:
            existing = df_c.loc[mask, "vanna_adj"]
            stronger = (adj - 1.0).__abs__() > (existing - 1.0).abs()
            df_c.loc[mask & stronger, "vanna_adj"]  = adj
            df_c.loc[mask & stronger, "vanna_note"] = note

    df_c["pts_raw"]    = (df_c["gex_raw"].abs() * pts_per_unit).clip(upper=strike_cap)
    df_c["pts_impact"] = (df_c["pts_raw"] * df_c["vanna_adj"]).round(2).clip(lower=0, upper=strike_cap*1.5)

    # ── Bear cascade ───────────────────────────────────────────────────────
    bear = df_c[df_c["strike"] <= spot_close].sort_values("strike", ascending=False).copy()
    bear["cascade_direction"] = "BEAR"
    bear["cumulative_pts"]    = bear["pts_impact"].cumsum().round(2)
    bear["role_label"] = bear.apply(lambda r: r["vanna_note"] if r["vanna_note"] else
        ("Accelerates fall" if r["gex_raw"] < 0 else "Brakes fall"), axis=1)

    b_fuel  = bear[bear["gex_raw"] < 0]["pts_impact"].sum()
    b_brake = bear[bear["gex_raw"] >= 0]["pts_impact"].sum()
    b_net   = max(0, b_fuel - b_brake * 0.5)

    # ── Bull cascade ───────────────────────────────────────────────────────
    bull = df_c[df_c["strike"] > spot_close].sort_values("strike", ascending=True).copy()
    bull["cascade_direction"] = "BULL"
    bull["cumulative_pts"]    = bull["pts_impact"].cumsum().round(2)
    bull["role_label"] = bull.apply(lambda r: r["vanna_note"] if r["vanna_note"] else
        ("Accelerates rise" if r["gex_raw"] < 0 else "Brakes rise"), axis=1)

    u_fuel  = bull[bull["gex_raw"] < 0]["pts_impact"].sum()
    u_brake = bull[bull["gex_raw"] >= 0]["pts_impact"].sum()
    u_net   = max(0, u_fuel - u_brake * 0.5)

    # ── Build cascade rows ─────────────────────────────────────────────────
    now_str = datetime.now(IST).isoformat()
    cascade_rows = []

    for part_df in [bear, bull]:
        if part_df.empty: continue
        direction = part_df.iloc[0]["cascade_direction"]
        for _, row in part_df.iterrows():
            adj_val = row["vanna_adj"]
            if adj_val > 1.05:   adj_pct = f"+{(adj_val-1)*100:.0f}%"
            elif adj_val < 0.95: adj_pct = f"-{(1-adj_val)*100:.0f}%"
            else:                adj_pct = "--"

            cascade_rows.append({
                "date":             date_str,
                "expiry_flag":      expiry_flag,
                "expiry_code":      expiry_code,
                "gex_type":         gex_type,
                "cascade_direction":direction,
                "strike":           row["strike"],
                "strike_type":      row.get("strike_type",""),
                "gex_raw":          round(row["gex_raw"], 6),
                "pts_raw":          round(row["pts_raw"], 2),
                "vanna_adj":        round(row["vanna_adj"], 4),
                "vanna_adj_pct":    adj_pct,
                "vanna_note":       row["vanna_note"],
                "pts_impact":       round(row["pts_impact"], 2),
                "cumulative_pts":   round(row["cumulative_pts"], 2),
                "role":             row["role_label"],
                "bear_fuel_pts":    round(b_fuel, 2),
                "bear_brake_pts":   round(b_brake, 2),
                "bear_net_pts":     round(b_net, 2),
                "bull_fuel_pts":    round(u_fuel, 2),
                "bull_brake_pts":   round(u_brake, 2),
                "bull_net_pts":     round(u_net, 2),
                "iv_regime":        iv_regime,
                "iv_skew":          round(iv_skew, 4),
                "spot_close":       spot_close,
                "vanna_zones_json": vanna_zones_json,
                "fetched_at":       now_str,
            })

    return cascade_rows

# ============================================================================
# DHAN API FETCHER
# ============================================================================

class DhanFetcher:
    def __init__(self, token: str):
        self.headers = {
            "access-token": token,
            "client-id":    DHAN_CLIENT_ID,
            "Content-Type": "application/json",
        }
        self.base = "https://api.dhan.co/v2"

    def fetch(self, strike_type: str, opt_type: str,
              from_date: str, to_date: str,
              expiry_code: int, expiry_flag: str) -> dict:
        payload = {
            "exchangeSegment": "NSE_FNO",
            "interval":        "1D",
            "securityId":      NIFTY_SEC_ID,
            "instrument":      "OPTIDX",
            "expiryFlag":      expiry_flag,
            "expiryCode":      expiry_code,
            "strike":          strike_type,
            "drvOptionType":   opt_type,
            "requiredData":    ["open","high","low","close","volume","oi","iv","strike","spot"],
            "fromDate":        from_date,
            "toDate":          to_date,
        }
        try:
            r = requests.post(f"{self.base}/charts/rollingoption",
                              headers=self.headers, json=payload, timeout=30)
            if r.status_code == 200:
                return r.json().get("data", {})
            return {"error": f"HTTP {r.status_code}: {r.text[:150]}"}
        except Exception as e:
            return {"error": str(e)}

    def build_row(self, date_str, strike_type, ce_day, pe_day, flag, code, tte) -> dict:
        spot   = ce_day.get("spot",  0) or pe_day.get("spot",  0) or 0
        strike = ce_day.get("strike",0) or pe_day.get("strike",0) or 0
        if spot == 0 or strike == 0: return {}

        c_oi   = ce_day.get("oi",     0) or 0
        p_oi   = pe_day.get("oi",     0) or 0
        c_vol  = ce_day.get("volume", 0) or 0
        p_vol  = pe_day.get("volume", 0) or 0
        c_iv   = ce_day.get("iv",    15) or 15
        p_iv   = pe_day.get("iv",    15) or 15
        c_oi0  = ce_day.get("oi_open", c_oi)
        p_oi0  = pe_day.get("oi_open", p_oi)

        civ = c_iv/100 if c_iv > 1 else c_iv
        piv = p_iv/100 if p_iv > 1 else p_iv

        cg=bs.gamma(spot,strike,tte,RISK_FREE_RATE,civ)
        pg=bs.gamma(spot,strike,tte,RISK_FREE_RATE,piv)
        cd=bs.call_delta(spot,strike,tte,RISK_FREE_RATE,civ)
        pd_=bs.put_delta(spot,strike,tte,RISK_FREE_RATE,piv)
        cv=bs.vanna(spot,strike,tte,RISK_FREE_RATE,civ)
        pv=bs.vanna(spot,strike,tte,RISK_FREE_RATE,piv)
        cc=bs.charm(spot,strike,tte,RISK_FREE_RATE,civ)
        pc=bs.charm(spot,strike,tte,RISK_FREE_RATE,piv)
        ct=bs.theta(spot,strike,tte,RISK_FREE_RATE,civ,"call")
        pt=bs.theta(spot,strike,tte,RISK_FREE_RATE,piv,"put")
        cve=bs.vega(spot,strike,tte,RISK_FREE_RATE,civ)
        pve=bs.vega(spot,strike,tte,RISK_FREE_RATE,piv)
        csp=bs.speed(spot,strike,tte,RISK_FREE_RATE,civ)
        psp=bs.speed(spot,strike,tte,RISK_FREE_RATE,piv)
        czm=bs.zomma(spot,strike,tte,RISK_FREE_RATE,civ)
        pzm=bs.zomma(spot,strike,tte,RISK_FREE_RATE,piv)

        S2L=spot**2*CONTRACT_SIZE/SCALING_FACTOR
        SL =spot   *CONTRACT_SIZE/SCALING_FACTOR

        c_oi_chg = c_oi - c_oi0
        p_oi_chg = p_oi - p_oi0
        c_oigex  =  c_oi_chg*cg*S2L
        p_oigex  = -p_oi_chg*pg*S2L

        return {
            "date":date_str,"expiry_flag":flag,"expiry_code":code,
            "strike":strike,"strike_type":strike_type,
            "spot_open":ce_day.get("open",spot),"spot_high":spot,"spot_low":spot,"spot_close":spot,
            "call_oi":c_oi,"call_oi_open":c_oi0,"call_volume":c_vol,
            "call_iv":c_iv,"call_open":ce_day.get("open",0),"call_high":ce_day.get("high",0),
            "call_low":ce_day.get("low",0),"call_close":ce_day.get("close",0),
            "put_oi":p_oi,"put_oi_open":p_oi0,"put_volume":p_vol,
            "put_iv":p_iv,"put_open":pe_day.get("open",0),"put_high":pe_day.get("high",0),
            "put_low":pe_day.get("low",0),"put_close":pe_day.get("close",0),
            "call_oi_chg":c_oi_chg,"put_oi_chg":p_oi_chg,
            "tte":tte,
            "call_gamma":cg,"put_gamma":pg,"call_delta":cd,"put_delta":pd_,
            "call_vanna":cv,"put_vanna":pv,"call_charm":cc,"put_charm":pc,
            "call_theta":ct,"put_theta":pt,"call_vega":cve,"put_vega":pve,
            "call_speed":csp,"put_speed":psp,"call_zomma":czm,"put_zomma":pzm,
            "call_gex":c_oi*cg*S2L,"put_gex":-(p_oi*pg*S2L),"net_gex":(c_oi*cg-p_oi*pg)*S2L,
            "call_dex":c_oi*cd*SL,"put_dex":p_oi*pd_*SL,"net_dex":(c_oi*cd+p_oi*pd_)*SL,
            "call_vanna_exp":c_oi*cv*SL,"put_vanna_exp":p_oi*pv*SL,"net_vanna":(c_oi*cv+p_oi*pv)*SL,
            "call_charm_exp":c_oi*cc*SL,"put_charm_exp":p_oi*pc*SL,"net_charm":(c_oi*cc+p_oi*pc)*SL,
            "call_theta_exp":c_oi*ct*SL,"put_theta_exp":p_oi*pt*SL,"net_theta":(c_oi*ct+p_oi*pt)*SL,
            "call_vega_exp":c_oi*cve*SL,"put_vega_exp":p_oi*pve*SL,"net_vega":(c_oi*cve+p_oi*pve)*SL,
            "call_speed_exp":c_oi*csp*S2L,"put_speed_exp":p_oi*psp*S2L,"net_speed":(c_oi*csp+p_oi*psp)*S2L,
            "call_zomma_exp":c_oi*czm*S2L,"put_zomma_exp":p_oi*pzm*S2L,"net_zomma":(c_oi*czm+p_oi*pzm)*S2L,
            "call_oi_gex":c_oigex,"put_oi_gex":p_oigex,"net_oi_gex":c_oigex+p_oigex,
            "total_call_oi":0,"total_put_oi":0,"total_volume":0,"pcr_oi":0,
            "net_gex_all":0,"net_dex_all":0,"net_vanna_all":0,"net_charm_all":0,
            "fetched_at":datetime.now(IST).isoformat(),
        }

# ============================================================================
# DATE HELPERS
# ============================================================================

def trading_dates(start: date, end: date) -> list:
    out, cur = [], start
    while cur <= end:
        if cur.weekday() < 5: out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out

def extract_day(data: dict, target: str) -> dict:
    for i, ts in enumerate(data.get("timestamp",[])):
        try:
            dt = datetime.fromtimestamp(ts, tz=pytz.UTC).astimezone(IST)
            if dt.strftime("%Y-%m-%d") == target:
                return {k: data.get(k,[None]*100)[i]
                        for k in ["open","high","low","close","volume","oi","iv","strike","spot"]}
        except: continue
    return {}

def tte_from_flag(flag: str) -> float:
    return 7/365 if flag == "WEEK" else 30/365

def estimate_iv_regime(rows: list) -> tuple:
    """Estimate IV regime for the day from ATM call/put IV."""
    if not rows: return "FLAT", 0.0
    df = pd.DataFrame(rows)
    # ATM = strike closest to spot_close
    spot = df["spot_close"].iloc[0] if "spot_close" in df.columns else df["spot_open"].iloc[0]
    atm  = df.iloc[(df["strike"]-spot).abs().argsort().iloc[:1]]
    c_iv = float(atm["call_iv"].values[0]) if not atm.empty else 15.0
    p_iv = float(atm["put_iv"].values[0])  if not atm.empty else 15.0
    skew = c_iv - p_iv
    # Use absolute IV level as proxy for regime
    # (without intraday bars we approximate: IV>20 = expanding, <13 = compressing)
    avg  = (c_iv + p_iv) / 2
    if avg > 18: regime = "EXPANDING"
    elif avg < 13: regime = "COMPRESSING"
    else: regime = "FLAT"
    return regime, round(skew, 2)

# ============================================================================
# UI — HEADER + STATS
# ============================================================================

st.markdown("""
<div class="main-header">
    <div class="main-title">📥 NYZTrade — NIFTY 5-Year Historical Downloader v2</div>
    <div class="sub-title">
        Dhan Rolling Options API · Daily OHLCV + 8 Greeks + GEX/DEX/VANNA/CHARM/THETA/VEGA/SPEED/ZOMMA
        + Cascade Mathematics · Robust Checkpointing · SQLite + CSV Export
    </div>
</div>
""", unsafe_allow_html=True)

stats = db.get_stats()
c1,c2,c3,c4,c5 = st.columns(5)
for col,val,lbl in [
    (c1, f"{stats["days"]:,}",          "Trading Days"),
    (c2, f"{stats["rows"]:,}",          "Strike Rows"),
    (c3, f"{stats["cascade_rows"]:,}",  "Cascade Rows"),
    (c4, f"{stats["mb"]:.1f} MB",       "DB Size"),
    (c5, f"{ckpt.completed_count:,}",     "Checkpointed"),
]:
    col.markdown(f'<div class="stat-card"><div class="stat-val">{val}</div><div class="stat-lbl">{lbl}</div></div>',
                 unsafe_allow_html=True)

# ── Checkpoint status ─────────────────────────────────────────────────────────
st.markdown("")
if ckpt.completed_count > 0:
    pct = int(ckpt.completed_count / max(ckpt.total_dates,1) * 100)
    st.markdown(f"""
    <div class="ckpt-box">
        ⏱️ &nbsp;<b>Checkpoint Active</b> &nbsp;·&nbsp;
        Started: {ckpt.started_at[:19] if ckpt.started_at else "—"} &nbsp;·&nbsp;
        Last saved: {ckpt.last_saved[:19] if ckpt.last_saved else "—"} &nbsp;·&nbsp;
        Last date: <b>{ckpt.current_date}</b> &nbsp;·&nbsp;
        Progress: <b>{ckpt.completed_count}/{ckpt.total_dates} ({pct}%)</b> &nbsp;·&nbsp;
        Rows: {ckpt.stats["rows"]:,} &nbsp;·&nbsp;
        Errors: {len(ckpt.failed_dates)}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Dhan Access Token")
    token = st.text_input("Access Token", type="password",
                          placeholder="Paste Dhan API token here")

    st.markdown("---")
    st.markdown("### 📅 Date Range")
    today     = datetime.now(IST).date()
    five_yrs  = today - timedelta(days=5*365)
    start_d   = st.date_input("Start", value=five_yrs, max_value=today)
    end_d     = st.date_input("End",   value=today,    max_value=today)

    st.markdown("---")
    st.markdown("### 📆 Expiry Configs")
    exp_opts = st.multiselect("Expiry (flag-code)",
        ["WEEK-1","WEEK-2","MONTH-1"], default=["WEEK-1"])

    st.markdown("---")
    st.markdown("### ⚙️ Options")
    skip_done   = st.checkbox("Skip completed dates (resume)", value=True)
    delay_ms    = st.slider("API delay (ms)", 300, 2000, 500, 100)
    ckpt_every  = st.slider("Checkpoint every N dates", 1, 20, 5,
                             help="Save checkpoint after every N completed dates")
    run_cascade = st.checkbox("Compute cascade math", value=True,
                              help="Adds cascade rows for both standard GEX and Enhanced OI GEX")

    st.markdown("---")
    st.markdown(f"**Strikes:** {len(STRIKE_RANGE)} (ATM±10)")
    st.markdown(f"**Cascade scalar:** 0.010 pts/B | cap 150 pts")

    st.markdown("---")
    if st.button("🗑️ Reset Checkpoint", use_container_width=True):
        ckpt.reset()
        st.success("Checkpoint cleared"); st.rerun()

# ── Control buttons ────────────────────────────────────────────────────────────
col_a, col_b, col_c = st.columns([1,1,2])
with col_a: start_btn = st.button("🚀 Start / Resume", type="primary", use_container_width=True)
with col_b: stop_btn  = st.button("⏹ Stop",           use_container_width=True)

# ── Session stop flag ─────────────────────────────────────────────────────────
if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = False
if stop_btn:
    st.session_state.stop_flag = True
    st.warning("⏹ Stop requested — will finish current date and save checkpoint.")

# ── Download loop ──────────────────────────────────────────────────────────────
if start_btn:
    st.session_state.stop_flag = False

    if not token or len(token) < 20:
        st.error("❌ Enter a valid Dhan access token"); st.stop()
    if not exp_opts:
        st.error("❌ Select at least one expiry config"); st.stop()

    fetcher     = DhanFetcher(token)
    dates       = trading_dates(start_d, end_d)
    exp_configs = [{"flag": o.split("-")[0], "code": int(o.split("-")[1])} for o in exp_opts]
    total       = len(dates) * len(exp_configs)

    ckpt.mark_started(total)

    overall_bar  = st.progress(0, text="Starting…")
    current_info = st.empty()
    log_lines    = []
    log_display  = st.empty()
    done_count   = ckpt.completed_count   # start from saved progress

    def log(msg, lvl="info"):
        ts = datetime.now(IST).strftime("%H:%M:%S")
        icon = {"ok":"✅","err":"❌","warn":"⚠️","ckpt":"💾","info":"→"}.get(lvl,"→")
        log_lines.insert(0, f"{ts}  {icon}  {msg}")
        log_display.markdown(
            '<div class="log-box">' + "<br>".join(log_lines[:80]) + "</div>",
            unsafe_allow_html=True)

    log(f"Dates: {len(dates)} | Configs: {len(exp_configs)} | Total: {total}")
    log(f"Resuming from checkpoint — {ckpt.completed_count} already done")

    for date_str in dates:
        if st.session_state.stop_flag:
            log("Stop flag set — saving checkpoint and exiting", "ckpt")
            ckpt.save()
            break

        for ec in exp_configs:
            flag, code = ec["flag"], ec["code"]
            combo_key  = f"{date_str} [{flag}-{code}]"

            # ── CHECKPOINT CHECK ──────────────────────────────────────────
            if skip_done and ckpt.is_done(date_str, flag, code):
                done_count += 1
                ckpt.mark_skip()
                overall_bar.progress(done_count/total, text=f"Skipping {combo_key} (checkpointed)")
                continue
            if skip_done and db.date_exists(date_str, flag, code):
                done_count += 1
                ckpt.mark_ok(date_str, flag, code, 0)
                overall_bar.progress(done_count/total, text=f"Skipping {combo_key} (in DB)")
                continue

            current_info.info(f"**Fetching:** `{combo_key}` — {len(STRIKE_RANGE)} strikes…")
            log(f"Fetching {combo_key}")

            tte    = tte_from_flag(flag)
            t_dt   = datetime.strptime(date_str, "%Y-%m-%d")
            from_d = (t_dt - timedelta(days=1)).strftime("%Y-%m-%d")
            to_d   = (t_dt + timedelta(days=1)).strftime("%Y-%m-%d")

            rows   = []
            errors = 0

            for st_type in STRIKE_RANGE:
                if st.session_state.stop_flag: break

                ce_raw = fetcher.fetch(st_type,"CALL",from_d,to_d,code,flag)
                time.sleep(delay_ms/1000)
                pe_raw = fetcher.fetch(st_type,"PUT", from_d,to_d,code,flag)
                time.sleep(delay_ms/1000)

                if "error" in ce_raw or "error" in pe_raw:
                    errors += 1
                    log(f"  {st_type}: {ce_raw.get('error',pe_raw.get('error',''))[:60]}", "err")
                    continue

                ce_day = extract_day(ce_raw.get("ce",{}), date_str)
                pe_day = extract_day(pe_raw.get("pe",{}), date_str)
                if not ce_day or not pe_day: continue

                row = fetcher.build_row(date_str, st_type, ce_day, pe_day, flag, code, tte)
                if row: rows.append(row)

            if st.session_state.stop_flag:
                log("Stop during strike fetch — saving checkpoint", "ckpt")
                ckpt.save()
                break

            if rows:
                # Cross-strike aggregates
                tc_oi  = sum(r["call_oi"]   for r in rows)
                tp_oi  = sum(r["put_oi"]    for r in rows)
                t_vol  = sum(r["call_volume"]+r["put_volume"] for r in rows)
                pcr    = tp_oi/tc_oi if tc_oi > 0 else 0
                ng_all = sum(r["net_gex"]   for r in rows)
                nd_all = sum(r["net_dex"]   for r in rows)
                nv_all = sum(r["net_vanna"] for r in rows)
                nc_all = sum(r["net_charm"] for r in rows)
                spot_c = rows[0]["spot_close"]

                for r in rows:
                    r["total_call_oi"] = tc_oi; r["total_put_oi"] = tp_oi
                    r["total_volume"]  = t_vol; r["pcr_oi"]        = round(pcr,4)
                    r["net_gex_all"]   = ng_all; r["net_dex_all"]  = nd_all
                    r["net_vanna_all"] = nv_all; r["net_charm_all"]= nc_all

                saved = db.insert_rows(rows, date_str, flag, code)

                # ── CASCADE MATH ──────────────────────────────────────────
                casc_rows_total = 0
                if run_cascade:
                    iv_regime, iv_skew = estimate_iv_regime(rows)

                    for gex_type in ["standard", "enhanced_oi"]:
                        cr = compute_cascade_for_day(
                            rows, spot_c, iv_regime, iv_skew,
                            gex_type, date_str, flag, code)
                        if cr:
                            db.insert_cascade(cr)
                            casc_rows_total += len(cr)

                log(f"  {combo_key}: {saved} rows | cascade {casc_rows_total} rows | "
                    f"net GEX={ng_all:.3f}B | IV={iv_regime if run_cascade else 'skipped'}", "ok")
                ckpt.mark_ok(date_str, flag, code, saved)
            else:
                log(f"  {combo_key}: no data ({errors} errors)", "err")
                ckpt.mark_fail(date_str, flag, code)

            done_count += 1
            overall_bar.progress(done_count/total,
                text=f"Progress: {done_count}/{total} ({done_count/total*100:.1f}%) | "
                     f"Last: {combo_key}")

            # ── CHECKPOINT SAVE ───────────────────────────────────────────
            if ckpt.stats["ok"] % ckpt_every == 0 and ckpt.stats["ok"] > 0:
                ckpt.save()
                db.save_checkpoint_log(ckpt)
                log(f"💾 Checkpoint saved — {ckpt.completed_count} done, "
                    f"{ckpt.stats["rows"]:,} rows", "ckpt")

    # Final checkpoint
    ckpt.save()
    db.save_checkpoint_log(ckpt)
    overall_bar.progress(1.0, text=f"✅ Session complete — {done_count}/{total} processed")
    log(f"Session done. {ckpt.stats["ok"]} ok | {ckpt.stats["err"]} errors | "
        f"{ckpt.stats["rows"]:,} rows", "ok")
    st.rerun()

# ── Retry failed dates ────────────────────────────────────────────────────────
if ckpt.failed_dates:
    st.markdown("---")
    with st.expander(f"⚠️ {len(ckpt.failed_dates)} Failed Dates", expanded=False):
        st.markdown("These dates returned no data. They will be retried next run.")
        for fd in ckpt.failed_dates[-50:]:
            st.code(fd)
        if st.button("🗑️ Clear failed list"):
            ckpt._data["failed"] = []
            ckpt.save()
            st.rerun()

# ── Data preview ──────────────────────────────────────────────────────────────
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Daily Data", "🌊 Cascade Data", "📤 Export"])

with tab1:
    if stats["rows"] > 0:
        conn = sqlite3.connect(DB_PATH)
        prev = pd.read_sql("""SELECT date,expiry_flag,expiry_code,strike_type,strike,
            spot_close,call_oi,put_oi,call_iv,put_iv,
            net_gex,net_dex,net_vanna,net_charm,net_oi_gex,pcr_oi,net_gex_all
            FROM nifty_daily ORDER BY date DESC,strike LIMIT 100""", conn)
        conn.close()
        for col in ["net_gex","net_dex","net_vanna","net_charm","net_oi_gex","net_gex_all"]:
            if col in prev.columns: prev[col] = prev[col].round(4)
        st.dataframe(prev, use_container_width=True, hide_index=True, height=380)
    else:
        st.info("No daily data yet.")

with tab2:
    if stats["cascade_rows"] > 0:
        conn = sqlite3.connect(DB_PATH)
        cprev = pd.read_sql("""SELECT date,expiry_flag,expiry_code,gex_type,
            cascade_direction,strike,strike_type,gex_raw,pts_raw,
            vanna_adj_pct,pts_impact,cumulative_pts,role,
            bear_fuel_pts,bear_net_pts,bull_fuel_pts,bull_net_pts,
            iv_regime,spot_close
            FROM nifty_cascade ORDER BY date DESC,cascade_direction,strike LIMIT 200""", conn)
        conn.close()
        st.dataframe(cprev, use_container_width=True, hide_index=True, height=400)

        # Cascade summary
        conn = sqlite3.connect(DB_PATH)
        csumm = pd.read_sql("""SELECT date,expiry_flag,gex_type,
            MAX(bear_fuel_pts) bear_fuel, MAX(bear_net_pts) bear_net,
            MAX(bull_fuel_pts) bull_fuel, MAX(bull_net_pts) bull_net,
            MAX(iv_regime) iv_regime, MAX(spot_close) spot
            FROM nifty_cascade
            GROUP BY date,expiry_flag,gex_type
            ORDER BY date DESC LIMIT 60""", conn)
        conn.close()
        st.markdown("#### Daily Cascade Summary")
        st.dataframe(csumm, use_container_width=True, hide_index=True, height=300)
    else:
        st.info("No cascade data yet.")

with tab3:
    st.markdown("### 📤 Export")
    ce1,ce2 = st.columns(2)
    with ce1:
        ex_start = st.date_input("From", value=five_yrs, key="ex_s")
        ex_end   = st.date_input("To",   value=today,    key="ex_e")

    with ce2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📥 Export Daily CSV"):
            if stats["rows"] > 0:
                csv = db.export_csv("nifty_daily",
                    ex_start.strftime("%Y-%m-%d"), ex_end.strftime("%Y-%m-%d"))
                st.download_button("💾 Download Daily CSV", csv,
                    f"nifty_daily_{ex_start}_{ex_end}.csv", "text/csv",
                    use_container_width=True)

        if st.button("📥 Export Cascade CSV"):
            if stats["cascade_rows"] > 0:
                csv = db.export_csv("nifty_cascade",
                    ex_start.strftime("%Y-%m-%d"), ex_end.strftime("%Y-%m-%d"))
                st.download_button("💾 Download Cascade CSV", csv,
                    f"nifty_cascade_{ex_start}_{ex_end}.csv", "text/csv",
                    use_container_width=True)

        if st.button("📦 Download SQLite DB"):
            if os.path.exists(DB_PATH):
                with open(DB_PATH,"rb") as f:
                    st.download_button("💾 Download .db", f,
                        "nyztrade_nifty_historical_v2.db",
                        "application/octet-stream", use_container_width=True)

# ── Variable reference ─────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📚 Column Reference — Both Tables"):
    st.markdown("""
    ### nifty_daily — 65+ columns per row
    **Identifiers:** date · expiry_flag · expiry_code · strike · strike_type
    **Spot OHLC:** spot_open/high/low/close
    **Options Raw:** call/put OI · OI_open · OI_chg · volume · IV · OHLC
    **Greeks (per contract):** gamma · delta · vanna · charm · theta · vega · speed · zomma
    **Exposures (Billions = OI × Greek × S^2 or S × LotSize):**
    net_gex · net_dex · net_vanna · net_charm · net_theta · net_vega · net_speed · net_zomma · net_oi_gex
    **Cross-Strike:** total_call_oi · total_put_oi · total_volume · pcr_oi · net_gex_all · net_vanna_all

    ### nifty_cascade — cascade math per row
    **gex_type:** standard (total OI) or enhanced_oi (OI change)
    **cascade_direction:** BEAR (below spot) or BULL (above spot)
    **strike / strike_type / gex_raw / pts_raw** — raw cascade inputs
    **vanna_adj / vanna_adj_pct / vanna_note** — VANNA zone adjustment applied
    **pts_impact** — adjusted cascade points for this strike
    **cumulative_pts** — running total down/up the cascade chain
    **role** — Accelerates fall / Brakes fall / Support Floor / Trap Door / Vacuum / Resistance
    **bear/bull fuel/brake/net pts** — daily cascade summary (same for all rows of that date)
    **iv_regime** — EXPANDING / COMPRESSING / FLAT
    **vanna_zones_json** — all detected VANNA flip zones for the day
    """)

st.markdown("""
<div style="text-align:center;font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:rgba(255,255,255,0.2);margin-top:20px;">
NYZTrade Analytics · NIFTY Historical Downloader v2 · For research only · Not financial advice
</div>
""", unsafe_allow_html=True)
