"""
NYZTrade GEX Research Data Collector — Streamlit App
=====================================================
Works on: Streamlit Cloud, local PC, any server.

REPO STRUCTURE NEEDED:
  app.py  (or research_app.py — rename to app.py for Streamlit Cloud)
  nyztrade_historical_gex.py
  requirements.txt
"""

import streamlit as st

# ── Crash-safe import with clear error message ─────────────────────────────────
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import nyztrade_historical_gex as col
    COLLECTOR_OK = True
except ImportError as e:
    COLLECTOR_OK = False
    IMPORT_ERROR = str(e)
except Exception as e:
    COLLECTOR_OK = False
    IMPORT_ERROR = str(e)

import sqlite3
import json
import time
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

# ── Work directory: try current dir first, fall back to /tmp ─────────────────────
import os

def _pick_work_dir():
    """Pick a writable working directory. Current dir preferred (local),
    /tmp fallback for read-only cloud environments."""
    candidates = [
        Path('.') / 'nyztrade_data',   # local: subfolder in project
        Path('/tmp') / 'nyztrade_research',  # Cloud / any server
    ]
    for p in candidates:
        try:
            p.mkdir(parents=True, exist_ok=True)
            test = p / '.write_test'
            test.write_text('ok')
            test.unlink()
            return p
        except Exception:
            continue
    return Path('/tmp/nyztrade_research')

WORK_DIR = _pick_work_dir()
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Always override collector paths to our writable WORK_DIR
if COLLECTOR_OK:
    try:
        col.DB_PATH         = WORK_DIR / 'nyztrade_research.db'
        col.RAW_DIR         = WORK_DIR / 'bhavcopy_cache'
        col.OHLCV_DIR       = WORK_DIR / 'ohlcv_cache'
        col.EXPORT_DIR      = WORK_DIR / 'research_export'
        col.CHECKPOINT_FILE = WORK_DIR / 'nyztrade_checkpoint.json'
        for d in [col.RAW_DIR, col.OHLCV_DIR, col.EXPORT_DIR]:
            d.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        pass  # will show error later in UI

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYZTrade GEX Research Collector",
    page_icon="📊",
    layout="wide",
)

# ── Show import error prominently if collector failed ───────────────────────────
if not COLLECTOR_OK:
    st.error("❌ Could not import `nyztrade_historical_gex.py`")
    st.code(IMPORT_ERROR)
    st.markdown("""
    **Fix:** Make sure both files are in the same folder / GitHub repo:
    ```
    app.py                        ← this file
    nyztrade_historical_gex.py    ← the collector engine
    requirements.txt              ← with: pandas, numpy, scipy, requests, streamlit
    ```
    """)
    st.stop()

# ── Styling ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.step-box {
    border-left: 4px solid #0ea5e9;
    background: rgba(14,165,233,0.06);
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
}
.stat-row { display:flex; gap:16px; flex-wrap:wrap; margin:8px 0; }
.stat-card {
    background: rgba(30,58,138,0.08);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 8px;
    padding: 12px 18px;
    min-width: 140px;
}
.stat-val { font-size: 1.5rem; font-weight: 700; color: #3b82f6; }
.stat-lbl { font-size: 0.75rem; color: #6b7280; }
.safe-pill {
    display:inline-block;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.35);
    color: #10b981;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────────
# Always reset running=False on page load.
# run_sync() is synchronous — if the page is loading, no pipeline is running.
# This prevents the "stuck disabled buttons" bug after browser refresh.
st.session_state['running'] = False

for key, default in [
    ('prog', 0.0), ('prog_msg', ''),
    ('error', ''), ('step_done', ''),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── DB connection — fresh per call (sqlite3 not thread-safe when shared) ──────────
def fresh_conn():
    """Always return a fresh connection — avoids sqlite3.ProgrammingError across threads."""
    return col.init_db()

# ── Synchronous runner with live UI updates ──────────────────────────────────────
# Streamlit Cloud does not support persistent background threads.
# We run synchronously and update the UI via placeholder widgets.

_prog_bar = None   # set before running
_prog_txt = None

def prog_cb(pct, msg):
    """Update progress bar and text in real time."""
    st.session_state['prog']     = float(pct)
    st.session_state['prog_msg'] = str(msg)
    if _prog_bar is not None:
        try:   _prog_bar.progress(float(pct), str(msg))
        except Exception: pass
    if _prog_txt is not None:
        try:   _prog_txt.caption(f"⏳ {msg}")
        except Exception: pass

def run_sync(fn, prog_placeholder, txt_placeholder):
    """Run fn(conn) synchronously, updating placeholders in real time."""
    global _prog_bar, _prog_txt
    _prog_bar = prog_placeholder
    _prog_txt = txt_placeholder
    st.session_state['running'] = True
    st.session_state['error']   = ''
    try:
        c = col.init_db()
        fn(c)
        c.close()
        st.session_state['step_done'] = '✅ Done!'
    except Exception as e:
        import traceback
        st.session_state['error'] = f"{e}\n{traceback.format_exc()}"
    finally:
        st.session_state['running'] = False
        _prog_bar = None
        _prog_txt = None

# ── Quick DB stats — fresh connection, full try/except ────────────────────────────
def stats():
    s = {'bhavcopy_raw':0,'index_ohlcv':0,'gex_per_strike':0,'gex_daily_summary':0,
         'dl':{},'pending':0}
    try:
        c = fresh_conn()
        for t in ['bhavcopy_raw','index_ohlcv','gex_per_strike','gex_daily_summary']:
            try:
                row = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
                s[t] = row[0] if row else 0
            except Exception:
                s[t] = 0
        try:
            dl = c.execute("SELECT status,COUNT(*) FROM download_log GROUP BY status").fetchall()
            s['dl'] = dict(dl)
        except Exception:
            s['dl'] = {}
        try:
            row = c.execute("""
                SELECT COUNT(*) FROM (
                    SELECT DISTINCT b.trade_date,b.symbol FROM bhavcopy_raw b
                    LEFT JOIN gex_daily_summary g
                        ON b.trade_date=g.trade_date AND b.symbol=g.symbol
                    WHERE g.trade_date IS NULL AND b.underlying_value>0)
            """).fetchone()
            s['pending'] = row[0] if row else 0
        except Exception:
            s['pending'] = 0
        c.close()
    except Exception:
        pass
    return s

def checkpoint():
    try:
        if col.CHECKPOINT_FILE.exists():
            return json.loads(col.CHECKPOINT_FILE.read_text())
    except Exception: pass
    return {}

def qry(sql, params=(), fetchall=True):
    """Safe query helper — fresh connection, returns [] on any error."""
    try:
        c = fresh_conn()
        cur = c.execute(sql, params)
        result = cur.fetchall() if fetchall else cur.fetchone()
        c.close()
        return result if result is not None else ([] if fetchall else None)
    except Exception:
        return [] if fetchall else None

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 NYZTrade")
    st.markdown("**GEX Research Collector**")
    st.caption("Historical GEX pipeline for academic research")
    st.divider()

    st.markdown("**📅 Date Range**")
    c1, c2 = st.columns(2)
    with c1: start_d = st.date_input("From",
                           value=date(2019, 1, 1),
                           min_value=date(2010, 1, 1),
                           max_value=date.today(),
                           key='start')
    with c2: end_d   = st.date_input("To",
                           value=date.today(),
                           min_value=date(2010, 1, 1),
                           max_value=date.today(),
                           key='end')

    st.markdown("**📌 Indices**")
    sel_syms = st.multiselect("", col.SYMBOLS, default=col.SYMBOLS)

    st.divider()
    st.markdown("**🛡️ Safety**")
    for pill in ["WAL Journal Mode", "Disk Cache First", "Checkpoint File", "Idempotent Writes"]:
        st.markdown(f'<span class="safe-pill">✓ {pill}</span>', unsafe_allow_html=True)
    st.caption("Safe to Ctrl-C and restart anytime.\nAll progress is preserved.")

    st.divider()
    # Emergency reset — clears stuck "running" state
    if st.button("🔄 Reset (if buttons stuck)", use_container_width=True,
                 help="Click if buttons appear greyed out when nothing is running"):
        st.session_state['running']  = False
        st.session_state['prog']     = 0.0
        st.session_state['prog_msg'] = ''
        st.session_state['error']    = ''
        st.session_state['step_done'] = ''
        st.rerun()

    st.divider()
    db = col.DB_PATH
    if db.exists():
        st.caption(f"DB: `{db}`  ({db.stat().st_size/1024**2:.1f} MB)")
    else:
        st.caption("DB: not created yet")

    if st.button("🔄 Refresh Stats", use_container_width=True):
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📊 NYZTrade Historical GEX Research Collector")
st.caption(
    "Collects NSE Bhavcopy options OI + Index OHLCV, then computes the full "
    "GEX / VANNA / Cascade pipeline (identical to the live dashboard) over 5+ years."
)

# Persistent placeholders for progress (updated live during sync run)
_header_prog = st.empty()
_header_txt  = st.empty()

if st.session_state.get('step_done'):
    st.success(st.session_state['step_done'])
    # Don't clear — keep showing until user navigates

if st.session_state.get('error'):
    st.error(f"❌ {st.session_state['error'][:500]}")
    if st.button("Clear error"):
        st.session_state['error'] = ''
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
t1, t2, t3, t4, t5, t6 = st.tabs([
    "🚀 Run Pipeline",
    "📥 Step 1 · Bhavcopy",
    "📈 Step 2 · OHLCV",
    "⚙️ Step 3 · Compute GEX",
    "📤 Export",
    "📋 Status & Preview",
])


# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — FULL PIPELINE
# ───────────────────────────────────────────────────────────────────────────────
with t1:
    st.markdown("### Run Complete Pipeline")
    st.markdown("Runs all steps sequentially. **Safe to re-run** — skips already completed work.")

    try:
        s = stats()
    except Exception:
        s = {'bhavcopy_raw':0,'index_ohlcv':0,'gex_daily_summary':0,'pending':0}

    # Welcome message when DB is empty
    if s['bhavcopy_raw'] == 0 and not st.session_state['running']:
        st.success(
            "✅ **App is ready!** Click **Run Full Pipeline** below to start collecting "
            "5 years of NSE Bhavcopy options data + Index OHLCV. "
            "First run takes 3–6 hours. Safe to stop and resume anytime."
        )

    # Stats row
    st.markdown('<div class="stat-row">', unsafe_allow_html=True)
    for label, val in [
        ("Options Rows", f"{s['bhavcopy_raw']:,}"),
        ("OHLCV Rows",   f"{s['index_ohlcv']:,}"),
        ("GEX Days",     f"{s['gex_daily_summary']:,}"),
        ("Pending GEX",  f"{s['pending']:,}"),
    ]:
        st.markdown(
            f'<div class="stat-card"><div class="stat-val">{val}</div>'
            f'<div class="stat-lbl">{label}</div></div>',
            unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Steps overview
    steps = [
        ("1", "📥 Bhavcopy Download",   "Download NSE options OI + LTP per strike (EOD)", "~1–3 hours"),
        ("2", "📈 OHLCV Download",       "Download index Open/High/Low/Close/Volume",       "~5–10 min"),
        ("3", "⚙️ GEX Computation",      "BS solver → GEX/VANNA/Cascade on all data",       "~45 min–2 hrs"),
        ("4", "📐 Return Variables",      "Compute next-day returns, realized vol, ranges",  "< 5 min"),
        ("5", "📤 Export CSVs",           "Write MASTER_DATASET.csv and all outputs",        "< 5 min"),
    ]
    for num, icon_name, desc, est in steps:
        st.markdown(
            f'<div class="step-box"><b>{icon_name}</b> — {desc} &nbsp;&nbsp;'
            f'<span style="color:#6b7280;font-size:0.82rem">⏱ {est}</span></div>',
            unsafe_allow_html=True)

    st.divider()
    if st.button("🚀 Run Full Pipeline", type="primary",
                 use_container_width=True):
        def _all(c):
            col.BhavCopyDownloader(c).download_range(start_d, end_d, prog_cb)
            col.OHLCVDownloader(c).download_range(start_d, end_d, prog_cb)
            col.GEXEngine(c).compute_all(prog_cb)
            col.compute_returns(c, prog_cb)
            col.export_all(c, prog_cb)
        run_sync(_all, _header_prog, _header_txt)
        st.session_state["step_done"] = "✅ Done! Refresh page to see updated stats."

    st.caption("💡 **Tip:** Start before sleeping. Typical run time: 3–4 hours for 7 years of data.")


# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — BHAVCOPY DOWNLOAD
# ───────────────────────────────────────────────────────────────────────────────
with t2:
    st.markdown("### 📥 Step 1 — NSE Bhavcopy Download")
    st.info(
        "Downloads one ZIP file per trading day from NSE archives. "
        "Each file contains OI + LTP per strike per expiry for all index options. "
        "~1,764 files for 7 years. **Already downloaded files are skipped automatically.**"
    )

    dl = qry("SELECT status,COUNT(*) FROM download_log GROUP BY status")
    dl_dict = dict(dl) if dl else {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Downloaded",  dl_dict.get('ok', 0))
    c2.metric("Holidays",    dl_dict.get('holiday', 0))
    c3.metric("Errors",      dl_dict.get('error', 0))

    st.divider()
    ca, cb = st.columns(2)
    with ca:
        if st.button("📥 Download Bhavcopy", type="primary",
                     use_container_width=True):
            _p = st.empty(); _t = st.empty()
            run_sync(lambda c: col.BhavCopyDownloader(c).download_range(start_d, end_d, prog_cb), _p, _t)
            st.session_state['step_done'] = '✅ Bhavcopy download complete!'
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("🔁 Retry Failed Dates",
                         use_container_width=True,
                         help="Clears error status — retries on next download run"):
                c2 = col.init_db()
                c2.execute("DELETE FROM download_log WHERE status='error'")
                c2.commit(); c2.close()
                st.success("Cleared. Click Download Bhavcopy to retry.")
                st.rerun()
        with col_r2:
            if st.button("🏖 Skip Errors (mark as holiday)",
                         use_container_width=True,
                         help="If 1-2 dates keep failing, NSE has no file for them — safe to skip"):
                c2 = col.init_db()
                c2.execute("UPDATE download_log SET status='holiday' WHERE status='error'")
                c2.commit(); c2.close()
                st.success("Marked as holidays. These dates will be permanently skipped.")
                st.rerun()
    with cb:
        # Show recent log
        failed = qry("SELECT trade_date,error_msg FROM download_log WHERE status='error' LIMIT 10")
        if failed:
            st.warning(f"{len(failed)} failed dates:")
            st.dataframe(pd.DataFrame(failed, columns=['Date','Error']),
                         use_container_width=True, hide_index=True)

    # Progress

    # Show sample of what columns NSE is returning (helps debug format changes)
    sample_row = qry(
        "SELECT trade_date FROM download_log WHERE status='error' LIMIT 1",
        fetchall=False)
    if dl_dict.get('error', 0) > 0:
        with st.expander(f"⚠️ {dl_dict.get('error',0)} errors — click to fix"):
            st.markdown("""
**Root cause:** NSE changed Bhavcopy column names in Jan 2024.
New format uses: `TckrSymb`, `XpryDt`, `OptnTp`, `StrkPric`, `OpnIntrst` etc.

**Fix in 2 clicks:**
1. Click **🔁 Retry Failed Dates** button (clears error status)
2. Click **📥 Download Bhavcopy** again

The updated `nyztrade_historical_gex.py` handles all NSE formats automatically.
Make sure you uploaded the **latest** version to GitHub.
            """)
            err_rows = qry(
                "SELECT trade_date, error_msg FROM download_log WHERE status='error' LIMIT 20")
            if err_rows:
                st.dataframe(pd.DataFrame(err_rows, columns=['Date','Error']),
                             use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**Most recent downloads:**")
    recent = qry("SELECT trade_date, status, rows_stored FROM download_log ORDER BY trade_date DESC LIMIT 15")
    if recent:
        st.dataframe(pd.DataFrame(recent, columns=['Date','Status','Rows']),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No downloads yet. Click the button above to start.")


# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 — OHLCV
# ───────────────────────────────────────────────────────────────────────────────
with t3:
    st.markdown("### 📈 Step 2 — Index OHLCV Download")
    st.info(
        "Downloads Open / High / Low / Close / Volume for NIFTY, BANKNIFTY, "
        "FINNIFTY, MIDCPNIFTY from NSE historical index API. "
        "Used for intraday range and realized volatility (dependent variables)."
    )

    ohlcv_cov = qry("SELECT symbol, MIN(trade_date), MAX(trade_date), COUNT(*) FROM index_ohlcv GROUP BY symbol")

    if ohlcv_cov:
        df_ov = pd.DataFrame(ohlcv_cov, columns=['Symbol','From','To','Days'])
        st.dataframe(df_ov, use_container_width=True, hide_index=True)
    else:
        st.warning("No OHLCV data yet.")

    st.divider()
    ca, cb = st.columns(2)
    with ca:
        if st.button("📥 Download OHLCV", type="primary",
                     use_container_width=True):
            _p = st.empty(); _t = st.empty()
            run_sync(lambda c: col.OHLCVDownloader(c).download_range(start_d, end_d, prog_cb), _p, _t)
            st.session_state['step_done'] = '✅ OHLCV download complete! Stats updated below.'
    with cb:
        if st.button("📐 Compute Returns from OHLCV",
                     use_container_width=True):
            _p = st.empty(); _t = st.empty()
            run_sync(lambda c: col.compute_returns(c, prog_cb), _p, _t)
            st.session_state['step_done'] = '✅ Return variables computed!'


    # Preview
    st.divider()
    sym_sel = st.selectbox("Preview symbol", col.SYMBOLS, key='ohlcv_sym')
    df_prev = pd.read_sql("""
        SELECT trade_date, open, high, low, close, volume, change_pct
        FROM index_ohlcv WHERE symbol=? ORDER BY trade_date DESC LIMIT 20
    """, fresh_conn(), params=(sym_sel,))
    if not df_prev.empty:
        st.dataframe(df_prev, use_container_width=True, hide_index=True)
    else:
        st.info(f"No OHLCV data for {sym_sel} yet.")


# ───────────────────────────────────────────────────────────────────────────────
# TAB 4 — GEX COMPUTE
# ───────────────────────────────────────────────────────────────────────────────
with t4:
    st.markdown("### ⚙️ Step 3 — GEX Analytics Computation")
    st.info(
        "Runs the **exact same** Black-Scholes IV solver + GEX/VANNA/Cascade "
        "pipeline as the live NYZTrade dashboard, over all downloaded historical data. "
        "Skips already-computed dates automatically."
    )

    s = stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Already Computed", f"{s['gex_daily_summary']:,}")
    c2.metric("Pending",          f"{s['pending']:,}")
    cp = checkpoint()
    c3.metric("Last Checkpoint",  cp.get('last_gex', '—'))

    st.markdown("""
    **What gets computed per trading day:**
    - IV solved via bisection for every strike × expiry
    - BS Gamma, Vanna, Delta (vectorised numpy)
    - Net GEX, VANNA, DEX, Enhanced OI GEX
    - GEX Flip Zones, VANNA Flip Zones (Vacuum/Support/Trap/Resistance)
    - Bear & Bull Cascade Mathematics
    - All stored to `gex_per_strike` + `gex_daily_summary` tables
    """)

    st.divider()
    ca, cb = st.columns(2)
    with ca:
        if st.button("⚙️ Compute GEX (pending only)", type="primary",
                     use_container_width=True):
            _p = st.empty(); _t = st.empty()
            run_sync(lambda c: col.GEXEngine(c).compute_all(prog_cb), _p, _t)
            st.session_state['step_done'] = '✅ GEX computation complete!'
    with cb:
        if st.button("📐 Step 4: Compute Returns",
                     use_container_width=True):
            _p = st.empty(); _t = st.empty()
            run_sync(lambda c: col.compute_returns(c, prog_cb), _p, _t)
            st.session_state['step_done'] = '✅ Return variables computed!'

    if st.session_state['running']:
        st.progress(st.session_state['prog'], st.session_state['prog_msg'])

    # Per-symbol GEX coverage
    st.divider()
    st.markdown("**GEX Computed Coverage:**")
    gex_cov = qry("""SELECT symbol, MIN(trade_date), MAX(trade_date), COUNT(*), AVG(net_gex_total) as avg_gex, COUNT(CASE WHEN gex_regime='POSITIVE' THEN 1 END) as pos_days, COUNT(CASE WHEN gex_regime='NEGATIVE' THEN 1 END) as neg_days FROM gex_daily_summary GROUP BY symbol""")
    if gex_cov:
        df_gc = pd.DataFrame(gex_cov, columns=[
            'Symbol','From','To','Days','Avg GEX (B)','Positive Days','Negative Days'])
        df_gc['Avg GEX (B)'] = df_gc['Avg GEX (B)'].round(4)
        st.dataframe(df_gc, use_container_width=True, hide_index=True)
    else:
        st.info("No GEX data computed yet.")


# ───────────────────────────────────────────────────────────────────────────────
# TAB 5 — EXPORT
# ───────────────────────────────────────────────────────────────────────────────
with t5:
    st.markdown("### 📤 Export Research Datasets")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Files generated:**
        | File | Contents |
        |------|----------|
        | `MASTER_DATASET.csv` | GEX + OHLCV merged ← **use this** |
        | `GEX_MAIN_DATASET.csv` | GEX analytics only |
        | `OHLCV_ALL.csv` | All index OHLCV |
        | `OHLCV_{SYMBOL}.csv` | Per-symbol OHLCV |
        | `GEX_PER_STRIKE.csv` | Strike-level data |
        | `GEX_{SYMBOL}.csv` | Per-symbol GEX |
        """)
    with c2:
        st.markdown("""
        **Dependent variables in MASTER_DATASET:**
        | Variable | Description |
        |----------|-------------|
        | `open/high/low/close` | Index OHLCV |
        | `volume` | Traded volume |
        | `index_return_1d` | Next day return % |
        | `index_return_3d` | 3-day return % |
        | `index_return_5d` | 5-day return % |
        | `index_intraday_range` | (H-L)/L % |
        | `realized_vol_5d` | 5-day realised vol |
        | `realized_vol_21d` | 21-day realised vol |
        """)

    st.divider()

    if st.button("📤 Generate All Export Files", type="primary",
                 use_container_width=False):
        _p = st.empty(); _t = st.empty()
        run_sync(lambda c: col.export_all(c, prog_cb), _p, _t)
        st.session_state['step_done'] = '✅ Export complete! Download files below.'

    if st.session_state['running']:
        st.progress(st.session_state['prog'], st.session_state['prog_msg'])

    # Download buttons for each file
    st.divider()
    st.markdown("**Download Files:**")
    export_dir = col.EXPORT_DIR
    if export_dir.exists():
        csvs = sorted(export_dir.glob('*.csv'))
        if csvs:
            for f in csvs:
                size_mb = f.stat().st_size / 1024**2
                col_n, col_s, col_d = st.columns([4, 1, 2])
                col_n.markdown(f"`{f.name}`")
                col_s.caption(f"{size_mb:.1f} MB")
                with open(f, 'rb') as fp:
                    col_d.download_button(
                        label="⬇ Download",
                        data=fp.read(),
                        file_name=f.name,
                        mime='text/csv',
                        key=f"dl_{f.name}",
                        use_container_width=True,
                    )
        else:
            st.info("No export files yet. Click 'Generate' above.")
    else:
        st.info("Export directory not found. Click 'Generate' above.")


# ───────────────────────────────────────────────────────────────────────────────
# TAB 6 — STATUS & PREVIEW
# ───────────────────────────────────────────────────────────────────────────────
with t6:
    st.markdown("### 📋 Status & Data Preview")

    ptab1, ptab2, ptab3, ptab4, ptab5 = st.tabs([
        "📊 Database", "📈 OHLCV Preview", "⚙️ GEX Preview",
        "📁 Per-Strike", "📝 Log"
    ])

    with ptab1:
        s = stats()
        st.markdown("**Table Row Counts:**")
        for t, d in [
            ('bhavcopy_raw',      'Raw NSE options data'),
            ('index_ohlcv',       'Index OHLCV'),
            ('gex_per_strike',    'GEX per strike per day'),
            ('gex_daily_summary', 'Daily GEX summary (main table)'),
        ]:
            col1, col2, col3 = st.columns([3, 1, 3])
            col1.markdown(f"`{t}`")
            col2.markdown(f"**{s[t]:,}**")
            col3.caption(d)

        st.divider()
        st.markdown("**Checkpoint state:**")
        cp = checkpoint()
        if cp: st.json(cp)
        else:  st.info("No checkpoint yet.")

        st.divider()
        st.markdown("**Files on disk:**")
        file_data = []
        for d, label in [(col.RAW_DIR,'bhavcopy_cache'),
                         (col.OHLCV_DIR,'ohlcv_cache'),
                         (col.EXPORT_DIR,'research_export')]:
            if d.exists():
                files = list(d.glob('*'))
                size  = sum(f.stat().st_size for f in files if f.is_file()) / 1024**2
                file_data.append({'Folder': label, 'Files': len(files), 'Size (MB)': round(size,1)})
        if col.DB_PATH.exists():
            file_data.append({'Folder': 'nyztrade_research.db',
                               'Files': 1,
                               'Size (MB)': round(col.DB_PATH.stat().st_size/1024**2,1)})
        if file_data:
            st.dataframe(pd.DataFrame(file_data), use_container_width=True, hide_index=True)

    with ptab2:
        sym = st.selectbox("Symbol", col.SYMBOLS, key='prev_ohlcv')
        df  = pd.read_sql("""
            SELECT trade_date, open, high, low, close, volume, change_pct
            FROM index_ohlcv WHERE symbol=? ORDER BY trade_date DESC LIMIT 30
        """, fresh_conn(), params=(sym,))
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info(f"No OHLCV for {sym} yet.")

    with ptab3:
        sym2 = st.selectbox("Symbol", col.SYMBOLS, key='prev_gex')
        df2  = pd.read_sql("""
            SELECT trade_date, spot_price, net_gex_total, gex_regime,
                   pcr, iv_skew, bear_cascade_net, bull_cascade_net,
                   index_return_1d, realized_vol_5d,
                   open, high, low, close, volume
            FROM gex_daily_summary g
            LEFT JOIN index_ohlcv o USING(trade_date, symbol)
            WHERE g.symbol=? ORDER BY trade_date DESC LIMIT 30
        """, fresh_conn(), params=(sym2,))
        if not df2.empty:
            st.dataframe(df2, use_container_width=True, hide_index=True)
        else:
            st.info(f"No GEX data for {sym2} yet.")

    with ptab4:
        sym3 = st.selectbox("Symbol", col.SYMBOLS, key='prev_ps')
        dates = qry("SELECT DISTINCT trade_date FROM gex_per_strike WHERE symbol=? ORDER BY trade_date DESC LIMIT 20", (sym3,))
        dt_list = [r[0] for r in dates]
        if dt_list:
            dt_sel = st.selectbox("Date", dt_list, key='prev_dt')
            df3 = pd.read_sql("""
                SELECT strike_price, call_oi, put_oi, call_iv, put_iv,
                       net_gex, net_vanna, net_dex, enhanced_oi_gex
                FROM gex_per_strike
                WHERE symbol=? AND trade_date=? ORDER BY strike_price
            """, fresh_conn(), params=(sym3, dt_sel))
            st.dataframe(df3, use_container_width=True, hide_index=True)
        else:
            st.info("No per-strike GEX data yet. Run Step 3.")

    with ptab5:
        log_path = Path('nyztrade_gex_research.log')
        if log_path.exists():
            lines = log_path.read_text(errors='replace').split('\n')
            st.text_area("Recent log (last 100 lines)",
                         '\n'.join(lines[-100:]), height=350)
        else:
            st.info("Log file not found. It will appear once the pipeline starts running.")


# ── Footer ───────────────────────────────────────────────────────────────────────
st.divider()
col_f1, col_f2, col_f3 = st.columns(3)
col_f1.caption("NYZTrade Analytics | Dr. Niyas N")
col_f2.caption("Data: NSE Bhavcopy (FREE public archive)")
col_f3.caption("Safe: WAL + disk cache + checkpoint resume")
