"""
NYZTrade GEX Research Data Collector — Streamlit UI
====================================================
Visual interface for running the historical GEX data pipeline.
All data is safe — DB uses WAL mode + disk caching before every write.
"""

import streamlit as st
import sqlite3
import pandas as pd
import json
import time
import threading
import subprocess
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

# ── Import the collector engine ─────────────────────────────────────────────────
# The collector module must be in the same folder as this app.py
sys.path.insert(0, str(Path(__file__).parent))
import nyztrade_historical_gex as collector

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYZTrade GEX Research Collector",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: rgba(30,58,138,0.08);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.step-header {
    background: linear-gradient(90deg,rgba(14,165,233,0.15),rgba(14,165,233,0));
    border-left: 4px solid #0ea5e9;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 12px 0 6px 0;
    font-weight: 600;
}
.safe-badge {
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.4);
    color: #10b981;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.78rem;
    font-weight: 600;
}
.warn-badge {
    background: rgba(245,158,11,0.12);
    border: 1px solid rgba(245,158,11,0.35);
    color: #f59e0b;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.78rem;
}
.stButton > button {
    font-weight: 600;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_db():
    return collector.init_db()

def db_stats(conn):
    stats = {}
    for t in ['bhavcopy_raw','index_ohlcv','gex_per_strike','gex_daily_summary']:
        try: stats[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except: stats[t] = 0
    for sym in collector.SYMBOLS:
        g = conn.execute("SELECT MIN(trade_date),MAX(trade_date),COUNT(*) FROM gex_daily_summary WHERE symbol=?",(sym,)).fetchone()
        o = conn.execute("SELECT COUNT(*) FROM index_ohlcv WHERE symbol=?",(sym,)).fetchone()
        stats[f'gex_{sym}']  = g if g and g[0] else (None,None,0)
        stats[f'ohlcv_{sym}'] = o[0] if o else 0
    dl = conn.execute("SELECT status,COUNT(*) FROM download_log GROUP BY status").fetchall()
    stats['download_log'] = dict(dl)
    return stats

def get_checkpoint():
    try:
        if collector.CHECKPOINT_FILE.exists():
            return json.loads(collector.CHECKPOINT_FILE.read_text())
    except Exception: pass
    return {}

def export_files():
    files = []
    if collector.EXPORT_DIR.exists():
        for f in sorted(collector.EXPORT_DIR.glob('*.csv')):
            size = f.stat().st_size / 1024**2
            files.append({'File': f.name, 'Size (MB)': round(size,2),
                          'Rows': pd.read_csv(f, nrows=1, on_bad_lines='skip').shape[0]})
    return files

def run_step_bg(fn, args=(), kwargs={}):
    """Run a collector step in background thread and update session state."""
    st.session_state['running'] = True
    st.session_state['step_log'] = []
    def target():
        try:
            fn(*args, **kwargs)
        except Exception as e:
            st.session_state['step_error'] = str(e)
        finally:
            st.session_state['running'] = False
    t = threading.Thread(target=target, daemon=True)
    t.start()


# ── Session state defaults ───────────────────────────────────────────────────────
if 'running' not in st.session_state:   st.session_state['running'] = False
if 'step_log' not in st.session_state:  st.session_state['step_log'] = []
if 'step_error' not in st.session_state:st.session_state['step_error'] = ''
if 'progress' not in st.session_state:  st.session_state['progress'] = 0.0
if 'prog_msg' not in st.session_state:  st.session_state['prog_msg'] = ''

def progress_cb(pct, msg):
    st.session_state['progress'] = float(pct)
    st.session_state['prog_msg'] = msg


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://via.placeholder.com/180x50/1B3A6B/FFFFFF?text=NYZTrade", width=180)
    st.markdown("### 📊 GEX Research Collector")
    st.caption("Historical GEX data pipeline for academic research")
    st.markdown("---")

    st.markdown("**📅 Date Range**")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", date(2019, 1, 1), min_value=date(2010,1,1))
    with col2:
        end_date = st.date_input("End", date.today())

    st.markdown("**📌 Symbols**")
    selected_symbols = st.multiselect(
        "Indices to collect",
        options=collector.SYMBOLS,
        default=collector.SYMBOLS,
    )

    st.markdown("---")
    st.markdown("**🛡️ Safety Status**")
    st.markdown('<span class="safe-badge">✅ WAL Journal Mode</span>', unsafe_allow_html=True)
    st.markdown('<span class="safe-badge">✅ Disk Cache Before DB</span>', unsafe_allow_html=True)
    st.markdown('<span class="safe-badge">✅ Checkpoint File</span>', unsafe_allow_html=True)
    st.caption("Data is safe on crash, power-off, or Ctrl-C.\nRestart any step — it resumes from where it stopped.")

    st.markdown("---")
    db_path = collector.DB_PATH
    if db_path.exists():
        size_mb = db_path.stat().st_size / 1024**2
        st.metric("DB Size", f"{size_mb:.1f} MB")
    st.caption(f"DB: `{db_path.absolute()}`")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📊 NYZTrade Historical GEX Research Collector")
st.caption("Collects 5-year NSE Bhavcopy options data + Index OHLCV and computes the full dashboard GEX/VANNA/Cascade pipeline historically.")

conn = get_db()

# ── Tab layout ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚀 Pipeline", "📈 OHLCV", "📊 GEX Compute", "📤 Export", "📋 Status", "📚 Data Preview"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PIPELINE (run all steps)
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Run Complete Pipeline")
    st.markdown("Runs all 5 steps sequentially. Safe to re-run — skips already completed work.")

    col_a, col_b = st.columns([2,1])
    with col_a:
        st.markdown("""
        **Steps:**
        1. 📥 Download NSE Bhavcopy (options OI + LTP) — ~3-5 hours
        2. 📈 Download Index OHLCV (open/high/low/close/volume) — ~20 min
        3. ⚙️ Compute GEX, VANNA, Cascade (dashboard pipeline) — ~4-8 hours
        4. 📐 Compute return variables (dependent variables for regression)
        5. 📤 Export all CSVs (MASTER_DATASET.csv + per-symbol)
        """)
    with col_b:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        stats = db_stats(conn)
        st.metric("Options Rows", f"{stats['bhavcopy_raw']:,}")
        st.metric("OHLCV Rows",   f"{stats['index_ohlcv']:,}")
        st.metric("GEX Days",     f"{stats['gex_daily_summary']:,}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Progress bar (shown when running)
    prog_container = st.empty()
    if st.session_state['running']:
        prog_container.progress(st.session_state['progress'], st.session_state['prog_msg'])

    if st.session_state.get('step_error'):
        st.error(f"Error: {st.session_state['step_error']}")

    col_run, col_stop = st.columns([2,1])
    with col_run:
        if st.button("🚀 Run Full Pipeline", type="primary",
                     disabled=st.session_state['running'],
                     use_container_width=True):
            def run_all():
                conn2 = collector.init_db()
                collector.BhavCopyDownloader(conn2).download_range(
                    start_date, end_date, progress_cb)
                collector.OHLCVDownloader(conn2).download_range(
                    start_date, end_date, progress_cb)
                collector.GEXEngine(conn2).compute_all(progress_cb)
                collector.compute_returns(conn2, progress_cb)
                collector.export_all(conn2, progress_cb)
                conn2.close()
            run_step_bg(run_all)
            st.rerun()
    with col_stop:
        if st.button("⏹ Stop", disabled=not st.session_state['running'],
                     use_container_width=True):
            st.session_state['running'] = False
            st.success("Stop requested. Current operation will finish its current row, then stop. Data already collected is safe.")

    if st.session_state['running']:
        time.sleep(2); st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — OHLCV (individual step)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📈 Index OHLCV Download")
    st.markdown("Downloads Open/High/Low/Close/Volume for all selected indices from NSE historical index API.")

    st.info("""
    **Source**: NSE Historical Index API (free, public)
    **Covers**: NIFTY 50, NIFTY BANK, NIFTY FIN SERVICE, NIFTY MIDCAP SELECT
    **Data**: Daily OHLCV + % change
    **Used for**: Intraday range, realized volatility, next-day returns (dependent variables in regression)
    """)

    # Show existing OHLCV coverage
    ohlcv_rows = conn.execute("SELECT symbol,MIN(trade_date),MAX(trade_date),COUNT(*) FROM index_ohlcv GROUP BY symbol").fetchall()
    if ohlcv_rows:
        st.markdown("**Current OHLCV Coverage:**")
        df_ov = pd.DataFrame(ohlcv_rows, columns=['Symbol','First Date','Last Date','Rows'])
        st.dataframe(df_ov, use_container_width=True, hide_index=True)
    else:
        st.warning("No OHLCV data yet. Click Download to start.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Download OHLCV", type="primary",
                     disabled=st.session_state['running'],
                     use_container_width=True):
            def run_ohlcv():
                conn2 = collector.init_db()
                collector.OHLCVDownloader(conn2).download_range(start_date, end_date, progress_cb)
                conn2.close()
            run_step_bg(run_ohlcv)
            st.rerun()
    with col2:
        if st.button("📐 Compute Returns from OHLCV",
                     disabled=st.session_state['running'],
                     use_container_width=True):
            def run_returns():
                conn2 = collector.init_db()
                collector.compute_returns(conn2, progress_cb)
                conn2.close()
            run_step_bg(run_returns)
            st.rerun()

    if st.session_state['running']:
        st.progress(st.session_state['progress'], st.session_state['prog_msg'])
        time.sleep(2); st.rerun()

    # Preview latest OHLCV
    st.markdown("---")
    st.markdown("**Preview: Latest OHLCV data**")
    sym_sel = st.selectbox("Symbol", collector.SYMBOLS, key='ohlcv_preview_sym')
    df_prev = pd.read_sql(
        "SELECT trade_date,open,high,low,close,volume,change_pct FROM index_ohlcv WHERE symbol=? ORDER BY trade_date DESC LIMIT 30",
        conn, params=(sym_sel,))
    if not df_prev.empty:
        # Colour change_pct
        st.dataframe(df_prev.style.applymap(
            lambda v: 'color: #10b981' if isinstance(v,float) and v>0 else
                      ('color: #ef4444' if isinstance(v,float) and v<0 else ''),
            subset=['change_pct']),
            use_container_width=True, hide_index=True)
    else:
        st.info("No OHLCV data for this symbol yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GEX COMPUTE
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### ⚙️ GEX Analytics Computation")
    st.markdown("Runs the **exact same** BS Greeks + GEX/VANNA/Cascade pipeline as the live dashboard, over all downloaded historical data.")

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        pending = conn.execute("""
            SELECT COUNT(DISTINCT b.trade_date||b.symbol) FROM bhavcopy_raw b
            LEFT JOIN gex_daily_summary g ON b.trade_date=g.trade_date AND b.symbol=g.symbol
            WHERE g.trade_date IS NULL AND b.underlying_value>0
        """).fetchone()[0]
        st.metric("Pending Computation", f"{pending:,} day-symbols")
    with col_m2:
        done = conn.execute("SELECT COUNT(*) FROM gex_daily_summary").fetchone()[0]
        st.metric("Computed", f"{done:,} day-symbols")
    with col_m3:
        cp = get_checkpoint()
        st.metric("Last Checkpoint", cp.get('last_gex', 'none'))

    st.markdown("---")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⚙️ Compute GEX (pending only)", type="primary",
                     disabled=st.session_state['running'], use_container_width=True):
            def run_gex():
                conn2 = collector.init_db()
                collector.GEXEngine(conn2).compute_all(progress_cb)
                conn2.close()
            run_step_bg(run_gex)
            st.rerun()
    with col_b2:
        if st.button("📥 Download Bhavcopy Only",
                     disabled=st.session_state['running'], use_container_width=True):
            def run_dl():
                conn2 = collector.init_db()
                collector.BhavCopyDownloader(conn2).download_range(start_date, end_date, progress_cb)
                conn2.close()
            run_step_bg(run_dl)
            st.rerun()

    if st.session_state['running']:
        st.progress(st.session_state['progress'], st.session_state['prog_msg'])
        time.sleep(2); st.rerun()

    # Download log
    st.markdown("---")
    st.markdown("**Download Log Summary**")
    dl_log = conn.execute("""
        SELECT status, COUNT(*) as count FROM download_log GROUP BY status
    """).fetchall()
    if dl_log:
        df_dl = pd.DataFrame(dl_log, columns=['Status','Days'])
        st.dataframe(df_dl, use_container_width=True, hide_index=True)

    # Failed dates
    failed = conn.execute(
        "SELECT trade_date,error_msg FROM download_log WHERE status='error' LIMIT 20"
    ).fetchall()
    if failed:
        st.warning(f"{len(failed)} failed downloads:")
        st.dataframe(pd.DataFrame(failed, columns=['Date','Error']),
                     use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📤 Export Research CSVs")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("""
        **Files generated:**
        - `MASTER_DATASET.csv` — GEX + OHLCV merged ← **use this for regression**
        - `GEX_MAIN_DATASET.csv` — GEX analytics only
        - `OHLCV_ALL.csv` — all index OHLCV
        - `OHLCV_{SYMBOL}.csv` — per-symbol OHLCV
        - `GEX_PER_STRIKE.csv` — strike-level cross-sectional data
        - `GEX_{SYMBOL}.csv` — per-symbol GEX summary
        """)
    with col_e2:
        st.markdown("""
        **Dependent variables in MASTER_DATASET:**
        - `index_return_1d` — next day return %
        - `index_return_3d` — 3-day return %
        - `index_return_5d` — 5-day return %
        - `index_intraday_range` — (H-L)/L %
        - `realized_vol_5d` — 5-day realised vol
        - `realized_vol_21d` — 21-day realised vol
        - `open`, `high`, `low`, `close`, `volume`
        """)

    st.markdown("---")

    if st.button("📤 Export All CSVs", type="primary",
                 disabled=st.session_state['running'], use_container_width=False):
        def run_export():
            conn2 = collector.init_db()
            collector.export_all(conn2, progress_cb)
            conn2.close()
        run_step_bg(run_export)
        st.rerun()

    if st.session_state['running']:
        st.progress(st.session_state['progress'], st.session_state['prog_msg'])
        time.sleep(2); st.rerun()

    # List export files with download buttons
    st.markdown("---")
    st.markdown("**Available Export Files:**")
    if collector.EXPORT_DIR.exists():
        csvs = sorted(collector.EXPORT_DIR.glob('*.csv'))
        if csvs:
            for f in csvs:
                size = f.stat().st_size / 1024**2
                col_fn, col_sz, col_dl = st.columns([4,1,2])
                col_fn.markdown(f"`{f.name}`")
                col_sz.caption(f"{size:.1f} MB")
                with open(f,'rb') as fp:
                    col_dl.download_button(
                        label="⬇ Download",
                        data=fp.read(),
                        file_name=f.name,
                        mime='text/csv',
                        key=f"dl_{f.name}",
                        use_container_width=True,
                    )
        else:
            st.info("No export files yet. Click Export to generate.")
    else:
        st.info("Export directory not found. Click Export to generate.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — STATUS
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 📋 Collection Status")

    if st.button("🔄 Refresh Status"):
        st.rerun()

    # Per-symbol coverage table
    st.markdown("**Per-Symbol Coverage:**")
    cov_rows = []
    for sym in collector.SYMBOLS:
        g = conn.execute("SELECT MIN(trade_date),MAX(trade_date),COUNT(*) FROM gex_daily_summary WHERE symbol=?",(sym,)).fetchone()
        o = conn.execute("SELECT MIN(trade_date),MAX(trade_date),COUNT(*) FROM index_ohlcv WHERE symbol=?",(sym,)).fetchone()
        b = conn.execute("SELECT COUNT(DISTINCT trade_date) FROM bhavcopy_raw WHERE symbol=?",(sym,)).fetchone()
        cov_rows.append({
            'Symbol':     sym,
            'Bhavcopy Days': b[0] if b else 0,
            'OHLCV Days': o[2] if o and o[0] else 0,
            'OHLCV From': o[0] if o and o[0] else '—',
            'OHLCV To':   o[1] if o and o[1] else '—',
            'GEX Days':   g[2] if g and g[0] else 0,
            'GEX From':   g[0] if g and g[0] else '—',
            'GEX To':     g[1] if g and g[1] else '—',
        })
    st.dataframe(pd.DataFrame(cov_rows), use_container_width=True, hide_index=True)

    # DB table sizes
    st.markdown("---")
    st.markdown("**Database Table Sizes:**")
    tbl_rows = []
    for t in ['bhavcopy_raw','index_ohlcv','gex_per_strike','gex_daily_summary','download_log']:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        tbl_rows.append({'Table': t, 'Rows': f"{n:,}"})
    st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)

    # Checkpoint
    st.markdown("---")
    st.markdown("**Checkpoint (safe resume state):**")
    cp = get_checkpoint()
    if cp:
        st.json(cp)
    else:
        st.info("No checkpoint yet — will be created when first step runs.")

    # Log viewer
    st.markdown("---")
    st.markdown("**Recent Log:**")
    log_path = Path('nyztrade_gex_research.log')
    if log_path.exists():
        lines = log_path.read_text().split('\n')
        st.text_area("Log", '\n'.join(lines[-80:]), height=300)
    else:
        st.info("Log file not found yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DATA PREVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### 📚 Data Preview")

    preview_tab1, preview_tab2, preview_tab3, preview_tab4 = st.tabs(
        ["MASTER_DATASET", "OHLCV", "GEX Summary", "Per-Strike GEX"])

    with preview_tab1:
        st.caption("GEX + OHLCV merged — the main regression table")
        sym_p = st.selectbox("Symbol", collector.SYMBOLS, key='master_sym')
        lim_p = st.slider("Rows to show", 10, 200, 30, key='master_rows')
        df_m = pd.read_sql("""
            SELECT g.trade_date, g.symbol, o.open, o.high, o.low, o.close, o.volume, o.change_pct,
                   g.net_gex_total, g.gex_regime, g.gex_flip_zone_1,
                   g.dominant_call_wall, g.dominant_put_wall,
                   g.bear_cascade_net, g.bull_cascade_net, g.pcr,
                   g.iv_skew, g.atm_call_iv, g.net_vanna_total,
                   g.index_return_1d, g.index_return_3d, g.realized_vol_5d
            FROM gex_daily_summary g
            LEFT JOIN index_ohlcv o ON g.trade_date=o.trade_date AND g.symbol=o.symbol
            WHERE g.symbol=? ORDER BY g.trade_date DESC LIMIT ?
        """, conn, params=(sym_p, lim_p))
        if not df_m.empty:
            st.dataframe(df_m.style.applymap(
                lambda v: 'color: #10b981' if isinstance(v,(int,float)) and v>0 else
                          ('color: #ef4444' if isinstance(v,(int,float)) and v<0 else ''),
                subset=[c for c in ['net_gex_total','index_return_1d','change_pct','bear_cascade_net']
                        if c in df_m.columns]),
                use_container_width=True, hide_index=True)
            st.caption(f"{len(df_m)} rows shown")
        else:
            st.info("No data yet. Run the pipeline first.")

    with preview_tab2:
        sym_o = st.selectbox("Symbol", collector.SYMBOLS, key='ohlcv_tab_sym')
        df_o  = pd.read_sql(
            "SELECT * FROM index_ohlcv WHERE symbol=? ORDER BY trade_date DESC LIMIT 50",
            conn, params=(sym_o,))
        if not df_o.empty:
            st.dataframe(df_o, use_container_width=True, hide_index=True)
        else:
            st.info("No OHLCV data. Run Step 2 (OHLCV Download).")

    with preview_tab3:
        sym_g = st.selectbox("Symbol", collector.SYMBOLS, key='gex_tab_sym')
        df_g  = pd.read_sql("""
            SELECT trade_date,spot_price,net_gex_total,gex_regime,pcr,iv_skew,
                   bear_cascade_net,bull_cascade_net,index_return_1d,realized_vol_5d
            FROM gex_daily_summary WHERE symbol=? ORDER BY trade_date DESC LIMIT 50
        """, conn, params=(sym_g,))
        if not df_g.empty:
            st.dataframe(df_g, use_container_width=True, hide_index=True)
        else:
            st.info("No GEX data. Run Step 3 (Compute GEX).")

    with preview_tab4:
        col_ps1, col_ps2 = st.columns(2)
        with col_ps1:
            sym_ps = st.selectbox("Symbol", collector.SYMBOLS, key='ps_sym')
        with col_ps2:
            dates_avail = conn.execute(
                "SELECT DISTINCT trade_date FROM gex_per_strike WHERE symbol=? ORDER BY trade_date DESC LIMIT 30",
                (sym_ps,)).fetchall()
            date_list = [r[0] for r in dates_avail]
            dt_sel = st.selectbox("Date", date_list if date_list else ["No data"],
                                  key='ps_date')
        df_ps = pd.read_sql("""
            SELECT strike_price,call_oi,put_oi,call_iv,put_iv,
                   call_gamma,call_vanna,net_gex,net_vanna,net_dex,enhanced_oi_gex
            FROM gex_per_strike WHERE symbol=? AND trade_date=?
            ORDER BY strike_price
        """, conn, params=(sym_ps, dt_sel)) if date_list else pd.DataFrame()
        if not df_ps.empty:
            st.dataframe(df_ps.style.applymap(
                lambda v: 'background-color: rgba(16,185,129,0.1)' if isinstance(v,(int,float)) and v>0 else
                          ('background-color: rgba(239,68,68,0.1)' if isinstance(v,(int,float)) and v<0 else ''),
                subset=['net_gex','net_vanna']),
                use_container_width=True, hide_index=True)
        else:
            st.info("No per-strike data. Run Step 3 (Compute GEX).")


# ── Footer ───────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "NYZTrade Analytics | Dr. Niyas N | GEX Research Collector v1.0 | "
    "Data: NSE Bhavcopy (FREE) | Safe: WAL + disk cache + checkpoint"
)
