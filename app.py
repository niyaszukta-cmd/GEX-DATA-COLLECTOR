"""
NYZTrade GEX Collector — Streamlit App
Dhan API only. NIFTY ATM ± 15 strikes.
"""
import streamlit as st

st.set_page_config(
    page_title="NYZTrade GEX Collector",
    page_icon="📊",
    layout="wide",
)

try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import gex_dhan_collector as gc
    READY = True
except Exception as e:
    READY = False
    ERR   = str(e)

import pandas as pd
import time
from datetime import date, timedelta

if not READY:
    st.error(f"❌ Import failed: {ERR}")
    st.code("Make sure gex_dhan_collector.py is in the same repo folder.")
    st.stop()

for k, v in [('running',False),('prog',0.0),('msg',''),('done',''),('err','')]:
    if k not in st.session_state:
        st.session_state[k] = v

st.session_state['running'] = False

_pbar = None

def prog(pct, msg):
    st.session_state['prog'] = float(pct)
    st.session_state['msg']  = str(msg)
    global _pbar
    if _pbar:
        try:
            _pbar.progress(float(pct), str(msg))
        except Exception:
            pass

def fresh_conn():
    return gc.init_db()

def qry(sql, params=(), one=False):
    try:
        c = fresh_conn()
        r = c.execute(sql, params)
        result = r.fetchone() if one else r.fetchall()
        c.close()
        return result
    except Exception:
        return None if one else []

def get_client():
    cid = st.session_state.get("dhan_client_id", "")
    tok = st.session_state.get("dhan_token", "")
    if not cid or not tok:
        st.error("Enter Dhan Client ID and Access Token in the sidebar first.")
        return None
    return gc.DhanClient(cid, tok)

def run(fn):
    client = get_client()
    if not client:
        return
    conn = fresh_conn()
    collector = gc.GEXCollector(conn, client)
    try:
        fn(collector)
        conn.close()
    except Exception as e:
        import traceback
        st.session_state['err'] = f"{e}\n{traceback.format_exc()[:600]}"
        try:
            conn.close()
        except Exception:
            pass

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## NYZTrade GEX")
    st.caption("Dhan API · NIFTY ATM ±15")
    st.divider()

    st.markdown("**🔑 Dhan Credentials**")
    client_id = st.text_input(
        "Client ID",
        value=st.session_state.get("dhan_client_id", ""),
        key="inp_cid",
        placeholder="1100480354"
    )
    token = st.text_input(
        "Access Token",
        type="password",
        value=st.session_state.get("dhan_token", ""),
        key="inp_tok",
        placeholder="eyJ0eX..."
    )
    if client_id:
        st.session_state["dhan_client_id"] = client_id
    if token:
        st.session_state["dhan_token"] = token

    if client_id and token:
        st.success("Credentials saved")
    else:
        st.warning("Enter credentials above")

    st.divider()
    st.markdown("**📅 Date Range**")
    c1, c2 = st.columns(2)
    with c1:
        start_d = st.date_input("From", date(2024, 1, 1),
                                 min_value=date(2019,1,1),
                                 max_value=date.today(), key="sd")
    with c2:
        end_d = st.date_input("To", date.today(),
                               min_value=date(2019,1,1),
                               max_value=date.today(), key="ed")

    st.divider()
    st.caption(f"Strikes: ATM ±{gc.ATM_RANGE} (step ₹{gc.STRIKE_INTERVAL})")
    st.caption(f"Total per day: {gc.ATM_RANGE*2+1} strikes × 2 = {(gc.ATM_RANGE*2+1)*2} calls")
    st.caption(f"Lot size: {gc.LOT_SIZE}  |  r: {gc.RISK_FREE*100:.1f}%")

    st.divider()
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state['running'] = False
        st.session_state['done']    = ''
        st.session_state['err']     = ''
        st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📊 NYZTrade GEX Collector")
st.caption("Fetches NIFTY options (ATM ±15 strikes) from Dhan API · Computes BS GEX/VANNA/DEX")

_pbar = st.empty()

if st.session_state.get('done'):
    st.success(st.session_state['done'])
if st.session_state.get('err'):
    with st.expander("❌ Error details"):
        st.code(st.session_state['err'])

# ── Stats ──────────────────────────────────────────────────────────────────────
s = {}
for t in ['nifty_ohlcv', 'options_raw', 'gex_per_strike', 'gex_daily']:
    row = qry(f"SELECT COUNT(*) FROM {t}", one=True)
    s[t] = row[0] if row else 0

ca, cb, cc, cd = st.columns(4)
ca.metric("NIFTY Days",    f"{s['nifty_ohlcv']:,}")
cb.metric("Options Rows",  f"{s['options_raw']:,}")
cc.metric("GEX Strikes",   f"{s['gex_per_strike']:,}")
cd.metric("GEX Days",      f"{s['gex_daily']:,}")

gex_r = qry("SELECT MIN(trade_date),MAX(trade_date) FROM gex_daily", one=True)
if gex_r and gex_r[0]:
    st.caption(f"GEX: **{gex_r[0]}** → **{gex_r[1]}**")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "🚀 Run All",
    "📈 Step 1 · Prices",
    "📥 Step 2 · Options",
    "⚙️ Step 3 · GEX",
    "📤 Export",
])

# ── RUN ALL ────────────────────────────────────────────────────────────────────
with t1:
    st.markdown("### Run Complete Pipeline")

    days_est = (end_d - start_d).days * 5 // 7
    calls_est = days_est * (gc.ATM_RANGE*2+1) * 2
    mins_est  = calls_est // 600

    st.info(f"""
**Estimate for {start_d} → {end_d}:**
- ~{days_est} trading days
- ~{calls_est:,} Dhan API calls (Step 2)
- ~{mins_est} minutes for Step 2
""")

    st.markdown("""
| Step | Action | Estimated time |
|------|--------|----------------|
| 1 | NIFTY daily closing prices | ~1 min |
| 2 | Options LTP+OI per strike | ~{} min |
| 3 | BS GEX/VANNA/DEX compute | ~2 min |
""".format(mins_est))

    if st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True):
        def _all(col):
            col.fetch_nifty_prices(start_d, end_d, prog)
            col.fetch_options_data(start_d, end_d, prog)
            col.compute_gex(prog)
            col.export(prog)
        run(_all)
        st.session_state['done'] = "Pipeline complete! Check Export tab."

# ── PRICES ─────────────────────────────────────────────────────────────────────
with t2:
    st.markdown("### Step 1 — NIFTY Daily Prices")
    st.info("Fetches NIFTY 50 OHLCV from Dhan. Used to calculate ATM = round(spot / 50) × 50")

    cov = qry("SELECT MIN(trade_date),MAX(trade_date),COUNT(*) FROM nifty_ohlcv", one=True)
    if cov and cov[0]:
        st.success(f"✅ {cov[2]} trading days: {cov[0]} → {cov[1]}")
    else:
        st.warning("No data yet.")

    if st.button("📈 Download NIFTY Prices", type="primary"):
        run(lambda c: c.fetch_nifty_prices(start_d, end_d, prog))
        st.session_state['done'] = "NIFTY prices downloaded!"

    rows = qry("SELECT trade_date,open,high,low,close FROM nifty_ohlcv ORDER BY trade_date DESC LIMIT 10")
    if rows:
        st.dataframe(
            pd.DataFrame(rows, columns=['Date','Open','High','Low','Close']),
            use_container_width=True, hide_index=True)

# ── OPTIONS ────────────────────────────────────────────────────────────────────
with t3:
    st.markdown("### Step 2 — Options LTP + OI")
    st.info(
        f"For each day, fetches LTP and OI for **{(gc.ATM_RANGE*2+1)*2} contracts** "
        "(ATM±15 × CE+PE). Uses Dhan security ID lookup per contract."
    )

    fl = qry("SELECT status, COUNT(*) FROM fetch_log GROUP BY status")
    if fl:
        fld = dict(fl)
        ca, cb, cc = st.columns(3)
        ca.metric("Fetched",  fld.get('ok', 0))
        cb.metric("No Data",  fld.get('no_data', 0))
        cc.metric("Errors",   fld.get('error', 0))

        nifty_n = qry("SELECT COUNT(*) FROM nifty_ohlcv", one=True)
        n = nifty_n[0] if nifty_n else 0
        expected = n * (gc.ATM_RANGE*2+1) * 2
        done_n = fld.get('ok',0) + fld.get('no_data',0)
        if expected > 0:
            st.progress(min(done_n/expected, 1.0),
                        f"{done_n:,} / {expected:,} ({done_n*100//expected}%)")

    if st.button("📥 Fetch Options Data", type="primary"):
        run(lambda c: c.fetch_options_data(start_d, end_d, prog))
        st.session_state['done'] = "Options data fetched!"

    sample = qry("""
        SELECT trade_date, strike, option_type, close, open_interest, spot_price
        FROM options_raw ORDER BY trade_date DESC, strike LIMIT 20
    """)
    if sample:
        st.markdown("**Latest options rows:**")
        st.dataframe(
            pd.DataFrame(sample, columns=['Date','Strike','Type','LTP','OI','Spot']),
            use_container_width=True, hide_index=True)

# ── GEX ────────────────────────────────────────────────────────────────────────
with t4:
    st.markdown("### Step 3 — Compute GEX / VANNA / DEX")
    st.markdown("""
    For each day and each strike:
    1. Solve IV via Black-Scholes bisection (LTP → IV)
    2. Compute Gamma, Vanna, Delta
    3. `net_gex = (Call_OI × Call_Gamma − Put_OI × Put_Gamma) × Spot² / 1e9`
    4. Aggregate to daily: total_gex, regime, flip zone, walls, PCR, IV skew
    """)

    pending_r = qry("""
        SELECT COUNT(DISTINCT o.trade_date) FROM options_raw o
        LEFT JOIN gex_daily g ON o.trade_date=g.trade_date
        WHERE g.trade_date IS NULL
    """, one=True)
    pending_n = pending_r[0] if pending_r else 0

    ca, cb = st.columns(2)
    ca.metric("Computed", s['gex_daily'])
    cb.metric("Pending",  pending_n)

    if st.button("⚙️ Compute GEX", type="primary"):
        run(lambda c: c.compute_gex(prog))
        st.session_state['done'] = "GEX computation complete!"

    if st.button("🔄 Clear GEX + Recompute All"):
        c2 = fresh_conn()
        c2.execute("DELETE FROM gex_daily")
        c2.execute("DELETE FROM gex_per_strike")
        c2.commit(); c2.close()
        st.success("Cleared. Click Compute GEX.")
        st.rerun()

    gex_rows = qry("""
        SELECT trade_date, spot_price, total_net_gex, gex_regime,
               dominant_call_wall, dominant_put_wall, gex_flip_zone,
               pcr, iv_skew, atm_call_iv
        FROM gex_daily ORDER BY trade_date DESC LIMIT 15
    """)
    if gex_rows:
        st.markdown("**Latest GEX daily:**")
        st.dataframe(
            pd.DataFrame(gex_rows, columns=[
                'Date','Spot','Net GEX','Regime',
                'Call Wall','Put Wall','Flip Zone',
                'PCR','IV Skew','ATM Call IV']),
            use_container_width=True, hide_index=True)

    latest_d = qry("SELECT MAX(trade_date) FROM gex_per_strike", one=True)
    if latest_d and latest_d[0]:
        sk = qry("""
            SELECT strike, call_iv, put_iv, call_oi, put_oi, net_gex, net_vanna
            FROM gex_per_strike WHERE trade_date=? ORDER BY strike
        """, (latest_d[0],))
        if sk:
            st.markdown(f"**Per-strike GEX ({latest_d[0]}):**")
            st.dataframe(
                pd.DataFrame(sk, columns=[
                    'Strike','Call IV','Put IV','Call OI','Put OI','Net GEX','Net VANNA']),
                use_container_width=True, hide_index=True)

# ── EXPORT ─────────────────────────────────────────────────────────────────────
with t5:
    st.markdown("### Export CSVs")
    st.markdown("""
    | File | Contents | Use for |
    |------|----------|---------|
    | `GEX_DAILY.csv` | Daily aggregate GEX | **Regression analysis** |
    | `GEX_PER_STRIKE.csv` | Strike-level GEX | Cross-sectional studies |
    | `NIFTY_OHLCV.csv` | NIFTY OHLCV | Dependent variables |
    | `OPTIONS_RAW.csv` | Raw LTP + OI | Data audit |
    """)

    if st.button("📤 Export All CSVs", type="primary"):
        run(lambda c: c.export(prog))
        st.session_state['done'] = "Export complete!"

    try:
        csvs = sorted(gc.EXPORT_DIR.glob('*.csv')) if gc.EXPORT_DIR.exists() else []
    except Exception:
        csvs = []

    if csvs:
        for f in csvs:
            try:
                sz  = f.stat().st_size / 1024
                ca, cb, cc = st.columns([4,1,2])
                ca.markdown(f"`{f.name}`")
                cb.caption(f"{sz:.0f} KB")
                with open(f,'rb') as fp:
                    cc.download_button(
                        "⬇ Download", fp.read(),
                        file_name=f.name, mime='text/csv',
                        key=f"dl_{f.name}")
            except Exception:
                pass
    else:
        st.info("No files yet. Run pipeline then click Export.")

st.divider()
ca, cb, cc = st.columns(3)
ca.caption("NYZTrade Analytics | Dr. Niyas N")
cb.caption("Dhan API | NIFTY ATM ±15 | BS Greeks")
cc.caption(f"DB: {gc.DB_PATH}")
