"""
NYZTrade Historical GEX Research Processor
==========================================
Reuses the EXACT same analytics pipeline as the NYZTrade dashboard.

DATA COLLECTED:
  1. NSE Bhavcopy options OI + LTP per strike (EOD) — 2019-present, FREE
  2. NSE Index OHLCV (open/high/low/close/volume) — FREE

SAFETY FEATURES:
  - Every download cached to disk BEFORE DB insert (resume safely after crash)
  - WAL journal mode: DB never corrupts on power-off / Ctrl-C
  - Checkpoint JSON: tracks exact last processed item
  - All DB writes use INSERT OR IGNORE / INSERT OR REPLACE (idempotent)
  - Progress logged to nyztrade_gex_research.log (persistent)

OUTPUTS (research_export/):
  MASTER_DATASET.csv        <- GEX + OHLCV merged, one row/day/symbol
  GEX_MAIN_DATASET.csv      <- GEX analytics only
  OHLCV_ALL.csv             <- All index OHLCV
  OHLCV_{SYMBOL}.csv        <- Per-symbol OHLCV
  GEX_PER_STRIKE.csv        <- Strike-level cross-sectional data
  GEX_{SYMBOL}.csv          <- Per-symbol GEX summary

USAGE:
  python nyztrade_historical_gex.py --download   # Step 1: NSE Bhavcopy options
  python nyztrade_historical_gex.py --ohlcv      # Step 2: Index OHLCV
  python nyztrade_historical_gex.py --compute    # Step 3: GEX analytics
  python nyztrade_historical_gex.py --returns    # Step 4: Return variables
  python nyztrade_historical_gex.py --export     # Step 5: Export CSVs
  python nyztrade_historical_gex.py --summary    # Check progress
  python nyztrade_historical_gex.py --all        # Run all steps
"""

import os, sys, io, time, sqlite3, zipfile, json, logging, argparse
import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('nyztrade_gex_research.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('NYZTrade')

# ── Paths ──────────────────────────────────────────────────────────────────────
DB_PATH         = Path('nyztrade_research.db')
RAW_DIR         = Path('bhavcopy_cache')
OHLCV_DIR       = Path('ohlcv_cache')
EXPORT_DIR      = Path('research_export')
CHECKPOINT_FILE = Path('nyztrade_checkpoint.json')

for d in [RAW_DIR, OHLCV_DIR, EXPORT_DIR]:
    d.mkdir(exist_ok=True)

# ── Symbols ────────────────────────────────────────────────────────────────────
SYMBOLS = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']

INDEX_CONFIG = {
    'NIFTY':      {'lot_size': 50,  'unit_divisor': 1e9, 'unit_label': 'B',
                   'total_cascade_pot': 500,  'nse_index': 'NIFTY 50'},
    'BANKNIFTY':  {'lot_size': 15,  'unit_divisor': 1e9, 'unit_label': 'B',
                   'total_cascade_pot': 1000, 'nse_index': 'NIFTY BANK'},
    'FINNIFTY':   {'lot_size': 40,  'unit_divisor': 1e9, 'unit_label': 'B',
                   'total_cascade_pot': 200,  'nse_index': 'NIFTY FIN SERVICE'},
    'MIDCPNIFTY': {'lot_size': 75,  'unit_divisor': 1e9, 'unit_label': 'B',
                   'total_cascade_pot': 300,  'nse_index': 'NIFTY MIDCAP SELECT'},
}

RISK_FREE_RATE = 0.065

# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def save_checkpoint(key: str, value):
    cp = {}
    if CHECKPOINT_FILE.exists():
        try: cp = json.loads(CHECKPOINT_FILE.read_text())
        except Exception: cp = {}
    cp[key] = value
    CHECKPOINT_FILE.write_text(json.dumps(cp, indent=2))

def load_checkpoint(key: str, default=None):
    if not CHECKPOINT_FILE.exists(): return default
    try: return json.loads(CHECKPOINT_FILE.read_text()).get(key, default)
    except Exception: return default


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute('PRAGMA cache_size=30000')
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS bhavcopy_raw (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_date TEXT NOT NULL, symbol TEXT NOT NULL,
        expiry_date TEXT NOT NULL, option_type TEXT NOT NULL,
        strike_price REAL NOT NULL, open_interest REAL DEFAULT 0,
        oi_change REAL DEFAULT 0, ltp REAL DEFAULT 0,
        settle_price REAL DEFAULT 0, contracts REAL DEFAULT 0,
        underlying_value REAL DEFAULT 0,
        UNIQUE(trade_date,symbol,expiry_date,option_type,strike_price)
    );
    CREATE INDEX IF NOT EXISTS idx_raw_date ON bhavcopy_raw(trade_date);
    CREATE INDEX IF NOT EXISTS idx_raw_sym  ON bhavcopy_raw(symbol,trade_date);

    CREATE TABLE IF NOT EXISTS index_ohlcv (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_date TEXT NOT NULL, symbol TEXT NOT NULL,
        open REAL DEFAULT 0, high REAL DEFAULT 0,
        low  REAL DEFAULT 0, close REAL DEFAULT 0,
        volume REAL DEFAULT 0, change_pct REAL DEFAULT 0,
        source TEXT DEFAULT 'NSE',
        UNIQUE(trade_date,symbol)
    );
    CREATE INDEX IF NOT EXISTS idx_ohlcv_sym ON index_ohlcv(symbol,trade_date);

    CREATE TABLE IF NOT EXISTS gex_per_strike (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_date TEXT NOT NULL, symbol TEXT NOT NULL,
        expiry_date TEXT NOT NULL, strike_price REAL NOT NULL,
        tte_days REAL, spot_price REAL,
        call_oi REAL DEFAULT 0, put_oi REAL DEFAULT 0,
        oi_total REAL DEFAULT 0, pcr_strike REAL DEFAULT 0,
        call_iv REAL DEFAULT 0, put_iv REAL DEFAULT 0, iv_avg REAL DEFAULT 0,
        call_gamma REAL DEFAULT 0, put_gamma REAL DEFAULT 0,
        call_vanna REAL DEFAULT 0, put_vanna REAL DEFAULT 0,
        call_delta REAL DEFAULT 0, put_delta REAL DEFAULT 0,
        net_gex REAL DEFAULT 0, net_vanna REAL DEFAULT 0,
        net_dex REAL DEFAULT 0, enhanced_oi_gex REAL DEFAULT 0,
        UNIQUE(trade_date,symbol,expiry_date,strike_price)
    );
    CREATE INDEX IF NOT EXISTS idx_ps_date ON gex_per_strike(trade_date,symbol);

    CREATE TABLE IF NOT EXISTS gex_daily_summary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_date TEXT NOT NULL, symbol TEXT NOT NULL, spot_price REAL,
        net_gex_total REAL, net_gex_positive REAL, net_gex_negative REAL,
        gex_ratio REAL, gex_regime TEXT,
        net_vanna_total REAL, net_vanna_positive REAL, net_vanna_negative REAL,
        vanna_regime TEXT, net_dex_total REAL, dex_regime TEXT,
        gex_flip_zone_1 REAL, gex_flip_zone_2 REAL, n_flip_zones INTEGER,
        dominant_call_wall REAL, dominant_put_wall REAL, max_pain_strike REAL,
        total_call_oi REAL, total_put_oi REAL, pcr REAL, total_oi REAL,
        atm_call_iv REAL, atm_put_iv REAL, iv_skew REAL, iv_regime TEXT,
        avg_call_iv REAL, avg_put_iv REAL,
        bear_cascade_fuel REAL, bear_cascade_absorption REAL, bear_cascade_net REAL,
        bull_cascade_fuel REAL, bull_cascade_absorption REAL, bull_cascade_net REAL,
        cascade_bias TEXT, n_vanna_zones INTEGER, n_vacuum_zones INTEGER,
        n_resistance_ceilings INTEGER, n_trap_doors INTEGER, n_support_floors INTEGER,
        nearest_vacuum_above REAL, nearest_support_below REAL, nearest_trap_below REAL,
        n_strikes INTEGER, n_expiries INTEGER,
        index_return_1d REAL, index_return_3d REAL, index_return_5d REAL,
        index_intraday_range REAL, realized_vol_5d REAL, realized_vol_21d REAL,
        created_at TEXT DEFAULT (datetime('now')),
        UNIQUE(trade_date,symbol)
    );
    CREATE INDEX IF NOT EXISTS idx_sum_date ON gex_daily_summary(trade_date,symbol);

    CREATE TABLE IF NOT EXISTS download_log (
        trade_date TEXT PRIMARY KEY, status TEXT,
        rows_stored INTEGER DEFAULT 0, error_msg TEXT,
        updated_at TEXT DEFAULT (datetime('now'))
    );
    """)
    conn.commit()
    return conn


# ═══════════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES — same as dashboard
# ═══════════════════════════════════════════════════════════════════════════════
def _d1(S, K, T, r, sig):
    return (np.log(S/np.maximum(K,1e-6))+(r+0.5*sig**2)*T)/np.maximum(sig*np.sqrt(max(T,1/365)),1e-10)

def bs_gamma_v(S,K,T,r,s): return norm.pdf(_d1(S,K,T,r,s))/np.maximum(S*s*np.sqrt(max(T,1/365)),1e-10)
def bs_vanna_v(S,K,T,r,s):
    d1=_d1(S,K,T,r,s); return -norm.pdf(d1)*(d1-s*np.sqrt(max(T,1/365)))/np.maximum(s,1e-10)
def bs_dc(S,K,T,r,s): return norm.cdf(_d1(S,K,T,r,s))
def bs_dp(S,K,T,r,s): return norm.cdf(_d1(S,K,T,r,s))-1.0

def bs_price(S,K,T,r,sig,ot='CE'):
    if T<=0 or sig<=0: return max(S-K,0) if ot=='CE' else max(K-S,0)
    d1=_d1(S,np.array([K]),T,r,np.array([sig]))[0]; d2=d1-sig*np.sqrt(T)
    return float(S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)) if ot=='CE' \
           else float(K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1))

def solve_iv(price,S,K,T,r,ot='CE',tol=1e-5,n=80) -> float:
    if price<=0 or T<=0: return 0.0
    if price < (max(S-K,0) if ot=='CE' else max(K-S,0)): return 0.0
    lo,hi = 0.001,5.0
    for _ in range(n):
        mid=( lo+hi)/2; p=bs_price(S,K,T,r,mid,ot)
        if abs(p-price)<tol: return mid*100
        if p<price: lo=mid
        else:       hi=mid
    return mid*100


# ═══════════════════════════════════════════════════════════════════════════════
# GEX ANALYTICS — same as dashboard
# ═══════════════════════════════════════════════════════════════════════════════
def gamma_flip_zones(df,spot):
    df_s=df.sort_values('strike_price').reset_index(drop=True); out=[]
    for i in range(len(df_s)-1):
        g1,g2=float(df_s.loc[i,'net_gex']),float(df_s.loc[i+1,'net_gex'])
        if g1*g2<0:
            k1,k2=float(df_s.loc[i,'strike_price']),float(df_s.loc[i+1,'strike_price'])
            out.append({'strike':k1+(k2-k1)*abs(g1)/(abs(g1)+abs(g2)),'below_spot':(k1+(k2-k1)*abs(g1)/(abs(g1)+abs(g2)))<spot})
    return out

def vanna_flip_zones(df,spot):
    df_s=df.sort_values('strike_price').reset_index(drop=True); out=[]
    rm={(True,True):('VACUUM_ZONE','#10b981'),(True,False):('RESISTANCE_CEILING','#ef4444'),
        (False,True):('SUPPORT_FLOOR','#06b6d4'),(False,False):('TRAP_DOOR','#f59e0b')}
    for i in range(len(df_s)-1):
        v1,v2=float(df_s.loc[i,'net_vanna']),float(df_s.loc[i+1,'net_vanna'])
        if v1*v2<0:
            k1,k2=float(df_s.loc[i,'strike_price']),float(df_s.loc[i+1,'strike_price'])
            k=k1+(k2-k1)*abs(v1)/(abs(v1)+abs(v2))
            role,color=rm.get((k>spot,v1>0),('NEUTRAL','#94a3b8'))
            out.append({'strike':k,'role':role,'color':color,'above':k>spot,'pos2neg':v1>0})
    return out

def iv_trend(df):
    ac=df['call_iv'].replace(0,np.nan).mean() or 25; ap=df['put_iv'].replace(0,np.nan).mean() or 25
    sk=ap-ac; rg='EXPANDING' if sk>5 else('COMPRESSING' if sk<-2 else 'FLAT')
    return {'regime':rg,'skew':round(sk,2)}

def gex_cascade(df,spot,cfg,vzones,iv_regime):
    if df.empty: return {}
    pot=cfg['total_cascade_pot']
    VA={'SUPPORT_FLOOR':{'COMPRESSING':-0.60,'FLAT':-0.35,'EXPANDING':0.20},
        'TRAP_DOOR':{'COMPRESSING':0,'FLAT':0,'EXPANDING':0.30},
        'VACUUM_ZONE':{'COMPRESSING':0,'FLAT':0,'EXPANDING':-0.50},
        'RESISTANCE_CEILING':{'COMPRESSING':0,'FLAT':0,'EXPANDING':0.20}}
    df_s=df.sort_values('strike_price').reset_index(drop=True)
    bear=df_s[df_s['strike_price']<=spot].sort_values('strike_price',ascending=False)
    bull=df_s[df_s['strike_price']>spot].sort_values('strike_price',ascending=True)
    ivl=df_s['strike_price'].diff().abs().mode(); ivl=float(ivl.iloc[0]) if len(ivl)>0 else 50.0
    res={}
    for direction,sub in [('BEAR',bear),('BULL',bull)]:
        if sub.empty: res[direction]={'fuel':0,'absorption':0,'net':0}; continue
        gv=sub['net_gex'].fillna(0).to_numpy(float); tot=np.abs(gv).sum()
        if tot==0: res[direction]={'fuel':0,'absorption':0,'net':0}; continue
        fuel=0.0; ab=0.0
        for _,row in sub.iterrows():
            g=float(row['net_gex']); w=abs(g)/tot; rp=w*pot/2
            adj=0.0; md=float('inf'); cl=None
            for z in vzones:
                dd=abs(z['strike']-float(row['strike_price']))
                if dd<md: md=dd; cl=z
            if cl and md<ivl: adj=VA.get(cl['role'],{}).get(iv_regime,0)*rp
            ap_=max(0.0,rp+adj)
            if direction=='BEAR':
                if g<0: fuel+=ap_
                else: ab+=ap_
            else:
                if g<0: fuel+=ap_
                else: ab+=ap_
        res[direction]={'fuel':round(fuel,2),'absorption':round(ab,2),'net':round(max(0,fuel-ab*0.5),2)}
    return res

def enhanced_oi_gex(df,spot):
    dist=(df['strike_price']-spot).abs(); dw=1-(dist/max(dist.max(),1))*0.5
    ai=(df['call_iv'].fillna(25)+df['put_iv'].fillna(25))/2
    ia=(ai/max(ai.mean(),1)).clip(0.5,2.0)
    oc=df['call_oi'].fillna(0)*0.05; op_=df['put_oi'].fillna(0)*0.05
    raw=(oc*df['call_gamma'].abs().fillna(0)-op_*df['put_gamma'].abs().fillna(0))*ia*dw
    sc=df['net_gex'].abs().mean()/raw.abs().mean() if raw.abs().mean()>0 and df['net_gex'].abs().mean()>0 else 1.0
    return raw*sc


# ═══════════════════════════════════════════════════════════════════════════════
# BHAVCOPY DOWNLOADER
# ═══════════════════════════════════════════════════════════════════════════════
class BhavCopyDownloader:
    HDR={'User-Agent':'Mozilla/5.0','Accept-Encoding':'gzip, deflate','Connection':'keep-alive'}

    def __init__(self,conn):
        self.conn=conn; self.ses=requests.Session(); self.ses.headers.update(self.HDR)
        try: self.ses.get('https://www.nseindia.com',timeout=10); time.sleep(1)
        except Exception: pass

    def _urls(self,d):
        dn=d.strftime('%Y%m%d'); do=d.strftime('%d%b%Y').upper()
        return [f'https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{dn}_F_0000.csv.zip',
                f'https://nsearchives.nseindia.com/content/fo/fo{do}bhav.csv.zip',
                f'https://archives.nseindia.com/content/fo/fo{do}bhav.csv.zip']

    def _fetch(self,d):
        cache=RAW_DIR/f"{d.strftime('%Y%m%d')}.csv"
        if cache.exists():
            try: return pd.read_csv(cache)
            except Exception: pass
        for url in self._urls(d):
            try:
                r=self.ses.get(url,timeout=30)
                if r.status_code==200 and len(r.content)>500:
                    zf=zipfile.ZipFile(io.BytesIO(r.content))
                    df=pd.read_csv(zf.open(zf.namelist()[0]))
                    df.to_csv(cache,index=False); return df
            except Exception as e: log.debug(f"  {url}: {e}")
        return None

    def _norm(self,df):
        df.columns=[c.strip().upper().replace(' ','_') for c in df.columns]
        AL={'SYM':['SYMBOL','TRADINGSYMBOL'],'EXP':['EXPIRY_DT','EXPIRYDATE','EXPIRY_DATE'],
            'OPT':['OPTION_TYP','OPTIONTYPE','OPTION_TYPE'],'STK':['STRIKE_PR','STRIKEPRICE','STRIKE_PRICE'],
            'OI':['OPEN_INT','OPENINTEREST','OPEN_INTEREST','OI'],'OIC':['CHG_IN_OI','CHANGE_OI','OI_CHANGE'],
            'LTP':['LAST','LTP','CLOSE'],'SET':['SETTLE_PR','SETTLEMENT_PRICE'],
            'CTR':['CONTRACTS','NO_OF_CONTRACTS'],'UND':['UNDERLYING_VALUE','UNDERLYING']}
        def fc(ks):
            for k in ks:
                if k in df.columns: return k
            return None
        c={k:fc(v) for k,v in AL.items()}
        if not all(c[k] for k in ['SYM','EXP','OPT','STK','OI']): return None
        def sa(col,d=0):
            return pd.to_numeric(df[col],errors='coerce').fillna(d) if col and col in df.columns else pd.Series([d]*len(df))
        return pd.DataFrame({'symbol':df[c['SYM']].astype(str).str.strip(),
            'expiry':df[c['EXP']].astype(str).str.strip(),'opttype':df[c['OPT']].astype(str).str.strip(),
            'strike':pd.to_numeric(df[c['STK']],errors='coerce'),'oi':sa(c['OI']),
            'oi_chg':sa(c['OIC']),'ltp':sa(c['LTP']),'settle':sa(c['SET']),
            'contracts':sa(c['CTR']),'underlying':sa(c['UND'])})

    def download_range(self,start,end,progress_cb=None):
        cur=start; total=(end-start).days+1; done=0
        while cur<=end:
            done+=1
            if cur.weekday()>=5: cur+=timedelta(days=1); continue
            ds=cur.strftime('%Y-%m-%d')
            row=self.conn.execute("SELECT status FROM download_log WHERE trade_date=?",(ds,)).fetchone()
            if row and row[0] in ('ok','holiday'):
                if progress_cb: progress_cb(done/total,f"Skip {ds}")
                cur+=timedelta(days=1); continue
            if progress_cb: progress_cb(done/total,f"Downloading {ds}...")
            log.info(f"Bhavcopy {ds}...")
            raw=self._fetch(cur)
            if raw is None:
                self.conn.execute("INSERT OR REPLACE INTO download_log(trade_date,status) VALUES(?,?)",(ds,'holiday'))
                self.conn.commit(); cur+=timedelta(days=1); time.sleep(0.5); continue
            df=self._norm(raw)
            if df is None:
                self.conn.execute("INSERT OR REPLACE INTO download_log(trade_date,status,error_msg) VALUES(?,?,?)",(ds,'error','col fail'))
                self.conn.commit(); cur+=timedelta(days=1); continue
            mask=df['symbol'].isin(SYMBOLS)&df['opttype'].isin(['CE','PE'])
            df=df[mask].dropna(subset=['strike'])
            if df.empty:
                self.conn.execute("INSERT OR REPLACE INTO download_log(trade_date,status) VALUES(?,?)",(ds,'holiday'))
                self.conn.commit(); cur+=timedelta(days=1); continue
            rows=[(ds,r.symbol,r.expiry,r.opttype,float(r.strike),float(r.oi),float(r.oi_chg),
                   float(r.ltp),float(r.settle),float(r.contracts),float(r.underlying))
                  for r in df.itertuples()]
            self.conn.executemany("""INSERT OR IGNORE INTO bhavcopy_raw
                (trade_date,symbol,expiry_date,option_type,strike_price,open_interest,oi_change,
                 ltp,settle_price,contracts,underlying_value) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",rows)
            self.conn.execute("INSERT OR REPLACE INTO download_log(trade_date,status,rows_stored) VALUES(?,?,?)",(ds,'ok',len(rows)))
            self.conn.commit(); save_checkpoint('last_bhavcopy',ds)
            log.info(f"  {ds}: {len(rows)} rows"); time.sleep(1.2); cur+=timedelta(days=1)


# ═══════════════════════════════════════════════════════════════════════════════
# OHLCV DOWNLOADER
# ═══════════════════════════════════════════════════════════════════════════════
class OHLCVDownloader:
    HDR={'User-Agent':'Mozilla/5.0','Accept':'application/json,*/*',
         'Referer':'https://www.nseindia.com/'}

    def __init__(self,conn):
        self.conn=conn; self.ses=requests.Session(); self.ses.headers.update(self.HDR)
        try:
            self.ses.get('https://www.nseindia.com',timeout=15); time.sleep(1.5)
            self.ses.get('https://www.nseindia.com/market-data/live-equity-market',timeout=10); time.sleep(1)
        except Exception: pass

    def _fetch_chunk(self,symbol,start,end):
        idx=INDEX_CONFIG.get(symbol,{}).get('nse_index',symbol)
        cache=OHLCV_DIR/f"{symbol}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.json"
        if cache.exists():
            try: return json.loads(cache.read_text())
            except Exception: pass
        url="https://www.nseindia.com/api/historicalindices"
        params={'indexType':idx,'from':start.strftime('%d-%m-%Y'),'to':end.strftime('%d-%m-%Y')}
        try:
            r=self.ses.get(url,params=params,timeout=30)
            if r.status_code==200:
                data=r.json().get('data',[])
                if data: cache.write_text(json.dumps(data)); return data
            log.debug(f"  OHLCV {symbol} {start}-{end}: {r.status_code}")
        except Exception as e: log.debug(f"  {e}")
        return []

    def _parse(self,data,symbol):
        rows=[]; cn={'date':['HistoricalDate','date','INDEX_DATE'],
            'o':['OPEN','open'],'h':['HIGH','high'],'l':['LOW','low'],
            'c':['CLOSING','CLOSE','close','Closing Index Value'],'v':['TRADED_VOLUME','volume']}
        for item in data:
            try:
                def gv(keys):
                    for k in keys:
                        if k in item: return item[k]
                    return None
                raw_dt=str(gv(cn['date']) or '').strip()
                td=None
                for fmt in ['%d-%b-%Y','%Y-%m-%d','%d/%m/%Y']:
                    try: td=datetime.strptime(raw_dt,fmt).strftime('%Y-%m-%d'); break
                    except: pass
                if not td: continue
                o=float(str(gv(cn['o']) or '0').replace(',',''))
                h=float(str(gv(cn['h']) or '0').replace(',',''))
                l=float(str(gv(cn['l']) or '0').replace(',',''))
                c_=float(str(gv(cn['c']) or '0').replace(',',''))
                v=float(str(gv(cn['v']) or '0').replace(',',''))
                chg=((c_-o)/o*100) if o>0 else 0
                rows.append((td,symbol,o,h,l,c_,v,round(chg,4)))
            except Exception: continue
        return rows

    def download_range(self,start,end,progress_cb=None):
        for si,sym in enumerate(SYMBOLS):
            if progress_cb: progress_cb(si/len(SYMBOLS),f"OHLCV: {sym}")
            log.info(f"OHLCV {sym}...")
            all_rows=[]; cs=start
            while cs<=end:
                ce=min(cs+timedelta(days=180),end)
                data=self._fetch_chunk(sym,cs,ce)
                if data:
                    r=self._parse(data,sym); all_rows.extend(r)
                    log.info(f"  {sym} {cs}→{ce}: {len(r)} rows")
                else:
                    log.warning(f"  {sym} {cs}→{ce}: no data")
                cs=ce+timedelta(days=1); time.sleep(2)
            if all_rows:
                self.conn.executemany("""INSERT OR REPLACE INTO index_ohlcv
                    (trade_date,symbol,open,high,low,close,volume,change_pct) VALUES(?,?,?,?,?,?,?,?)""",all_rows)
                self.conn.commit(); save_checkpoint(f'ohlcv_{sym}',str(end))
                log.info(f"  {sym}: {len(all_rows)} total rows")
        if progress_cb: progress_cb(1.0,"OHLCV complete")


# ═══════════════════════════════════════════════════════════════════════════════
# GEX COMPUTE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class GEXEngine:

    def __init__(self,conn): self.conn=conn

    def _tte(self,td,exp):
        for fmt in ['%d-%b-%Y','%Y-%m-%d','%d/%m/%Y','%d-%b-%y']:
            try: return max((datetime.strptime(exp.strip(),fmt)-datetime.strptime(td,'%Y-%m-%d')).days/365.0,1/365)
            except: pass
        return 7/365

    def _pending(self):
        return self.conn.execute("""
            SELECT DISTINCT b.trade_date,b.symbol FROM bhavcopy_raw b
            LEFT JOIN gex_daily_summary g ON b.trade_date=g.trade_date AND b.symbol=g.symbol
            WHERE g.trade_date IS NULL AND b.underlying_value>0 ORDER BY b.trade_date,b.symbol
        """).fetchall()

    def compute_all(self,progress_cb=None):
        pending=self._pending(); total=len(pending)
        log.info(f"Computing GEX for {total} pairs...")
        for i,(ds,sym) in enumerate(pending,1):
            if progress_cb: progress_cb(i/total,f"GEX: {ds} {sym}")
            try: self._day(ds,sym)
            except Exception as e: log.error(f"  {ds} {sym}: {e}",exc_info=True)
            if i%10==0: save_checkpoint('last_gex',f"{ds}_{sym}")
        log.info("GEX done.")

    def _day(self,ds,sym):
        cfg=INDEX_CONFIG.get(sym,INDEX_CONFIG['NIFTY'])
        div=cfg['unit_divisor']; lot=cfg['lot_size']
        rows=self.conn.execute("""SELECT expiry_date,option_type,strike_price,
            open_interest,oi_change,ltp,settle_price,underlying_value
            FROM bhavcopy_raw WHERE trade_date=? AND symbol=?""",(ds,sym)).fetchall()
        if not rows: return
        df=pd.DataFrame(rows,columns=['expiry','opt','strike','oi','oic','ltp','set','und'])
        df['price']=df.apply(lambda r:r['set'] if r['ltp']==0 else r['ltp'],axis=1)
        spot=float(df['und'].replace(0,np.nan).dropna().iloc[0]) if df['und'].replace(0,np.nan).dropna().any() else 0
        if spot<=0: return
        ar=[]
        for exp in df['expiry'].unique():
            tte=self._tte(ds,exp); de=df[df['expiry']==exp]
            calls=de[de['opt']=='CE'].set_index('strike'); puts=de[de['opt']=='PE'].set_index('strike')
            ks=sorted(set(calls.index)|set(puts.index));
            if not ks: continue
            K=np.array(ks,float)
            civ=np.zeros(len(K)); piv=np.zeros(len(K))
            for j,k in enumerate(K):
                if k in calls.index: civ[j]=solve_iv(float(calls.loc[k,'price']),spot,k,tte,RISK_FREE_RATE,'CE')
                if k in puts.index:  piv[j]=solve_iv(float(puts.loc[k,'price']),spot,k,tte,RISK_FREE_RATE,'PE')
            ai=int(np.argmin(np.abs(K-spot))); ac=civ[ai] or 25; ap=piv[ai] or 25
            civ[civ==0]=ac; piv[piv==0]=ap
            cf=np.clip(civ,1,500)/100; pf=np.clip(piv,1,500)/100
            cg=bs_gamma_v(spot,K,tte,RISK_FREE_RATE,cf); pg=bs_gamma_v(spot,K,tte,RISK_FREE_RATE,pf)
            cv=bs_vanna_v(spot,K,tte,RISK_FREE_RATE,cf); pv_=bs_vanna_v(spot,K,tte,RISK_FREE_RATE,pf)
            cd=bs_dc(spot,K,tte,RISK_FREE_RATE,cf); pd_=bs_dp(spot,K,tte,RISK_FREE_RATE,pf)
            for j,k in enumerate(K):
                co=float(calls.loc[k,'oi'])*lot if k in calls.index else 0
                po=float(puts.loc[k,'oi'])*lot  if k in puts.index  else 0
                ar.append({'expiry':exp,'strike_price':k,'tte_days':round(tte*365,1),
                    'call_oi':co,'put_oi':po,'call_iv':round(civ[j],2),'put_iv':round(piv[j],2),
                    'call_gamma':cg[j],'put_gamma':pg[j],'call_vanna':cv[j],'put_vanna':pv_[j],
                    'call_delta':cd[j],'put_delta':pd_[j],
                    'net_gex':(co*cg[j]-po*pg[j])*spot**2/div,
                    'net_vanna':(co*cv[j]-po*pv_[j])/div,
                    'net_dex':(co*cd[j]+po*pd_[j])/div,
                    'oi_total':co+po,'pcr_strike':po/max(co,1)})
        if not ar: return
        df2=pd.DataFrame(ar); df2['enhanced_oi_gex']=enhanced_oi_gex(df2,spot)
        for _,r in df2.iterrows():
            self.conn.execute("""INSERT OR IGNORE INTO gex_per_strike
                (trade_date,symbol,expiry_date,strike_price,tte_days,spot_price,
                 call_oi,put_oi,oi_total,pcr_strike,call_iv,put_iv,iv_avg,
                 call_gamma,put_gamma,call_vanna,put_vanna,call_delta,put_delta,
                 net_gex,net_vanna,net_dex,enhanced_oi_gex)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ds,sym,r['expiry'],r['strike_price'],r['tte_days'],spot,
                 r['call_oi'],r['put_oi'],r['oi_total'],r['pcr_strike'],
                 r['call_iv'],r['put_iv'],(r['call_iv']+r['put_iv'])/2,
                 r['call_gamma'],r['put_gamma'],r['call_vanna'],r['put_vanna'],
                 r['call_delta'],r['put_delta'],r['net_gex'],r['net_vanna'],r['net_dex'],r['enhanced_oi_gex']))
        self.conn.commit()
        fl=gamma_flip_zones(df2,spot); vz=vanna_flip_zones(df2,spot)
        ivi=iv_trend(df2); cas=gex_cascade(df2,spot,cfg,vz,ivi['regime'])
        tg=df2['net_gex'].sum(); pg=df2[df2['net_gex']>0]['net_gex'].sum()
        ng=df2[df2['net_gex']<0]['net_gex'].sum()
        tv=df2['net_vanna'].sum(); td_=df2['net_dex'].sum()
        tc=df2['call_oi'].sum(); tp=df2['put_oi'].sum(); pcr=tp/max(tc,1)
        mp=float(df2.groupby('strike_price')['oi_total'].sum().idxmax()) if not df2.empty else 0
        pm=df2['net_gex']>0; nm=df2['net_gex']<0
        dc=float(df2[pm].loc[df2[pm]['net_gex'].idxmax(),'strike_price']) if pm.any() else 0
        dp=float(df2[nm].loc[df2[nm]['net_gex'].idxmin(),'strike_price']) if nm.any() else 0
        fs=sorted(fl,key=lambda z:abs(z['strike']-spot))
        f1=fs[0]['strike'] if fs else 0; f2=fs[1]['strike'] if len(fs)>1 else 0
        ar_=df2.iloc[(df2['strike_price']-spot).abs().argsort().iloc[0]]
        vr=[z['role'] for z in vz]
        ab=[z for z in vz if z['above']]; bl=[z for z in vz if not z['above']]
        def mnz(lst,role): return min([z['strike'] for z in lst if z['role']==role],key=lambda k:abs(k-spot),default=0)
        nva=mnz(ab,'VACUUM_ZONE'); nsu=mnz(bl,'SUPPORT_FLOOR'); ntr=mnz(bl,'TRAP_DOOR')
        bear=cas.get('BEAR',{}); bull=cas.get('BULL',{})
        gb='POSITIVE' if tg>0 else('NEGATIVE' if tg<0 else 'NEUTRAL')
        vb='BULLISH_IV' if tv>0 else('BEARISH_IV' if tv<0 else 'NEUTRAL')
        db='LONG_DELTA' if td_>0 else('SHORT_DELTA' if td_<0 else 'NEUTRAL')
        cb='BEAR' if bear.get('net',0)>bull.get('net',0) else('BULL' if bull.get('net',0)>bear.get('net',0) else 'NEUTRAL')
        self.conn.execute("""INSERT OR REPLACE INTO gex_daily_summary
            (trade_date,symbol,spot_price,net_gex_total,net_gex_positive,net_gex_negative,
             gex_ratio,gex_regime,net_vanna_total,net_vanna_positive,net_vanna_negative,vanna_regime,
             net_dex_total,dex_regime,gex_flip_zone_1,gex_flip_zone_2,n_flip_zones,
             dominant_call_wall,dominant_put_wall,max_pain_strike,total_call_oi,total_put_oi,
             pcr,total_oi,atm_call_iv,atm_put_iv,iv_skew,iv_regime,avg_call_iv,avg_put_iv,
             bear_cascade_fuel,bear_cascade_absorption,bear_cascade_net,bull_cascade_fuel,
             bull_cascade_absorption,bull_cascade_net,cascade_bias,n_vanna_zones,n_vacuum_zones,
             n_resistance_ceilings,n_trap_doors,n_support_floors,nearest_vacuum_above,
             nearest_support_below,nearest_trap_below,n_strikes,n_expiries)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ds,sym,spot,tg,pg,ng,pg/abs(ng) if ng else 0,gb,
             tv,df2[df2['net_vanna']>0]['net_vanna'].sum(),df2[df2['net_vanna']<0]['net_vanna'].sum(),vb,
             td_,db,f1,f2,len(fl),dc,dp,mp,tc,tp,pcr,tc+tp,
             float(ar_['call_iv']),float(ar_['put_iv']),ivi['skew'],ivi['regime'],
             df2['call_iv'].replace(0,np.nan).mean() or 0,df2['put_iv'].replace(0,np.nan).mean() or 0,
             bear.get('fuel',0),bear.get('absorption',0),bear.get('net',0),
             bull.get('fuel',0),bull.get('absorption',0),bull.get('net',0),cb,
             len(vz),vr.count('VACUUM_ZONE'),vr.count('RESISTANCE_CEILING'),
             vr.count('TRAP_DOOR'),vr.count('SUPPORT_FLOOR'),nva,nsu,ntr,len(df2),df2['expiry'].nunique()))
        self.conn.commit()


# ═══════════════════════════════════════════════════════════════════════════════
# RETURN VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════
def compute_returns(conn,progress_cb=None):
    log.info("Computing return variables...")
    for i,sym in enumerate(SYMBOLS):
        if progress_cb: progress_cb(i/len(SYMBOLS),f"Returns: {sym}")
        # Prefer OHLCV (has real H/L for intraday range)
        orows=conn.execute("SELECT trade_date,open,high,low,close FROM index_ohlcv WHERE symbol=? ORDER BY trade_date",(sym,)).fetchall()
        if orows:
            df=pd.DataFrame(orows,columns=['trade_date','open','high','low','close'])
        else:
            brows=conn.execute("SELECT trade_date,AVG(underlying_value) as close FROM bhavcopy_raw WHERE symbol=? AND underlying_value>0 GROUP BY trade_date ORDER BY trade_date",(sym,)).fetchall()
            if not brows: continue
            df=pd.DataFrame(brows,columns=['trade_date','close'])
            df['open']=df['close']; df['high']=df['close']; df['low']=df['close']
        df=df.sort_values('trade_date').reset_index(drop=True)
        df['lr']=np.log(df['close']/df['close'].shift(1))
        df['r1']=df['close'].pct_change(1).shift(-1)*100
        df['r3']=df['close'].pct_change(3).shift(-3)*100
        df['r5']=df['close'].pct_change(5).shift(-5)*100
        df['rv5']=df['lr'].rolling(5).std()*np.sqrt(252)*100
        df['rv21']=df['lr'].rolling(21).std()*np.sqrt(252)*100
        df['idr']=(df['high']-df['low'])/df['low'].replace(0,np.nan)*100
        def s(v): return float(v) if pd.notna(v) and not np.isinf(v) else None
        for _,row in df.iterrows():
            conn.execute("""UPDATE gex_daily_summary SET
                index_return_1d=?,index_return_3d=?,index_return_5d=?,
                index_intraday_range=?,realized_vol_5d=?,realized_vol_21d=?
                WHERE trade_date=? AND symbol=?""",
                (s(row['r1']),s(row['r3']),s(row['r5']),s(row['idr']),s(row['rv5']),s(row['rv21']),row['trade_date'],sym))
        conn.commit(); log.info(f"  {sym}: {len(df)} days")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
def export_all(conn,progress_cb=None):
    log.info(f"Exporting to {EXPORT_DIR}/")
    steps=6; done=0
    def tick(msg):
        nonlocal done; done+=1
        if progress_cb: progress_cb(done/steps,msg)
        log.info(f"  {msg}")

    gex=pd.read_sql("SELECT * FROM gex_daily_summary ORDER BY symbol,trade_date",conn)
    gex.to_csv(EXPORT_DIR/'GEX_MAIN_DATASET.csv',index=False)
    tick(f"GEX_MAIN_DATASET.csv — {len(gex):,} rows")

    ohlcv=pd.read_sql("SELECT * FROM index_ohlcv ORDER BY symbol,trade_date",conn)
    ohlcv.to_csv(EXPORT_DIR/'OHLCV_ALL.csv',index=False)
    tick(f"OHLCV_ALL.csv — {len(ohlcv):,} rows")

    for sym in SYMBOLS:
        o=ohlcv[ohlcv['symbol']==sym]
        if not o.empty: o.to_csv(EXPORT_DIR/f'OHLCV_{sym}.csv',index=False)
    tick("Per-symbol OHLCV CSVs")

    for sym in SYMBOLS:
        g=gex[gex['symbol']==sym]
        if not g.empty: g.to_csv(EXPORT_DIR/f'GEX_{sym}.csv',index=False)
    tick("Per-symbol GEX CSVs")

    ps=pd.read_sql("SELECT * FROM gex_per_strike ORDER BY trade_date,symbol,strike_price",conn)
    ps.to_csv(EXPORT_DIR/'GEX_PER_STRIKE.csv',index=False)
    tick(f"GEX_PER_STRIKE.csv — {len(ps):,} rows")

    # MASTER: GEX + OHLCV merged
    master=pd.merge(gex,ohlcv[['trade_date','symbol','open','high','low','close','volume','change_pct']],
                    on=['trade_date','symbol'],how='left')
    master.to_csv(EXPORT_DIR/'MASTER_DATASET.csv',index=False)
    tick(f"MASTER_DATASET.csv — {len(master):,} rows  <- USE FOR REGRESSION")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
def summary(conn):
    print("\n"+"="*65)
    print("NYZTRADE RESEARCH DATABASE")
    print("="*65)
    for t,d in [('bhavcopy_raw','Options OI raw'),('index_ohlcv','Index OHLCV'),
                ('gex_per_strike','GEX per strike'),('gex_daily_summary','Daily GEX summary')]:
        n=conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:<25} {n:>10,}  {d}")
    print()
    for sym in SYMBOLS:
        g=conn.execute("SELECT MIN(trade_date),MAX(trade_date),COUNT(*) FROM gex_daily_summary WHERE symbol=?",(sym,)).fetchone()
        o=conn.execute("SELECT COUNT(*) FROM index_ohlcv WHERE symbol=?",(sym,)).fetchone()
        gstr=f"GEX {g[0]}→{g[1]} ({g[2]} days)" if g and g[0] else "GEX: none"
        print(f"  {sym:<12}: {gstr}  |  OHLCV: {o[0]} days")
    dl=conn.execute("SELECT status,COUNT(*) FROM download_log GROUP BY status").fetchall()
    print(f"\n  Downloads: {dict(dl)}")
    print(f"\n  Export files in {EXPORT_DIR}/:")
    for f in sorted(EXPORT_DIR.glob('*.csv')):
        print(f"    {f.name:<35} {f.stat().st_size/1024**2:.1f} MB")
    print("="*65)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--download',action='store_true')
    ap.add_argument('--ohlcv',action='store_true')
    ap.add_argument('--compute',action='store_true')
    ap.add_argument('--returns',action='store_true')
    ap.add_argument('--export',action='store_true')
    ap.add_argument('--summary',action='store_true')
    ap.add_argument('--all',action='store_true')
    ap.add_argument('--start',default='2019-01-01')
    ap.add_argument('--end',default=str(date.today()))
    args=ap.parse_args()
    conn=init_db(); s=date.fromisoformat(args.start); e=date.fromisoformat(args.end)
    if args.all or args.download: BhavCopyDownloader(conn).download_range(s,e)
    if args.all or args.ohlcv:   OHLCVDownloader(conn).download_range(s,e)
    if args.all or args.compute:  GEXEngine(conn).compute_all()
    if args.all or args.returns:  compute_returns(conn)
    if args.all or args.export:   export_all(conn)
    if args.summary or args.all:  summary(conn)
    conn.close()

if __name__=='__main__': main()
