# -*- coding: utf-8 -*-
"""
app.py — Markowitz Portfolio Optimizer con Login Supabase
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple
import logging, io, json, requests, re as _re
from supabase import create_client, Client

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Markowitz Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Mono', monospace;
    letter-spacing: -0.5px;
}
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}
section[data-testid="stSidebar"] {
    background: #13151c;
    border-right: 1px solid #1e2130;
}
.metric-card {
    background: #13151c;
    border: 1px solid #1e2130;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 12px;
}
.metric-card .label {
    font-size: 11px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'DM Mono', monospace;
    font-size: 22px;
    font-weight: 500;
    color: #e8eaf0;
}
.metric-card .value.positive { color: #34d399; }
.metric-card .value.negative { color: #f87171; }
.stButton > button {
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    letter-spacing: 0.5px;
    transition: background 0.2s;
    width: 100%;
}
.stButton > button:hover { background: #2563eb; }
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: #1a1d27 !important;
    border: 1px solid #2a2d3a !important;
    color: #e8eaf0 !important;
    border-radius: 8px !important;
}
.stDataFrame { border-radius: 10px; overflow: hidden; }
div[data-testid="stExpander"] {
    background: #13151c;
    border: 1px solid #1e2130;
    border-radius: 10px;
}
.section-title {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3b82f6;
    margin: 28px 0 12px 0;
}
</style>
""", unsafe_allow_html=True)




# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE — connessione
# ══════════════════════════════════════════════════════════════════════════════

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

@st.cache_resource
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = get_supabase()

# ══════════════════════════════════════════════════════════════════════════════
# AUTH — login / registrazione / logout
# ══════════════════════════════════════════════════════════════════════════════

if "user" not in st.session_state:
    st.session_state.user = None
if "access_token" not in st.session_state:
    st.session_state.access_token = None

def do_login(email: str, password: str):
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        st.session_state.user         = res.user
        st.session_state.access_token = res.session.access_token
        return True, None
    except Exception as e:
        return False, str(e)

def do_register(email: str, password: str):
    try:
        res = supabase.auth.sign_up({"email": email, "password": password})
        return True, None
    except Exception as e:
        return False, str(e)

def do_logout():
    supabase.auth.sign_out()
    st.session_state.user         = None
    st.session_state.access_token = None
    st.session_state.current_portfolio_id   = None
    st.session_state.portfolio_tickers      = DEFAULT_TICKERS.copy()
    st.session_state.manual_weights         = {}
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# DB — portafogli Supabase
# ══════════════════════════════════════════════════════════════════════════════

def db_client():
    """Restituisce il client Supabase autenticato con il token dell'utente."""
    if st.session_state.access_token:
        supabase.auth.set_session(
            st.session_state.access_token,
            st.session_state.access_token
        )
    return supabase

def load_portfolios() -> List[Dict]:
    """Carica tutti i portafogli dell'utente corrente."""
    try:
        res = db_client().table("portfolios").select("*").order("created_at").execute()
        return res.data or []
    except Exception:
        return []

def load_portfolio_assets(portfolio_id: str) -> Tuple[List[str], Dict[str, float], str]:
    """Carica ticker, pesi e benchmark di un portafoglio."""
    try:
        p = db_client().table("portfolios").select("benchmark").eq("id", portfolio_id).single().execute()
        benchmark = p.data.get("benchmark", "SPXTR") if p.data else "SPXTR"
        a = db_client().table("portfolio_assets").select("*").eq("portfolio_id", portfolio_id).execute()
        assets = a.data or []
        tickers = [x["ticker"] for x in assets]
        weights = {x["ticker"]: x["weight"] for x in assets}
        return tickers, weights, benchmark
    except Exception:
        return [], {}, "SPXTR"

def save_portfolio(name: str, tickers: List[str], weights: Dict[str, float],
                   benchmark: str, portfolio_id: Optional[str] = None) -> str:
    """Salva o aggiorna un portafoglio. Restituisce l'ID."""
    uid = st.session_state.user.id
    if portfolio_id:
        db_client().table("portfolios").update({
            "name": name, "benchmark": benchmark, "updated_at": "now()"
        }).eq("id", portfolio_id).execute()
        db_client().table("portfolio_assets").delete().eq("portfolio_id", portfolio_id).execute()
    else:
        res = db_client().table("portfolios").insert({
            "user_id": uid, "name": name, "benchmark": benchmark
        }).execute()
        portfolio_id = res.data[0]["id"]

    assets = [{"portfolio_id": portfolio_id, "ticker": t, "weight": weights.get(t, 0.0)}
              for t in tickers]
    if assets:
        db_client().table("portfolio_assets").insert(assets).execute()
    return portfolio_id

def delete_portfolio(portfolio_id: str):
    db_client().table("portfolios").delete().eq("id", portfolio_id).execute()

# ══════════════════════════════════════════════════════════════════════════════
# LOGICA DI CALCOLO
# ══════════════════════════════════════════════════════════════════════════════

PERIODS_PER_YEAR = 252

def download_data(tickers: List[str], benchmark: str,
                  n_bars: int = 15_000,
                  tv_username: Optional[str] = None,
                  tv_password: Optional[str] = None) -> pd.DataFrame:
    """Scarica prezzi giornalieri da TradingView."""
    from tvDatafeed import TvDatafeed, Interval
    tv = TvDatafeed(username=tv_username or None, password=tv_password or None)
    all_symbols = list(dict.fromkeys(tickers + ([benchmark] if benchmark not in tickers else [])))
    raw_data: Dict[str, pd.Series] = {}
    invalid: List[str] = []

    progress = st.progress(0)
    status   = st.empty()

    for idx, symbol in enumerate(all_symbols):
        status.text(f"⬇️  Download: {symbol} ({idx+1}/{len(all_symbols)})")
        try:
            # Gestisce formato EXCHANGE:SYMBOL (es. MIL:ENI) e simbolo semplice (es. AAPL)
            if ":" in symbol:
                exch, sym = symbol.split(":", 1)
            else:
                exch, sym = "", symbol
            tv_data = tv.get_hist(symbol=sym, exchange=exch,
                                  interval=Interval.in_daily, n_bars=n_bars)
            if tv_data is not None and not tv_data.empty:
                tv_data.index = tv_data.index.normalize()
                raw_data[symbol] = tv_data["close"]
            else:
                invalid.append(symbol)
        except Exception as e:
            invalid.append(symbol)
        progress.progress((idx + 1) / len(all_symbols))

    progress.empty()
    status.empty()

    if invalid:
        st.warning(f"⚠️ Ticker non trovati: {', '.join(invalid)}")

    df = pd.DataFrame(raw_data)
    valid_cols = [c for c in df.columns if df[c].notna().any()]
    if not valid_cols:
        raise ValueError("Nessun dato valido scaricato.")

    min_date = max(df[c].dropna().index.min() for c in valid_cols)
    max_date = min(df[c].dropna().index.max() for c in valid_cols)
    df = df.loc[(df.index >= min_date) & (df.index <= max_date)].dropna(how="any")
    if df.empty:
        raise ValueError("Nessun dato comune disponibile.")
    return df


def compute_stats(price_df: pd.DataFrame, benchmark: str):
    all_ret  = price_df.pct_change().dropna()
    port_ret = all_ret.drop(columns=[benchmark], errors="ignore") \
               if benchmark not in [t for t in price_df.columns if t != benchmark] \
               else all_ret.copy()
    # se benchmark non è nei tickers lo escludiamo dal portafoglio
    if benchmark in all_ret.columns and benchmark not in st.session_state.get("tickers", []):
        port_ret = all_ret.drop(columns=[benchmark], errors="ignore")
    else:
        port_ret = all_ret.copy()

    bench_ret = all_ret[benchmark]
    mu        = port_ret.mean() * PERIODS_PER_YEAR
    cov       = port_ret.cov()  * PERIODS_PER_YEAR
    return all_ret, port_ret, bench_ret, mu, cov


def simulate_frontier(port_ret: pd.DataFrame, mu: pd.Series,
                      cov: pd.DataFrame, n: int = 10_000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(42)
    na = len(port_ret.columns)
    res = np.zeros((3, n))
    weights_record = []
    for i in range(n):
        w = np.random.random(na); w /= w.sum()
        weights_record.append(w)
        ret = w @ mu.values
        std = np.sqrt(w @ cov.values @ w)
        res[0, i] = std; res[1, i] = ret
        res[2, i] = ret / std if std > 0 else 0
    df = pd.DataFrame(res.T, columns=["Rischio", "Rendimento", "Sharpe Ratio"])
    wdf = pd.DataFrame(weights_record, columns=port_ret.columns)
    df["Weights"] = [
        "<br>".join(f"{c}: {wdf.iloc[i][c]:.2%}" for c in wdf.columns)
        for i in range(len(wdf))
    ]
    return df, wdf


def get_optimal_weights(res_df, wdf):
    idx = int(res_df["Sharpe Ratio"].idxmax())
    return wdf.iloc[idx].values


def normalize_weights(tickers, weight_map):
    raw = np.array([weight_map.get(t, 0.0) for t in tickers])
    s = raw.sum()
    if s < 1e-8:
        raise ValueError("Somma pesi = 0")
    return raw / s


def calc_drawdowns(ret: pd.Series) -> pd.Series:
    cum  = (1 + ret).cumprod()
    peak = cum.cummax()
    return (cum - peak) / peak


def calc_ann_returns(ret: pd.Series) -> pd.Series:
    return ret.groupby(ret.index.year).apply(lambda x: (1 + x).prod() - 1)


def calc_ann_max_dd(ret: pd.Series) -> pd.Series:
    return ret.groupby(ret.index.year).apply(lambda x: calc_drawdowns(x).min())


def calc_rolling_ann(series: pd.Series, years: int) -> pd.Series:
    w = PERIODS_PER_YEAR * years
    return series.rolling(window=w, min_periods=w).mean() * PERIODS_PER_YEAR


def calc_recovery_times(cum: pd.Series) -> List[int]:
    times = []
    peak = cum.iloc[0]; trough = cum.iloc[0]; t_date = cum.index[0]
    for date, val in cum.items():
        if val >= peak:
            if trough < peak:
                times.append((date - t_date).days)
            peak = val; trough = val; t_date = date
        elif val < trough:
            trough = val; t_date = date
    return times


def compute_rolling_optimal(port_ret: pd.DataFrame) -> pd.DataFrame:
    tickers    = port_ret.columns.tolist()
    rebal_dates = port_ret.index[PERIODS_PER_YEAR:]
    df_opt = pd.DataFrame(index=rebal_dates, columns=tickers, dtype=float)
    for date in rebal_dates:
        i    = port_ret.index.get_loc(date)
        hist = port_ret.iloc[:i]
        mu_h = hist.mean() * PERIODS_PER_YEAR
        sg   = hist.cov()  * PERIODS_PER_YEAR
        try:
            raw = np.linalg.inv(sg.values).dot(mu_h.values)
        except np.linalg.LinAlgError:
            raw = np.ones(len(tickers))
        raw = np.where(raw < 0, 0, raw)
        d   = raw.sum()
        df_opt.loc[date] = raw / d if d > 1e-8 else np.ones(len(tickers)) / len(tickers)
    return df_opt


# ══════════════════════════════════════════════════════════════════════════════
# RICERCA LIVE — TradingView Symbol Search API (tutti gli strumenti mondiali)
# ══════════════════════════════════════════════════════════════════════════════

import requests
import re as _re

@st.cache_data(ttl=30)
def search_tv_live(query: str, limit: int = 12) -> List[Dict]:
    """
    Interroga l'API pubblica di TradingView symbol search in tempo reale.
    Copre: azioni (tutti gli exchange mondiali), ETF, indici, crypto,
    materie prime, bond, forex.
    Nessun database locale — risultati sempre aggiornati.
    """
    q = query.strip()
    if not q:
        return []
    try:
        url = "https://symbol-search.tradingview.com/symbol_search/v3/"
        params = {
            "text": q,
            "hl": "0",
            "exchange": "",
            "lang": "it",
            "search_type": "undefined",
            "domain": "production",
            "sort_by_country": "IT",
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Referer": "https://www.tradingview.com/",
            "Origin": "https://www.tradingview.com",
        }
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        if resp.status_code != 200:
            return []
        raw = resp.json()
        symbols_list = raw.get("symbols", raw) if isinstance(raw, dict) else raw
        results = []
        for item in symbols_list[:limit]:
            sym      = _re.sub(r"<[^>]+>", "", str(item.get("symbol", "")))
            exchange = str(item.get("exchange", ""))
            desc     = _re.sub(r"<[^>]+>", "", str(item.get("description", "")))
            stype    = str(item.get("type", ""))
            if not sym:
                continue
            # Icona per tipo strumento
            type_icon = {
                "stock": "📈", "fund": "📦", "index": "🗂️",
                "crypto": "🪙", "forex": "💱", "futures": "⚙️",
                "bond": "🏦", "economic": "📊",
            }.get(stype, "📌")
            results.append({
                "symbol":      sym,
                "exchange":    exchange,
                "description": desc,
                "type":        stype,
                "type_icon":   type_icon,
                "display":     f"{type_icon} {sym} ({exchange}) — {desc}",
                "tv_symbol":   f"{exchange}:{sym}" if exchange else sym,
            })
        return results
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — inizializzazione
# ══════════════════════════════════════════════════════════════════════════════

import json
import streamlit.components.v1 as _components

# ── PERSISTENZA PORTAFOGLIO su localStorage del browser ──────────────────────

DEFAULT_TICKERS = ["ROST", "CHD", "TSCO", "ORLY", "SHW", "BALL",
                   "ROL", "IEX", "AAON", "AAPL", "RMS", "AZO", "APH", "DHR", "MTD"]

if "portfolio_tickers" not in st.session_state:
    st.session_state.portfolio_tickers = DEFAULT_TICKERS.copy()
if "manual_weights" not in st.session_state:
    st.session_state.manual_weights = {}
if "do_reset" not in st.session_state:
    st.session_state.do_reset = False
if "ls_loaded" not in st.session_state:
    st.session_state.ls_loaded = False

# Componente HTML che legge localStorage e lo inietta in un campo hidden
# Al primo caricamento della pagina, invia i dati salvati a Streamlit
_LS_KEY_TICKERS = "mkw_tickers"
_LS_KEY_WEIGHTS = "mkw_weights"

_load_script = f"""
<script>
(function() {{
    const tickers = localStorage.getItem("{_LS_KEY_TICKERS}");
    const weights = localStorage.getItem("{_LS_KEY_WEIGHTS}");
    // Invia i dati al server Streamlit tramite query param al primo load
    if (tickers && !sessionStorage.getItem("mkw_sent")) {{
        sessionStorage.setItem("mkw_sent", "1");
        const url = new URL(window.location.href);
        url.searchParams.set("mkw_t", tickers);
        url.searchParams.set("mkw_w", weights || "{{}}");
        window.history.replaceState(null, "", url.toString());
        window.location.reload();
    }}
}})();
</script>
"""

# Al primo caricamento, legge i query params iniettati dallo script
if not st.session_state.ls_loaded:
    params = st.query_params
    if "mkw_t" in params:
        try:
            loaded_t = json.loads(params["mkw_t"])
            loaded_w = json.loads(params.get("mkw_w", "{}"))
            if isinstance(loaded_t, list) and len(loaded_t) > 0:
                st.session_state.portfolio_tickers = loaded_t
                st.session_state.manual_weights    = loaded_w
            # Pulisce i query params dall'URL
            st.query_params.clear()
        except Exception:
            pass
    st.session_state.ls_loaded = True

# Inietta lo script di lettura solo al primo load (prima della sidebar)
if not st.session_state.get("ls_script_done"):
    _components.html(_load_script, height=0)
    st.session_state.ls_script_done = True


def _save_to_localstorage(tickers: list, weights: dict):
    """Salva portafoglio e pesi nel localStorage del browser."""
    t_json = json.dumps(tickers)
    w_json = json.dumps(weights)
    save_script = f"""
<script>
localStorage.setItem("{_LS_KEY_TICKERS}", {repr(t_json)});
localStorage.setItem("{_LS_KEY_WEIGHTS}", {repr(w_json)});
</script>
"""
    _components.html(save_script, height=0)

# Applica reset se richiesto nel ciclo precedente
if st.session_state.do_reset:
    n_t = len(st.session_state.portfolio_tickers)
    eq  = round(1.0 / n_t, 6) if n_t > 0 else 1.0
    # Svuota manual_weights — i widget leggeranno equal_w come default
    st.session_state.manual_weights = {}
    st.session_state.do_reset = False


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — CONFIGURAZIONE
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📐 Markowitz\nPortfolio Optimizer")
    st.markdown("---")

    st.markdown('<p class="section-title">Credenziali TradingView</p>', unsafe_allow_html=True)
    tv_user = st.text_input("Username (opzionale)", placeholder="lascia vuoto per accesso anonimo")
    tv_pass = st.text_input("Password (opzionale)", type="password")

    st.markdown('<p class="section-title">Benchmark</p>', unsafe_allow_html=True)
    benchmark = st.text_input("Simbolo benchmark", value="SPXTR")

    # ── RICERCA TICKER — autocomplete live ──────────────────────────────────
    st.markdown('<p class="section-title">🔍 Cerca & Aggiungi Ticker</p>', unsafe_allow_html=True)

    from streamlit_searchbox import st_searchbox

    def _tv_search(query: str) -> list:
        """Funzione di ricerca chiamata ad ogni carattere digitato."""
        if not query or len(query.strip()) < 1:
            return []
        results = search_tv_live(query, limit=12)
        # Restituisce lista di tuple (label_display, valore_da_salvare)
        return [
            (item["display"], item["tv_symbol"])
            for item in results
        ]

    selected = st_searchbox(
        _tv_search,
        key       = "ticker_searchbox",
        placeholder = "es. ENI, Apple, Bitcoin, Gold...",
        label     = "Cerca strumento",
        clear_on_submit = True,
        debounce  = 300,   # ms di attesa prima di chiamare l'API
    )

    if selected:
        tv_sym = selected
        if tv_sym not in st.session_state.portfolio_tickers:
            st.session_state.portfolio_tickers.append(tv_sym)
            _save_to_localstorage(
                st.session_state.portfolio_tickers,
                st.session_state.manual_weights
            )
            st.rerun()
        else:
            st.caption(f"✅ {tv_sym} già nel portafoglio")

    # ── LISTA TICKER CON PESI ─────────────────────────────────────────────────
    st.markdown('<p class="section-title">📋 Portafoglio corrente</p>', unsafe_allow_html=True)

    n_tickers = len(st.session_state.portfolio_tickers)
    equal_w   = round(1.0 / n_tickers, 6) if n_tickers > 0 else 1.0

    for ticker in list(st.session_state.portfolio_tickers):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(
                f"<span style='font-family:DM Mono;font-size:12px;color:#e8eaf0'>{ticker}</span>",
                unsafe_allow_html=True
            )
        with col2:
            widget_key = f"w_{ticker}"
            if widget_key not in st.session_state:
                st.session_state[widget_key] = round(equal_w, 6)
            new_w = st.number_input(
                label="peso",
                min_value=0.0, max_value=1.0,
                value=round(float(st.session_state[widget_key]), 6),
                step=0.0001, format="%.4f",
                key=widget_key,
                label_visibility="collapsed"
            )
            st.session_state.manual_weights[ticker] = new_w
        with col3:
            if st.button("🗑️", key=f"del_{ticker}"):
                st.session_state.portfolio_tickers.remove(ticker)
                st.session_state.manual_weights.pop(ticker, None)
                st.session_state.pop(f"w_{ticker}", None)
                _save_to_localstorage(
                    st.session_state.portfolio_tickers,
                    st.session_state.manual_weights
                )
                st.rerun()

    # Salva automaticamente i pesi correnti ad ogni interazione
    _save_to_localstorage(
        st.session_state.portfolio_tickers,
        st.session_state.manual_weights
    )

    # Stato pesi
    if n_tickers > 0:
        total_w = sum(
            st.session_state.manual_weights.get(t, equal_w)
            for t in st.session_state.portfolio_tickers
        )
        is_equal = all(
            abs(st.session_state.manual_weights.get(t, equal_w) - equal_w) < 1e-4
            for t in st.session_state.portfolio_tickers
        )
        if is_equal:
            st.caption(f"⚖️ Equal weight: {equal_w:.2%} per asset")
        else:
            color = "#34d399" if abs(total_w - 1.0) < 0.015 else "#f87171"
            icon  = "✅" if abs(total_w - 1.0) < 0.015 else "⚠️"
            st.markdown(
                f"<span style='font-size:12px;color:{color}'>Σ pesi: {total_w:.4f} {icon}</span>",
                unsafe_allow_html=True
            )
            if abs(total_w - 1.0) > 0.015:
                st.caption("I pesi verranno normalizzati automaticamente a 1.0")

    # Reset Equal Weight — cancella i widget e svuota manual_weights
    if st.button("🔄 Reset Equal Weight"):
        st.session_state.manual_weights = {}
        for t in list(st.session_state.portfolio_tickers):
            st.session_state.pop(f"w_{t}", None)
        st.session_state.do_reset = False
        _save_to_localstorage(st.session_state.portfolio_tickers, {})
        st.rerun()

    st.markdown('<p class="section-title">Parametri simulazione</p>', unsafe_allow_html=True)
    num_portfolios = st.slider("Portafogli simulati", 1000, 20000, 10000, step=1000)
    n_bars         = st.slider("Barre storiche (giorni)", 2000, 15000, 10000, step=1000)

    st.markdown("---")
    run_btn = st.button("🚀  AVVIA ANALISI")


# ── Costruzione lista ticker e pesi finali ───────────────────────────────────
parsed_tickers: List[str] = list(st.session_state.portfolio_tickers)
n_assets = len(parsed_tickers)
equal_weight = 1.0 / n_assets if n_assets > 0 else 1.0

# Recupera pesi (equal weight se non impostati manualmente)
raw_weights = np.array([
    st.session_state.manual_weights.get(t, equal_weight)
    for t in parsed_tickers
])
# Normalizza sempre a 1.0
total_raw = raw_weights.sum()
if total_raw > 1e-8:
    raw_weights = raw_weights / total_raw

custom_weights_map: Dict[str, float] = {
    t: float(raw_weights[i]) for i, t in enumerate(parsed_tickers)
}

parse_error = n_assets == 0

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<h1 style="font-size:2rem; margin-bottom:4px;">
  📈 Frontiera Efficiente di Markowitz
</h1>
<p style="color:#6b7280; font-size:14px; margin-bottom:32px;">
  Ottimizzazione Monte Carlo · Benchmark comparison · Rolling analytics
</p>
""", unsafe_allow_html=True)

if not run_btn:
    st.info("👈  Configura il portafoglio nella barra laterale e premi **AVVIA ANALISI**.")
    st.stop()

if parse_error or not parsed_tickers:
    st.error("Correggi gli errori nel pannello laterale prima di procedere.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# ESECUZIONE ANALISI
# ══════════════════════════════════════════════════════════════════════════════

st.session_state["tickers"] = parsed_tickers

with st.spinner("Connessione a TradingView e download dati..."):
    try:
        price_df = download_data(
            parsed_tickers, benchmark, n_bars=n_bars,
            tv_username=tv_user or None, tv_password=tv_pass or None
        )
    except Exception as e:
        st.error(f"❌ Errore download dati: {e}")
        st.stop()

with st.spinner("Calcolo frontiera efficiente..."):
    all_ret, port_ret, bench_ret, mu, cov = compute_stats(price_df, benchmark)
    res_df, wdf = simulate_frontier(port_ret, mu, cov, num_portfolios)
    opt_w   = get_optimal_weights(res_df, wdf)
    sel_w   = normalize_weights(port_ret.columns.tolist(), custom_weights_map)
    opt_ret = port_ret.dot(opt_w)
    sel_ret = port_ret.dot(sel_w)

with st.spinner("Calcolo metriche e grafici..."):
    ann_ret_opt = calc_ann_returns(opt_ret)
    ann_ret_sel = calc_ann_returns(sel_ret)
    ann_ret_ben = calc_ann_returns(bench_ret)
    ann_dd_opt  = calc_ann_max_dd(opt_ret)
    ann_dd_sel  = calc_ann_max_dd(sel_ret)
    ann_dd_ben  = calc_ann_max_dd(bench_ret)

    common_idx  = opt_ret.index.intersection(bench_ret.index).intersection(sel_ret.index)
    cum_opt  = (1 + opt_ret.loc[common_idx]).cumprod()
    cum_sel  = (1 + sel_ret.loc[common_idx]).cumprod()
    cum_ben  = (1 + bench_ret.loc[common_idx]).cumprod()

    rec_opt = calc_recovery_times(cum_opt)
    rec_sel = calc_recovery_times(cum_sel)
    rec_ben = calc_recovery_times(cum_ben)

st.success("✅ Analisi completata!")

# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — init
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_TICKERS = ["ROST","CHD","TSCO","ORLY","SHW","BALL",
                   "ROL","IEX","AAON","AAPL","RMS","AZO","APH","DHR","MTD"]

for key, default in [
    ("portfolio_tickers", DEFAULT_TICKERS.copy()),
    ("manual_weights",    {}),
    ("do_reset",          False),
    ("current_portfolio_id", None),
    ("current_portfolio_name", "Nuovo portafoglio"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════════════════════
# PAGINA LOGIN — mostrata se utente non loggato
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.user:
    st.markdown("""
    <div style="max-width:420px;margin:80px auto 0 auto;">
    <h1 style="font-family:DM Mono;font-size:1.8rem;margin-bottom:4px;">📈 Markowitz</h1>
    <p style="color:#6b7280;margin-bottom:32px;">Portfolio Optimizer · Accedi o registrati</p>
    </div>
    """, unsafe_allow_html=True)

    col_center = st.columns([1,2,1])[1]
    with col_center:
        tab_login, tab_reg = st.tabs(["🔑 Accedi", "📝 Registrati"])

        with tab_login:
            email_l    = st.text_input("Email", key="login_email")
            password_l = st.text_input("Password", type="password", key="login_pass")
            if st.button("Accedi", key="btn_login"):
                if email_l and password_l:
                    ok, err = do_login(email_l, password_l)
                    if ok:
                        st.rerun()
                    else:
                        st.error(f"❌ {err}")
                else:
                    st.warning("Inserisci email e password")

        with tab_reg:
            email_r    = st.text_input("Email", key="reg_email")
            password_r = st.text_input("Password (min 6 caratteri)", type="password", key="reg_pass")
            if st.button("Registrati", key="btn_reg"):
                if email_r and password_r:
                    ok, err = do_register(email_r, password_r)
                    if ok:
                        st.success("✅ Registrazione completata! Controlla la tua email per confermare, poi accedi.")
                    else:
                        st.error(f"❌ {err}")
                else:
                    st.warning("Inserisci email e password")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# APP PRINCIPALE — solo per utenti loggati
# ══════════════════════════════════════════════════════════════════════════════

# Reset pesi
if st.session_state.do_reset:
    n_t = len(st.session_state.portfolio_tickers)
    eq  = round(1.0 / n_t, 6) if n_t > 0 else 1.0
    st.session_state.manual_weights = {}
    for t in st.session_state.portfolio_tickers:
        st.session_state.pop(f"w_{t}", None)
    st.session_state.do_reset = False

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Header utente
    user_email = st.session_state.user.email
    st.markdown(f"## 📐 Markowitz")
    st.caption(f"👤 {user_email}")
    if st.button("🚪 Logout"):
        do_logout()
    st.markdown("---")

    # ── GESTIONE PORTAFOGLI SALVATI ───────────────────────────────────────────
    st.markdown('<p class="section-title">💾 I tuoi portafogli</p>', unsafe_allow_html=True)
    portfolios = load_portfolios()

    if portfolios:
        port_names = {p["name"]: p["id"] for p in portfolios}
        selected_name = st.selectbox("Carica portafoglio", ["— Nuovo —"] + list(port_names.keys()),
                                     key="sel_portfolio")
        col_load, col_del = st.columns([3,1])
        with col_load:
            if st.button("📂 Carica", key="btn_load") and selected_name != "— Nuovo —":
                pid = port_names[selected_name]
                tickers, weights, bench = load_portfolio_assets(pid)
                if tickers:
                    # Reset widget keys prima di cambiare portafoglio
                    for t in st.session_state.portfolio_tickers:
                        st.session_state.pop(f"w_{t}", None)
                    st.session_state.portfolio_tickers      = tickers
                    st.session_state.manual_weights         = weights
                    st.session_state.current_portfolio_id   = pid
                    st.session_state.current_portfolio_name = selected_name
                    st.rerun()
        with col_del:
            if st.button("🗑️", key="btn_del_port") and selected_name != "— Nuovo —":
                delete_portfolio(port_names[selected_name])
                if st.session_state.current_portfolio_id == port_names[selected_name]:
                    st.session_state.current_portfolio_id   = None
                    st.session_state.portfolio_tickers      = DEFAULT_TICKERS.copy()
                    st.session_state.manual_weights         = {}
                st.rerun()
    else:
        st.caption("Nessun portafoglio salvato")

    # Salva portafoglio corrente
    st.markdown('<p class="section-title">💾 Salva portafoglio</p>', unsafe_allow_html=True)
    port_name_input = st.text_input("Nome portafoglio",
                                    value=st.session_state.current_portfolio_name,
                                    key="port_name_input")
    if st.button("💾 Salva", key="btn_save"):
        if port_name_input.strip():
            pid = save_portfolio(
                name         = port_name_input.strip(),
                tickers      = st.session_state.portfolio_tickers,
                weights      = st.session_state.manual_weights,
                benchmark    = st.session_state.get("benchmark_val", "SPXTR"),
                portfolio_id = st.session_state.current_portfolio_id,
            )
            st.session_state.current_portfolio_id   = pid
            st.session_state.current_portfolio_name = port_name_input.strip()
            st.success("✅ Salvato!")
        else:
            st.warning("Inserisci un nome")

    st.markdown("---")

    # ── CREDENZIALI TV ────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Credenziali TradingView</p>', unsafe_allow_html=True)
    tv_user = st.text_input("Username (opzionale)", placeholder="accesso anonimo")
    tv_pass = st.text_input("Password (opzionale)", type="password")

    st.markdown('<p class="section-title">Benchmark</p>', unsafe_allow_html=True)
    benchmark = st.text_input("Simbolo benchmark", value="SPXTR", key="benchmark_val")

    # ── RICERCA TICKER ────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">🔍 Cerca & Aggiungi Ticker</p>', unsafe_allow_html=True)

    from streamlit_searchbox import st_searchbox

    def _tv_search(query: str) -> list:
        if not query or len(query.strip()) < 1:
            return []
        results = search_tv_live(query, limit=12)
        return [(item["display"], item["tv_symbol"]) for item in results]

    selected = st_searchbox(
        _tv_search,
        key             = "ticker_searchbox",
        placeholder     = "es. ENI, Apple, Bitcoin, Gold...",
        label           = "Cerca strumento",
        clear_on_submit = True,
        debounce        = 300,
    )

    if selected:
        tv_sym = selected
        if tv_sym not in st.session_state.portfolio_tickers:
            st.session_state.portfolio_tickers.append(tv_sym)
            st.rerun()
        else:
            st.caption(f"✅ {tv_sym} già nel portafoglio")

    # ── LISTA TICKER CON PESI ─────────────────────────────────────────────────
    st.markdown('<p class="section-title">📋 Portafoglio corrente</p>', unsafe_allow_html=True)

    n_tickers = len(st.session_state.portfolio_tickers)
    equal_w   = round(1.0 / n_tickers, 6) if n_tickers > 0 else 1.0

    for ticker in list(st.session_state.portfolio_tickers):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(
                f"<span style='font-family:DM Mono;font-size:12px;color:#e8eaf0'>{ticker}</span>",
                unsafe_allow_html=True
            )
        with col2:
            widget_key = f"w_{ticker}"
            if widget_key not in st.session_state:
                st.session_state[widget_key] = round(
                    float(st.session_state.manual_weights.get(ticker, equal_w)), 6
                )
            new_w = st.number_input(
                label="peso", min_value=0.0, max_value=1.0,
                value=round(float(st.session_state[widget_key]), 6),
                step=0.0001, format="%.4f",
                key=widget_key, label_visibility="collapsed"
            )
            st.session_state.manual_weights[ticker] = new_w
        with col3:
            if st.button("🗑️", key=f"del_{ticker}"):
                st.session_state.portfolio_tickers.remove(ticker)
                st.session_state.manual_weights.pop(ticker, None)
                st.session_state.pop(f"w_{ticker}", None)
                st.rerun()

    # Stato pesi
    if n_tickers > 0:
        total_w  = sum(st.session_state.manual_weights.get(t, equal_w)
                       for t in st.session_state.portfolio_tickers)
        is_equal = all(abs(st.session_state.manual_weights.get(t, equal_w) - equal_w) < 1e-4
                       for t in st.session_state.portfolio_tickers)
        if is_equal:
            st.caption(f"⚖️ Equal weight: {equal_w:.2%} per asset")
        else:
            color = "#34d399" if abs(total_w - 1.0) < 0.015 else "#f87171"
            icon  = "✅" if abs(total_w - 1.0) < 0.015 else "⚠️"
            st.markdown(f"<span style='font-size:12px;color:{color}'>Σ pesi: {total_w:.4f} {icon}</span>",
                        unsafe_allow_html=True)
            if abs(total_w - 1.0) > 0.015:
                st.caption("I pesi verranno normalizzati automaticamente a 1.0")

    if st.button("🔄 Reset Equal Weight"):
        st.session_state.manual_weights = {}
        for t in list(st.session_state.portfolio_tickers):
            st.session_state.pop(f"w_{t}", None)
        st.session_state.do_reset = False
        st.rerun()

    st.markdown('<p class="section-title">Parametri simulazione</p>', unsafe_allow_html=True)
    num_portfolios = st.slider("Portafogli simulati", 1000, 20000, 10000, step=1000)
    n_bars         = st.slider("Barre storiche (giorni)", 2000, 15000, 10000, step=1000)
    st.markdown("---")
    run_btn = st.button("🚀  AVVIA ANALISI")

# ── Pesi finali ───────────────────────────────────────────────────────────────
parsed_tickers: List[str] = list(st.session_state.portfolio_tickers)
n_assets     = len(parsed_tickers)
equal_weight = 1.0 / n_assets if n_assets > 0 else 1.0
raw_weights  = np.array([st.session_state.manual_weights.get(t, equal_weight)
                         for t in parsed_tickers])
total_raw    = raw_weights.sum()
if total_raw > 1e-8:
    raw_weights = raw_weights / total_raw
custom_weights_map = {t: float(raw_weights[i]) for i, t in enumerate(parsed_tickers)}
parse_error  = n_assets == 0

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<h1 style="font-size:2rem; margin-bottom:4px;">
  📈 Frontiera Efficiente di Markowitz
</h1>
<p style="color:#6b7280; font-size:14px; margin-bottom:32px;">
  Ottimizzazione Monte Carlo · Benchmark comparison · Rolling analytics
</p>
""", unsafe_allow_html=True)

if not run_btn:
    # Mostra portafoglio corrente in attesa
    if parsed_tickers:
        st.info(f"👈 Portafoglio: **{st.session_state.current_portfolio_name}** "
                f"({len(parsed_tickers)} asset) · Premi **AVVIA ANALISI** per calcolare")
    else:
        st.info("👈 Configura il portafoglio nella barra laterale e premi **AVVIA ANALISI**.")
    st.stop()

if parse_error:
    st.error("Aggiungi almeno un ticker al portafoglio.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# ESECUZIONE ANALISI
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("Connessione a TradingView e download dati..."):
    try:
        price_df = download_data(parsed_tickers, benchmark, n_bars=n_bars,
                                 tv_username=tv_user or None, tv_password=tv_pass or None)
    except Exception as e:
        st.error(f"❌ Errore download dati: {e}")
        st.stop()

with st.spinner("Calcolo frontiera efficiente..."):
    all_ret, port_ret, bench_ret, mu, cov = compute_stats(price_df, benchmark)
    res_df, wdf = simulate_frontier(port_ret, mu, cov, num_portfolios)
    opt_w       = get_optimal_weights(res_df, wdf)
    sel_w       = normalize_weights(port_ret.columns.tolist(), custom_weights_map)
    opt_ret     = port_ret.dot(opt_w)
    sel_ret     = port_ret.dot(sel_w)

with st.spinner("Calcolo metriche..."):
    ann_ret_opt = calc_ann_returns(opt_ret)
    ann_ret_sel = calc_ann_returns(sel_ret)
    ann_ret_ben = calc_ann_returns(bench_ret)
    ann_dd_opt  = calc_ann_max_dd(opt_ret)
    ann_dd_sel  = calc_ann_max_dd(sel_ret)
    ann_dd_ben  = calc_ann_max_dd(bench_ret)
    common_idx  = opt_ret.index.intersection(bench_ret.index).intersection(sel_ret.index)
    cum_opt  = (1 + opt_ret.loc[common_idx]).cumprod()
    cum_sel  = (1 + sel_ret.loc[common_idx]).cumprod()
    cum_ben  = (1 + bench_ret.loc[common_idx]).cumprod()
    rec_opt = calc_recovery_times(cum_opt)
    rec_sel = calc_recovery_times(cum_sel)
    rec_ben = calc_recovery_times(cum_ben)

st.success("✅ Analisi completata!")

# ══════════════════════════════════════════════════════════════════════════════
# TAB — grafici (identici all'app originale)
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌐 Frontiera Efficiente",
    "📊 Performance & Drawdown",
    "📈 Rendimenti Cumulativi",
    "🔄 Rolling Analytics",
    "⚖️  Pesi & Recovery",
])


# ── TAB 1: Frontiera Efficiente ───────────────────────────────────────────────
with tab1:
    tickers_list = port_ret.columns.tolist()

    risk_opt = float(np.sqrt(opt_w @ cov.values @ opt_w))
    ret_opt  = float(opt_w @ mu.values)
    risk_sel = float(np.sqrt(sel_w @ cov.values @ sel_w))
    ret_sel  = float(sel_w @ mu.values)

    opt_label = "<br>".join(f"{t}: {w:.2%}" for t, w in zip(tickers_list, opt_w))
    sel_label = "<br>".join(f"{t}: {w:.2%}" for t, w in zip(tickers_list, sel_w))

    fig = px.scatter(
        res_df, x="Rischio", y="Rendimento", color="Sharpe Ratio",
        color_continuous_scale="Plasma", hover_data=["Weights"],
        opacity=0.6, title="",
    )
    fig.update_layout(
        paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
        font=dict(color="#e8eaf0", family="DM Mono"),
        height=600,
        xaxis=dict(gridcolor="#1e2130", title="Rischio (Deviazione Standard)"),
        yaxis=dict(gridcolor="#1e2130", title="Rendimento Atteso Annualizzato"),
        legend=dict(bgcolor="#13151c", bordercolor="#1e2130"),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    fig.data[0].update(marker=dict(size=3, opacity=0.35))
    fig.add_scatter(
        x=[risk_opt], y=[ret_opt], mode="markers",
        marker=dict(color="#f87171", size=16, line=dict(width=2, color="white")),
        name="Portafoglio Ottimale", hovertext=[opt_label], hoverinfo="text",
    )
    fig.add_scatter(
        x=[risk_sel], y=[ret_sel], mode="markers",
        marker=dict(color="#34d399", size=16, line=dict(width=2, color="white")),
        name="Portafoglio Selezionato", hovertext=[sel_label], hoverinfo="text",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metriche affiancate
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">🔴 Ottimale — Rendimento atteso</div>
            <div class="value positive">{ret_opt:.2%}</div>
        </div>
        <div class="metric-card">
            <div class="label">🔴 Ottimale — Rischio</div>
            <div class="value">{risk_opt:.2%}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">🟢 Selezionato — Rendimento atteso</div>
            <div class="value positive">{ret_sel:.2%}</div>
        </div>
        <div class="metric-card">
            <div class="label">🟢 Selezionato — Rischio</div>
            <div class="value">{risk_sel:.2%}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        sharpe_opt = ret_opt / risk_opt if risk_opt > 0 else 0
        sharpe_sel = ret_sel / risk_sel if risk_sel > 0 else 0
        st.markdown(f"""<div class="metric-card">
            <div class="label">Sharpe Ratio — Ottimale</div>
            <div class="value">{sharpe_opt:.3f}</div>
        </div>
        <div class="metric-card">
            <div class="label">Sharpe Ratio — Selezionato</div>
            <div class="value">{sharpe_sel:.3f}</div>
        </div>""", unsafe_allow_html=True)


# ── TAB 2: Performance & Drawdown ─────────────────────────────────────────────
with tab2:
    years_idx = ann_ret_opt.index.tolist()
    x         = list(range(len(years_idx)))

    fig_ret = go.Figure()
    w = 0.28
    for i_shift, (series, name, color) in enumerate([
        (ann_ret_opt, "Ottimale",    "#3b82f6"),
        (ann_ret_sel, "Selezionato", "#34d399"),
        (ann_ret_ben, "Benchmark",   "#f59e0b"),
    ]):
        fig_ret.add_bar(
            x=[xi + (i_shift - 1) * w for xi in x],
            y=series.values * 100, name=name,
            marker_color=color, width=w,
        )
        fig_ret.add_hline(
            y=series.mean() * 100,
            line=dict(color=color, dash="dot", width=1),
            annotation_text=f"μ {name}: {series.mean()*100:.1f}%",
            annotation_position="top right",
        )

    fig_ret.update_layout(
        title="Rendimenti Annualizzati (%)",
        barmode="overlay",
        paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
        font=dict(color="#e8eaf0", family="DM Mono"),
        xaxis=dict(tickvals=x, ticktext=[str(y) for y in years_idx],
                   gridcolor="#1e2130"),
        yaxis=dict(gridcolor="#1e2130", title="%"),
        height=420, legend=dict(bgcolor="#13151c"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_ret, use_container_width=True)

    fig_dd = go.Figure()
    for i_shift, (series, name, color) in enumerate([
        (ann_dd_opt, "Ottimale",    "#3b82f6"),
        (ann_dd_sel, "Selezionato", "#34d399"),
        (ann_dd_ben, "Benchmark",   "#f59e0b"),
    ]):
        fig_dd.add_bar(
            x=[xi + (i_shift - 1) * w for xi in x],
            y=series.values * 100, name=name,
            marker_color=color, width=w,
        )

    fig_dd.update_layout(
        title="Drawdown Massimi Annuali (%)",
        barmode="overlay",
        paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
        font=dict(color="#e8eaf0", family="DM Mono"),
        xaxis=dict(tickvals=x, ticktext=[str(y) for y in years_idx],
                   gridcolor="#1e2130"),
        yaxis=dict(gridcolor="#1e2130", title="%"),
        height=420, legend=dict(bgcolor="#13151c"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # Tabella riepilogativa
    st.markdown('<p class="section-title">Riepilogo statistiche</p>', unsafe_allow_html=True)
    summary = pd.DataFrame({
        "": ["Rend. medio annuo (%)", "Rend. mediano annuo (%)",
             "Drawdown medio (%)", "Drawdown mediano (%)",
             "Recovery medio (giorni)"],
        "Ottimale": [
            f"{ann_ret_opt.mean()*100:.2f}%",
            f"{ann_ret_opt.median()*100:.2f}%",
            f"{ann_dd_opt.mean()*100:.2f}%",
            f"{ann_dd_opt.median()*100:.2f}%",
            f"{np.mean(rec_opt):.0f}" if rec_opt else "n/a",
        ],
        "Selezionato": [
            f"{ann_ret_sel.mean()*100:.2f}%",
            f"{ann_ret_sel.median()*100:.2f}%",
            f"{ann_dd_sel.mean()*100:.2f}%",
            f"{ann_dd_sel.median()*100:.2f}%",
            f"{np.mean(rec_sel):.0f}" if rec_sel else "n/a",
        ],
        "Benchmark": [
            f"{ann_ret_ben.mean()*100:.2f}%",
            f"{ann_ret_ben.median()*100:.2f}%",
            f"{ann_dd_ben.mean()*100:.2f}%",
            f"{ann_dd_ben.median()*100:.2f}%",
            f"{np.mean(rec_ben):.0f}" if rec_ben else "n/a",
        ],
    })
    st.dataframe(summary.set_index(""), use_container_width=True)


# ── TAB 3: Rendimenti Cumulativi ──────────────────────────────────────────────
with tab3:
    fig_cum = go.Figure()
    fig_cum.add_scatter(x=cum_opt.index,  y=cum_opt.values,
                        name="Ottimale",    line=dict(color="#3b82f6", width=2))
    fig_cum.add_scatter(x=cum_sel.index,  y=cum_sel.values,
                        name="Selezionato", line=dict(color="#34d399", width=2))
    fig_cum.add_scatter(x=cum_ben.index,  y=cum_ben.values,
                        name=benchmark,     line=dict(color="#f59e0b", width=2, dash="dot"))
    fig_cum.update_layout(
        title="Rendimenti Cumulativi (scala logaritmica)",
        paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
        font=dict(color="#e8eaf0", family="DM Mono"),
        yaxis=dict(type="log", gridcolor="#1e2130", title="Valore cumulativo (log)"),
        xaxis=dict(gridcolor="#1e2130"),
        height=500, legend=dict(bgcolor="#13151c"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Drawdown chart
    dd_opt_s = calc_drawdowns(opt_ret.loc[common_idx])
    dd_sel_s = calc_drawdowns(sel_ret.loc[common_idx])
    dd_ben_s = calc_drawdowns(bench_ret.loc[common_idx])

    fig_dd2 = go.Figure()
    fig_dd2.add_scatter(x=dd_opt_s.index, y=dd_opt_s.values * 100,
                        name="Ottimale",    fill="tozeroy",
                        line=dict(color="#3b82f6"), fillcolor="rgba(59,130,246,0.15)")
    fig_dd2.add_scatter(x=dd_sel_s.index, y=dd_sel_s.values * 100,
                        name="Selezionato", fill="tozeroy",
                        line=dict(color="#34d399"), fillcolor="rgba(52,211,153,0.15)")
    fig_dd2.add_scatter(x=dd_ben_s.index, y=dd_ben_s.values * 100,
                        name=benchmark,     fill="tozeroy",
                        line=dict(color="#f59e0b", dash="dot"),
                        fillcolor="rgba(245,158,11,0.10)")
    fig_dd2.update_layout(
        title="Drawdown nel tempo (%)",
        paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
        font=dict(color="#e8eaf0", family="DM Mono"),
        yaxis=dict(gridcolor="#1e2130", title="Drawdown (%)"),
        xaxis=dict(gridcolor="#1e2130"),
        height=380, legend=dict(bgcolor="#13151c"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_dd2, use_container_width=True)


# ── TAB 4: Rolling Analytics ──────────────────────────────────────────────────
with tab4:
    roll_year = st.select_slider(
        "Orizzonte rolling (anni)", options=[1, 3, 5, 10, 15, 20], value=5
    )
    roll_opt = calc_rolling_ann(opt_ret, roll_year)
    roll_sel = calc_rolling_ann(sel_ret, roll_year)
    roll_ben = calc_rolling_ann(bench_ret, roll_year)

    fig_roll = go.Figure()
    fig_roll.add_scatter(x=roll_opt.index, y=roll_opt.values * 100,
                         name="Ottimale",    line=dict(color="#3b82f6", width=2))
    fig_roll.add_scatter(x=roll_sel.index, y=roll_sel.values * 100,
                         name="Selezionato", line=dict(color="#34d399", width=2))
    fig_roll.add_scatter(x=roll_ben.index, y=roll_ben.values * 100,
                         name=benchmark,     line=dict(color="#f59e0b", width=2, dash="dot"))
    fig_roll.add_hline(y=0, line=dict(color="#6b7280", dash="dash", width=1))
    fig_roll.update_layout(
        title=f"Rolling Returns Annualizzati — Finestra {roll_year} anni (%)",
        paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
        font=dict(color="#e8eaf0", family="DM Mono"),
        yaxis=dict(gridcolor="#1e2130", title="%"),
        xaxis=dict(gridcolor="#1e2130"),
        height=420, legend=dict(bgcolor="#13151c"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_roll, use_container_width=True)

    # Distribuzione rolling
    st.markdown('<p class="section-title">Distribuzione rendimenti rolling</p>', unsafe_allow_html=True)
    fig_dist = go.Figure()
    for series, name, color in [
        (roll_opt.dropna(), "Ottimale",    "#3b82f6"),
        (roll_sel.dropna(), "Selezionato", "#34d399"),
        (roll_ben.dropna(), "Benchmark",   "#f59e0b"),
    ]:
        fig_dist.add_trace(go.Histogram(
            x=series * 100, name=name,
            marker_color=color, opacity=0.55,
            nbinsx=50,
        ))
    fig_dist.update_layout(
        barmode="overlay",
        paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
        font=dict(color="#e8eaf0", family="DM Mono"),
        xaxis=dict(gridcolor="#1e2130", title="Rendimento Annualizzato (%)"),
        yaxis=dict(gridcolor="#1e2130", title="Frequenza"),
        height=380, legend=dict(bgcolor="#13151c"),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_dist, use_container_width=True)


# ── TAB 5: Pesi & Recovery ────────────────────────────────────────────────────
with tab5:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">Pesi portafoglio ottimale (Sharpe)</p>',
                    unsafe_allow_html=True)
        opt_w_df = pd.DataFrame({
            "Ticker": port_ret.columns.tolist(),
            "Peso (%)": [f"{w:.2%}" for w in opt_w],
        }).sort_values("Peso (%)", ascending=False)
        fig_pie_opt = go.Figure(go.Pie(
            labels=opt_w_df["Ticker"], values=opt_w,
            hole=0.45,
            marker=dict(colors=px.colors.sequential.Plasma_r[:len(opt_w)]),
        ))
        fig_pie_opt.update_layout(
            paper_bgcolor="#0d0f14", font=dict(color="#e8eaf0", family="DM Mono"),
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(bgcolor="#13151c"),
        )
        st.plotly_chart(fig_pie_opt, use_container_width=True)
        st.dataframe(opt_w_df.set_index("Ticker"), use_container_width=True)

    with col2:
        st.markdown('<p class="section-title">Pesi portafoglio selezionato</p>',
                    unsafe_allow_html=True)
        sel_w_df = pd.DataFrame({
            "Ticker": port_ret.columns.tolist(),
            "Peso (%)": [f"{w:.2%}" for w in sel_w],
        }).sort_values("Peso (%)", ascending=False)
        fig_pie_sel = go.Figure(go.Pie(
            labels=sel_w_df["Ticker"], values=sel_w,
            hole=0.45,
            marker=dict(colors=px.colors.sequential.Viridis[:len(sel_w)]),
        ))
        fig_pie_sel.update_layout(
            paper_bgcolor="#0d0f14", font=dict(color="#e8eaf0", family="DM Mono"),
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(bgcolor="#13151c"),
        )
        st.plotly_chart(fig_pie_sel, use_container_width=True)
        st.dataframe(sel_w_df.set_index("Ticker"), use_container_width=True)

    # Recovery times
    st.markdown('<p class="section-title">Analisi recovery times</p>', unsafe_allow_html=True)
    portfolios_names = ["Ottimale", "Selezionato", benchmark]
    avg_rec = [
        np.mean(rec_opt) if rec_opt else 0,
        np.mean(rec_sel) if rec_sel else 0,
        np.mean(rec_ben) if rec_ben else 0,
    ]
    colors_rec = ["#3b82f6", "#34d399", "#f59e0b"]

    fig_rec = go.Figure(go.Bar(
        x=portfolios_names, y=avg_rec,
        marker_color=colors_rec,
        text=[f"{v:.0f} giorni" for v in avg_rec],
        textposition="outside",
    ))
    fig_rec.update_layout(
        title="Tempo medio di Recovery (giorni)",
        paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
        font=dict(color="#e8eaf0", family="DM Mono"),
        yaxis=dict(gridcolor="#1e2130", title="Giorni"),
        xaxis=dict(gridcolor="#1e2130"),
        height=360, margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_rec, use_container_width=True)

    # Scatter recovery individuali
    rec_data = {"Ottimale": rec_opt, "Selezionato": rec_sel, benchmark: rec_ben}
    fig_scat = go.Figure()
    for i, (label, times) in enumerate(rec_data.items()):
        if times:
            jitter = np.random.uniform(-0.15, 0.15, len(times))
            fig_scat.add_scatter(
                x=[i + j for j in jitter], y=times,
                mode="markers", name=label,
                marker=dict(color=colors_rec[i], size=8, opacity=0.7),
            )
    fig_scat.update_layout(
        title="Recovery Times Individuali",
        paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
        font=dict(color="#e8eaf0", family="DM Mono"),
        xaxis=dict(tickvals=[0, 1, 2], ticktext=portfolios_names, gridcolor="#1e2130"),
        yaxis=dict(gridcolor="#1e2130", title="Giorni"),
        height=360, legend=dict(bgcolor="#13151c"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_scat, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='color:#374151; font-size:12px; font-family:DM Mono;'>"
    "Markowitz Portfolio Optimizer · dati via TradingView (tvdatafeed) · "
    "Nessuna raccomandazione finanziaria</p>",
    unsafe_allow_html=True,
)
