# -*- coding: utf-8 -*-
"""
app.py — Streamlit Web App: Frontiera Efficiente di Markowitz
=============================================================
Come avviare:
    pip install streamlit tvdatafeed numpy pandas plotly seaborn matplotlib
    streamlit run app.py

Come pubblicare gratis:
    1. Crea account su https://streamlit.io
    2. Carica questo file su GitHub
    3. Collega il repo su share.streamlit.io → Deploy
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
import logging
import io

# ── Page config ──────────────────────────────────────────────────────────────
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
# LOGICA DI CALCOLO (backend puro, nessuna dipendenza da Streamlit)
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
            tv_data = tv.get_hist(symbol=symbol, exchange="",
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
# DATABASE TICKER — nome completo → simbolo
# ══════════════════════════════════════════════════════════════════════════════

TICKER_DB: Dict[str, str] = {
    # S&P 500 — tutti i principali
    "Apple": "AAPL", "Microsoft": "MSFT", "Amazon": "AMZN", "Alphabet": "GOOGL",
    "Google": "GOOGL", "Meta": "META", "Tesla": "TSLA", "Nvidia": "NVDA",
    "Berkshire Hathaway": "BRK.B", "JPMorgan": "JPM", "Johnson & Johnson": "JNJ",
    "Visa": "V", "Mastercard": "MA", "Procter & Gamble": "PG", "UnitedHealth": "UNH",
    "Exxon Mobil": "XOM", "Home Depot": "HD", "Chevron": "CVX", "Abbvie": "ABBV",
    "Eli Lilly": "LLY", "Costco": "COST", "PepsiCo": "PEP", "Coca Cola": "KO",
    "Merck": "MRK", "Netflix": "NFLX", "Walt Disney": "DIS", "Salesforce": "CRM",
    "Adobe": "ADBE", "Oracle": "ORCL", "Intel": "INTC", "AMD": "AMD",
    "Qualcomm": "QCOM", "Texas Instruments": "TXN", "Broadcom": "AVGO",
    "Goldman Sachs": "GS", "Morgan Stanley": "MS", "Bank of America": "BAC",
    "Citigroup": "C", "Wells Fargo": "WFC", "American Express": "AXP",
    "Boeing": "BA", "Caterpillar": "CAT", "Deere": "DE", "3M": "MMM",
    "General Electric": "GE", "Honeywell": "HON", "Raytheon": "RTX",
    "Lockheed Martin": "LMT", "Northrop Grumman": "NOC",
    "Starbucks": "SBUX", "McDonald's": "MCD", "Nike": "NKE",
    "Pfizer": "PFE", "Moderna": "MRNA", "Bristol-Myers Squibb": "BMY",
    "Palo Alto Networks": "PANW", "CrowdStrike": "CRWD", "Fortinet": "FTNT",
    "Airbnb": "ABNB", "Uber": "UBER", "Palantir": "PLTR",
    "Snowflake": "SNOW", "Datadog": "DDOG", "Cloudflare": "NET",
    "PayPal": "PYPL", "Block": "SQ", "Intuit": "INTU", "ServiceNow": "NOW",
    "Workday": "WDAY", "Zoom": "ZM", "Spotify": "SPOT",
    "Accenture": "ACN", "Automatic Data Processing": "ADP",
    "Fiserv": "FI", "Fidelity National": "FIS", "Global Payments": "GPN",
    "Moody's": "MCO", "S&P Global": "SPGI", "Intercontinental Exchange": "ICE",
    "CME Group": "CME", "Nasdaq Inc": "NDAQ",
    "Abbott Laboratories": "ABT", "Medtronic": "MDT", "Stryker": "SYK",
    "Boston Scientific": "BSX", "Zimmer Biomet": "ZBH", "Becton Dickinson": "BDX",
    "Thermo Fisher": "TMO", "Agilent": "A", "Waters": "WAT",
    "Waste Management": "WM", "Republic Services": "RSG",
    "American Tower": "AMT", "Crown Castle": "CCI", "Prologis": "PLD",
    "Simon Property": "SPG", "Realty Income": "O",
    "NextEra Energy": "NEE", "Duke Energy": "DUK", "Southern Company": "SO",
    "Dominion Energy": "D", "Exelon": "EXC", "Sempra": "SRE",
    "ConocoPhillips": "COP", "Pioneer Natural Resources": "PXD",
    "Schlumberger": "SLB", "Halliburton": "HAL", "Baker Hughes": "BKR",
    "Colgate-Palmolive": "CL", "Kimberly-Clark": "KMB", "Clorox": "CLX",
    "Estee Lauder": "EL", "Church & Dwight": "CHD",
    "Ross Stores": "ROST", "TJX Companies": "TJX", "Burlington": "BURL",
    "Tractor Supply": "TSCO", "O'Reilly Automotive": "ORLY",
    "Sherwin-Williams": "SHW", "Ball Corporation": "BALL",
    "Rollins": "ROL", "IDEX Corporation": "IEX",
    "AAON Inc": "AAON", "AutoZone": "AZO", "Amphenol": "APH",
    "Danaher": "DHR", "Mettler-Toledo": "MTD",
    "Roper Technologies": "ROP", "Parker Hannifin": "PH",
    "Illinois Tool Works": "ITW", "Emerson Electric": "EMR",
    "Rockwell Automation": "ROK", "Fortive": "FTV",
    "Verisk Analytics": "VRSK", "Fair Isaac": "FICO", "Gartner": "IT",
    "Costar Group": "CSGP", "Zillow": "Z",
    "Charter Communications": "CHTR", "Comcast": "CMCSA",
    "T-Mobile": "TMUS", "Verizon": "VZ", "AT&T": "T",
    "Motorola Solutions": "MSI", "Keysight": "KEYS",
    "Analog Devices": "ADI", "Microchip Technology": "MCHP",
    "KLA Corporation": "KLAC", "Lam Research": "LRCX",
    "Applied Materials": "AMAT", "ASML Holding": "ASML",
    "Marvell Technology": "MRVL", "Micron": "MU", "Western Digital": "WDC",
    "Seagate": "STX", "Corning": "GLW",
    "FedEx": "FDX", "UPS": "UPS", "Norfolk Southern": "NSC",
    "Union Pacific": "UNP", "CSX": "CSX",
    "American Airlines": "AAL", "Delta": "DAL", "Southwest": "LUV",
    "United Airlines": "UAL", "Carnival": "CCL", "Royal Caribbean": "RCL",
    "Marriott": "MAR", "Hilton": "HLT", "Wyndham": "WH",
    "Las Vegas Sands": "LVS", "MGM Resorts": "MGM", "Wynn": "WYNN",
    "Chipotle": "CMG", "Yum Brands": "YUM", "Darden": "DRI",
    "Dollar General": "DG", "Dollar Tree": "DLTR", "Target": "TGT",
    "Walmart": "WMT", "Kroger": "KR", "Sysco": "SYY",
    "CVS Health": "CVS", "Walgreens": "WBA", "Cardinal Health": "CAH",
    "McKesson": "MCK", "AmerisourceBergen": "ABC",
    "Cigna": "CI", "Humana": "HUM", "Centene": "CNC",
    "Regeneron": "REGN", "Biogen": "BIIB", "Vertex": "VRTX",
    "Gilead": "GILD", "Amgen": "AMGN", "Celanese": "CE",
    "Dow Chemical": "DOW", "DuPont": "DD", "LyondellBasell": "LYB",
    "Air Products": "APD", "Linde": "LIN", "PPG Industries": "PPG",
    "Freeport-McMoRan": "FCX", "Newmont": "NEM", "Nucor": "NUE",
    # ETF
    "S&P 500 ETF SPY": "SPY", "Nasdaq ETF QQQ": "QQQ",
    "Total Market VTI": "VTI", "World ETF VT": "VT",
    "Emerging Markets VWO": "VWO", "Bond ETF BND": "BND",
    "Gold ETF GLD": "GLD", "Silver ETF SLV": "SLV",
    "Real Estate ETF VNQ": "VNQ", "Tech ETF FTEC": "FTEC",
    "Dividend ETF VYM": "VYM", "Growth ETF VUG": "VUG",
    "Value ETF VTV": "VTV", "Small Cap ETF VB": "VB",
    "Europe ETF VGK": "VGK", "Asia ETF VPL": "VPL",
    # Indici
    "S&P 500 Total Return": "SPXTR", "Nasdaq 100": "NDX",
    "Dow Jones": "DJI", "Russell 2000": "RUT", "VIX": "VIX",
    # Europa
    "LVMH": "LVMH", "SAP": "SAP", "Nestlé": "NESN",
    "Novartis": "NVS", "Roche": "ROG", "Hermes": "RMS",
    "Volkswagen": "VOW3", "BMW": "BMW", "Mercedes": "MBG",
    "Siemens": "SIE", "Allianz": "ALV", "Airbus": "AIR",
    "Total Energies": "TTE", "BNP Paribas": "BNP",
    "Unilever": "UL", "AstraZeneca": "AZN", "GlaxoSmithKline": "GSK",
    "BP": "BP", "Shell": "SHEL", "Rio Tinto": "RIO",
}

def search_ticker(query: str) -> List[Tuple[str, str]]:
    """
    Cerca ticker per nome o simbolo nel database.
    Restituisce lista di (nome_display, simbolo).
    Se non trovato nel DB, restituisce lista vuota
    (il simbolo viene comunque aggiunto come custom).
    """
    q = query.strip().lower()
    if not q or len(q) < 1:
        return []

    results: Dict[str, str] = {}  # symbol → display_name

    for name, symbol in TICKER_DB.items():
        # Match esatto simbolo o nome
        if symbol.lower() == q or name.lower() == q:
            results[symbol] = f"{symbol} — {name}"
        # Match parziale su simbolo o nome
        elif q in symbol.lower() or q in name.lower():
            results[symbol] = f"{symbol} — {name}"

    # Ordina: prima i match esatti sul simbolo
    sorted_results = sorted(
        results.items(),
        key=lambda x: (0 if x[0].lower().startswith(q) else 1)
    )
    return [(display, sym) for sym, display in sorted_results[:8]]


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — inizializzazione
# ══════════════════════════════════════════════════════════════════════════════

if "portfolio_tickers" not in st.session_state:
    st.session_state.portfolio_tickers = [
        "ROST", "CHD", "TSCO", "ORLY", "SHW", "BALL",
        "ROL", "IEX", "AAON", "AAPL", "RMS", "AZO", "APH", "DHR", "MTD"
    ]
if "manual_weights" not in st.session_state:
    st.session_state.manual_weights = {}   # {ticker: float} — vuoto = equal weight
if "do_reset" not in st.session_state:
    st.session_state.do_reset = False

# Applica reset se richiesto nel ciclo precedente
if st.session_state.do_reset:
    st.session_state.manual_weights = {}
    # Cancella anche i widget number_input dal session state
    for t in st.session_state.portfolio_tickers:
        st.session_state.pop(f"w_{t}", None)
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

    # ── RICERCA TICKER ────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">🔍 Cerca & Aggiungi Ticker</p>', unsafe_allow_html=True)
    search_query = st.text_input(
        "Cerca per nome o simbolo",
        placeholder="es. Apple, ACN, Nvidia, SPY...",
        key="ticker_search"
    )

    if search_query and len(search_query.strip()) >= 1:
        sym_upper = search_query.strip().upper()
        results   = search_ticker(search_query)

        if results:
            # Mostra i risultati trovati nel DB
            for display, symbol in results:
                already = symbol in st.session_state.portfolio_tickers
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    color = "#6b7280" if already else "#e8eaf0"
                    st.markdown(
                        f"<span style='font-size:12px;font-family:DM Mono;color:{color}'>{display}</span>",
                        unsafe_allow_html=True
                    )
                with col_b:
                    if already:
                        st.markdown("✅", unsafe_allow_html=True)
                    else:
                        if st.button("➕", key=f"add_{symbol}"):
                            st.session_state.portfolio_tickers.append(symbol)
                            st.rerun()
        else:
            # Non trovato nel DB → aggiunta come simbolo custom diretto
            already = sym_upper in st.session_state.portfolio_tickers
            if already:
                st.success(f"✅ {sym_upper} già nel portafoglio")
            else:
                st.info(f"Simbolo **{sym_upper}** non trovato nel database.\nVerrà aggiunto come simbolo custom (TradingView verificherà la disponibilità dei dati).")
                if st.button(f"➕ Aggiungi {sym_upper} come custom"):
                    st.session_state.portfolio_tickers.append(sym_upper)
                    st.rerun()

    # ── LISTA TICKER CON PESI ─────────────────────────────────────────────────
    st.markdown('<p class="section-title">📋 Portafoglio corrente</p>', unsafe_allow_html=True)

    n_tickers = len(st.session_state.portfolio_tickers)
    equal_w   = round(1.0 / n_tickers, 6) if n_tickers > 0 else 1.0

    tickers_to_remove = []
    for ticker in list(st.session_state.portfolio_tickers):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(
                f"<span style='font-family:DM Mono;font-size:12px;color:#e8eaf0'>{ticker}</span>",
                unsafe_allow_html=True
            )
        with col2:
            stored_w = st.session_state.manual_weights.get(ticker, equal_w)
            new_w = st.number_input(
                label="peso",
                min_value=0.0, max_value=1.0,
                value=float(stored_w),
                step=0.01, format="%.4f",
                key=f"w_{ticker}",
                label_visibility="collapsed"
            )
            # Registra sempre il valore corrente
            st.session_state.manual_weights[ticker] = new_w
        with col3:
            if st.button("🗑️", key=f"del_{ticker}"):
                tickers_to_remove.append(ticker)

    for t in tickers_to_remove:
        st.session_state.portfolio_tickers.remove(t)
        st.session_state.manual_weights.pop(t, None)
        st.session_state.pop(f"w_{t}", None)
    if tickers_to_remove:
        st.rerun()

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

    # Bottone reset — usa flag per evitare conflitti con number_input
    if st.button("🔄 Reset Equal Weight"):
        st.session_state.do_reset = True
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
