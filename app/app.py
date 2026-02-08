"""
Cross-Market Contagion Dashboard
================================
Streamlit app for analyzing contagion and volatility spillovers
between cryptocurrency and traditional financial markets.

Run: streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import community as community_louvain
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from pathlib import Path
from datetime import date, timedelta
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Cross-Market Contagion",
    page_icon="\U0001F310",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONSTANTS
# ============================================================

DATA_DIR = Path(__file__).parent.parent / "data"

TRADFI = ["Gold", "NASDAQ 100", "S&P 500", "US Treasury 20Y+", "US Dollar Index", "VIX"]
CRYPTO = ["Bitcoin", "Ethereum", "BNB", "Solana", "Aave (DeFi proxy)"]

SHORT = {
    "Gold": "Gold", "NASDAQ 100": "NDX", "S&P 500": "SPX",
    "US Treasury 20Y+": "TLT", "US Dollar Index": "USD", "VIX": "VIX",
    "Bitcoin": "BTC", "Ethereum": "ETH", "BNB": "BNB",
    "Solana": "SOL", "Aave (DeFi proxy)": "AAVE",
}

# Tickers for yfinance update (mirrored from src/data_loader.py)
TRADFI_TICKERS = {
    "SPY": "S&P 500", "QQQ": "NASDAQ 100", "GLD": "Gold",
    "TLT": "US Treasury 20Y+", "UUP": "US Dollar Index", "^VIX": "VIX",
}
CRYPTO_TICKERS = {
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana",
    "BNB-USD": "BNB", "AAVE-USD": "Aave (DeFi proxy)",
}

CORR_WINDOW = 60
NET_THRESHOLD = 0.3
NET_MIN_OBS = 10
GRANGER_MAX_LAG = 5
VAR_MAX_LAG = 10
VAR_MIN_LAG = 1
FEVD_HORIZON = 10
ROLL_WINDOW = 200
ROLL_STEP = 5

TRADFI_COLOR = "#4A90D9"
CRYPTO_COLOR = "#E8873A"


# ============================================================
# DATA LOADING (cached)
# ============================================================

@st.cache_data(ttl=3600)
def load_data():
    """Load returns and events from CSV files."""
    returns = pd.read_csv(DATA_DIR / "returns.csv", index_col="Date", parse_dates=True)
    events = pd.read_csv(DATA_DIR / "events.csv", index_col=0, parse_dates=["start", "end"])

    tradfi = [c for c in TRADFI if c in returns.columns]
    crypto = [c for c in CRYPTO if c in returns.columns]
    all_assets = tradfi + crypto

    return returns, events, tradfi, crypto, all_assets


def update_data_from_yfinance():
    """Download new data from yfinance, append to CSVs, clear cache."""
    import yfinance as yf

    prices = pd.read_csv(DATA_DIR / "prices.csv", index_col="Date", parse_dates=True)
    last_date = prices.index[-1]
    start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end = date.today().strftime("%Y-%m-%d")

    if start >= end:
        return 0  # nothing to update

    all_tickers = {**TRADFI_TICKERS, **CRYPTO_TICKERS}
    raw = yf.download(list(all_tickers.keys()), start=start, end=end, auto_adjust=True)

    if raw.empty or len(raw) == 0:
        return 0

    if isinstance(raw.columns, pd.MultiIndex):
        new_prices = raw["Close"]
    else:
        new_prices = raw[["Close"]]
        new_prices.columns = list(all_tickers.keys())

    new_prices.columns = [all_tickers.get(c, c) for c in new_prices.columns]

    # Reindex to match existing column order
    new_prices = new_prices.reindex(columns=prices.columns)

    # Forward-fill crypto for weekends (match data_loader logic)
    crypto_cols = [CRYPTO_TICKERS[t] for t in CRYPTO_TICKERS if CRYPTO_TICKERS[t] in new_prices.columns]
    new_prices[crypto_cols] = new_prices[crypto_cols].ffill(limit=4)

    # Append to prices
    combined_prices = pd.concat([prices, new_prices])
    combined_prices = combined_prices[~combined_prices.index.duplicated(keep="last")]
    combined_prices.sort_index(inplace=True)

    # Filter to TradFi trading calendar — drop weekends/holidays where
    # all TradFi columns are NaN (crypto trades 24/7 but we align to TradFi).
    tradfi_cols = [TRADFI_TICKERS[t] for t in TRADFI_TICKERS
                   if TRADFI_TICKERS[t] in combined_prices.columns]
    combined_prices = combined_prices.dropna(subset=tradfi_cols, how="all")

    combined_prices.to_csv(DATA_DIR / "prices.csv")

    # Recompute returns
    returns = np.log(combined_prices / combined_prices.shift(1))
    returns = returns.dropna(how="all")
    returns.to_csv(DATA_DIR / "returns.csv")

    n_new = len(new_prices)

    # Clear all cached data
    st.cache_data.clear()

    return n_new


# ============================================================
# COMPUTATION FUNCTIONS (cached)
# ============================================================

@st.cache_data
def compute_rolling_cross_correlation(returns, tradfi, crypto, window=CORR_WINDOW):
    """Compute mean |rolling correlation| between crypto and TradFi."""
    cross_series = []
    for t in tradfi:
        for c in crypto:
            rc = returns[t].rolling(window).corr(returns[c]).abs()
            cross_series.append(rc)
    cross_df = pd.concat(cross_series, axis=1)
    return cross_df.mean(axis=1).dropna()


@st.cache_data
def compute_period_corr(returns, assets, start=None, end=None, min_obs=NET_MIN_OBS):
    """Correlation matrix for a time window, filtering low-data assets."""
    if start is not None and end is not None:
        subset = returns.loc[start:end, assets]
    else:
        subset = returns[assets]
    counts = subset.count()
    valid = counts[counts >= min_obs].index.tolist()
    return subset[valid].corr(), valid


@st.cache_data
def build_network(returns, assets, start=None, end=None,
                  threshold=NET_THRESHOLD, min_obs=NET_MIN_OBS):
    """Build correlation graph + metrics for a given period."""
    corr_mat, valid_assets = compute_period_corr(returns, assets, start, end, min_obs)
    excluded = [a for a in assets if a not in valid_assets]

    G = nx.Graph()
    G.add_nodes_from(valid_assets)
    for i in range(len(valid_assets)):
        for j in range(i + 1, len(valid_assets)):
            rho = corr_mat.iloc[i, j]
            if not np.isnan(rho) and abs(rho) > threshold:
                G.add_edge(valid_assets[i], valid_assets[j],
                           weight=abs(rho), raw_corr=rho)

    deg = nx.degree_centrality(G)
    bet = nx.betweenness_centrality(G, weight="weight")

    if G.number_of_edges() > 0:
        comms = community_louvain.best_partition(G, weight="weight", random_state=42)
    else:
        comms = {n: i for i, n in enumerate(G.nodes())}

    density = nx.density(G)
    degrees = dict(G.degree())
    avg_deg = np.mean(list(degrees.values())) if degrees else 0.0
    top_bridge = max(bet, key=bet.get) if bet else None

    return {
        "G": G, "corr_mat": corr_mat, "valid_assets": valid_assets,
        "excluded": excluded, "degree_centrality": deg,
        "betweenness_centrality": bet, "communities": comms,
        "density": density, "avg_degree": avg_deg,
        "top_bridge": top_bridge, "n_communities": len(set(comms.values())),
    }


@st.cache_data
def run_granger_tests(returns, tradfi, crypto, max_lag=GRANGER_MAX_LAG):
    """Run pairwise Granger causality tests in both directions."""
    rows = []
    for crypto_asset in crypto:
        c_s = returns[crypto_asset].dropna()
        for tradfi_asset in tradfi:
            t_s = returns[tradfi_asset].dropna()
            for y_s, x_s, cause_name, effect_name, direction in [
                (t_s, c_s, SHORT[crypto_asset], SHORT[tradfi_asset], "Crypto \u2192 TradFi"),
                (c_s, t_s, SHORT[tradfi_asset], SHORT[crypto_asset], "TradFi \u2192 Crypto"),
            ]:
                combined = pd.concat([y_s, x_s], axis=1).dropna()
                if len(combined) < max_lag + 20:
                    rows.append({"cause": cause_name, "effect": effect_name,
                                 "direction": direction, "best_lag": None,
                                 "p_value": np.nan, "significant": False})
                    continue
                try:
                    result = grangercausalitytests(combined.values, maxlag=max_lag, verbose=False)
                    pvals = {lag: result[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)}
                    best_lag = min(pvals, key=pvals.get)
                    best_p = pvals[best_lag]
                except Exception:
                    best_lag, best_p = None, np.nan
                rows.append({"cause": cause_name, "effect": effect_name,
                             "direction": direction, "best_lag": best_lag,
                             "p_value": round(best_p, 4) if not np.isnan(best_p) else np.nan,
                             "significant": best_p < 0.05 if not np.isnan(best_p) else False})
    return pd.DataFrame(rows)


@st.cache_data
def compute_var_spillover(returns, all_assets, horizon=FEVD_HORIZON):
    """Fit VAR, compute FEVD, build Diebold-Yilmaz spillover table."""
    first_dates = {a: returns[a].first_valid_index() for a in all_assets}
    common_start = max(first_dates.values())
    var_data = returns.loc[common_start:, all_assets].dropna()

    model = VAR(var_data)
    aic_vals = {}
    for lag in range(VAR_MIN_LAG, VAR_MAX_LAG + 1):
        try:
            aic_vals[lag] = model.fit(lag).aic
        except Exception:
            pass

    best_lag = min(aic_vals, key=aic_vals.get)
    var_result = model.fit(best_lag)

    fevd = var_result.fevd(horizon)
    decomp = fevd.decomp
    n_vars = var_result.neqs
    if decomp.shape[0] == n_vars and decomp.shape[1] == horizon:
        last_step = decomp[:, horizon - 1, :]
    elif decomp.shape[0] == horizon and decomp.shape[1] == n_vars:
        last_step = decomp[horizon - 1]
    else:
        raise ValueError(f"Unexpected FEVD decomp shape: {decomp.shape}")

    fevd_df = pd.DataFrame(last_step, index=var_result.names, columns=var_result.names)
    n = len(fevd_df)
    pct = fevd_df * 100
    from_others = pd.Series([pct.iloc[i].sum() - pct.iloc[i, i] for i in range(n)], index=fevd_df.index)
    to_others = pd.Series([pct.iloc[:, j].sum() - pct.iloc[j, j] for j in range(n)], index=fevd_df.index)
    net = to_others - from_others
    total_spillover = from_others.sum() / n
    directional = pd.DataFrame({"FROM_others": from_others, "TO_others": to_others, "NET": net})
    return var_data, best_lag, fevd_df, directional, total_spillover


@st.cache_data
def compute_rolling_spillover(var_data, var_lag, horizon=FEVD_HORIZON,
                              window=ROLL_WINDOW, step=ROLL_STEP):
    """Compute rolling Diebold-Yilmaz total spillover index."""
    dates = var_data.index
    n = len(var_data)
    n_vars = var_data.shape[1]
    results = []
    for end_idx in range(window, n, step):
        w = var_data.iloc[end_idx - window:end_idx]
        end_date = dates[end_idx - 1]
        try:
            fit = VAR(w).fit(var_lag)
            fevd = fit.fevd(horizon)
            decomp = fevd.decomp
            if decomp.shape[0] == n_vars and decomp.shape[1] == horizon:
                last = decomp[:, horizon - 1, :]
            elif decomp.shape[0] == horizon and decomp.shape[1] == n_vars:
                last = decomp[horizon - 1]
            else:
                continue
            off_diag = last.sum() - np.trace(last)
            results.append((end_date, (off_diag / n_vars) * 100))
        except Exception:
            continue
    if not results:
        return pd.Series(dtype=float)
    return pd.Series([r[1] for r in results], index=[r[0] for r in results],
                     name="Total Spillover Index")


@st.cache_data
def compute_var_spillover_custom(returns, all_assets, start, end,
                                 horizon=FEVD_HORIZON):
    """Fit VAR on a custom date range, compute FEVD + Diebold-Yilmaz table.

    With short samples the covariance matrix can lose positive-definiteness
    at higher lags (overfitting).  We try lags in AIC order and fall back
    to lower lags until FEVD succeeds.
    """
    subset = returns.loc[start:end, all_assets].dropna()

    if len(subset) < 30:
        return None, None, None, None, None

    n_vars = len(all_assets)
    model = VAR(subset)

    # Cap max lag more aggressively: need at least (n_vars * lag + 1) < nobs
    # to keep enough degrees of freedom for a well-conditioned estimate.
    max_possible = min(VAR_MAX_LAG, max(1, (len(subset) - 1) // n_vars - 1))

    aic_vals = {}
    for lag in range(VAR_MIN_LAG, max_possible + 1):
        try:
            aic_vals[lag] = model.fit(lag).aic
        except Exception:
            pass

    if not aic_vals:
        return None, None, None, None, None

    # Try lags in AIC order; skip those where FEVD fails (non-PD covariance)
    sorted_lags = sorted(aic_vals, key=aic_vals.get)

    for best_lag in sorted_lags:
        try:
            var_result = model.fit(best_lag)
            fevd = var_result.fevd(horizon)
            decomp = fevd.decomp
            nv = var_result.neqs
            if decomp.shape[0] == nv and decomp.shape[1] == horizon:
                last_step = decomp[:, horizon - 1, :]
            elif decomp.shape[0] == horizon and decomp.shape[1] == nv:
                last_step = decomp[horizon - 1]
            else:
                continue  # unexpected shape, try next lag

            fevd_df = pd.DataFrame(last_step, index=var_result.names,
                                   columns=var_result.names)
            n = len(fevd_df)
            pct = fevd_df * 100
            from_others = pd.Series(
                [pct.iloc[i].sum() - pct.iloc[i, i] for i in range(n)],
                index=fevd_df.index)
            to_others = pd.Series(
                [pct.iloc[:, j].sum() - pct.iloc[j, j] for j in range(n)],
                index=fevd_df.index)
            net = to_others - from_others
            total_spillover = from_others.sum() / n
            directional = pd.DataFrame({
                "FROM_others": from_others, "TO_others": to_others, "NET": net})
            return subset, best_lag, fevd_df, directional, total_spillover
        except (np.linalg.LinAlgError, ValueError):
            continue  # non-PD covariance or other numerical issue, try next

    # All lags failed
    return None, None, None, None, None


@st.cache_data
def run_granger_tests_custom(returns, tradfi, crypto, start, end,
                              max_lag=GRANGER_MAX_LAG):
    """Run pairwise Granger tests on a custom date range."""
    subset = returns.loc[start:end]
    return run_granger_tests(subset, tradfi, crypto, max_lag)


@st.cache_data
def compute_findings_data(returns, events, tradfi, crypto, all_assets):
    """Compute all numbers needed for the Key Findings tab."""
    # 1. Spillover
    _, _, _, directional, total_spill = compute_var_spillover(returns, all_assets)
    dir_short = directional.copy()
    dir_short.index = [SHORT.get(n, n) for n in dir_short.index]
    top_transmitter = dir_short["TO_others"].idxmax()
    second_transmitter_val = dir_short["TO_others"].nlargest(2)
    second_transmitter = second_transmitter_val.index[1] if len(second_transmitter_val) > 1 else "N/A"

    # 2. Granger
    granger_df = run_granger_tests(returns, tradfi, crypto)
    n_sig = int(granger_df["significant"].sum())
    n_total = len(granger_df.dropna(subset=["p_value"]))
    n_c2t = int(granger_df[(granger_df["significant"]) &
                           (granger_df["direction"] == "Crypto \u2192 TradFi")].shape[0])
    n_t2c = int(granger_df[(granger_df["significant"]) &
                           (granger_df["direction"] == "TradFi \u2192 Crypto")].shape[0])

    # 3. Network densities per event + community mixing
    densities = {}
    mixed_events = []
    for event_name, row in events.iterrows():
        net = build_network(returns, all_assets, start=row["start"], end=row["end"])
        densities[event_name] = net["density"]
        # Check community mixing
        comm_groups = {}
        for node, cid in net["communities"].items():
            comm_groups.setdefault(cid, []).append(node)
        for cid, members in comm_groups.items():
            has_t = any(m in tradfi for m in members)
            has_c = any(m in crypto for m in members)
            if has_t and has_c:
                mixed_events.append(event_name)
                break

    peak_event = max(densities, key=densities.get)
    peak_density = densities[peak_event]

    return {
        "total_spill": total_spill,
        "top_transmitter": top_transmitter,
        "second_transmitter": second_transmitter,
        "n_sig": n_sig,
        "n_total": n_total,
        "n_c2t": n_c2t,
        "n_t2c": n_t2c,
        "densities": densities,
        "peak_event": peak_event,
        "peak_density": peak_density,
        "mixed_events": mixed_events,
    }


# ============================================================
# PLOTLY VISUALIZATION HELPERS
# ============================================================

def create_network_plotly(net_data, tradfi, crypto):
    """Create an interactive Plotly network graph."""
    G = net_data["G"]
    comms = net_data["communities"]
    deg = net_data["degree_centrality"]
    bet = net_data["betweenness_centrality"]

    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data for this period", showarrow=False, font=dict(size=20))
        return fig

    pos = nx.spring_layout(G, k=1.8, iterations=80, seed=42, weight="weight")

    edge_traces = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        raw = d["raw_corr"]
        w = d["weight"]
        color = "rgba(100,100,100,0.4)" if raw >= 0 else "rgba(200,60,60,0.5)"
        width = 1 + 4 * w
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line=dict(width=width, color=color), hoverinfo="text",
            text=f"{SHORT.get(u, u)} \u2014 {SHORT.get(v, v)}<br>\u03C1 = {raw:.3f}",
            showlegend=False))

    node_x, node_y, node_text, node_color, node_size, node_labels = [], [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        label = SHORT.get(node, node)
        node_labels.append(label)
        dc = deg.get(node, 0); bc = bet.get(node, 0)
        comm_id = comms.get(node, -1)
        asset_type = "TradFi" if node in tradfi else "Crypto"
        node_text.append(
            f"<b>{label}</b> ({asset_type})<br>"
            f"Degree centrality: {dc:.3f}<br>"
            f"Betweenness centrality: {bc:.3f}<br>"
            f"Community: {comm_id}")
        node_color.append(TRADFI_COLOR if node in tradfi else CRYPTO_COLOR)
        node_size.append(20 + dc * 40)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=node_size, color=node_color, line=dict(width=1.5, color="white")),
        text=node_labels, textposition="top center",
        textfont=dict(size=11, color="white"),
        hovertext=node_text, hoverinfo="text", showlegend=False)

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), height=500)
    return fig


def add_event_vlines(fig, events):
    """Add vertical event lines + annotations to a plotly figure."""
    colors = px.colors.qualitative.Set2
    for i, (name, row) in enumerate(events.iterrows()):
        mid = row["start"] + (row["end"] - row["start"]) / 2
        color = colors[i % len(colors)]
        fig.add_vrect(x0=row["start"], x1=row["end"],
                      fillcolor=color, opacity=0.12, layer="below", line_width=0)
        fig.add_annotation(x=mid, y=1.0, yref="paper", text=name, showarrow=False,
                           font=dict(size=9, color=color), textangle=-35, yanchor="bottom")


def show_explanation(text):
    """Show an expandable explanation if the toggle is on."""
    if st.session_state.get("show_explanations", True):
        with st.expander("How to read this chart"):
            st.markdown(text)


def custom_info_banner(period_ctx):
    """Show an info banner when Custom Range mode is active."""
    if period_ctx["mode"] == "custom":
        s = period_ctx["start"].strftime("%b %d, %Y")
        e = period_ctx["end"].strftime("%b %d, %Y")
        st.info(f"Showing analysis for **{s}** — **{e}**. "
                "Preset events outside this range are hidden.")


# ============================================================
# SIDEBAR HELPERS
# ============================================================

def get_period_context(events):
    """
    Read sidebar state and return the active period context.

    Returns
    -------
    dict with keys:
        mode : "preset" | "custom"
        name : str  (period label for titles)
        start : pd.Timestamp | None
        end : pd.Timestamp | None
        events_for_plots : pd.DataFrame  (events to shade on time-series charts)
    """
    mode = st.session_state.get("analysis_mode", "Preset Events")

    if mode == "Custom Range":
        s = st.session_state.get("custom_start")
        e = st.session_state.get("custom_end")
        label = st.session_state.get("custom_label", "").strip()
        if s and e:
            start = pd.Timestamp(s)
            end = pd.Timestamp(e)
            name = label if label else f"Custom ({start.strftime('%b %d, %Y')}\u2013{end.strftime('%b %d, %Y')})"
            # Only keep preset events that overlap with the custom range
            visible = events[
                (events["start"] <= end) & (events["end"] >= start)
            ]
            return {"mode": "custom", "name": name, "start": start, "end": end,
                    "events_for_plots": visible}
        # Fall through to preset if dates missing

    return {"mode": "preset", "name": None, "start": None, "end": None,
            "events_for_plots": events}


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar(returns):
    with st.sidebar:
        st.title("\U0001F310 Cross-Market Contagion")
        st.caption("Crypto \u2194 TradFi Spillover Analysis")

        st.markdown("---")

        # --- Show explanations toggle ---
        st.toggle("Show chart explanations", value=True, key="show_explanations")

        st.markdown("---")

        # --- Custom Analysis ---
        st.subheader("Custom Analysis")

        st.radio("Analysis mode", ["Preset Events", "Custom Range"],
                 key="analysis_mode", horizontal=True)

        if st.session_state.get("analysis_mode") == "Custom Range":
            min_date = returns.index[0].date()
            max_date = returns.index[-1].date()

            st.date_input("Start date", value=max_date - timedelta(days=90),
                          min_value=min_date, max_value=max_date, key="custom_start")
            st.date_input("End date", value=max_date,
                          min_value=min_date, max_value=max_date, key="custom_end")
            st.text_input("Event label (optional)", key="custom_label",
                          placeholder="e.g. My Custom Event")

            s = st.session_state.get("custom_start")
            e = st.session_state.get("custom_end")
            if s and e:
                delta = (e - s).days
                if delta < 30:
                    st.warning(f"Range is {delta} days. Minimum 30 days recommended for reliable statistics.")

        st.markdown("---")

        # --- Update Data ---
        st.subheader("Update Data")
        last_date = returns.index[-1].strftime("%b %d, %Y")
        st.caption(f"Data through: **{last_date}**")

        if st.button("Update Data", use_container_width=True):
            with st.spinner("Downloading latest prices from Yahoo Finance..."):
                try:
                    n_new = update_data_from_yfinance()
                    if n_new > 0:
                        st.success(f"Added {n_new} new trading days.")
                        st.rerun()
                    else:
                        st.info("Data is already up to date.")
                except Exception as exc:
                    st.error(f"Update failed: {exc}")

        st.markdown("---")

        # --- Project info ---
        st.markdown("""
        **Research question**: Do crypto and traditional
        financial markets merge into one interconnected
        system during crises?

        **Methods**: Rolling correlations, correlation
        networks, Granger causality, VAR, Diebold-Yilmaz
        spillover index.
        """)

        st.markdown("---")

        st.markdown("**TradFi assets**")
        for a in TRADFI:
            st.markdown(f"&nbsp;&nbsp;\U0001F535 {SHORT.get(a, a)} \u2014 {a}")
        st.markdown("**Crypto assets**")
        for a in CRYPTO:
            st.markdown(f"&nbsp;&nbsp;\U0001F7E0 {SHORT.get(a, a)} \u2014 {a}")

        st.markdown("---")

        # --- About ---
        st.subheader("About")
        st.markdown(
            "**Author**: Valerii Kulikovskyi\n"
            "[\U0001F4CE GitHub](https://github.com/bolerp/cross-market-contagion)  \n"
            "[\U0001F4C4 LinkedIn](https://www.linkedin.com/in/valerii-kulikovskyi/)"
        )
        st.caption("Built with Streamlit \u2022 Data: 2019\u20132025")


# ============================================================
# TAB 0 — KEY FINDINGS
# ============================================================

def render_tab_findings(returns, events, tradfi, crypto, all_assets):
    st.header("Cross-Market Contagion: What We Found")

    with st.spinner("Computing summary statistics..."):
        f = compute_findings_data(returns, events, tradfi, crypto, all_assets)

    # ---- The Big Picture ----
    st.subheader("The Big Picture")
    st.markdown(
        "Crypto and traditional markets are becoming increasingly interconnected. "
        f"On average, **{f['total_spill']:.1f}%** of price volatility spills over between assets in "
        "our system. During major crises, this connectedness surges \u2014 network density jumps "
        f"from a baseline level to **{f['peak_density']:.2f}** during the {f['peak_event']}. "
        "However, true cross-market contagion (where crypto and traditional assets merge into "
        "a single cluster) remains rare."
    )

    st.markdown("---")

    # ---- Three Key Discoveries ----
    st.subheader("Three Key Discoveries")

    # Metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Spillover", f"{f['total_spill']:.1f}%", help="Diebold-Yilmaz index")
    m2.metric("Significant Granger Pairs", f"{f['n_sig']}/{f['n_total']}",
              help="Pairs with p < 0.05")
    m3.metric("Peak Crisis Density", f"{f['peak_density']:.2f}",
              f"{f['peak_event']}")

    st.markdown("")

    # Finding 1
    col1, col2, col3 = st.columns(3)

    with col1:
        mixed_list = ", ".join(f["mixed_events"]) if f["mixed_events"] else "none"
        n_mixed = len(f["mixed_events"])
        st.markdown(f"""
        #### \U0001F9EC 1. Rare True Contagion

        Community detection found that crypto and traditional assets merged into
        a single cluster in only **{n_mixed}** out of 6 crisis events ({mixed_list}).
        In most crises \u2014 even when correlations spike \u2014 the two worlds stay
        in separate communities. This means high correlation \u2260 true contagion.
        """)

    with col2:
        st.markdown(f"""
        #### \U0001F4E1 2. Crypto Leads TradFi

        Of {f['n_sig']} statistically significant Granger-causal pairs,
        **{f['n_c2t']}** flow from crypto to traditional markets, while only
        **{f['n_t2c']}** go the other direction. Crypto prices carry predictive
        information about future stock, bond, and dollar movements \u2014
        not the other way around.
        """)

    with col3:
        st.markdown(f"""
        #### \U0001F3DB\uFE0F 3. BTC Is Systemically Important

        In the Diebold-Yilmaz spillover analysis, **{f['second_transmitter']}** is
        the second-largest transmitter of volatility across the whole system,
        right after **{f['top_transmitter']}** (NASDAQ). This puts Bitcoin
        on par with major equity indices in terms of systemic influence.
        """)

    st.markdown("---")

    # ---- What This Means ----
    st.subheader("What This Means")
    st.markdown(
        "For portfolio managers: holding crypto alongside traditional assets provides less "
        "diversification than widely assumed, especially during crises when correlations spike. "
        "For risk assessment: Bitcoin should be monitored as a systemically important asset \u2014 "
        "shocks originating in crypto markets now propagate into equities and currencies with "
        "statistical significance. The era of crypto as an isolated asset class is ending."
    )

    st.markdown("---")

    # ---- Event Timeline ----
    st.subheader("Event Timeline: Network Density Across Crises")

    baseline_net = build_network(returns, all_assets)
    baseline_density = baseline_net["density"]

    timeline_events = list(events.index)
    timeline_densities = [f["densities"][e] for e in timeline_events]
    timeline_dates = [events.loc[e, "start"] for e in timeline_events]

    fig_tl = go.Figure()

    # Baseline reference line
    fig_tl.add_hline(y=baseline_density, line_dash="dash", line_color="gray",
                     annotation_text=f"Baseline: {baseline_density:.2f}",
                     annotation_position="bottom left")

    # Event markers
    fig_tl.add_trace(go.Scatter(
        x=timeline_dates, y=timeline_densities,
        mode="markers+lines+text",
        marker=dict(size=16, color=[
            "#d62728" if d > baseline_density + 0.15 else "#ff7f0e" if d > baseline_density else "#2ca02c"
            for d in timeline_densities
        ], line=dict(width=1.5, color="white")),
        text=[f"{d:.2f}" for d in timeline_densities],
        textposition="top center",
        textfont=dict(size=11),
        hovertext=[f"{name}<br>Density: {d:.2f}" for name, d in zip(timeline_events, timeline_densities)],
        hoverinfo="text",
        showlegend=False,
    ))

    # Event name labels below
    for dt, name in zip(timeline_dates, timeline_events):
        fig_tl.add_annotation(x=dt, y=-0.08, yref="paper",
                              text=name, showarrow=False,
                              font=dict(size=9), textangle=-30)

    fig_tl.update_layout(
        yaxis_title="Graph Density",
        xaxis_title="",
        height=320,
        margin=dict(l=50, r=20, t=20, b=80),
    )
    st.plotly_chart(fig_tl, use_container_width=True)

    show_explanation(
        "Each dot represents a crisis event. The y-axis shows **graph density** \u2014 "
        "the fraction of all possible connections between assets that actually exist. "
        "Higher density means assets moved more in sync during that crisis. "
        "Red dots indicate the biggest spikes in market interconnectedness. "
        "The dashed line shows the baseline (full-period) density for comparison."
    )


# ============================================================
# TAB 1 — NETWORK EVOLUTION
# ============================================================

def render_tab_network(returns, events, tradfi, crypto, all_assets, period_ctx):
    st.header("Network Evolution")
    custom_info_banner(period_ctx)
    st.caption(
        "Correlation-based networks for each period. "
        f"Edge exists if |\u03C1| > {NET_THRESHOLD}. "
        "Node size \u221D degree centrality."
    )

    # Period selector
    if period_ctx["mode"] == "custom":
        # Custom range active — show it as the selected option
        net = build_network(returns, all_assets,
                            start=period_ctx["start"], end=period_ctx["end"])
        label = period_ctx["name"]

        days = (period_ctx["end"] - period_ctx["start"]).days
        if days < 30:
            st.warning(f"Custom range is only {days} days. Results may be unreliable.")
    else:
        period_options = ["Baseline (Full Period)"] + list(events.index)
        period = st.selectbox("Select period", period_options, key="net_period")

        if period == "Baseline (Full Period)":
            net = build_network(returns, all_assets)
            label = "Full Period"
        else:
            row = events.loc[period]
            net = build_network(returns, all_assets, start=row["start"], end=row["end"])
            label = f"{period} ({row['start'].strftime('%b %Y')}\u2013{row['end'].strftime('%b %Y')})"

    col_graph, col_metrics = st.columns([3, 2])

    with col_graph:
        fig = create_network_plotly(net, tradfi, crypto)
        fig.update_layout(title=dict(
            text=f"{label} &nbsp;|&nbsp; density = {net['density']:.2f}",
            font=dict(size=14)))
        st.plotly_chart(fig, use_container_width=True)

        if net["excluded"]:
            st.info(f"Excluded (insufficient data): "
                    f"{', '.join(SHORT.get(e, e) for e in net['excluded'])}")

        show_explanation(
            "Each circle is a financial asset. **Blue = traditional** (stocks, bonds, gold), "
            "**orange = crypto**. Lines connect assets that moved together (correlation above "
            f"{NET_THRESHOLD}). Thicker lines = stronger connection. Bigger circles = more "
            "connections. Hover over any circle to see its exact scores. If you see blue and "
            "orange circles connected by many thick lines, that means the two markets were "
            "highly intertwined during this period."
        )

    with col_metrics:
        st.subheader("Graph Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Density", f"{net['density']:.2f}")
        m2.metric("Communities", net["n_communities"])
        m3.metric("Avg Degree", f"{net['avg_degree']:.1f}")

        if net["top_bridge"]:
            st.metric("Top Bridge (betweenness)",
                      SHORT.get(net["top_bridge"], net["top_bridge"]),
                      f"score = {net['betweenness_centrality'][net['top_bridge']]:.3f}")

        st.subheader("Centrality Metrics")
        cent_rows = []
        for node in net["valid_assets"]:
            label_n = SHORT.get(node, node)
            asset_type = "TradFi" if node in tradfi else "Crypto"
            cent_rows.append({
                "Asset": label_n, "Type": asset_type,
                "Degree": round(net["degree_centrality"].get(node, 0), 3),
                "Betweenness": round(net["betweenness_centrality"].get(node, 0), 3),
                "Community": net["communities"].get(node, -1)})
        cent_df = pd.DataFrame(cent_rows).sort_values("Betweenness", ascending=False)
        st.dataframe(cent_df, use_container_width=True, hide_index=True)


# ============================================================
# TAB 2 — CORRELATION DYNAMICS
# ============================================================

def render_tab_correlation(returns, events, tradfi, crypto, all_assets, period_ctx):
    st.header("Correlation Dynamics")
    custom_info_banner(period_ctx)

    # --- Rolling cross-market correlation ---
    st.subheader(f"Mean |Cross-Market Correlation| (rolling {CORR_WINDOW}-day)")
    avg_corr = compute_rolling_cross_correlation(returns, tradfi, crypto)

    # In custom mode: crop to [start - 90d, end + 30d] to account for 60-day window
    if period_ctx["mode"] == "custom":
        view_start = period_ctx["start"] - pd.Timedelta(days=90)
        view_end = period_ctx["end"] + pd.Timedelta(days=30)
        avg_corr_plot = avg_corr.loc[view_start:view_end]
    else:
        avg_corr_plot = avg_corr

    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(
        x=avg_corr_plot.index, y=avg_corr_plot.values,
        mode="lines", line=dict(color="#1f77b4", width=1.5),
        fill="tozeroy", fillcolor="rgba(31,119,180,0.1)",
        name="Mean |cross-corr|"))
    add_event_vlines(fig_corr, period_ctx["events_for_plots"])

    # Highlight custom range
    if period_ctx["mode"] == "custom":
        fig_corr.add_vrect(x0=period_ctx["start"], x1=period_ctx["end"],
                           fillcolor="yellow", opacity=0.15, layer="below",
                           line=dict(color="yellow", width=2, dash="dash"))
        fig_corr.add_annotation(
            x=period_ctx["start"] + (period_ctx["end"] - period_ctx["start"]) / 2,
            y=1.05, yref="paper", text=f"\u2B50 {period_ctx['name']}", showarrow=False,
            font=dict(size=10, color="yellow"))

    fig_corr.update_layout(
        yaxis_title="Mean |Correlation|", xaxis_title="Date",
        height=350, margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig_corr, use_container_width=True)

    show_explanation(
        "This line shows how connected crypto and traditional markets are over time. "
        "When the line goes up, the two markets are moving more in sync. "
        "Colored bands mark crisis events. Look for spikes during crises \u2014 that suggests "
        "the shock spread between markets (contagion)."
    )

    # --- Heatmap for selected period ---
    st.subheader("Correlation Heatmap")

    if period_ctx["mode"] == "custom":
        # Heatmap computed ONLY for the custom range
        corr_mat, valid = compute_period_corr(
            returns, all_assets, start=period_ctx["start"], end=period_ctx["end"])
        heat_label = period_ctx["name"]
    else:
        period_options = ["Full Period"] + list(events.index)
        period = st.selectbox("Select period", period_options, key="corr_period")
        if period == "Full Period":
            corr_mat, valid = compute_period_corr(returns, all_assets)
        else:
            row = events.loc[period]
            corr_mat, valid = compute_period_corr(
                returns, all_assets, start=row["start"], end=row["end"])
        heat_label = period

    labels = [SHORT.get(a, a) for a in valid]
    corr_vals = corr_mat.values

    fig_heat = go.Figure(data=go.Heatmap(
        z=corr_vals, x=labels, y=labels,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=np.round(corr_vals, 2), texttemplate="%{text}",
        textfont=dict(size=9), colorbar=dict(title="\u03C1")))
    fig_heat.update_layout(
        title=dict(text=heat_label, font=dict(size=13)),
        height=500, margin=dict(l=60, r=20, t=40, b=60),
        yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_heat, use_container_width=True)

    show_explanation(
        "Each cell shows how closely two assets move together. "
        "**Red = move in the same direction** (positive correlation), "
        "**blue = move in opposite directions** (negative correlation), "
        "**white = no relationship**. Numbers closer to +1 or \u22121 mean stronger connection. "
        "Look for red cells between blue-group (TradFi) and orange-group (crypto) assets \u2014 "
        "those indicate cross-market dependence."
    )


# ============================================================
# TAB 3 — SPILLOVER ANALYSIS
# ============================================================

def render_tab_spillover(returns, events, all_assets, tradfi, crypto, period_ctx):
    st.header("Spillover Analysis")
    custom_info_banner(period_ctx)

    # In custom mode: fit VAR on custom range only for bar charts
    if period_ctx["mode"] == "custom":
        with st.spinner("Fitting VAR model on custom range..."):
            result = compute_var_spillover_custom(
                returns, all_assets, period_ctx["start"], period_ctx["end"])
            cust_var_data, cust_lag, cust_fevd_df, cust_dir, cust_total = result

        if cust_total is None:
            st.error("Not enough data in the custom range to fit a VAR model (minimum 30 days).")
            return

        best_lag_display = cust_lag
        total_spill_display = cust_total
        directional_display = cust_dir
    else:
        cust_var_data = None  # not used

    # Always compute full-period VAR (needed for rolling spillover baseline)
    with st.spinner("Fitting VAR model and computing spillover index..."):
        var_data, best_lag, fevd_df, directional, total_spill = compute_var_spillover(
            returns, all_assets)

    if period_ctx["mode"] != "custom":
        best_lag_display = best_lag
        total_spill_display = total_spill
        directional_display = directional

    st.caption(
        f"VAR({best_lag_display}) \u2022 FEVD horizon = {FEVD_HORIZON} days "
        f"\u2022 Total Spillover Index = **{total_spill_display:.1f}%**")

    # --- Rolling spillover ---
    st.subheader(f"Rolling Total Spillover ({ROLL_WINDOW}-day window)")
    with st.spinner("Computing rolling spillover (~180 VAR fits)..."):
        roll_spill = compute_rolling_spillover(var_data, best_lag)

    if len(roll_spill) > 0:
        # In custom mode: crop to [start - 230d, end + 30d] to account for 200-day window
        if period_ctx["mode"] == "custom":
            view_start = period_ctx["start"] - pd.Timedelta(days=230)
            view_end = period_ctx["end"] + pd.Timedelta(days=30)
            roll_plot = roll_spill.loc[view_start:view_end]
        else:
            roll_plot = roll_spill

        if len(roll_plot) > 0:
            mean_val = roll_plot.mean()
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(
                x=roll_plot.index, y=roll_plot.values,
                mode="lines", line=dict(color="#1f77b4", width=1.5),
                fill="tozeroy", fillcolor="rgba(31,119,180,0.1)",
                name="Total Spillover"))
            fig_roll.add_hline(y=mean_val, line_dash="dash", line_color="red",
                               annotation_text=f"Mean = {mean_val:.1f}%",
                               annotation_position="bottom right")
            add_event_vlines(fig_roll, period_ctx["events_for_plots"])

            if period_ctx["mode"] == "custom":
                fig_roll.add_vrect(x0=period_ctx["start"], x1=period_ctx["end"],
                                   fillcolor="yellow", opacity=0.15, layer="below",
                                   line=dict(color="yellow", width=2, dash="dash"))

            fig_roll.update_layout(
                yaxis_title="Total Spillover Index (%)", xaxis_title="Date",
                height=350, margin=dict(l=50, r=20, t=30, b=40))
            st.plotly_chart(fig_roll, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean", f"{roll_plot.mean():.1f}%")
            c2.metric("Std", f"{roll_plot.std():.1f}%")
            c3.metric("Min", f"{roll_plot.min():.1f}%", f"{roll_plot.idxmin().strftime('%b %Y')}")
            c4.metric("Max", f"{roll_plot.max():.1f}%", f"{roll_plot.idxmax().strftime('%b %Y')}")
        else:
            st.warning("No rolling spillover data available for the selected time window. "
                       "The 200-day rolling window requires data starting ~230 days before your range.")

        show_explanation(
            "This line tracks how much **volatility leaks** between all the assets in our system over time. "
            "Higher values mean that a shock to one asset spreads more to others. "
            "The red dashed line is the average. When the blue line rises above it during a colored band "
            "(a crisis), that crisis made markets more interconnected than usual."
        )

    # --- Directional spillovers bar chart ---
    st.subheader("Directional Spillovers: TO vs FROM")

    dir_df = directional_display.copy()
    dir_df.index = [SHORT.get(n, n) for n in dir_df.index]
    dir_df = dir_df.sort_values("NET", ascending=True)

    crypto_short = {SHORT[c] for c in crypto}

    col_to, col_from, col_net = st.columns(3)

    with col_to:
        sorted_to = dir_df.sort_values("TO_others", ascending=True)
        colors_to = [CRYPTO_COLOR if n in crypto_short else TRADFI_COLOR for n in sorted_to.index]
        fig_to = go.Figure(go.Bar(y=sorted_to.index, x=sorted_to["TO_others"],
                                  orientation="h", marker_color=colors_to))
        fig_to.update_layout(title="Transmits TO others", xaxis_title="%", height=400,
                             margin=dict(l=60, r=10, t=40, b=30))
        st.plotly_chart(fig_to, use_container_width=True)

    with col_from:
        sorted_from = dir_df.sort_values("FROM_others", ascending=True)
        colors_from = [CRYPTO_COLOR if n in crypto_short else TRADFI_COLOR for n in sorted_from.index]
        fig_from = go.Figure(go.Bar(y=sorted_from.index, x=sorted_from["FROM_others"],
                                    orientation="h", marker_color=colors_from))
        fig_from.update_layout(title="Receives FROM others", xaxis_title="%", height=400,
                               margin=dict(l=60, r=10, t=40, b=30))
        st.plotly_chart(fig_from, use_container_width=True)

    with col_net:
        net_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in dir_df["NET"]]
        fig_net = go.Figure(go.Bar(y=dir_df.index, x=dir_df["NET"],
                                   orientation="h", marker_color=net_colors))
        fig_net.add_vline(x=0, line_color="white", line_width=1)
        fig_net.update_layout(title="NET (+ transmitter, \u2013 receiver)", xaxis_title="%",
                              height=400, margin=dict(l=60, r=10, t=40, b=30))
        st.plotly_chart(fig_net, use_container_width=True)

    show_explanation(
        "These three charts show who **sends** and who **receives** volatility. "
        "**Left**: which assets spread the most turbulence to others. "
        "**Center**: which assets absorb the most turbulence from others. "
        "**Right**: the net balance \u2014 green bars are net **transmitters** (sources of shocks), "
        "red bars are net **receivers** (absorbers). Blue = traditional assets, orange = crypto."
    )


# ============================================================
# TAB 4 — GRANGER CAUSALITY
# ============================================================

def render_tab_granger(returns, tradfi, crypto, period_ctx):
    st.header("Granger Causality")
    custom_info_banner(period_ctx)
    st.caption(
        "Pairwise Granger causality tests (lags 1\u20135). "
        "Color = p-value. Green = significant at 5%. "
        "Granger causality \u2260 true causality \u2014 it measures predictive content.")

    if period_ctx["mode"] == "custom":
        # Check data length in custom range
        custom_len = len(returns.loc[period_ctx["start"]:period_ctx["end"]].dropna(how="all"))
        if custom_len < 100:
            st.warning(
                f"\u26A0\uFE0F Custom range contains only **{custom_len} trading days**. "
                "Granger causality results may be unreliable with fewer than 100 observations. "
                "Consider expanding the date range for more robust results."
            )
        with st.spinner("Running Granger causality tests on custom range..."):
            granger_df = run_granger_tests_custom(
                returns, tradfi, crypto, period_ctx["start"], period_ctx["end"])
    else:
        with st.spinner("Running Granger causality tests (60 pairs)..."):
            granger_df = run_granger_tests(returns, tradfi, crypto)

    n_sig = granger_df["significant"].sum()
    n_total = len(granger_df.dropna(subset=["p_value"]))
    st.info(f"Significant pairs: **{n_sig}** / {n_total} ({100*n_sig/n_total:.0f}%) at 5% level")

    crypto_labels = [SHORT[c] for c in crypto]
    tradfi_labels = [SHORT[t] for t in tradfi]

    col1, col2 = st.columns(2)

    for col, direction, title in [
        (col1, "Crypto \u2192 TradFi", "Crypto \u2192 TradFi"),
        (col2, "TradFi \u2192 Crypto", "TradFi \u2192 Crypto"),
    ]:
        with col:
            subset = granger_df[granger_df["direction"] == direction]
            if direction == "Crypto \u2192 TradFi":
                piv = subset.pivot(index="cause", columns="effect", values="p_value")
                sig_piv = subset.pivot(index="cause", columns="effect", values="significant")
            else:
                piv = subset.pivot(index="effect", columns="cause", values="p_value")
                sig_piv = subset.pivot(index="effect", columns="cause", values="significant")

            piv = piv.reindex(index=crypto_labels, columns=tradfi_labels)
            sig_piv = sig_piv.reindex(index=crypto_labels, columns=tradfi_labels)

            annot = piv.copy().astype(object)
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    v = piv.iloc[i, j]
                    s = sig_piv.iloc[i, j]
                    if pd.isna(v):
                        annot.iloc[i, j] = "\u2014"
                    else:
                        star = " *" if s else ""
                        annot.iloc[i, j] = f"{v:.3f}{star}"

            fig_g = go.Figure(data=go.Heatmap(
                z=piv.values, x=tradfi_labels, y=crypto_labels,
                colorscale=[[0, "#2ca02c"], [0.05, "#98df8a"],
                            [0.1, "#ffdd57"], [0.2, "#ff7f0e"], [1.0, "#d62728"]],
                zmin=0, zmax=0.2,
                text=annot.values, texttemplate="%{text}", textfont=dict(size=11),
                colorbar=dict(title="p-value", len=0.6),
                hovertemplate="Cause: %{y}<br>Effect: %{x}<br>p-value: %{z:.4f}<extra></extra>"))
            fig_g.update_layout(
                title=dict(text=f"{title}<br><sub>* = significant at 5%</sub>", font=dict(size=13)),
                yaxis_title="Crypto", xaxis_title="TradFi",
                height=350, margin=dict(l=60, r=20, t=60, b=60),
                yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_g, use_container_width=True)

    show_explanation(
        "These heatmaps test whether one asset's past prices help predict another's future prices. "
        "**Green cells = statistically significant** (the past of one genuinely helps predict the other). "
        "**Red cells = not significant** (no predictive relationship found). "
        "The left chart checks if crypto predicts traditional assets; the right checks the reverse. "
        "More green on the left than the right means crypto leads traditional markets."
    )

    with st.expander("View all significant pairs"):
        sig = granger_df[granger_df["significant"]].sort_values("p_value")
        if len(sig) > 0:
            st.dataframe(sig[["cause", "effect", "direction", "best_lag", "p_value"]],
                         use_container_width=True, hide_index=True)
        else:
            st.write("No significant pairs found.")


# ============================================================
# MAIN
# ============================================================

def main():
    returns, events, tradfi, crypto, all_assets = load_data()

    render_sidebar(returns)
    period_ctx = get_period_context(events)

    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "\U0001F4CA Key Findings",
        "\U0001F578\uFE0F Network Evolution",
        "\U0001F4C8 Correlation Dynamics",
        "\U0001F4C9 Spillover Analysis",
        "\U0001F50D Granger Causality",
    ])

    with tab0:
        render_tab_findings(returns, events, tradfi, crypto, all_assets)

    with tab1:
        render_tab_network(returns, events, tradfi, crypto, all_assets, period_ctx)

    with tab2:
        render_tab_correlation(returns, events, tradfi, crypto, all_assets, period_ctx)

    with tab3:
        render_tab_spillover(returns, events, all_assets, tradfi, crypto, period_ctx)

    with tab4:
        render_tab_granger(returns, tradfi, crypto, period_ctx)


if __name__ == "__main__":
    main()
