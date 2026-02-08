"""
Data loader for Cross-Market Contagion project.
Downloads daily price data for TradFi and Crypto assets,
synchronizes dates, and computes log returns.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================
# ASSET CONFIGURATION
# ============================================================

TRADFI_TICKERS = {
    "SPY": "S&P 500",
    "QQQ": "NASDAQ 100",
    "GLD": "Gold",
    "TLT": "US Treasury 20Y+",
    "UUP": "US Dollar Index",
    "^VIX": "VIX",
}

CRYPTO_TICKERS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "BNB-USD": "BNB",
    "AAVE-USD": "Aave (DeFi proxy)",
}

# Full date range
START_DATE = "2019-01-01"
END_DATE = "2025-02-01"

# Key crisis events for analysis
EVENTS = {
    "COVID Crash": ("2020-02-20", "2020-04-15"),
    "China Crypto Ban": ("2021-05-10", "2021-06-30"),
    "Luna/Terra Collapse": ("2022-05-01", "2022-06-15"),
    "FTX Collapse": ("2022-11-01", "2022-12-15"),
    "SVB Crisis": ("2023-03-01", "2023-04-15"),
    "BTC ETF Approval": ("2024-01-01", "2024-02-15"),
}

DATA_DIR = Path(__file__).parent.parent / "data"


def download_prices(tickers: dict, start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for a dict of tickers."""
    all_tickers = list(tickers.keys())
    print(f"Downloading: {', '.join(tickers.values())}")
    
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True)
    
    # yf.download returns MultiIndex columns when multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = all_tickers
    
    # Rename columns to readable names
    prices.columns = [tickers.get(col, col) for col in prices.columns]
    
    return prices


def synchronize_dates(tradfi: pd.DataFrame, crypto: pd.DataFrame) -> pd.DataFrame:
    """
    Synchronize TradFi and Crypto data.
    
    Strategy: use TradFi trading days as the base calendar.
    For crypto, forward-fill weekends/holidays (crypto trades 24/7 
    but we want aligned dates for correlation analysis).
    """
    # Crypto: resample to only include dates where TradFi traded
    combined = pd.concat([tradfi, crypto], axis=1)
    
    # Forward fill crypto data for weekends/holidays (max 4 days gap)
    crypto_cols = crypto.columns.tolist()
    combined[crypto_cols] = combined[crypto_cols].ffill(limit=4)
    
    # Keep only rows where TradFi data exists (trading days)
    tradfi_cols = tradfi.columns.tolist()
    combined = combined.dropna(subset=tradfi_cols, how="all")
    
    return combined


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Compute returns from price data."""
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return returns.dropna(how="all")


def get_event_data(returns: pd.DataFrame, event_name: str) -> pd.DataFrame:
    """Extract returns for a specific event window."""
    if event_name not in EVENTS:
        raise ValueError(f"Unknown event: {event_name}. Choose from: {list(EVENTS.keys())}")
    
    start, end = EVENTS[event_name]
    return returns.loc[start:end]


def print_data_summary(prices: pd.DataFrame, returns: pd.DataFrame):
    """Print summary statistics for quick sanity check."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Trading days: {len(prices)}")
    print(f"Assets: {len(prices.columns)}")
    print(f"\nMissing data (% of total):")
    missing = (prices.isna().sum() / len(prices) * 100).round(1)
    for asset, pct in missing.items():
        status = "⚠️" if pct > 5 else "✓"
        print(f"  {status} {asset}: {pct}%")
    
    print(f"\nAnnualized volatility (from returns):")
    annual_vol = (returns.std() * np.sqrt(252) * 100).round(1)
    for asset, vol in annual_vol.items():
        print(f"  {asset}: {vol}%")


def main():
    """Main pipeline: download, sync, compute returns, save."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Step 1: Downloading TradFi data...")
    tradfi_prices = download_prices(TRADFI_TICKERS, START_DATE, END_DATE)
    
    print("\nStep 2: Downloading Crypto data...")
    crypto_prices = download_prices(CRYPTO_TICKERS, START_DATE, END_DATE)
    
    print("\nStep 3: Synchronizing dates...")
    all_prices = synchronize_dates(tradfi_prices, crypto_prices)
    
    print(f"\nStep 4: Computing log returns...")
    all_returns = compute_returns(all_prices, method="log")
    
    # Save
    all_prices.to_csv(DATA_DIR / "prices.csv")
    all_returns.to_csv(DATA_DIR / "returns.csv")
    
    # Also save events metadata
    events_df = pd.DataFrame(EVENTS, index=["start", "end"]).T
    events_df.to_csv(DATA_DIR / "events.csv")
    
    print(f"\nSaved to {DATA_DIR}/")
    print(f"  - prices.csv ({all_prices.shape})")
    print(f"  - returns.csv ({all_returns.shape})")
    print(f"  - events.csv")
    
    print_data_summary(all_prices, all_returns)
    
    return all_prices, all_returns


if __name__ == "__main__":
    prices, returns = main()
