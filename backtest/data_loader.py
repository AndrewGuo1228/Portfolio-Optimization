"""
data_loader.py
--------------
Load historical price data from Data/ directory.
Builds the inputs that optimizer.py and regime detection need.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "Data"


def load_price_returns(
    tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Build a daily log-return wide DataFrame (columns=tickers, index=date).
    Tries individual {TICKER}.csv first; falls back to merged_close.csv.

    Returns
    -------
    pd.DataFrame  — DatetimeIndex, columns = tickers, values = pct_change returns
    """
    frames: dict[str, pd.Series] = {}

    # --- Try individual files first ---
    for ticker in tickers:
        path = DATA_DIR / f"{ticker}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.lower()
            date_col  = next((c for c in df.columns if c in ("date", "time", "timestamp")), None)
            close_col = next((c for c in df.columns if c in ("close", "adj close", "adjclose")), None)
            if date_col is None or close_col is None:
                print(f"[data_loader] {ticker}.csv: cannot identify date/close columns, skipping")
                continue
            s = df.set_index(date_col)[close_col].rename(ticker)
            s.index = pd.to_datetime(s.index, errors="coerce")
            s = pd.to_numeric(s, errors="coerce")
            frames[ticker] = s
        except Exception as e:
            print(f"[data_loader] {ticker}.csv read error: {e}")

    # --- Fallback: read from merged_close.csv for any missing tickers ---
    missing = [t for t in tickers if t not in frames]
    if missing:
        merged_path = DATA_DIR / "merged_close.csv"
        if merged_path.exists():
            try:
                mc = pd.read_csv(merged_path, index_col=0, parse_dates=True)
                mc.index = pd.to_datetime(mc.index, errors="coerce")
                for ticker in missing:
                    if ticker in mc.columns:
                        frames[ticker] = pd.to_numeric(mc[ticker], errors="coerce").rename(ticker)
                    else:
                        print(f"[data_loader] {ticker} not found in merged_close.csv either, skipping")
            except Exception as e:
                print(f"[data_loader] merged_close.csv read error: {e}")

    if not frames:
        raise ValueError(f"No price data found for any of: {tickers}")

    price_df = pd.DataFrame(frames).sort_index()
    price_df = price_df[price_df.index.notna()]

    if start_date:
        price_df = price_df[price_df.index >= pd.Timestamp(start_date)]
    if end_date:
        price_df = price_df[price_df.index <= pd.Timestamp(end_date)]

    returns_df = price_df.pct_change().dropna(how="all")
    return returns_df


def load_ohlcv(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load OHLCV for a single ticker. Returns DataFrame with DatetimeIndex
    and lowercase columns: open, high, low, close, volume.
    Used by regime detection (build_features requires this schema).
    """
    path = DATA_DIR / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"[data_loader] {path} not found")

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()

    date_col = next((c for c in df.columns if c in ("date", "time", "timestamp")), None)
    if date_col:
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date")

    df = df.sort_index()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]

    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]

    # Ensure numeric
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_betas(
    returns_df: pd.DataFrame,
    benchmark: str = "SPY",
    window: int = 252,
) -> pd.DataFrame:
    """
    Compute each ticker's beta vs benchmark using full-history OLS.
    Returns DataFrame with columns: symbol, beta.
    """
    if benchmark not in returns_df.columns:
        print(f"[data_loader] benchmark {benchmark} not in returns_df, using beta=1.0 for all")
        return pd.DataFrame([
            {"symbol": t, "beta": 1.0} for t in returns_df.columns
        ])

    bench = returns_df[benchmark].dropna()
    betas: dict[str, float] = {}

    for ticker in returns_df.columns:
        if ticker == benchmark:
            betas[ticker] = 1.0
            continue
        asset   = returns_df[ticker].dropna()
        aligned = pd.concat([asset, bench], axis=1).dropna().tail(window)
        if len(aligned) < 30:
            betas[ticker] = 1.0
            continue
        cov = aligned.cov()
        beta = cov.iloc[0, 1] / max(cov.iloc[1, 1], 1e-12)
        betas[ticker] = float(beta)

    return pd.DataFrame([{"symbol": t, "beta": b} for t, b in betas.items()])
