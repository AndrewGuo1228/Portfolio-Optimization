"""
regime_builder.py
-----------------
Build regime_drift_stats and regime_lab_ticker_summary from OHLCV data.
Replaces the IB-dependent run_daily_risk.py workflow for backtest use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from regime.regime_models import run_regime_model_hybrid_v2, resolve_mixed_regimes
from regime.regime_calibration import _smooth_regime_labels_asymmetric

SMOOTH_WINDOW    = 5
DOWNTREND_WINDOW = 1
MIN_SEGMENT_DAYS = 7


def _strip_suffix(label: str) -> str:
    for base in ("UPTREND", "DOWNTREND", "RANGE"):
        if str(label).upper().startswith(base):
            return base
    return "MIXED"


def _merge_short_segments(labels: list, min_days: int) -> list:
    """Merge segments shorter than min_days into the preceding segment."""
    filtered = labels.copy()
    i = 0
    while i < len(labels):
        base = labels[i]
        j = i
        while j < len(labels) and labels[j] == base:
            j += 1
        if (j - i) < min_days and i > 0:
            prev = filtered[i - 1]
            for k in range(i, j):
                filtered[k] = prev
        i = j
    return filtered


def build_regime_drift_stats(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    smooth_window: int = SMOOTH_WINDOW,
    downtrend_window: int = DOWNTREND_WINDOW,
    min_segment_days: int = MIN_SEGMENT_DAYS,
) -> pd.DataFrame:
    """
    Run hybrid_v2 regime detection on each ticker's OHLCV,
    compute annualised conditional returns per regime.

    Returns
    -------
    pd.DataFrame  columns: ticker, regime, annualized_return, days
    Matches the schema of reports/regime_drift_stats.csv.
    """
    rows = []
    for ticker, ohlcv in ohlcv_by_ticker.items():
        try:
            feat, _ = run_regime_model_hybrid_v2(ohlcv)   # returns (df, dict)
            feat = resolve_mixed_regimes(ohlcv, feat)

            raw_labels = feat["drift_regime_label"].tolist() \
                if "drift_regime_label" in feat.columns \
                else feat["drift_regime"].tolist()

            smoothed   = _smooth_regime_labels_asymmetric(
                raw_labels, window=smooth_window, downtrend_window=downtrend_window,
            )
            base_labels = [_strip_suffix(l) for l in smoothed]
            filtered    = _merge_short_segments(base_labels, min_segment_days)

            close       = pd.to_numeric(feat["close"], errors="coerce")
            daily_rets  = close.pct_change()
            base_series = pd.Series(filtered, index=feat.index)

            for regime in ("UPTREND", "RANGE", "DOWNTREND"):
                mask   = (base_series == regime)
                ret_s  = daily_rets[mask].dropna()
                ann    = float(ret_s.mean() * 252) if len(ret_s) >= 20 else None
                rows.append({
                    "ticker":            ticker.upper(),
                    "regime":            regime,
                    "annualized_return": ann,
                    "days":              int(mask.sum()),
                })
        except Exception as e:
            print(f"[regime_builder] {ticker} drift stats failed: {e}")

    return pd.DataFrame(rows)


def build_regime_summary(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    as_of_date: str | None = None,
) -> dict:
    """
    Build the equivalent of regime_lab_ticker_summary.json for a given date.

    Returns
    -------
    dict  {ticker: {current_drift_regime_base, days_in_current_regime, ...}}
    """
    summary: dict = {}
    for ticker, ohlcv in ohlcv_by_ticker.items():
        try:
            df = ohlcv.copy()
            if as_of_date:
                df = df[df.index <= pd.Timestamp(as_of_date)]
            if df.empty:
                continue

            feat, _ = run_regime_model_hybrid_v2(df)
            feat = resolve_mixed_regimes(df, feat)

            raw_labels = feat["drift_regime_label"].tolist() \
                if "drift_regime_label" in feat.columns \
                else feat["drift_regime"].tolist()

            smoothed = _smooth_regime_labels_asymmetric(
                raw_labels, window=SMOOTH_WINDOW, downtrend_window=DOWNTREND_WINDOW,
            )

            today_base = _strip_suffix(smoothed[-1])
            prev_base  = _strip_suffix(smoothed[-2]) if len(smoothed) >= 2 else today_base

            count = 0
            for lbl in reversed(smoothed):
                if _strip_suffix(lbl) == today_base:
                    count += 1
                else:
                    break

            p_trend_2w = float(feat["p_trend_2w"].iloc[-1]) \
                if "p_trend_2w" in feat.columns else 0.5

            # iv_percentile: backtest default = 50 (neutral, no RSI/IV discount)
            iv_pct = 50.0

            summary[ticker.upper()] = {
                "current_drift_regime_base":    today_base,
                "prev_drift_regime_base":        prev_base,
                "days_in_current_regime":        count,
                "current_trend_probability_2w":  p_trend_2w,
                "trend_mode": "theta" if p_trend_2w < 0.40 else "price",
                "is_buffer_day":                 False,
                "transition_progress":           min(1.0, count / 5),
                "iv_percentile":                 iv_pct,
                "theta_eligible":                p_trend_2w < 0.40,
            }
        except Exception as e:
            print(f"[regime_builder] {ticker} summary failed: {e}")

    return summary
