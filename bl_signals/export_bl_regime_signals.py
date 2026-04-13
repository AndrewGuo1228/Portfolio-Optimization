#!/usr/bin/env python3
"""
Export IV/RV regime history for downstream Black-Litterman research.

This script lives in Portfolio-Optimization and imports the main risk model as
an external dependency. It does not modify or copy code into the main project.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
PORTFOLIO_OPT_ROOT = THIS_FILE.parents[1]
DEFAULT_MAIN_MODEL_ROOT = PORTFOLIO_OPT_ROOT.parent / "Portfolio Risk Profile"
DEFAULT_OUT_DIR = PORTFOLIO_OPT_ROOT / "bl_signals"

STATES = ("deep_green", "light_green", "warning")
REQUIRED_HISTORY_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume", "impliedVol"]


@dataclass(frozen=True)
class BlueStats:
    count: int
    avg_30d: float | None
    hit_30d: float | None
    best_30d: float | None
    worst_30d: float | None
    avg_60d: float | None
    hit_60d: float | None
    best_60d: float | None
    worst_60d: float | None


def _install_main_model_path(main_model_root: Path) -> None:
    root = main_model_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Main model root not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _parse_symbols(raw: str | None, input_df: pd.DataFrame | None = None) -> list[str]:
    if raw:
        return sorted({s.strip().upper() for s in raw.split(",") if s.strip()})
    if input_df is not None and "symbol" in input_df.columns:
        return sorted(input_df["symbol"].dropna().astype(str).str.upper().unique().tolist())
    return []


def _load_history_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ["symbol", "date", "close"] if c not in df.columns]
    if missing:
        raise ValueError(f"Input history CSV missing required columns: {missing}")

    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "symbol"]).sort_values(["symbol", "date"])
    for col in REQUIRED_HISTORY_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[REQUIRED_HISTORY_COLUMNS]


def _fetch_history_from_ibkr(
    symbols: list[str],
    main_model_root: Path,
    host: str,
    port: int,
    client_id: int,
    timeout: int,
    duration: str,
    bar_size: str,
    market_data_type: int,
) -> pd.DataFrame:
    _install_main_model_path(main_model_root)
    from ib_async import IB, util  # type: ignore
    from autonomous_quant_agent.data.historical_data import fetch_hist_for_symbols

    util.startLoop()
    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=timeout)
        if not ib.isConnected():
            raise RuntimeError("Failed to connect to IBKR/TWS.")
        hist = fetch_hist_for_symbols(
            ib,
            symbols=symbols,
            duration=duration,
            bar_size=bar_size,
            market_data_type=market_data_type,
        )
        if hist is None or hist.empty:
            raise RuntimeError("IBKR historical fetch returned no rows.")
        hist["symbol"] = hist["symbol"].astype(str).str.upper()
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
        return hist.dropna(subset=["date", "symbol"]).sort_values(["symbol", "date"])
    finally:
        if ib.isConnected():
            ib.disconnect()


def _rolling_percentile_current(series: pd.Series, window: int = 252, min_periods: int = 20) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.rolling(window, min_periods=min_periods).apply(
        lambda x: float((pd.Series(x) <= pd.Series(x).iloc[-1]).mean() * 100.0),
        raw=False,
    )


def _compute_iv_analytics_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["date", "impliedVol"]].copy()
    out["current_iv"] = pd.to_numeric(out["impliedVol"], errors="coerce")
    out["iv_mean_12m"] = out["current_iv"].rolling(252, min_periods=20).mean()
    out["iv_std_12m"] = out["current_iv"].rolling(252, min_periods=20).std()
    out["iv_percentile"] = _rolling_percentile_current(out["current_iv"])
    out["iv_zscore_vs_12m"] = (out["current_iv"] - out["iv_mean_12m"]) / out["iv_std_12m"]
    return out.drop(columns=["impliedVol"])


def _segment_ids(state: pd.Series) -> tuple[pd.Series, pd.Series]:
    segment_id = pd.Series(pd.NA, index=state.index, dtype="Int64")
    segment_day = pd.Series(pd.NA, index=state.index, dtype="Int64")
    current_id = 0
    prev_state = None
    day = 0
    for idx, value in state.items():
        if pd.isna(value):
            prev_state = None
            day = 0
            continue
        if value != prev_state:
            current_id += 1
            day = 1
            prev_state = value
        else:
            day += 1
        segment_id.loc[idx] = current_id
        segment_day.loc[idx] = day
    return segment_id, segment_day


def compute_iv_rv_state_history(
    hist_df: pd.DataFrame,
    reg_df: pd.DataFrame,
    ewma_span: int = 10,
    rv_window: int = 20,
) -> pd.DataFrame:
    """Mirror dashboard.plot_iv_vs_rv state logic as a dataframe exporter."""
    base = hist_df.copy().sort_values("date").reset_index(drop=True)
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base["close"] = pd.to_numeric(base["close"], errors="coerce")
    base["impliedVol"] = pd.to_numeric(base["impliedVol"], errors="coerce")
    base = base.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
    full_dates = pd.Index(base["date"], name="date")

    out = pd.DataFrame({"date": full_dates})
    out["close"] = base.set_index("date")["close"].reindex(full_dates).to_numpy()
    out["daily_return"] = pd.Series(out["close"]).pct_change().to_numpy()
    out["state_available"] = False
    out["iv_rv_state"] = pd.NA
    for col in [
        "iv_smooth",
        "rv_raw",
        "rv_smooth",
        "vrp",
        "rolling_pos_ratio_15d",
        "orange_cond",
        "blue_exit",
        "blue_trigger_active",
        "warning_active",
        "segment_id",
        "segment_day",
    ]:
        out[col] = pd.NA

    plot_df = base.dropna(subset=["impliedVol", "close"]).copy().reset_index(drop=True)
    if len(plot_df) < rv_window + 5:
        return out

    iv_smooth = plot_df["impliedVol"].ewm(span=ewma_span, adjust=False).mean()
    log_ret = np.log(plot_df["close"] / plot_df["close"].shift(1))
    rv_raw = log_ret.rolling(rv_window, min_periods=max(5, rv_window // 2)).std() * np.sqrt(252)
    rv_smooth = rv_raw.ewm(span=ewma_span, adjust=False).mean()

    calc = plot_df[["date", "close"]].copy()
    calc["iv_smooth"] = iv_smooth
    calc["rv_raw"] = rv_raw
    calc["rv_smooth"] = rv_smooth
    calc = calc.dropna(subset=["iv_smooth", "rv_smooth"]).reset_index(drop=True)
    if calc.empty:
        return out

    raw_dates = pd.to_datetime(calc["date"], errors="coerce")
    vrp_s = pd.Series(calc["iv_smooth"].to_numpy(dtype=float) - calc["rv_smooth"].to_numpy(dtype=float), index=raw_dates)
    rolling_pos_ratio = (vrp_s > 0).rolling(15, min_periods=8).mean()
    deep_green_cond = (rolling_pos_ratio >= 0.80).fillna(False)

    vp_full = pd.Series(np.nan, index=raw_dates)
    if reg_df is not None and not reg_df.empty and {"date", "vol_pressure_score"}.issubset(reg_df.columns):
        vp_reg = reg_df[["date", "vol_pressure_score"]].copy()
        vp_reg["date"] = pd.to_datetime(vp_reg["date"], errors="coerce")
        vp_reg = vp_reg.dropna(subset=["date"]).set_index("date")["vol_pressure_score"]
        vp_full = pd.to_numeric(vp_reg.reindex(raw_dates).ffill(), errors="coerce")

    if vp_full.notna().sum() >= 20:
        vp_above = (vp_full > vp_full.rolling(20).quantile(0.995)).fillna(False)
        orange_cond = (vp_above.rolling(2, min_periods=2).sum() >= 2).fillna(False)
        vp_below_median = vp_full < vp_full.rolling(20, min_periods=5).quantile(0.50)
        vp_sustained_exit = (vp_below_median.rolling(3, min_periods=3).sum() >= 3).fillna(False)
    else:
        orange_cond = pd.Series(False, index=raw_dates)
        vp_sustained_exit = pd.Series(False, index=raw_dates)

    vrp_low = (vrp_s < vrp_s.rolling(60, min_periods=20).quantile(0.03)).fillna(False)
    close_s = pd.Series(calc["close"].to_numpy(dtype=float), index=raw_dates)
    rolling_high_40d = close_s.rolling(40, min_periods=20).max()
    is_drawdown_10pct = ((close_s - rolling_high_40d) / rolling_high_40d <= -0.10).fillna(False)
    vrp_negative = vrp_s < 0
    blue_exit = vrp_low & is_drawdown_10pct & vrp_negative

    state_list: list[str] = []
    blue_trigger_dates: list[pd.Timestamp] = []
    in_warning = False
    for i in range(len(raw_dates)):
        t = raw_dates.iloc[i]
        if orange_cond[t]:
            in_warning = True
            state_list.append("warning")
        elif in_warning:
            if blue_exit[t]:
                in_warning = False
                state_list.append("light_green")
                blue_trigger_dates.append(t)
            elif vp_sustained_exit[t]:
                in_warning = False
                state_list.append("light_green")
            else:
                state_list.append("warning")
        elif deep_green_cond[t]:
            state_list.append("deep_green")
        else:
            state_list.append("light_green")

    state_series = pd.Series(state_list, index=raw_dates)
    segment_id, segment_day = _segment_ids(state_series)
    calc_out = pd.DataFrame(
        {
            "date": raw_dates,
            "iv_smooth": calc["iv_smooth"].to_numpy(),
            "rv_raw": calc["rv_raw"].to_numpy(),
            "rv_smooth": calc["rv_smooth"].to_numpy(),
            "vrp": vrp_s.to_numpy(),
            "rolling_pos_ratio_15d": rolling_pos_ratio.to_numpy(),
            "iv_rv_state": state_series.to_numpy(),
            "orange_cond": orange_cond.to_numpy(dtype=bool),
            "blue_exit": blue_exit.to_numpy(dtype=bool),
            "blue_trigger_active": raw_dates.isin(blue_trigger_dates),
            "warning_active": (state_series == "warning").to_numpy(dtype=bool),
            "state_available": True,
            "segment_id": segment_id.to_numpy(),
            "segment_day": segment_day.to_numpy(),
        }
    )

    out = out.drop(columns=[c for c in calc_out.columns if c in out.columns and c != "date"]).merge(
        calc_out, on="date", how="left"
    )
    out["state_available"] = out["state_available"].map(lambda x: bool(x) if pd.notna(x) else False)
    for col in ["orange_cond", "blue_exit", "blue_trigger_active", "warning_active"]:
        out[col] = out[col].map(lambda x: bool(x) if pd.notna(x) else False)
    return out


def _segment_max_drawdown(prices: pd.Series) -> float | None:
    if len(prices.dropna()) < 2:
        return None
    roll_max = prices.expanding().max()
    return float(((prices - roll_max) / roll_max).min())


def _state_max_drawdown(df: pd.DataFrame, state: str) -> float | None:
    values: list[float] = []
    sub = df[df["iv_rv_state"].eq(state)].copy()
    if sub.empty or "segment_id" not in sub.columns:
        return None
    for _, seg in sub.groupby("segment_id", dropna=True):
        mdd = _segment_max_drawdown(pd.to_numeric(seg["close"], errors="coerce"))
        if mdd is not None:
            values.append(mdd)
    return min(values) if values else None


def _blue_forward_stats(df: pd.DataFrame) -> BlueStats:
    price_by_date = {
        pd.Timestamp(row["date"]): float(row["close"])
        for _, row in df.dropna(subset=["date", "close"]).iterrows()
    }
    dates_sorted = sorted(price_by_date)
    triggers = [
        pd.Timestamp(d)
        for d in df.loc[df["blue_trigger_active"].fillna(False), "date"].tolist()
    ]

    def lookup(p0: float, td: pd.Timestamp, cal_days: int) -> float | None:
        target = td + pd.Timedelta(days=cal_days)
        future = [d for d in dates_sorted if d >= target]
        if not future or p0 == 0:
            return None
        return (price_by_date[future[0]] / p0) - 1.0

    r30: list[float] = []
    r60: list[float] = []
    for td in triggers:
        p0 = price_by_date.get(td)
        if p0 is None:
            continue
        f30 = lookup(p0, td, 30)
        f60 = lookup(p0, td, 60)
        if f30 is not None:
            r30.append(float(f30))
        if f60 is not None:
            r60.append(float(f60))

    def avg(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    def hit(values: list[float]) -> float | None:
        return float((np.array(values) > 0).mean()) if values else None

    return BlueStats(
        count=len(triggers),
        avg_30d=avg(r30),
        hit_30d=hit(r30),
        best_30d=float(np.max(r30)) if r30 else None,
        worst_30d=float(np.min(r30)) if r30 else None,
        avg_60d=avg(r60),
        hit_60d=hit(r60),
        best_60d=float(np.max(r60)) if r60 else None,
        worst_60d=float(np.min(r60)) if r60 else None,
    )


def compute_stats_by_state(history_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for symbol, sub in history_df.groupby("symbol"):
        sub = sub.sort_values("date").copy()
        blue = _blue_forward_stats(sub)
        for state in STATES:
            mask = sub["iv_rv_state"].eq(state)
            r = pd.to_numeric(sub.loc[mask, "daily_return"], errors="coerce").dropna()
            days = int(mask.sum())
            seg_count = int(sub.loc[mask, "segment_id"].dropna().nunique()) if days else 0
            avg_segment_duration = (
                float(sub.loc[mask].groupby("segment_id")["date"].size().mean())
                if seg_count
                else None
            )
            accumulated_return = float((1.0 + r).prod() - 1.0) if len(r) else None
            row = {
                "symbol": symbol,
                "state": state,
                "days": days,
                "segments": seg_count,
                "avg_segment_duration": avg_segment_duration,
                "avg_daily_return": float(r.mean()) if len(r) else None,
                "annualized_return": float(r.mean() * 252.0) if len(r) else None,
                "annualized_vol": float(r.std() * np.sqrt(252.0)) if len(r) >= 2 else None,
                "hit_rate": float((r > 0).mean()) if len(r) else None,
                "max_drawdown": _state_max_drawdown(sub, state),
                "accumulated_return": accumulated_return,
                "cumulative_return": accumulated_return,
                "blue_trigger_count": blue.count if state == "warning" else None,
                "blue_30d_avg_return": blue.avg_30d if state == "warning" else None,
                "blue_30d_hit_rate": blue.hit_30d if state == "warning" else None,
                "blue_30d_best_return": blue.best_30d if state == "warning" else None,
                "blue_30d_worst_return": blue.worst_30d if state == "warning" else None,
                "blue_60d_avg_return": blue.avg_60d if state == "warning" else None,
                "blue_60d_hit_rate": blue.hit_60d if state == "warning" else None,
                "blue_60d_best_return": blue.best_60d if state == "warning" else None,
                "blue_60d_worst_return": blue.worst_60d if state == "warning" else None,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _run_hybrid_v2(hist_symbol_df: pd.DataFrame, main_model_root: Path) -> pd.DataFrame:
    _install_main_model_path(main_model_root)
    from autonomous_quant_agent.config.settings import REGIME_MODEL_HYBRID_V2
    from autonomous_quant_agent.dashboard.regime_lab import run_regime_lab_pipeline

    reg_df, _summary = run_regime_lab_pipeline(hist_symbol_df, model_version=REGIME_MODEL_HYBRID_V2)
    reg_df["date"] = pd.to_datetime(reg_df["date"], errors="coerce")
    return reg_df.dropna(subset=["date"]).sort_values("date")


def _select_regime_columns(reg_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date",
        "vol_pressure_score",
        "vol_level_score",
        "vol_regime",
        "realized_vol",
        "realized_vol_expansion",
        "atr_pct",
        "atr_expansion",
        "p_trend",
        "p_trend_2w",
        "drift_regime",
        "drift_regime_label",
        "drift_regime_source",
        "trend_score",
        "direction_score",
        "classification_layer",
        "leverage_entry_suitability",
        "reason_text",
    ]
    out = reg_df[[c for c in cols if c in reg_df.columns]].copy()
    if "realized_vol" in out.columns:
        out = out.rename(columns={"realized_vol": "realized_vol_20d"})
    return out


def build_signal_history(hist_df: pd.DataFrame, symbols: Iterable[str], main_model_root: Path) -> pd.DataFrame:
    all_rows: list[pd.DataFrame] = []
    for symbol in symbols:
        print(f"Processing {symbol}...")
        sub = hist_df[hist_df["symbol"].astype(str).str.upper().eq(symbol)].copy()
        if sub.empty:
            print(f"  skipped: no history rows")
            continue

        reg_df = _run_hybrid_v2(sub, main_model_root=main_model_root)
        state_df = compute_iv_rv_state_history(sub, reg_df=reg_df)
        iv_hist = _compute_iv_analytics_history(sub[["date", "impliedVol"]].copy())
        reg_cols = _select_regime_columns(reg_df)

        merged = (
            state_df.merge(iv_hist, on="date", how="left")
            .merge(reg_cols, on="date", how="left")
            .sort_values("date")
        )
        merged["symbol"] = symbol
        all_rows.append(merged)

    if not all_rows:
        raise RuntimeError("No symbol histories were processed.")

    out = pd.concat(all_rows, ignore_index=True)
    first_cols = ["date", "symbol"]
    out = out[first_cols + [c for c in out.columns if c not in first_cols]]
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def write_dictionary(path: Path) -> None:
    content = """# BL Signal Dictionary

This folder contains exported regime signals prepared for future Black-Litterman optimization backtests. The exporter lives in `Portfolio-Optimization/bl_signals/` and imports the main risk model functions from `Portfolio Risk Profile`; it does not modify the main model project.

## Files

### `bl_regime_history.csv`

One row per `date × symbol`. This is the primary point-in-time signal table for constructing Black-Litterman views at any historical rebalance date.

Core fields:

- `date`: observation date.
- `symbol`: ticker.
- `close`: close price used for return/stat calculations.
- `daily_return`: simple daily return from close-to-close.
- `state_available`: whether the IV/RV state machine had enough IV and price data to assign a state.
- `iv_rv_state`: `deep_green`, `light_green`, or `warning`; blank when `state_available` is false.
- `iv_smooth`: EWMA-smoothed implied volatility, matching the dashboard IV/RV chart logic.
- `rv_raw`: 20-day annualized realized volatility from log returns.
- `rv_smooth`: EWMA-smoothed realized volatility.
- `vrp`: `iv_smooth - rv_smooth`.
- `rolling_pos_ratio_15d`: rolling 15-observation share of days where `vrp > 0`.
- `orange_cond`: warning entry trigger based on extreme `vol_pressure_score`.
- `blue_exit`: blue re-entry condition based on low VRP, drawdown, and negative VRP.
- `blue_trigger_active`: true only on state-machine warning-exit dates.
- `warning_active`: true when `iv_rv_state == warning`.
- `segment_id`: contiguous state segment identifier.
- `segment_day`: trading day number within the current segment.
- `current_iv`: point-in-time implied volatility.
- `iv_mean_12m`: trailing 252-observation IV mean.
- `iv_std_12m`: trailing 252-observation IV standard deviation.
- `iv_percentile`: point-in-time IV percentile within trailing 252 observations.
- `iv_zscore_vs_12m`: point-in-time IV z-score versus trailing 252 observations.
- `vol_pressure_score`: Hybrid V2 volatility pressure score.
- `vol_level_score`: Hybrid V2 volatility level score.
- `vol_regime`: Hybrid V2 volatility regime.
- `realized_vol_20d`, `realized_vol_expansion`, `atr_pct`, `atr_expansion`: Hybrid V2 volatility features.
- `p_trend`: raw Hybrid V2 trend probability.
- `p_trend_2w`: 10-trading-day smoothed trend probability, the Regime Lab 2W Moving Trend Probability.
- `drift_regime`, `drift_regime_label`, `drift_regime_source`: Hybrid V2 drift classification.
- `trend_score`, `direction_score`, `classification_layer`: Hybrid V2 trend/drift diagnostics.
- `leverage_entry_suitability`, `reason_text`: Hybrid V2 interpretive outputs.

### `bl_regime_stats_by_state.csv`

Aggregated historical performance by `symbol × state`.

Fields:

- `symbol`: ticker.
- `state`: `deep_green`, `light_green`, or `warning`.
- `days`: number of state days.
- `segments`: number of contiguous state segments.
- `avg_segment_duration`: average segment length in trading days.
- `avg_daily_return`: average daily simple return during that state.
- `annualized_return`: `avg_daily_return × 252`.
- `annualized_vol`: daily return standard deviation during that state, annualized by `sqrt(252)`.
- `hit_rate`: share of state days with positive daily return.
- `max_drawdown`: worst segment-local drawdown observed during that state.
- `accumulated_return`: compounded return over all days assigned to that state.
- `cumulative_return`: alias of `accumulated_return` for compatibility.
- `blue_trigger_count`: number of blue re-entry trigger dates, populated on the `warning` row.
- `blue_30d_avg_return`, `blue_30d_hit_rate`, `blue_30d_best_return`, `blue_30d_worst_return`: 30-calendar-day forward return stats after blue triggers.
- `blue_60d_avg_return`, `blue_60d_hit_rate`, `blue_60d_best_return`, `blue_60d_worst_return`: 60-calendar-day forward return stats after blue triggers.

## Source Logic

- IV/RV three-state state machine mirrors `autonomous_quant_agent.dashboard.charts.plot_iv_vs_rv`.
- Hybrid V2 features come from `autonomous_quant_agent.dashboard.regime_lab.run_regime_lab_pipeline(..., model_version=REGIME_MODEL_HYBRID_V2)`.
- Historical data can be supplied as an input CSV or fetched through `autonomous_quant_agent.data.historical_data.fetch_hist_for_symbols`.

## Black-Litterman Usage

Fields most suitable for constructing `Q` / view magnitude:

- `iv_rv_state`
- `avg_daily_return`
- `annualized_return`
- `accumulated_return`
- `hit_rate`
- `blue_30d_avg_return`
- `blue_60d_avg_return`
- `p_trend_2w`
- `drift_regime`

Typical interpretation:

- `deep_green`: constructive or positive view.
- `light_green`: neutral-to-constructive view.
- `warning`: cautious, underweight, or negative view.
- `blue_trigger_active`: possible re-entry/recovery event, best treated as an event overlay rather than a fourth state.

Fields most suitable for constructing `Omega` / confidence:

- `annualized_vol`
- `max_drawdown`
- `hit_rate`
- `days`
- `segments`
- `avg_segment_duration`
- `rolling_pos_ratio_15d`
- `vrp`
- `iv_percentile`
- `iv_zscore_vs_12m`
- `vol_pressure_score`
- `vol_level_score`
- `classification_layer`
- `state_available`

Suggested confidence treatment:

- More state days and more independent segments imply higher statistical confidence.
- Higher annualized vol or deeper max drawdown imply lower confidence.
- `state_available == false` should prevent construction of an IV/RV state-based view for that row.
- Extreme `vol_pressure_score` can increase confidence in warning/caution views.
"""
    path.write_text(content, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export BL regime signal history.")
    parser.add_argument("--main-model-root", default=str(DEFAULT_MAIN_MODEL_ROOT))
    parser.add_argument("--input-csv", default="", help="Existing history CSV with symbol/date/OHLCV/impliedVol columns.")
    parser.add_argument("--symbols", default="", help="Comma-separated tickers. Required for --fetch-ibkr; optional for --input-csv.")
    parser.add_argument("--fetch-ibkr", action="store_true", help="Fetch history from IBKR instead of --input-csv.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7496)
    parser.add_argument("--client-id", type=int, default=610)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--duration", default="26 Y")
    parser.add_argument("--bar-size", default="1 day")
    parser.add_argument("--market-data-type", type=int, default=1)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    main_model_root = Path(args.main_model_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.fetch_ibkr:
        symbols = _parse_symbols(args.symbols)
        if not symbols:
            raise ValueError("--symbols is required with --fetch-ibkr")
        hist_df = _fetch_history_from_ibkr(
            symbols=symbols,
            main_model_root=main_model_root,
            host=args.host,
            port=args.port,
            client_id=args.client_id,
            timeout=args.timeout,
            duration=args.duration,
            bar_size=args.bar_size,
            market_data_type=args.market_data_type,
        )
    else:
        if not args.input_csv:
            raise ValueError("Provide --input-csv or use --fetch-ibkr.")
        hist_df = _load_history_csv(Path(args.input_csv).expanduser().resolve())
        symbols = _parse_symbols(args.symbols, input_df=hist_df)

    if not symbols:
        raise ValueError("No symbols selected for export.")

    signal_df = build_signal_history(hist_df=hist_df, symbols=symbols, main_model_root=main_model_root)
    stats_df = compute_stats_by_state(signal_df)

    history_path = out_dir / "bl_regime_history.csv"
    stats_path = out_dir / "bl_regime_stats_by_state.csv"
    dictionary_path = out_dir / "bl_signal_dictionary.md"

    signal_df.to_csv(history_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    write_dictionary(dictionary_path)

    print(f"Wrote {history_path} ({len(signal_df)} rows)")
    print(f"Wrote {stats_path} ({len(stats_df)} rows)")
    print(f"Wrote {dictionary_path}")


if __name__ == "__main__":
    main()
