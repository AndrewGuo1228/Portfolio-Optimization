"""
runner.py
---------
Rolling BL backtest main entry point.

Usage:
    python backtest/runner.py

All parameters live in configs/backtest_config.yaml.
"""

from __future__ import annotations

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── make package importable regardless of working directory ──────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from backtest.data_loader   import load_price_returns, load_ohlcv, compute_betas
from backtest.regime_builder import build_regime_drift_stats, build_regime_summary
from bl.optimizer           import run_bl_optimization

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helper — write all files optimizer.load_bl_inputs() expects
# ---------------------------------------------------------------------------

def _write_optimizer_inputs(
    reports_dir: Path,
    tickers: list[str],
    drift_stats_df: pd.DataFrame,
    regime_summary: dict,
    vol_hist_rows: list[dict],
    portfolio_value: float,
    betas_df: pd.DataFrame,
) -> None:
    """
    Write the CSV/JSON files that load_bl_inputs() reads.

    load_bl_inputs() uses:
        project_root = reports_dir.parent
        portfolioExposure/  ← project_root
        asset_config.json   ← project_root
        dashboard_user_prefs.json ← project_root
        regime_lab_ticker_summary.json ← reports_dir
        vol_history.csv        ← reports_dir
        iv_state_entry_log.csv ← reports_dir
        regime_drift_stats.csv ← reports_dir
    """
    project_root = reports_dir.parent
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. regime_drift_stats.csv  →  reports_dir
    drift_stats_df.to_csv(reports_dir / "regime_drift_stats.csv", index=False)

    # 2. regime_lab_ticker_summary.json  →  reports_dir
    with open(reports_dir / "regime_lab_ticker_summary.json", "w") as f:
        json.dump(regime_summary, f, indent=2)

    # 3. vol_history.csv  →  reports_dir
    pd.DataFrame(vol_hist_rows).to_csv(reports_dir / "vol_history.csv", index=False)

    # 4. iv_state_entry_log.csv  →  reports_dir  (dummy — no history needed)
    pd.DataFrame(columns=[
        "ticker", "entry_date", "checkpoint_date", "consistency_score_raw"
    ]).to_csv(reports_dir / "iv_state_entry_log.csv", index=False)

    # 5. portfolioExposure/  →  project_root  (mirrors original project layout)
    n = len(tickers)
    eq_weight = 1.0 / max(n, 1)
    exp_rows = [
        {
            "symbol":               t,
            "delta_gamma_exposure": portfolio_value * eq_weight,
            "max_exposure":         portfolio_value * eq_weight,
            "beta_exposure":        portfolio_value * eq_weight,
            "extrinsic_value":      0.0,
            "dollar_theta":         0.0,
            "beta":                 float(
                betas_df[betas_df["symbol"] == t]["beta"].values[0]
                if (betas_df["symbol"] == t).any() else 1.0
            ),
            "source_file": "BACKTEST",
        }
        for t in tickers
    ]
    exp_rows.append({
        "symbol":               "PORTFOLIO_TOTAL",
        "delta_gamma_exposure": portfolio_value,
        "max_exposure":         portfolio_value,
        "beta_exposure":        portfolio_value,
        "extrinsic_value":      0.0,
        "dollar_theta":         0.0,
        "beta":                 1.0,
        "source_file":          "BACKTEST",
    })
    port_dir = project_root / "portfolioExposure"
    port_dir.mkdir(exist_ok=True)
    today_str = datetime.today().strftime("%Y%m%d")
    pd.DataFrame(exp_rows).to_csv(
        port_dir / f"portfolio_exposure_{today_str}.csv", index=False
    )

    # 6. asset_config.json  →  project_root
    asset_config = {"assets": {t: {"asset_class": "equity"} for t in tickers}}
    with open(project_root / "asset_config.json", "w") as f:
        json.dump(asset_config, f)

    # 7. dashboard_user_prefs.json  →  project_root
    prefs = {
        "current_portfolio_size": portfolio_value,
        "strategic_weights": {t: eq_weight for t in tickers},
    }
    with open(project_root / "dashboard_user_prefs.json", "w") as f:
        json.dump(prefs, f)


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def run_backtest(config: dict):
    tickers         = config["tickers"]
    start_date      = config["start_date"]
    end_date        = config["end_date"]
    rebalance_freq  = config.get("rebalance_freq", "ME")   # pandas month-end
    portfolio_value = float(config.get("portfolio_value", 1_000_000))
    beta_limit      = float(config.get("beta_limit", 1.2))
    vol_limit       = float(config.get("vol_limit_factor", 1.2))
    cov_method      = config.get("cov_method", "kendall")

    print(f"[backtest] Tickers:   {tickers}")
    print(f"[backtest] Period:    {start_date} → {end_date}")
    print(f"[backtest] Rebalance: {rebalance_freq}")

    # ── Load price returns (full history for rolling slicing) ──────────────
    print("[backtest] Loading price returns...")
    returns_df = load_price_returns(tickers, start_date, end_date)
    print(f"  {returns_df.shape[0]} trading days, {returns_df.shape[1]} tickers")

    # ── Load OHLCV for regime detection ────────────────────────────────────
    print("[backtest] Loading OHLCV...")
    ohlcv_full: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            ohlcv_full[t] = load_ohlcv(t, start_date, end_date)
        except FileNotFoundError:
            print(f"  [warn] {t}.csv not found, regime will use fallback")

    # ── Rebalance date sequence ─────────────────────────────────────────────
    # "ME" = month-end in pandas >= 2.2; "M" works in older versions
    try:
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_freq)
    except ValueError:
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq="M")

    reports_dir = RESULTS_DIR / "tmp_reports"

    all_weights: list[dict]     = []
    all_signals: list[pd.DataFrame] = []

    for rebal_date in rebalance_dates:
        date_str = rebal_date.strftime("%Y-%m-%d")
        print(f"\n[backtest] ── Rebalance {date_str} ──")

        # Slice history up to rebal_date (no look-ahead)
        hist_to_date   = returns_df[returns_df.index <= rebal_date]
        ohlcv_to_date  = {
            t: df[df.index <= rebal_date]
            for t, df in ohlcv_full.items()
        }

        if hist_to_date.shape[0] < 60:
            print("  Not enough history yet, skipping")
            continue

        # ── Beta estimates ──────────────────────────────────────────────────
        betas_df = compute_betas(hist_to_date)

        # ── Regime drift stats ──────────────────────────────────────────────
        drift_stats = build_regime_drift_stats(ohlcv_to_date)

        # ── Regime summary (current state) ──────────────────────────────────
        regime_summary = build_regime_summary(ohlcv_to_date, as_of_date=date_str)

        # ── Vol history (simplified: realized_vol + neutral RSI) ────────────
        vol_hist_rows: list[dict] = []
        for t in tickers:
            t_rets = hist_to_date.get(t) if hasattr(hist_to_date, "get") else \
                     hist_to_date[t] if t in hist_to_date.columns else None
            if t_rets is None:
                continue
            rv = float(t_rets.rolling(20).std().iloc[-1] * np.sqrt(252)) \
                 if len(t_rets) >= 20 else 0.20
            vol_hist_rows.append({
                "date":             date_str,
                "ticker":           t,
                "realized_vol_20d": rv,
                "rsi_6":            50.0,  # neutral — no IV data in backtest
            })

        # ── Write all input files for optimizer ─────────────────────────────
        _write_optimizer_inputs(
            reports_dir=reports_dir,
            tickers=tickers,
            drift_stats_df=drift_stats,
            regime_summary=regime_summary,
            vol_hist_rows=vol_hist_rows,
            portfolio_value=portfolio_value,
            betas_df=betas_df,
        )

        # ── Run BL optimisation ─────────────────────────────────────────────
        try:
            signals = run_bl_optimization(
                hist_df=None,
                betas_df=betas_df,
                reports_dir=reports_dir,
                portfolio_value=portfolio_value,
                cov_method=cov_method,
                beta_limit=beta_limit,
                vol_limit_factor=vol_limit,
                price_returns_df=hist_to_date,  # DatetimeIndex df, cols=tickers
            )

            if signals is not None and not signals.empty:
                signals["rebal_date"] = date_str
                all_signals.append(signals)
                weights = {row["ticker"]: row["bl_weight"] for _, row in signals.iterrows()}
                weights["date"] = date_str
                all_weights.append(weights)
                print(f"  BL weights: { {k: f'{v:.3f}' for k, v in weights.items() if k != 'date'} }")
            else:
                print("  BL returned no signals")

        except Exception as e:
            import traceback
            print(f"  [backtest] BL failed on {date_str}: {e}")
            traceback.print_exc()

    # ── Persist results ─────────────────────────────────────────────────────
    if not all_weights:
        print("\n[backtest] No weights generated — check data coverage.")
        return None

    weights_df = pd.DataFrame(all_weights).set_index("date")
    weights_df.to_csv(RESULTS_DIR / "bl_weights_history.csv")

    if all_signals:
        signals_df = pd.concat(all_signals, ignore_index=True)
        signals_df.to_csv(RESULTS_DIR / "bl_signals_history.csv", index=False)

    _compute_portfolio_returns(returns_df, weights_df, RESULTS_DIR)

    print(f"\n[backtest] Done. Results in: {RESULTS_DIR}")
    return weights_df


# ---------------------------------------------------------------------------
# Performance attribution
# ---------------------------------------------------------------------------

def _compute_portfolio_returns(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    results_dir: Path,
) -> None:
    portfolio_rets: list[dict] = []
    ew_rets:        list[dict] = []
    current_weights: dict | None = None

    for date, row in returns_df.iterrows():
        date_str = date.strftime("%Y-%m-%d")
        if date_str in weights_df.index:
            current_weights = weights_df.loc[date_str].dropna().to_dict()

        if current_weights is None:
            continue

        # BL portfolio daily return
        bl_ret = sum(
            float(w) * float(row[t])
            for t, w in current_weights.items()
            if t in row.index and pd.notna(row[t]) and pd.notna(w)
        )
        portfolio_rets.append({"date": date, "return": bl_ret})

        # Equal-weight daily return
        held = [t for t in current_weights if t in row.index]
        ew_ret = float(row[held].mean()) if held else 0.0
        ew_rets.append({"date": date, "return": ew_ret})

    if not portfolio_rets:
        return

    perf_df = pd.DataFrame(portfolio_rets).set_index("date")
    ew_df   = pd.DataFrame(ew_rets).set_index("date")

    perf_df["bl_cumret"] = (1 + perf_df["return"]).cumprod() - 1
    ew_df["ew_cumret"]   = (1 + ew_df["return"]).cumprod() - 1

    result = perf_df[["return", "bl_cumret"]].join(ew_df[["ew_cumret"]], how="inner")
    result.columns = ["bl_return", "bl_cumret", "ew_cumret"]

    if "SPY" in returns_df.columns:
        spy = returns_df["SPY"].loc[result.index]
        result["spy_cumret"] = (1 + spy).cumprod() - 1

    result.to_csv(results_dir / "performance.csv")

    # Summary stats
    ann_bl  = float(perf_df["return"].mean() * 252)
    vol_bl  = float(perf_df["return"].std()  * np.sqrt(252))
    sharpe  = ann_bl / vol_bl if vol_bl > 0 else np.nan

    print("\n" + "=" * 50)
    print("Backtest Performance Summary")
    print("=" * 50)
    print(f"BL  Total Return : {result['bl_cumret'].iloc[-1]*100:+.1f}%")
    print(f"EW  Total Return : {result['ew_cumret'].iloc[-1]*100:+.1f}%")
    if "spy_cumret" in result:
        print(f"SPY Total Return : {result['spy_cumret'].iloc[-1]*100:+.1f}%")
    print(f"BL  Ann. Return  : {ann_bl*100:+.1f}%")
    print(f"BL  Ann. Vol     : {vol_bl*100:.1f}%")
    print(f"BL  Sharpe Ratio : {sharpe:.2f}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "configs" / "backtest_config.yaml"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_backtest(config)
