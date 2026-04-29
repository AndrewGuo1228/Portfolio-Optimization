"""
optimizer.py
------------
Black-Litterman portfolio optimization core.

Pure computation module — no file I/O except in load_bl_inputs().
All persistence is handled by the Step 3 main function.

Functions
---------
load_bl_inputs          : Load all BL input data from reports/
get_valid_tickers       : Filter exposure_df to valid BL tickers
compute_consistency_scores : Decay-weighted IV-state consistency per ticker
compute_expected_returns   : Blend theta yield + price return → Q vector
compute_covariance         : Pearson-EWMA or Kendall correlation → Σ
compute_equilibrium_pi     : Reverse-optimize implied Π and λ
compute_omega              : Diagonal uncertainty matrix Ω
compute_bl_posterior       : Standard BL posterior μ_BL
compute_optimal_weights    : SLSQP-constrained mean–variance weights
_validate_outputs          : Pretty-print diagnostics (testing only)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. load_bl_inputs
# ---------------------------------------------------------------------------

def load_bl_inputs(reports_dir: Path) -> dict:
    """
    Read all BL input data from the reports directory and adjacent files.

    Parameters
    ----------
    reports_dir : Path
        Path to the ``reports/`` directory (e.g. ``Path("reports")``).
        ``portfolioExposure/`` and ``asset_config.json`` are resolved one
        level up at the project root (``reports_dir.parent``).

    Returns
    -------
    dict with keys:
        exposure_df, regime_summary, vol_history, entry_log,
        asset_config, regime_drift_df

    Note
    ----
    Theta yield CSVs (theta_yield_summary_*.csv) are intentionally NOT loaded
    here — they belong to Layer 2.  The files remain on disk for Layer 2 use.
    """
    reports_dir = Path(reports_dir)
    project_root = reports_dir.parent

    # 1. Latest position snapshot
    exposure_files = sorted(
        (project_root / "portfolioExposure").glob("portfolio_exposure_*.csv")
    )
    if not exposure_files:
        raise FileNotFoundError(
            f"No portfolio_exposure_*.csv files found in "
            f"{project_root / 'portfolioExposure'}"
        )
    exposure_df = pd.read_csv(exposure_files[-1])

    # BL optimization 从聚合敞口中扣除 THETA_HARVEST 腿的贡献
    # portfolio_exposure_*.csv 是按 symbol 聚合的视图，没有 source_file 列；
    # 需要从 Greeks/greeks_*.csv（有 source_file）中读取 TH 腿的贡献并减去。
    _greeks_dir = project_root / "Greeks"
    _greeks_files = sorted(_greeks_dir.glob("greeks_*.csv"))
    if _greeks_files:
        try:
            _greeks_df = pd.read_csv(_greeks_files[-1])
            if "source_file" in _greeks_df.columns:
                _sym_col_e = "ticker" if "ticker" in exposure_df.columns else "symbol"
                _th_agg = (
                    _greeks_df[_greeks_df["source_file"] == "THETA_HARVEST"]
                    .groupby("symbol")[["delta_gamma_exposure", "max_exposure"]]
                    .sum()
                    .reset_index()
                    .rename(columns={
                        "symbol": _sym_col_e,
                        "delta_gamma_exposure": "_th_dge",
                        "max_exposure": "_th_me",
                    })
                )
                if not _th_agg.empty:
                    exposure_df = exposure_df.merge(_th_agg, on=_sym_col_e, how="left")
                    exposure_df["_th_dge"] = exposure_df["_th_dge"].fillna(0.0)
                    exposure_df["_th_me"] = exposure_df["_th_me"].fillna(0.0)
                    exposure_df["delta_gamma_exposure"] -= exposure_df["_th_dge"]
                    exposure_df["max_exposure"] = (
                        exposure_df["max_exposure"] - exposure_df["_th_me"]
                    ).clip(lower=0.0)
                    _n_adj = (exposure_df["_th_dge"] != 0).sum()
                    exposure_df.drop(columns=["_th_dge", "_th_me"], inplace=True)
                    if _n_adj > 0:
                        print(f"[BL] 已从 {_n_adj} 个 symbol 的敞口中扣除 THETA_HARVEST 腿的贡献")
        except Exception as _e_th:
            print(f"[BL] THETA_HARVEST 敞口调整失败（不影响优化）: {_e_th}")

    # 2. Regime summary (drift regime state, days_in_current_regime, …)
    regime_summary = json.loads(
        (reports_dir / "regime_lab_ticker_summary.json").read_text(encoding="utf-8")
    )

    # 3. Vol history (realized_vol_20d, 504-day percentile)
    vol_history = pd.read_csv(
        reports_dir / "vol_history.csv",
        parse_dates=["date"],
    )

    # 4. IV state entry log (consistency scores)
    entry_log = pd.read_csv(
        reports_dir / "iv_state_entry_log.csv",
        parse_dates=["entry_date", "checkpoint_date"],
    )

    # 5. Asset config (SHV special flags, fixed_return, skip_regime, …)
    asset_config = json.loads(
        (project_root / "asset_config.json").read_text(encoding="utf-8")
    )

    # 6. Regime drift stats (annualised return per ticker/regime)
    regime_drift_path = reports_dir / "regime_drift_stats.csv"
    if regime_drift_path.exists():
        regime_drift_df = pd.read_csv(regime_drift_path)
    else:
        regime_drift_df = None
        print("[BL] regime_drift_stats.csv 不存在，"
              "price_return 将使用 fallback 0.0")

    # 7. Strategic weights from dashboard_user_prefs.json
    #    Used by compute_equilibrium_pi() to anchor Π to the intended
    #    long-run allocation instead of the current (possibly drifted) weights.
    _prefs_path_opt = project_root / "dashboard_user_prefs.json"
    _strategic_weights: dict = {}
    if _prefs_path_opt.exists():
        try:
            _prefs_opt = json.loads(
                _prefs_path_opt.read_text(encoding="utf-8")
            )
            _strategic_weights = _prefs_opt.get("strategic_weights", {})
            if _strategic_weights:
                print(
                    f"[BL] strategic_weights loaded: "
                    f"{len(_strategic_weights)} tickers"
                )
        except Exception as _e_prefs:
            print(f"[BL] strategic_weights load failed (ignored): {_e_prefs}")

    return {
        "exposure_df":        exposure_df,
        "regime_summary":     regime_summary,
        "vol_history":        vol_history,
        "entry_log":          entry_log,
        "asset_config":       asset_config,
        "regime_drift_df":    regime_drift_df,
        "strategic_weights":  _strategic_weights,
    }


# ---------------------------------------------------------------------------
# 2. get_valid_tickers
# ---------------------------------------------------------------------------

def get_valid_tickers(
    exposure_df: pd.DataFrame,
    asset_config: dict,
) -> list[str]:
    """
    Return tickers that participate in BL optimisation.

    Excluded
    --------
    - delta_gamma_exposure <= 0  (net short or zero)
    - max_exposure <= 0
    - PORTFOLIO_TOTAL sentinel row
    - QQQ  (used only as benchmark, not a held position)

    SHV is *included* when delta_gamma_exposure > 0 (equity-like cash
    position: qty × price is always positive for a long holding).
    """
    # Support both 'ticker' and 'symbol' column names
    sym_col = "ticker" if "ticker" in exposure_df.columns else "symbol"

    mask = (
        (exposure_df["delta_gamma_exposure"] > 0)
        & (exposure_df["max_exposure"] > 0)
        & (~exposure_df[sym_col].isin(["PORTFOLIO_TOTAL", "QQQ"]))
    )
    return exposure_df[mask][sym_col].str.upper().tolist()


# ---------------------------------------------------------------------------
# 3. compute_consistency_scores
# ---------------------------------------------------------------------------

def compute_consistency_scores(
    tickers: list[str],
    entry_log: pd.DataFrame,
    decay: float = 0.98,
) -> dict[str, float]:
    """
    Decay-weighted average consistency_score_raw per ticker.

    Rows with NaN checkpoint_date (new-entry stubs) are excluded.

    Returns
    -------
    dict[ticker → float in [0, 1]]
    Missing tickers get 0.5 (neutral prior).
    """
    scores: dict[str, float] = {}

    for ticker in tickers:
        rows = entry_log[entry_log["ticker"] == ticker].sort_values(
            "checkpoint_date"
        )
        rows = rows.dropna(subset=["checkpoint_date", "consistency_score_raw"])

        if len(rows) == 0:
            scores[ticker] = 0.5
            continue

        raw = rows["consistency_score_raw"].values.astype(float)
        n = len(raw)
        w = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=float)
        w /= w.sum()
        scores[ticker] = float(np.dot(w, raw))

    return scores


# ---------------------------------------------------------------------------
# 3b. compute_price_return_from_regime
# ---------------------------------------------------------------------------

def compute_price_return_from_regime(
    ticker: str,
    regime_summary: dict,
    regime_drift_df,
    asset_config: dict | None = None,
) -> tuple[float, float]:
    """
    Return (price_return, confidence) based on the drift regime state machine.

    Parameters
    ----------
    ticker          : str   — uppercase ticker symbol
    regime_summary  : dict  — regime_lab_ticker_summary.json content
    regime_drift_df : pd.DataFrame | None  — regime_drift_stats.csv content
    asset_config    : dict | None  — asset_config.json content; used to
                       short-circuit cash_equivalent assets with confidence=1.0

    Returns
    -------
    price_return : float  annualised return estimate for the current regime
    confidence   : float  in (0, 1], decay-based confidence on days in regime
    """
    # Cash-equivalent assets (e.g. SHV): fixed return, full confidence
    if asset_config is not None:
        cfg = asset_config.get("assets", {}).get(ticker, {})
        if cfg.get("asset_class") == "cash_equivalent":
            return float(cfg.get("fixed_return", 0.037)), 1.0

    info = regime_summary.get(ticker, {})

    current_regime = info.get("current_drift_regime_base", "UPTREND")
    days_in_regime = int(info.get("days_in_current_regime", 1))
    prev_regime    = info.get("prev_drift_regime_base", "UPTREND")
    trans_progress = float(info.get("transition_progress", 1.0))
    is_buffer_day  = bool(info.get("is_buffer_day", False))

    if regime_drift_df is None:
        return 0.0, 0.5

    def get_ann_return(regime: str) -> float:
        row = regime_drift_df[
            (regime_drift_df["ticker"] == ticker.upper()) &
            (regime_drift_df["regime"] == regime)
        ]
        if row.empty or pd.isna(row["annualized_return"].values[0]):
            return 0.0
        return float(row["annualized_return"].values[0])

    r_up   = get_ann_return("UPTREND")
    r_rng  = get_ann_return("RANGE")
    r_down = get_ann_return("DOWNTREND")

    regime_returns = {
        "UPTREND":   r_up,
        "RANGE":     r_rng,
        "DOWNTREND": r_down,
        "MIXED":     (r_up + r_rng) / 2.0,
    }

    # Buffer day: maintain DOWNTREND view with very low confidence
    if is_buffer_day:
        return r_down, float(1 - np.exp(-1 / 7))   # ≈ 13%

    # Confidence (solely based on days persisted in current regime)
    confidence = float(1 - np.exp(-days_in_regime / 7))

    # Asymmetric transition window:
    #   entering DOWNTREND  (current=DOWN, prev≠DOWN) →  5d
    #   exiting  DOWNTREND  (prev=DOWN, current≠DOWN) → 10d cautious recovery
    #   UPTREND ↔ RANGE                               →  7d
    if current_regime == "DOWNTREND" and prev_regime != "DOWNTREND":
        trans_days = 5    # entering DOWNTREND
    elif prev_regime == "DOWNTREND":
        trans_days = 10   # exiting DOWNTREND
    else:
        trans_days = 7    # UPTREND ↔ RANGE

    # Stable phase: regime has persisted long enough
    if days_in_regime >= trans_days:
        return (
            float(regime_returns.get(current_regime, 0.0)),
            confidence,
        )

    # Transition phase (t³ blend curve)
    t         = days_in_regime / trans_days
    t_cubed   = t ** 3

    if trans_progress < 1.0 and prev_regime != current_regime:
        # Mid-transition reversal: weight from in-progress transition
        weight = trans_progress * (1 - t_cubed)
        price_return = (
            weight * float(regime_returns.get(prev_regime, 0.0)) +
            (1 - weight) * float(regime_returns.get(current_regime, 0.0))
        )
    else:
        # Forward transition: blend prev → current via t³
        price_return = (
            (1 - t_cubed) * float(regime_returns.get(prev_regime, 0.0)) +
            t_cubed * float(regime_returns.get(current_regime, 0.0))
        )

    return float(price_return), confidence


# ---------------------------------------------------------------------------
# 3b. compute_rsi_vol_discount
# ---------------------------------------------------------------------------

def compute_rsi_vol_discount(
    rsi: float,
    vol_percentile: float,
) -> float:
    """
    RSI + Vol dual-dimension discount on Q views.

    Single dimension: mild penalty starts at RSI≥75 (up to −30%) or
                      vol_percentile≥0.75 (up to −25%).
    Combined:         product penalty dominates when both are elevated;
                      max combined discount is −90%.
    Return value is the multiplier in [0.20, 1.0].
    """
    # RSI solo penalty (RSI 75→90 → 0%→30%)
    if rsi <= 75:
        rsi_solo = 0.0
    else:
        rsi_solo = ((rsi - 75) / 15) ** 1.5 * 0.30
        rsi_solo = min(rsi_solo, 0.30)

    # Vol solo penalty (percentile 0.75→0.90 → 0%→25%)
    if vol_percentile <= 0.75:
        vol_solo = 0.0
    else:
        vol_solo = ((vol_percentile - 0.75) / 0.15) ** 1.5 * 0.25
        vol_solo = min(vol_solo, 0.25)

    # Combined (product) penalty — dominates when both dimensions elevated
    rsi_factor      = min(max((rsi - 70) / 20, 0.0), 1.0)
    vol_factor      = min(max((vol_percentile - 0.65) / 0.25, 0.0), 1.0)
    combined_penalty = rsi_factor * vol_factor * 0.90

    total_penalty = max(combined_penalty, rsi_solo + vol_solo)
    return max(1.0 - total_penalty, 0.20)


# ---------------------------------------------------------------------------
# 4. compute_expected_returns
# ---------------------------------------------------------------------------

def compute_expected_returns(
    tickers: list[str],
    regime_summary: dict,
    vol_history: pd.DataFrame,
    asset_config: dict,
    regime_drift_df=None,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute the Q (views) vector and per-ticker price uncertainty (price_omega).

    Layer 1 — pure price-return optimisation.
    Q = price_return (drift regime state machine) × discount.
    discount = compute_rsi_vol_discount(RSI-6, IV-percentile).
    IV percentile (12M) is read from regime_lab_ticker_summary.json via
    regime_summary[ticker]["iv_percentile"] (0–100 raw → divided by 100).
    RSI-6 is read from vol_history["rsi_6"] (written by workflow).
    Fallbacks: IV percentile → 0.5; RSI → 50 (neutral, no penalty).

    Returns
    -------
    expected_returns : pd.Series  (annualised, index=ticker)
    price_omega      : pd.Series  (variance of the price-return view, index=ticker)
    """
    exp_returns:  dict[str, float] = {}
    price_omegas: dict[str, float] = {}

    for ticker in tickers:
        # ── SHV: cash-equivalent fixed return ──────────────────────────────
        cfg = asset_config.get("assets", {}).get(ticker, {})
        if cfg.get("asset_class") == "cash_equivalent":
            exp_returns[ticker]  = float(cfg.get("fixed_return", 0.037))
            price_omegas[ticker] = 0.001 ** 2   # near-zero uncertainty
            continue

        # ── Price return (from drift regime state machine) ──────────────────
        price_return, price_regime_confidence = compute_price_return_from_regime(
            ticker, regime_summary, regime_drift_df, asset_config=asset_config
        )

        # Realized vol → price_omega, modulated by regime confidence
        ticker_vol_s = vol_history[vol_history["ticker"] == ticker][
            "realized_vol_20d"
        ]
        price_return_std = (
            float(ticker_vol_s.iloc[-1]) if len(ticker_vol_s) > 0 else 0.20
        )
        price_omega = price_return_std ** 2 / max(price_regime_confidence, 0.05)

        # ── RSI + IV discount ───────────────────────────────────────────────
        # vol_percentile = 12M IV percentile from regime_summary (0–1 scale).
        # Source: regime_lab_ticker_summary.json["iv_percentile"] (0–100 raw).
        # Fallback to 0.5 (neutral) if field absent (e.g. bond ETFs w/o options).
        _iv_raw = (regime_summary or {}).get(ticker, {}).get("iv_percentile", None)
        if _iv_raw is not None:
            try:
                vol_percentile = float(_iv_raw) / 100.0
            except (TypeError, ValueError):
                vol_percentile = 0.5
        else:
            vol_percentile = 0.5

        # RSI-6 from vol_history (written by workflow; fallback = 50 neutral)
        rsi_series = vol_history[vol_history["ticker"] == ticker]["rsi_6"]
        if len(rsi_series) > 0:
            try:
                rsi = float(rsi_series.iloc[-1])
                if pd.isna(rsi):
                    rsi = 50.0
            except (TypeError, ValueError):
                rsi = 50.0
        else:
            rsi = 50.0

        discount = compute_rsi_vol_discount(rsi, vol_percentile)

        # ── Layer 1: Q = price_return × discount ────────────────────────────
        exp_returns[ticker]  = price_return * discount
        price_omegas[ticker] = price_omega

    return pd.Series(exp_returns), pd.Series(price_omegas)


# ---------------------------------------------------------------------------
# 5. compute_covariance
# ---------------------------------------------------------------------------

def compute_covariance(
    price_returns_df: pd.DataFrame,
    tickers: list[str],
    method: str = "kendall",
    window: int = 504,
    halflife: int = 60,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute an annualised covariance matrix.

    Parameters
    ----------
    price_returns_df : pd.DataFrame
        Daily log-return matrix (columns = ticker symbols).
    tickers : list[str]
        Desired ticker order; only tickers present in the DataFrame are used.
    method : {"kendall", "pearson"}
        "kendall"  → Kendall τ correlation → Pearson r conversion → Σ
                     (robust to fat tails; O(n²) pairwise)
        "pearson"  → exponentially weighted covariance
    window : int
        Most-recent trading days to include (default 504 ≈ 2 Y).
    halflife : int
        EWMA halflife in days (pearson only, default 60).

    Returns
    -------
    Sigma : np.ndarray  shape (m, m)  annualised covariance
    valid : list[str]   tickers actually included (same order as Sigma rows)
    """
    from scipy.stats import kendalltau  # lazy import keeps module importable

    valid  = [t for t in tickers if t in price_returns_df.columns]
    recent = price_returns_df[valid].iloc[-window:].dropna(how="all")

    if method == "pearson":
        n = len(recent)
        w = np.array(
            [np.exp(-np.log(2) / halflife * (n - 1 - i)) for i in range(n)],
            dtype=float,
        )
        w /= w.sum()
        mean    = (recent.values * w[:, None]).sum(axis=0)
        demeaned = recent.values - mean
        Sigma   = (demeaned * w[:, None]).T @ demeaned * 252.0

    elif method == "kendall":
        cols     = recent.columns.tolist()
        m        = len(cols)
        corr_mat = np.eye(m)

        for i in range(m):
            for j in range(i + 1, m):
                x = recent[cols[i]].dropna()
                y = recent[cols[j]].dropna()
                common = x.index.intersection(y.index)
                if len(common) < 20:
                    r = 0.0
                else:
                    tau, _ = kendalltau(x[common].values, y[common].values)
                    r = float(np.sin(np.pi / 2 * tau))
                corr_mat[i, j] = r
                corr_mat[j, i] = r

        # Ensure positive-definiteness
        eigvals = np.linalg.eigvalsh(corr_mat)
        if eigvals.min() < 1e-8:
            corr_mat += (-eigvals.min() + 1e-6) * np.eye(m)

        vols  = recent.std().values * np.sqrt(252.0)
        D     = np.diag(vols)
        Sigma = D @ corr_mat @ D

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'pearson' or 'kendall'.")

    return Sigma, valid


# ---------------------------------------------------------------------------
# 6. compute_equilibrium_pi
# ---------------------------------------------------------------------------

def _compress_q(q: float) -> float:
    """
    Double log-compression for Q values used only in λ back-calculation.
    log(1 + log(1+q)) — stronger smoothing than single log1p.
    Preserves relative ordering; smooths extreme positive views.
    Negative views are passed through unchanged (already conservative).
    """
    if q <= 0:
        return q
    return float(np.log1p(np.log1p(q)))


def compute_equilibrium_pi(
    Sigma: np.ndarray,
    w_current: np.ndarray,
    expected_returns: np.ndarray,
    tickers: list[str] | None = None,
    strategic_weights: dict | None = None,
) -> tuple[np.ndarray, float]:
    """
    Reverse-optimise to recover the implied equilibrium return vector Π
    and the implied risk-aversion coefficient λ.

    λ 反推时只使用主要持仓（w > 5%）且期望收益合理（|Q| < 2.0）的资产，
    避免小仓位或异常收益拉偏 lambda。

    Π 的权重基础：
    - 若 strategic_weights 覆盖 ≥ 50% 的 tickers，用 strategic_weights
      归一化后计算 Π（锚定长期目标权重，避免随短期漂移而波动）。
    - 否则 fallback 到 w_current（原有行为）。

    λ 反推始终用 w_current + major_mask（不受此改动影响）。

    λ is clipped to [0.5, 10.0] for numerical stability.

    Parameters
    ----------
    tickers : list[str] | None
        Ticker labels aligned with w_current / expected_returns.
        Used only for diagnostic printing; safe to omit.

    Returns
    -------
    Pi            : np.ndarray  implied equilibrium returns
    lambda_implied : float
    """
    expected_returns = np.asarray(expected_returns)
    _tickers = tickers if tickers is not None else [str(i) for i in range(len(w_current))]

    # 只用主要持仓且 Q 合理的资产反推 lambda
    major_mask = (w_current > 0.05) & (np.abs(expected_returns) < 2.0)

    print("[Lambda 诊断]")
    print(f"  全量 tickers: {len(w_current)}")
    print(f"  major_mask 选中: {major_mask.sum()} 个")

    if major_mask.sum() >= 3:
        # 有足够的主要持仓，用过滤后的资产反推
        w_major = w_current[major_mask]
        w_major = w_major / w_major.sum()  # 归一化
        Q_major = expected_returns[major_mask]
        Sigma_major = Sigma[np.ix_(major_mask, major_mask)]

        # 对数压缩：只用于 lambda 反推，平滑高 Q 资产对 lambda 的拉偏
        Q_major_compressed = np.array([_compress_q(q) for q in Q_major])

        portfolio_return_compressed = float(Q_major_compressed @ w_major)
        portfolio_variance_major    = float(w_major @ Sigma_major @ w_major)

        if portfolio_variance_major > 1e-10:
            lambda_implied = portfolio_return_compressed / (
                2 * portfolio_variance_major
            )
        else:
            lambda_implied = 2.5

        # 诊断：选中资产 + Q 压缩前后对比
        print(f"  选中的 tickers (Q_original → Q_compressed):")
        _major_tickers = [t for i, t in enumerate(_tickers) if major_mask[i]]
        for t, q_orig, q_comp in zip(
            _major_tickers, Q_major, Q_major_compressed
        ):
            _i = list(_tickers).index(t)
            print(
                f"    {t}: weight={w_current[_i]:.3f}, "
                f"Q={q_orig:.4f} → {q_comp:.4f}"
            )
        print(f"  portfolio_return_compressed = {portfolio_return_compressed:.4f}")
        print(f"  portfolio_variance_major    = {portfolio_variance_major:.6f}")
    else:
        # 主要持仓不足3个，fallback 到全量反推
        portfolio_variance = float(w_current @ Sigma @ w_current)
        portfolio_return   = float(expected_returns @ w_current)
        if portfolio_variance > 1e-10:
            lambda_implied = portfolio_return / (
                2 * portfolio_variance
            )
        else:
            lambda_implied = 2.5

        print(f"  ⚠️  主要持仓不足3个，fallback 到全量反推")
        print(f"  portfolio_return   = {portfolio_return:.4f}")
        print(f"  portfolio_variance = {portfolio_variance:.6f}")

    lambda_implied = float(np.clip(lambda_implied, 0.5, 10.0))
    print(f"  lambda_implied = {lambda_implied:.3f}")

    # ── Determine w_for_pi (strategic anchor or fallback to w_current) ──────
    # λ was derived from w_current; Π can use a different (more stable)
    # weight vector to reduce sensitivity to short-term position drift.
    _sw = strategic_weights or {}
    _coverage = sum(1 for t in _tickers if t in _sw)
    _threshold = max(1, int(len(_tickers) * 0.5))

    if _sw and _coverage >= _threshold:
        _w_strat = np.array(
            [_sw.get(t, 0.0) for t in _tickers], dtype=float
        )
        _strat_total = _w_strat.sum()
        if _strat_total > 1e-10:
            w_for_pi = _w_strat / _strat_total
            print(
                f"  [Π] using strategic_weights "
                f"(coverage {_coverage}/{len(_tickers)})"
            )
        else:
            w_for_pi = w_current
            print("  [Π] strategic_weights sum ≈ 0 — fallback to w_current")
    else:
        w_for_pi = w_current
        if _sw:
            print(
                f"  [Π] strategic_weights coverage insufficient "
                f"({_coverage}/{len(_tickers)}) — using w_current"
            )

    Pi = lambda_implied * Sigma @ w_for_pi

    return Pi, lambda_implied


# ---------------------------------------------------------------------------
# 7. compute_omega
# ---------------------------------------------------------------------------

def compute_omega(
    tickers: list[str],
    price_omegas: pd.Series,
    vol_history: "pd.DataFrame | None" = None,
    qqq_vol: float = 0.20,
) -> "tuple[np.ndarray, dict[str, float]]":
    """
    Build the diagonal uncertainty matrix Ω.

    Layer 1 — each diagonal entry is the price_omega (from compute_expected_returns)
    scaled by a volatility penalty when realized vol exceeds ``qqq_vol × 2.0``.

    The penalty grows quadratically with the excess:
        vol_penalty = 1 + ((realized_vol - threshold) / threshold)²

    Returns
    -------
    Omega        : np.ndarray  shape (n, n)  diagonal matrix with Ω_i > 0
    vol_penalties: dict[str, float]  per-ticker multiplier (1.0 = no penalty)
    """
    omega_diag:   list[float]       = []
    vol_penalties: dict[str, float] = {}

    vol_threshold = qqq_vol * 2.0

    for ticker in tickers:
        p_omega = float(price_omegas.get(ticker, 0.20 ** 2))

        # ── Vol penalty: only punish excess above QQQ × 2.0 ─────────────────
        if vol_history is not None:
            ticker_vol_s = vol_history[vol_history["ticker"] == ticker]["realized_vol_20d"]
            realized_vol = float(ticker_vol_s.iloc[-1]) if len(ticker_vol_s) > 0 else qqq_vol
        else:
            realized_vol = qqq_vol

        if realized_vol <= vol_threshold:
            vol_penalty = 1.0
        else:
            excess      = (realized_vol - vol_threshold) / vol_threshold
            vol_penalty = 1.0 + excess ** 2

        # ── Layer 1: Ω_i = price_omega × vol_penalty ────────────────────────
        omega_diag.append(p_omega * vol_penalty)
        vol_penalties[ticker] = round(vol_penalty, 3)

    return np.diag(omega_diag), vol_penalties


# ---------------------------------------------------------------------------
# 8. compute_bl_posterior
# ---------------------------------------------------------------------------

def compute_bl_posterior(
    Pi: np.ndarray,
    Sigma: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float = 0.05,
) -> np.ndarray:
    """
    Standard Black-Litterman posterior mean μ_BL.

    With P = I (one absolute view per asset):

        M      = (τΣ)⁻¹ + Ω⁻¹
        μ_BL   = M⁻¹ [(τΣ)⁻¹ Π + Ω⁻¹ Q]

    Returns
    -------
    mu_BL : np.ndarray  shape (n,)
    """
    n           = len(Pi)
    P           = np.eye(n)
    tau_sig_inv = np.linalg.inv(tau * Sigma)
    omega_inv   = np.linalg.inv(Omega)

    M     = tau_sig_inv + P.T @ omega_inv @ P
    mu_BL = np.linalg.inv(M) @ (tau_sig_inv @ Pi + P.T @ omega_inv @ Q)

    return mu_BL


# ---------------------------------------------------------------------------
# 9. compute_optimal_weights
# ---------------------------------------------------------------------------

def compute_optimal_weights(
    mu_BL: np.ndarray,
    Sigma: np.ndarray,
    lambda_implied: float,
    w_current: np.ndarray,
    tickers: list[str],
    betas: dict[str, float],
    qqq_vol: float,
    lambda_override: float | None = None,
    beta_limit: float = 1.2,
    vol_limit_factor: float = 1.2,
) -> tuple[np.ndarray, bool]:
    """
    SLSQP mean–variance optimisation with portfolio-level constraints.

    Objective (minimise)
    --------------------
    - μ_BL·w  +  λ · wᵀΣw

    Constraints
    -----------
    - Σ wᵢ = 1   (fully invested)
    - β_portfolio ≤ 1.2
    - σ_portfolio ≤ QQQ_vol × 1.2

    Bounds
    ------
    All tickers: [0.0, 0.70]  (no shorts; max 70% in any single name)
    The optimiser will naturally under-weight SHV if other assets offer
    better risk-adjusted expected returns.

    Returns
    -------
    w_star  : np.ndarray  optimal weights (sums to 1, clipped ≥ 0)
    success : bool
    """
    from scipy.optimize import minimize  # lazy import

    lam = float(lambda_override) if lambda_override is not None else lambda_implied
    n   = len(tickers)
    print(f"[BL DEBUG] SLSQP starting, n={n}")

    def objective(w: np.ndarray) -> float:
        return float(-(mu_BL @ w) + lam * (w @ Sigma @ w))

    def grad(w: np.ndarray) -> np.ndarray:
        return -mu_BL + 2.0 * lam * Sigma @ w

    beta_vec = np.array([betas.get(t, 1.0) for t in tickers], dtype=float)

    constraints = [
        {   # fully invested
            "type": "eq",
            "fun":  lambda w: w.sum() - 1.0,
            "jac":  lambda w: np.ones(n),
        },
        {   # portfolio beta ≤ beta_limit
            "type": "ineq",
            "fun":  lambda w: beta_limit - float(beta_vec @ w),
            "jac":  lambda w: -beta_vec,
        },
        {   # portfolio volatility² ≤ (QQQ_vol × vol_limit_factor)²
            "type": "ineq",
            "fun":  lambda w: (qqq_vol * vol_limit_factor) ** 2 - float(w @ Sigma @ w),
            "jac":  lambda w: -2.0 * (Sigma @ w),
        },
    ]

    # Uniform bounds: all tickers [0%, 70%]
    bounds = [(0.0, 0.70)] * n

    result = minimize(
        objective,
        x0=w_current,
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    print(f"[BL DEBUG] SLSQP result.success={result.success}")
    print(f"[BL DEBUG] SLSQP result.message={result.message}")
    print(f"[BL DEBUG] SLSQP result.x={result.x}")
    print(f"[BL DEBUG] entering {'success' if result.success else 'FAILURE'} branch")

    if result.success:
        w_star = np.clip(result.x, 0.0, None)
        w_star /= w_star.sum()

        # ── Risk contribution diagnostics ────────────────────────────────────
        _sigma_w   = Sigma @ w_star
        _rc        = w_star * _sigma_w
        _total_var = float(w_star @ _sigma_w)
        _rc_pct    = _rc / _total_var if _total_var > 0 else _rc
        _rr        = np.where(
            _rc > 1e-12,
            mu_BL / _rc,
            np.where(mu_BL > 0, np.inf, -np.inf)
        )
        _order = np.argsort(_rr)[::-1]

        lines = []
        lines.append("=" * 72)
        lines.append("[BL] Optimal weight risk decomposition")
        lines.append(f"  Total portfolio variance: {_total_var:.6f}  "
                     f"(ann. vol ≈ {(_total_var**0.5)*252**0.5*100:.1f}%)")
        lines.append(f"  {'Ticker':>8}  {'w_star':>7}  {'w_curr':>7}  {'Δw':>7}  "
                     f"{'RC':>9}  {'RC%':>6}  {'mu_BL':>8}  {'mu/RC':>10}")
        lines.append("  " + "-" * 68)
        for _i in _order:
            _t     = tickers[_i]
            _ws    = float(w_star[_i])
            _wc    = float(w_current[_i])
            _rc_i  = float(_rc[_i])
            _rcp_i = float(_rc_pct[_i]) * 100
            _mu_i  = float(mu_BL[_i])
            _rr_i  = float(_rr[_i])
            _rr_s  = f"{_rr_i:>10.1f}" if np.isfinite(_rr_i) else f"{'inf':>10}"
            lines.append(f"  {_t:>8}  {_ws:>7.3f}  {_wc:>7.3f}  {_ws-_wc:>+7.3f}  "
                         f"  {_rc_i:>7.5f}  {_rcp_i:>5.1f}%  {_mu_i:>8.5f}  {_rr_s}")
        lines.append("=" * 72)
        diag_text = "\n".join(lines)
        print(diag_text)          # terminal
        # also write to file so dashboard can display it
        try:
            import pathlib
            _diag_path = pathlib.Path(__file__).parent.parent / "reports" / "bl" / "last_rc_diag.txt"
            _diag_path.parent.mkdir(parents=True, exist_ok=True)
            _diag_path.write_text(diag_text, encoding="utf-8")
        except Exception:
            pass

        return w_star, True
    else:
        print(f"[BL] Optimisation did not converge: {result.message}")
        return w_current, False


# ---------------------------------------------------------------------------
# 10. generate_rebalance_signals
# ---------------------------------------------------------------------------

def generate_rebalance_signals(
    tickers: list[str],
    w_current: np.ndarray,
    w_star: np.ndarray,
    mu_BL: np.ndarray,
    Pi: np.ndarray,
    Q: np.ndarray,
    lambda_implied: float,
    portfolio_value: float,
    consistency_scores: dict[str, float],
    threshold: float = 0.10,
    regime_summary: dict | None = None,
    regime_drift_df=None,
    asset_config: dict | None = None,
    vol_penalties: "dict[str, float] | None" = None,
    vol_history: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """
    Build a per-ticker rebalance signal DataFrame from BL outputs.

    Layer 1 — pure price-return optimisation.  Theta-related columns
    (theta_yield, theta_confidence, n_obs_theta) are not produced here;
    they belong to Layer 2.

    Rebalance flag is set when |w_star - w_current| > threshold.
    portfolio_rebalance_triggered is True when > 30% of tickers need rebalancing.

    Output columns
    --------------
    ticker, current_drift_regime_base, theta_eligible, current_weight,
    bl_weight, delta_weight, needs_rebalance, mu_equilibrium, q_view,
    price_return, mu_bl, price_confidence, vol_penalty, rsi_6,
    vol_percentile, discount, consistency_score, lambda_implied,
    target_delta_gamma, action, reason, portfolio_rebalance_triggered
    """
    rows = []
    rebalance_count = 0

    for i, ticker in enumerate(tickers):
        delta           = float(w_star[i] - w_current[i])
        needs_rebalance = abs(delta) > threshold
        if needs_rebalance:
            rebalance_count += 1

        q_val       = float(Q[i])
        mu_bl_val   = float(mu_BL[i])
        pi_val      = float(Pi[i])
        consistency = float(consistency_scores.get(ticker, 0.5))
        bl_w        = float(w_star[i])

        # Drift regime label for display
        _rs_info      = (regime_summary or {}).get(ticker, {})
        drift_regime  = str(_rs_info.get("current_drift_regime_base", "—")).upper()

        # Theta eligibility (from daily workflow → regime_lab_ticker_summary.json)
        _theta_eligible      = bool(_rs_info.get("theta_eligible", False))
        _theta_eligible_icon = "🟢" if _theta_eligible else "🔴"

        # Regime-based price return + confidence
        _rs = regime_summary if regime_summary is not None else {}
        price_ret, price_conf = compute_price_return_from_regime(
            ticker, _rs, regime_drift_df, asset_config=asset_config
        )

        # ── RSI + IV discount diagnostics (for CSV) ──────────────────────────
        _rsi_diag      = 50.0
        _vp_diag       = 0.5
        _discount_diag = 1.0
        # vol_percentile = 12M IV percentile (same source as compute_expected_returns)
        _iv_raw_diag = (_rs or {}).get(ticker, {}).get("iv_percentile", None)
        if _iv_raw_diag is not None:
            try:
                _vp_diag = float(_iv_raw_diag) / 100.0
            except (TypeError, ValueError):
                _vp_diag = 0.5
        if vol_history is not None:
            _rsi_s = vol_history[vol_history["ticker"] == ticker]["rsi_6"]
            if len(_rsi_s) > 0:
                try:
                    _r = float(_rsi_s.iloc[-1])
                    _rsi_diag = _r if not pd.isna(_r) else 50.0
                except (TypeError, ValueError):
                    pass
        _discount_diag = compute_rsi_vol_discount(_rsi_diag, _vp_diag)

        # ── action ──────────────────────────────────────────────────────────
        if bl_w == 0.0:
            action = "Liquidate"
        elif delta > threshold:
            action = "Increase"
        elif delta < -threshold:
            action = "Reduce"
        else:
            action = "Hold"

        # ── reason ──────────────────────────────────────────────────────────
        reasons: list[str] = []
        if bl_w == 0.0:
            if q_val < 0:
                reasons.append("Negative Q view")
        elif abs(delta) > threshold:
            if abs(delta) > 0.15:
                reasons.append(f"Large adjustment {delta*100:+.0f}%")
            if mu_bl_val > pi_val * 1.2 and pi_val > 0:
                reasons.append("BL expected significantly above equilibrium")
            elif mu_bl_val < pi_val * 0.8 and pi_val > 0:
                reasons.append("BL expected below equilibrium")
            if q_val < 0:
                reasons.append("Negative Q view")
            if consistency < 0.4:
                reasons.append("Low historical accuracy")
        else:
            if q_val < 0:
                reasons.append("Negative Q view — weight adequate")
            elif mu_bl_val > pi_val * 1.2 and pi_val > 0:
                reasons.append("BL above equilibrium — constraint-limited")
            else:
                reasons.append("Near-optimal allocation")

        reason = " | ".join(reasons) if reasons else "—"

        rows.append({
            "ticker":                  ticker,
            "current_drift_regime_base": drift_regime,
            "theta_eligible":          _theta_eligible_icon,
            "current_weight":          round(float(w_current[i]), 4),
            "bl_weight":               round(bl_w, 4),
            "delta_weight":            round(delta, 4),
            "needs_rebalance":         needs_rebalance,
            "mu_equilibrium":          round(pi_val, 4),
            "mu_bl":                   round(mu_bl_val, 4),
            "q_view":                  round(q_val, 4),
            "price_return":            round(price_ret, 6),
            "price_confidence":        round(price_conf, 4),
            "vol_penalty":             round(vol_penalties.get(ticker, 1.0) if vol_penalties else 1.0, 3),
            "rsi_6":                   round(_rsi_diag, 2),
            "vol_percentile":          round(_vp_diag, 4),
            "discount":                round(_discount_diag, 4),
            "consistency_score":       round(consistency, 3),
            "lambda_implied":          round(lambda_implied, 4),
            "target_delta_gamma":      round(bl_w * portfolio_value, 0),
            "action":                  action,
            "reason":                  reason,
        })

    df = pd.DataFrame(rows)

    # Portfolio-level rebalance warning
    rebalance_ratio = rebalance_count / max(len(tickers), 1)
    df["portfolio_rebalance_triggered"] = rebalance_ratio > 0.30

    return df


# ---------------------------------------------------------------------------
# 11. run_bl_optimization
# ---------------------------------------------------------------------------

def run_bl_optimization(
    hist_df: "pd.DataFrame | None",
    betas_df: "pd.DataFrame | None",
    reports_dir: Path,
    portfolio_value: float,
    cov_method: str = "kendall",
    lambda_override: float | None = None,
    beta_limit: float = 1.2,
    vol_limit_factor: float = 1.2,
    price_returns_df: "pd.DataFrame | None" = None,
) -> "pd.DataFrame | None":
    """
    Complete BL optimization pipeline.

    Reads all inputs from reports_dir, runs the full BL pipeline,
    writes reports/bl/bl_signals_YYYYMMDD.csv, and returns the
    signals DataFrame.  Returns None on any failure.
    """
    try:
        # ── 1. Load inputs ──────────────────────────────────────────────────
        inputs = load_bl_inputs(reports_dir)

        # ── 2. Valid tickers ────────────────────────────────────────────────
        tickers = get_valid_tickers(
            inputs["exposure_df"],
            inputs["asset_config"],
        )
        if not tickers:
            print("[BL] No valid tickers — skipping optimization")
            return None

        # ── 3. Current weights (delta_gamma_exposure normalized) ────────────
        sym_col = (
            "ticker" if "ticker" in inputs["exposure_df"].columns else "symbol"
        )
        exposure = inputs["exposure_df"].set_index(
            inputs["exposure_df"][sym_col].str.upper()
        )["delta_gamma_exposure"]
        dge = pd.Series({t: float(exposure.get(t, 0)) for t in tickers})
        total = dge.sum()
        if total <= 0:
            print("[BL] delta_gamma_exposure sum is zero")
            return None

        # ── 4. Price return matrix ──────────────────────────────────────────
        # Accept pre-built price_returns_df (dashboard path) or build from hist_df
        if price_returns_df is not None and not price_returns_df.empty:
            _pret_df = price_returns_df
        else:
            price_dict: dict[str, pd.Series] = {}
            if hist_df is not None and not hist_df.empty:
                _h = hist_df.copy()
                _h["_sym"]  = _h["symbol"].astype(str).str.upper()
                _h["_dt"]   = pd.to_datetime(_h["date"], errors="coerce")
                _h["close"] = pd.to_numeric(_h["close"], errors="coerce")
                _h = _h.dropna(subset=["_sym", "_dt", "close"])
                for sym, grp in _h.groupby("_sym"):
                    if sym in tickers and sym != "SHV":
                        price_dict[sym] = (
                            grp.sort_values("_dt")
                            .set_index("_dt")["close"]
                            .pct_change()
                        )
            _pret_df = pd.DataFrame(price_dict).dropna(how="all")

        # ── 4b. Persist price returns cache (keep workflow and dashboard consistent) ─
        if not _pret_df.empty:
            _cache_dir = Path(reports_dir) / "bl"
            _cache_dir.mkdir(exist_ok=True)
            _cache_path = _cache_dir / "price_returns_cache.csv"
            _pret_df.to_csv(_cache_path)
            print(f"[BL] Price returns cached → {_cache_path}")

        # ── 5. Covariance matrix (fallback to diagonal if no price data) ────
        cov_tickers_input = [t for t in tickers if t != "SHV"]
        if not _pret_df.empty and _pret_df.shape[1] >= 2:
            Sigma, cov_tickers = compute_covariance(
                _pret_df, cov_tickers_input, method=cov_method,
            )
        else:
            # Diagonal fallback: use realized_vol_20d from vol_history
            print("[BL] No price return data — using diagonal vol Sigma fallback")
            _vol_h = inputs["vol_history"]
            _vols  = np.array([
                float(_vol_h[_vol_h["ticker"] == t]["realized_vol_20d"].iloc[-1])
                if len(_vol_h[_vol_h["ticker"] == t]) > 0 else 0.20
                for t in cov_tickers_input
            ])
            Sigma       = np.diag(_vols ** 2)
            cov_tickers = cov_tickers_input[:]

        # SHV: near-zero variance, zero covariance with others
        if "SHV" in tickers:
            shv_var = (0.001) ** 2
            n_ext = len(cov_tickers) + 1
            Sigma_full = np.zeros((n_ext, n_ext))
            Sigma_full[: len(cov_tickers), : len(cov_tickers)] = Sigma
            Sigma_full[-1, -1] = shv_var
            Sigma       = Sigma_full
            cov_tickers = cov_tickers + ["SHV"]

        # Align tickers to covariance order
        tickers     = [t for t in tickers if t in cov_tickers]
        ticker_idx  = {t: i for i, t in enumerate(cov_tickers)}
        idx         = [ticker_idx[t] for t in tickers]
        Sigma       = Sigma[np.ix_(idx, idx)]
        w_current   = np.array(
            [float(dge.get(t, 0)) for t in tickers], dtype=float
        )
        w_current   = w_current / w_current.sum()

        # ── 5b. Persist Sigma cache (overwrite) ─────────────────────────────
        _sig_cache_dir = Path(reports_dir) / "bl"
        _sig_cache_dir.mkdir(exist_ok=True)
        np.save(str(_sig_cache_dir / "sigma_cache.npy"), Sigma)
        pd.Series(tickers).to_csv(
            _sig_cache_dir / "sigma_tickers.csv", index=False, header=True
        )
        print(f"[BL] Sigma ({Sigma.shape[0]}×{Sigma.shape[0]}) cached "
              f"→ {_sig_cache_dir / 'sigma_cache.npy'}")

        # ── 6. Consistency scores ───────────────────────────────────────────
        consistency_scores = compute_consistency_scores(
            tickers, inputs["entry_log"]
        )

        # ── 7. Expected returns Q (Layer 1: pure price return) ───────────────
        Q_series, price_omegas = compute_expected_returns(
            tickers,
            inputs["regime_summary"],
            inputs["vol_history"],
            inputs["asset_config"],
            regime_drift_df=inputs["regime_drift_df"],
        )
        Q = Q_series.reindex(tickers).fillna(0).values

        # ── 8. Equilibrium prior Π ──────────────────────────────────────────
        Pi, lambda_implied_calc = compute_equilibrium_pi(
            Sigma, w_current, Q,
            tickers=tickers,
            strategic_weights=inputs.get("strategic_weights", {}),
        )

        # ── 9. QQQ volatility (needed by compute_omega below) ────────────────
        qqq_vol_s = inputs["vol_history"][
            inputs["vol_history"]["ticker"] == "QQQ"
        ]["realized_vol_20d"]
        qqq_vol = float(qqq_vol_s.iloc[-1]) if len(qqq_vol_s) > 0 else 0.20

        # ── 10. Omega (Layer 1: price_omega × vol_penalty) ───────────────────
        Omega, vol_penalties = compute_omega(
            tickers,
            price_omegas,
            vol_history=inputs["vol_history"],
            qqq_vol=qqq_vol,
        )

        # ── 11. BL posterior ─────────────────────────────────────────────────
        mu_BL = compute_bl_posterior(Pi, Sigma, Q, Omega)

        # ── 12. Beta vector ─────────────────────────────────────────────────
        betas_dict: dict[str, float] = {}
        if betas_df is not None and not betas_df.empty:
            for _, row in betas_df.iterrows():
                sym  = str(row.get("symbol", "")).upper()
                beta = row.get("beta", 1.0)
                if sym and pd.notna(beta):
                    betas_dict[sym] = float(beta)

        # ── 13. Optimal weights ─────────────────────────────────────────────
        print(f"[BL DEBUG] calling compute_optimal_weights")
        print(f"[BL DEBUG] mu_BL: {mu_BL}")
        print(f"[BL DEBUG] w_current: {w_current}")
        print(f"[BL DEBUG] lambda: {lambda_implied_calc}")
        w_star, success = compute_optimal_weights(
            mu_BL, Sigma,
            lambda_implied_calc,
            w_current, tickers,
            betas_dict, qqq_vol,
            lambda_override=lambda_override,
            beta_limit=beta_limit,
            vol_limit_factor=vol_limit_factor,
        )

        # ── 14. Rebalance signals ───────────────────────────────────────────
        signals_df = generate_rebalance_signals(
            tickers=tickers,
            w_current=w_current,
            w_star=w_star,
            mu_BL=mu_BL,
            Pi=Pi,
            Q=Q,
            lambda_implied=lambda_implied_calc,
            portfolio_value=portfolio_value,
            consistency_scores=consistency_scores,
            regime_summary=inputs["regime_summary"],
            regime_drift_df=inputs["regime_drift_df"],
            asset_config=inputs["asset_config"],
            vol_penalties=vol_penalties,
            vol_history=inputs["vol_history"],
        )

        # ── 验证：打印每个 ticker 的 drift regime + price return + RSI + discount ──
        print("\n[BL] Layer-1 验证 (ticker | drift | rsi_6 | vol_pct | discount | price_return | q_view):")
        _hdr = "  %-8s  %-12s  %7s  %8s  %9s  %12s  %10s"
        print(_hdr % ("ticker", "drift_regime", "rsi_6", "vol_pct", "discount", "price_return", "q_view"))
        print("  " + "-" * 75)
        for _, _vrow in signals_df.iterrows():
            _t   = str(_vrow.get("ticker", ""))
            _dr  = str(_vrow.get("current_drift_regime_base", "?"))
            _rsi = _vrow.get("rsi_6",         float("nan"))
            _vp  = _vrow.get("vol_percentile", float("nan"))
            _dsc = _vrow.get("discount",        float("nan"))
            _pr  = _vrow.get("price_return",    float("nan"))
            _qv  = _vrow.get("q_view",          float("nan"))
            print(_hdr % (
                _t,
                _dr,
                "%.1f"   % _rsi              if pd.notna(_rsi) else "—",
                "%.1f%%" % (_vp * 100)       if pd.notna(_vp)  else "—",
                "%.3f"   % _dsc              if pd.notna(_dsc) else "—",
                "%+.2f%%" % (_pr * 100)      if pd.notna(_pr)  else "—",
                "%+.2f%%" % (_qv * 100)      if pd.notna(_qv)  else "—",
            ))

        # ── SHV bl_weight check ──────────────────────────────────────────────
        _shv_row = signals_df[signals_df["ticker"] == "SHV"]
        if not _shv_row.empty:
            _shv_bw = float(_shv_row["bl_weight"].iloc[0])
            _shv_cw = float(_shv_row["current_weight"].iloc[0])
            print(f"\n[BL] SHV: current_weight={_shv_cw:.3f}  bl_weight={_shv_bw:.3f}")
        print(f"[BL] lambda_implied = {lambda_implied_calc:.3f}")

        # ── 15. Write output ────────────────────────────────────────────────
        bl_dir = reports_dir / "bl"
        bl_dir.mkdir(exist_ok=True)
        today      = pd.Timestamp.today().strftime("%Y%m%d")
        output_path = bl_dir / f"bl_signals_{today}.csv"
        signals_df.to_csv(output_path, index=False)

        # ── 16. Console summary ─────────────────────────────────────────────
        print(f"\n{'='*50}")
        print(f"BL Optimization complete ({cov_method})")
        print(f"{'='*50}")
        print(f"Lambda implied : {lambda_implied_calc:.3f}")
        print(f"QQQ vol        : {qqq_vol*100:.1f}%")
        print(
            f"Portfolio rebalance triggered: "
            f"{signals_df['portfolio_rebalance_triggered'].iloc[0]}"
        )
        print(
            f"\n{'Ticker':<8} {'Curr%':>6} {'BL%':>6} "
            f"{'Δ%':>6} {'μ_BL':>7} {'Rebal':>6}"
        )
        print("-" * 45)
        for _, row in signals_df.iterrows():
            flag = "⚠️ " if row["needs_rebalance"] else "  "
            print(
                f"{row['ticker']:<8} "
                f"{row['current_weight']*100:>5.1f}% "
                f"{row['bl_weight']*100:>5.1f}% "
                f"{row['delta_weight']*100:>+5.1f}% "
                f"{row['mu_bl']*100:>+6.1f}% "
                f"{flag}"
            )
        print(f"{'='*50}\n")
        print(f"  ✅ BL signals saved → {output_path}")

        return signals_df

    except Exception as _e:
        import traceback
        print(f"[BL] Optimization failed: {_e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# _validate_outputs  (testing only — not for production use)
# ---------------------------------------------------------------------------

def _validate_outputs(
    tickers: list[str],
    w_current: np.ndarray,
    w_star: np.ndarray,
    mu_BL: np.ndarray,
    Pi: np.ndarray,
    lambda_implied: float,
    Sigma: np.ndarray,
) -> None:
    print("\n=== BL Step 2 Validation ===")
    print(f"Tickers ({len(tickers)}): {tickers}")
    print(f"Lambda implied: {lambda_implied:.3f}")

    print("\nΠ (equilibrium prior):")
    for t, v in zip(tickers, Pi):
        print(f"  {t:8s}: {v*100:+7.2f}%")

    print("\nμ_BL (posterior):")
    for t, v in zip(tickers, mu_BL):
        print(f"  {t:8s}: {v*100:+7.2f}%")

    print("\nCurrent → BL weights:")
    for t, wc, wb in zip(tickers, w_current, w_star):
        print(f"  {t:8s}: {wc*100:5.1f}% → {wb*100:5.1f}%")

    print(f"\nWeight sum : {w_star.sum():.6f}")
    port_vol = float(np.sqrt(w_star @ Sigma @ w_star))
    print(f"Port vol   : {port_vol*100:.1f}%")

    # Validation checks
    assert abs(w_star.sum() - 1.0) < 1e-3,   "FAIL: weights do not sum to 1"
    assert all(wb <= 0.701 for wb in w_star), "FAIL: weight > 70%"
    assert all(wb >= -1e-6 for wb in w_star), "FAIL: negative weight"
    assert 0.5 <= lambda_implied <= 20.0,      "FAIL: lambda out of range"
    assert all(v > -0.50 for v in mu_BL),     "FAIL: mu_BL below -50%"
    assert all(v < 2.00  for v in mu_BL),     "FAIL: mu_BL above +200%"
    print("\n✅ All assertion checks passed")
