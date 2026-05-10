"""
bl_relative_views.py
====================
Black-Litterman with Regime-Aware Relative (Paired) Views — Core Implementation
ORIE 5730  |  Portfolio Optimisation Project

This file contains the three core components that distinguish our approach
from a standard Black-Litterman model:

  1. Regime scoring  — continuous [-1, +1] signal via t³ cubic blend
  2. Relative view construction  — "asset i outperforms asset j" (P ≠ I)
  3. BL posterior + CVXPY optimizer — general form supporting any P matrix

─────────────────────────────────────────────────────────────────────────────
Standard BL uses absolute views (P = I):
    each row of P selects a single asset  →  "asset i will return q_i"

Our extension uses RELATIVE (paired) views:
    each row of P has +1 (outperformer) and -1 (underperformer)
    →  "asset i will OUTPERFORM asset j by q_ij"

This design forces the model to express genuine cross-sectional conviction
rather than making independent per-asset bets, and naturally pairs regime
leaders with regime laggards.
─────────────────────────────────────────────────────────────────────────────

Full backtest pipeline:  backtest/runner_v2.py
CVXPY optimiser (reference):  bl/cvxpy_optimizer.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
    _CVXPY_AVAILABLE = True
except ImportError:
    _CVXPY_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — REGIME SCORING
#
# Each asset's regime state is converted to a continuous score in [-1, +1].
# We use a t³ (cubic) blend during regime transitions to avoid step-function
# jumps in views on rebalance dates.
#
# Transition windows (asymmetric, to match market dynamics):
#   • Entering DOWNTREND  →  5 days   (faster risk-off)
#   • Exiting  DOWNTREND  →  10 days  (slower risk-on)
#   • All other changes   →  7 days
#
# Confidence formula:  c_i = 1 − exp(−days_in_regime / 7)
#   →  rises from 0 toward 1 the longer a regime persists
# ═════════════════════════════════════════════════════════════════════════════

_BASE_SCORE = {"UPTREND": 1.0, "RANGE": 0.0, "DOWNTREND": -1.0, "MIXED": 0.0}

_TRANS_DAYS = {
    ("OTHER",      "DOWNTREND"): 5,   # entering bear:  fast
    ("DOWNTREND",  "OTHER"):     10,  # exiting  bear:  slow
}
_DEFAULT_TRANS = 7


def _trans_days(current: str, prev: str) -> int:
    """Return asymmetric transition window for a regime change."""
    if current == "DOWNTREND" and prev != "DOWNTREND":
        return _TRANS_DAYS[("OTHER", "DOWNTREND")]
    if prev == "DOWNTREND" and current != "DOWNTREND":
        return _TRANS_DAYS[("DOWNTREND", "OTHER")]
    return _DEFAULT_TRANS


def regime_score(snap: dict) -> float:
    """
    Map a regime snapshot to a continuous score in [-1, +1].

    During a transition (days < trans_days), interpolate between the previous
    and current score using a t³ (cubic ease-in) curve:

        score = (1 - t³) * s_prev + t³ * s_curr,   where t = days / trans_days

    A cubic curve spends more time near the previous state and accelerates
    toward the new state, which mirrors typical momentum dynamics.

    Parameters
    ----------
    snap : dict
        Keys: current (str), prev (str), days (int), stable (bool)
        Produced by get_regime_snapshot() in runner_v2.py

    Returns
    -------
    float in [-1, +1]
    """
    current = snap["current"]
    prev    = snap["prev"]
    days    = snap["days"]
    stable  = snap["stable"]

    s_curr = _BASE_SCORE.get(current, 0.0)
    s_prev = _BASE_SCORE.get(prev,    0.0)

    if stable or prev == current:
        return s_curr

    tdays   = _trans_days(current, prev)
    t_cubed = (days / tdays) ** 3
    return (1 - t_cubed) * s_prev + t_cubed * s_curr


def regime_confidence(snap: dict, conf_denom: float = 7.0) -> float:
    """
    Exponential confidence in the current regime label.

        c = 1 − exp(−days / conf_denom)

    With conf_denom = 7: c reaches ~0.50 after 5 days, ~0.75 after 10 days,
    ~0.86 after 14 days.

    This scales view magnitudes so that very fresh regime calls have minimal
    impact and only well-established regimes drive large tilts.
    """
    return float(1.0 - np.exp(-snap["days"] / conf_denom))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RELATIVE (PAIRED) VIEW CONSTRUCTION
#
# For every pair (i, j) where  score_i − score_j ≥ min_score_gap,
# we create one row of P:
#
#   P[row, i] = +1    (outperformer)
#   P[row, j] = -1    (underperformer)
#
# The view magnitude is:
#   q_ij = (r_i * disc_i − r_j * disc_j) * avg_conf
#
# where:
#   r_i     = regime-conditional annualised return for asset i
#   disc_i  = RSI + realised-vol overbought discount ∈ [0.20, 1.00]
#   avg_conf = (conf_i + conf_j) / 2   (regime confidence)
#
# View uncertainty (diagonal Omega):
#   ω_ij = (σ_i² + σ_j²) / avg_conf
#
# Interpretation:  higher realised vol or lower confidence → wider uncertainty
# → the BL posterior tilts less aggressively toward that view.
# ═════════════════════════════════════════════════════════════════════════════

def build_relative_views(
    tickers: list[str],
    regime_snap: dict[str, dict],
    eff_ret: dict[str, float],
    discounts: dict[str, float],
    rv_ann: dict[str, float],
    conf_denom: float = 7.0,
    min_score_gap: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct relative BL views: "asset i outperforms asset j".

    Parameters
    ----------
    tickers       : ordered list of n asset tickers
    regime_snap   : {ticker: snap_dict}  (see regime_score for snap format)
    eff_ret       : {ticker: annualised regime-conditional return}
    discounts     : {ticker: overbought discount in [0.20, 1.00]}
    rv_ann        : {ticker: annualised 20-day realised volatility}
    conf_denom    : exponential decay denominator for confidence (default 7)
    min_score_gap : minimum regime-score difference to form a pair (default 0.30)

    Returns
    -------
    P     : (K, n)  view matrix  — K rows, each with +1 and -1
    Q_vec : (K,)    view return vector (annualised outperformance)
    Omega : (K, K)  diagonal uncertainty matrix
    """
    n = len(tickers)

    # Per-asset regime signals
    scores = {}
    confs  = {}
    for t in tickers:
        snap = regime_snap.get(t)
        if snap:
            scores[t] = regime_score(snap)
            confs[t]  = regime_confidence(snap, conf_denom)
        else:
            scores[t] = 0.0
            confs[t]  = 0.10          # low confidence when no regime data

    P_rows: list[np.ndarray] = []
    Q_rows: list[float]      = []
    O_rows: list[float]      = []

    for ii, ti in enumerate(tickers):
        for jj, tj in enumerate(tickers):
            if ii == jj:
                continue

            gap = scores[ti] - scores[tj]
            if gap < min_score_gap:
                continue                   # not enough regime divergence

            # ── View magnitude ────────────────────────────────────────────
            # Regime-conditional expected outperformance, scaled by:
            #   (a) overbought discount  — reduces views for stretched assets
            #   (b) average confidence   — reduces views for fresh regime calls
            avg_conf = (confs[ti] + confs[tj]) / 2.0
            q_ij = (
                eff_ret.get(ti, 0.08) * discounts.get(ti, 1.0)
                - eff_ret.get(tj, 0.08) * discounts.get(tj, 1.0)
            ) * avg_conf

            # ── View uncertainty ──────────────────────────────────────────
            # Proportional to sum of realized variances, inversely scaled by
            # confidence so low-conviction pairs get wider uncertainty bands.
            sigma_i = rv_ann.get(ti, 0.20)
            sigma_j = rv_ann.get(tj, 0.20)
            omega_ij = (sigma_i ** 2 + sigma_j ** 2) / max(avg_conf, 0.05)

            row        = np.zeros(n)
            row[ii]    = +1.0
            row[jj]    = -1.0
            P_rows.append(row)
            Q_rows.append(q_ij)
            O_rows.append(omega_ij)

    # Fallback: if no regime pairs qualify, place near-zero absolute views
    # so the BL posterior stays close to the equilibrium prior.
    if not P_rows:
        P_rows = list(np.eye(n))
        Q_rows = [0.06] * n
        O_rows = [0.04] * n

    return np.array(P_rows), np.array(Q_rows), np.diag(O_rows)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BLACK-LITTERMAN POSTERIOR (GENERAL FORM)
#
# Standard formula (He & Litterman 1999):
#
#   μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹  ·  [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]
#
# This is identical for both absolute views (P = I, K = n) and relative
# views (P general, K = number of pairs).  The only difference is that
# relative views do NOT directly specify individual asset returns; they
# constrain *differences*, so the posterior is determined by the pull
# between paired assets AND the equilibrium prior simultaneously.
#
# Parameters used in our backtest:
#   τ = 1.0    (views weighted equally with the equilibrium prior)
#   λ = 2.5    (fixed risk-aversion; prevents degenerate clamping to λ=10)
# ═════════════════════════════════════════════════════════════════════════════

def bl_posterior(
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    Sigma: np.ndarray,
    w_eq: np.ndarray,
    tau: float = 1.0,
    lam_fix: float | None = 2.5,
) -> tuple[np.ndarray, float]:
    """
    Compute the Black-Litterman posterior expected returns.

    Supports both absolute views (P = I) and relative views
    (general P with +1 / -1 entries for cross-asset comparisons).

    Parameters
    ----------
    P      : (K, n)  view matrix
    Q      : (K,)    view return vector
    Omega  : (K, K)  diagonal view uncertainty matrix
    Sigma  : (n, n)  Kendall-tau asset covariance (rolling 252-day)
    w_eq   : (n,)    equilibrium weights (prior portfolio)
    tau    : float   prior uncertainty scaling (τ=1.0 → views equal weight)
    lam_fix: float   fixed risk-aversion coefficient (None → infer from data)

    Returns
    -------
    mu_BL : (n,)  posterior expected returns
    lam   :       risk-aversion coefficient used
    """
    n = Sigma.shape[0]

    # Implied equilibrium returns: Π = λ Σ w_eq
    pvar = float(w_eq @ Sigma @ w_eq)
    if lam_fix is not None:
        lam = lam_fix
    else:
        pret = float(Q.mean())
        lam  = float(np.clip(pret / (2 * pvar + 1e-12), 0.5, 10.0))

    Pi = lam * Sigma @ w_eq

    # Posterior precision matrices
    tSinv   = np.linalg.solve(tau * Sigma + 1e-8 * np.eye(n), np.eye(n))
    K       = len(Q)
    Oinv    = np.linalg.solve(Omega + 1e-10 * np.eye(K), np.eye(K))
    PtOinvP = P.T @ Oinv @ P

    # Combined precision and RHS
    M   = tSinv + PtOinvP
    rhs = tSinv @ Pi + P.T @ Oinv @ Q

    mu_BL = np.linalg.solve(M + 1e-8 * np.eye(n), rhs)
    return mu_BL, lam


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CONSTRAINED MV OPTIMISATION (CVXPY / OSQP)
#
# Given the BL posterior μ_BL and covariance Σ, solve the QCQP:
#
#   max_w   w'μ_BL − (λ/2) w'Σw
#   s.t.    Σ w_i = 1          (fully invested)
#           0 ≤ w_i ≤ 0.70     (long-only, position cap)
#           β'w ≤ 1.20         (market-beta ceiling)
#           w'Σw ≤ (1.2·σ_QQQ)² (portfolio-vol ceiling)
#
# The vol and beta ceilings are hard constraints, not penalties, so the
# solver guarantees feasibility at every rebalance.  OSQP (an ADMM-based
# QP solver) handles the quad_form constraint efficiently.
# ═════════════════════════════════════════════════════════════════════════════

def optimize_mv_cvxpy(
    mu_bl: np.ndarray,
    Sigma: np.ndarray,
    betas: np.ndarray,
    qqq_vol: float,
    lam: float = 2.5,
    beta_limit: float = 1.20,
    vol_limit_factor: float = 1.20,
    w_max: float = 0.70,
) -> np.ndarray:
    """
    Solve the constrained mean-variance problem using CVXPY + OSQP.

    Parameters
    ----------
    mu_bl          : (n,)   BL posterior expected returns (annualised)
    Sigma          : (n, n) asset covariance matrix
    betas          : (n,)   asset betas vs SPY
    qqq_vol        : float  annualised realised volatility of QQQ (benchmark)
    lam            : float  risk-aversion coefficient
    beta_limit     : float  portfolio-beta upper bound (default 1.20)
    vol_limit_factor: float portfolio vol ≤ qqq_vol × factor (default 1.20)
    w_max          : float  per-asset weight cap (default 0.70)

    Returns
    -------
    w_star : (n,) optimal weights, summing to 1, non-negative
    """
    if not _CVXPY_AVAILABLE:
        raise ImportError(
            "cvxpy is not installed.  Run:  pip install cvxpy\n"
            "A projected-gradient fallback is available in runner_v2.py."
        )

    n          = len(mu_bl)
    vol_max_sq = (qqq_vol * vol_limit_factor) ** 2

    w = cp.Variable(n, nonneg=True)

    # Objective: maximise risk-adjusted expected return
    objective = cp.Maximize(
        mu_bl @ w - (lam / 2) * cp.quad_form(w, Sigma)
    )

    # Hard constraints
    constraints = [
        cp.sum(w) == 1,                        # fully invested
        w <= w_max,                            # position cap
        betas @ w <= beta_limit,               # market-beta ceiling
        cp.quad_form(w, Sigma) <= vol_max_sq,  # volatility ceiling
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-8, eps_rel=1e-8)

    if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and w.value is not None:
        result = np.clip(w.value, 0.0, w_max)
        total  = result.sum()
        return result / total if total > 1e-10 else np.ones(n) / n

    # Fallback: equal-weight on solver failure
    return np.ones(n) / n


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MINIMAL SELF-CONTAINED DEMO
#
# Demonstrates the full pipeline on a toy 3-asset example.
# No market data required — shows how P, Q, Omega are constructed
# and how the BL posterior shifts weights away from equilibrium.
# ═════════════════════════════════════════════════════════════════════════════

def demo():
    """
    Toy 3-asset example: XLK (tech), XLU (utilities), GLD (gold).

    Assumed regime snapshot:
      XLK  → UPTREND  (stable, 12 days)
      XLU  → RANGE    (stable,  8 days)
      GLD  → DOWNTREND (stable, 6 days)

    Relative views constructed:
      XLK outperforms XLU  (score gap = 1.0 − 0.0 = 1.0 ≥ 0.30)
      XLK outperforms GLD  (score gap = 1.0 − (−1.0) = 2.0 ≥ 0.30)
      XLU outperforms GLD  (score gap = 0.0 − (−1.0) = 1.0 ≥ 0.30)
    """
    print("=" * 60)
    print("BL RELATIVE VIEWS — DEMO (3-asset toy example)")
    print("=" * 60)

    tickers = ["XLK", "XLU", "GLD"]
    n = len(tickers)

    # Regime snapshots
    regime_snap = {
        "XLK": {"current": "UPTREND",   "prev": "UPTREND",   "days": 12, "stable": True},
        "XLU": {"current": "RANGE",      "prev": "RANGE",     "days": 8,  "stable": True},
        "GLD": {"current": "DOWNTREND",  "prev": "RANGE",     "days": 6,  "stable": False},
    }

    # Effective returns (regime-conditional historical mean, annualised)
    eff_ret = {"XLK": 0.18, "XLU": 0.06, "GLD": 0.04}

    # Overbought discounts (RSI + realised-vol; 1.0 = no discount)
    discounts = {"XLK": 0.85, "XLU": 1.00, "GLD": 1.00}

    # Annualised 20-day realised vols
    rv_ann = {"XLK": 0.22, "XLU": 0.14, "GLD": 0.17}

    # ── Step 1: Build relative views ─────────────────────────────────────────
    P, Q, Omega = build_relative_views(
        tickers, regime_snap, eff_ret, discounts, rv_ann,
        conf_denom=7.0, min_score_gap=0.30,
    )

    print(f"\nNumber of views (K) : {len(Q)}")
    print(f"P matrix shape      : {P.shape}")
    for k in range(len(Q)):
        pos = [tickers[i] for i in range(n) if P[k, i] > 0]
        neg = [tickers[i] for i in range(n) if P[k, i] < 0]
        print(f"  View {k+1}: {pos[0]:>4} outperforms {neg[0]:<4}  |  "
              f"q = {Q[k]*100:+.2f}%  ω = {Omega[k,k]:.4f}")

    # ── Step 2: BL posterior ──────────────────────────────────────────────────
    # Simple diagonal covariance (annualised variances)
    Sigma = np.diag([rv_ann[t] ** 2 for t in tickers])
    w_eq  = np.array([1/3, 1/3, 1/3])     # equal-weight prior

    mu_BL, lam = bl_posterior(P, Q, Omega, Sigma, w_eq, tau=1.0, lam_fix=2.5)

    print(f"\nBL posterior μ (annualised):")
    for t, m in zip(tickers, mu_BL):
        print(f"  {t}: {m*100:+.2f}%")

    # ── Step 3: CVXPY optimisation ────────────────────────────────────────────
    if _CVXPY_AVAILABLE:
        betas   = np.array([1.15, 0.40, 0.10])
        qqq_vol = 0.22

        w_star = optimize_mv_cvxpy(mu_BL, Sigma, betas, qqq_vol, lam=lam)

        print(f"\nOptimal weights (CVXPY / OSQP):")
        for t, w in zip(tickers, w_star):
            print(f"  {t}: {w*100:.1f}%")
        port_vol = float(np.sqrt(w_star @ Sigma @ w_star))
        print(f"  Portfolio vol (ann): {port_vol*100:.1f}%")
    else:
        print("\n[cvxpy not installed — skipping optimisation step]")

    print("=" * 60)


if __name__ == "__main__":
    demo()
