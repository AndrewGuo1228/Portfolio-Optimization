from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    REGIME_HYBRID_DIR_DOWN_MAX,
    REGIME_HYBRID_DIR_UP_MIN,
    REGIME_HYBRID_EXPANDING_LEVEL_MIN,
    REGIME_HYBRID_EXPANDING_PRESSURE_MIN,
    REGIME_HYBRID_HIGH_VOL_LEVEL_MIN,
    REGIME_HYBRID_LOOKBACK_DAYS,
    REGIME_HYBRID_LOW_VOL_LEVEL_MAX,
    REGIME_HYBRID_MIN_PERIODS,
    REGIME_HYBRID_MOD_ADX_PCTL,
    REGIME_HYBRID_MOD_GAP_PCTL,
    REGIME_HYBRID_MOD_PERSIST_PCTL,
    REGIME_HYBRID_STRONG_ADX_PCTL,
    REGIME_HYBRID_STRONG_GAP_PCTL,
    REGIME_HYBRID_STRONG_PERSIST_PCTL,
    REGIME_HYBRID_TREND_CUTOFF,
    REGIME_HYBRID_W_ADX,
    REGIME_HYBRID_W_GAP,
    REGIME_HYBRID_W_PERSIST,
    REGIME_MODEL_BASELINE_V1,
    REGIME_MODEL_HYBRID_V2,
    REGIME_PTREND2W_NEUTRAL_MAX,
    REGIME_PTREND2W_RANGE_FAVORABLE_MAX,
)
from regime.regime_detection import (
    build_features,
    compute_score_neutral,
    detect_regimes_blocks,
    trend_score_rules,
    trend_weight_alpha_theta,
)


def _rolling_pct_rank(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.rolling(window=window, min_periods=min_periods).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan,
        raw=False,
    )


def _clip01(x: pd.Series) -> pd.Series:
    return x.clip(lower=0.0, upper=1.0)


def _interpret_leverage_suitability_v2(drift_regime: str, vol_regime: str) -> str:
    if drift_regime == "RANGE" and vol_regime == "LOW_VOL":
        return "FAVORABLE_RANGE_ENTRY"
    if drift_regime == "RANGE" and vol_regime == "NORMAL_VOL":
        return "FAVORABLE_TO_NEUTRAL"
    if drift_regime == "RANGE" and vol_regime == "EXPANDING_VOL":
        return "CAUTION_VOL_EXPANDING"
    if drift_regime == "DOWNTREND" and vol_regime in {"EXPANDING_VOL", "HIGH_VOL"}:
        return "DANGEROUS"
    if drift_regime == "UPTREND" and vol_regime == "LOW_VOL":
        return "SELECTIVE_UPTREND_OK"
    if vol_regime == "HIGH_VOL":
        return "CAUTION_HIGH_VOL"
    return "NEUTRAL"


def evaluate_shared_leverage_entry_suitability(
    drift_regime: str,
    base_vol_regime: str,
    p_trend_2w: pd.Series | None,
    squeeze_detected: bool = False,
) -> dict[str, str]:
    # Strip multi-timeframe source suffix (_W / _M / _FB) so that
    # UPTREND_W, UPTREND_M, etc. are treated identically to UPTREND.
    drift = str(drift_regime).upper().split("_")[0]
    effective_vol = "SQUEEZE" if squeeze_detected else str(base_vol_regime).upper()

    # Evaluation order is explicit and shared across all dashboard paths:
    # 1) squeeze override -> effective vol regime
    # 2) dangerous states
    # 3) caution states
    # 4) favorable states with persistence
    # 5) otherwise NEUTRAL
    if drift == "DOWNTREND" and effective_vol in {"HIGH_VOL", "SQUEEZE"}:
        return {"effective_vol_regime": effective_vol, "leverage_entry_suitability": "DANGEROUS"}

    if drift == "DOWNTREND" or effective_vol in {"HIGH_VOL", "EXPANDING_VOL", "SQUEEZE"}:
        return {"effective_vol_regime": effective_vol, "leverage_entry_suitability": "RISK_CAUTION"}

    if drift == "RANGE" and effective_vol in {"LOW_VOL", "NORMAL_VOL"}:
        gated = _apply_favorable_persistence_gate("FAVORABLE_RANGE_ENTRY", p_trend_2w if p_trend_2w is not None else pd.Series(dtype=float))
        return {"effective_vol_regime": effective_vol, "leverage_entry_suitability": gated}

    if drift == "UPTREND" and effective_vol in {"LOW_VOL", "NORMAL_VOL"}:
        gated = _apply_favorable_persistence_gate("FAVORABLE_TO_NEUTRAL", p_trend_2w if p_trend_2w is not None else pd.Series(dtype=float))
        return {"effective_vol_regime": effective_vol, "leverage_entry_suitability": gated}

    # Explicit fallbacks:
    # - RANGE or UPTREND with failed persistence -> NEUTRAL (handled by gate)
    # - MIXED -> NEUTRAL by default
    # - anything not clearly favorable or dangerous -> NEUTRAL
    return {"effective_vol_regime": effective_vol, "leverage_entry_suitability": "NEUTRAL"}


def _persistence_ratio_below(series: pd.Series, threshold: float, window: int) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < window:
        return None
    tail = s.iloc[-window:]
    return float((tail < threshold).mean())


def _apply_favorable_persistence_gate(
    base_label: str,
    p_trend_2w: pd.Series,
    threshold: float = 0.40,
) -> str:
    s = pd.to_numeric(p_trend_2w, errors="coerce").dropna()
    if s.empty:
        return "UNKNOWN"
    current = float(s.iloc[-1])

    # Only gate the two favorable labels; other labels pass through unchanged.
    if base_label not in {"FAVORABLE_RANGE_ENTRY", "FAVORABLE_TO_NEUTRAL"}:
        return base_label

    if not (current < threshold):
        return "NEUTRAL"

    p20 = _persistence_ratio_below(s, threshold=threshold, window=20)
    if p20 is not None and p20 >= 0.95:
        return "FAVORABLE_RANGE_ENTRY"

    p10 = _persistence_ratio_below(s, threshold=threshold, window=10)
    if p10 is not None and p10 >= 0.90:
        return "FAVORABLE_TO_NEUTRAL"

    return "NEUTRAL"


def run_regime_model_baseline_v1(ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = detect_regimes_blocks(ohlcv)
    trend_score = trend_score_rules(out, adx_hi=28, gap_hi=1.2, persist_min=0.5)
    out["trend_score"] = trend_score
    score_neutral = compute_score_neutral(adx_hi=28, adx_mid=20, gap_hi=1.2, gap_mid=0.7)
    weights = trend_weight_alpha_theta(trend_score, score_neutral=score_neutral, scale=0.2)
    out = out.join(weights)
    out["p_trend_2w"] = out["p_trend"].rolling(window=10, min_periods=1).mean()
    out["drift_regime"] = out["state"].map({"TREND": "UPTREND", "RANGE": "RANGE"}).fillna("RANGE")
    out["vol_regime"] = "N/A"
    out["classification_layer"] = "BASELINE_RULES"
    out["reason_text"] = "Baseline V1 rule blocks."
    out["persist_sign"] = np.where(
        pd.to_numeric(out.get("vwap_sign_persist_5"), errors="coerce") >= 0,
        "ABOVE_VWAP",
        "BELOW_VWAP",
    )

    latest = out.iloc[-1]
    p2w = float(latest["p_trend_2w"]) if pd.notna(latest.get("p_trend_2w")) else np.nan
    if pd.isna(p2w):
        suitability = "UNKNOWN"
    elif p2w <= REGIME_PTREND2W_RANGE_FAVORABLE_MAX:
        suitability = "FAVORABLE_RANGE_ENTRY"
    elif p2w <= REGIME_PTREND2W_NEUTRAL_MAX:
        suitability = "NEUTRAL"
    else:
        suitability = "TREND_RISK_CAUTION"
    suitability = _apply_favorable_persistence_gate(suitability, out["p_trend_2w"])

    summary = {
        "model_version": REGIME_MODEL_BASELINE_V1,
        "current_trend_probability": float(latest["p_trend"]) if pd.notna(latest.get("p_trend")) else None,
        "current_trend_probability_2w": float(latest["p_trend_2w"]) if pd.notna(latest.get("p_trend_2w")) else None,
        "current_regime_state": latest.get("state"),
        "current_drift_regime": latest.get("drift_regime"),
        "current_vol_regime": latest.get("vol_regime"),
        "current_classification_layer": latest.get("classification_layer"),
        "leverage_entry_suitability": suitability,
        "reason_text": str(latest.get("reason_text", "Baseline V1 classification.")),
        "layer_pct_strong_rule": None,
        "layer_pct_moderate_rule": None,
        "layer_pct_score_fallback": None,
        "rows": len(out),
    }
    return out, summary


def run_regime_model_hybrid_v2(
    ohlcv: pd.DataFrame,
    lookback: int = REGIME_HYBRID_LOOKBACK_DAYS,
) -> tuple[pd.DataFrame, dict]:
    feat = build_features(ohlcv, atr_len=14, adx_len=14, vwap_win=30).copy()
    minp = max(int(REGIME_HYBRID_MIN_PERIODS), min(lookback, 40))

    feat["gap_abs"] = pd.to_numeric(feat["vwap_gap_norm_atr"], errors="coerce").abs()
    feat["persist_abs"] = pd.to_numeric(feat["vwap_sign_persist_5"], errors="coerce").abs()
    feat["persist_sign"] = np.where(
        pd.to_numeric(feat["vwap_sign_persist_5"], errors="coerce") >= 0,
        "ABOVE_VWAP",
        "BELOW_VWAP",
    )

    feat["adx_pct_rank"] = _rolling_pct_rank(feat["adx"], window=lookback, min_periods=minp)
    feat["gap_pct_rank"] = _rolling_pct_rank(feat["gap_abs"], window=lookback, min_periods=minp)
    feat["persist_pct_rank"] = _rolling_pct_rank(feat["persist_abs"], window=lookback, min_periods=minp)

    feat["atr_pct_rank"] = _rolling_pct_rank(feat["atr_pct"], window=lookback, min_periods=minp)
    feat["atr_expansion"] = feat["atr"] / feat["atr"].rolling(20, min_periods=10).mean()
    feat["atr_expansion_rank"] = _rolling_pct_rank(feat["atr_expansion"], window=lookback, min_periods=minp)

    ret1 = feat["close"].pct_change(1)
    feat["r5"] = feat["close"].pct_change(5)
    feat["r10"] = feat["close"].pct_change(10)
    feat["r20"] = feat["close"].pct_change(20)
    feat["direction_score_raw"] = 0.40 * feat["r5"] + 0.35 * feat["r10"] + 0.25 * feat["r20"]
    feat["direction_score"] = feat["direction_score_raw"].ewm(span=3, adjust=False).mean()

    feat["realized_vol"] = ret1.rolling(20, min_periods=10).std() * np.sqrt(252.0)
    feat["realized_vol_rank"] = _rolling_pct_rank(feat["realized_vol"], window=lookback, min_periods=minp)
    feat["realized_vol_expansion"] = feat["realized_vol"] / feat["realized_vol"].rolling(60, min_periods=20).mean()
    feat["realized_vol_expansion_rank"] = _rolling_pct_rank(
        feat["realized_vol_expansion"], window=lookback, min_periods=minp
    )

    strong_rule = (
        (feat["adx_pct_rank"] >= REGIME_HYBRID_STRONG_ADX_PCTL)
        & (feat["gap_pct_rank"] >= REGIME_HYBRID_STRONG_GAP_PCTL)
        & (feat["persist_pct_rank"] >= REGIME_HYBRID_STRONG_PERSIST_PCTL)
    )
    moderate_rule = (
        (feat["adx_pct_rank"] >= REGIME_HYBRID_MOD_ADX_PCTL)
        & (feat["gap_pct_rank"] >= REGIME_HYBRID_MOD_GAP_PCTL)
        & (feat["persist_pct_rank"] >= REGIME_HYBRID_MOD_PERSIST_PCTL)
    ) & (~strong_rule)

    trend_score = (
        REGIME_HYBRID_W_ADX * _clip01(feat["adx_pct_rank"])
        + REGIME_HYBRID_W_GAP * _clip01(feat["gap_pct_rank"])
        + REGIME_HYBRID_W_PERSIST * _clip01(feat["persist_pct_rank"])
    )
    trend_score = _clip01(trend_score)
    feat["trend_score"] = trend_score

    p_trend = trend_score.copy()
    p_trend = np.where(strong_rule, np.maximum(p_trend, 0.85), p_trend)
    p_trend = np.where(moderate_rule, np.maximum(p_trend, 0.65), p_trend)
    feat["p_trend"] = _clip01(pd.Series(p_trend, index=feat.index))
    feat["p_trend_2w"] = feat["p_trend"].rolling(window=10, min_periods=1).mean()
    feat["regime"] = (feat["p_trend"] >= REGIME_HYBRID_TREND_CUTOFF).astype(float)
    feat["state"] = feat["regime"].map({0.0: "RANGE", 1.0: "TREND"})

    feat["classification_layer"] = np.where(
        strong_rule,
        "STRONG_RULE",
        np.where(moderate_rule, "MODERATE_RULE", "SCORE_FALLBACK"),
    )

    # Drift regime (direction-aware)
    drift = np.where(
        feat["state"] == "RANGE",
        np.where(
            feat["direction_score"] > REGIME_HYBRID_DIR_UP_MIN,
            "MIXED",
            np.where(feat["direction_score"] < REGIME_HYBRID_DIR_DOWN_MAX, "MIXED", "RANGE"),
        ),
        np.where(
            feat["direction_score"] > REGIME_HYBRID_DIR_UP_MIN,
            "UPTREND",
            np.where(feat["direction_score"] < REGIME_HYBRID_DIR_DOWN_MAX, "DOWNTREND", "MIXED"),
        ),
    )
    feat["drift_regime"] = drift

    # Vol regime (expansion pressure explicitly matters)
    vol_level = 0.5 * _clip01(feat["atr_pct_rank"]) + 0.5 * _clip01(feat["realized_vol_rank"])
    vol_pressure = (
        0.5 * _clip01(feat["atr_expansion_rank"])
        + 0.3 * _clip01(feat["realized_vol_expansion_rank"])
        + 0.2 * _clip01(feat["atr_pct_rank"])
    )
    feat["vol_level_score"] = _clip01(vol_level)
    feat["vol_pressure_score"] = _clip01(vol_pressure)

    vol_regime = np.where(
        (feat["vol_pressure_score"] >= REGIME_HYBRID_EXPANDING_PRESSURE_MIN)
        & (feat["vol_level_score"] >= REGIME_HYBRID_EXPANDING_LEVEL_MIN),
        "EXPANDING_VOL",
        np.where(
            feat["vol_level_score"] >= REGIME_HYBRID_HIGH_VOL_LEVEL_MIN,
            "HIGH_VOL",
            np.where(feat["vol_level_score"] <= REGIME_HYBRID_LOW_VOL_LEVEL_MAX, "LOW_VOL", "NORMAL_VOL"),
        ),
    )
    feat["vol_regime"] = vol_regime

    feat["leverage_entry_suitability"] = [
        evaluate_shared_leverage_entry_suitability(
            drift_regime=d,
            base_vol_regime=v,
            p_trend_2w=feat["p_trend_2w"].iloc[: i + 1],
            squeeze_detected=False,
        )["leverage_entry_suitability"]
        for i, (d, v) in enumerate(zip(feat["drift_regime"], feat["vol_regime"]))
    ]
    feat["reason_text"] = (
        "layer="
        + feat["classification_layer"].astype(str)
        + "; trend="
        + feat["trend_score"].round(2).astype(str)
        + "; dir="
        + feat["direction_score"].round(3).astype(str)
        + "; vol="
        + feat["vol_regime"].astype(str)
    )

    latest = feat.iloc[-1]
    layer_mix = feat["classification_layer"].value_counts(normalize=True)
    summary = {
        "model_version": REGIME_MODEL_HYBRID_V2,
        "current_trend_probability": float(latest.get("p_trend")) if pd.notna(latest.get("p_trend")) else None,
        "current_trend_probability_2w": float(latest.get("p_trend_2w")) if pd.notna(latest.get("p_trend_2w")) else None,
        "current_regime_state": latest.get("state"),
        "current_drift_regime": latest.get("drift_regime"),
        "current_vol_regime": latest.get("vol_regime"),
        "current_classification_layer": latest.get("classification_layer"),
        "leverage_entry_suitability": latest.get("leverage_entry_suitability"),
        "reason_text": str(latest.get("reason_text", "")),
        "layer_pct_strong_rule": float(layer_mix.get("STRONG_RULE", 0.0) * 100.0),
        "layer_pct_moderate_rule": float(layer_mix.get("MODERATE_RULE", 0.0) * 100.0),
        "layer_pct_score_fallback": float(layer_mix.get("SCORE_FALLBACK", 0.0) * 100.0),
        "rows": len(feat),
    }
    return feat, summary


# ---------------------------------------------------------------------------
# Multi-timeframe MIXED resolver
# ---------------------------------------------------------------------------

_VALID_REGIMES: frozenset[str] = frozenset({"UPTREND", "RANGE", "DOWNTREND"})

_SOURCE_SUFFIX: dict[str, str] = {
    "daily":    "",      # no suffix — original daily label
    "weekly":   "_W",
    "monthly":  "_M",
    "fallback": "_FB",
}


def _build_rolling_bars(
    daily_ohlcv: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Build a series of trailing-window synthetic OHLCV bars.

    For each trading day t, the synthetic bar aggregates the trailing
    `window` calendar days ending at t (inclusive):
        open   = open  of t-(window-1)   [first bar in the window]
        high   = max(high  over window)
        low    = min(low   over window)
        close  = close of t              [current day]
        volume = sum(volume over window)

    This is fully backward-looking — no forward bias.
    Rows with fewer than `window` preceding bars are dropped.
    """
    df = daily_ohlcv.copy()
    result = pd.DataFrame(index=df.index)
    result["open"]   = df["open"].shift(window - 1)
    result["high"]   = df["high"].rolling(window, min_periods=window).max()
    result["low"]    = df["low"].rolling(window, min_periods=window).min()
    result["close"]  = df["close"]
    result["volume"] = df["volume"].rolling(window, min_periods=window).sum()
    return result.dropna()


def resolve_mixed_regimes(
    ohlcv: pd.DataFrame,
    daily_feat: pd.DataFrame,
    lookback_weekly: int = 52,
    lookback_monthly: int = 12,
    window_weekly: int = 5,
    window_monthly: int = 21,
) -> pd.DataFrame:
    """
    Resolve MIXED drift-regime labels using progressively longer timeframes.

    Algorithm (no forward bias at any level):
      1. For each day already labelled UPTREND / RANGE / DOWNTREND → keep as-is.
      2. For MIXED days → build trailing `window_weekly`-day synthetic bars,
         run hybrid_v2 with lookback=lookback_weekly.
         If the result is a valid 3-class label → use it (source = "weekly").
      3. If still MIXED → build trailing `window_monthly`-day synthetic bars,
         run hybrid_v2 with lookback=lookback_monthly.
         If valid → use it (source = "monthly").
      4. Still MIXED → RANGE (source = "fallback").

    Output columns added / replaced:
        drift_regime        — resolved 3-class label: UPTREND / RANGE / DOWNTREND
                              (replaces original which could be MIXED)
        drift_regime_source — origin of the label: daily / weekly / monthly / fallback
        drift_regime_label  — display label with suffix:
                              UPTREND | UPTREND_W | UPTREND_M
                              RANGE   | RANGE_W   | RANGE_M   | RANGE_FB
                              DOWNTREND | DOWNTREND_W | DOWNTREND_M

    All other columns in daily_feat are preserved unchanged.
    """
    feat = daily_feat.copy()

    # Initialise source columns for non-MIXED rows
    feat["drift_regime_source"] = "daily"
    feat["drift_regime_label"]  = feat["drift_regime"].astype(str)

    mixed_mask = feat["drift_regime"] == "MIXED"
    if not mixed_mask.any():
        return feat

    # ── Direction classifier for weekly / monthly resolution ─────────────
    #
    # We do NOT reuse hybrid_v2 here because hybrid_v2 can itself produce MIXED,
    # which defeats the purpose of the resolution layer.  Instead we use a simple
    # net-return classifier on the synthetic bars:
    #
    #   net_return = close / open - 1  (already encoded in the rolling bar)
    #   EWM-smoothed over span=3 to reduce day-to-day noise.
    #
    # Deadbands (annualised ~1 vol day):
    #   weekly  (5-bar)  : ±0.5 %  → catches most directional weeks
    #   monthly (21-bar) : ±1.5 %  → broader but still decisive
    #
    # Returns: pd.Series of "UPTREND" / "RANGE" / "DOWNTREND" aligned on
    # daily_feat.index.  This classifier NEVER returns MIXED.

    def _direction_classify(
        bars: pd.DataFrame,
        deadband: float,
    ) -> pd.Series:
        """
        Classify each synthetic bar as UPTREND / RANGE / DOWNTREND.

        net_return = close/open - 1  (window cumulative return)
        EWM-smoothed (span=3) to reduce noise at bar boundaries.
        Result is re-indexed onto the full daily index (NaN outside bar range).
        """
        try:
            net_ret = (bars["close"] / bars["open"].replace(0, float("nan"))) - 1.0
            smoothed = net_ret.ewm(span=3, adjust=False).mean()
            labels = pd.Series("RANGE", index=bars.index, dtype=str)
            labels[smoothed >  deadband] = "UPTREND"
            labels[smoothed < -deadband] = "DOWNTREND"
            # Re-index onto full daily index; missing rows become NaN
            return labels.reindex(feat.index)
        except Exception:
            return pd.Series(dtype=str)

    weekly_bars  = _build_rolling_bars(ohlcv, window_weekly)
    monthly_bars = _build_rolling_bars(ohlcv, window_monthly)

    weekly_regimes  = _direction_classify(weekly_bars,  deadband=0.005)   # ±0.5 %
    monthly_regimes = _direction_classify(monthly_bars, deadband=0.015)   # ±1.5 %

    # ── Resolve each MIXED date ───────────────────────────────────────────
    for date in feat.index[mixed_mask]:
        resolved = None
        source   = None

        # 1. Try weekly direction
        if date in weekly_regimes.index:
            w = str(weekly_regimes.loc[date])
            if w in _VALID_REGIMES:
                resolved, source = w, "weekly"

        # 2. Try monthly direction
        if resolved is None and date in monthly_regimes.index:
            m = str(monthly_regimes.loc[date])
            if m in _VALID_REGIMES:
                resolved, source = m, "monthly"

        # 3. Fallback — only if both windows are also flat (very rare)
        if resolved is None:
            resolved, source = "RANGE", "fallback"

        suffix = _SOURCE_SUFFIX[source]
        feat.loc[date, "drift_regime"]        = resolved
        feat.loc[date, "drift_regime_source"] = source
        feat.loc[date, "drift_regime_label"]  = resolved + suffix

    return feat
