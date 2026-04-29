from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import settings
from regime.regime_models import (
    run_regime_model_hybrid_v2,
    resolve_mixed_regimes,
)

TARGET_REGIMES = ("UPTREND", "DOWNTREND", "RANGE")
DIAG_ONLY_REGIMES = ("MIXED",)


# ---------------------------------------------------------------------------
# Inlined from dashboard/charts.py — no streamlit dependency
# ---------------------------------------------------------------------------

def _smooth_regime_labels_asymmetric(
    labels: list,
    window: int,
    downtrend_window: int = 2,
) -> list:
    """Asymmetric backward-looking smoother.

    DOWNTREND is treated with higher sensitivity: if any label in the short
    ``downtrend_window`` lookback starts with "DOWNTREND", the output flips
    immediately to that label.  All other regimes use the standard
    ``window``-day rolling mode.
    """
    from collections import Counter
    result: list = []
    for i in range(len(labels)):
        down_w = min(downtrend_window, i + 1)
        recent_short = labels[max(0, i - down_w + 1): i + 1]
        downtrend_labels = [l for l in recent_short if str(l).startswith("DOWNTREND")]
        if downtrend_labels:
            result.append(Counter(downtrend_labels).most_common(1)[0][0])
        else:
            normal_w = min(window, i + 1)
            recent = labels[max(0, i - normal_w + 1): i + 1]
            result.append(Counter(recent).most_common(1)[0][0])
    return result


# ---------------------------------------------------------------------------
# Helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _strip_regime_suffix(label: str) -> str:
    s = str(label).upper()
    for base in ("UPTREND", "DOWNTREND", "RANGE"):
        if s.startswith(base):
            return base
    return "MIXED"


@dataclass
class RegimeDriftCalibrationResult:
    summary_df: pd.DataFrame
    lookup: dict[str, dict[str, dict[str, float]]]


def _weighted_mu(avg5: float, avg10: float, avg20: float) -> float:
    vals = np.array([avg5, avg10, avg20], dtype=float)
    w = np.array(
        [
            float(settings.MC_REGIME_DRIFT_FWD_5_WEIGHT),
            float(settings.MC_REGIME_DRIFT_FWD_10_WEIGHT),
            float(settings.MC_REGIME_DRIFT_FWD_20_WEIGHT),
        ],
        dtype=float,
    )
    mask = np.isfinite(vals)
    if not mask.any():
        return np.nan
    ww = w[mask]
    vv = vals[mask]
    s = float(ww.sum())
    if s <= 0:
        return np.nan
    return float(np.dot(vv, ww / s))


def _shrink_weight(count: int) -> float:
    c = max(int(count), 0)
    min_n = int(settings.MC_REGIME_DRIFT_MIN_SAMPLES)
    shrink_n = max(int(settings.MC_REGIME_DRIFT_SHRINK_COUNT), min_n)
    if c < min_n:
        return 0.0
    return float(np.clip(c / max(shrink_n, 1), 0.0, 1.0))


_SMOOTH_WINDOW    = 5   # rolling-mode window for non-downtrend regimes
_DOWNTREND_WINDOW = 1   # fast-trigger window for downtrend detection
_MIN_SEGMENT_DAYS = 7   # segments shorter than this are merged into preceding regime


# ---------------------------------------------------------------------------
# Main calibration function (unchanged logic)
# ---------------------------------------------------------------------------

def calibrate_regime_conditional_drift(
    hist_df: pd.DataFrame,
    symbols: list[str],
    mu_base_by_symbol: dict[str, float],
    regime_df_map: "dict[str, pd.DataFrame] | None" = None,
) -> RegimeDriftCalibrationResult:
    rows: list[dict] = []
    lookup: dict[str, dict[str, dict[str, float]]] = {}

    if hist_df is None or hist_df.empty:
        return RegimeDriftCalibrationResult(summary_df=pd.DataFrame(rows), lookup=lookup)

    h = hist_df.copy()
    h["symbol"] = h["symbol"].astype(str).str.upper()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")

    for sym in [str(s).upper() for s in symbols]:
        hs = h[h["symbol"] == sym].copy().sort_values("date")
        mu_base = float(mu_base_by_symbol.get(sym, np.nan))

        try:
            if regime_df_map is not None and sym in regime_df_map:
                precomp = regime_df_map[sym]
                if precomp is None or (hasattr(precomp, "empty") and precomp.empty):
                    lookup[sym] = {}
                    continue
                d = precomp.copy()
                if "date" not in d.columns and getattr(d.index, "name", None) == "date":
                    d = d.reset_index()
            else:
                if hs.empty or not {"open", "high", "low", "close", "volume"}.issubset(hs.columns):
                    lookup[sym] = {}
                    continue
                ohlcv = hs.set_index("date")[["open", "high", "low", "close", "volume"]].copy()
                feat, _ = run_regime_model_hybrid_v2(ohlcv)
                feat = resolve_mixed_regimes(ohlcv, feat)
                if feat is None or feat.empty:
                    lookup[sym] = {}
                    continue
                d = feat.reset_index().rename(columns={"index": "date"}).copy()
        except Exception:
            lookup[sym] = {}
            continue

        if d is None or d.empty:
            lookup[sym] = {}
            continue
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d["close"] = pd.to_numeric(d["close"], errors="coerce")
        d["drift_regime"] = d["drift_regime"].astype(str).str.upper()
        if "drift_regime_label" not in d.columns:
            d["drift_regime_label"] = d["drift_regime"]
        if "drift_regime_source" not in d.columns:
            d["drift_regime_source"] = "daily"

        _raw_labels = d["drift_regime_label"].tolist()
        _smoothed = _smooth_regime_labels_asymmetric(
            _raw_labels,
            window=_SMOOTH_WINDOW,
            downtrend_window=_DOWNTREND_WINDOW,
        )
        d["drift_regime_label_smoothed"] = _smoothed
        d["drift_regime_base"] = (
            pd.Series(_smoothed, index=d.index).map(_strip_regime_suffix)
        )

        _base_list = d["drift_regime_base"].tolist()
        _filtered = _base_list.copy()
        _i = 0
        while _i < len(_base_list):
            _seg_base = _base_list[_i]
            _j = _i
            while _j < len(_base_list) and _base_list[_j] == _seg_base:
                _j += 1
            if (_j - _i) < _MIN_SEGMENT_DAYS and _i > 0:
                _prev = _filtered[_i - 1]
                for _k in range(_i, _j):
                    _filtered[_k] = _prev
            _i = _j
        d["drift_regime_base_filtered"] = _filtered

        d["fwd_5"]  = d["close"].shift(-5)  / d["close"] - 1.0
        d["fwd_10"] = d["close"].shift(-10) / d["close"] - 1.0
        d["fwd_20"] = d["close"].shift(-20) / d["close"] - 1.0

        sym_lookup: dict[str, dict[str, float]] = {}
        for regime in list(TARGET_REGIMES) + list(DIAG_ONLY_REGIMES):
            rg = d[d["drift_regime_base_filtered"] == regime].copy()
            if rg.empty:
                rows.append({
                    "symbol": sym, "regime": regime,
                    "is_target_regime": bool(regime in TARGET_REGIMES),
                    "count": 0,
                    "avg_fwd_5": np.nan, "med_fwd_5": np.nan,
                    "avg_fwd_10": np.nan, "med_fwd_10": np.nan,
                    "avg_fwd_20": np.nan, "med_fwd_20": np.nan,
                    "mu_regime_raw": np.nan, "mu_regime_final": mu_base,
                    "mu_regime_final_median": mu_base, "mu_base": mu_base,
                    "shrinkage_weight": 0.0, "shrinkage_note": "no_samples",
                    "used_for_mc": False,
                    "label_source_daily_pct": np.nan,
                    "label_source_weekly_pct": np.nan,
                    "label_source_monthly_pct": np.nan,
                    "label_source_fallback_pct": np.nan,
                })
                continue

            count_all      = int(rg.shape[0])
            count_complete = int(rg[["fwd_5", "fwd_10", "fwd_20"]].dropna().shape[0])
            fwd5_s  = pd.to_numeric(rg["fwd_5"],  errors="coerce")
            fwd10_s = pd.to_numeric(rg["fwd_10"], errors="coerce")
            fwd20_s = pd.to_numeric(rg["fwd_20"], errors="coerce")
            avg5  = float(fwd5_s.mean());  med5  = float(fwd5_s.median())
            avg10 = float(fwd10_s.mean()); med10 = float(fwd10_s.median())
            avg20 = float(fwd20_s.mean()); med20 = float(fwd20_s.median())
            mu_raw        = _weighted_mu(avg5, avg10, avg20)
            mu_raw_median = _weighted_mu(med5, med10, med20)
            w = _shrink_weight(count_complete)
            if np.isfinite(mu_raw):
                mu_final = float(w * mu_raw + (1.0 - w) * mu_base)
            else:
                mu_final = float(mu_base); w = 0.0
            if np.isfinite(mu_raw_median):
                mu_final_median = float(w * mu_raw_median + (1.0 - w) * mu_base)
            else:
                mu_final_median = float(mu_base)

            if not np.isfinite(mu_raw):
                note = "no_forward_returns"
            elif count_complete < int(settings.MC_REGIME_DRIFT_MIN_SAMPLES):
                note = "thin_data_baseline_dominant"
            elif w < 0.5:
                note = "partial_shrinkage"
            else:
                note = "data_driven"

            src_counts = rg["drift_regime_source"].value_counts() if "drift_regime_source" in rg.columns else pd.Series(dtype=int)
            def _src_pct(key: str) -> float:
                return round(float(src_counts.get(key, 0)) / max(count_all, 1) * 100, 1)

            rows.append({
                "symbol": sym, "regime": regime,
                "is_target_regime": bool(regime in TARGET_REGIMES),
                "count": count_complete, "count_raw_labels": count_all,
                "avg_fwd_5": avg5, "med_fwd_5": med5,
                "avg_fwd_10": avg10, "med_fwd_10": med10,
                "avg_fwd_20": avg20, "med_fwd_20": med20,
                "mu_regime_raw": mu_raw, "mu_regime_final": mu_final,
                "mu_regime_final_median": mu_final_median, "mu_base": mu_base,
                "shrinkage_weight": w, "shrinkage_note": note,
                "used_for_mc": bool(regime in TARGET_REGIMES),
                "label_source_daily_pct":    _src_pct("daily"),
                "label_source_weekly_pct":   _src_pct("weekly"),
                "label_source_monthly_pct":  _src_pct("monthly"),
                "label_source_fallback_pct": _src_pct("fallback"),
            })

            sym_lookup[regime] = {
                "count": float(count_complete),
                "avg_fwd_5": avg5, "avg_fwd_10": avg10, "avg_fwd_20": avg20,
                "mu_regime_raw": mu_raw, "mu_regime_final": mu_final,
                "mu_base": mu_base, "shrinkage_weight": w,
            }
        lookup[sym] = sym_lookup

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["symbol", "regime"]).reset_index(drop=True)
    return RegimeDriftCalibrationResult(summary_df=out, lookup=lookup)
