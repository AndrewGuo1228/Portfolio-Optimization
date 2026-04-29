"""
regime_detection.py
Two ways to label daily TREND vs RANGE:

A) Rules + block persistence (your original approach)
   - build_features() -> predict_daily() -> smooth_blocks()
B) KMeans + sliding-window (probability) smoother
   - build_features() -> detect_regimes_kmeans() with trend_prob + window smooth

Plot helper:
- plot_regime_dots(...)  # one colored dot per day (blue RANGE, red TREND)

Input df: columns ['open','high','low','close','volume'] with DatetimeIndex.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Wilder helpers (ATR / ADX)
# ---------------------------
def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift(1)
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)

def _wilder_rma(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1/n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _wilder_rma(_true_range(df['high'], df['low'], df['close']), n)

def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    h, l, c = df['high'], df['low'], df['close']
    up   = h.diff()
    down = -l.diff()  # = (l.shift(1) - l)

    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr        = _true_range(h, l, c)
    tr_rma    = _wilder_rma(pd.Series(tr, index=df.index), n)
    plus_rma  = _wilder_rma(pd.Series(plus_dm, index=df.index), n)
    minus_rma = _wilder_rma(pd.Series(minus_dm, index=df.index), n)

    di_plus  = 100.0 * (plus_rma  / tr_rma).replace([np.inf, -np.inf], np.nan)
    di_minus = 100.0 * (minus_rma / tr_rma).replace([np.inf, -np.inf], np.nan)

    dx      = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    adx_val = _wilder_rma(dx, n)
    return pd.DataFrame({'adx': adx_val, 'di_plus': di_plus, 'di_minus': di_minus})

def vwap_rolling(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    pv = tp * df['volume']
    return pv.rolling(window, min_periods=1).sum() / df['volume'].rolling(window, min_periods=1).sum()

# ---------------------------
# Feature builder (shared)
# ---------------------------
def build_features(df: pd.DataFrame, atr_len: int = 14, adx_len: int = 14, vwap_win: int = 20) -> pd.DataFrame:
    out = df.copy()

    # Volatility scale
    out['atr'] = atr(out, atr_len)
    out['atr_pct'] = out['atr'] / out['close']

    # Trend strength
    adx_df = adx(out, adx_len)
    out = out.join(adx_df)

    # VWAP & normalized gap
    out['vwap'] = vwap_rolling(out, vwap_win)
    out['vwap_gap'] = out['close'] - out['vwap']
    out['vwap_gap_norm_atr'] = out['vwap_gap'] / out['atr']

    # Light persistence proxy: sign of (close - vwap) averaged
    sign_gap = np.sign(out['vwap_gap']).replace(0, np.nan)
    out['vwap_sign_persist_5'] = sign_gap.rolling(5).mean()

    return out

# =======================================
# A) RULES → RUN-LENGTH "BLOCK" SMOOTHER
# =======================================
def predict_daily(features: pd.DataFrame,
                  adx_hi: float = 25.0,
                  adx_mid: float = 20.0,
                  gap_hi: float = 0.8,
                  gap_mid: float = 0.5,
                  persist_min: float = 0.2) -> pd.Series:
    """
    Rule-based daily TREND prediction (1=TREND, 0=RANGE).
    """
    adx = features['adx']
    gap = features['vwap_gap_norm_atr'].abs()
    per = features['vwap_sign_persist_5']

    cond_strong = (adx >= adx_hi) & (gap >= gap_hi)
    cond_mod    = (adx >= adx_mid) & (gap >= gap_mid) & (per >= persist_min)

    pred = (cond_strong | cond_mod).astype(float)  # 1.0 TREND, 0.0 RANGE
    return pred

def smooth_blocks(pred_raw: pd.Series,
                  min_trend_block: int = 3,
                  min_range_block: int = 3,
                  fill_small_gaps: bool = True) -> pd.Series:
    """
    Enforce blockwise persistence by run-length filtering.
    """
    s = pred_raw.copy()
    if s.isna().all():
        return s

    s = s.ffill().fillna(0.0)  # assume early unknowns as RANGE
    vals = s.values.astype(float)
    n = len(vals)

    def flip_runs(vals, target_val, min_len, flip_to):
        i = 0
        while i < n:
            v = vals[i]
            j = i
            while j < n and vals[j] == v:
                j += 1
            run_len = j - i
            if v == target_val and run_len < min_len:
                vals[i:j] = flip_to
            i = j

    # Remove tiny TREND blips
    if min_trend_block and min_trend_block > 1:
        flip_runs(vals, target_val=1.0, min_len=min_trend_block, flip_to=0.0)

    # Optionally fill tiny RANGE holes inside TREND
    if fill_small_gaps and min_range_block and min_range_block > 1:
        flip_runs(vals, target_val=0.0, min_len=min_range_block, flip_to=1.0)

    return pd.Series(vals, index=s.index)

def detect_regimes_blocks(df: pd.DataFrame,
                          atr_len: int = 14,
                          adx_len: int = 14,
                          vwap_win: int = 30,
                          adx_hi: float = 28.0,
                          adx_mid: float = 20.0,
                          gap_hi: float = 1.2,
                          gap_mid: float = 0.7,
                          persist_min: float = 0.5,
                          min_trend_block: int = 3,
                          min_range_block: int = 3,
                          fill_small_gaps: bool = True) -> pd.DataFrame:
    """
    Rules + run-length blocks.
    """
    feat = build_features(df, atr_len=atr_len, adx_len=adx_len, vwap_win=vwap_win)
    pred_raw = predict_daily(feat, adx_hi=adx_hi, adx_mid=adx_mid, gap_hi=gap_hi, gap_mid=gap_mid, persist_min=persist_min)
    regime = smooth_blocks(pred_raw,
                           min_trend_block=min_trend_block,
                           min_range_block=min_range_block,
                           fill_small_gaps=fill_small_gaps)
    out = feat.copy()
    out['pred_raw'] = pred_raw
    out['regime'] = regime
    out['state'] = out['regime'].map({0.0: 'RANGE', 1.0: 'TREND'})
    return out

# =======================================
# B) KMEANS + SLIDING WINDOW (PROB-BASED)
# =======================================
def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(skipna=True)
    sd = df.std(ddof=0, skipna=True).replace(0, np.nan)
    return (df - mu) / sd

def _fit_kmeans_map_trend(features: pd.DataFrame,
                          cols=('z_adx','z_atr_pct','z_vwap_gap_norm_atr','z_vwap_sign_persist_5'),
                          random_state: int = 42):
    """
    Fit KMeans(k=2) on z-scored feature columns and decide which cluster is TREND
    by comparing mean ADX in each cluster.
    """
    from sklearn.cluster import KMeans
    X_df = features.dropna(subset=list(cols)).loc[:, list(cols)]
    X = X_df.values

    km = KMeans(n_clusters=2, n_init=20, random_state=random_state)
    km.fit(X)

    labels = pd.Series(km.labels_, index=X_df.index, name='cluster')
    cluster_adx = features.loc[X_df.index, 'adx'].groupby(labels).mean()
    trend_cluster = int(cluster_adx.idxmax())
    return km, trend_cluster, X_df.index

def _soft_probs_from_dist(km, X: np.ndarray) -> np.ndarray:
    """Softmax over negative centroid distances."""
    d = km.transform(X)
    logits = -d
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)

def smooth_window_prob(trend_prob: pd.Series,
                       window: int = 7,
                       min_trend_frac: float = 0.60,
                       center: bool = True,
                       min_valid_frac: float = 0.6,
                       tie_bias: float = 0.5) -> pd.Series:
    """
    Sliding-window smoothing based on averaged probability.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    min_periods = max(1, int(np.ceil(min_valid_frac * window)))

    roll = trend_prob.rolling(window, center=center, min_periods=min_periods).mean()
    lab = (roll > min_trend_frac).astype(float)

    # exact ties
    tie_mask = roll.eq(min_trend_frac)
    if tie_mask.any():
        if tie_bias == 0.5:
            lab.loc[tie_mask] = np.nan
            lab = lab.ffill()
        else:
            lab.loc[tie_mask] = float(1 if tie_bias >= 1 else 0)

    # warmup
    if lab.notna().any():
        first_idx = lab.first_valid_index()
        lab = lab.ffill()
        lab.loc[:first_idx] = lab.loc[first_idx]
    else:
        lab[:] = 0.0

    return lab

def detect_regimes_kmeans(df: pd.DataFrame,
                          atr_len: int = 14,
                          adx_len: int = 14,
                          vwap_win: int = 30,
                          window: int = 7,
                          min_trend_frac: float = 0.60,
                          center: bool = True,
                          random_state: int = 42) -> pd.DataFrame:
    """
    KMeans on z-scored features + sliding-window probability smoother.

    Adds columns:
      ['atr','atr_pct','adx','di_plus','di_minus','vwap','vwap_gap','vwap_gap_norm_atr',
       'vwap_sign_persist_5','regime_raw','trend_prob','regime','state']
    """
    feat = build_features(df, atr_len=atr_len, adx_len=adx_len, vwap_win=vwap_win)

    # build z-features for clustering
    z_cols_src = ['adx', 'atr_pct', 'vwap_gap_norm_atr', 'vwap_sign_persist_5']
    z = _zscore(feat[z_cols_src].copy())
    z.columns = [f"z_{c}" for c in z_cols_src]
    feat = feat.join(z)

    # fit & map trend
    km, trend_cluster, fit_idx = _fit_kmeans_map_trend(feat, random_state=random_state)
    # predict on all valid rows
    mask = feat[['z_adx','z_atr_pct','z_vwap_gap_norm_atr','z_vwap_sign_persist_5']].notna().all(axis=1)
    X = feat.loc[mask, ['z_adx','z_atr_pct','z_vwap_gap_norm_atr','z_vwap_sign_persist_5']].values
    probs = np.full((len(feat), 2), np.nan)
    probs[mask] = _soft_probs_from_dist(km, X)

    trend_prob = pd.Series(np.nan, index=feat.index)
    trend_prob.loc[mask] = probs[mask, trend_cluster]

    regime_raw = pd.Series(np.nan, index=feat.index)
    regime_raw.loc[mask] = (km.predict(X) == trend_cluster).astype(float)

    out = feat.copy()
    out['regime_raw'] = regime_raw
    out['trend_prob'] = trend_prob

    # sliding-window smoother on probability
    out['regime'] = smooth_window_prob(out['trend_prob'],
                                       window=window,
                                       min_trend_frac=min_trend_frac,
                                       center=center)
    out['state'] = out['regime'].map({0.0: 'RANGE', 1.0: 'TREND'})
    return out

# ---------------------------
# Plot: one dot per day
# ---------------------------
def plot_regime_dots(df, price_col='close',
                     range_color='blue', trend_color='red',
                     size=14, alpha=0.9, title='Daily regime dots'):
    """
    Scatter only: one dot per day.
    RANGE (0.0) = blue, TREND (1.0) = red.
    """
    if 'regime' not in df.columns:
        raise ValueError("df must contain a 'regime' column (0.0 RANGE, 1.0 TREND).")

    x = df.index
    y = df[price_col].values
    r = df['regime'].values

    mask_trend = (r == 1.0)
    mask_range = (r == 0.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(x[mask_range], y[mask_range], s=size, alpha=alpha, color=range_color, label='RANGE')
    ax.scatter(x[mask_trend], y[mask_trend], s=size, alpha=alpha, color=trend_color, label='TREND')
    ax.set_title(title)
    ax.set_xlabel('Date'); ax.set_ylabel(price_col.capitalize())
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best')
    plt.tight_layout()
    return fig, ax
    
def _clip01(x: pd.Series) -> pd.Series:
    return x.clip(lower=0.0, upper=1.0)


def trend_score_rules(features: pd.DataFrame,
                      adx_hi: float,
                      gap_hi: float,
                      persist_min: float,
                      w_adx: float = 1.0,
                      w_gap: float = 1.0,
                      w_persist: float = 1.0) -> pd.Series:
    """
    Continuous trend strength score in [0, 1].

    This measures *how strong trend structure is*,
    NOT whether we should trade alpha or theta.
    """
    adx = features["adx"]
    gap = features["vwap_gap_norm_atr"].abs()
    per = features["vwap_sign_persist_5"]

    s_adx = _clip01(adx / adx_hi)
    s_gap = _clip01(gap / gap_hi)
    s_per = _clip01(per.abs() / persist_min)

    denom = w_adx + w_gap + w_persist
    score = (w_adx * s_adx + w_gap * s_gap + w_persist * s_per) / denom
    score.name = "trend_score"
    return score

# =======================================
# Alpha–Theta regime dominance (centered probability/weight)
# =======================================

def compute_score_neutral(adx_hi: float, adx_mid: float,
                          gap_hi: float, gap_mid: float) -> float:
    """
    Neutral score where alpha ~= theta (borderline between RANGE and TREND).

    We assume persistence at threshold maps to 1.0 contribution in the score
    (because score uses abs(persist) / persist_min clipped to [0,1]).
    """
    return ((adx_mid / adx_hi) + (gap_mid / gap_hi) + 1.0) / 3.0


def trend_weight_alpha_theta(trend_score: pd.Series,
                             score_neutral: float,
                             scale: float = 0.2,
                             clip: float = 0.999) -> pd.DataFrame:
    """
    Convert trend_score into:
      - trend_excess = trend_score - score_neutral
      - w in [-1, +1] via tanh(excess/scale)
      - p_trend in [0,1] via (1+w)/2

    scale controls how quickly you "commit" to alpha vs theta.
    """
    if scale <= 0:
        raise ValueError("scale must be > 0")

    excess = trend_score.astype(float) - float(score_neutral)
    w = np.tanh(excess / scale)

    # optional: avoid exactly +/-1 for numerical stability in downstream use
    if clip is not None:
        w = np.clip(w, -clip, clip)

    p_trend = 0.5 * (1.0 + w)
    p_range = 1.0 - p_trend

    return pd.DataFrame({
        "trend_excess": excess,
        "trend_weight": pd.Series(w, index=trend_score.index),
        "p_trend": pd.Series(p_trend, index=trend_score.index),
        "p_range": pd.Series(p_range, index=trend_score.index),
    }, index=trend_score.index)

def summarize_regime(out):
    """
    Parameters
    ----------
    out : DataFrame
        Output of detect_regimes_* for ONE symbol

    Returns
    -------
    dict with:
        - structural_trend_weight
        - p_trend_2w
    """
    structural = out["trend_weight"].resample("YE").mean().iloc[-2]

    p_trend_w = (
        out["p_trend"]
        .resample("W-FRI")
        .mean()
        .iloc[-1]
    )

    # --- Classification based on STRUCTURE ONLY ---
    if structural < 0.2:
        regime_character = "RANGE_DOMINATED"
    elif structural < 0.35:
        regime_character = "MIXED"
    else:
        regime_character = "TREND_DOMINATED"

    return {
        "structural_trend_weight": structural,
        "p_trend_w": p_trend_w,
        "regime_character": regime_character,
    }