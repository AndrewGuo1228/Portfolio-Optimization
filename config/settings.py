"""
settings.py
-----------
Standalone settings for the BL + Regime backtest package.
All constants copied verbatim from the original settings.py
(only the ones needed by optimizer, regime_models, regime_calibration).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent.parent   # Portfolio-Optimization/
DATA_DIR    = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "results"

# ---------------------------------------------------------------------------
# Regime Lab model selectors
# ---------------------------------------------------------------------------
REGIME_MODEL_BASELINE_V1 = "baseline_v1"
REGIME_MODEL_HYBRID_V2   = "hybrid_v2"

# Suitability thresholds (2-week smoothed trend probability)
REGIME_PTREND2W_RANGE_FAVORABLE_MAX = 0.35
REGIME_PTREND2W_NEUTRAL_MAX         = 0.60

# ---------------------------------------------------------------------------
# Regime Hybrid V2 — detection parameters
# ---------------------------------------------------------------------------
REGIME_HYBRID_LOOKBACK_DAYS = 252
REGIME_HYBRID_MIN_PERIODS   = 40

# Structural percentile rule thresholds
REGIME_HYBRID_STRONG_ADX_PCTL    = 0.80
REGIME_HYBRID_STRONG_GAP_PCTL    = 0.80
REGIME_HYBRID_STRONG_PERSIST_PCTL= 0.70
REGIME_HYBRID_MOD_ADX_PCTL       = 0.65
REGIME_HYBRID_MOD_GAP_PCTL       = 0.65
REGIME_HYBRID_MOD_PERSIST_PCTL   = 0.55

# Trend score weights
REGIME_HYBRID_W_ADX      = 0.40
REGIME_HYBRID_W_GAP      = 0.35
REGIME_HYBRID_W_PERSIST  = 0.25
REGIME_HYBRID_TREND_CUTOFF = 0.55

# Direction thresholds on weighted return slope
REGIME_HYBRID_DIR_UP_MIN   =  0.010
REGIME_HYBRID_DIR_DOWN_MAX = -0.010

# Volatility regime thresholds
REGIME_HYBRID_EXPANDING_PRESSURE_MIN = 0.70
REGIME_HYBRID_EXPANDING_LEVEL_MIN    = 0.55
REGIME_HYBRID_HIGH_VOL_LEVEL_MIN     = 0.75
REGIME_HYBRID_LOW_VOL_LEVEL_MAX      = 0.35

# ---------------------------------------------------------------------------
# Regime-conditional drift calibration (forward-return based)
# ---------------------------------------------------------------------------
MC_REGIME_DRIFT_FWD_5_WEIGHT  = 0.40
MC_REGIME_DRIFT_FWD_10_WEIGHT = 0.35
MC_REGIME_DRIFT_FWD_20_WEIGHT = 0.25
MC_REGIME_DRIFT_MIN_SAMPLES   = 12
MC_REGIME_DRIFT_SHRINK_COUNT  = 80

# ---------------------------------------------------------------------------
# BL optimisation — tunable parameters
# ---------------------------------------------------------------------------
LAMBDA_CLIP_MIN  = 2.0
LAMBDA_CLIP_MAX  = 6.0
CONFIDENCE_DENOM = 7      # confidence = 1 - exp(-days / this)

TRANS_DAYS_ENTER_DOWNTREND = 5
TRANS_DAYS_EXIT_DOWNTREND  = 10
TRANS_DAYS_OTHER           = 7

VOL_PENALTY_THRESHOLD = 2.0   # × QQQ realized vol

RSI_PENALTY_START      = 75
RSI_PENALTY_MAX        = 90
VOL_PCT_PENALTY_START  = 0.65
VOL_PCT_PENALTY_MAX    = 0.90
DISCOUNT_FLOOR         = 0.20

DEFAULT_BETA_LIMIT       = 1.2
DEFAULT_VOL_LIMIT_FACTOR = 1.2
DEFAULT_COV_METHOD       = "kendall"

# ---------------------------------------------------------------------------
# Regime smoothing defaults (used by regime_builder)
# ---------------------------------------------------------------------------
SMOOTH_WINDOW    = 5
DOWNTREND_WINDOW = 1
MIN_SEGMENT_DAYS = 7
