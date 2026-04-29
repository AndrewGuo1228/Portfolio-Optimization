# BL Portfolio Optimization — Backtest

Standalone Black-Litterman optimization and rolling backtest package.
No IBKR API required. All market data comes from local CSV files.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Edit parameters
#    Open configs/backtest_config.yaml and set tickers / dates

# 3. Run backtest
python backtest/runner.py

# 4. View results
#    results/
#    ├── bl_weights_history.csv   — BL recommended weights at each rebalance
#    ├── bl_signals_history.csv   — Full signal detail (regime, mu_BL, etc.)
#    └── performance.csv          — BL vs Equal-Weight vs SPY cumulative returns
```

## Data Format

Place price files in `Data/`. Two formats supported:

**Individual files** (preferred for OHLCV + regime detection):
```
Data/SPY.csv
Data/GLD.csv
...
```
Required columns: `date, open, high, low, close, volume`
Date format: `YYYY-MM-DD` or `YYYYMMDD`

**Wide-format merged file** (fallback for price-return only):
```
Data/merged_close.csv
```
Format: `date` index, one column per ticker with closing prices.

## Directory Structure

```
Portfolio-Optimization/
├── Data/                      # Price CSV files (already present)
├── bl/
│   └── optimizer.py           # Black-Litterman core (no changes from original)
├── regime/
│   ├── regime_detection.py    # Feature engineering (ADX, VWAP, etc.)
│   ├── regime_models.py       # Hybrid V2 regime classifier
│   └── regime_calibration.py  # Forward-return drift calibration
├── config/
│   └── settings.py            # All tunable constants
├── backtest/
│   ├── data_loader.py         # Load prices, compute betas
│   ├── regime_builder.py      # Build regime inputs from OHLCV
│   └── runner.py              # Rolling backtest main loop
├── configs/
│   └── backtest_config.yaml   # ← Edit this to change parameters
├── results/                   # Output files written here
└── requirements.txt
```

## Key Parameters (backtest_config.yaml)

| Parameter | Default | Description |
|---|---|---|
| `tickers` | [GLD, QQQ, SPY, TLT, IWM, HYG] | Assets to optimize |
| `start_date` | 2018-01-01 | Backtest start |
| `end_date` | 2024-12-31 | Backtest end |
| `rebalance_freq` | ME | ME=monthly, QE=quarterly, W=weekly |
| `beta_limit` | 1.2 | Max portfolio beta |
| `cov_method` | kendall | kendall (robust) or pearson (fast) |
| `confidence_denom` | 7 | Regime confidence decay rate |

## How It Works

1. **Regime Detection** — For each rebalance date, hybrid_v2 runs on OHLCV history
   up to that date to classify each ticker as UPTREND / RANGE / DOWNTREND.

2. **Drift Calibration** — Historical forward returns (5d / 10d / 20d) are computed
   conditional on regime label, then shrunk toward a base drift for thin samples.

3. **BL Optimization** — Expected returns Q = regime_drift × RSI/vol discount.
   Covariance Σ estimated via Kendall τ. SLSQP solves for optimal weights under
   beta and volatility constraints.

4. **Performance** — BL weights applied forward to next rebalance period.
   Results compared to Equal-Weight and SPY buy-and-hold.
