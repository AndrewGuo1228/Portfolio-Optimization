# BL Signal Dictionary

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
