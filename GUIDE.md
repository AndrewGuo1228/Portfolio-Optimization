# 操作指南 — BL Portfolio Optimization Backtest
**Black-Litterman 组合优化回测系统**

---

## 目录
1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [运行第一次回测](#3-运行第一次回测)
4. [读懂输出结果](#4-读懂输出结果)
5. [参数调节指南](#5-参数调节指南)
6. [模型原理简介](#6-模型原理简介)
7. [实验迭代建议](#7-实验迭代建议)
8. [常见问题](#8-常见问题)

---

## 1. 环境准备

### 安装依赖

```bash
cd Portfolio-Optimization
pip install -r requirements.txt
```

所需的库（requirements.txt 里已列好）：
- `numpy` / `pandas` — 数据处理
- `scipy` — 协方差矩阵计算 + SLSQP 优化
- `matplotlib` — Regime 检测内部作图（不影响回测）
- `pyyaml` — 读取配置文件

### 目录结构确认

```
Portfolio-Optimization/
├── Data/                    ← 你的价格数据放这里
│   ├── SPY.csv
│   ├── GLD.csv
│   └── merged_close.csv     ← 或者用这个宽表
├── bl/                      ← BL 优化核心（不需要改）
├── regime/                  ← Regime 检测（不需要改）
├── config/settings.py       ← 模型常量（高级调参用）
├── backtest/                ← 回测引擎（不需要改）
├── configs/
│   └── backtest_config.yaml ← ⭐ 你主要改这个文件
├── results/                 ← 输出结果自动写到这里
└── GUIDE.md                 ← 本文件
```

---

## 2. 数据准备

### 方案 A：每个 ticker 一个 CSV 文件（推荐）

文件放在 `Data/` 目录下，命名为 `{TICKER}.csv`，例如 `SPY.csv`、`GLD.csv`。

**必须包含的列：**

| 列名 | 说明 | 备注 |
|------|------|------|
| `date` | 日期 | YYYY-MM-DD 或 YYYYMMDD 均可 |
| `open` | 开盘价 | Regime 检测需要 |
| `high` | 最高价 | Regime 检测需要 |
| `low` | 最低价 | Regime 检测需要 |
| `close` | 收盘价 | BL 收益计算用 |
| `volume` | 成交量 | Regime 检测需要 |

示例（`SPY.csv` 前几行）：
```
date,open,high,low,close,volume
2018-01-02,267.84,268.81,267.41,268.77,86655500
2018-01-03,268.96,270.07,268.53,270.47,90070400
...
```

### 方案 B：使用合并宽表（已有 merged_close.csv）

如果只有收盘价数据，放一个 `merged_close.csv` 在 `Data/` 目录：

```
date,SPY,GLD,TLT,QQQ,...
2018-01-02,268.77,122.40,126.80,156.87,...
2018-01-03,270.47,122.80,127.12,157.92,...
```

> ⚠️ **注意**：使用宽表时 Regime 检测会降级（只有收盘价，无 OHLCV），
> BL 计算仍然正常，但 Regime 信号质量会下降。

---

## 3. 运行第一次回测

### Step 1：编辑配置文件

打开 `configs/backtest_config.yaml`，至少修改以下几个字段：

```yaml
# 要包含的标的（Data/ 目录下必须有对应的 CSV）
tickers:
  - SPY
  - GLD
  - TLT
  - QQQ
  - IWM

# 回测区间
start_date: "2018-01-01"
end_date:   "2024-12-31"

# 调仓频率
rebalance_freq: "ME"    # ME=月末, QE=季末, W=每周
```

### Step 2：运行

```bash
# 在 Portfolio-Optimization/ 目录下执行
python backtest/runner.py
```

你会看到类似这样的进度输出：
```
[backtest] Tickers:   ['SPY', 'GLD', 'TLT', 'QQQ', 'IWM']
[backtest] Period:    2018-01-01 → 2024-12-31
[backtest] Rebalance: ME

[backtest] ── Rebalance 2018-01-31 ──
  BL weights: {'SPY': '0.412', 'GLD': '0.183', 'TLT': '0.221', ...}

[backtest] ── Rebalance 2018-02-28 ──
  ...

==================================================
Backtest Performance Summary
==================================================
BL  Total Return :  +87.3%
EW  Total Return :  +61.2%
SPY Total Return :  +95.1%
BL  Ann. Return  :  +9.8%
BL  Ann. Vol     :  11.4%
BL  Sharpe Ratio :  0.86
==================================================
```

### Step 3：查看结果

结果自动写入 `results/` 目录：

```
results/
├── bl_weights_history.csv    ← 每期调仓权重
├── bl_signals_history.csv    ← 完整信号（含 Regime、mu_BL 等）
└── performance.csv           ← BL vs EW vs SPY 每日收益对比
```

---

## 4. 读懂输出结果

### bl_weights_history.csv

每次调仓的 BL 推荐权重。

| 列 | 说明 |
|----|------|
| `date` (index) | 调仓日期 |
| `SPY`, `GLD`, ... | 各 ticker 的推荐权重（0~1，合计=1） |

示例：
```
date,       SPY,   GLD,   TLT,   QQQ,   IWM
2018-01-31, 0.412, 0.183, 0.221, 0.094, 0.090
2018-02-28, 0.385, 0.210, 0.198, 0.127, 0.080
```

### bl_signals_history.csv

每次调仓时每个 ticker 的完整诊断信息。关键列：

| 列 | 说明 |
|----|------|
| `rebal_date` | 调仓日期 |
| `ticker` | 标的 |
| `current_drift_regime_base` | 当日 Regime（UPTREND / RANGE / DOWNTREND） |
| `current_weight` | 调仓前的权重 |
| `bl_weight` | BL 推荐权重 |
| `delta_weight` | 变化量（正=加仓，负=减仓） |
| `q_view` | BL 对该标的的预期收益率视图 |
| `mu_bl` | BL 后验预期收益率 |
| `price_return` | Regime 驱动的价格收益预测 |
| `price_confidence` | 对当前 Regime 判断的置信度（0~1） |
| `discount` | RSI+波动率折扣系数 |
| `action` | 建议操作（Increase / Reduce / Hold） |
| `reason` | 操作原因说明 |

### performance.csv

每日累计收益对比。

| 列 | 说明 |
|----|------|
| `bl_return` | BL 策略当日收益率 |
| `bl_cumret` | BL 策略累计收益率 |
| `ew_cumret` | 等权基准累计收益率 |
| `spy_cumret` | SPY Buy & Hold 累计收益率（如有） |

---

## 5. 参数调节指南

所有参数集中在 `configs/backtest_config.yaml`。

### 调仓频率

```yaml
rebalance_freq: "ME"    # 月末 ← 推荐起点
rebalance_freq: "QE"    # 季末 ← 换手率更低，运行更快
rebalance_freq: "W"     # 每周 ← 换手率高，交易成本更大
```

### 风险约束

```yaml
beta_limit: 1.2          # 组合 Beta 上限（相对 SPY）
                         # 提高 → 允许更激进的仓位
                         # 降低 → 更保守

vol_limit_factor: 1.2    # 组合波动率 ≤ QQQ_vol × 这个值
                         # 提高 → 允许更高波动率的组合
                         # 降低 → 强制更低波动
```

### Regime 参数

```yaml
smooth_window: 5         # Regime 标签平滑窗口（天）
                         # 增大 → 信号更稳定，但反应更慢
                         # 减小 → 信号更敏感

downtrend_window: 1      # DOWNTREND 触发窗口（天）
                         # = 1: 一天就触发（快速保护，默认）
                         # = 2: 连续2天才触发

min_segment_days: 7      # 短于这个天数的 Regime 片段会被合并
                         # 增大 → 减少信号抖动
```

### BL 模型参数

```yaml
confidence_denom: 7      # Regime 置信度衰减速率
                         # confidence = 1 - exp(-days_in_regime / 7)
                         # 增大 → 置信度积累更慢（更保守）
                         # 减小 → 置信度积累更快（更激进）

trans_days_enter_downtrend: 5   # 进入 DOWNTREND 的过渡天数
trans_days_exit_downtrend: 10   # 离开 DOWNTREND 的过渡天数（更谨慎）
trans_days_other: 7             # UPTREND ↔ RANGE 过渡天数
```

### 协方差方法

```yaml
cov_method: "kendall"    # Kendall τ → 更稳健，对尾部风险不敏感（推荐）
cov_method: "pearson"    # EWMA Pearson → 计算更快，但对异常值敏感
```

---

## 6. 模型原理简介

### 整体流程

```
OHLCV 数据
    │
    ▼
Regime 检测 (Hybrid V2)
    │  → UPTREND / RANGE / DOWNTREND
    │  → 每日趋势概率 p_trend_2w
    │
    ▼
Drift 校准
    │  → 计算每个 Regime 下的历史条件收益率
    │  → 用前向收益（5d/10d/20d 加权）估算
    │
    ▼
期望收益 Q (BL 视图)
    │  → Q = regime_drift × RSI折扣 × Vol折扣
    │  → 防止在高 RSI 或高波动率时过于激进
    │
    ▼
Black-Litterman 后验
    │  → 均衡先验 Π = λ·Σ·w_strategic
    │  → 后验 μ_BL = 混合(Π, Q, 不确定性 Ω)
    │
    ▼
SLSQP 约束优化
    │  → 最大化 μ_BL·w - λ·w'Σw
    │  → 约束: β ≤ 1.2, σ ≤ QQQ_vol×1.2, Σw=1
    │
    ▼
权重输出 w*
```

### Regime 三种状态的含义

| Regime | 含义 | BL 的处理方式 |
|--------|------|--------------|
| **UPTREND** | 趋势向上，价格动量强 | Q 为正，加大权重 |
| **RANGE** | 横盘震荡，无明显方向 | Q 接近中性，维持 |
| **DOWNTREND** | 趋势向下，动量转弱 | Q 为负或低，减少权重 |

### 置信度的作用

进入一个新 Regime 后，置信度从 0 开始积累：
```
第1天:  confidence ≈ 13%   → 轻微调整
第7天:  confidence ≈ 63%   → 中等调整
第14天: confidence ≈ 86%   → 较大调整
第30天: confidence ≈ 99%   → 充分体现
```
这防止了对短暂 Regime 切换做出过度反应。

### RSI + 波动率折扣

当标的处于超买（RSI > 75）或高波动率（IV 百分位 > 65%）时，
BL 视图 Q 会被折扣，最多打到 20%（`discount_floor`）。
这是对"追高"行为的内置保护。

---

## 7. 实验迭代建议

### 快速验证工作流

每次修改参数后，用较短时间段快速验证：

```yaml
# 快速测试用（2-3年，季度调仓）
start_date: "2020-01-01"
end_date:   "2023-12-31"
rebalance_freq: "QE"
```

确认结果合理后，再改回完整时间段和月度调仓。

### 实验方向建议

**实验1：Regime 敏感度**
```yaml
# 更快响应
smooth_window: 3
downtrend_window: 1

# vs 更稳定
smooth_window: 10
downtrend_window: 2
```

**实验2：保守 vs 激进**
```yaml
# 保守
beta_limit: 0.8
vol_limit_factor: 0.9
confidence_denom: 14

# vs 激进
beta_limit: 1.5
vol_limit_factor: 1.5
confidence_denom: 5
```

**实验3：调仓频率**
```yaml
# 月度 vs 季度 → 比较换手率和 Sharpe 的 trade-off
rebalance_freq: "ME"
# vs
rebalance_freq: "QE"
```

**实验4：标的组合**
```yaml
# 纯股票
tickers: [SPY, QQQ, IWM, XLK, XLF, XLE]

# vs 股债混合
tickers: [SPY, QQQ, TLT, GLD, HYG, LQD]

# vs 全球分散
tickers: [SPY, EFA, EEM, TLT, GLD, USO]
```

### 对比结果的方法

把不同实验的 `performance.csv` 放到 Excel 里对比：
- **BL Sharpe** = 年化收益 / 年化波动率
- **Max Drawdown** = `(cumret - cumret.cummax()).min()`
- **BL vs SPY 超额收益** = `bl_cumret.iloc[-1] - spy_cumret.iloc[-1]`

---

## 8. 常见问题

**Q: 运行时提示 "Not enough history yet, skipping"**

A: 前60个交易日用于 Regime 检测预热，不会产生信号。这是正常的，
   第一个信号通常在 `start_date` 后约3个月出现。

**Q: 某个 ticker 没有出现在权重里**

A: 可能原因：
- `Data/{TICKER}.csv` 文件不存在 → 检查文件名是否完全匹配
- 该 ticker 当期 `delta_gamma_exposure <= 0` → 正常，BL 会把它排除出本期优化

**Q: BL failed on ... SLSQP did not converge**

A: 约束条件过紧，优化无解。尝试：
```yaml
beta_limit: 1.5         # 放宽 beta 约束
vol_limit_factor: 1.5   # 放宽波动率约束
```

**Q: 所有权重都接近均等权重**

A: 可能 Regime 信号太弱（标的都处于 RANGE 状态），或置信度积累不足。
   尝试缩短 `confidence_denom`（如从 7 改为 4）让置信度积累更快。

**Q: 想测试某一段特定市场环境**

A: 直接修改时间段：
```yaml
# 测试熊市（2022年）
start_date: "2021-06-01"
end_date:   "2023-03-31"

# 测试牛市（2020-2021年）
start_date: "2020-04-01"
end_date:   "2021-12-31"

# 测试高波动（2020年3月）
start_date: "2020-01-01"
end_date:   "2020-12-31"
```

**Q: 如何加入新的数据源？**

A: 在 `Data/` 目录里放一个新的 `{TICKER}.csv`，
   然后把 ticker 名字加进 `backtest_config.yaml` 的 `tickers` 列表里。
   系统会自动识别。

---

## 附：核心参数速查表

| 参数 | 默认值 | 调大效果 | 调小效果 |
|------|--------|----------|----------|
| `beta_limit` | 1.2 | 更激进 | 更保守 |
| `vol_limit_factor` | 1.2 | 允许更高波动 | 更低波动 |
| `confidence_denom` | 7 | 置信度积累慢 | 置信度积累快 |
| `smooth_window` | 5 | 信号更稳定 | 信号更敏感 |
| `downtrend_window` | 1 | 保护更慢 | 保护更快 |
| `min_segment_days` | 7 | 减少假信号 | 响应更快 |
| `trans_days_enter_downtrend` | 5 | 进熊更慢 | 进熊更快 |
| `trans_days_exit_downtrend` | 10 | 出熊更慢（谨慎） | 出熊更快 |
| `discount_floor` | 0.20 | — | 对高RSI/高Vol更严格 |
