---
title: PM Workspace SX5E
emoji: 📊
colorFrom: blue
colorTo: cyan
sdk: docker
pinned: false
app_port: 7860
---

# PM Workspace — EuroStoxx 50

> **Quantitative portfolio dashboard** — Black-Litterman × Gradient Descent optimization, benchmarked on the EuroStoxx 50.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Dash](https://img.shields.io/badge/Dash-2.18-informational)](https://dash.plotly.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

PM Workspace simulates the monitoring and portfolio management dashboard that an institutional fund manager would use for a benchmark-constrained strategy on the **EuroStoxx 50**. It is structured around three modules:

| Module | Route | Description |
|--------|-------|-------------|
| **00 — SX5E Replication** | `/replication` | Index reconstruction from STOXX methodology files (free-float market cap). Annualized tracking error ~1%. |
| **01 — Portfolio Constructor** | `/cockpit` | Black-Litterman engine × gradient descent optimizer. Dynamic sector views, absolute or relative-to-peers performance. UCITS-compliant constraints. |
| **02 — Holdings** | `/holdings` | Optimized weights, BL diagnostics, ex-ante risk metrics, sector decomposition vs benchmark. |
| **03 — Reporting & Simulation** | `/whatif` | Bloomberg PORT-style statistics, what-if analysis, out-of-sample simulation with dynamic risk-free rate. |

**Reference**: Oliva, J. (2025). *Multi-objective Portfolio Optimization Via Gradient Descent*. [arXiv:2507.16717](https://arxiv.org/abs/2507.16717)

---

## Architecture

```
portfolio-optimizer-sx5e/
├── quant_engine/          # PyTorch optimization engine
│   ├── black_litterman.py    ← BL engine v5 (Woodbury, dynamic universe)
│   ├── sparsemax.py          ← O(n log n) simplex projection
│   ├── losses.py             ← Sharpe (BL mode) + CVaR objectives
│   ├── constraints.py        ← UCITS 5/10/40, TE, ICB sector budgets
│   ├── model.py              ← PortfolioOptimizer (Adam + early stopping)
│   └── backtester.py         ← Walk-forward backtest (12m train / 1m test)
│
├── pm_workspace/          # Dash dashboard (frontend)
│   ├── app.py                ← Entry point, global layout, navigation
│   ├── pages/                ← 5 pages (home, replication, cockpit, holdings, whatif)
│   └── utils/
│       ├── quant_bridge.py   ← Dash ↔ Engine interface (Pydantic validation)
│       └── design.py         ← Shared design tokens (Bloomberg dark theme)
│
├── setup/                 # Data pipeline scripts
├── nightly/               # Daily price update
├── data/                  # Parquet files (see Data section)
│
├── METHODOLOGY.md         # Mathematical methods (BL, sparsemax, CVaR, walk-forward)
└── TECHNICAL_SPECIFICATION.md  # Data schemas, architecture, deployment
```

---

## Key Technical Features

### Black-Litterman Engine (v5)
- **Woodbury k×k formula** — inverts the k×k view uncertainty matrix instead of the n×n covariance, enabling large universes
- **Dynamic universe**: `N_r = benchmark_assets ∪ view_targets` — eliminates mean-reversion bias for mid-caps targeted by views
- **Three implementation choices**: excess return conversion, James-Stein shrinkage on δ (toward 2.5), "alpha" view type (outperformance vs CAPM consensus)
- **ViewAccumulator v4**: IR-based selection, redundancy/contradiction detection, mcap-weighted P matrix

### Sparsemax Projection
Replaces softmax to produce **exactly-zero weights** without ad hoc thresholding. Algorithm runs in O(n log n) and is fully compatible with PyTorch autograd.

```
z ∈ ℝⁿ  →  sparsemax(z)  →  w ∈ Δⁿ  (Σwᵢ=1, wᵢ≥0, sparse)
```

### Multi-Objective Loss
```
L(z) = −λ_Sharpe · Sharpe_BL(w) + λ_CVaR · CVaR(w) + Σᵢ λᵢ · Cᵢ(w)
```

- **Sharpe BL mode**: numerator `w · μ_BL`, denominator `σ_hist` (BL modifies expectations, not risk)
- **CVaR differentiable**: ReLU formulation (Rockafellar & Uryasev, eq. 16 in Oliva 2025)
- **UCITS 5/10/40** constraints with normalized penalties (v3 calibration)
- **ICB sector deviation** constraint with auto-widened corridor in BL mode (±10% → ±20%)

### Walk-Forward Backtest
12-month training / 1-month test rolling windows. Point-in-time loading (no look-ahead), dead ticker management (anti-survivorship bias), dynamic risk-free rate (€STR/EONIA).

---

## Data Requirements

The following Parquet files are required to run the full dashboard:

| File | Path | Description |
|------|------|-------------|
| `sx5e_daily_panel.parquet` | `data/processed/` | Daily SX5E index (weights + returns) |
| `master_history.parquet` | `data/processed/` | OHLCV history, all tickers (~150 MB) |
| `ref_universe.parquet` | `data/reference/` | ISIN reference (ICB sectors, fundamentals) |
| `sx5e_compositions_cache.parquet` | `data/cache/` | Monthly STOXX compositions |

See [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md) for full Parquet schemas.

---

## Local Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure paths

Via environment variables:
```bash
export PROJECT_ROOT="/path/to/portfolio-optimizer-sx5e"
export QUANT_ENGINE_ROOT="/path/to/portfolio-optimizer-sx5e/quant_engine"
```

Or edit `pm_workspace/app.py` directly (lines 37-38).

### 3. Build data (first run)

```bash
python setup/build_reference.py        # ref_universe.parquet
python setup/ingest_compositions.py    # sx5e_compositions_cache.parquet
python setup/build_sx5e_index.py       # sx5e_daily_panel.parquet
```

### 4. Launch

```bash
cd pm_workspace
python app.py
# → http://localhost:8050
```

### 5. Production (Gunicorn)

```bash
gunicorn pm_workspace.app:server --bind 0.0.0.0:8050 --workers 1
```

---

## Usage Flow

1. **`/REPLICATION BENCHMARK`** — Check reconstruction quality: rolling tracking error, OHLCV snapshot, compositions table
2. **`/PORTFOLIO CONSTRUCTEUR`** — Set T0, lookback window, BL views (sector/absolute/relative), UCITS constraints
3. **`/HOLDING`** → optimization runs automatically — inspect optimal weights, BL diagnostics, sector radar
4. **`/REPORTING & SIMULATION`** — Simulate out-of-sample Buy & Hold, compare against benchmark

## References

- He, G. & Litterman, R. (1999). *The Intuition Behind Black-Litterman Model Portfolios*. Goldman Sachs.
- Martins, A. & Astudillo, R. (2016). *From Softmax to Sparsemax*. ICML 2016.
- Rockafellar, R.T. & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk*. Journal of Risk.
- Oliva, J. (2025). *Black-Litterman Portfolio Optimization via Differentiable Sparsemax Projections*. [arXiv:2507.16717](https://arxiv.org/abs/2507.16717)
