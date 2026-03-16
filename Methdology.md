# METHODOLOGY — PM Workspace · EuroStoxx 50

> **Stack** : Python 3.10 · PyTorch · Dash · Plotly · Parquet
> **Reference** : Oliva, J. (2025). *Multi-objective Portfolio Optimization Via Gradient Descent*. [arXiv:2507.16717](https://arxiv.org/abs/2507.16717)
> **Benchmark** : EuroStoxx 50 (STOXX methodology, free-float market cap weighted)

---

## Table of Contents

1. [Module 00 — SX5E Replication](#module-00--sx5e-replication)
2. [Module 01 — Black-Litterman Engine](#module-01--black-litterman-engine)
3. [Module 02 — Sparsemax Projection](#module-02--sparsemax-projection)
4. [Module 03 — Multi-Objective Loss](#module-03--multi-objective-loss)
5. [Module 04 — Differentiable Constraints](#module-04--differentiable-constraints)
6. [Module 05 — Gradient Descent Optimizer](#module-05--gradient-descent-optimizer)
8. [Hyperparameter Reference](#hyperparameter-reference)

---

## Module 00 — SX5E Replication

### Objective

Reconstruct the EuroStoxx 50 index from first principles using STOXX methodology files (free-float market cap data through October 2024), then verify the reconstruction quality via a full backtest against the official index level.

### Pipeline

**1. Index Member Retrieval**

Constituent data is sourced from STOXX official composition files (monthly). Each file provides, for each member: ISIN, ticker, company name, free-float shares outstanding, and the reference price used in the official calculation. These are stored in `sx5e_compositions_cache.parquet`.

**2. ICB Sector Approximation**

The STOXX files do not contain ICB classification directly. Sector mapping is reconstructed by:
- Cross-referencing ISINs against the reference universe (`ref_universe.parquet`) which contains `ICB_Industry`, `ICB_Sector`, and `ICB_SuperSector` from Euronext data
- Fallback: Yahoo Finance metadata (`YF_Sector`, `YF_Industry`) for tickers not present in the ICB reference

**3. Weight Reconstruction**

Official STOXX index weights are derived from:

```
w_i = (FFShares_i × RefPrice_i) / Σ_j (FFShares_j × RefPrice_j)
```

capped at 10% per constituent (STOXX capping rule). Capped weights are renormalized.

Because the exact reference prices are only available through October 2024, subsequent months use a **weight-drift approximation**: weights are propagated using daily price returns and renormalized monthly, with composition changes applied at each rebalancing date.

**4. Reconstruction Quality**

| Metric | Result |
|--------|--------|
| Annualized Tracking Error | ~1% |
| Correlation with official SX5E | > 0.999 |
| Max monthly deviation | < 0.3% |

The dashboard (`/replication`) displays this audit via rolling tracking error, monthly deviations, and a composition snapshot table linked to the selected date range.

---

## Module 01 — Black-Litterman Engine

**Source**: `quant_engine/black_litterman.py` v5
**Reference**: He & Litterman (1992), Cheung (2009)

### Theoretical Framework

The Black-Litterman model starts from the CAPM **equilibrium returns** (the implied returns that would make the current market portfolio optimal) and updates them with **investor views** using Bayes' theorem.

**Equilibrium prior** (reverse optimization):

```
Π = δ · Σ · w_mkt
```

Where:
- `δ` = risk aversion coefficient (calibrated via James-Stein shrinkage toward 2.5, see below)
- `Σ` = covariance matrix of asset returns (T × N, empirical)
- `w_mkt` = market-cap weights of the universe

**Posterior Black-Litterman returns** (Woodbury formula):

```
μ_BL = Π + τΣP^T (Ω + PτΣP^T)^{-1} (Q − PΠ)
```

Where:
- `τ` = uncertainty scalar on the prior (default 0.025, adapts to T/N ratio)
- `P` ∈ ℝ^{k×n} = pick matrix encoding which assets each view refers to
- `Q` ∈ ℝ^k = view returns (annualized, converted to daily before injection)
- `Ω` ∈ ℝ^{k×k} = diagonal view uncertainty matrix

The Woodbury formula inverts the k×k matrix `(Ω + PτΣP^T)` instead of the n×n matrix, which is critical for large universes (k ≪ n).

### Three Key Implementation Choices

**Choice 1 — Excess return conversion**

Raw BL returns `μ_BL` are annualized. Before injection into the optimizer, they are converted to daily excess returns over the risk-free rate:

```python
mu_bl_daily = (mu_bl_annual / 252) - (rf_annual / 252)
```

This ensures compatibility with the Sharpe numerator `w · μ_BL − r_f,daily`.

**Choice 2 — James-Stein shrinkage on δ**

The raw risk-aversion `δ_raw = (μ_market − r_f) / σ²_market` can be unstable (sensitive to the lookback window). A shrinkage toward the canonical value 2.5 (He & Litterman 1999) is applied:

```python
δ = (1 − α) · δ_raw + α · 2.5   # α = delta_shrinkage (default 0.5)
```

This prevents extreme prior returns in low-vol regimes.

**Choice 3 — "Alpha" view type**

In addition to standard absolute and relative views, the engine supports `type = "alpha"` views, which express outperformance vs the **CAPM consensus** (implied Π) rather than vs another asset. This avoids inadvertently encoding the prior into the view itself.

### Dynamic Universe (v5)

Prior versions applied a static `max_universe_size` filter (top-N by market cap). This introduced a systematic bias: assets ranked 51+ received an implicit "no view" prior, meaning their weights were pulled toward zero even when targeted by investor views.

v5 replaces the static filter with a **dynamic universe**:

```
N_r = {Benchmark assets} ∪ {Assets targeted by active views}
```

Mid-cap assets explicitly cited in views are included in the restricted space, ensuring their BL posterior is correctly estimated. Woodbury still runs on a k×k matrix.

### ViewAccumulator (Not finish yet)

Views from the dashboard are collected and validated before being passed to the BL engine:

- **IR-based selection**: views are ranked by information ratio `|Q_i| / σ(view_i)`. Low-IR views can be discarded to avoid adding noise.
- **Redundancy detection**: two views on the same sector and direction are merged.
- **Contradiction detection**: opposing views on the same asset are flagged with a warning.
- **P matrix construction**: for sector views, `P_ij = w_mkt_j / Σ_{k∈sector} w_mkt_k` (mcap-weighted exposure within the sector, normalized to 1). This is more precise than equal-weighting within the sector.

---

## Module 02 — Sparsemax Projection

**Source**: `quant_engine/sparsemax.py`
**Reference**: Martins & Astudillo (2016)

### Why Not Softmax?

Softmax maps any input vector z to strictly positive weights: `w_i = exp(z_i) / Σ exp(z_j) > 0`. Over 50+ assets, this dilutes capital across irrelevant positions. Post-hoc thresholding (e.g., drop positions below 0.5%) is non-differentiable and arbitrary.

Sparsemax produces **exactly-zero weights** for non-relevant assets without any threshold, by projecting onto the probability simplex rather than the positive orthant.

### Algorithm — O(n log n)

```
Input : z ∈ ℝⁿ  (unconstrained pre-weights)
Output: w ∈ Δⁿ  (valid portfolio weights, potentially sparse)

1. Sort z in descending order: z_{(1)} ≥ z_{(2)} ≥ ... ≥ z_{(n)}
2. Find the support size:
       k(z) = max{ k | 1 + k·z_{(k)} > Σ_{j=1}^{k} z_{(j)} }
3. Compute the threshold:
       τ(z) = (Σ_{j=1}^{k(z)} z_{(j)} − 1) / k(z)
4. Project:
       wᵢ = max(zᵢ − τ(z), 0)
```

The implementation adds numerical stabilization: `z ← z − max(z)` before sorting, which prevents overflow without affecting the result.

### Autograd Compatibility

The entire computation uses standard PyTorch operations (`torch.sort`, `torch.cumsum`, `torch.clamp`). PyTorch's autograd engine computes the gradient of sparsemax correctly through these operations — no custom backward pass is required at this stage.

The gradient of sparsemax at w is the projection of the upstream gradient onto the active support:

```
∂L/∂z = ∂L/∂w − (Σᵢ∈S ∂L/∂wᵢ / |S|) · 1_S
```

where S = {i | wᵢ > 0} is the active support. Assets with wᵢ = 0 receive zero gradient.

---

## Module 03 — Multi-Objective Loss

**Source**: `quant_engine/losses.py`

All objectives are formulated as **minimization** problems. The total loss is:

```
L(z) = −λ_Sharpe · Sharpe(w) + λ_CVaR · CVaR(w) + Σᵢ λᵢ · Cᵢ(w)
```

where `w = sparsemax(z)`.

### Objective 1 — Sharpe Ratio

**Historical mode** (`expected_returns = None`):

```
Sharpe_hist = (mean(R_hist) − r_f,daily) / std(R_hist) + ε
```

**Black-Litterman mode** (`expected_returns = μ_BL`):

```
Sharpe_BL = (w · μ_BL − r_f,daily) / std(R_hist) + ε
```

The numerator is replaced by the BL-adjusted prospective return. The denominator remains the historical standard deviation — the covariance matrix Σ is the best available risk estimator; BL modifies expectations, not risk.

This formulation is consistent with Oliva et al. (2025): "any differentiable financial objective can replace the Sharpe numerator."

### Objective 2 — CVaR (Expected Shortfall)

The CVaR at confidence level α is the expected loss in the worst α fraction of days:

```
CVaR_α = VaR_α + (1/α) · E[max(−R − VaR_α, 0)]
```

Differentiable implementation via ReLU (Rockafellar & Uryasev, 2000; eq. 16 in Oliva 2025):

```python
R     = returns @ weights
var   = -torch.quantile(R, alpha)          # VaR as positive loss
cvar  = var + relu(-R - var).mean() / alpha
```

The `torch.quantile` call is not differentiable w.r.t. `alpha` but is differentiable w.r.t. `weights` via the `relu` term. This is exactly what is needed for gradient-based weight optimization.

### Total Loss Assembly

```python
L = λ_sharpe · (−Sharpe) + λ_cvar · CVaR + Σ constraint_terms
```

Default weights: `λ_sharpe = 10.0`, `λ_cvar = 100.0` (CVaR receives a higher coefficient because it operates on daily losses which are an order of magnitude smaller than annualized Sharpe values).

---

## Module 04 — Differentiable Constraints

**Source**: `quant_engine/constraints.py` v3

All constraints are implemented as **smooth penalty terms** added to the loss. Hard constraints are approximated by continuous functions compatible with autograd.

### Constraint Normalization (v3)

A critical calibration issue identified in v2: if constraint violations are expressed as raw fractions (e.g., `relu(w - 0.10).sum()` = 0.02 for a 2% violation), they are numerically negligible compared to a Sharpe term of ~5.0. The optimizer simply ignores them.

v3 normalizes all violations to be **competitive with the Sharpe** by dividing by a reference scale:

```python
penalty = relu(w - 0.10).sum() / 0.01   # 2% violation → 2.0 (not 0.02)
```

With `λ_UCITS = 10` and a 2% violation: cost = 10 × 2.0 = 20.0. This is 4× the Sharpe contribution at Sharpe = 0.5 — the constraint is now enforced.

### UCITS 5/10/40 Rule (Article 52 OPCVM Directive)

**Rule 1 — 10% per issuer cap:**

```
C₁₀% = Σᵢ relu(wᵢ − 0.10) / 0.01
```

**Rule 2 — Sum of positions > 5% cannot exceed 40%:**

Two modes:
- **Soft** (exploration): `mask_i = sigmoid(T · (wᵢ − 0.05))` — differentiable approximation of the indicator function
- **STE** (production): `mask_i = round(sigmoid(...))` with straight-through estimator for the backward pass — binary in forward, smooth gradient in backward

```
C₅₋₄₀% = relu(Σᵢ wᵢ · mask_i − 0.40) / 0.01
```

### Tracking Error

Penalizes annualized tracking error vs the SX5E synthetic benchmark:

```
TE = std(R_portfolio − R_benchmark)    (daily)
C_TE = relu(TE − te_max) / 0.001
```

`te_max = 0.004` daily ≈ 6.3%/year. The scale 0.001 (0.1 bp) makes a 10bp violation equal to 10 penalty units.

### Sectoral Budget Constraints (ICB)

The portfolio universe is mapped to `n_sectors` ICB industries via a binary mask matrix M ∈ {0,1}^{n×s}:

```
w_sector = M^T · w    ∈ ℝˢ
C_sector = relu(w_sector − budget_sector).sum() / 0.01
```

**Sectoral deviation from benchmark** (new in v3):

```
deviation = |M^T · w − w_benchmark_sector|
C_dev = relu(deviation − tolerance).sum() / 0.01
```

Default tolerance: ±10% per ICB industry. In BL mode, this is automatically widened to ±20% (`activate_bl_corridor()`) to allow views to express themselves without being suppressed by this constraint.

### Cardinality via STE (Straight-Through Estimator)

To control the number of active positions without a non-differentiable argmax:

```python
binary_mask = round(sigmoid(α · (w − threshold)))
k_active = binary_mask.sum()
```

The round function is non-differentiable, but the STE substitutes the identity function in the backward pass: `∂round/∂x ← 1`. This lets the gradient flow through the cardinality constraint.

- `constraint_cardinality_min`: penalizes `relu(k_min − k_active)`
- `constraint_cardinality_max`: penalizes `relu(k_active − k_max)`

---

## Module 05 — Gradient Descent Optimizer

**Source**: `quant_engine/model.py` (PortfolioOptimizer class)

### Optimization Variable

The **only optimized parameter** is `z ∈ ℝⁿ` — a vector of unconstrained pre-weights. Portfolio weights are always obtained through the sparsemax projection:

```
w = sparsemax(z) ∈ Δⁿ    (simplex, non-negative, sum to 1)
```

This is equivalent to optimizing over the simplex directly, but the unconstrained parametrization makes gradient steps unrestricted.

**Initialization**:

```python
if init_scale > 0:
    z = torch.randn(n_assets) * init_scale    # recommended: 0.1
else:
    z = torch.zeros(n_assets)                  # → equal-weight after sparsemax
```

`init_scale = 0.0` creates an equal-weight initialization where all assets have identical pre-weights. If cardinality constraints are active, this may create a **degenerate plateau** where no gradient signal can select among assets at epoch 1. Using `init_scale = 0.1` breaks this symmetry.

### Training Loop

```
for epoch in 1..N:
    w = sparsemax(z)                           # projection
    penalty_terms = collect_constraints(w)     # constraint penalties
    loss, components = compute_total_loss(w)   # objectives + penalties
    if w_prev is not None:
        loss += λ_turnover · 0.5 · |w − w_prev|₁   # turnover penalty
    loss.backward()                            # autograd
    adam.step()                                # update z
```

**Optimizer**: Adam (`lr = 1e-3` default, configurable). Adam is chosen over SGD because the constraint landscape creates sparse, direction-varying gradients that benefit from per-parameter learning rate adaptation.

**Warmup** (optional): the first `warmup_epochs` iterations use softmax instead of sparsemax. This avoids the "too many zeros too early" problem where sparsemax collapses to a near-degenerate solution before the optimizer has explored the weight space.

### Early Stopping with Constraint Guard

Early stopping is triggered when:
1. The loss has not improved by more than `min_delta` for `patience` consecutive epochs
2. **AND** the total constraint penalty is below 5% of the absolute loss value

If condition 1 is met but condition 2 is not (constraints still violated), the patience counter is reset and optimization continues. This prevents stopping at a "good loss but infeasible" solution.

### Final Metrics

After convergence, the following metrics are computed and logged:

| Metric | Formula |
|--------|---------|
| Sharpe (annualized) | `(mean(R) − r_f,daily) / std(R) × √252` |
| CVaR (α = 5%) | `VaR₅% + (1/0.05) · E[max(−R − VaR, 0)]` |
| Volatility (annualized) | `std(R) × √252` |
| Tracking Error | `std(R_portfolio − R_benchmark) × √252` |
| Herfindahl Index | `Σᵢ wᵢ²` (concentration) |
| UCITS 10% check | `max(w) ≤ 0.10` |
| UCITS 5-40% check | `Σ(w_i : w_i > 0.05) ≤ 0.40` |

---

## Hyperparameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `τ` (tau) | 0.025 | Prior uncertainty scalar in BL |
| `δ_shrinkage` | 0.5 | James-Stein shrinkage toward δ=2.5 |
| `init_scale` | 0.1 | Std of random z initialization |
| `n_epochs` | 500 | Maximum training epochs |
| `lr` | 1e-3 | Adam learning rate |
| `patience` | 50 | Early stopping patience |
| `min_delta` | 1e-4 | Min improvement to reset patience |
| `λ_sharpe` | 10.0 | Sharpe objective weight |
| `λ_cvar` | 100.0 | CVaR objective weight |
| `cvar_alpha` | 0.05 | CVaR tail level (5%) |
| `λ_ucits_10pct` | 10.0 | UCITS 10% constraint weight |
| `λ_ucits_5_40pct` | 5.0 | UCITS 5-40% constraint weight |
| `λ_tracking_error` | 3.0 | TE constraint weight |
| `te_max` | 0.004 | Max daily TE (~6.3%/year) |
| `sector_tolerance` | 0.10 | ICB sectoral deviation tolerance (normal) |
| `sector_tolerance_bl` | 0.20 | ICB sectoral deviation tolerance (BL mode) |
| `train_months` | 12 | WFO training window length |
| `test_months` | 1 | WFO test window length |
| `step_months` | 1 | WFO step size |

---

## References

- He, G. & Litterman, R. (1999). *The Intuition Behind Black-Litterman Model Portfolios*. Goldman Sachs Asset Management.
- Cheung, W. (2009). *The Black-Litterman Model Explained*. Journal of Asset Management, 11(4).
- Martins, A. & Astudillo, R. (2016). *From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification*. ICML 2016.
- Rockafellar, R.T. & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk*. Journal of Risk.
- Oliva, J. (2025). *Black-Litterman Portfolio Optimization via Differentiable Sparsemax Projections*. [arXiv:2507.16717](https://arxiv.org/abs/2507.16717).
