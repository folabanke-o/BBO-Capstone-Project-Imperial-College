> **Version:** Beta (Week 10 submission)
> This is a preliminary version of the model card submitted at Week 10.
> It will be updated with final round results by Week 13.
# BBO Capstone Project — Model Card

> Framework: Mitchell et al. (2019) *Model Cards for Model Reporting*

---

## 1. Overview

| Field | Value |
|---|---|
| **Name** | BBO Capstone Optimisation Pipeline |
| **Type** | Sequential Bayesian Optimisation with adaptive surrogate selection |
| **Version** | Week 10 (final) |
| **Functions** | 8 synthetic black-box functions, F1(2D) to F8(8D) |
| **Rounds** | 10 weekly iterations, one irreversible query per function per round |
| **Repository** | This public GitHub repository |

---

## 2. Intended Use

**Suitable tasks:**
- Single-objective maximisation with d in {2, 3, 4, 5, 6, 8} and one query per round
- Small-n settings (n < 50) with calibrated GP uncertainty
- Late-stage convergence: GP Mean and EI xi=0 tuned for final-round exploitation

**Use cases to avoid:**
- Noisy black-box functions (current EI assumes noiseless evaluations)
- Batch queries (pipeline generates one query per function per round)
- Multi-modal functions without a global search phase before exploitation

---

## 3. Strategy Details Across Ten Rounds

| Week | Gains | Primary Method | Key Event |
|---|---|---|---|
| W1 | 4/8 | GP UCB uniform | Broad initial sweep |
| W2 | 2/8 | GP EI constrained | First per-function bounds |
| W3 | 3/8 | GP EI multi-method | Thompson sampling added (F8) |
| W4 | 4/8 | GP EI anchored | Anchor-to-ATB fix introduced |
| W5 | 2/8 | GP EI LOO-CV | Surrogate selection via LOO-CV |
| W6 | 2/8 | GP + DKL + MC Dropout | DKL found F5 dim1-near-zero (+507 units) |
| W7 | 3/8 | ARD kernels + ZoMBI | Colleague ARD suggestion applied |
| W8 | 5/8 | ARD Matern + GP Mean | Best round: ZoMBI F1 breakthrough |
| W9 | 4/8 | GP Mean tight | F3, F5, F7 new ATBs |
| W10 | TBD | LS Ensemble + micro-bounds | Final round |

### 3.1 Surrogate architectures

- **GP ARD-Matern-5/2 (default W8+):** sklearn `GaussianProcessRegressor`, ConstantKernel x Matern(nu=2.5, ARD) + WhiteKernel, Y standardised, `ls_init=sqrt(d)/10`, `n_restarts=15`, `random_state=42`
- **MC Dropout MLP (Gal and Ghahramani 2016):** Input(d)->Lin(max(32,8d))->ReLU->Drop(0.1)->Lin(max(16,4d))->ReLU->Drop(0.1)->Lin(1), T=50 stochastic passes
- **Deep Kernel Learning (Wilson et al. 2016):** NN feature extractor Input(d)->Lin(max(16,4d))->ReLU->Lin(2) jointly trained with GPyTorch exact GP
- **Length scale ensemble (arXiv LB-BO 2025):** 5 GPs with fixed ls in {0.01, 0.05, 0.1, 0.5, 1.0}, EI averaged across ensemble

### 3.2 Acquisition schedule

- W1-W2: UCB kappa=2.0 (uniform)
- W3-W7: EI xi=0.001-0.010 (per-function)
- W8-W10: GP Mean (converging functions); EI xi=0 (F8); Grid step 0.001-0.003 (F6)
- ZoMBI applied to F1 (W7-W9) and F4 (W8-W10)

### 3.3 Key design decisions

- **Anchor-to-ATB rule (W5+):** all search bounds anchored to confirmed all-time best input
- **LOO-CV surrogate selection (W5+):** per-function surrogate chosen by lowest LOO-CV RMSE
- **Colleague suggestions (W8):** ARD kernels; Y scaling on GP; Matern replacing RBF
- **Pinned ZoMBI (W9-W10):** W2 best input forced into memory for F4

---

## 4. Performance

| Fn | Dim | Initial | ATB | ATB Week | Key Methods | Unresolved | Gains/10 |
|---|---|---|---|---|---|---|---|
| F1 | 2D | 7.71e-16 | 7.62e-9 | W9 | GP, ZoMBI, GP Mean | No | 9 |
| F2 | 2D | 0.611 | 0.764 | W8 | GP, DKL, GP Mean | No | 8 |
| F3 | 3D | -0.035 | -5.1e-4 | W9 | GP, MC Drop, GP Mean | No | 7 |
| F4 | 4D | -4.026 | 0.534 | W2 | GP, ZoMBI, LS Ensemble | Yes | 1 |
| F5 | 4D | 1089 | 4440 | W9 | GP, DKL, GP Mean | No | 6 |
| F6 | 5D | -0.714 | -0.680 | W4 | GP Grid | Yes | 3 |
| F7 | 6D | 1.365 | 2.747 | W9 | GP, GP Mean | No | 6 |
| F8 | 8D | 9.598 | 9.996 | W7 | GP, UCB, EI xi=0 | No | 3 |

**Metrics used:**
- All-time best output (primary)
- LOO-CV RMSE (surrogate selection)
- L2 distance from ATB input (query drift diagnostic)
- Gains per round (2/8 to 5/8 across ten rounds)

---

## 5. Assumptions and Limitations

| Assumption | Justification | Risk |
|---|---|---|
| Proximity: optimum near confirmed best input | Proximity pattern held across all 10 rounds | Misses better basins in unexplored space |
| Stationarity: ARD Matern uniform correlation | Standard GP prior | Violated by F5 (DKL required) |
| Noiseless evaluation | Portal returns deterministic outputs | GP Mean/EI xi=0 invalid if noisy |

**Computational constraints:**
- One query per function per week; no iterative refinement
- n < 50; GP cannot identify dimension importance reliably at 6D or 8D
- LOO-CV across 4 surrogates: 4x compute relative to single GP

**Sampling biases:**
- Early broad queries persist and bias GP mean in imbalanced functions (F4: 30 negative, 3 positive)
- By W10, all search boxes within 0.02-0.08 radius of confirmed best; global optima outside this invisible

---

## 6. Ethical Considerations

**Transparency:** Every decision citable: inline paper citations, method_config dictionaries, LOO-CV model cards, ARD diagnostic printouts, weekly post-mortem markdown files.

**Reproducibility:** Fixed `random_state=42` throughout; `requirements.txt` pins library versions. Full query history in repository.

**Real-world relevance:** Irreversible-query discipline maps to drug trials, A/B tests, infrastructure configuration, any setting with expensive delayed feedback.

**Limitations of this card:** Describes a student pipeline with a budget of one query per week. Not suitable for benchmarking commercial BBO systems or functions outside this capstone programme.

---

*Model card follows Mitchell et al. (2019). Last updated: Week 10.*
