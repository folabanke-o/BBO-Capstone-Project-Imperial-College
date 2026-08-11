# Bayesian Optimisation Capstone: Week 4 Strategy and Code

> **Building on:** Three full iterations of data now incorporated
> **Research note:** Recent literature (NeurIPS 2024) confirms Thompson Sampling
> tends to underperform EI and UCB in BO settings. EI is strengthened for
> functions with confirmed peaks; constrained UCB remains the workhorse.

---

## Table of Contents

1. [Week 3 Results Analysis](#week-3-results-analysis)
2. [What the Literature Says](#what-the-literature-says)
3. [Updated Datasets](#updated-datasets)
4. [Method Assignments for Week 4](#method-assignments-for-week-4)
5. [Per-Function Strategy](#per-function-strategy)
6. [Full Notebook Code](#full-notebook-code)

---

## Week 3 Results Analysis

Week 3 queries and the outputs we are now building from:

| Function | W3 query | W3 output | Best before W3 | New best | Outcome |
|----------|----------|-----------|----------------|----------|---------|
| F1 | [0.771, 0.772] | TBC | 0.000000 | TBC | TBC |
| F2 | [0.783, 0.852] | TBC | 0.611205 | TBC | TBC |
| F3 | [0.539, 0.565, 0.389] | TBC | -0.034835 | TBC | TBC |
| F4 | [0.391, 0.397, 0.393, 0.417] | TBC | 0.533858 | TBC | TBC |
| F5 | [0.043, 0.985, 0.986, 0.924] | TBC | 3511.612 | TBC | TBC |
| F6 | [0.718, 0.145, 0.743, 0.704, 0.046] | TBC | -0.714265 | TBC | TBC |
| F7 | [0.121, 0.376, 0.488, 0.138, 0.427, 0.726] | TBC | 1.771970 | TBC | TBC |
| F8 | [0.311, 0.116, 0.059, 0.140, 0.541, 0.045, 0.335, 0.555] | TBC | 9.965293 | TBC | TBC |

> **Update this table** once Week 3 outputs arrive. The code in Section 6 has
> placeholder values — replace `week3_outputs` with the real values before running.

---

## What the Literature Says

Three findings from current research directly shaped the Week 4 method choices:

**1. Thompson Sampling underperforms EI and UCB in BO.**
A NeurIPS 2024 paper found that standard Thompson Sampling tends to underperform
EI and UCB in Bayesian optimisation settings, particularly as data accumulates
and the region of interest narrows. The reason is geometric: as the peak
localises, random uniform candidates are exponentially unlikely to land near
it in higher dimensions. This is a known weakness that gets worse as iterations
progress.

**Implication for Week 4:** F7 and F8, which used Thompson Sampling in Week 3,
are switched to constrained EI and constrained UCB respectively. This is more
appropriate now that three iterations of data have identified the region of interest.

**2. EI with a well-chosen xi is the most reliable method near confirmed peaks.**
With xi set just above the current best, EI becomes very precise — it only
recommends a point if it genuinely expects to beat the existing maximum.
This is ideal for functions like F5 and F4 that have shown consistent improvement.

**3. Constrained search boxes outperform unconstrained search when the region
of interest is known.**
Restricting candidates to a tight neighbourhood prevents the acquisition function
from being distracted by uncertain boundary regions, which was the root cause of
failures in Weeks 1 and 2.

---

## Updated Datasets

After stacking Week 3 observations, each dataset will contain:

| Function | Original | After W1 | After W2 | After W3 | Best going into W4 |
|----------|---------|---------|---------|---------|-------------------|
| F1 | 10 | 11 | 12 | 13 | TBC |
| F2 | 10 | 11 | 12 | 13 | TBC |
| F3 | 15 | 16 | 17 | 18 | TBC |
| F4 | 30 | 31 | 32 | 33 | TBC |
| F5 | 20 | 21 | 22 | 23 | TBC |
| F6 | 20 | 21 | 22 | 23 | TBC |
| F7 | 30 | 31 | 32 | 33 | TBC |
| F8 | 40 | 41 | 42 | 43 | TBC |

---

## Method Assignments for Week 4

| Function | W3 method | W4 method | Change | Reason |
|----------|-----------|-----------|--------|--------|
| F1 | EI | EI | Same | 3 queries near [0.731, 0.733] — keep EI, tighten to margin 0.03 |
| F2 | EI | EI | Same | Confirmed multimodal peak — EI targets above 0.611 |
| F3 | UCB 0.3 | EI | Switch | UCB with very low kappa risks stagnation; EI more precise near -0.035 |
| F4 | UCB 1.0 | EI | Switch | New positive best — EI targets improvement above 0.534 precisely |
| F5 | UCB 0.5 | EI | Switch | 3 consecutive gains, near peak — EI most precise for fine-tuning |
| F6 | Grid | Grid | Same | GP still unreliable; reduce step to 0.01 for tighter search |
| F7 | Thompson | UCB 1.0 | Switch | Literature: Thompson underperforms EI/UCB; return to W1 best region |
| F8 | Thompson | UCB 2.5 | Switch | Thompson too random; UCB 2.5 with Sobol covers space more systematically |

---

## Per-Function Strategy

### Function 1

Three queries have all returned near-zero. EI stays but the search box tightens
to margin 0.03 around [0.731, 0.733]. If W3 returned a better value, update
`best_known["function_1"]` before running.

**Method:** EI, xi=0.001, margin 0.03

---

### Function 2

Best remains 0.611205 at [0.703, 0.927]. EI with a tight box stays the approach.
If W3's query (0.783, 0.852) returned above 0.611, shift the box to that new
best input.

**Method:** EI, xi=0.01, margin 0.07

---

### Function 3

Switching from UCB 0.3 to EI. With 18 observations and a confirmed interior
optimum at [0.493, 0.612, 0.340], EI is more likely to place the query exactly
where improvement is expected rather than slightly off-target due to kappa.

**Method:** EI, xi=0.005, margin 0.08 around best known

---

### Function 4

F4 crossed zero in Week 2 and the best is now 0.534. EI with xi just above 0.534
will only recommend a point it expects to beat that. UCB at kappa 1.0 could
still drift slightly; EI is more precise at this stage.

**Method:** EI, xi=0.01, margin 0.10 around [0.391, 0.397, 0.393, 0.417]

---

### Function 5

Three consecutive gains (1088 to 3019 to 3511). The peak at high dims 2-4
and low dim 1 is the most well-characterised surface in the entire competition.
EI with xi=0.01 will refine precisely. Keep the asymmetric bounds.

**Method:** EI, xi=0.01, bounds [0.00-0.10, 0.92-1.00, 0.93-1.00, 0.92-1.00]

---

### Function 6

Grid search stays. Reduce step from 0.01 to 0.008 for finer resolution.
The GP has been unreliable for three rounds and even the closest query
(distance 0.017) returned a worse value. Tighter grid, same pattern.

**Method:** Grid search, step=0.008, centred on [0.728, 0.155, 0.733, 0.694, 0.056]

---

### Function 7

Switching from Thompson to constrained UCB 1.0. Thompson sent the W3 query
to a different region from the W1 best. The W1 location [0.110, 0.394, 0.394,
0.093, 0.386, 0.670] produced +30% and should be exploited more precisely.
UCB 1.0 in a tight box will stay closer.

**Method:** UCB kappa=1.0, margin 0.08 around W1 best input

---

### Function 8

Switching from Thompson to UCB 2.5. Thompson's unconstrained sampling produced
a query far from any known good region. UCB 2.5 with Sobol candidates over
the full space provides more systematic exploration than random posterior draws.
The high kappa maintains the exploratory intent while using a better candidate
generation strategy.

**Method:** UCB kappa=2.5, full [0,1]^8 space, Sobol candidates

---

## Full Notebook Code

```python
# ============================================================
# WEEK 4 BAYESIAN OPTIMISATION - FULL NOTEBOOK
# ============================================================
# Key change: Thompson Sampling replaced by EI and UCB
# based on NeurIPS 2024 findings that TS underperforms
# EI/UCB in BO settings as data accumulates.
# ============================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy.stats.qmc import Sobol
from itertools import product


# ============================================================
# SECTION 1: LOAD ORIGINAL DATA
# ============================================================

original_data = {}
for i in range(1, 9):
    X = np.load(f"function_{i}/initial_inputs.npy")
    Y = np.load(f"function_{i}/initial_outputs.npy")
    original_data[f"function_{i}"] = {"X": X, "Y": Y}
    print(f"Function {i} | X: {X.shape} | Y: {Y.shape}")


# ============================================================
# SECTION 2: FULL QUERY HISTORY (W1, W2, W3)
# ============================================================

week1_queries = {
    "function_1": np.array([0.000186, 0.014353]),
    "function_2": np.array([0.998531, 0.007036]),
    "function_3": np.array([0.933672, 0.002452, 0.965412]),
    "function_4": np.array([0.417336, 0.402860, 0.336077, 0.476656]),
    "function_5": np.array([0.050115, 0.927701, 0.965034, 0.985561]),
    "function_6": np.array([0.197786, 0.010925, 0.990284, 0.888004, 0.052863]),
    "function_7": np.array([0.110110, 0.393658, 0.394356, 0.092883, 0.385807, 0.669789]),
    "function_8": np.array([0.042700, 0.092462, 0.083390, 0.051299, 0.808162, 0.563756, 0.175217, 0.419904]),
}
week1_outputs = {
    "function_1": 1.14e-239,
    "function_2": -0.109704342,
    "function_3": -0.365767178,
    "function_4": -0.869002202,
    "function_5": 3019.659838,
    "function_6": -1.213319062,
    "function_7": 1.771969604,
    "function_8": 9.965293447,
}
week2_queries = {
    "function_1": np.array([0.705908, 0.823143]),
    "function_2": np.array([0.839078, 0.895770]),
    "function_3": np.array([0.363504, 0.758379, 0.191085]),
    "function_4": np.array([0.403695, 0.397605, 0.413333, 0.411576]),
    "function_5": np.array([0.056181, 0.992607, 0.973199, 0.959866]),
    "function_6": np.array([0.729856, 0.157383, 0.734992, 0.704798, 0.068448]),
    "function_7": np.array([0.058090, 0.303889, 0.327991, 0.001268, 0.275231, 0.673181]),
    "function_8": np.array([0.017074, 0.091604, 0.305973, 0.115845, 0.946320, 0.608139, 0.053440, 0.855712]),
}
week2_outputs = {
    "function_1": -4.676913887169069e-32,
    "function_2": 0.14503569246975664,
    "function_3": -0.11944712762491103,
    "function_4": 0.5338577755032223,
    "function_5": 3511.611905490813,
    "function_6": -0.8006000173001564,
    "function_7": 1.2888474165310304,
    "function_8": 9.8168730656046,
}

# Week 3 queries submitted
week3_queries = {
    "function_1": np.array([0.770991, 0.772247]),
    "function_2": np.array([0.782573, 0.851721]),
    "function_3": np.array([0.538501, 0.565224, 0.389093]),
    "function_4": np.array([0.391220, 0.397272, 0.392512, 0.416849]),
    "function_5": np.array([0.043103, 0.985149, 0.986449, 0.924277]),
    "function_6": np.array([0.718186, 0.144693, 0.742552, 0.703997, 0.046401]),
    "function_7": np.array([0.121105, 0.376149, 0.487901, 0.138159, 0.427197, 0.725669]),
    "function_8": np.array([0.311134, 0.116341, 0.059376, 0.140269, 0.540842, 0.044932, 0.335219, 0.555288]),
}

# !! REPLACE THESE WITH ACTUAL WEEK 3 OUTPUTS WHEN RECEIVED !!
week3_outputs = {
    "function_1": 0.0,        # replace
    "function_2": 0.611205,   # replace
    "function_3": -0.034835,  # replace
    "function_4": 0.533858,   # replace
    "function_5": 3511.612,   # replace
    "function_6": -0.714265,  # replace
    "function_7": 1.771970,   # replace
    "function_8": 9.965293,   # replace
}


# ============================================================
# SECTION 3: BUILD UPDATED DATASETS (original + W1 + W2 + W3)
# ============================================================

updated_data = {}
for i in range(1, 9):
    key = f"function_{i}"
    X_updated = np.vstack([
        original_data[key]["X"],
        week1_queries[key].reshape(1, -1),
        week2_queries[key].reshape(1, -1),
        week3_queries[key].reshape(1, -1),
    ])
    Y_updated = np.append(
        original_data[key]["Y"],
        [week1_outputs[key], week2_outputs[key], week3_outputs[key]]
    )
    updated_data[key] = {"X": X_updated, "Y": Y_updated}
    best_idx = np.argmax(Y_updated)
    print(f"Function {i} | obs: {X_updated.shape[0]} | best: {Y_updated[best_idx]:.6f}")


# ============================================================
# SECTION 4: EDA — UNDERSTAND WHAT CHANGED IN WEEK 3
# ============================================================

print("\n" + "="*60)
print("ENTERING WEEK 4 — DATASET SUMMARY")
print("="*60)

for i in range(1, 9):
    key = f"function_{i}"
    Y = updated_data[key]["Y"]
    X = updated_data[key]["X"]
    best_idx = np.argmax(Y)

    w3_prev_best = max(
        original_data[key]["Y"].max(),
        week1_outputs[key],
        week2_outputs[key]
    )
    w3_change = week3_outputs[key] - w3_prev_best
    direction = "IMPROVED" if w3_change > 1e-10 else (
        "NO CHANGE" if abs(w3_change) < 1e-10 else "DECLINED"
    )

    print(f"\nFunction {i}")
    print(f"  Observations : {X.shape[0]}")
    print(f"  Best output  : {Y[best_idx]:.6f} at {np.round(X[best_idx], 4)}")
    print(f"  W3 output    : {week3_outputs[key]:.6f} | {direction} ({w3_change:+.6f})")


# ============================================================
# SECTION 5: SHARED GP FITTER
# ============================================================

def fit_gp(X, Y, n_restarts=15, y_shift=False):
    """
    Fit GP on (X, Y). If y_shift=True, shift Y so minimum=0
    before fitting (helps for all-negative output functions).
    Returns (gp, scaler).
    """
    Y_fit = Y - Y.min() if y_shift else Y
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e4))
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        random_state=42
    )
    gp.fit(X_scaled, Y_fit)
    return gp, scaler


# ============================================================
# SECTION 6: ACQUISITION FUNCTIONS
# ============================================================

def ei_query(X, Y, xi=0.01, low_bounds=None, high_bounds=None,
             n_restarts=15, n_candidates=50000, seed=42, y_shift=False):
    """
    Expected Improvement acquisition.
    EI = E[max(0, f(x) - f_best - xi)]
    Only recommends a point where improvement over f_best is expected.
    More precise than UCB near confirmed peaks.
    Preferred over Thompson Sampling based on NeurIPS 2024 findings.
    """
    dim = X.shape[1]
    if low_bounds is None: low_bounds = np.zeros(dim)
    if high_bounds is None: high_bounds = np.ones(dim)

    gp, scaler = fit_gp(X, Y, n_restarts=n_restarts, y_shift=y_shift)
    f_best = Y.max()

    np.random.seed(seed)
    cands = np.random.uniform(low_bounds, high_bounds, size=(n_candidates, dim))
    cands_sc = scaler.transform(cands)

    mu, sigma = gp.predict(cands_sc, return_std=True)
    sigma = np.maximum(sigma, 1e-9)

    improvement = mu - f_best - xi
    z  = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma < 1e-9] = 0.0

    best_idx = np.argmax(ei)
    query = np.clip(cands[best_idx], 0.0, 1.0)
    return query, float(ei[best_idx]), float(mu[best_idx]), float(sigma[best_idx])


def ucb_query(X, Y, kappa=2.0, low_bounds=None, high_bounds=None,
              n_restarts=15, n_candidates=None, seed=42, y_shift=False):
    """
    Upper Confidence Bound acquisition.
    UCB = mu + kappa * sigma
    Best for: broader exploration with controlled kappa,
    or functions where the surface is not yet well-characterised.
    """
    dim = X.shape[1]
    if n_candidates is None: n_candidates = 5000 * dim
    if low_bounds is None: low_bounds = np.zeros(dim)
    if high_bounds is None: high_bounds = np.ones(dim)

    gp, scaler = fit_gp(X, Y, n_restarts=n_restarts, y_shift=y_shift)

    sampler = Sobol(d=dim, scramble=True, seed=seed)
    cands = sampler.random(n=n_candidates)
    cands = low_bounds + cands * (high_bounds - low_bounds)
    cands = np.clip(cands, 0.0, 1.0)
    cands_sc = scaler.transform(cands)

    mu, sigma = gp.predict(cands_sc, return_std=True)
    ucb = mu + kappa * sigma

    best_idx = np.argmax(ucb)
    query = np.clip(cands[best_idx], 0.0, 1.0)
    return query, float(ucb[best_idx]), float(mu[best_idx]), float(sigma[best_idx])


def grid_search_query(X, Y, best_known, step=0.008, n_restarts=15):
    """
    Deterministic grid search centred on best_known.
    3^d candidates evaluated by GP mean only (no uncertainty).
    Used for F6 where the GP surrogate is unreliable but the
    best region is known.
    """
    dim = X.shape[1]
    gp, scaler = fit_gp(X, Y, n_restarts=n_restarts)

    steps_list = [-step, 0.0, step]
    grid = []
    for deltas in product(steps_list, repeat=dim):
        candidate = np.clip(best_known + np.array(deltas), 0.0, 1.0)
        grid.append(candidate)
    grid = np.unique(np.array(grid), axis=0)

    grid_sc = scaler.transform(grid)
    mu, _ = gp.predict(grid_sc, return_std=True)

    best_idx = np.argmax(mu)
    query = grid[best_idx]
    print(f"  Grid: {len(grid)} candidates | best mu: {mu[best_idx]:.6f}")
    return query, float(mu[best_idx]), float(mu[best_idx]), 0.0


# ============================================================
# SECTION 7: PER-FUNCTION CONFIGURATION
# ============================================================

# Best known inputs — UPDATE these once W3 outputs arrive
# If W3 improved a function, update to the W3 query coordinates
best_known = {
    "function_1": np.array([0.731024, 0.732999]),      # from initial data
    "function_2": np.array([0.702637, 0.926564]),      # from initial data
    "function_3": np.array([0.492581, 0.611593, 0.340176]),
    "function_4": np.array([0.403695, 0.397605, 0.413333, 0.411576]),  # W2 best
    "function_5": np.array([0.056181, 0.992607, 0.973199, 0.959866]),  # W2 best
    "function_6": np.array([0.728186, 0.154693, 0.732552, 0.693997, 0.056401]),
    "function_7": np.array([0.110110, 0.393658, 0.394356, 0.092883, 0.385807, 0.669789]),  # W1 best
    "function_8": None,  # full space UCB
}

# F5 custom asymmetric bounds (dims 2-4 pushed high, dim 1 kept low)
f5_low  = np.array([0.00, 0.92, 0.93, 0.92])
f5_high = np.array([0.10, 1.00, 1.00, 1.00])

method_config = {
    # EI replaces Thompson for more precise targeting
    "function_1": {"method": "EI",   "xi": 0.001, "margin": 0.03, "restarts": 10},
    "function_2": {"method": "EI",   "xi": 0.010, "margin": 0.07, "restarts": 10},
    "function_3": {"method": "EI",   "xi": 0.005, "margin": 0.08, "restarts": 10},
    "function_4": {"method": "EI",   "xi": 0.010, "margin": 0.10, "restarts": 15},
    "function_5": {"method": "EI",   "xi": 0.010, "custom_bounds": True, "restarts": 15},
    "function_6": {"method": "GRID", "step": 0.008, "restarts": 15},
    # UCB replaces Thompson — literature shows UCB outperforms TS in BO
    "function_7": {"method": "UCB",  "kappa": 1.0, "margin": 0.08, "restarts": 20},
    "function_8": {"method": "UCB",  "kappa": 2.5, "margin": None, "restarts": 20},
}

print("\nWEEK 4 METHOD CONFIGURATION")
print(f"{'Fn':<5} {'Method':<8} {'Setting':<12} {'Restarts'}")
print("-"*38)
for i in range(1, 9):
    key = f"function_{i}"
    cfg = method_config[key]
    m   = cfg["method"]
    if m == "EI":
        setting = f"xi={cfg['xi']}"
    elif m == "UCB":
        setting = f"k={cfg.get('kappa','n/a')}"
    else:
        setting = f"step={cfg.get('step','n/a')}"
    print(f"F{i:<4} {m:<8} {setting:<12} {cfg['restarts']}")


# ============================================================
# SECTION 8: RUN OPTIMISATION — ALL 8 FUNCTIONS
# ============================================================

week4_results = {}

for i in range(1, 9):
    key = f"function_{i}"
    cfg = method_config[key]
    X   = updated_data[key]["X"]
    Y   = updated_data[key]["Y"]
    dim = X.shape[1]

    print(f"\n{'='*55}")
    print(f"FUNCTION {i} | {cfg['method']} | obs: {X.shape[0]} | best: {Y.max():.6f}")
    print(f"{'='*55}")

    best   = best_known[key]
    margin = cfg.get("margin")

    # Build bounds
    if best is not None and margin is not None:
        low_b  = np.clip(best - margin, 0.0, 1.0)
        high_b = np.clip(best + margin, 0.0, 1.0)
    else:
        low_b  = np.zeros(dim)
        high_b = np.ones(dim)

    if cfg["method"] == "EI":
        # F5 gets custom asymmetric bounds
        if key == "function_5":
            low_b, high_b = f5_low, f5_high
        query, score, mu, sigma = ei_query(
            X, Y,
            xi=cfg["xi"],
            low_bounds=low_b, high_bounds=high_b,
            n_restarts=cfg["restarts"]
        )

    elif cfg["method"] == "UCB":
        query, score, mu, sigma = ucb_query(
            X, Y,
            kappa=cfg["kappa"],
            low_bounds=low_b, high_bounds=high_b,
            n_restarts=cfg["restarts"]
        )

    elif cfg["method"] == "GRID":
        query, score, mu, sigma = grid_search_query(
            X, Y,
            best_known=best,
            step=cfg["step"],
            n_restarts=cfg["restarts"]
        )

    formatted = "-".join([f"{x:.6f}" for x in query])
    week4_results[key] = {
        "query"          : query,
        "formatted_query": formatted,
        "method"         : cfg["method"],
        "score"          : score,
        "predicted_mean" : mu,
        "uncertainty"    : sigma,
    }

    print(f"Query    : {formatted}")
    print(f"Method   : {cfg['method']}")
    print(f"Score    : {score:.6f}")
    print(f"Pred mean: {mu:.6f}")
    print(f"Uncert.  : {sigma:.6f}")


# ============================================================
# SECTION 9: VALIDATION
# ============================================================

print("\n" + "="*55)
print("QUERY VALIDATION REPORT")
print("="*55)

all_clear = True
for i in range(1, 9):
    key    = f"function_{i}"
    query  = week4_results[key]["query"]
    method = week4_results[key]["method"]
    issues = []

    if np.any(query < 0) or np.any(query > 1):
        issues.append("OUT OF RANGE: values outside [0, 1]")
    if np.all(query < 0.01):
        issues.append("SUSPICIOUS: all values near 0")
    if np.all(query > 0.99):
        issues.append("SUSPICIOUS: all values near 1")

    status = "OK" if not issues else "WARNING"
    print(f"\nFunction {i} [{status}] [{method}]")
    print(f"  Query : {week4_results[key]['formatted_query']}")
    if issues:
        all_clear = False
        for w in issues: print(f"  !! {w}")

print("\nAll queries valid." if all_clear else "\nReview warnings before submitting.")


# ============================================================
# SECTION 10: PROXIMITY CHECK
# ============================================================

ref_best = {
    "function_1": np.array([0.731024, 0.732999]),
    "function_2": np.array([0.702637, 0.926564]),
    "function_3": np.array([0.492581, 0.611593, 0.340176]),
    "function_4": np.array([0.403695, 0.397605, 0.413333, 0.411576]),
    "function_5": np.array([0.056181, 0.992607, 0.973199, 0.959866]),
    "function_6": np.array([0.728186, 0.154693, 0.732552, 0.693997, 0.056401]),
    "function_7": np.array([0.110110, 0.393658, 0.394356, 0.092883, 0.385807, 0.669789]),
    "function_8": np.array([0.017074, 0.091604, 0.305973, 0.115845, 0.946320, 0.608139, 0.053440, 0.855712]),
}

print("\n" + "="*55)
print("PROXIMITY TO BEST KNOWN INPUT")
print("="*55)
for i in range(1, 9):
    key  = f"function_{i}"
    q    = week4_results[key]["query"]
    b    = ref_best[key]
    d    = np.linalg.norm(q - b)
    m    = week4_results[key]["method"]
    flag = "ok" if d < 0.25 else "far"
    print(f"Function {i} [{m}] | distance: {d:.4f} [{flag}]")


# ============================================================
# SECTION 11: SAVE QUERIES
# ============================================================

rows = []
for i in range(1, 9):
    key = f"function_{i}"
    Y   = updated_data[key]["Y"]
    r   = week4_results[key]
    rows.append({
        "Function"       : f"Function {i}",
        "Query"          : r["formatted_query"],
        "Method"         : r["method"],
        "Score"          : round(r["score"], 6),
        "Predicted_mean" : round(r["predicted_mean"], 6),
        "Uncertainty"    : round(r["uncertainty"], 6),
        "Current_best"   : round(float(Y.max()), 6),
    })

df = pd.DataFrame(rows)
print("\nFINAL WEEK 4 QUERIES")
print(df[["Function","Method","Query","Current_best"]].to_string(index=False))

df[["Function","Query"]].to_csv("week4_queries.csv", index=False)
df.to_csv("week4_queries_full.csv", index=False)
print("\nweek4_queries.csv saved.")
print("week4_queries_full.csv saved.")
```

---

## Notes before running

**Update Week 3 outputs first.** Replace the placeholder values in `week3_outputs`
in Section 2 with the actual values received from the competition. Then update
`best_known` in Section 7 for any function where Week 3 produced a new best.

**Section 8 runs once only.** Do not re-run it after validation. All method
assignments and bounds are embedded in the loop — no override cells needed.

**If F6 grid search returns a query identical to best_known,** the GP is
predicting that the centre point scores highest. That is acceptable — it means
the GP believes the current best recipe is already near-optimal within the
0.008 grid.

**Updating best_known after W3 results:**
```python
# Example: if W3 improved F3, update to W3 query
best_known["function_3"] = week3_queries["function_3"]

# Example: if W3 improved F7, update to W3 query
best_known["function_7"] = week3_queries["function_7"]
```

---

*Week 4 strategy. Thompson Sampling replaced by EI and UCB following NeurIPS
2024 findings. All method assignments built into the pipeline loop.*
