# Bayesian Optimisation Capstone: Week 5 Strategy

> **Research basis:** Bayesian optimisation reliably converges close to the
> optimum within roughly 19 to 20 unique evaluations, and sampling shifts from
> exploration-dominated to exploitation-dominated as evaluations stabilise
> (iGEM Imperial 2025; arXiv 2512.12649). With 33 to 47 observations per
> function now, the project is firmly in the exploitation phase.

---

## Week 4 Results — Full Analysis

| Fn | Init | W1 | W2 | W3 | W4 | Running best | W4 outcome |
|----|------|----|----|----|----|--------------|-----------|
| F1 | ~0 | ~0 | ~0 | ~0 | -1.61e-16 | ~0 | No gain |
| F2 | 0.6112 | -0.1097 | 0.1450 | 0.0917 | **0.3325** | 0.6112 | Best since initial data |
| F3 | -0.0348 | -0.3658 | -0.1194 | -0.0214 | **-0.0061** | **-0.0061** | **NEW BEST** |
| F4 | -4.0255 | -0.8690 | 0.5339 | 0.2314 | **0.4507** | 0.5339 | Second-best ever |
| F5 | 1088.9 | 3019.7 | 3511.6 | 3214.8 | **3905.2** | **3905.2** | **NEW BEST (+21.5%)** |
| F6 | -0.7143 | -1.2133 | -0.8006 | -0.7036 | **-0.6802** | **-0.6802** | **NEW BEST** |
| F7 | 1.3650 | 1.7720 | 1.2888 | 1.9192 | **2.0189** | **2.0189** | **NEW BEST** |
| F8 | 9.5985 | 9.9653 | 9.8169 | 9.6172 | **9.9609** | 9.9653 | Within 0.0044 of best |

**Headline: 4 of 8 functions set new all-time bests in Week 4.** This is the
strongest round so far. F3, F5, F6, and F7 all improved on their previous best.
F2, F4, and F8 did not set new records but each posted their second-best
result ever, indicating the search is converging even where it has not yet
overtaken the record.

**What worked:**
- F5's dim 4 floor fix (raised to 0.95 in W4) fully reversed the W3 decline,
  recovering and then exceeding the W2 peak by 21.5%.
- F6 and F7's strategy of exploiting the immediately prior week's best input
  with a tight EI/Grid box produced back-to-back new bests.
- F8's return to the W1 best region with UCB kappa 1.5 came within 0.0044 of
  the all-time best, confirming that region is correct.

**What still needs attention:**
- F1 remains stuck near zero after four queries. The function's peak is
  narrower than any constraint box tried so far.
- F4 has not beaten its W2 result despite three close attempts, suggesting
  the W2 coordinates may sit very near a sharp local optimum.

---

## Week 5 Method Assignments

| Fn | Method | Setting | Bounds | Reasoning |
|----|--------|---------|--------|-----------|
| F1 | Grid | step=0.015 | 5x5 grid around [0.731, 0.733] | EI has failed 4 times; switch to exhaustive local grid |
| F2 | EI | xi=0.003, margin 0.04 | around W4 best [0.733, 0.897] | Best-yet result; tighten further to finish convergence |
| F3 | EI | xi=0.002, margin 0.04 | around W4 best [0.578, 0.529, 0.427] | 3 consecutive gains; keep exploiting same trajectory |
| F4 | EI | xi=0.003, margin 0.025 | very tight around W2 best [0.404, 0.398, 0.413, 0.412] | Test if W2 exact point is the true local optimum |
| F5 | EI | xi=0.005, dim4≥0.96 | [0.00-0.12, 0.94-1.00, 0.93-1.00, **0.96-1.00**] | Raise dim 4 floor further; keep exploiting peak |
| F6 | Grid | step=0.004 | centred on W4 best [0.712, 0.139, 0.749, 0.710, 0.040] | 3 consecutive gains; tighten grid further |
| F7 | EI | xi=0.005, margin 0.05 | around W4 best (new best input) | 2 consecutive new bests; tighten and continue |
| F8 | UCB | kappa=1.0, margin 0.05 | around W1 best [0.043, 0.092, 0.083, 0.051, 0.808, 0.564, 0.175, 0.420] | 0.0044 from beating all-time best; final push |

---

## Per-Function Reasoning

**F1 — switching to grid search.** Four EI queries across four weeks have all
returned values indistinguishable from zero. The function's peak appears
narrower than any EI constraint box tried (margins from 0.10 down to 0.025).
A 5x5 deterministic grid with step 0.015 around [0.731, 0.733] will
systematically sample 25 points within a 0.06 x 0.06 box, scored by GP mean,
giving the best chance of landing inside whatever narrow peak exists.

**F2 — tighten further.** W4's 0.3325 is the best result since the initial
dataset. The trend across W2 to W4 (0.145 → 0.092 → 0.333) shows EI converging.
Margin reduced from 0.06 to 0.04 and xi reduced from 0.005 to 0.003 for sharper
exploitation.

**F3 — stay the course.** Three consecutive weeks of improvement (W2 to W4)
on the same EI strategy. No reason to change method; tighten margin from 0.06
to 0.04 to continue refining.

**F4 — test the exact W2 point.** Three weeks of nearby queries (W2, W3, W4)
have all clustered around [0.39-0.40, 0.39-0.40, 0.41-0.42, 0.41-0.43] without
beating W2's exact 0.5339. A very tight margin of 0.025 centred specifically
on the W2 coordinates will test whether that specific point is a sharp local
optimum that nearby queries keep narrowly missing.

**F5 — raise the dim 4 floor again.** The W3 decline (dim4=0.924) and W4
recovery (dim4=0.999) confirm dim 4 is the critical dimension. Raising the
floor from 0.95 to 0.96 locks in the productive range while still allowing
some flexibility.

**F6 — tighten the grid.** Three consecutive gains from grid search
(step 0.008 → 0.006, now 0.004). The pattern of incremental improvement
through finer grids is working; continue it.

**F7 — exploit the new best precisely.** Two consecutive new bests confirm
this region is genuinely good. Margin reduced from 0.08 to 0.05.

**F8 — finish the job.** W4 missed the all-time best by only 0.0044.
A tighter UCB box (margin 0.05, kappa reduced to 1.0) should be enough to
finally beat the W1 record.

---

## Full Notebook Code

```python
# ============================================================
# WEEK 5 BAYESIAN OPTIMISATION — FULL NOTEBOOK
# ============================================================
# W4 confirmed new bests: F3, F5, F6, F7
# W4 close misses: F2 (best since initial), F4 (2nd best), F8 (0.0044 from best)
# W4 still struggling: F1 (4 queries, no signal)
# Strategy: tighten EI margins where converging, switch F1 to grid,
# raise F5 dim4 floor further, finish F8 with tighter UCB.
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
# SECTION 2: COMPLETE QUERY HISTORY — ALL 4 WEEKS
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
    "function_1": 1.13739520e-239, "function_2": -0.10970434, "function_3": -0.36576718,
    "function_4": -0.86900220, "function_5": 3019.65984, "function_6": -1.21331906,
    "function_7": 1.77196960, "function_8": 9.96529345,
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
    "function_1": -4.67691389e-32, "function_2": 0.14503569, "function_3": -0.11944713,
    "function_4": 0.53385778, "function_5": 3511.61191, "function_6": -0.80060002,
    "function_7": 1.28884742, "function_8": 9.81687307,
}
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
week3_outputs = {
    "function_1": 5.07920747e-29, "function_2": 0.09168177, "function_3": -0.02142785,
    "function_4": 0.23137215, "function_5": 3214.79278, "function_6": -0.70363809,
    "function_7": 1.91924944, "function_8": 9.61721741,
}
week4_queries = {
    "function_1": np.array([0.755943, 0.708019]),
    "function_2": np.array([0.732540, 0.896588]),
    "function_3": np.array([0.578454, 0.529081, 0.427267]),
    "function_4": np.array([0.391082, 0.388623, 0.421155, 0.426886]),
    "function_5": np.array([0.073441, 0.999981, 0.958412, 0.999276]),
    "function_6": np.array([0.712186, 0.138693, 0.748552, 0.709997, 0.040401]),
    "function_7": np.array([0.062595, 0.413172, 0.555018, 0.158477, 0.386036, 0.699222]),
    "function_8": np.array([0.007140, 0.131981, 0.126210, 0.111602, 0.760069, 0.618063, 0.244768, 0.488294]),
}
week4_outputs = {
    "function_1": -1.6059656084544877e-16,
    "function_2": 0.33247906550220524,
    "function_3": -0.0060824161981239185,
    "function_4": 0.4507094767251023,
    "function_5": 3905.150046905718,
    "function_6": -0.6801523243364344,
    "function_7": 2.01886276802693,
    "function_8": 9.9609195726939,
}


# ============================================================
# SECTION 3: BUILD UPDATED DATASETS (original + W1-W4)
# ============================================================

updated_data = {}
for i in range(1, 9):
    key = f"function_{i}"
    X_updated = np.vstack([
        original_data[key]["X"],
        week1_queries[key].reshape(1, -1),
        week2_queries[key].reshape(1, -1),
        week3_queries[key].reshape(1, -1),
        week4_queries[key].reshape(1, -1),
    ])
    Y_updated = np.append(
        original_data[key]["Y"],
        [week1_outputs[key], week2_outputs[key], week3_outputs[key], week4_outputs[key]]
    )
    updated_data[key] = {"X": X_updated, "Y": Y_updated}
    best_idx = np.argmax(Y_updated)
    print(f"Function {i} | obs: {X_updated.shape[0]} | best: {Y_updated[best_idx]:.6f} "
          f"at {np.round(X_updated[best_idx], 4)}")


# ============================================================
# SECTION 4: EDA — STATUS ENTERING WEEK 5
# ============================================================

print("\n" + "="*65)
print("ENTERING WEEK 5 — RUNNING BESTS")
print("="*65)

all_bests = {
    "function_1": {"val": 7.71e-16,    "inp": np.array([0.731024, 0.732999]),                              "src": "initial"},
    "function_2": {"val": 0.61120522,  "inp": np.array([0.702637, 0.926564]),                              "src": "initial"},
    "function_3": {"val": -0.00608242, "inp": np.array([0.578454, 0.529081, 0.427267]),                    "src": "W4"},
    "function_4": {"val": 0.53385778,  "inp": np.array([0.403695, 0.397605, 0.413333, 0.411576]),          "src": "W2"},
    "function_5": {"val": 3905.15005,  "inp": np.array([0.073441, 0.999981, 0.958412, 0.999276]),          "src": "W4"},
    "function_6": {"val": -0.68015232, "inp": np.array([0.712186, 0.138693, 0.748552, 0.709997, 0.040401]),"src": "W4"},
    "function_7": {"val": 2.01886277,  "inp": np.array([0.062595, 0.413172, 0.555018, 0.158477, 0.386036, 0.699222]), "src": "W4"},
    "function_8": {"val": 9.96529345,  "inp": np.array([0.042700, 0.092462, 0.083390, 0.051299, 0.808162, 0.563756, 0.175217, 0.419904]), "src": "W1"},
}

for i in range(1, 9):
    key = f"function_{i}"
    b = all_bests[key]
    print(f"F{i} | best={b['val']:.6f} (from {b['src']}) | {np.round(b['inp'], 4)}")


# ============================================================
# SECTION 5: SHARED GP FITTER
# ============================================================

def fit_gp(X, Y, n_restarts=15, y_shift=False):
    Y_fit = Y - Y.min() if y_shift else Y
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e4))
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=n_restarts, random_state=42
    )
    gp.fit(X_scaled, Y_fit)
    return gp, scaler


# ============================================================
# SECTION 6: ACQUISITION FUNCTIONS
# ============================================================

def ei_query(X, Y, xi=0.01, low_bounds=None, high_bounds=None,
             n_restarts=15, n_candidates=60000, seed=42, y_shift=False):
    """Expected Improvement. EI = E[max(0, f(x) - f_best - xi)]"""
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
    z = improvement / sigma
    ei_vals = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei_vals[sigma < 1e-9] = 0.0

    best_idx = np.argmax(ei_vals)
    query = np.clip(cands[best_idx], 0.0, 1.0)
    return query, float(ei_vals[best_idx]), float(mu[best_idx]), float(sigma[best_idx])


def ucb_query(X, Y, kappa=2.0, low_bounds=None, high_bounds=None,
              n_restarts=15, n_candidates=None, seed=42, y_shift=False):
    """UCB = mu + kappa * sigma"""
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


def grid_search_query(X, Y, best_known, step=0.006, n_restarts=15):
    """Deterministic grid centred on best_known, scored by GP mean only."""
    dim = X.shape[1]
    gp, scaler = fit_gp(X, Y, n_restarts=n_restarts)
    grid = []
    for deltas in product([-step, 0.0, step], repeat=dim):
        candidate = np.clip(best_known + np.array(deltas), 0.0, 1.0)
        grid.append(candidate)
    grid = np.unique(np.array(grid), axis=0)
    grid_sc = scaler.transform(grid)
    mu, _ = gp.predict(grid_sc, return_std=True)
    best_idx = np.argmax(mu)
    query = grid[best_idx]
    print(f"  Grid: {len(grid)} candidates | best mu: {mu[best_idx]:.6f}")
    return query, float(mu[best_idx]), float(mu[best_idx]), 0.0


def fine_grid_query(X, Y, centre, step=0.015, n_per_dim=5, n_restarts=10):
    """
    Wider, denser grid for functions where standard methods have repeatedly
    failed (e.g. F1). Builds an n_per_dim^d grid around centre with spacing
    step, scored by GP mean.
    """
    dim = X.shape[1]
    gp, scaler = fit_gp(X, Y, n_restarts=n_restarts)
    offsets = np.linspace(-(n_per_dim//2)*step, (n_per_dim//2)*step, n_per_dim)
    grid = []
    for deltas in product(offsets, repeat=dim):
        candidate = np.clip(centre + np.array(deltas), 0.0, 1.0)
        grid.append(candidate)
    grid = np.unique(np.array(grid), axis=0)
    grid_sc = scaler.transform(grid)
    mu, _ = gp.predict(grid_sc, return_std=True)
    best_idx = np.argmax(mu)
    query = grid[best_idx]
    print(f"  Fine grid: {len(grid)} candidates | best mu: {mu[best_idx]:.6f}")
    return query, float(mu[best_idx]), float(mu[best_idx]), 0.0


# ============================================================
# SECTION 7: PER-FUNCTION CONFIGURATION
# ============================================================

best_known = {
    "function_1": np.array([0.731024, 0.732999]),
    "function_2": np.array([0.732540, 0.896588]),   # W4 best-since-initial
    "function_3": np.array([0.578454, 0.529081, 0.427267]),  # W4 NEW BEST
    "function_4": np.array([0.403695, 0.397605, 0.413333, 0.411576]),  # W2 best (still unbeaten)
    "function_5": np.array([0.073441, 0.999981, 0.958412, 0.999276]),  # W4 NEW BEST
    "function_6": np.array([0.712186, 0.138693, 0.748552, 0.709997, 0.040401]),  # W4 NEW BEST
    "function_7": np.array([0.062595, 0.413172, 0.555018, 0.158477, 0.386036, 0.699222]),  # W4 NEW BEST
    "function_8": np.array([0.042700, 0.092462, 0.083390, 0.051299, 0.808162, 0.563756, 0.175217, 0.419904]),  # W1 best
}

# F5 dim 4 floor raised again: 0.95 -> 0.96
f5_low  = np.array([0.00, 0.94, 0.93, 0.96])
f5_high = np.array([0.12, 1.00, 1.00, 1.00])

method_config = {
    "function_1": {"method": "FINEGRID", "step": 0.015, "n_per_dim": 5, "restarts": 10},
    "function_2": {"method": "EI",   "xi": 0.003, "margin": 0.040, "restarts": 10, "y_shift": False},
    "function_3": {"method": "EI",   "xi": 0.002, "margin": 0.040, "restarts": 12, "y_shift": True},
    "function_4": {"method": "EI",   "xi": 0.003, "margin": 0.025, "restarts": 15, "y_shift": False},
    "function_5": {"method": "EI",   "xi": 0.005, "custom_bounds": True, "restarts": 15, "y_shift": False},
    "function_6": {"method": "GRID", "step": 0.004, "restarts": 15, "y_shift": True},
    "function_7": {"method": "EI",   "xi": 0.005, "margin": 0.050, "restarts": 20, "y_shift": False},
    "function_8": {"method": "UCB",  "kappa": 1.0, "margin": 0.050, "restarts": 20, "y_shift": False},
}

print("\nWEEK 5 METHOD CONFIGURATION")
print(f"{'Fn':<5} {'Method':<10} {'Setting':<14} {'Restarts'}")
print("-"*45)
for i in range(1, 9):
    key = f"function_{i}"
    cfg = method_config[key]
    m = cfg["method"]
    if m == "EI":
        setting = f"xi={cfg['xi']}, margin={cfg.get('margin','custom')}"
    elif m == "UCB":
        setting = f"k={cfg['kappa']}, margin={cfg['margin']}"
    elif m == "GRID":
        setting = f"step={cfg['step']}"
    else:
        setting = f"step={cfg['step']}, n={cfg['n_per_dim']}"
    print(f"F{i:<4} {m:<10} {setting:<14} {cfg['restarts']}")


# ============================================================
# SECTION 8: RUN OPTIMISATION — ONE LOOP, NO OVERRIDES NEEDED
# ============================================================

week5_results = {}

for i in range(1, 9):
    key = f"function_{i}"
    cfg = method_config[key]
    X   = updated_data[key]["X"]
    Y   = updated_data[key]["Y"]
    dim = X.shape[1]
    best = best_known[key]
    margin = cfg.get("margin")

    print(f"\n{'='*58}")
    print(f"FUNCTION {i} | {cfg['method']} | obs: {X.shape[0]} | best: {Y.max():.6f}")
    print(f"{'='*58}")

    low_b  = np.zeros(dim)
    high_b = np.ones(dim)
    if best is not None and margin is not None:
        low_b  = np.clip(best - margin, 0.0, 1.0)
        high_b = np.clip(best + margin, 0.0, 1.0)

    if cfg["method"] == "EI":
        if key == "function_5":
            low_b, high_b = f5_low, f5_high
        query, score, mu, sigma = ei_query(
            X, Y, xi=cfg["xi"], low_bounds=low_b, high_bounds=high_b,
            n_restarts=cfg["restarts"], y_shift=cfg["y_shift"]
        )
    elif cfg["method"] == "UCB":
        query, score, mu, sigma = ucb_query(
            X, Y, kappa=cfg["kappa"], low_bounds=low_b, high_bounds=high_b,
            n_restarts=cfg["restarts"], y_shift=cfg["y_shift"]
        )
    elif cfg["method"] == "GRID":
        query, score, mu, sigma = grid_search_query(
            X, Y, best_known=best, step=cfg["step"], n_restarts=cfg["restarts"]
        )
    elif cfg["method"] == "FINEGRID":
        query, score, mu, sigma = fine_grid_query(
            X, Y, centre=best, step=cfg["step"], n_per_dim=cfg["n_per_dim"],
            n_restarts=cfg["restarts"]
        )

    formatted = "-".join([f"{x:.6f}" for x in query])
    week5_results[key] = {
        "query": query, "formatted_query": formatted,
        "method": cfg["method"], "score": score,
        "predicted_mean": mu, "uncertainty": sigma,
    }

    dist = np.linalg.norm(query - best)
    print(f"Query    : {formatted}")
    print(f"Score    : {score:.6f} | Pred mean: {mu:.6f} | Dist from best: {dist:.4f}")


# ============================================================
# SECTION 9: VALIDATION
# ============================================================

print("\n" + "="*58)
print("QUERY VALIDATION REPORT")
print("="*58)

all_clear = True
for i in range(1, 9):
    key    = f"function_{i}"
    query  = week5_results[key]["query"]
    method = week5_results[key]["method"]
    issues = []

    if np.any(query < 0) or np.any(query > 1):
        issues.append("OUT OF RANGE")
    if np.all(query < 0.01):
        issues.append("SUSPICIOUS: all near 0")
    if np.all(query > 0.99):
        issues.append("SUSPICIOUS: all near 1")

    status = "OK" if not issues else "WARNING"
    print(f"\nFunction {i} [{status}] [{method}]")
    print(f"  Query : {week5_results[key]['formatted_query']}")
    if issues:
        all_clear = False
        for w in issues: print(f"  !! {w}")

print("\nAll queries valid." if all_clear else "\nReview warnings before submitting.")


# ============================================================
# SECTION 10: PROXIMITY CHECK
# ============================================================

print("\n" + "="*58)
print("PROXIMITY TO RUNNING BEST INPUT")
print("="*58)
for i in range(1, 9):
    key  = f"function_{i}"
    q    = week5_results[key]["query"]
    b    = best_known[key]
    d    = np.linalg.norm(q - b)
    m    = week5_results[key]["method"]
    flag = "ok" if d < 0.20 else "far — review"
    print(f"Function {i} [{m}] | distance: {d:.4f} [{flag}]")


# ============================================================
# SECTION 11: SAVE QUERIES
# ============================================================

rows = []
for i in range(1, 9):
    key = f"function_{i}"
    Y   = updated_data[key]["Y"]
    r   = week5_results[key]
    rows.append({
        "Function":       f"Function {i}",
        "Query":          r["formatted_query"],
        "Method":         r["method"],
        "Score":          round(r["score"], 6),
        "Predicted_mean": round(r["predicted_mean"], 6),
        "Uncertainty":    round(r["uncertainty"], 6),
        "Current_best":   round(float(Y.max()), 6),
    })

df = pd.DataFrame(rows)
print("\nFINAL WEEK 5 QUERIES")
print(df[["Function","Method","Query","Current_best"]].to_string(index=False))

df[["Function","Query"]].to_csv("week5_queries.csv", index=False)
df.to_csv("week5_queries_full.csv", index=False)
print("\nweek5_queries.csv saved.")
print("week5_queries_full.csv saved.")
```

---

## Notes before running

**Section 8 runs once only.** All bounds, methods, and the new FINEGRID
function for F1 are embedded directly. Do not re-run after validation.

**F1's new approach is a genuine strategy change.** Four EI attempts have
failed. The fine_grid_query function builds a 5x5 = 25-point grid spaced
0.015 apart around [0.731, 0.733], spanning a 0.06 x 0.06 box, scored purely
by GP mean. This is the most exhaustive local search attempted on F1 so far.

**F5's dim 4 floor moves from 0.95 to 0.96**, tightening further around the
confirmed productive zone after W4's recovery to a new all-time best.

**F8 is one tight query away from a new record.** Margin reduced to 0.05 and
kappa reduced to 1.0 for the most precise exploitation attempt yet around the
W1 best input.

---

*Week 5 strategy. Four new all-time bests achieved in Week 4 (F3, F5, F6, F7).
Strategy for Week 5 tightens exploitation across the board, with F1 receiving
a fundamentally different grid-based approach after repeated EI failures.*
