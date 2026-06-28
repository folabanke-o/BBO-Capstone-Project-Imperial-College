# ============================================================
# WEEK 4 BAYESIAN OPTIMISATION — FULL NOTEBOOK
# ============================================================
# Methods: EI for F1-F5, F7 | Grid for F6 | UCB for F8
# Research: xi=0.01 recommended for exploitation (arxiv 2211.09504)
#           EI more exploitative than UCB at equivalent scores
#           (arxiv 1911.12809)
# W3 confirmed gains: F3 (first gain), F6 (first gain), F7 (new best)
# W3 declines to fix: F5 dim4 dropped 0.960->0.924, F8 broad explore failed
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
# SECTION 2: COMPLETE QUERY HISTORY — ALL 3 WEEKS
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
    "function_1": 1.13739520e-239,
    "function_2": -0.10970434,
    "function_3": -0.36576718,
    "function_4": -0.86900220,
    "function_5": 3019.65984,
    "function_6": -1.21331906,
    "function_7": 1.77196960,
    "function_8": 9.96529345,
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
    "function_1": -4.67691389e-32,
    "function_2": 0.14503569,
    "function_3": -0.11944713,
    "function_4": 0.53385778,
    "function_5": 3511.61191,
    "function_6": -0.80060002,
    "function_7": 1.28884742,
    "function_8": 9.81687307,
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
    "function_1": 5.07920747e-29,
    "function_2": 0.09168177,
    "function_3": -0.02142785,
    "function_4": 0.23137215,
    "function_5": 3214.79278,
    "function_6": -0.70363809,
    "function_7": 1.91924944,
    "function_8": 9.61721741,
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
    print(f"Function {i} | obs: {X_updated.shape[0]} | best: {Y_updated[best_idx]:.6f} "
          f"at {np.round(X_updated[best_idx], 4)}")


# ============================================================
# SECTION 4: EDA — STATUS ENTERING WEEK 4
# ============================================================

print("\n" + "="*65)
print("ENTERING WEEK 4 — RUNNING BESTS")
print("="*65)

all_bests = {
    "function_1": {"val": 7.71e-16,    "inp": np.array([0.731024, 0.732999]),                               "src": "initial"},
    "function_2": {"val": 0.61120522,  "inp": np.array([0.702637, 0.926564]),                               "src": "initial"},
    "function_3": {"val": -0.02142785, "inp": np.array([0.538501, 0.565224, 0.389093]),                     "src": "W3"},
    "function_4": {"val": 0.53385778,  "inp": np.array([0.403695, 0.397605, 0.413333, 0.411576]),           "src": "W2"},
    "function_5": {"val": 3511.61191,  "inp": np.array([0.056181, 0.992607, 0.973199, 0.959866]),           "src": "W2"},
    "function_6": {"val": -0.70363809, "inp": np.array([0.718186, 0.144693, 0.742552, 0.703997, 0.046401]),"src": "W3"},
    "function_7": {"val": 1.91924944,  "inp": np.array([0.121105, 0.376149, 0.487901, 0.138159, 0.427197, 0.725669]),"src": "W3"},
    "function_8": {"val": 9.96529345,  "inp": np.array([0.042700, 0.092462, 0.083390, 0.051299, 0.808162, 0.563756, 0.175217, 0.419904]),"src": "W1"},
}

for i in range(1, 9):
    key = f"function_{i}"
    b = all_bests[key]
    print(f"F{i} | best={b['val']:.6f} (from {b['src']}) | {np.round(b['inp'], 4)}")


# ============================================================
# SECTION 5: SHARED GP FITTER
# ============================================================

def fit_gp(X, Y, n_restarts=15, y_shift=False):
    """
    Fit GP surrogate on (X, Y).
    y_shift=True shifts Y so minimum=0 — improves fitting for
    functions with all-negative outputs (F3, F6).
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
             n_restarts=15, n_candidates=60000, seed=42, y_shift=False):
    """
    Expected Improvement (EI).
    EI = E[max(0, f(x) - f_best - xi)]
    xi=0.01 is recommended for exploitation (arxiv 2211.09504).
    EI is more exploitative than UCB at equivalent scores (arxiv 1911.12809).
    Best suited for: confirmed peak regions with a meaningful f_best reference.
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
    ei_vals = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei_vals[sigma < 1e-9] = 0.0

    best_idx = np.argmax(ei_vals)
    query = np.clip(cands[best_idx], 0.0, 1.0)
    return query, float(ei_vals[best_idx]), float(mu[best_idx]), float(sigma[best_idx])


def ucb_query(X, Y, kappa=2.0, low_bounds=None, high_bounds=None,
              n_restarts=15, n_candidates=None, seed=42, y_shift=False):
    """
    Upper Confidence Bound (UCB).
    UCB = mu + kappa * sigma.
    Used for F8 where EI's hard floor on f_best is less appropriate
    and controlled exploitation of a known region is needed.
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


def grid_search_query(X, Y, best_known, step=0.006, n_restarts=15):
    """
    Deterministic grid search centred on best_known.
    3^d candidates scored by GP mean only (no uncertainty).
    Used for F6: GP surrogate unreliable but optimum tightly localised.
    Step reduced from 0.008 to 0.006 for finer W4 resolution.
    """
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


# ============================================================
# SECTION 7: PER-FUNCTION CONFIGURATION
# ============================================================

# Running best inputs — all confirmed from real outputs
best_known = {
    "function_1": np.array([0.731024, 0.732999]),
    "function_2": np.array([0.702637, 0.926564]),
    "function_3": np.array([0.538501, 0.565224, 0.389093]),      # W3 first gain
    "function_4": np.array([0.403695, 0.397605, 0.413333, 0.411576]),  # W2 best
    "function_5": np.array([0.056181, 0.992607, 0.973199, 0.959866]),  # W2 best
    "function_6": np.array([0.718186, 0.144693, 0.742552, 0.703997, 0.046401]),  # W3 first gain
    "function_7": np.array([0.121105, 0.376149, 0.487901, 0.138159, 0.427197, 0.725669]),  # W3 new best
    "function_8": np.array([0.042700, 0.092462, 0.083390, 0.051299, 0.808162, 0.563756, 0.175217, 0.419904]),  # W1 best
}

# F5 asymmetric bounds — dim 4 floor raised to 0.95
# Root cause of W3 decline: dim 4 dropped from 0.960 (W2) to 0.924 (W3)
f5_low  = np.array([0.00, 0.93, 0.93, 0.95])
f5_high = np.array([0.10, 1.00, 1.00, 1.00])

method_config = {
    "function_1": {"method": "EI",   "xi": 0.001, "margin": 0.025, "restarts": 10, "y_shift": False},
    "function_2": {"method": "EI",   "xi": 0.005, "margin": 0.060, "restarts": 10, "y_shift": False},
    "function_3": {"method": "EI",   "xi": 0.005, "margin": 0.060, "restarts": 12, "y_shift": True},
    "function_4": {"method": "EI",   "xi": 0.005, "margin": 0.080, "restarts": 15, "y_shift": False},
    "function_5": {"method": "EI",   "xi": 0.005, "custom_bounds": True, "restarts": 15, "y_shift": False},
    "function_6": {"method": "GRID", "step": 0.006, "restarts": 15, "y_shift": True},
    "function_7": {"method": "EI",   "xi": 0.010, "margin": 0.080, "restarts": 20, "y_shift": False},
    "function_8": {"method": "UCB",  "kappa": 1.5, "margin": 0.120, "restarts": 20, "y_shift": False},
}

print("\nWEEK 4 METHOD CONFIGURATION")
print(f"{'Fn':<5} {'Method':<8} {'Setting':<12} {'y_shift':<9} {'Restarts'}")
print("-"*45)
for i in range(1, 9):
    key = f"function_{i}"
    cfg = method_config[key]
    m   = cfg["method"]
    setting = f"xi={cfg['xi']}" if m == "EI" else (f"k={cfg['kappa']}" if m == "UCB" else f"step={cfg['step']}")
    print(f"F{i:<4} {m:<8} {setting:<12} {str(cfg['y_shift']):<9} {cfg['restarts']}")


# ============================================================
# SECTION 8: RUN OPTIMISATION — ONE LOOP, NO OVERRIDES NEEDED
# ============================================================

week4_results = {}

for i in range(1, 9):
    key = f"function_{i}"
    cfg = method_config[key]
    X   = updated_data[key]["X"]
    Y   = updated_data[key]["Y"]
    dim = X.shape[1]
    best   = best_known[key]
    margin = cfg.get("margin")

    print(f"\n{'='*58}")
    print(f"FUNCTION {i} | {cfg['method']} | obs: {X.shape[0]} | best: {Y.max():.6f}")
    print(f"{'='*58}")

    # Build bounds from margin
    low_b  = np.zeros(dim)
    high_b = np.ones(dim)
    if best is not None and margin is not None:
        low_b  = np.clip(best - margin, 0.0, 1.0)
        high_b = np.clip(best + margin, 0.0, 1.0)

    if cfg["method"] == "EI":
        if key == "function_5":
            low_b, high_b = f5_low, f5_high
        query, score, mu, sigma = ei_query(
            X, Y,
            xi=cfg["xi"],
            low_bounds=low_b, high_bounds=high_b,
            n_restarts=cfg["restarts"],
            y_shift=cfg["y_shift"]
        )
    elif cfg["method"] == "UCB":
        query, score, mu, sigma = ucb_query(
            X, Y,
            kappa=cfg["kappa"],
            low_bounds=low_b, high_bounds=high_b,
            n_restarts=cfg["restarts"],
            y_shift=cfg["y_shift"]
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
    query  = week4_results[key]["query"]
    method = week4_results[key]["method"]
    issues = []

    if np.any(query < 0) or np.any(query > 1):
        issues.append("OUT OF RANGE")
    if np.all(query < 0.01):
        issues.append("SUSPICIOUS: all near 0")
    if np.all(query > 0.99):
        issues.append("SUSPICIOUS: all near 1")

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

print("\n" + "="*58)
print("PROXIMITY TO RUNNING BEST INPUT")
print("="*58)
for i in range(1, 9):
    key  = f"function_{i}"
    q    = week4_results[key]["query"]
    b    = best_known[key]
    d    = np.linalg.norm(q - b)
    m    = week4_results[key]["method"]
    flag = "ok" if d < 0.20 else "far — review"
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
        "Function":       f"Function {i}",
        "Query":          r["formatted_query"],
        "Method":         r["method"],
        "Score":          round(r["score"], 6),
        "Predicted_mean": round(r["predicted_mean"], 6),
        "Uncertainty":    round(r["uncertainty"], 6),
        "Current_best":   round(float(Y.max()), 6),
    })

df = pd.DataFrame(rows)
print("\nFINAL WEEK 4 QUERIES")
print(df[["Function","Method","Query","Current_best"]].to_string(index=False))

df[["Function","Query"]].to_csv("week4_queries.csv", index=False)
df.to_csv("week4_queries_full.csv", index=False)
print("\nweek4_queries.csv saved.")
print("week4_queries_full.csv saved.")
