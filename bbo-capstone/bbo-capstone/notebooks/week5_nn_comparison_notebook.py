# ============================================================
# WEEK 5 — GP QUERIES + NEURAL NETWORK SURROGATE COMPARISON
# ============================================================
# GP remains the surrogate used to generate this week's queries
# (see Section 8 — unchanged in method from Week 4 strategy).
# NEW this week: a small MLP is trained per function purely as
# a comparison surrogate. Results feed into the model cards.
#
# Why GP stays primary: with 13-47 observations per function,
# Bayesian neural networks have been shown to underperform GPs
# due to underfitting on small data (Li et al. 2024, arXiv
# 2305.20028). Standard GPs remain competitive because of their
# strong priors and exact inference (same source). The NN is
# trained here for documentation and comparison, not to replace
# the GP-driven query strategy.
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from scipy.stats import norm
from scipy.stats.qmc import Sobol
from itertools import product

torch.manual_seed(42)
np.random.seed(42)


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
    print(f"Function {i} | obs: {X_updated.shape[0]} | best: {Y_updated[best_idx]:.6f}")


# ============================================================
# SECTION 4: GP FITTER AND ACQUISITION FUNCTIONS (UNCHANGED)
# ============================================================
# This is the same GP-based pipeline used in Weeks 1-4.
# It remains the method used to generate this week's actual
# queries in Section 8.

def fit_gp(X, Y, n_restarts=15, y_shift=False):
    Y_fit = Y - Y.min() if y_shift else Y
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e4))
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    )
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts, random_state=42)
    gp.fit(X_scaled, Y_fit)
    return gp, scaler


def ei_query(X, Y, xi=0.01, low_bounds=None, high_bounds=None,
             n_restarts=15, n_candidates=60000, seed=42, y_shift=False):
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
    return query, float(mu[best_idx]), float(mu[best_idx]), 0.0


def fine_grid_query(X, Y, centre, step=0.015, n_per_dim=5, n_restarts=10):
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
    return query, float(mu[best_idx]), float(mu[best_idx]), 0.0


# ============================================================
# SECTION 5: NEURAL NETWORK SURROGATE ARCHITECTURE
# ============================================================
# A single, small architecture used across all 8 functions.
# Kept deliberately compact given 13-47 observations per
# function. Width and depth scale lightly with input dimension
# to avoid the same network being too large for 2D functions
# and too small for 8D ones.

class MLPSurrogate(nn.Module):
    """
    Small feed-forward surrogate.
    Architecture: Input -> Linear(h1) -> ReLU -> Dropout
                        -> Linear(h2) -> ReLU -> Dropout
                        -> Linear(1)
    h1, h2 scale with input dimension: h1 = max(16, 4*dim), h2 = max(8, 2*dim)
    Dropout used as a (crude) uncertainty proxy via MC Dropout at inference.
    """
    def __init__(self, input_dim, dropout=0.15):
        super().__init__()
        h1 = max(16, 4 * input_dim)
        h2 = max(8, 2 * input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_nn_surrogate(X, Y, epochs=300, lr=0.01, dropout=0.15, weight_decay=1e-3, verbose=False):
    """
    Trains MLPSurrogate on (X, Y). Inputs are already in [0,1].
    Outputs are standardised (mean 0, std 1) before training,
    consistent with the ICLR 2024 BNN study's preprocessing
    recommendation (arXiv 2305.20028).
    """
    dim = X.shape[1]
    y_mean, y_std = Y.mean(), Y.std() if Y.std() > 0 else 1.0
    Y_scaled = (Y - y_mean) / y_std

    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y_scaled, dtype=torch.float32).reshape(-1, 1)

    model = MLPSurrogate(dim, dropout=dropout)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        model.train()
        optimiser.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, Y_t)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
        if verbose and epoch % 50 == 0:
            print(f"    epoch {epoch:4d} | loss: {loss.item():.4f}")

    return model, (y_mean, y_std), losses


def nn_predict_with_uncertainty(model, X, y_mean, y_std, n_mc=30):
    """
    MC Dropout: run forward pass n_mc times with dropout active
    to estimate predictive mean and std (proxy for GP-style
    uncertainty). Standard MC Dropout approach (Gal & Ghahramani 2016).
    """
    model.train()  # keep dropout active
    X_t = torch.tensor(X, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for _ in range(n_mc):
            preds.append(model(X_t).numpy().flatten())
    preds = np.array(preds)
    mu = preds.mean(axis=0) * y_std + y_mean
    sigma = preds.std(axis=0) * y_std
    return mu, sigma


# ============================================================
# SECTION 6: LEAVE-ONE-OUT COMPARISON — GP vs NN
# ============================================================
# For each function, perform leave-one-out cross-validation
# comparing GP and NN prediction error. This is the basis for
# each function's model card.

def loo_compare(X, Y, y_shift=False, nn_epochs=200):
    """
    Returns dict with GP and NN leave-one-out RMSE, plus the
    number of observations used (informs the model card).
    """
    loo = LeaveOneOut()
    gp_errs, nn_errs = [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # GP
        gp, scaler = fit_gp(X_train, Y_train, n_restarts=5, y_shift=y_shift)
        Y_fit_test = Y_test - Y_train.min() if y_shift else Y_test
        mu_gp, _ = gp.predict(scaler.transform(X_test), return_std=True)
        gp_pred = mu_gp + Y_train.min() if y_shift else mu_gp
        gp_errs.append((gp_pred[0] - Y_test[0]) ** 2)

        # NN
        model, (y_mean, y_std), _ = train_nn_surrogate(
            X_train, Y_train, epochs=nn_epochs, verbose=False
        )
        mu_nn, _ = nn_predict_with_uncertainty(model, X_test, y_mean, y_std, n_mc=10)
        nn_errs.append((mu_nn[0] - Y_test[0]) ** 2)

    return {
        "gp_rmse": float(np.sqrt(np.mean(gp_errs))),
        "nn_rmse": float(np.sqrt(np.mean(nn_errs))),
        "n_obs": X.shape[0],
        "dim": X.shape[1],
    }


print("\n" + "="*60)
print("LEAVE-ONE-OUT COMPARISON: GP vs NN SURROGATE")
print("="*60)

y_shift_map = {
    "function_1": False, "function_2": False, "function_3": True,
    "function_4": False, "function_5": False, "function_6": True,
    "function_7": False, "function_8": False,
}

comparison_results = {}
for i in range(1, 9):
    key = f"function_{i}"
    X = updated_data[key]["X"]
    Y = updated_data[key]["Y"]
    print(f"\nFunction {i} | {X.shape[0]} obs, {X.shape[1]}D ... running LOO-CV")
    result = loo_compare(X, Y, y_shift=y_shift_map[key], nn_epochs=150)
    comparison_results[key] = result
    winner = "GP" if result["gp_rmse"] < result["nn_rmse"] else "NN"
    print(f"  GP RMSE: {result['gp_rmse']:.4f} | NN RMSE: {result['nn_rmse']:.4f} | Better: {winner}")


# ============================================================
# SECTION 7: PER-FUNCTION CONFIGURATION (GP — unchanged method)
# ============================================================

best_known = {
    "function_1": np.array([0.731024, 0.732999]),
    "function_2": np.array([0.732540, 0.896588]),
    "function_3": np.array([0.578454, 0.529081, 0.427267]),
    "function_4": np.array([0.403695, 0.397605, 0.413333, 0.411576]),
    "function_5": np.array([0.073441, 0.999981, 0.958412, 0.999276]),
    "function_6": np.array([0.712186, 0.138693, 0.748552, 0.709997, 0.040401]),
    "function_7": np.array([0.062595, 0.413172, 0.555018, 0.158477, 0.386036, 0.699222]),
    "function_8": np.array([0.042700, 0.092462, 0.083390, 0.051299, 0.808162, 0.563756, 0.175217, 0.419904]),
}

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


# ============================================================
# SECTION 8: RUN GP-BASED OPTIMISATION — GENERATES THIS WEEK'S QUERIES
# ============================================================
# This section is unchanged in method from previous weeks.
# The NN trained above is NOT used here — it is a comparison
# tool only, documented in the model cards (Section 9).

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
    print(f"Query    : {formatted}")
    print(f"Score    : {score:.6f} | Pred mean: {mu:.6f}")


# ============================================================
# SECTION 9: MODEL CARD DATA EXPORT
# ============================================================
# Compiles per-function model card data: architecture summary,
# parameter count, LOO comparison, and recommendation.

model_cards = {}
for i in range(1, 9):
    key = f"function_{i}"
    dim = updated_data[key]["X"].shape[1]
    h1 = max(16, 4 * dim)
    h2 = max(8, 2 * dim)
    n_params = (dim * h1 + h1) + (h1 * h2 + h2) + (h2 * 1 + 1)
    comp = comparison_results[key]

    model_cards[key] = {
        "function": f"F{i}",
        "dimensions": dim,
        "n_observations": comp["n_obs"],
        "gp_kernel": "ConstantKernel x RBF + WhiteKernel",
        "nn_architecture": f"Input({dim}) -> Linear({h1}) -> ReLU -> Dropout(0.15) -> Linear({h2}) -> ReLU -> Dropout(0.15) -> Linear(1)",
        "nn_param_count": n_params,
        "gp_loo_rmse": round(comp["gp_rmse"], 6),
        "nn_loo_rmse": round(comp["nn_rmse"], 6),
        "recommended_surrogate": "GP" if comp["gp_rmse"] <= comp["nn_rmse"] else "NN",
        "method_used_this_week": method_config[key]["method"],
    }

print("\n" + "="*60)
print("MODEL CARD SUMMARY")
print("="*60)
for key, card in model_cards.items():
    print(f"\n{card['function']} | dim={card['dimensions']} | obs={card['n_observations']} | "
          f"params={card['nn_param_count']}")
    print(f"  GP LOO-RMSE: {card['gp_loo_rmse']} | NN LOO-RMSE: {card['nn_loo_rmse']} | "
          f"Recommended: {card['recommended_surrogate']}")

cards_df = pd.DataFrame(model_cards).T
cards_df.to_csv("model_cards_week5.csv")
print("\nmodel_cards_week5.csv saved.")


# ============================================================
# SECTION 10: VALIDATION AND SAVE QUERIES (unchanged)
# ============================================================

print("\n" + "="*58)
print("QUERY VALIDATION REPORT")
print("="*58)
all_clear = True
for i in range(1, 9):
    key = f"function_{i}"
    query = week5_results[key]["query"]
    issues = []
    if np.any(query < 0) or np.any(query > 1):
        issues.append("OUT OF RANGE")
    status = "OK" if not issues else "WARNING"
    print(f"Function {i} [{status}] Query: {week5_results[key]['formatted_query']}")
    if issues:
        all_clear = False

rows = []
for i in range(1, 9):
    key = f"function_{i}"
    r = week5_results[key]
    rows.append({"Function": f"Function {i}", "Query": r["formatted_query"], "Method": r["method"]})
pd.DataFrame(rows).to_csv("week5_queries.csv", index=False)
print("\nweek5_queries.csv saved.")
print("All queries valid." if all_clear else "Review warnings.")
