# ============================================================
# WEEK 6 BAYESIAN OPTIMISATION — NEURAL NETWORK SURROGATES
# ============================================================
#
# WEEK 5 POST-MORTEM (informs every W6 decision):
#   Only F3 and F8 improved — the worst round of the project.
#   Root causes identified from the data:
#   - F2, F4, F7: search boxes drifted from the confirmed
#     all-time best input, wasting the query on worse terrain.
#   - F5: dim3 dropped to 0.935 in the W5 query (was 0.958
#     in the W4 best input); that single change cost ~285 pts.
#   - F6: grid step 0.004 moved to a cell worse than W4.
#   - F1: fundamental problem — function near-zero everywhere
#     within the explored region; may need a broader search.
#
# WEEK 6 STRATEGY:
#   Three surrogates are trained per function. LOO-CV RMSE
#   determines which one generates the actual query. The
#   search box for each function is derived precisely from the
#   all-time best input, not the most recent query.
#
# NEURAL NETWORK ARCHITECTURES:
#   Architecture 1 — Vanilla GP (sklearn baseline, all weeks)
#   Architecture 2 — MC Dropout MLP (Gal & Ghahramani 2016)
#     Dropout is kept ACTIVE at inference (model.train() mode)
#     to approximate posterior uncertainty via T=50 forward
#     passes. This is the recommended practical BNN approach
#     at small data scales (BioBO 2024; arXiv 2306.01095).
#     Architecture: Input(d) -> Linear(h1) -> ReLU -> Drop(p)
#                           -> Linear(h2) -> ReLU -> Drop(p)
#                           -> Linear(1)
#     h1=max(32,8d), h2=max(16,4d); p=0.1; Adam lr=0.001
#   Architecture 3 — Deep Kernel Learning (Wilson et al. 2016)
#     A neural feature extractor maps d-dim inputs to a small
#     latent space, on which an exact GP RBF kernel operates.
#     NN weights and GP hyperparameters are trained jointly by
#     maximising the GP marginal log likelihood via GPyTorch.
#     Architecture: Input(d)->Linear(h)->ReLU->Linear(lat)
#     Latent dim=2; h=max(16,4d); Adam lr=0.01, 200 epochs.
#
# REFERENCES:
#   Gal & Ghahramani (2016) arXiv:1506.02142 — MC Dropout
#   Wilson et al. (2016) ICML — Deep Kernel Learning
#   Li et al. ICLR 2024 arXiv:2305.20028 — BNN surrogate study
#   GPyTorch DKL docs: docs.gpytorch.ai (2024)
#   BioBO (2024) arXiv:2509.19988 — MC Dropout in BBO
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gpytorch
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
# SECTION 2: COMPLETE QUERY HISTORY — WEEKS 1–5
# ============================================================

week1_queries = {
    "function_1": np.array([0.000186, 0.014353]),
    "function_2": np.array([0.998531, 0.007036]),
    "function_3": np.array([0.933672, 0.002452, 0.965412]),
    "function_4": np.array([0.417336, 0.402860, 0.336077, 0.476656]),
    "function_5": np.array([0.050115, 0.927701, 0.965034, 0.985561]),
    "function_6": np.array([0.197786, 0.010925, 0.990284, 0.888004, 0.052863]),
    "function_7": np.array([0.110110, 0.393658, 0.394356, 0.092883, 0.385807, 0.669789]),
    "function_8": np.array([0.042700, 0.092462, 0.083390, 0.051299,
                             0.808162, 0.563756, 0.175217, 0.419904]),
}
week1_outputs = {
    "function_1": 1.13739520e-239, "function_2": -0.10970434,
    "function_3": -0.36576718,     "function_4": -0.86900220,
    "function_5": 3019.65984,      "function_6": -1.21331906,
    "function_7": 1.77196960,      "function_8": 9.96529345,
}
week2_queries = {
    "function_1": np.array([0.705908, 0.823143]),
    "function_2": np.array([0.839078, 0.895770]),
    "function_3": np.array([0.363504, 0.758379, 0.191085]),
    "function_4": np.array([0.403695, 0.397605, 0.413333, 0.411576]),
    "function_5": np.array([0.056181, 0.992607, 0.973199, 0.959866]),
    "function_6": np.array([0.729856, 0.157383, 0.734992, 0.704798, 0.068448]),
    "function_7": np.array([0.058090, 0.303889, 0.327991, 0.001268, 0.275231, 0.673181]),
    "function_8": np.array([0.017074, 0.091604, 0.305973, 0.115845,
                             0.946320, 0.608139, 0.053440, 0.855712]),
}
week2_outputs = {
    "function_1": -4.67691389e-32, "function_2":  0.14503569,
    "function_3": -0.11944713,     "function_4":  0.53385778,
    "function_5":  3511.61191,     "function_6": -0.80060002,
    "function_7":  1.28884742,     "function_8":  9.81687307,
}
week3_queries = {
    "function_1": np.array([0.770991, 0.772247]),
    "function_2": np.array([0.782573, 0.851721]),
    "function_3": np.array([0.538501, 0.565224, 0.389093]),
    "function_4": np.array([0.391220, 0.397272, 0.392512, 0.416849]),
    "function_5": np.array([0.043103, 0.985149, 0.986449, 0.924277]),
    "function_6": np.array([0.718186, 0.144693, 0.742552, 0.703997, 0.046401]),
    "function_7": np.array([0.121105, 0.376149, 0.487901, 0.138159, 0.427197, 0.725669]),
    "function_8": np.array([0.311134, 0.116341, 0.059376, 0.140269,
                             0.540842, 0.044932, 0.335219, 0.555288]),
}
week3_outputs = {
    "function_1":  5.07920747e-29, "function_2":  0.09168177,
    "function_3": -0.02142785,     "function_4":  0.23137215,
    "function_5":  3214.79278,     "function_6": -0.70363809,
    "function_7":  1.91924944,     "function_8":  9.61721741,
}
week4_queries = {
    "function_1": np.array([0.755943, 0.708019]),
    "function_2": np.array([0.732540, 0.896588]),
    "function_3": np.array([0.578454, 0.529081, 0.427267]),
    "function_4": np.array([0.391082, 0.388623, 0.421155, 0.426886]),
    "function_5": np.array([0.073441, 0.999981, 0.958412, 0.999276]),
    "function_6": np.array([0.712186, 0.138693, 0.748552, 0.709997, 0.040401]),
    "function_7": np.array([0.062595, 0.413172, 0.555018, 0.158477, 0.386036, 0.699222]),
    "function_8": np.array([0.007140, 0.131981, 0.126210, 0.111602,
                             0.760069, 0.618063, 0.244768, 0.488294]),
}
week4_outputs = {
    "function_1": -1.60596561e-16, "function_2":  0.33247907,
    "function_3": -0.00608242,     "function_4":  0.45070948,
    "function_5":  3905.150047,    "function_6": -0.68015232,
    "function_7":  2.01886280,     "function_8":  9.96091957,
}
week5_queries = {
    "function_1": np.array([0.746024, 0.747999]),
    "function_2": np.array([0.772411, 0.856620]),
    "function_3": np.array([0.618443, 0.512242, 0.466082]),
    "function_4": np.array([0.378726, 0.395461, 0.438325, 0.427135]),
    "function_5": np.array([0.115771, 0.997836, 0.934529, 0.999358]),
    "function_6": np.array([0.708186, 0.134693, 0.744552, 0.705997, 0.036401]),
    "function_7": np.array([0.022545, 0.447401, 0.567314, 0.109668, 0.425415, 0.724518]),
    "function_8": np.array([0.077857, 0.141235, 0.128950, 0.097085,
                             0.760277, 0.523742, 0.158135, 0.465728]),
}
week5_outputs = {
    "function_1": -5.34109637e-23, "function_2":  0.20291815,
    "function_3": -0.00319822,     "function_4":  0.29101365,
    "function_5":  3620.40360,     "function_6": -0.70486982,
    "function_7":  1.46989852,     "function_8":  9.98947835,
}


# ============================================================
# SECTION 3: BUILD DATASETS (original + W1–W5)
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
        week5_queries[key].reshape(1, -1),
    ])
    Y_updated = np.append(
        original_data[key]["Y"],
        [week1_outputs[key], week2_outputs[key], week3_outputs[key],
         week4_outputs[key], week5_outputs[key]]
    )
    updated_data[key] = {"X": X_updated, "Y": Y_updated}
    best_idx = np.argmax(Y_updated)
    print(f"F{i} | obs: {X_updated.shape[0]} | best: {Y_updated[best_idx]:.6f} "
          f"at {np.round(X_updated[best_idx], 4)}")


# ============================================================
# SECTION 4: EDA — STATUS ENTERING WEEK 6
# ============================================================
# All-time bests: use the CONFIRMED best input, not the most
# recent query. This is the core fix from W5 post-mortem.
# ============================================================

print("\n" + "=" * 65)
print("ENTERING WEEK 6 — ALL-TIME BESTS (CONFIRMED)")
print("=" * 65)

# Each entry uses the input that produced the all-time best output.
# F1, F2: initial data still holds.  F3, F8: W5 new best.
# F4, F5, F6, F7: W2/W4 best — do NOT use the W5 query inputs.
all_bests = {
    "function_1": {
        "val": 7.71e-16,
        "inp": np.array([0.731024, 0.732999]),
        "src": "Initial",
        "note": "Never improved — initial data holds",
    },
    "function_2": {
        "val": 0.61120522,
        "inp": np.array([0.702637, 0.926564]),
        "src": "Initial",
        "note": "W4 was 0.332, W5 was 0.203; initial still best",
    },
    "function_3": {
        "val": -0.00319822,
        "inp": np.array([0.618443, 0.512242, 0.466082]),
        "src": "W5",
        "note": "Three consecutive gains; keep converging",
    },
    "function_4": {
        "val": 0.53385778,
        "inp": np.array([0.403695, 0.397605, 0.413333, 0.411576]),
        "src": "W2",
        "note": "W2 best unbeaten for 3 weeks; W5 moved too far",
    },
    "function_5": {
        "val": 3905.150047,
        "inp": np.array([0.073441, 0.999981, 0.958412, 0.999276]),
        "src": "W4",
        "note": "W5 dropped dim3 to 0.935 — floor must stay at 0.95+",
    },
    "function_6": {
        "val": -0.68015232,
        "inp": np.array([0.712186, 0.138693, 0.748552, 0.709997, 0.040401]),
        "src": "W4",
        "note": "W5 grid step moved to worse cell; return to W4 coords",
    },
    "function_7": {
        "val": 2.01886280,
        "inp": np.array([0.062595, 0.413172, 0.555018, 0.158477,
                          0.386036, 0.699222]),
        "src": "W4",
        "note": "W5 EI drifted far; return to W4 best as anchor",
    },
    "function_8": {
        "val": 9.98947835,
        "inp": np.array([0.077857, 0.141235, 0.128950, 0.097085,
                          0.760277, 0.523742, 0.158135, 0.465728]),
        "src": "W5",
        "note": "New all-time best; exploit this region",
    },
}

for i in range(1, 9):
    key = f"function_{i}"
    b = all_bests[key]
    print(f"F{i} | {b['val']:.6g} ({b['src']}) | {b['note']}")


# ============================================================
# SECTION 5: ARCHITECTURE 1 — VANILLA GP BASELINE
# ============================================================
# ConstantKernel × RBF + WhiteKernel, sklearn.
# Unchanged across all prior weeks. Reference baseline.
# y_shift=True shifts Y by its minimum before fitting —
# improves numerical stability for all-negative functions.
# ============================================================

def fit_gp(X, Y, n_restarts=15, y_shift=False):
    """Fit the standard GP surrogate used in Weeks 1–5."""
    Y_fit = Y - Y.min() if y_shift else Y
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e4))
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=n_restarts, random_state=42
    )
    gp.fit(X_sc, Y_fit)
    return gp, scaler


# ============================================================
# SECTION 6: ARCHITECTURE 2 — MC DROPOUT MLP
# ============================================================
# MC Dropout (Gal & Ghahramani, 2016): dropout remains ACTIVE
# at inference time (model.train() mode), making each forward
# pass use a different sub-network. The variance across T
# passes is the epistemic uncertainty estimate.
#
# Hyperparameters (grounded in BioBO 2024, arXiv 2509.19988):
#   - Two hidden layers: h1=max(32,8d), h2=max(16,4d)
#   - Dropout p=0.1 (conservative — less aggressive than 0.5,
#     appropriate for very small n where variance is already
#     high; 0.5 can cause instability with <20 observations)
#   - Adam lr=0.001, weight_decay=1e-3, full-batch
#   - Output standardised (mean 0, std 1) before training
#   - T=50 MC passes at inference
# ============================================================

class MCDropoutMLP(nn.Module):
    """
    Two-hidden-layer MLP with MC Dropout for uncertainty.
    h1 and h2 scale with input dimensionality d to keep the
    parameter-to-observation ratio reasonable at small n.
    """
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.h1 = max(32, 8 * input_dim)
        self.h2 = max(16, 4 * input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.h1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.h1, self.h2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.h2, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_mc_dropout(X, Y, epochs=300, lr=0.001, dropout=0.1,
                     weight_decay=1e-3, verbose=False):
    """
    Train MCDropoutMLP. Standardises Y before fitting.
    Returns model and (y_mean, y_std) for inverse transform.
    """
    y_mean = float(Y.mean())
    y_std  = float(Y.std()) if Y.std() > 1e-12 else 1.0
    Y_s    = (Y - y_mean) / y_std

    X_t = torch.tensor(X,   dtype=torch.float32)
    Y_t = torch.tensor(Y_s, dtype=torch.float32).reshape(-1, 1)

    model   = MCDropoutMLP(X.shape[1], dropout=dropout)
    opt     = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_t), Y_t)
        loss.backward()
        opt.step()
        if verbose and epoch % 100 == 0:
            print(f"    [MC-MLP] epoch {epoch:4d} | loss {loss.item():.5f}")

    return model, (y_mean, y_std)


def mc_predict(model, X_np, y_mean, y_std, n_mc=50):
    """
    MC Dropout inference.
    model.train() keeps dropout active during forward passes.
    Returns predictive mean and std in original Y scale.
    Implements equations 12–13 from arXiv 2306.01095:
      mu_MC  = (1/T) sum_t f_t(x)
      var_MC = (1/T) sum_t f_t(x)^2 - mu_MC^2
    """
    model.train()
    X_t = torch.tensor(X_np, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for _ in range(n_mc):
            preds.append(model(X_t).numpy().flatten())
    preds  = np.array(preds)                  # (n_mc, n_candidates)
    mu     = preds.mean(axis=0) * y_std + y_mean
    sigma  = preds.std(axis=0)  * y_std
    return mu, np.maximum(sigma, 1e-9)


# ============================================================
# SECTION 7: ARCHITECTURE 3 — DEEP KERNEL LEARNING (DKL)
# ============================================================
# Deep Kernel Learning (Wilson et al. 2016, ICML):
# A small neural feature extractor maps d-dimensional inputs
# to a 2D latent space. An exact GP RBF kernel operates on
# that latent space. NN weights and GP hyperparameters are
# trained jointly by maximising the GP marginal log likelihood.
#
# Why 2D latent space: keeps the number of learnable NN
# parameters low (critical when n < 50), while still allowing
# the NN to learn a non-stationary input transformation.
# GPyTorch handles the end-to-end marginal likelihood training.
#
# Hyperparameters (GPyTorch official DKL docs, 2024):
#   - Feature extractor: Input(d)->Linear(h)->ReLU->Linear(2)
#     h = max(16, 4d)
#   - GP: RBF kernel on 2D latent, GaussianLikelihood
#   - Adam lr=0.01, 200 epochs
# ============================================================

class DKLFeatureExtractor(nn.Module):
    """
    Lightweight NN feature extractor for DKL.
    Maps d-dim input to 2D latent space for the GP kernel.
    Architecture: Input(d) -> Linear(h) -> ReLU -> Linear(2)
    h = max(16, 4*d) — kept small to avoid overfitting at n<50.
    """
    def __init__(self, input_dim):
        super().__init__()
        h = max(16, 4 * input_dim)
        self.h = h
        self.net = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.ReLU(),
            nn.Linear(h, 2),
        )

    def forward(self, x):
        return self.net(x)


class DKLGPModel(gpytorch.models.ExactGP):
    """
    Exact GP model with deep kernel.
    The covariance kernel operates on the 2D feature space
    produced by DKLFeatureExtractor, not the raw inputs.
    Both NN weights and GP hyperparameters are learnt jointly.
    """
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        z = self.feature_extractor(x)          # map to 2D latent
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(z),
            self.covar_module(z)
        )


def train_dkl(X, Y, epochs=200, lr=0.01, verbose=False):
    """
    Train DKL model on (X, Y). Standardises Y before fitting.
    Optimises NN weights + GP hyperparameters jointly via
    negative marginal log likelihood (Type-II MLE).
    Returns model, likelihood, (y_mean, y_std), scaler.
    """
    y_mean = float(Y.mean())
    y_std  = float(Y.std()) if Y.std() > 1e-12 else 1.0
    Y_s    = (Y - y_mean) / y_std

    scaler  = StandardScaler()
    X_s     = scaler.fit_transform(X)
    train_x = torch.tensor(X_s, dtype=torch.float32)
    train_y = torch.tensor(Y_s, dtype=torch.float32)

    feat_ext   = DKLFeatureExtractor(X.shape[1])
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model      = DKLGPModel(train_x, train_y, likelihood, feat_ext)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(
        list(model.feature_extractor.parameters())
        + list(model.covar_module.parameters())
        + list(model.mean_module.parameters())
        + list(likelihood.parameters()),
        lr=lr
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        if verbose and epoch % 50 == 0:
            print(f"    [DKL] epoch {epoch:4d} | -MLL {loss.item():.5f}")

    return model, likelihood, (y_mean, y_std), scaler


def dkl_predict(model, likelihood, X_np, y_mean, y_std, scaler):
    """
    DKL prediction using exact GP posterior on the latent space.
    Returns mean and std in original Y scale.
    """
    model.eval()
    likelihood.eval()
    X_t = torch.tensor(scaler.transform(X_np), dtype=torch.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(X_t))
    mu    = pred.mean.numpy()   * y_std + y_mean
    sigma = pred.stddev.numpy() * y_std
    return mu, np.maximum(sigma, 1e-9)


# ============================================================
# SECTION 8: ACQUISITION FUNCTIONS (SURROGATE-AGNOSTIC)
# ============================================================

def expected_improvement(mu, sigma, f_best, xi=0.01):
    """
    Closed-form EI: E[max(0, f(x) - f_best - xi)]
    Compatible with any surrogate that returns (mu, sigma).
    xi = exploration–exploitation trade-off parameter.
    Lower xi = more exploitation (closer to f_best).
    """
    sigma = np.maximum(sigma, 1e-9)
    Z  = (mu - f_best - xi) / sigma
    ei = (mu - f_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei


def upper_confidence_bound(mu, sigma, kappa=1.5):
    """UCB = mu + kappa * sigma."""
    return mu + kappa * sigma


# ============================================================
# SECTION 9: LEAVE-ONE-OUT SURROGATE COMPARISON
# ============================================================
# For each function, all three surrogates are evaluated by
# leave-one-out CV. The surrogate with the lowest RMSE is
# selected to generate the actual Week 6 query.
# ============================================================

def loo_rmse_all(X, Y, y_shift=False):
    """
    LOO-CV RMSE for GP, MC Dropout, and DKL.
    y_shift passed to GP only (not needed for NN surrogates
    which standardise Y internally).
    """
    loo = LeaveOneOut()
    gp_sq, mc_sq, dkl_sq = [], [], []

    for train_idx, test_idx in loo.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        Ytr, Yte = Y[train_idx], Y[test_idx]

        # — GP —
        gp, sc = fit_gp(Xtr, Ytr, n_restarts=5, y_shift=y_shift)
        mu_g, _ = gp.predict(sc.transform(Xte), return_std=True)
        pred_g   = float(mu_g[0]) + (Ytr.min() if y_shift else 0)
        gp_sq.append((pred_g - Yte[0]) ** 2)

        # — MC Dropout —
        try:
            m_mc, (ym, ys) = train_mc_dropout(Xtr, Ytr, epochs=200)
            mu_mc, _ = mc_predict(m_mc, Xte, ym, ys, n_mc=20)
            mc_sq.append((float(mu_mc[0]) - Yte[0]) ** 2)
        except Exception:
            mc_sq.append((float(np.mean(Ytr)) - Yte[0]) ** 2)

        # — DKL —
        try:
            m_dkl, lik, (ym2, ys2), sc2 = train_dkl(Xtr, Ytr, epochs=80)
            mu_dkl, _ = dkl_predict(m_dkl, lik, Xte, ym2, ys2, sc2)
            dkl_sq.append((float(mu_dkl[0]) - Yte[0]) ** 2)
        except Exception:
            dkl_sq.append((float(np.mean(Ytr)) - Yte[0]) ** 2)

    return {
        "GP":         float(np.sqrt(np.mean(gp_sq))),
        "MC_Dropout": float(np.sqrt(np.mean(mc_sq))),
        "DKL":        float(np.sqrt(np.mean(dkl_sq))),
    }


# ============================================================
# SECTION 10: PER-FUNCTION CONFIGURATION
# ============================================================
# Bounds derived from the ALL-TIME BEST INPUT, not W5 query.
# This is the primary fix from the W5 post-mortem.
# ============================================================

# y_shift map — True for functions with all-negative outputs
y_shift_map = {
    "function_1": False, "function_2": False, "function_3": True,
    "function_4": False, "function_5": False, "function_6": True,
    "function_7": False, "function_8": False,
}

# F5: asymmetric hard bounds — never let dim3 or dim4 fall below 0.95
f5_low  = np.array([0.00, 0.94, 0.95, 0.96])
f5_high = np.array([0.10, 1.00, 1.00, 1.00])

method_config = {
    # F1: tight EI around initial best — FINEGRID and EI have both failed;
    # switching to a slightly wider EI box to avoid degenerate near-zero outputs
    "function_1": {"acq": "EI",   "xi": 0.0001, "margin": 0.025, "restarts": 12},

    # F2: very tight EI around initial best [0.703, 0.927]
    # W5 query at [0.772, 0.857] was 0.087 away — it overshot
    "function_2": {"acq": "EI",   "xi": 0.003,  "margin": 0.035, "restarts": 12},

    # F3: extremely tight EI — three consecutive gains; best is W5 itself
    # Use W5 best as anchor and keep squeezing
    "function_3": {"acq": "EI",   "xi": 0.001,  "margin": 0.030, "restarts": 12},

    # F4: return to W2 best [0.404, 0.398, 0.413, 0.412] as anchor
    # Very tight — W2 best has held for 3 weeks, meaning a sharp peak
    "function_4": {"acq": "EI",   "xi": 0.002,  "margin": 0.018, "restarts": 15},

    # F5: EI with asymmetric bounds — dim3 floor at 0.95, dim4 floor at 0.96
    # W5 dropped dim3 to 0.935 and cost 285 points
    "function_5": {"acq": "EI",   "xi": 0.005,  "custom_bounds": True, "restarts": 15},

    # F6: grid search — GP still unreliable at 25 obs in 5D
    # Return anchor to W4 best [0.712, 0.139, 0.749, 0.710, 0.040]
    # Tighten step from 0.004 to 0.003
    "function_6": {"acq": "GRID", "step": 0.003, "restarts": 15},

    # F7: UCB kappa=0.8 around W4 best — EI in W5 drifted 0.22 away
    # kappa < 1 for strong exploitation; W4 best was genuine optimum signal
    "function_7": {"acq": "UCB",  "kappa": 0.8, "margin": 0.035, "restarts": 20},

    # F8: tight EI around W5 best — W5 IS the current best, exploit it
    "function_8": {"acq": "EI",   "xi": 0.002,  "margin": 0.055, "restarts": 20},
}


# ============================================================
# SECTION 11: RUN LOO-CV — SELECT BEST SURROGATE PER FUNCTION
# ============================================================

print("\n" + "=" * 68)
print("LOO-CV SURROGATE COMPARISON (lower RMSE = selected)")
print(f"{'Fn':<5} {'#obs':>5} {'GP-RMSE':>11} {'MC-RMSE':>11} {'DKL-RMSE':>11}  Winner")
print("-" * 68)

surrogate_selection = {}

for i in range(1, 9):
    key  = f"function_{i}"
    X    = updated_data[key]["X"]
    Y    = updated_data[key]["Y"]
    rmse = loo_rmse_all(X, Y, y_shift=y_shift_map[key])
    winner = min(rmse, key=rmse.get)
    surrogate_selection[key] = {"rmse": rmse, "winner": winner}
    print(f"F{i:<4} {X.shape[0]:>5} {rmse['GP']:>11.4f} "
          f"{rmse['MC_Dropout']:>11.4f} {rmse['DKL']:>11.4f}  {winner}")


# ============================================================
# SECTION 12: GENERATE QUERIES — ONE LOOP, NO OVERRIDE CELLS
# ============================================================

N_CANDS = 60000    # candidate pool for EI / UCB
week6_results = {}

for i in range(1, 9):
    key    = f"function_{i}"
    cfg    = method_config[key]
    X      = updated_data[key]["X"]
    Y      = updated_data[key]["Y"]
    dim    = X.shape[1]
    best   = all_bests[key]["inp"]
    margin = cfg.get("margin")
    winner = surrogate_selection[key]["winner"]
    yshift = y_shift_map[key]

    print(f"\n{'=' * 62}")
    print(f"F{i} | {cfg['acq']} | surrogate={winner} | obs={X.shape[0]} "
          f"| best={Y.max():.6f}")
    print(f"{'=' * 62}")

    # — Build candidate bounds from ALL-TIME BEST INPUT —
    low_b  = np.zeros(dim)
    high_b = np.ones(dim)
    if best is not None and margin is not None:
        low_b  = np.clip(best - margin, 0.0, 1.0)
        high_b = np.clip(best + margin, 0.0, 1.0)
    if key == "function_5":
        low_b, high_b = f5_low, f5_high   # asymmetric override for F5

    # — F6: deterministic grid, GP mean only (no acquisition) —
    if cfg["acq"] == "GRID":
        step = cfg["step"]
        gp, sc = fit_gp(X, Y, n_restarts=cfg["restarts"], y_shift=yshift)
        grid = []
        for deltas in product([-step, 0.0, step], repeat=dim):
            grid.append(np.clip(best + np.array(deltas), 0.0, 1.0))
        grid     = np.unique(np.array(grid), axis=0)
        mu_g, _  = gp.predict(sc.transform(grid), return_std=True)
        best_idx = np.argmax(mu_g)
        query    = grid[best_idx]
        score    = float(mu_g[best_idx])
        mu_out, sigma_out = score, 0.0
        print(f"  Grid: {len(grid)} candidates | best mu: {score:.6f}")

    else:
        # — Draw random candidates within bounds —
        np.random.seed(42)
        cands = np.random.uniform(low_b, high_b, size=(N_CANDS, dim))

        # — Fit the selected surrogate on full data —
        if winner == "GP":
            gp, sc = fit_gp(X, Y, n_restarts=cfg["restarts"], y_shift=yshift)
            mu, sigma = gp.predict(sc.transform(cands), return_std=True)
            if yshift:
                mu = mu + Y.min()     # undo the y_shift

        elif winner == "MC_Dropout":
            model_mc, (ym, ys) = train_mc_dropout(
                X, Y, epochs=300, dropout=0.1, verbose=False
            )
            mu, sigma = mc_predict(model_mc, cands, ym, ys, n_mc=50)

        else:  # DKL
            model_dkl, lik, (ym2, ys2), sc2 = train_dkl(
                X, Y, epochs=200, verbose=False
            )
            mu, sigma = dkl_predict(model_dkl, lik, cands, ym2, ys2, sc2)

        # — Apply acquisition function —
        f_best = Y.max()
        if cfg["acq"] == "EI":
            acq_vals = expected_improvement(mu, sigma, f_best, xi=cfg["xi"])
        else:   # UCB
            acq_vals = upper_confidence_bound(mu, sigma, kappa=cfg["kappa"])

        best_idx  = np.argmax(acq_vals)
        query     = np.clip(cands[best_idx], 0.0, 1.0)
        score     = float(acq_vals[best_idx])
        mu_out    = float(mu[best_idx])
        sigma_out = float(sigma[best_idx])

    formatted = "-".join([f"{x:.6f}" for x in query])
    week6_results[key] = {
        "query":           query,
        "formatted_query": formatted,
        "acq":             cfg["acq"],
        "surrogate":       ("GRID" if cfg["acq"] == "GRID" else winner),
        "score":           score,
        "predicted_mean":  mu_out,
        "uncertainty":     sigma_out,
    }
    dist = np.linalg.norm(query - best)
    print(f"Query     : {formatted}")
    print(f"Score     : {score:.6f} | Pred mean: {mu_out:.6f} | Dist from best: {dist:.4f}")


# ============================================================
# SECTION 13: QUERY VALIDATION
# ============================================================

print("\n" + "=" * 62)
print("QUERY VALIDATION REPORT")
print("=" * 62)

all_clear = True
for i in range(1, 9):
    key   = f"function_{i}"
    query = week6_results[key]["query"]
    acq   = week6_results[key]["acq"]
    surr  = week6_results[key]["surrogate"]
    issues = []
    if np.any(query < 0) or np.any(query > 1):
        issues.append("OUT OF RANGE")
    if np.all(query < 0.01):
        issues.append("SUSPICIOUS: all dims near 0")
    if np.all(query > 0.99):
        issues.append("SUSPICIOUS: all dims near 1")

    status = "OK" if not issues else "WARNING"
    print(f"\nF{i} [{status}] [{acq}] [surrogate: {surr}]")
    print(f"  Query : {week6_results[key]['formatted_query']}")
    if issues:
        all_clear = False
        for w in issues:
            print(f"  !! {w}")

print("\nAll queries valid." if all_clear else "\nReview warnings before submitting.")


# ============================================================
# SECTION 14: PROXIMITY CHECK
# ============================================================

print("\n" + "=" * 62)
print("PROXIMITY TO ALL-TIME BEST INPUT")
print("=" * 62)
for i in range(1, 9):
    key  = f"function_{i}"
    q    = week6_results[key]["query"]
    b    = all_bests[key]["inp"]
    d    = np.linalg.norm(q - b)
    surr = week6_results[key]["surrogate"]
    flag = "ok" if d < 0.20 else "FAR — review"
    print(f"F{i} [{surr}] | dist: {d:.4f} [{flag}]")


# ============================================================
# SECTION 15: PER-FUNCTION MODEL CARDS
# ============================================================
# Each model card documents: function characteristics,
# surrogate architectures with parameter counts, LOO-CV RMSE
# for all three surrogates, the surrogate selected, and the
# full Week 6 query with its predicted performance.
# ============================================================

print("\n\n" + "=" * 65)
print("PER-FUNCTION MODEL CARDS — WEEK 6")
print("=" * 65)

for i in range(1, 9):
    key    = f"function_{i}"
    X      = updated_data[key]["X"]
    Y      = updated_data[key]["Y"]
    dim    = X.shape[1]
    n_obs  = X.shape[0]
    rmse   = surrogate_selection[key]["rmse"]
    winner = surrogate_selection[key]["winner"]
    b      = all_bests[key]
    r      = week6_results[key]
    cfg    = method_config[key]

    # Architecture sizes
    h1_mc  = max(32, 8 * dim);  h2_mc = max(16, 4 * dim)
    p_mc   = (dim*h1_mc + h1_mc) + (h1_mc*h2_mc + h2_mc) + (h2_mc + 1)
    h_dkl  = max(16, 4 * dim)
    p_dkl  = (dim*h_dkl + h_dkl) + (h_dkl*2 + 2)

    gp_tag  = " ← SELECTED" if winner == "GP"         else ""
    mc_tag  = " ← SELECTED" if winner == "MC_Dropout" else ""
    dkl_tag = " ← SELECTED" if winner == "DKL"        else ""

    acq_str = (f"EI (xi={cfg['xi']}, margin={cfg.get('margin', 'custom')})"
               if cfg["acq"] == "EI"
               else f"UCB (kappa={cfg.get('kappa')}, margin={cfg.get('margin')})"
               if cfg["acq"] == "UCB"
               else f"Grid (step={cfg['step']})")

    print(f"\n{'─'*65}")
    print(f"  MODEL CARD — F{i}  ({dim}D | {n_obs} observations)")
    print(f"{'─'*65}")
    print(f"  All-time best  : {b['val']:.6g}  ({b['src']})")
    print(f"  Best input     : {np.round(b['inp'], 4)}")
    print(f"  Note           : {b['note']}")
    print()
    print(f"  ARCHITECTURES")
    print(f"    GP     : ConstantKernel × RBF + WhiteKernel (sklearn)")
    print(f"    MC-MLP : Input({dim}) -> Lin({h1_mc}) -> ReLU -> Drop(0.1)"
          f" -> Lin({h2_mc}) -> ReLU -> Drop(0.1) -> Lin(1)")
    print(f"             {p_mc} parameters | T=50 MC passes | Adam lr=0.001")
    print(f"    DKL    : NN: Input({dim})->Lin({h_dkl})->ReLU->Lin(2) "
          f"[{p_dkl} params]")
    print(f"             + RBF-GP on 2D latent | Adam lr=0.01 | 200 epochs")
    print()
    print(f"  LOO-CV RMSE")
    print(f"    GP         : {rmse['GP']:.5f}{gp_tag}")
    print(f"    MC Dropout : {rmse['MC_Dropout']:.5f}{mc_tag}")
    print(f"    DKL        : {rmse['DKL']:.5f}{dkl_tag}")
    print()
    print(f"  WEEK 6 QUERY")
    print(f"    Acquisition    : {acq_str}")
    print(f"    Surrogate      : {r['surrogate']}")
    print(f"    Query          : {r['formatted_query']}")
    print(f"    Predicted mean : {r['predicted_mean']:.6g}")
    print(f"    Uncertainty    : {r['uncertainty']:.6g}")
    d = np.linalg.norm(r['query'] - b['inp'])
    print(f"    Dist from best : {d:.4f}")


# ============================================================
# SECTION 16: SAVE QUERIES
# ============================================================

rows = []
for i in range(1, 9):
    key  = f"function_{i}"
    Y    = updated_data[key]["Y"]
    r    = week6_results[key]
    rmse = surrogate_selection[key]["rmse"]
    rows.append({
        "Function":        f"Function {i}",
        "Query":           r["formatted_query"],
        "Acquisition":     r["acq"],
        "Surrogate":       r["surrogate"],
        "GP_LOO_RMSE":     round(rmse["GP"],          4),
        "MC_LOO_RMSE":     round(rmse["MC_Dropout"],  4),
        "DKL_LOO_RMSE":    round(rmse["DKL"],         4),
        "Predicted_mean":  round(r["predicted_mean"], 6),
        "Uncertainty":     round(r["uncertainty"],    6),
        "Current_best":    round(float(Y.max()),      6),
    })

df = pd.DataFrame(rows)
print("\n\nFINAL WEEK 6 QUERIES")
print(df[["Function", "Surrogate", "Acquisition",
          "Query", "Current_best"]].to_string(index=False))

df[["Function", "Query"]].to_csv("week6_queries.csv", index=False)
df.to_csv("week6_queries_full.csv", index=False)
print("\nweek6_queries.csv      — submit this file")
print("week6_queries_full.csv — includes LOO-RMSE and model card data")
