# ============================================================
# WEEK 7 BAYESIAN OPTIMISATION — CONVERGENCE & FINE SEARCH
# ============================================================
#
# WEEK 6 POST-MORTEM:
#   Gains: F5 (NEW BEST +13%, DKL surrogate), F7 (NEW BEST +17.6%, GP)
#   Declines: F1, F2, F3, F4, F6, F8
#
#   Root causes:
#   F2: dim2 at 0.892 — still 0.034 below the initial best's 0.927
#   F3: MC Dropout moved dim2 and dim3 further from W5 best; regression
#   F4: dim1 drifted -0.018 and dim3 +0.018 from W2 best simultaneously
#   F6: grid step 0.003 moved ALL dims in wrong direction; W4 best holds
#   F8: L2 dist 0.114 from W5 best — too far in 8D space
#   F1: near-zero everywhere; broader EI search needed to escape flatness
#
#   Key findings confirmed:
#   F5: DKL found a genuinely better region (dim1 near 0 confirmed optimal)
#   F7: GP with UCB kappa=0.8 delivered the best-ever result (2.374)
#
# WEEK 7 STRATEGY:
#   F5, F7: tightest EI boxes ever — exploit these new bests immediately
#   F2: raise dim2 floor to 0.92 (hard constraint, not just a bound)
#   F3: revert to GP, tighter box around W5 best (MC Dropout caused drift)
#   F4: micro-box of margin 0.012 around W2 best — sharpest peak hypothesis
#   F6: switch from GRID to EI around W4 best; grid has plateaued
#   F8: tighten margin to 0.035 anchored firmly to W5 best
#   F1: widen search to margin 0.08 — 7 weeks near [0.73,0.73] found nothing
#
# THREE SURROGATE ARCHITECTURES (unchanged from W6):
#   1. Vanilla GP — ConstantKernel x RBF + WhiteKernel (sklearn)
#   2. MC Dropout MLP — Gal & Ghahramani (2016)
#   3. Deep Kernel Learning — Wilson et al. (2016) via GPyTorch
#
#   Surrogate selection via LOO-CV RMSE per function.
#   W6 lesson: where DKL won (F5) and GP won (F7), follow the data.
#   Research basis: EGP ensemble (arXiv:2205.14090) and convergence
#   monitoring (MDPI Mathematics 2025) confirm data-driven selection
#   outperforms fixed surrogate assignment.
#
# REFERENCES:
#   Rasmussen & Williams (2006) — GP theory
#   Jones et al. (1998) — Expected Improvement
#   Gal & Ghahramani (2016, arXiv:1506.02142) — MC Dropout
#   Wilson et al. (2016) / Gardner et al. (2018) — DKL via GPyTorch
#   Li et al. ICLR 2024 (arXiv:2305.20028) — BNN vs GP study
#   arXiv:2205.14090 — EGP ensemble surrogate
#   arXiv:2501.09262 — EI convergence theory
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
# SECTION 2: COMPLETE QUERY HISTORY — WEEKS 1-6
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
week6_queries = {
    "function_1": np.array([0.706220, 0.757972]),
    "function_2": np.array([0.737524, 0.891592]),
    "function_3": np.array([0.621632, 0.501799, 0.454372]),
    "function_4": np.array([0.385718, 0.396061, 0.431327, 0.422779]),
    "function_5": np.array([0.000563, 0.999140, 0.999023, 0.999846]),
    "function_6": np.array([0.709186, 0.141693, 0.745552, 0.706997, 0.037401]),
    "function_7": np.array([0.092365, 0.378563, 0.527314, 0.189186, 0.366707, 0.705150]),
    "function_8": np.array([0.031249, 0.188527, 0.117555, 0.143463,
                             0.713608, 0.547419, 0.184061, 0.520388]),
}
week6_outputs = {
    "function_1": -1.313446068e-16, "function_2":  0.544922639,
    "function_3": -0.014702510,     "function_4":  0.418765262,
    "function_5":  4412.974620,     "function_6": -0.767567783,
    "function_7":  2.374187172,     "function_8":  9.981432643,
}


# ============================================================
# SECTION 3: BUILD DATASETS (original + W1 to W6)
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
        week6_queries[key].reshape(1, -1),
    ])
    Y_updated = np.append(
        original_data[key]["Y"],
        [week1_outputs[key], week2_outputs[key], week3_outputs[key],
         week4_outputs[key], week5_outputs[key], week6_outputs[key]]
    )
    updated_data[key] = {"X": X_updated, "Y": Y_updated}
    best_idx = np.argmax(Y_updated)
    print(f"F{i} | obs: {X_updated.shape[0]} | best: {Y_updated[best_idx]:.6f} "
          f"at {np.round(X_updated[best_idx], 4)}")


# ============================================================
# SECTION 4: ALL-TIME BESTS ENTERING WEEK 7
# ============================================================
# CRITICAL: always anchor to the input that produced the ATB,
# not the most recent query. This is the core validated fix.
# ============================================================

print("\n" + "=" * 65)
print("ENTERING WEEK 7 — ALL-TIME BESTS (CONFIRMED)")
print("=" * 65)

all_bests = {
    "function_1": {
        "val": 7.71e-16,
        "inp": np.array([0.731024, 0.732999]),
        "src": "Initial",
        "note": "Near-flat function — widening search this week",
    },
    "function_2": {
        "val": 0.61120522,
        "inp": np.array([0.702637, 0.926564]),
        "src": "Initial",
        "note": "dim2 must be >= 0.92; W6 at 0.892 cost 0.066 points",
    },
    "function_3": {
        "val": -0.00319822,
        "inp": np.array([0.618443, 0.512242, 0.466082]),
        "src": "W5",
        "note": "MC Dropout drifted in W6; revert to GP with tighter box",
    },
    "function_4": {
        "val": 0.53385778,
        "inp": np.array([0.403695, 0.397605, 0.413333, 0.411576]),
        "src": "W2",
        "note": "Sharp peak hypothesis — micro-box margin 0.012",
    },
    "function_5": {
        "val": 4412.974620,
        "inp": np.array([0.000563, 0.999140, 0.999023, 0.999846]),
        "src": "W6",
        "note": "NEW BEST: DKL found dim1 near 0 is optimal; exploit tightly",
    },
    "function_6": {
        "val": -0.68015232,
        "inp": np.array([0.712186, 0.138693, 0.748552, 0.709997, 0.040401]),
        "src": "W4",
        "note": "Grid plateaued; switching to EI around W4 best",
    },
    "function_7": {
        "val": 2.37418717,
        "inp": np.array([0.092365, 0.378563, 0.527314, 0.189186, 0.366707, 0.705150]),
        "src": "W6",
        "note": "NEW BEST: tightest EI box yet; keep exploiting W6 region",
    },
    "function_8": {
        "val": 9.98947835,
        "inp": np.array([0.077857, 0.141235, 0.128950, 0.097085,
                          0.760277, 0.523742, 0.158135, 0.465728]),
        "src": "W5",
        "note": "Tighten margin to 0.035 — W6 was too far (dist 0.114)",
    },
}

for i in range(1, 9):
    key = f"function_{i}"
    b = all_bests[key]
    print(f"F{i} | {b['val']:.6g} ({b['src']}) | {b['note']}")


# ============================================================
# SECTION 5: ARCHITECTURE 1 — VANILLA GP
# ============================================================

def fit_gp(X, Y, n_restarts=15, y_shift=False):
    """
    Vanilla GP: ConstantKernel x RBF + WhiteKernel (sklearn).
    Unchanged from W1. y_shift=True offsets Y by its minimum
    before fitting, stabilising training for all-negative outputs.
    """
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
# Gal & Ghahramani (2016): dropout active at inference.
# T=50 stochastic forward passes estimate epistemic uncertainty.
# h1=max(32,8d), h2=max(16,4d) scales with input dimensionality.
# Dropout p=0.1; Adam lr=0.001, weight_decay=1e-3, 300 epochs.
# W6 lesson: MC Dropout won on F3 by LOO-RMSE but caused drift
# in the actual query. LOO-RMSE still selects it if it fits
# better; the tighter margin is the safety mechanism.
# ============================================================

class MCDropoutMLP(nn.Module):
    """
    Two-hidden-layer MLP. Dropout kept ACTIVE at inference.
    Stochastic forward passes approximate predictive uncertainty.
    Architecture: Input(d)->Lin(h1)->ReLU->Drop(p)->
                             Lin(h2)->ReLU->Drop(p)->Lin(1)
    """
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.h1 = max(32, 8 * input_dim)
        self.h2 = max(16, 4 * input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.h1), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(self.h1, self.h2),  nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(self.h2, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_mc_dropout(X, Y, epochs=300, lr=0.001, dropout=0.1,
                     weight_decay=1e-3, verbose=False):
    """Train MCDropoutMLP. Standardises Y before fitting."""
    y_mean = float(Y.mean())
    y_std  = float(Y.std()) if Y.std() > 1e-12 else 1.0
    Y_s    = (Y - y_mean) / y_std
    X_t    = torch.tensor(X,   dtype=torch.float32)
    Y_t    = torch.tensor(Y_s, dtype=torch.float32).reshape(-1, 1)
    model  = MCDropoutMLP(X.shape[1], dropout=dropout)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        loss = nn.MSELoss()(model(X_t), Y_t)
        loss.backward()
        opt.step()
        if verbose and epoch % 100 == 0:
            print(f"    [MC-MLP] epoch {epoch:4d} | loss {loss.item():.5f}")
    return model, (y_mean, y_std)


def mc_predict(model, X_np, y_mean, y_std, n_mc=50):
    """T=50 stochastic passes; model.train() keeps dropout active."""
    model.train()
    X_t = torch.tensor(X_np, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for _ in range(n_mc):
            preds.append(model(X_t).numpy().flatten())
    preds = np.array(preds)
    return preds.mean(axis=0) * y_std + y_mean, np.maximum(preds.std(axis=0) * y_std, 1e-9)


# ============================================================
# SECTION 7: ARCHITECTURE 3 — DEEP KERNEL LEARNING
# ============================================================
# Wilson et al. (2016): NN feature extractor + GP RBF kernel.
# Latent dim=2 keeps NN params low at n<50.
# NN weights and GP hyperparameters trained jointly via
# negative marginal log likelihood (Type-II MLE).
# W6 lesson: DKL found F5's dim1-near-zero region (new best +507
# units). This validates DKL for non-stationary landscapes.
# ============================================================

class DKLFeatureExtractor(nn.Module):
    """Input(d)->Linear(h)->ReLU->Linear(2). h=max(16,4d)."""
    def __init__(self, input_dim):
        super().__init__()
        h = max(16, 4 * input_dim)
        self.h = h
        self.net = nn.Sequential(
            nn.Linear(input_dim, h), nn.ReLU(), nn.Linear(h, 2)
        )

    def forward(self, x):
        return self.net(x)


class DKLGPModel(gpytorch.models.ExactGP):
    """Exact GP with ScaleKernel(RBF) on 2D latent features."""
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        z = self.feature_extractor(x)
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(z), self.covar_module(z)
        )


def train_dkl(X, Y, epochs=200, lr=0.01, verbose=False):
    """Train DKL. Standardises Y. Jointly optimises NN + GP via MLL."""
    y_mean = float(Y.mean())
    y_std  = float(Y.std()) if Y.std() > 1e-12 else 1.0
    Y_s    = (Y - y_mean) / y_std
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    train_x = torch.tensor(X_s, dtype=torch.float32)
    train_y = torch.tensor(Y_s, dtype=torch.float32)
    feat_ext   = DKLFeatureExtractor(X.shape[1])
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model      = DKLGPModel(train_x, train_y, likelihood, feat_ext)
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(
        list(model.feature_extractor.parameters())
        + list(model.covar_module.parameters())
        + list(model.mean_module.parameters())
        + list(likelihood.parameters()), lr=lr
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
    """Exact GP posterior on 2D latent space."""
    model.eval(); likelihood.eval()
    X_t = torch.tensor(scaler.transform(X_np), dtype=torch.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(X_t))
    return pred.mean.numpy() * y_std + y_mean, np.maximum(pred.stddev.numpy() * y_std, 1e-9)


# ============================================================
# SECTION 8: ACQUISITION FUNCTIONS
# ============================================================

def expected_improvement(mu, sigma, f_best, xi=0.01):
    """
    Closed-form EI (Jones et al. 1998, eq.15).
    Works with mu, sigma from any surrogate.
    xi=0.001 for convergence exploitation; higher xi explores more.
    EI convergence theory (arXiv:2501.09262): EI achieves the
    optimal convergence rate under GP prior assumptions.
    """
    sigma = np.maximum(sigma, 1e-9)
    Z  = (mu - f_best - xi) / sigma
    ei = (mu - f_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei


def upper_confidence_bound(mu, sigma, kappa=0.8):
    """UCB = mu + kappa * sigma. Low kappa for strong exploitation."""
    return mu + kappa * sigma


# ============================================================
# SECTION 9: LOO-CV SURROGATE COMPARISON
# ============================================================

def loo_rmse_all(X, Y, y_shift=False):
    """
    Leave-one-out RMSE for all three surrogates.
    The lowest RMSE determines which surrogate generates the query.
    Based on Li et al. ICLR 2024 recommendation for small-n selection.
    """
    loo = LeaveOneOut()
    gp_sq, mc_sq, dkl_sq = [], [], []

    for train_idx, test_idx in loo.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        Ytr, Yte = Y[train_idx], Y[test_idx]

        # GP
        gp, sc = fit_gp(Xtr, Ytr, n_restarts=5, y_shift=y_shift)
        mu_g, _ = gp.predict(sc.transform(Xte), return_std=True)
        pred_g  = float(mu_g[0]) + (Ytr.min() if y_shift else 0)
        gp_sq.append((pred_g - Yte[0]) ** 2)

        # MC Dropout
        try:
            m_mc, (ym, ys) = train_mc_dropout(Xtr, Ytr, epochs=200)
            mu_mc, _ = mc_predict(m_mc, Xte, ym, ys, n_mc=20)
            mc_sq.append((float(mu_mc[0]) - Yte[0]) ** 2)
        except Exception:
            mc_sq.append((float(np.mean(Ytr)) - Yte[0]) ** 2)

        # DKL
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

y_shift_map = {
    "function_1": False, "function_2": False, "function_3": True,
    "function_4": False, "function_5": False, "function_6": True,
    "function_7": False, "function_8": False,
}

# F5 asymmetric bounds: dim1 near 0 confirmed optimal in W6 (DKL)
# dim3 and dim4 floors raised — dim3 must stay above 0.996
f5_low  = np.array([0.000, 0.997, 0.996, 0.998])
f5_high = np.array([0.003, 1.000, 1.000, 1.000])

# F2 asymmetric bounds: hard dim2 floor at 0.92
f2_low  = np.array([0.670, 0.920])
f2_high = np.array([0.740, 0.960])

method_config = {
    # F1: widen search — 7 weeks near [0.73,0.73] found nothing new
    # Try broader EI with larger margin to escape the flat region
    "function_1": {"acq": "EI",  "xi": 0.0001, "margin": 0.080, "restarts": 12, "custom_bounds": False},

    # F2: hard asymmetric bounds — dim2 floor 0.92, ceiling 0.96
    # W6 at dim2=0.892 returned 0.545 vs initial best 0.611 at 0.927
    "function_2": {"acq": "EI",  "xi": 0.002,  "margin": None,  "restarts": 12, "custom_bounds": True},

    # F3: GP only (MC Dropout caused regression in W6), tight 0.025 margin
    "function_3": {"acq": "EI",  "xi": 0.001,  "margin": 0.025, "restarts": 12, "custom_bounds": False, "force_gp": True},

    # F4: micro-box 0.012 — sharpest peak hypothesis; W2 best held 5 weeks
    "function_4": {"acq": "EI",  "xi": 0.001,  "margin": 0.012, "restarts": 15, "custom_bounds": False},

    # F5: asymmetric bounds tightened further — dim1 [0,0.003]
    # DKL confirmed dim1 near-zero is optimal; exploit this fully
    "function_5": {"acq": "EI",  "xi": 0.005,  "margin": None,  "restarts": 15, "custom_bounds": True},

    # F6: switch from GRID to EI around W4 best
    # Grid has plateaued; EI with tight margin may find what grid missed
    "function_6": {"acq": "EI",  "xi": 0.002,  "margin": 0.025, "restarts": 15, "custom_bounds": False},

    # F7: ultra-tight EI around W6 best [0.092, 0.379, 0.527, 0.189, 0.367, 0.705]
    # W6 was a +17.6% jump; follow it immediately with the tightest box used
    "function_7": {"acq": "EI",  "xi": 0.001,  "margin": 0.020, "restarts": 20, "custom_bounds": False},

    # F8: tighten margin to 0.035 anchored to W5 best
    # W6 was too far (dist 0.114, 8 dims). W5 best at 9.9895 still holds.
    "function_8": {"acq": "UCB", "kappa": 0.8, "margin": 0.035, "restarts": 20, "custom_bounds": False},
}


# ============================================================
# SECTION 11: LOO-CV SURROGATE SELECTION
# ============================================================

print("\n" + "=" * 68)
print("LOO-CV SURROGATE COMPARISON")
print(f"{'Fn':<5} {'Obs':>4} {'GP-RMSE':>11} {'MC-RMSE':>11} {'DKL-RMSE':>11}  Winner")
print("-" * 68)

surrogate_selection = {}

for i in range(1, 9):
    key   = f"function_{i}"
    X     = updated_data[key]["X"]
    Y     = updated_data[key]["Y"]
    cfg   = method_config[key]
    rmse  = loo_rmse_all(X, Y, y_shift=y_shift_map[key])

    # F3 override: force GP regardless of LOO result
    if cfg.get("force_gp"):
        winner = "GP"
        rmse["note"] = "GP forced (MC Dropout caused drift in W6)"
    else:
        winner = min(rmse, key=lambda k: rmse[k])

    surrogate_selection[key] = {"rmse": rmse, "winner": winner}
    print(f"F{i:<4} {X.shape[0]:>4} {rmse['GP']:>11.4f} "
          f"{rmse['MC_Dropout']:>11.4f} {rmse['DKL']:>11.4f}  {winner}")


# ============================================================
# SECTION 12: GENERATE QUERIES — ONE LOOP
# ============================================================

N_CANDS    = 60000
week7_results = {}

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
    print(f"F{i} | {cfg['acq']} | surrogate={winner} | obs={X.shape[0]} | best={Y.max():.6f}")
    print(f"{'=' * 62}")

    # Build search bounds
    if cfg.get("custom_bounds") and key == "function_5":
        low_b, high_b = f5_low, f5_high
    elif cfg.get("custom_bounds") and key == "function_2":
        low_b, high_b = f2_low, f2_high
    elif best is not None and margin is not None:
        low_b  = np.clip(best - margin, 0.0, 1.0)
        high_b = np.clip(best + margin, 0.0, 1.0)
    else:
        low_b  = np.zeros(dim)
        high_b = np.ones(dim)

    np.random.seed(42)
    cands = np.random.uniform(low_b, high_b, size=(N_CANDS, dim))

    # Fit the selected surrogate
    if winner == "GP":
        gp, sc = fit_gp(X, Y, n_restarts=cfg["restarts"], y_shift=yshift)
        mu, sigma = gp.predict(sc.transform(cands), return_std=True)
        if yshift:
            mu = mu + Y.min()

    elif winner == "MC_Dropout":
        model_mc, (ym, ys) = train_mc_dropout(X, Y, epochs=300, dropout=0.1)
        mu, sigma = mc_predict(model_mc, cands, ym, ys, n_mc=50)

    else:  # DKL
        model_dkl, lik, (ym2, ys2), sc2 = train_dkl(X, Y, epochs=200)
        mu, sigma = dkl_predict(model_dkl, lik, cands, ym2, ys2, sc2)

    f_best = Y.max()

    if cfg["acq"] == "EI":
        acq_vals = expected_improvement(mu, sigma, f_best, xi=cfg["xi"])
    else:
        acq_vals = upper_confidence_bound(mu, sigma, kappa=cfg["kappa"])

    best_idx  = np.argmax(acq_vals)
    query     = np.clip(cands[best_idx], 0.0, 1.0)
    score     = float(acq_vals[best_idx])
    mu_out    = float(mu[best_idx])
    sigma_out = float(sigma[best_idx])

    formatted = "-".join([f"{x:.6f}" for x in query])
    week7_results[key] = {
        "query":           query,
        "formatted_query": formatted,
        "acq":             cfg["acq"],
        "surrogate":       winner,
        "score":           score,
        "predicted_mean":  mu_out,
        "uncertainty":     sigma_out,
    }

    dist = np.linalg.norm(query - best)
    print(f"Query     : {formatted}")
    print(f"Score     : {score:.6f} | Pred mean: {mu_out:.6f} | Dist from ATB: {dist:.4f}")
    print(f"Bounds    : low={np.round(low_b,3)} | high={np.round(high_b,3)}")


# ============================================================
# SECTION 13: VALIDATION
# ============================================================

print("\n" + "=" * 62)
print("QUERY VALIDATION REPORT")
print("=" * 62)

all_clear = True
for i in range(1, 9):
    key   = f"function_{i}"
    query = week7_results[key]["query"]
    acq   = week7_results[key]["acq"]
    surr  = week7_results[key]["surrogate"]
    issues = []
    if np.any(query < 0) or np.any(query > 1):
        issues.append("OUT OF RANGE")
    if np.all(query < 0.01) and i not in [5]:
        issues.append("SUSPICIOUS: all dims near 0")
    if np.all(query > 0.99):
        issues.append("SUSPICIOUS: all dims near 1")

    status = "OK" if not issues else "WARNING"
    print(f"\nF{i} [{status}] [{acq}] [surrogate: {surr}]")
    print(f"  Query : {week7_results[key]['formatted_query']}")
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
    q    = week7_results[key]["query"]
    b    = all_bests[key]["inp"]
    d    = np.linalg.norm(q - b)
    surr = week7_results[key]["surrogate"]
    flag = "ok" if d < 0.20 else "far — review"
    print(f"F{i} [{surr}] | dist: {d:.4f} [{flag}]")


# ============================================================
# SECTION 15: PER-FUNCTION MODEL CARDS
# ============================================================

print("\n\n" + "=" * 65)
print("PER-FUNCTION MODEL CARDS — WEEK 7")
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
    r      = week7_results[key]
    cfg    = method_config[key]

    h1_mc  = max(32, 8 * dim); h2_mc = max(16, 4 * dim)
    p_mc   = (dim*h1_mc + h1_mc) + (h1_mc*h2_mc + h2_mc) + (h2_mc + 1)
    h_dkl  = max(16, 4 * dim)
    p_dkl  = (dim*h_dkl + h_dkl) + (h_dkl*2 + 2)

    gp_tag  = " SELECTED" if winner == "GP"         else ""
    mc_tag  = " SELECTED" if winner == "MC_Dropout" else ""
    dkl_tag = " SELECTED" if winner == "DKL"        else ""
    if cfg.get("force_gp"):
        gp_tag = " SELECTED (forced: MC Dropout caused drift in W6)"

    acq_str = (f"EI xi={cfg['xi']} margin={cfg.get('margin','custom')}"
               if cfg["acq"] == "EI"
               else f"UCB kappa={cfg.get('kappa')} margin={cfg.get('margin')}")

    print(f"\n{'─'*65}")
    print(f"  MODEL CARD   F{i}  ({dim}D | {n_obs} obs)")
    print(f"{'─'*65}")
    print(f"  All-time best  : {b['val']:.6g}  ({b['src']})")
    print(f"  Best input     : {np.round(b['inp'], 4)}")
    print(f"  Strategy note  : {b['note']}")
    print()
    print(f"  ARCHITECTURES")
    print(f"    GP       : ConstantKernel x RBF + WhiteKernel (sklearn)")
    print(f"    MC-MLP   : Input({dim})->Lin({h1_mc})->ReLU->Drop(0.1)"
          f"->Lin({h2_mc})->ReLU->Drop(0.1)->Lin(1) | {p_mc} params | T=50")
    print(f"    DKL      : NN Input({dim})->Lin({h_dkl})->ReLU->Lin(2)"
          f" [{p_dkl} NN params] + RBF-GP on 2D latent")
    print()
    print(f"  LOO-CV RMSE")
    print(f"    GP         : {rmse['GP']:.5f}{gp_tag}")
    print(f"    MC Dropout : {rmse['MC_Dropout']:.5f}{mc_tag}")
    print(f"    DKL        : {rmse['DKL']:.5f}{dkl_tag}")
    print()
    print(f"  WEEK 7 QUERY")
    print(f"    Acquisition    : {acq_str}")
    print(f"    Surrogate      : {r['surrogate']}")
    print(f"    Query          : {r['formatted_query']}")
    print(f"    Predicted mean : {r['predicted_mean']:.6g}")
    print(f"    Uncertainty    : {r['uncertainty']:.6g}")
    d = np.linalg.norm(r['query'] - b['inp'])
    print(f"    Dist from ATB  : {d:.4f}")


# ============================================================
# SECTION 16: SAVE QUERIES
# ============================================================

rows = []
for i in range(1, 9):
    key  = f"function_{i}"
    Y    = updated_data[key]["Y"]
    r    = week7_results[key]
    rmse = surrogate_selection[key]["rmse"]
    rows.append({
        "Function":       f"Function {i}",
        "Query":          r["formatted_query"],
        "Acquisition":    r["acq"],
        "Surrogate":      r["surrogate"],
        "GP_LOO_RMSE":    round(rmse["GP"],          4),
        "MC_LOO_RMSE":    round(rmse["MC_Dropout"],  4),
        "DKL_LOO_RMSE":   round(rmse["DKL"],         4),
        "Predicted_mean": round(r["predicted_mean"],  6),
        "Uncertainty":    round(r["uncertainty"],     6),
        "Current_best":   round(float(Y.max()),       6),
    })

df = pd.DataFrame(rows)
print("\n\nFINAL WEEK 7 QUERIES")
print(df[["Function", "Surrogate", "Acquisition",
          "Query", "Current_best"]].to_string(index=False))

df[["Function", "Query"]].to_csv("week7_queries.csv", index=False)
df.to_csv("week7_queries_full.csv", index=False)
print("\nweek7_queries.csv      — submit this file")
print("week7_queries_full.csv — includes LOO-RMSE and model card data")
