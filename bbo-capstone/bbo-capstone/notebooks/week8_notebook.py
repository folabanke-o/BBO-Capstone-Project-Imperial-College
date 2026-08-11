# ============================================================
# WEEK 8 BAYESIAN OPTIMISATION — ARD KERNELS + MATERN + ZOMBI
# ============================================================
#
# WEEK 7 POST-MORTEM:
#   Gains: F5 (+0.58%), F7 (+7.98%), F8 (+0.065%)
#   Declines: F1, F2, F3, F4, F6
#
#   Root causes:
#   F3: dim3 drifted to 0.491 (was 0.466 at W5 best); crossed out of peak
#   F2: DKL raised dim2 to 0.954 but dropped dim1 to 0.671 (-0.032 from initial)
#   F4: W2 best unbeaten for 6 weeks; micro-box still not tight enough
#   F6: EI moved to worse cell; grid approach still better for F6
#   F1: Widened search to [0.81, 0.81] — even further from any signal
#
# WEEK 8 STRATEGY — THREE NEW TECHNIQUES:
#
# 1. AUTOMATIC RELEVANCE DETERMINATION (ARD) — Colleague suggestion
#    Replace scalar length_scale with vector length_scale (one per dim).
#    Colleague: "use automatic relevance determination (different
#    length_scales per dimension)".
#    Reference: Rasmussen & Williams (2006), Chapter 5.1
#    sklearn: RBF(length_scale=np.ones(d)) — vector instead of scalar
#    Impact: GP learns WHICH dims matter WITHOUT hand-coded bounds.
#    For F5: should learn dim1 is critical (short length_scale).
#    For F7/F8: identifies which of 6/8 dims drive the output.
#
# 2. MATERN KERNEL — Colleague suggestion + literature
#    "I'd also spend some time scaling the data as the outputs in
#    Function 1 are quite varied."
#    Literature: BOOST (2025) prioritises Matern over RBF for non-smooth
#    functions. RBF assumes infinite differentiability; Matern-2.5 is
#    twice differentiable — more realistic for black-box functions.
#    Added as a third kernel option alongside RBF in LOO comparison.
#    sklearn: Matern(length_scale=np.ones(d), nu=2.5)
#
# 3. Y SCALING ON GP — Colleague suggestion
#    "spend some time scaling the data as the outputs are quite varied"
#    Current: GP fitted on raw Y (F5 spans 1089 to 4438 — huge range).
#    Fix: StandardScaler on Y before GP fitting, invert after prediction.
#    Applied to ALL GP surrogate fits in W8.
#
# 4. LIGHTWEIGHT ZoMBI FOR F1 AND F4
#    ZoMBI (Siemenn et al. 2023, npj Computational Materials):
#    "zooming memory-based initialization — iteratively zooms in the
#    sampling search bounds using the m best-performing observations,
#    then prunes the memory of low-performing historical experiments."
#    F1 and F4 are needle-in-a-haystack problems: 0/16 and 3/35 good
#    observations respectively. Full ZoMBI implementation:
#    - ZOOM: bounds set from top-3 best inputs per dimension
#    - PRUNE: fit GP on only top-m=5 best observations (not all history)
#    This prevents bad early queries from biasing the GP mean.
#
# KERNEL SELECTION (LOO-CV RMSE across 4 surrogates):
#   1. GP + ARD RBF (NEW)
#   2. GP + ARD Matern-2.5 (NEW)
#   3. MC Dropout MLP (unchanged)
#   4. DKL (unchanged)
#   Winner per function generates the actual query.
#
# REFERENCES:
#   Rasmussen & Williams (2006) — ARD, GP theory
#   Jones et al. (1998) — Expected Improvement
#   Matern (1960), Stein (1999) — Matern kernel
#   BOOST arXiv:2508.02332 — Matern > RBF for non-smooth functions
#   Gal & Ghahramani (2016) — MC Dropout uncertainty
#   Wilson et al. (2016) / Gardner et al. (2018) — DKL via GPyTorch
#   Siemenn et al. (2023), npj Comp. Materials — ZoMBI
#   Li et al. ICLR 2024 arXiv:2305.20028 — BNN vs GP study
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gpytorch
import warnings
warnings.filterwarnings("ignore")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ConstantKernel, WhiteKernel, Matern
)
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
# SECTION 2: COMPLETE QUERY HISTORY — WEEKS 1-7
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
week7_queries = {
    "function_1": np.array([0.810503, 0.812929]),
    "function_2": np.array([0.670594, 0.953914]),
    "function_3": np.array([0.641201, 0.489832, 0.490912]),
    "function_4": np.array([0.392098, 0.396579, 0.425224, 0.422943]),
    "function_5": np.array([0.000017, 0.999957, 0.999922, 0.999992]),
    "function_6": np.array([0.687242, 0.162414, 0.724851, 0.692615, 0.036503]),
    "function_7": np.array([0.109098, 0.358650, 0.545872, 0.208942, 0.350961, 0.710041]),
    "function_8": np.array([0.103018, 0.152851, 0.147527, 0.130788,
                             0.794596, 0.495937, 0.186257, 0.448233]),
}
week7_outputs = {
    "function_1":  1.16714801e-47,  "function_2":  0.46037541,
    "function_3": -0.01407102,      "function_4":  0.48992037,
    "function_5":  4438.73290,      "function_6": -0.75896416,
    "function_7":  2.56363741,      "function_8":  9.99597080,
}


# ============================================================
# SECTION 3: BUILD DATASETS (original + W1 to W7)
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
        week7_queries[key].reshape(1, -1),
    ])
    Y_updated = np.append(
        original_data[key]["Y"],
        [week1_outputs[key], week2_outputs[key], week3_outputs[key],
         week4_outputs[key], week5_outputs[key], week6_outputs[key],
         week7_outputs[key]]
    )
    updated_data[key] = {"X": X_updated, "Y": Y_updated}
    best_idx = np.argmax(Y_updated)
    print(f"F{i} | obs: {X_updated.shape[0]} | best: {Y_updated[best_idx]:.6f} "
          f"at {np.round(X_updated[best_idx], 4)}")


# ============================================================
# SECTION 4: ALL-TIME BESTS ENTERING WEEK 8
# ============================================================
# CRITICAL: Always anchor bounds to the input that produced the
# all-time best — confirmed fix since Week 5.
# ============================================================

print("\n" + "=" * 65)
print("ENTERING WEEK 8 — ALL-TIME BESTS")
print("=" * 65)

all_bests = {
    "function_1": {
        "val": 7.71e-16,
        "inp": np.array([0.731024, 0.732999]),
        "src": "Initial",
        "note": "Near-flat; ZoMBI prune+zoom on top-5 obs applied W8",
    },
    "function_2": {
        "val": 0.61120522,
        "inp": np.array([0.702637, 0.926564]),
        "src": "Initial",
        "note": "W7 DKL raised dim2 to 0.954 but dim1 dropped to 0.671; "
                "W8 anchor both dims jointly closer to initial best",
    },
    "function_3": {
        "val": -0.00319822,
        "inp": np.array([0.618443, 0.512242, 0.466082]),
        "src": "W5",
        "note": "dim3 regressed to 0.491 in W7; pull back to 0.466 region",
    },
    "function_4": {
        "val": 0.53385778,
        "inp": np.array([0.403695, 0.397605, 0.413333, 0.411576]),
        "src": "W2",
        "note": "6 weeks unbeaten; ZoMBI prune+zoom applied; "
                "ARD will reveal which dims matter most",
    },
    "function_5": {
        "val": 4438.73290,
        "inp": np.array([0.000017, 0.999957, 0.999922, 0.999992]),
        "src": "W7",
        "note": "dim1 near zero confirmed optimal across W6+W7; "
                "tighten further to [0.000, 0.001]",
    },
    "function_6": {
        "val": -0.68015232,
        "inp": np.array([0.712186, 0.138693, 0.748552, 0.709997, 0.040401]),
        "src": "W4",
        "note": "EI failed in W7; return to grid search around W4 best",
    },
    "function_7": {
        "val": 2.56363741,
        "inp": np.array([0.109098, 0.358650, 0.545872, 0.208942, 0.350961, 0.710041]),
        "src": "W7",
        "note": "Two consecutive new bests; tight EI 0.018 margin",
    },
    "function_8": {
        "val": 9.99597080,
        "inp": np.array([0.103018, 0.152851, 0.147527, 0.130788,
                          0.794596, 0.495937, 0.186257, 0.448233]),
        "src": "W7",
        "note": "Function approaching ceiling ~10; tight UCB",
    },
}

for i in range(1, 9):
    key = f"function_{i}"
    b = all_bests[key]
    print(f"F{i} | {b['val']:.6g} ({b['src']}) | {b['note'][:60]}")


# ============================================================
# SECTION 5: ARCHITECTURE 1 — GP WITH ARD RBF (NEW)
# ============================================================
# ARD RBF: length_scale is a VECTOR of length d (one per dim).
# The GP learns a separate correlation length for each dimension.
# Short length_scale_i = dim i is highly relevant.
# Long  length_scale_i = dim i contributes little.
# Colleague: "automatic relevance determination (different
#             length_scales per dimension)"
# Reference: Rasmussen & Williams (2006), Chapter 5.1
# Y scaling added as colleague suggested ("scale the data").
# ============================================================

def fit_gp_ard_rbf(X, Y, n_restarts=15):
    """
    GP with ARD RBF kernel: per-dimension length scales.
    Y is standardised (zero mean, unit std) before fitting.
    Predictions are returned in original Y scale.
    ARD allows the GP to discover dimension importance automatically.
    """
    dim     = X.shape[1]
    y_mean  = float(Y.mean())
    y_std   = float(Y.std()) if Y.std() > 1e-12 else 1.0
    Y_s     = (Y - y_mean) / y_std

    scaler  = StandardScaler()
    X_s     = scaler.fit_transform(X)

    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))
        * RBF(length_scale=np.ones(dim),                # VECTOR — ARD
              length_scale_bounds=(1e-4, 1e4))
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=n_restarts, random_state=42
    )
    gp.fit(X_s, Y_s)
    return gp, scaler, y_mean, y_std


def predict_gp_ard(gp, scaler, X_np, y_mean, y_std):
    """Return predictions in original Y scale."""
    mu_s, sigma_s = gp.predict(scaler.transform(X_np), return_std=True)
    mu    = mu_s    * y_std + y_mean
    sigma = np.maximum(sigma_s * y_std, 1e-9)
    return mu, sigma


def get_ard_lengthscales(gp):
    """Extract per-dim length scales after fitting (diagnostic)."""
    for param_name, val in gp.kernel_.get_params().items():
        if 'length_scale' in param_name and hasattr(val, '__len__'):
            return np.array(val)
    return None


# ============================================================
# SECTION 6: ARCHITECTURE 2 — GP WITH ARD MATERN-2.5 (NEW)
# ============================================================
# Matern-2.5 kernel: twice differentiable functions.
# More appropriate than RBF for black-box functions that may
# have sharp features rather than infinite smoothness.
# Literature: BOOST (arXiv:2508.02332) — Matern > RBF for
# non-smooth objective functions in real-world BBO.
# Also includes ARD (vector length_scale).
# Y scaling applied identically to ARD-RBF.
# ============================================================

def fit_gp_ard_matern(X, Y, n_restarts=15, nu=2.5):
    """
    GP with ARD Matern-nu kernel.
    nu=2.5: twice differentiable (good general choice for BBO).
    nu=1.5: once differentiable (for sharper surfaces like F4).
    Y standardised before fitting; predictions returned in original scale.
    """
    dim    = X.shape[1]
    y_mean = float(Y.mean())
    y_std  = float(Y.std()) if Y.std() > 1e-12 else 1.0
    Y_s    = (Y - y_mean) / y_std

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))
        * Matern(length_scale=np.ones(dim),             # VECTOR — ARD
                 length_scale_bounds=(1e-4, 1e4),
                 nu=nu)
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=n_restarts, random_state=42
    )
    gp.fit(X_s, Y_s)
    return gp, scaler, y_mean, y_std


# ============================================================
# SECTION 7: ARCHITECTURE 3 — MC DROPOUT MLP (unchanged)
# ============================================================
# Gal & Ghahramani (2016): T=50 stochastic forward passes.
# Y standardised internally. Architecture scales with dim d.
# ============================================================

class MCDropoutMLP(nn.Module):
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
    y_mean = float(Y.mean()); y_std = float(Y.std()) if Y.std() > 1e-12 else 1.0
    Y_s    = (Y - y_mean) / y_std
    X_t    = torch.tensor(X,   dtype=torch.float32)
    Y_t    = torch.tensor(Y_s, dtype=torch.float32).reshape(-1, 1)
    model  = MCDropoutMLP(X.shape[1], dropout=dropout)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        nn.MSELoss()(model(X_t), Y_t).backward()
        opt.step()
    return model, (y_mean, y_std)


def mc_predict(model, X_np, y_mean, y_std, n_mc=50):
    model.train()
    X_t   = torch.tensor(X_np, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for _ in range(n_mc):
            preds.append(model(X_t).numpy().flatten())
    preds = np.array(preds)
    return preds.mean(axis=0) * y_std + y_mean, np.maximum(preds.std(axis=0) * y_std, 1e-9)


# ============================================================
# SECTION 8: ARCHITECTURE 4 — DEEP KERNEL LEARNING (unchanged)
# ============================================================

class DKLFeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        h = max(16, 4 * input_dim); self.h = h
        self.net = nn.Sequential(
            nn.Linear(input_dim, h), nn.ReLU(), nn.Linear(h, 2)
        )

    def forward(self, x):
        return self.net(x)


class DKLGPModel(gpytorch.models.ExactGP):
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
    y_mean = float(Y.mean()); y_std = float(Y.std()) if Y.std() > 1e-12 else 1.0
    Y_s = (Y - y_mean) / y_std
    scaler = StandardScaler(); X_s = scaler.fit_transform(X)
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
    for _ in range(epochs):
        optimizer.zero_grad(); loss = -mll(model(train_x), train_y)
        loss.backward(); optimizer.step()
    return model, likelihood, (y_mean, y_std), scaler


def dkl_predict(model, likelihood, X_np, y_mean, y_std, scaler):
    model.eval(); likelihood.eval()
    X_t = torch.tensor(scaler.transform(X_np), dtype=torch.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(X_t))
    return pred.mean.numpy() * y_std + y_mean, np.maximum(pred.stddev.numpy() * y_std, 1e-9)


# ============================================================
# SECTION 9: ACQUISITION FUNCTIONS
# ============================================================

def expected_improvement(mu, sigma, f_best, xi=0.01):
    """Closed-form EI (Jones et al. 1998). Surrogate-agnostic."""
    sigma = np.maximum(sigma, 1e-9)
    Z  = (mu - f_best - xi) / sigma
    ei = (mu - f_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei


def upper_confidence_bound(mu, sigma, kappa=0.8):
    """UCB = mu + kappa * sigma. Low kappa for exploitation."""
    return mu + kappa * sigma


# ============================================================
# SECTION 10: LIGHTWEIGHT ZoMBI (F1 and F4)
# ============================================================
# ZoMBI (Siemenn et al. 2023, npj Computational Materials):
# Zooming Memory-Based Initialization for needle-in-a-haystack.
# Two core steps implemented here:
#   ZOOM: per-dim bounds from top-m best inputs only
#   PRUNE: GP fitted only on top-m best observations
# This stops early bad queries biasing the GP mean.
# Applied to F1 (0/16 good obs) and F4 (3/35 above 0).
# ============================================================

def zombi_fit_and_bounds(X, Y, m=5):
    """
    ZoMBI lightweight implementation.
    1. PRUNE: select top-m observations by output value.
    2. ZOOM:  set per-dim bounds from min/max of top-m inputs,
              with a small buffer (20% of range, min 0.05).
    3. FIT:   GP fitted on pruned dataset (ARD Matern-2.5).
    Returns gp, scaler, y_mean, y_std, low_bounds, high_bounds.
    """
    m = min(m, len(Y))
    top_idx = np.argsort(Y)[-m:]
    X_top   = X[top_idx]
    Y_top   = Y[top_idx]

    # Zoom: per-dim bounds from top-m inputs
    dim     = X.shape[1]
    low_b   = np.zeros(dim)
    high_b  = np.ones(dim)
    for d in range(dim):
        lo  = X_top[:, d].min()
        hi  = X_top[:, d].max()
        buf = max((hi - lo) * 0.20, 0.05)
        low_b[d]  = max(0.0, lo - buf)
        high_b[d] = min(1.0, hi + buf)

    # Fit on pruned top-m data using ARD Matern-2.5
    gp, scaler, y_mean, y_std = fit_gp_ard_matern(X_top, Y_top, n_restarts=10, nu=2.5)
    return gp, scaler, y_mean, y_std, low_b, high_b


# ============================================================
# SECTION 11: LOO-CV — SELECT BEST SURROGATE (4 options)
# ============================================================

def loo_rmse_four(X, Y):
    """
    LOO-CV RMSE for all four surrogates:
      gp_ard_rbf, gp_ard_matern, mc_dropout, dkl.
    Lowest RMSE determines which surrogate generates the query.
    """
    loo = LeaveOneOut()
    sq = {"gp_ard_rbf": [], "gp_ard_matern": [], "mc_dropout": [], "dkl": []}

    for train_idx, test_idx in loo.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        Ytr, Yte = Y[train_idx], Y[test_idx]

        # ARD RBF
        try:
            gp1, sc1, ym1, ys1 = fit_gp_ard_rbf(Xtr, Ytr, n_restarts=5)
            mu1, _ = predict_gp_ard(gp1, sc1, Xte, ym1, ys1)
            sq["gp_ard_rbf"].append((float(mu1[0]) - Yte[0]) ** 2)
        except Exception:
            sq["gp_ard_rbf"].append((np.mean(Ytr) - Yte[0]) ** 2)

        # ARD Matern-2.5
        try:
            gp2, sc2, ym2, ys2 = fit_gp_ard_matern(Xtr, Ytr, n_restarts=5)
            mu2_s, _ = gp2.predict(sc2.transform(Xte), return_std=True)
            mu2 = mu2_s * ys2 + ym2
            sq["gp_ard_matern"].append((float(mu2[0]) - Yte[0]) ** 2)
        except Exception:
            sq["gp_ard_matern"].append((np.mean(Ytr) - Yte[0]) ** 2)

        # MC Dropout
        try:
            m_mc, (ym_mc, ys_mc) = train_mc_dropout(Xtr, Ytr, epochs=200)
            mu_mc, _ = mc_predict(m_mc, Xte, ym_mc, ys_mc, n_mc=20)
            sq["mc_dropout"].append((float(mu_mc[0]) - Yte[0]) ** 2)
        except Exception:
            sq["mc_dropout"].append((np.mean(Ytr) - Yte[0]) ** 2)

        # DKL
        try:
            m_dkl, lik, (ym_d, ys_d), sc_d = train_dkl(Xtr, Ytr, epochs=80)
            mu_dkl, _ = dkl_predict(m_dkl, lik, Xte, ym_d, ys_d, sc_d)
            sq["dkl"].append((float(mu_dkl[0]) - Yte[0]) ** 2)
        except Exception:
            sq["dkl"].append((np.mean(Ytr) - Yte[0]) ** 2)

    return {k: float(np.sqrt(np.mean(v))) for k, v in sq.items()}


# ============================================================
# SECTION 12: PER-FUNCTION CONFIGURATION
# ============================================================

# F5: asymmetric hard bounds confirmed by two consecutive weeks
f5_low  = np.array([0.000, 0.997, 0.997, 0.998])
f5_high = np.array([0.001, 1.000, 1.000, 1.000])

# F2: joint bounds anchoring dim1 closer to initial best
f2_low  = np.array([0.685, 0.910])
f2_high = np.array([0.730, 0.955])

method_config = {
    # F1: ZoMBI — near-flat NiaH; prune top-5, zoom bounds per dim
    "function_1": {"acq": "EI",   "xi": 0.0001, "zombi": True,  "restarts": 12},

    # F2: joint dim1+dim2 constraint; both need to be near initial best
    "function_2": {"acq": "EI",   "xi": 0.002,  "custom_bounds": True, "restarts": 12},

    # F3: tight GP, anchor to W5 best, pull dim3 back toward 0.466
    "function_3": {"acq": "EI",   "xi": 0.001,  "margin": 0.020, "restarts": 12},

    # F4: ZoMBI — classic NiaH; top-5 zoom+prune; Matern-1.5 for sharp peak
    "function_4": {"acq": "EI",   "xi": 0.001,  "zombi": True,  "restarts": 15,
                   "matern_nu": 1.5},

    # F5: ultra-tight asymmetric bounds; two consecutive new bests
    "function_5": {"acq": "EI",   "xi": 0.005,  "custom_bounds": True, "restarts": 15},

    # F6: back to GRID; EI failed W7; step 0.003 around W4 best
    "function_6": {"acq": "GRID", "step": 0.003, "restarts": 15},

    # F7: tight EI around W7 best; consecutive new bests W6+W7
    "function_7": {"acq": "EI",   "xi": 0.001,  "margin": 0.018, "restarts": 20},

    # F8: UCB kappa=0.8, tight margin around W7 best
    "function_8": {"acq": "UCB",  "kappa": 0.8, "margin": 0.030, "restarts": 20},
}


# ============================================================
# SECTION 13: LOO-CV SURROGATE SELECTION (all 8 functions)
# ============================================================

print("\n" + "=" * 70)
print("LOO-CV SURROGATE SELECTION (4 surrogates)")
print(f"{'Fn':<5} {'ARD-RBF':>11} {'ARD-Mat':>11} {'MC-Drop':>11} {'DKL':>11}  Winner")
print("-" * 70)

surrogate_selection = {}

for i in range(1, 9):
    key  = f"function_{i}"
    X    = updated_data[key]["X"]
    Y    = updated_data[key]["Y"]
    cfg  = method_config[key]

    rmse = loo_rmse_four(X, Y)
    winner = min(rmse, key=rmse.get)

    # F4 ZoMBI: force Matern-1.5 to respect sharp peak hypothesis
    if cfg.get("matern_nu"):
        winner = "gp_ard_matern"
        rmse["note"] = f"Matern nu={cfg['matern_nu']} forced for sharp peak"

    surrogate_selection[key] = {"rmse": rmse, "winner": winner}
    print(f"F{i:<4} {rmse['gp_ard_rbf']:>11.4f} {rmse['gp_ard_matern']:>11.4f} "
          f"{rmse['mc_dropout']:>11.4f} {rmse['dkl']:>11.4f}  {winner}")


# ============================================================
# SECTION 14: GENERATE QUERIES — ONE LOOP
# ============================================================

N_CANDS = 60000
week8_results = {}

for i in range(1, 9):
    key    = f"function_{i}"
    cfg    = method_config[key]
    X      = updated_data[key]["X"]
    Y      = updated_data[key]["Y"]
    dim    = X.shape[1]
    best   = all_bests[key]["inp"]
    margin = cfg.get("margin")
    winner = surrogate_selection[key]["winner"]

    print(f"\n{'=' * 62}")
    print(f"F{i} | {cfg['acq']} | {winner} | obs={X.shape[0]} | best={Y.max():.6f}")
    print(f"{'=' * 62}")

    # ── GRID for F6 ───────────────────────────────────────────
    if cfg["acq"] == "GRID":
        step = cfg["step"]
        gp, sc, ym, ys = fit_gp_ard_rbf(X, Y, n_restarts=cfg["restarts"])
        grid = []
        for deltas in product([-step, 0.0, step], repeat=dim):
            grid.append(np.clip(best + np.array(deltas), 0.0, 1.0))
        grid    = np.unique(np.array(grid), axis=0)
        mu_g, _ = predict_gp_ard(gp, sc, grid, ym, ys)
        bidx    = np.argmax(mu_g)
        query   = grid[bidx]
        score   = float(mu_g[bidx])
        mu_out, sigma_out = score, 0.0
        print(f"  Grid: {len(grid)} candidates | best mu: {score:.6f}")

    # ── ZoMBI for F1 and F4 ───────────────────────────────────
    elif cfg.get("zombi"):
        nu  = cfg.get("matern_nu", 2.5)
        gp, sc, ym, ys, low_b, high_b = zombi_fit_and_bounds(X, Y, m=5)
        print(f"  ZoMBI zoom: {np.round(low_b,3)} to {np.round(high_b,3)}")
        np.random.seed(42)
        cands  = np.random.uniform(low_b, high_b, size=(N_CANDS, dim))
        mu_s, sigma_s = gp.predict(sc.transform(cands), return_std=True)
        mu    = mu_s * ys + ym
        sigma = np.maximum(sigma_s * ys, 1e-9)
        f_best_pruned = Y[np.argsort(Y)[-5:]].max()
        acq_vals = expected_improvement(mu, sigma, f_best_pruned, xi=cfg["xi"])
        bidx     = np.argmax(acq_vals)
        query    = np.clip(cands[bidx], 0.0, 1.0)
        score    = float(acq_vals[bidx])
        mu_out   = float(mu[bidx])
        sigma_out = float(sigma[bidx])

    else:
        # ── Standard acquisition ──────────────────────────────
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

        if winner == "gp_ard_rbf":
            gp, sc, ym, ys = fit_gp_ard_rbf(X, Y, n_restarts=cfg["restarts"])
            mu, sigma = predict_gp_ard(gp, sc, cands, ym, ys)

        elif winner == "gp_ard_matern":
            nu = cfg.get("matern_nu", 2.5)
            gp, sc, ym, ys = fit_gp_ard_matern(X, Y, n_restarts=cfg["restarts"], nu=nu)
            mu_s, sigma_s = gp.predict(sc.transform(cands), return_std=True)
            mu    = mu_s * ys + ym
            sigma = np.maximum(sigma_s * ys, 1e-9)

        elif winner == "mc_dropout":
            model_mc, (ym_mc, ys_mc) = train_mc_dropout(X, Y, epochs=300)
            mu, sigma = mc_predict(model_mc, cands, ym_mc, ys_mc, n_mc=50)

        else:  # dkl
            m_dkl, lik, (ym_d, ys_d), sc_d = train_dkl(X, Y, epochs=200)
            mu, sigma = dkl_predict(m_dkl, lik, cands, ym_d, ys_d, sc_d)

        f_best = Y.max()
        if cfg["acq"] == "EI":
            acq_vals = expected_improvement(mu, sigma, f_best, xi=cfg["xi"])
        else:
            acq_vals = upper_confidence_bound(mu, sigma, kappa=cfg["kappa"])

        bidx      = np.argmax(acq_vals)
        query     = np.clip(cands[bidx], 0.0, 1.0)
        score     = float(acq_vals[bidx])
        mu_out    = float(mu[bidx])
        sigma_out = float(sigma[bidx])

    formatted = "-".join([f"{x:.6f}" for x in query])
    week8_results[key] = {
        "query":           query,
        "formatted_query": formatted,
        "acq":             cfg["acq"],
        "surrogate":       ("GRID" if cfg["acq"] == "GRID"
                            else "ZoMBI+" + winner if cfg.get("zombi")
                            else winner),
        "score":           score,
        "predicted_mean":  mu_out,
        "uncertainty":     sigma_out,
    }
    dist = np.linalg.norm(query - best)
    print(f"Query     : {formatted}")
    print(f"Score     : {score:.6f} | Pred mean: {mu_out:.6f} | Dist from ATB: {dist:.4f}")
    print(f"Bounds    : {np.round(low_b if cfg['acq'] != 'GRID' else best-cfg['step'], 3)}"
          f" to {np.round(high_b if cfg['acq'] != 'GRID' else best+cfg['step'], 3)}"
          if 'low_b' in dir() else "")


# ============================================================
# SECTION 15: VALIDATION
# ============================================================

print("\n" + "=" * 62)
print("QUERY VALIDATION REPORT")
print("=" * 62)

all_clear = True
for i in range(1, 9):
    key   = f"function_{i}"
    query = week8_results[key]["query"]
    surr  = week8_results[key]["surrogate"]
    issues = []
    if np.any(query < 0) or np.any(query > 1):
        issues.append("OUT OF RANGE")
    if np.all(query < 0.01) and i not in [5]:
        issues.append("SUSPICIOUS: all dims near 0")
    if np.all(query > 0.99):
        issues.append("SUSPICIOUS: all dims near 1")

    status = "OK" if not issues else "WARNING"
    print(f"\nF{i} [{status}] [{week8_results[key]['acq']}] [{surr}]")
    print(f"  Query : {week8_results[key]['formatted_query']}")
    if issues:
        all_clear = False
        for w in issues:
            print(f"  !! {w}")

print("\nAll queries valid." if all_clear else "\nReview warnings before submitting.")


# ============================================================
# SECTION 16: PROXIMITY CHECK
# ============================================================

print("\n" + "=" * 62)
print("PROXIMITY TO ALL-TIME BEST INPUT")
print("=" * 62)
for i in range(1, 9):
    key  = f"function_{i}"
    q    = week8_results[key]["query"]
    b    = all_bests[key]["inp"]
    d    = np.linalg.norm(q - b)
    surr = week8_results[key]["surrogate"]
    flag = "ok" if d < 0.20 else "far — review"
    print(f"F{i} [{surr}] | dist: {d:.4f} [{flag}]")


# ============================================================
# SECTION 17: ARD LENGTH SCALE DIAGNOSTIC
# ============================================================
# Print the per-dim length scales learned by the ARD kernel
# for each function. Short = dimension is important.
# Long = dimension contributes little.
# ============================================================

print("\n" + "=" * 62)
print("ARD LENGTH SCALE DIAGNOSTIC (shorter = more important dim)")
print("=" * 62)

for i in range(1, 9):
    key = f"function_{i}"
    X   = updated_data[key]["X"]
    Y   = updated_data[key]["Y"]
    try:
        gp, sc, ym, ys = fit_gp_ard_rbf(X, Y, n_restarts=5)
        ls = get_ard_lengthscales(gp)
        if ls is not None:
            ranked = np.argsort(ls)
            print(f"\nF{i} ({X.shape[1]}D): {np.round(ls, 4)}")
            print(f"  Most important dim: dim{ranked[0]+1} (length_scale={ls[ranked[0]]:.4f})")
            if len(ranked) > 1:
                print(f"  Least important  : dim{ranked[-1]+1} (length_scale={ls[ranked[-1]]:.4f})")
    except Exception as e:
        print(f"F{i}: could not extract ARD scales ({e})")


# ============================================================
# SECTION 18: PER-FUNCTION MODEL CARDS
# ============================================================

print("\n\n" + "=" * 65)
print("PER-FUNCTION MODEL CARDS — WEEK 8")
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
    r      = week8_results[key]
    cfg    = method_config[key]

    h1_mc  = max(32, 8 * dim); h2_mc = max(16, 4 * dim)
    p_mc   = (dim*h1_mc + h1_mc) + (h1_mc*h2_mc + h2_mc) + (h2_mc + 1)
    h_dkl  = max(16, 4 * dim); p_dkl = (dim*h_dkl + h_dkl) + (h_dkl*2 + 2)

    zombi_tag = " [ZoMBI: top-5 prune+zoom]" if cfg.get("zombi") else ""
    nu_tag    = f" Matern-{cfg.get('matern_nu', 2.5)}" if "gp_ard_matern" in winner else ""

    print(f"\n{'─'*65}")
    print(f"  MODEL CARD   F{i}  ({dim}D | {n_obs} obs){zombi_tag}")
    print(f"{'─'*65}")
    print(f"  All-time best  : {b['val']:.6g}  ({b['src']})")
    print(f"  Best input     : {np.round(b['inp'], 4)}")
    print(f"  Strategy       : {b['note'][:70]}")
    print()
    print(f"  ARCHITECTURES (NEW: ARD per-dim length_scales + Y-scaling)")
    print(f"    ARD-RBF      : ConstantKernel x RBF(d-dim ls) + WhiteKernel")
    print(f"    ARD-Matern{nu_tag}: ConstantKernel x Matern-nu(d-dim ls) + WhiteKernel")
    print(f"    MC-MLP       : Input({dim})->Lin({h1_mc})->ReLU->Drop(0.1)"
          f"->Lin({h2_mc})->ReLU->Drop(0.1)->Lin(1) | {p_mc}p | T=50")
    print(f"    DKL          : NN Input({dim})->Lin({h_dkl})->ReLU->Lin(2)"
          f" [{p_dkl}p] + RBF-GP")
    print()
    print(f"  LOO-CV RMSE")
    tags = {"gp_ard_rbf":"  SELECTED","gp_ard_matern":"  SELECTED",
            "mc_dropout":"  SELECTED","dkl":"  SELECTED"}
    for k in ["gp_ard_rbf","gp_ard_matern","mc_dropout","dkl"]:
        tag = tags[k] if k == winner else ""
        val = rmse.get(k, "n/a")
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        print(f"    {k:<14}: {val_str}{tag}")
    print()
    print(f"  WEEK 8 QUERY")
    print(f"    Acquisition    : {r['acq']} (xi={cfg.get('xi')} | kappa={cfg.get('kappa')})")
    print(f"    Surrogate      : {r['surrogate']}")
    print(f"    Query          : {r['formatted_query']}")
    print(f"    Predicted mean : {r['predicted_mean']:.6g}")
    print(f"    Dist from ATB  : {np.linalg.norm(r['query'] - b['inp']):.4f}")


# ============================================================
# SECTION 19: SAVE QUERIES
# ============================================================

rows = []
for i in range(1, 9):
    key  = f"function_{i}"
    Y    = updated_data[key]["Y"]
    r    = week8_results[key]
    rmse = surrogate_selection[key]["rmse"]
    rows.append({
        "Function":        f"Function {i}",
        "Query":           r["formatted_query"],
        "Acquisition":     r["acq"],
        "Surrogate":       r["surrogate"],
        "ARD_RBF_LOO":     round(rmse.get("gp_ard_rbf", 0),    4),
        "ARD_Mat_LOO":     round(rmse.get("gp_ard_matern", 0),  4),
        "MC_LOO":          round(rmse.get("mc_dropout", 0),     4),
        "DKL_LOO":         round(rmse.get("dkl", 0),            4),
        "Predicted_mean":  round(r["predicted_mean"],            6),
        "Uncertainty":     round(r["uncertainty"],               6),
        "Current_best":    round(float(Y.max()),                 6),
    })

df = pd.DataFrame(rows)
print("\n\nFINAL WEEK 8 QUERIES")
print(df[["Function", "Surrogate", "Acquisition",
          "Query", "Current_best"]].to_string(index=False))

df[["Function", "Query"]].to_csv("week8_queries.csv", index=False)
df.to_csv("week8_queries_full.csv", index=False)
print("\nweek8_queries.csv      — submit this file")
print("week8_queries_full.csv — LOO-RMSE and model card data")
