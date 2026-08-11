# Week 7 Strategy

## Context
Week 6 produced 2/8 gains. DKL's discovery of F5's dim1-near-zero region validated data-driven surrogate selection. Colleague suggestions adopted this week: ARD kernels, Y scaling, fixed random_state.

## New this week
- ARD (automatic relevance determination) kernels: per-dimension length scales replacing shared scalar
- Y standardisation on all GP fits (colleague suggestion)
- ZoMBI lightweight memory pruning for F1 (needle-in-a-haystack)
- Matern kernel added as alternative to RBF

## Per-function decisions
| Function | Surrogate | Acquisition | Key change |
|---|---|---|---|
| F1 | ZoMBI+GP | EI xi=0.0001 | Widened margin 0.080 |
| F2 | DKL | EI xi=0.002 | Hard dim2 floor 0.92 |
| F3 | GP (forced) | EI xi=0.001 | MC Dropout overridden after W6 drift |
| F4 | GP | EI xi=0.001 | Micro-box margin 0.012 |
| F5 | DKL | EI xi=0.005 | Asymmetric bounds dim1 in [0, 0.003] |
| F6 | GP | EI xi=0.002 | Switched from Grid to EI |
| F7 | GP | EI xi=0.001 | Margin 0.020 around W6 best |
| F8 | GP | UCB kappa=0.8 | Margin 0.035 |

## Gains: 3/8 (F5, F7, F8)
