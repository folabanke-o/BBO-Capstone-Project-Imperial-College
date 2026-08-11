# Week 8 Strategy

## Context
Week 7 produced 3/8 gains. F5 and F7 each improved for the third time. F3 and F2 declined due to surrogate drift. This week introduced GP Mean acquisition for convergence exploitation.

## New this week
- GP Mean acquisition (arXiv:2103.16649): pure exploitation with no uncertainty penalty, applied to functions near confirmed peak
- ARD Matern-5/2 as the default kernel (arXiv:2409.00011 recommendation)
- Pinned ZoMBI for F4: W2 best input forced into memory regardless of rank

## Per-function decisions
| Function | Surrogate | Acquisition | Key change |
|---|---|---|---|
| F1 | ZoMBI+ARD-Matern | EI xi=0.0001 | Top-5 prune and zoom |
| F2 | ARD-Matern | GP Mean | Custom bounds [0.685,0.730] x [0.910,0.955] |
| F3 | ARD-Matern | GP Mean | Margin 0.015 |
| F4 | Pinned ZoMBI | EI xi=0.0 | W2 best forced into memory |
| F5 | DKL | GP Mean | dim1 in [0.000, 0.003] |
| F6 | ARD-Matern | Grid step=0.003 | Reverted to grid |
| F7 | ARD-Matern | GP Mean | Margin 0.018 |
| F8 | ARD-Matern | UCB kappa=0.8 | Margin 0.030 |

## Gains: 5/8 (F1, F2, F3, F5, F7) — best round of the project
