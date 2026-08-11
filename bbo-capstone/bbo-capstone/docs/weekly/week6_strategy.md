# Week 6 Strategy

## Context
Weeks 5 produced only 2/8 gains due to search bounds drifting from the all-time best inputs. The anchor-to-ATB correction was validated in Week 4 and reinforced here.

## New this week
- Three surrogate options introduced: vanilla GP, MC Dropout MLP (Gal and Ghahramani 2016), Deep Kernel Learning (Wilson et al. 2016 via GPyTorch)
- Leave-one-out cross-validation RMSE used to select the best surrogate per function
- DKL discovered that F5 dim1 near zero is a qualitatively superior region, adding 507 output units in one step

## Per-function decisions
| Function | Surrogate | Acquisition | Outcome |
|---|---|---|---|
| F1 | GP | EI xi=0.001 | Declined |
| F2 | DKL | EI xi=0.002 | Declined |
| F3 | MC Dropout | EI xi=0.001 | Declined |
| F4 | GP | EI xi=0.001 | Declined |
| F5 | DKL | EI xi=0.005 | NEW ATB +507 |
| F6 | GP | Grid step=0.003 | Declined |
| F7 | GP | UCB kappa=0.8 | NEW ATB |
| F8 | GP | UCB kappa=0.8 | Declined |

## Gains: 2/8
