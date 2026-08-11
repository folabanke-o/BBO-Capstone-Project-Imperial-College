# Week 9 Strategy

## Context
Week 8 was the best round with 5/8 gains. F1 broke through after seven failed weeks via ZoMBI. F2 finally beat its initial best. This week continues the GP Mean approach for all converging functions.

## New this week
- GP Mean applied to F1, F2, F3, F5, F7 (all just set new all-time bests)
- F4: pinned ZoMBI with EI xi=0 (Knowledge Gradient approximation per arXiv:2512.17569)
- F6: finest grid step 0.002 around W4 best

## Per-function decisions
| Function | Acquisition | Bounds | Outcome |
|---|---|---|---|
| F1 | GP Mean | Margin 0.015 | NEW ATB 7.62e-9 |
| F2 | GP Mean | [0.695,0.712] x [0.918,0.935] | Declined |
| F3 | GP Mean | Margin 0.015 | NEW ATB -5.1e-4 |
| F4 | Pinned ZoMBI + EI xi=0 | Top-5 memory | Declined |
| F5 | GP Mean | dim1 in [0.000, 0.002] | NEW ATB 4440 |
| F6 | Grid step=0.002 | Around W4 best | Declined |
| F7 | GP Mean | Margin 0.015 | NEW ATB 2.747 |
| F8 | EI xi=0.001 | Margin 0.020 | Declined |

## Gains: 4/8 (F1, F3, F5, F7)
