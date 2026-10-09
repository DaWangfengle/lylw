# V17 G5-A | Tampere TRAIN-only posterior action alignment

No provided external test used. One frozen five-member V17 predictive distribution; only point action changes.

| Point action | Coordinate-macro MAE (m) |
|---|---:|
| mean | 7.202666 |
| component_median | 6.680664 |
| continuous_median | 6.820973 |

Primary continuous-median minus exact-mixture-mean MAE: -0.381693 m
95% simultaneous maxT upper: +0.350248 m
Fixed minimum improvement: 0.15 m
Decision: **NO_INTERNAL_PROMOTION**

Secondary component-centre-median minus mean: -0.522002 m
Shared energy score: 5.271725 m
Energy Score is identical across point actions by construction, not an independent improvement.
Independent-draw expected predictive Euclidean risk: {"component_median": 13.292911399144197, "continuous_median": 13.32357838901054, "mean": 14.228222511952536}
Mean geometric-median point displacement: 4.524535 m

Both median algorithms use no query labels. Student-t continuous-median is Monte Carlo approximate;
the V9 code already implements sampling and Weiszfeld median—do not call this method novel.
All experimental comparisons use identical five V17 member weights and the E5/L192/K192/A1 posterior.
Do not compare absolute error with G3/G4's different panels/role splits.
Internal confirmation coordinates can be spatially dependent; external tests remain untouched.
Result self hash: 16d622c8604ad9dd1a79663b129767370e5357ce91db2d2a3477c2f394271c71
