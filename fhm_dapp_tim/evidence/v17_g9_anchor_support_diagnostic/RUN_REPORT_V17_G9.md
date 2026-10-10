# G9 | Anchor candidate survival and rank-vs-support diagnosis

Retrospective FIT-only frozen E5 inference on previously exposed roles.
**No new blind holdout; no fitted reliability model.**

| Domain / historical role | Geometric FIT@5m | Top-L@5m | Consensus@5m | Base mass@5m | Anchor-T mass@5m | Base MAE m | Anchor-T MAE m |
|---|---:|---:|---:|---:|---:|---:|---:|
| tuji/calibration | 1.0000 | 1.0000 | 1.0000 | 0.5641 | 0.6618 | 2.7973 | 2.3906 |
| tuji/confirmation | 1.0000 | 1.0000 | 0.9986 | 0.5456 | 0.6329 | 2.9691 | 2.6578 |
| uts/calibration | 0.9380 | 0.9327 | 0.9276 | 0.1891 | 0.3482 | 7.1551 | 5.3803 |
| uts/confirmation | 0.9734 | 0.9670 | 0.9658 | 0.2184 | 0.3931 | 6.8192 | 5.1075 |
| tampere/confirmation | 0.9821 | 0.9821 | 0.9821 | 0.3046 | 0.4855 | 7.2027 | 5.9877 |

## Interpretation boundary

The full FIT dictionary is an *anchor-geometry reference*, not a Bayes-optimal predictor or full continuous model support.
Top-L union and consensus Top-K describe selected anchor IDs, not residual-adjusted continuous centres.
Both geometric anchor support and learned residual-centre support are saved separately, with floor/building matching by anchor ID.
A correct-neighbourhood mass is defined with held-out location labels and can NEVER be used as a deployable gating feature.
Large mass of a wrong anchor does NOT by itself identify why the retrieval model assigned it weight.
All results are descriptive for historical roles; no promotion, no new optimizer, no test access, no p-value.

Self hash: 5d5dfd911e56ff13f3d6335ff05275c0b2caad182ddfc563b514ae210c4984bd
