# G8-A | Member vs anchor probability sharpening: descriptive attribution

**No training or new blind confirmation.** This uses already-viewed TRAIN-internal TUJI / UTS / Tampere roles. No p-values or model promotion.

All four variants preserve original member centre/Student-t scale;
T=0.6 is fixed, independently for member and anchor weights.
Both-level factorized temperature must equal G6 whole-joint T=0.6
up to documented floating-point tolerance.

| Domain / historical role | Method | Macro MAE m | ES m |
|---|---|---:|---:|
| tuji / calibration | base | 2.797312 | 1.969971 |
| tuji / calibration | member | 2.797118 | 1.985456 |
| tuji / calibration | anchor | 2.390619 | 1.713443 |
| tuji / calibration | both | 2.390663 | 1.714640 |
| tuji / confirmation | base | 2.969113 | 2.102097 |
| tuji / confirmation | member | 2.968962 | 2.081672 |
| tuji / confirmation | anchor | 2.657774 | 1.867255 |
| tuji / confirmation | both | 2.657763 | 1.867301 |
| uts / calibration | base | 7.155118 | 5.024129 |
| uts / calibration | member | 7.155526 | 5.007852 |
| uts / calibration | anchor | 5.380266 | 4.123892 |
| uts / calibration | both | 5.380405 | 4.140842 |
| uts / confirmation | base | 6.819208 | 4.786196 |
| uts / confirmation | member | 6.819075 | 4.772178 |
| uts / confirmation | anchor | 5.107529 | 3.937965 |
| uts / confirmation | both | 5.107270 | 3.954783 |
| tampere / confirmation | base | 7.202666 | 5.212627 |
| tampere / confirmation | member | 7.202235 | 5.152498 |
| tampere / confirmation | anchor | 5.987721 | 4.996576 |
| tampere / confirmation | both | 5.986884 | 4.968090 |

## Factorial decomposition (historical confirmations only)

- tuji mae_2d_m: member -0.000151m; anchor -0.311340m; both -0.311351m; interaction +0.000140m
- tuji energy_score_2d_m: member -0.020425m; anchor -0.234842m; both -0.234796m; interaction +0.020471m
- uts mae_2d_m: member -0.000133m; anchor -1.711679m; both -1.711938m; interaction -0.000126m
- uts energy_score_2d_m: member -0.014019m; anchor -0.848232m; both -0.831414m; interaction +0.030837m
- tampere mae_2d_m: member -0.000431m; anchor -1.214945m; both -1.215782m; interaction -0.000406m
- tampere energy_score_2d_m: member -0.060129m; anchor -0.216051m; both -0.244537m; interaction +0.031643m

Full per-coordinate effects and BOTH/BASE trace decompositions are in JSON.
Both alternative nesting orders are reported for variance; splitting
cross-level centre disagreement into unique causal shares is not justified.
Descriptive contrasts do not establish attribution by intervention in
the training process; they isolate the fixed R3 post-prediction operator.
Negative-transfer locations, ES worsening and historical exposure persist.

Self hash: 3ba8ddf35d212ae8d2e61e119727893a82d5fd284f293e8c178676e057466988
