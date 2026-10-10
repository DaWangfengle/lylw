# G10: learned RSS/spatial evidence correction — retrospective pilot

**TRAIN ONLY:** Original TUJI + UTS CAL physical locations.
**No fitting:** Original E5 FIT rows, any CONFIRM rows, all Tampere rows.
**No independent confirmation:** All assessed domain confirmation
labels were seen in G4–G9; no p-values or method promotion.

| CONFIRM domain | Method | MAE m | ES m | P90 error m | P95 error m |
|---|---|---:|---:|---:|---:|
| tuji | base | 2.969113 | 2.102097 | 4.392200 | 4.860180 |
| tuji | anchor_t06 | 2.657774 | 1.867255 | 3.982890 | 4.466831 |
| tuji | evidence_correction | 2.609192 | 1.849013 | 3.862526 | 4.362310 |
| uts | base | 6.819208 | 4.786196 | 12.378209 | 13.963681 |
| uts | anchor_t06 | 5.107529 | 3.937965 | 9.447848 | 12.221708 |
| uts | evidence_correction | 4.962995 | 3.872844 | 9.151507 | 12.245514 |
| tampere | base | 7.202666 | 5.212627 | 12.911592 | 18.733934 |
| tampere | anchor_t06 | 5.987721 | 4.996576 | 8.911864 | 10.468628 |
| tampere | evidence_correction | 5.990838 | 4.981272 | 8.721600 | 10.550683 |

## Frozen Anchor-only contrasts

- tuji: ΔMAE -0.048582m; ΔES -0.018242m; MAE improve/harm 80/59; worst harm +0.6495m
- uts: ΔMAE -0.144534m; ΔES -0.065120m; MAE improve/harm 155/108; worst harm +2.5022m
- tampere: ΔMAE +0.003116m; ΔES -0.015304m; MAE improve/harm 46/66; worst harm +1.2890m

Four shared coefficients were learned only from two original CAL
roles using a fixed Dirac-mixture Energy Score + point-MAE + KL
surrogate. This train loss is NOT the continuous Student-t ES.
The three assessment domains were scored with paired continuous
Student-t 64+64 MC draws and original G8 sampling seed.
No validation selection, no adaptive temperature, no original
member-centre/scale update, no new blind holdout. A future truly
independent experiment is required before Q1/generalization claims.

Result self hash: 7fb4100c82c2f5d986fbb295b1375f8ad3e126d0470e5c8ce1b3b31bd25434f4
