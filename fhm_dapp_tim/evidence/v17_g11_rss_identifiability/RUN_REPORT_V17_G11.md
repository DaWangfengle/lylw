# G11 | Can observed RSS distinguish the V17 wrong spatial modes?

FIT-only reference signals, fixed original five-member checkpoints.
Historical CAL/CONFIRM only. **Not prospective/independent test.**

| Domain/role | V17 Anchor MAE m | Raw RSS 1NN MAE m | Raw RSS 5NN MAE m | AP visibility 1NN MAE m | Raw 1NN@10m | V17 Top1@10m |
|---|---:|---:|---:|---:|---:|---:|
| tuji/calibration | 2.3906 | 4.2988 | 3.4211 | 4.9228 | 0.9124 | 0.9942 |
| tuji/confirmation | 2.6578 | 4.4857 | 3.7514 | 4.8721 | 0.9228 | 0.9847 |
| uts/calibration | 5.3803 | 6.2194 | 6.2115 | 7.8452 | 0.8180 | 0.8095 |
| uts/confirmation | 5.1075 | 6.1155 | 6.3114 | 6.7679 | 0.8285 | 0.8041 |
| tampere/confirmation | 5.9877 | 6.9039 | 5.7642 | 8.0986 | 0.9196 | 0.9107 |

## Paired 10m ranking quadrants (post-hoc true XY)

- tuji/calibration / both_near: 137 sites; V17 MAE 2.390619305136457; raw 1NN MAE 4.29876747348238
- tuji/calibration / raw_near_v17_far: 0 sites; V17 MAE None; raw 1NN MAE None
- tuji/calibration / raw_far_v17_near: 0 sites; V17 MAE None; raw 1NN MAE None
- tuji/calibration / both_far: 0 sites; V17 MAE None; raw 1NN MAE None
- tuji/calibration: AP-detection near when both raw and V17 far = 0
- tuji/confirmation / both_near: 139 sites; V17 MAE 2.657773677591384; raw 1NN MAE 4.485710541018326
- tuji/confirmation / raw_near_v17_far: 0 sites; V17 MAE None; raw 1NN MAE None
- tuji/confirmation / raw_far_v17_near: 0 sites; V17 MAE None; raw 1NN MAE None
- tuji/confirmation / both_far: 0 sites; V17 MAE None; raw 1NN MAE None
- tuji/confirmation: AP-detection near when both raw and V17 far = 0
- uts/calibration / both_near: 195 sites; V17 MAE 4.254713624137192; raw 1NN MAE 4.210300669405755
- uts/calibration / raw_near_v17_far: 16 sites; V17 MAE 9.202999619075353; raw 1NN MAE 4.980268197205057
- uts/calibration / raw_far_v17_near: 15 sites; V17 MAE 6.9009358254035; raw 1NN MAE 15.234599404629588
- uts/calibration / both_far: 32 sites; V17 MAE 9.614918628849326; raw 1NN MAE 14.85638543838955
- uts/calibration: AP-detection near when both raw and V17 far = 5
- uts/confirmation / both_near: 203 sites; V17 MAE 4.224383845164338; raw 1NN MAE 4.138612935687671
- uts/confirmation / raw_near_v17_far: 16 sites; V17 MAE 6.05516904373277; raw 1NN MAE 4.932281514346306
- uts/confirmation / raw_far_v17_near: 10 sites; V17 MAE 7.162670284214514; raw 1NN MAE 16.486048127941494
- uts/confirmation / both_far: 34 sites; V17 MAE 9.330025021816041; raw 1NN MAE 15.425624117918138
- uts/confirmation: AP-detection near when both raw and V17 far = 8
- tampere/confirmation / both_near: 98 sites; V17 MAE 4.609807017568385; raw 1NN MAE 5.397763556034647
- tampere/confirmation / raw_near_v17_far: 5 sites; V17 MAE 8.298344795794723; raw 1NN MAE 5.414213562373095
- tampere/confirmation / raw_far_v17_near: 4 sites; V17 MAE 8.813445265040436; raw 1NN MAE 11.624168273328472
- tampere/confirmation / both_far: 5 sites; V17 MAE 28.423634764474855; raw 1NN MAE 34.137519201095174
- tampere/confirmation: AP-detection near when both raw and V17 far = 0

## Scientific boundary

1. All three point baselines rank ALL original FIT anchors; no true
   floor/building filtering or CAL/CONFIRM parameter fitting.
2. Raw RSS L2 uses the EXACT same model-seen AP L2 prototype law
   already used upstream in R3: it is an information benchmark,
   not a novel independent sensor observation.
3. AP-visibility Brier recomputes the original FIT detection frequency
   (unused by frozen R0 retrieval) from the SAME RSS scans; values
   clipped to -1 are indistinguishable from missing AP.
4. Margin uses TRUE location, with positives <=10m same floor
   and clearly wrong >=20m or different building/floor.
   It is for hindsight diagnosis only, never a deployed gate.
5. A wrong 1NN does not establish Bayes irreducibility; a correct
   1NN does not prove V17 learned representation discards all
   useful RSS information. Neither is an oracle performance bound.
6. Deterministic RSS baselines are point estimates: continuous
   Student-t Energy Score cannot be fairly compared to them.
7. All observed site outcomes were seen in G4–G10, so new independent
   folds are necessary for Q1 generalization claims.

Result self hash: f414517883ef959f3160deffd7f67eddb12a41765795328a6ff03170d50e578d
