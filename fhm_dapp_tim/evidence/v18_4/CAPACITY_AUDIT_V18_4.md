# V18.4 Role-Level Capacity Audit Evidence

Source: GitHub Actions dataset-audit run 37026901589 on the uploaded repository
datasets. The run completed successfully after the role-specific LODO test
update and reported `all_capacity_contracts_pass=true`.

Registered law:

- primary proposal quota before deduplication: 80
- computational support ceiling: 128

## Dataset roles

| Dataset | Role | Coordinate groups | Max positive non-self Delaunay degree | Required capacity | Margin to 128 |
|---|---:|---:|---:|---:|---:|
| UJIIndoorLoc | fit | 655 | 14 | 94 | 34 |
| UJIIndoorLoc | selection_refit | 794 | 16 | 96 | 32 |
| UJIIndoorLoc | full_training | 933 | 18 | 98 | 30 |
| UTSIndoorLoc | fit | 1026 | 15 | 95 | 33 |
| UTSIndoorLoc | selection_refit | 1246 | 17 | 97 | 31 |
| UTSIndoorLoc | full_training | 1466 | 19 | 99 | 29 |
| Tampere | fit | 312 | 10 | 90 | 38 |
| Tampere | selection_refit | 379 | 11 | 91 | 37 |
| Tampere | full_training | 446 | 12 | 92 | 36 |
| TUJI1 pooled | fit | 643 | 11 | 91 | 37 |
| TUJI1 pooled | selection_refit | 781 | 13 | 93 | 35 |
| TUJI1 pooled | full_training | 919 | 15 | 95 | 33 |

## TUJI1 leave-one-device-out

| Held-out device | Role seed | Role | Coordinate groups | Max positive degree | Required capacity | Margin |
|---|---:|---|---:|---:|---:|---:|
| A12 | 184451 | fit | 642 | 15 | 95 | 33 |
| A12 | 184451 | selection_refit | 780 | 15 | 95 | 33 |
| A12 | 184451 | full_training | 918 | 15 | 95 | 33 |
| POCO | 184452 | fit | 643 | 12 | 92 | 36 |
| POCO | 184452 | selection_refit | 781 | 14 | 94 | 34 |
| POCO | 184452 | full_training | 919 | 15 | 95 | 33 |
| S20 | 184453 | fit | 643 | 12 | 92 | 36 |
| S20 | 184453 | selection_refit | 781 | 12 | 92 | 36 |
| S20 | 184453 | full_training | 919 | 15 | 95 | 33 |
| S7 | 184454 | fit | 643 | 13 | 93 | 35 |
| S7 | 184454 | selection_refit | 781 | 15 | 95 | 33 |
| S7 | 184454 | full_training | 919 | 15 | 95 | 33 |
| tabS7 | 184455 | fit | 643 | 14 | 94 | 34 |
| tabS7 | 184455 | selection_refit | 781 | 17 | 97 | 31 |
| tabS7 | 184455 | full_training | 919 | 15 | 95 | 33 |

## Conclusion

No registered role approaches the support ceiling closely enough to require a
method change. The smallest audited margin is 29 slots on UTS full training.
For the roles actually used for gradient fitting and final refitting, the
smallest margins are 33 and 31 slots respectively.

This audit is outcome-free: it uses training coordinates and registered role
definitions only. It does not read or score reserved external test outcomes.
