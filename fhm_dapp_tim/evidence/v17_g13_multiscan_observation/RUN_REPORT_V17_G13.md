# G13 | Multi-scan AP visibility × conditional RSS evidence

**Retrospective TUJI/UTS only; all CONFIRM labels previously exposed.**
No Tampere, no official UJIIndoorLoc test, no E5 retraining.
Only two pooled coefficients learned on original CAL.

| Domain | Policy | Macro MAE m | Full Student-t ES m | P90 m | P95 m |
|---|---|---:|---:|---:|---:|
| tuji | base | 2.969113 | 2.102127 | 4.392200 | 4.860180 |
| tuji | anchor_t06 | 2.657774 | 1.867334 | 3.982890 | 4.466831 |
| tuji | visibility | 2.625764 | 1.852270 | 3.929792 | 4.406701 |
| tuji | strength | 2.451960 | 1.697630 | 3.666968 | 4.230616 |
| tuji | joint | 2.425471 | 1.670775 | 3.606305 | 4.152300 |
| uts | base | 6.819209 | 4.786230 | 12.378239 | 13.963684 |
| uts | anchor_t06 | 5.107528 | 3.937673 | 9.447879 | 12.221708 |
| uts | visibility | 5.028225 | 3.915181 | 9.138622 | 11.797025 |
| uts | strength | 4.621529 | 3.695636 | 9.026525 | 11.785207 |
| uts | joint | 4.579336 | 3.689665 | 8.929872 | 11.819741 |

## FIT-scan support and learned evidence coefficients

- tuji: FIT scans per reference site min/median/max=1/5.0/6; sites with >=2 scans=642/643
- uts: FIT scans per reference site min/median/max=1/5.0/24; sites with >=2 scans=934/945
- learned coefficients: {'visibility': 0.8374806046485901, 'strength': 0.9469252228736877}

**PROMISING_RETROSPECTIVE_EVIDENCE** (DESCRIPTIVE screening, not promotion).

AP detection clipping at -1, conditional independence,
FIT scan correlation, heterogeneous devices and reused
historical CONFIRM outcomes limit causal/generalization claims.
UJIIndoorLoc needs a separate, unviewed prospective protocol.

Result SHA: ff1b5feb913125ee25efb7abe5c58ac55d4d82673f5ae7eb2a4b684b7ec5c179
