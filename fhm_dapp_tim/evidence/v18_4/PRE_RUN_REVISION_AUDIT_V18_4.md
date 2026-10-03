# V18.4 pre-run mathematical revision audit

Status: **PASS**. Outcome-free diagnostics; no model selection.

Only UJI trainingData-derived fit/development roles and synthetic constructions were used.
No official validation or external test performance was accessed. No formal UJI protocol was run.

## CPU

Spatial loss: identical=0; near=0.08; far=19.28; finite nonzero logit-gradient norm=0.163659200856.

| r_radio | r_geometry | max epsilon | effective epsilon | realized KL | rho |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5 | 0.1 | 0.1 | 0.1 | 0.0961126302 |
| 0.5 | 0.25 | 0.1 | 0.05 | 0.05 | 0.0672935983 |
| 0.1 | 0.05 | 0.1 | 0.01 | 0.01 | 0.0300788202 |
| 0 | 0 | 0.1 | 0 | 0 | 0 |

Lower reliability did not increase realized KL for a fixed direction. Zero reliability preserved q bitwise; padding remained exactly zero.

Real-data smoke: fit=14126 rows/655 groups; development=2936 rows/139 groups; anchors=655; APs=520; s_C=10.4923105 m.

One 8-row forward/backward batch, zero model optimizer steps. All loss terms, model gradients, base and corrected posteriors were finite. Correction left model state bitwise unchanged and created no network gradients.

Training-batch loss terms: loss=15.5924089; anchor_nll=3.96330833; coordinate_nll=8.62640285; spatial_energy_loss=2.75937033; invariant_aux_loss=5.06356907; hierarchy_loss=1.81670558; domain_loss=2.77220201.

Residual bound=30 m; real effective-transform maximum singular value=30.0000016 m; synthetic oversized-transform cap and displacement checks passed.

Natural support sizes on 16 fit/development scans: [49, 55, 105, 51, 58, 67, 64, 48, 62, 64, 73, 54, 49, 44, 75, 44]; ceiling truncation fraction=0. All forced positive targets were preserved.

## CUDA

Spatial loss: identical=0; near=0.08; far=19.28; finite nonzero logit-gradient norm=0.163659200856.

| r_radio | r_geometry | max epsilon | effective epsilon | realized KL | rho |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5 | 0.1 | 0.1 | 0.1 | 0.0961126302 |
| 0.5 | 0.25 | 0.1 | 0.05 | 0.05 | 0.0672935983 |
| 0.1 | 0.05 | 0.1 | 0.01 | 0.01 | 0.0300788202 |
| 0 | 0 | 0.1 | 0 | 0 | 0 |

Lower reliability did not increase realized KL for a fixed direction. Zero reliability preserved q bitwise; padding remained exactly zero.

Real-data smoke: fit=14126 rows/655 groups; development=2936 rows/139 groups; anchors=655; APs=520; s_C=10.4923105 m.

One 8-row forward/backward batch, zero model optimizer steps. All loss terms, model gradients, base and corrected posteriors were finite. Correction left model state bitwise unchanged and created no network gradients.

Training-batch loss terms: loss=15.5280616; anchor_nll=3.93143725; coordinate_nll=8.62640285; spatial_energy_loss=2.78577856; invariant_aux_loss=5.01333237; hierarchy_loss=1.76425529; domain_loss=2.76631784.

Residual bound=30 m; real effective-transform maximum singular value=30.0000016 m; synthetic oversized-transform cap and displacement checks passed.

Natural support sizes on 16 fit/development scans: [49, 55, 105, 51, 58, 67, 64, 48, 62, 64, 73, 54, 49, 44, 75, 44]; ceiling truncation fraction=0. All forced positive targets were preserved.

## Revision semantics

A. Exact dimensionless half energy distance replaces iterative Sinkhorn; one lambda_spatial grid remains {0.10,0.25,0.50}.
B. Normalized reliability weights determine direction; max(r_radio,r_geometry) scales the registered maximum KL budget.
C. q -> temperature q_T -> trust correction -> final posterior. Fitting references stay frozen at training T=1; deployment energies/mismatches and KL use q_T. Scale calibration changes only component scales.
D. Local anisotropic Delaunay residual transforms are globally capped at 30 m, preserving their axes and singular-value ratio.

This checks numerical/design contracts on small samples; it does not establish localization accuracy, trained-model convergence, or exhaustive natural-support sufficiency.
Formal UJI prepare/develop/calibrate-freeze and all external formal evaluation remain pending review.
