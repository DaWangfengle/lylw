# G10 | Four-parameter RSS-evidence correction to the frozen V17 anchor law
**Fixed scientific protocol — 2026-10-10, before G10 outcomes.**

## Scientific question and falsifiability

G8 revealed nearly all Euclidean-MAE gains came from sharpening shared anchor probabilities. G9 found same-floor true-neighborhood candidates almost always present, but the strongest spatial modes were sometimes far from the true site. The simplest test of **extra deployable evidence** is whether one cross-domain, four-coefficient correction using only RSS, member agreement and spatial geometry outperforms the original anchor-only T=.6 law on MAE, full continuous Energy Score and hard tails *without retraining E5*.

This pilot cannot prove G9's posterior is wrong. ALL historical TRAIN-internal confirmation labels have already been viewed in G4–G9. They are **not blind/independent**. Prospective Q1 validation remains mandatory.

## One prespecified model

Keep original R3 E5 / member Top-L192 / alpha=1 consensus Top-K192 / Student-t df4 centres and scales. Original q[m,k]=w[m] c[k], member weights are unchanged.

Frozen comparator: q0_k = softmax_k(log(c_k)/0.6).

Single candidate: qtheta_k = softmax_k(log(c_k)/0.6 + Σ_j theta_j phi_j(r,k)), where theta_j = 1.5*tanh(v_j), and final continuous law Σ_mk w_m qtheta_k t4(y;mu_mk,Sigma_mk).

Four deployable features computed only from frozen R3 Top-K, per query and never from target labels:

1. Negative standardized RSS/prototype distance (rss_prototype_agreement).
2. Mean-centered fraction of member Top-L sets containing each selected anchor (member_topL_support).
3. Negative, per-query standardized across-member log-retrieval-probability disagreement (member_prior_log_agreement).
4. Log density-corrected shared consensus mass in a fixed same-building+floor 10m neighborhood, standardized within query (local_spatial_probability_coherence).

Building and floor are taken only from the **candidate anchor metadata**. True building/floor/XY labels must NEVER enter feature extraction. 10m came from G9 inspection, so the result is retrospective rather than independent discovery. Zero theta must exactly reproduce frozen anchor-only T=.6 probability and original G8 scoring.

## Exactly one training experiment

Train **once** on original TUJI G6 CAL (137 physical sites) and UTS G4 CAL (258 sites). No in-FIT, CONFIRM or Tampere data are fitted. Each physical site has equal weight within domain; TUJI and UTS each contribute half the total loss regardless of row counts. Tampere G5 CONFIRM has no target-domain coefficient fitting, but it is NOT an independent blind outcome because its labels were seen earlier.

Parameters: seed 20261010; initial raw theta all zero; Adam lr .06; 24 fixed epochs; minibatch24; coefficient cap ±1.5. No validation-based checkpoint selection.

The single frozen loss is:

L = [ES_Dirac(qtheta,y) + 0.30*EuclideanMAE(mean(qtheta),y)]/(10m)
    + 0.10*KL(qtheta || q0).

The training ES_Dirac is the differentiable **exact energy score of the discrete Dirac mixture at original member-averaged continuous component centers**. It is NOT the full continuous Student-t ES. It uses the true labels only in CAL training and does not update original member/anchor predictions. Do not claim this surrogate is calibrated for the full Student-t distribution.

At assessment, score original untempered Base, frozen Anchor-only T=.6, and the learned four-feature correction with exactly the same original R3 continuous Student-t law, 64+64 MC draws and G8 seed 19602. Save coordinate-macro MAE (m), continuous ES (m), site p90/p95 errors, all paired physical-site vectors, harm count and worst harm.

**Strict causal ordering:** lock first, collect TUJI+UTS CAL features only, fit and immutably save coefficients, THEN read any CONFIRM locations; do not use confirm to fit, select epochs, revise the feature set or set the hyperparameters. Original source/hash and all 15 old model checkpoints must match frozen prior G8/G9 provenance. Stop if lock exists after failure. Never rerun until a new predeclared design is created and independently justified.

## Interpretation and JCR Q1

A 4-param shared method that improves error, ES and tails in historically viewed TUJI/UTS/Tampere is promising **retrospective** evidence only, not Q1 evidence or prospective cross-domain success. If it loses to anchor-only T=.6 in meaningful domains, reject the simplistic correction; do not turn it into an adaptive temperature hyperparameter search. Future methods need unviewed spatially disjoint folds, proper E16/E25 absolute-MAE comparisons, full continuous ES, adverse-tail evaluation, and independent external replication.
