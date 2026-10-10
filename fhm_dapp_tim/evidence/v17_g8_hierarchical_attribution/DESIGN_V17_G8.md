# G8-A | Exact member-anchor temperature attribution

**Scientific freeze 2026-10-10, before G8 outcomes.** ONE small,
mechanistic, retrospective experiment. Reuse original G6 TUJI, G4 UTS,
G5 Tampere FIT-trained five-member E5 checkpoints. Historical CAL and
CONFIRM labels in all three domains have ALREADY been viewed. Do not
call G8 prospective, independent or blind external validation. No
provided test/official files. No retraining, search, or gate learning.

## Exact hypothesis

Original R3 dense joint mixture weight is member_weight times shared
anchor_consensus_probability: q[m,k] = w[m] * c[k], where the component
centres and Student-t scales remain member/anchor-specific. The
whole-joint temperature at T=.6 mathematically factorizes:

q_T[m,k] = softmax(log(w)/T)[m] * softmax(log(c)/T)[k].

This is exact in real arithmetic up to floating-point/clamp error.
Require per-batch original w*c reconstruction and independent
both-tempered vs original G6 whole-joint temperature comparison within
3e-6 abs/relative tolerance; otherwise STOP rather than silently
rewriting original posterior.

## Fixed 2×2 factorial, no search

All four variants preserve original E5/L192/K192/alpha1, member
weights/centres, Student-t scales df4, and G6 energy common randomness
seed19602, 2x64 draws per query.

| Variant | Member T | Anchor T |
|---|---:|---:|
| base | 1.0 | 1.0 |
| member | 0.6 | 1.0 |
| anchor | 1.0 | 0.6 |
| both | 0.6 | 0.6 |

BOTH is the exact G6 global **T-only, s=1** operator, NOT G7's s=.5
policy. Point MAE must be identical to G7 at fixed T if original
models/role/site grouping are unchanged. Energy Score differs with
Student-t scale; only compare within one common-RNG protocol.
Report the member/anchor main effects and interaction, for example:
I_MAE = (MAE_both - MAE_base) -
        (MAE_member - MAE_base) - (MAE_anchor - MAE_base).
No winner selection or new candidate promotions after viewing results.

## Total-variance trace identity

Coordinate-axis Student-t df=4 has variance 2 sigma² per dimension.
Record scalar trace in square metres (m²):
within = sum q_mk * 2 * (sigma_x² + sigma_y²);
between = sum q_mk * ||centre_mk - mean||².
A nested expansion conditioning on anchor first yields
between = between_anchor_of_member_mean + within_member_given_anchor.
Another valid expansion conditioning on member first yields
between = between_member_of_anchor_mean + within_anchor_given_member.
Validate both identities at query level in every batch. DO NOT label
one nesting as a uniquely identifiable causal decomposition: the
two levels interact through member-dependent anchor centres.
Store BASE and BOTH traces as physical-coordinate macro vectors.
Their relation to actual point error and ES is exploratory.

## Source and label isolation

TUJI uses original G6 TRAIN results_train MAT variable only, frozen
FIT/CAL/CONFIRM physical sites, and original .pt files matching G6 CAL
SHA256s. The MAT *container* physically includes results_test, but the
test variable is never requested/decoded. UTS uses original G4
TRAIN-only CAL/CONFIRM splits, member .pt SHA256s fixed in G7 from
original G4 CAL. Tampere uses original G5 TRAIN-only CONFIRM and
member .pt SHA256s in original G5 freeze. No scores from prior G4/G5
or G6 confirmation JSON are used to calculate G8 candidates.
No new model training. If old checkpoint missing, STOP instead of
retraining/reconstructing. G7 code and old protocols unchanged.

## Outcome and next decision

Five historically exposed role cohorts: TUJI CAL/CONFIRM, UTS
CAL/CONFIRM, Tampere CONFIRM. Score exact coordinate-macro
Euclidean MAE, Energy Score, all paired site error vectors, two
factorial effects and BOTH/BASE spatial variance trace vectors.
Report negative transfer and heavy tails, not just mean improvements.
No 95% gate or p-values on already-exposed cohorts. A follow-up
analysis can motivate a new physical domain-independent probability
reliability mechanism ONLY if one causal-operational level appears
consistently responsible, then validate it on genuinely untouched
development folds. Avoid engineering an adaptive gate prematurely.

A post-hoc temperature remains a strong baseline, not a JCR Q1
innovation. A Q1 method needs novel mathematical modelling plus
competitive absolute E16/E25 MAE, ES/tails, and independent multi-
dataset validation. User owns all Git operations; Codex owns local
scientific testing and execution only.
