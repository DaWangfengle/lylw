# G7 frozen TUJI calibration: UTS + Tampere transfer stress test

**Research question, fixed before G7 scoring (2026-10-10):** The G6 TUJI1
training-only CAL-selected T=0.6, Student-t scale=0.5 yields joint
MAE and ES improvement on G6 TUJI1 internal CONFIRM. Is that
specific TUJI-derived calibration direction at least beneficial in
two different RSS fingerprint domains **without any target fitting or
additional architecture/model training**?

G7 is intentionally **short**: reuse original 5-member E5 model checkpoints
from G4 UTS (FIT-trained) and G5 Tampere (FIT-trained) rather than training
a further 10 networks. Fail if either local checkpoint panel is unavailable
or hashes differ from historical G4/G5 manifests/freeze. Do not restore
models from other runs, rebuild old training checkpoints, or substitute 25
member outputs. The old V17 training, R3 joint law and Student-t df4 stay
unchanged. The only intervention is G6 post-hoc final joint-law calibration.

## Data and independence disclosures

G4 UTS training file physical-coordinate-disjoint historical **CAL** and
**CONFIRM** are scored. G5 Tampere training-file historical **CONFIRM**
is scored. All groups were held out from training their own frozen E5
members. However G4/G5 internal labels and outcomes were already
observed in previous chats and experiments. So even G7's positive
result is **NOT a new independent prospective test, external blind
benchmark, nor proof of universal transfer**. G7 is an honestly named
retrospective, frozen-policy transfer *stress test*. Because the method
was inspired by G4/G5, do not use G7 p-values as independent discovery
evidence. No access to UTS_test, Tampere Test files, UJI official/outer
or TUJI results_test. No new official datasets or benchmark scores.

G4/G5 data schemas and their original physical-site group assignments
are reused **unaltered**. The new code reads canonical dataset TRAIN
files and original on-disk historical manifest/freeze, not G4/G5
previous outcome or original confirmation metric vectors. It verifies
all five saved weight hashes and original FIT anchor dictionary for
each domain.

## Four frozen policies, no selection

At final R3 E5/L192/K192 alpha1 joint Student-t law, use fixed
T=.6 / s=.5 source policy as **the only primary candidate**.
The alternatives exist only for mechanism dissection:
- T1_S1: exact original V17 mean and predictive law;
- T0.6_S1: sharpen joint component weights, keep scale;
- T1_S0.5: original weights, shrink Student-t scales;
- T0.6_S0.5: source-frozen combined policy.

Fixed random number stream seed19602, original G6 draw count 64
per stream, identical random numbers for all four variants, original
G6 scoring function reused literally. Compute macro Euclidean MAE,
original continuous-mixture Energy Score, device-row breakdown,
per-site paired vectors, nominal predictive radius and 50/80/90%
radial empirical coverage (diagnostic, NOT guaranteed calibration).
Same T/different scale **must** have identical exact mixture mean and
thus equal MAE. No G4 gamma, G4 gate, G5 geometric median and no
new temperature candidates/learned uncertainty gates.

## Predeclared descriptive decision

Treat UTS historical CONFIRM and Tampere historical CONFIRM as two
domain-level decisions (G4 UTS CAL is extra descriptive only).
The fixed source candidate needs **on each domain**:
- candidate minus baseline coordinate-macro MAE <= -0.15m;
- candidate minus baseline coordinate-macro ES <= 0.00m.

If all hold: TWO_DOMAIN_DESCRIPTIVE_TRANSFER_SIGNAL.
Otherwise: TRANSFER_NOT_ESTABLISHED.

These thresholds are a practical screening convention, **not** a
prospective 95% significance claim. No calibration, threshold update or
model choice is permitted from the UTS or Tampere outcomes. Neither
domain's result may promote T-only in place of primary T+s. No
p-hacking via additional G7 runs. One immutable plan and one
irreversible result lock before scoring, to avoid subsequent
repeated output selection. If lock is created and later scoring
fails, preserve error and lock and STOP; do not delete to retry.

## Interpretation and JCR Q1 next step

- Both domains pass: investigate *why* learned joint component weights
  need global/conditional correction, and test the actual multi-model
  vs anchor-mode uncertainty decomposition; then design a theoretically
  motivated train-only reliability predictor plus fresh independent
  domains/folds and competitive full E16/25 model performance.
- Only one passes: global transfer is not established. Seek
  domain-shift observables and *conditional* correction with explicit
  baseline fallback in new, unviewed training-development splits.
- Neither passes: stop global post-hoc temperature line. Refocus on
  underlying retrieval/posterior learning and absolute MAE.

Every scenario preserves G1-G6 original claims, including the G6
positive TUJI-internal result. A post-hoc power transform is not itself
JCR Q1-worthy novelty. Focus on falsifiable transfer and first
principles rather than more engineering, audits, or hyperparameter grids.

The user handles ALL Git branch switching, committing and upload/push.
Codex executes local scientific Python only. Checkpoints remain local.
