# Run V18 from scratch with Codex

Use branch `v18-single-member`. Do not inspect or use V9/V17 result JSON files
to tune V18. Do not open `data/UJIndoorLoc/validationData.csv` during the main
experiment.

## 1. Verify code before GPU work

```bash
python -m pytest -q tests/test_coherent_marked_v9.py tests/test_uji_v18_experiment.py
python -m py_compile fhm_dapp_tim/fhm_dapp/v18_single.py
python -m py_compile fhm_dapp_tim/fhm_dapp/uji_v18_experiment.py
```

Stop and fix code if any test fails. Do not weaken or delete a protocol test to
make it pass.

## 2. Register the experiment

Run once from the repository root:

```bash
python -m fhm_dapp_tim.fhm_dapp.uji_v18_experiment register
```

Inspect `fhm_dapp_tim/evidence/v18_single/registry_v18.json` only for protocol
sanity: role sizes, non-overlap, hull checks, hashes and frozen configuration.
Do not alter the registry after registration.

## 3. Train the selection model

On CUDA:

```bash
python -m fhm_dapp_tim.fhm_dapp.uji_v18_experiment train-selection --device cuda
```

This may use only `member_fit` for gradient updates and `early_stop` for
epoch selection. The command then refits exactly one checkpoint on
`selection_refit`.

## 4. Evaluate development and freeze

```bash
python -m fhm_dapp_tim.fhm_dapp.uji_v18_experiment freeze --device cuda
```

This writes the development result and immutable selection freeze. Do not tune
V18_CONFIG, seed, Top-K, loss weights, architecture or epoch count after seeing
the development metrics. If a code bug is found, invalidate the entire V18 run,
delete only V18-generated artifacts, fix the bug, and restart from registration.

## 5. Train one final model

```bash
python -m fhm_dapp_tim.fhm_dapp.uji_v18_experiment train-final --device cuda
```

This refits exactly one checkpoint on `outer_refit` using the frozen epoch
count. It must not evaluate outer-test during training.

## 6. One-shot outer test

```bash
python -m fhm_dapp_tim.fhm_dapp.uji_v18_experiment outer --device cuda
```

After this command, do not modify V18 and rerun the same outer test as a tuning
loop. Treat `outer_test_result_v18.json` as the one-shot V18 result.

## 7. Report

Summarize:

- selected epoch;
- selection/final anchor counts;
- development and outer coordinate-macro MAE;
- development and outer coordinate-macro mixture NLL;
- development and outer coordinate-macro Energy Score;
- building/floor errors;
- posterior entropy and radial scale;
- training time;
- test output and git commit SHA.

Also explicitly verify from the checkpoint/config that:

- one checkpoint is used at inference;
- `residual_experts == 1`;
- `scan_feature_mode == "bounded_maximal_invariant"`;
- `prototype_metric == "none"`;
- `candidate_reranker == True`;
- no consensus or ensemble function is invoked.

Do not use the official UJI validation file until a separately preregistered
external-audit protocol is written.
