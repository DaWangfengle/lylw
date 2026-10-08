# V9 Reproduction Audit Note

The original design_v9.md referenced by the historical V9 registry is absent from the current repository. This new note exists only to satisfy the historical registry's requirement that a design-document path be present and hashed.

It does **not** redefine V9. The reproduction runner must call the constants, training functions, candidate fronts, scoring functions, member-extension rule, freeze logic, final refit, and outer-test evaluation directly from fhm_dapp_tim/fhm_dapp/uji_v9_experiment.py.

The only permitted adaptation is output-path isolation. No predictive parameter, seed, data role, metric, grid, or selection rule may be changed by this note.
