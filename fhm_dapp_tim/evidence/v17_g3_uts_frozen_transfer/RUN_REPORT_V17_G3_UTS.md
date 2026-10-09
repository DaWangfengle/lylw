# V17-G3: UTS one-shot fixed-G1 provided-test transfer

**Training input:** UTS_training.csv only; 25 original V17 R0 members.
**Evaluation:** one frozen provided UTS_test.csv attempt after immutable lock.
**Provenance:** test outcomes are user-attested unseen; adapter/schema was known.
**Not:** a claim that every file byte was never touched.

User attests UTS provided test labels/outcomes were never observed for method development; earlier schema/interface adapter checks are disclosed separately. This is one-shot, performance-blind-to-freeze provided-test evaluation, not proof of wholly unopened file bytes. UJI G1 historical STOP remains unchanged.

Protocol: v17_g3_uts_blind_external_fixed_geometry_transfer_20261008
Test scans: 388; test coordinate groups: 386

| Variant | Coordinate macro MAE (m) | Energy Score (m) | Row MAE (m) |
|---|---:|---:|---:|
| baseline | 7.684425 | 5.468975 | 7.687076 |
| g1_gamma_0p7 | 7.755909 | 5.627000 | 7.761380 |
| permuted_density | 7.702339 | 5.496777 | 7.701151 |

- G1 minus baseline macro MAE: +0.071484 m
- G1 minus baseline macro ES: +0.158025 m
- Paired familywise MAE upper: +0.159363 m
- Paired familywise ES upper: +0.205235 m
- Pre-registered G3 gate (including placebo MAE > baseline MAE): False; decision: TRANSFER_NOT_ESTABLISHED_ON_UTS
- The 25 cyclic E16 realizations share panel members and test coordinates; they are NOT independent replications.
- UJI G1 formal decision remains STOP regardless of these test outcomes.
- No post-test parameter selection or second attempt is authorized.

Immutable result self-SHA256: 8e9953ec4191676d1cc070bda757b83f1f5382703a39a2a4d0e4de97c1add0a0
