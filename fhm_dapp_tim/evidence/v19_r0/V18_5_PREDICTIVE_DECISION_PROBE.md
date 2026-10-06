# V18.5 Predictive Decision Probe

Read-only post-formal diagnostic. No training, validation, or external test.

| cell | T | eps | scale | mean MAE | predictive spatial median MAE | delta spatial-mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 1.00 | 0.000 | 1.00 | 7.284263 | 7.366557 | 0.082295 |
| calibration_only | 1.25 | 0.000 | 1.25 | 7.384087 | 7.331935 | -0.052152 |
| selected_final | 1.25 | 0.100 | 1.25 | 7.166467 | 7.232601 | 0.066134 |

Negative delta means the predictive spatial median improves Euclidean MAE.

Probe SHA256: 6194488728c2dd999a508c1f94cae2f9eaeff8108e19461903c3ae93320b6970
