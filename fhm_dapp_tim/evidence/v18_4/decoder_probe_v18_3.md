# V18.4 decoder mechanism probe on frozen V18.3

This is a read-only diagnostic on already-open V18.3 roles. It does not consume confirmation or official validation.

| Role | Decoder | Coordinate-macro MAE (m) | Improvement vs mean | Row MAE (m) |
|---|---|---:|---:|---:|
| early_stop | posterior_mean | 10.694250 | 0.00% | 10.525770 |
| early_stop | map_component | 4.396712 | 58.89% | 4.265161 |
| early_stop | top8_mean | 4.355324 | 59.27% | 4.315598 |
| early_stop | geometric_median | 4.805945 | 55.06% | 4.646433 |
| calibration | posterior_mean | 12.698292 | 0.00% | 12.187034 |
| calibration | map_component | 8.000909 | 36.99% | 7.523208 |
| calibration | top8_mean | 7.387286 | 41.82% | 6.881653 |
| calibration | geometric_median | 7.475349 | 41.13% | 7.004540 |

Decision: loss_aligned_geometric_median_supported

Use weighted geometric median as the registered V18.4 point decoder while keeping the full posterior for NLL/ES/uncertainty.

The predictive posterior is unchanged by this probe; decoder comparisons do not constitute new NLL/Energy-Score models.
