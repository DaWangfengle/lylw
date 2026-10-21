# Publication status: exploratory results with known limitations

Archived on 2026-10-21. These are the original DeepSeek exploratory scripts,
results, figures and available execution logs, accompanied by the
[independent three-structure review](../../docs/reviews/DEEPSEEK_THREE_STRUCTURE_REVIEW_20261021.md).

The archived results have not been corrected or rerun for publication. In
particular, the `1G+2A` combination in
[run_validation.py](../../fhm_dapp_tim/evidence/v17_gmrf_bandwidth_v1/run_validation.py)
does not apply the new graph propagation to its parent logits. Interaction
figures and H6/H7 claims derived from that branch must not be treated as valid
mechanism attribution. See the review for the synthetic counterexample,
aggregation inconsistencies, mathematical corrections and statistical limits.

Keep the original reports as historical evidence; their final-model and
significance claims are subject to the review. This archive does not establish
that the original three-structure method was repaired, disproved, or promoted.
The source scripts are retained for inspection, not as a recommendation to rerun
evaluation on already examined sites.

Publication checks: nine added Python scripts parse successfully; all five
saved JSON files parse without NaN/Inf. The original frozen source/protocol,
checkpoint and historical-result hashes match the previous integrity records.
These checks do not validate the exploratory scientific claims. No training or
new evaluation was performed as part of publication.

Model binaries, caches, ACL-repair material and write-probe scratch files are
excluded from this commit.
