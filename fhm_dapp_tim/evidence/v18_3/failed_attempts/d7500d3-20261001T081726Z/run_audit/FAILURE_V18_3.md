# V18.3 first attempt: implementation failure

The registered prepare succeeded. The first train-selection epoch completed training, but early-stop evaluation failed at `uji_v18_3_experiment.py:934`: a gradient-bearing diagnostic was converted directly to NumPy. No selection checkpoint, calibration result, confirmation lock/result, final checkpoint, or benchmark result was produced.

All scientific source files and the registered configuration remain unchanged. Confirmation content was never read after prepare, and the official benchmark was never read. Full stdout, phase timestamps, CUDA runtime checks, and I/O records are preserved in this directory.

The proposed correction detaches prediction tensors only at the evaluation-statistics boundary. Synthetic CPU and strict CUDA checks reproduce the original error and verify that the proposed correction completes finite statistics while preserving every prediction value bit for bit. The separate regression-test proposal exercises multi-batch evaluation and both correlations. Neither proposal has been applied to repository source.

The supplied instruction allows implementation fixes only before formal prepare. Applying this fix and registering a fresh attempt therefore requires the user's explicit exception. No scientific phase will be restarted without that authorization.
