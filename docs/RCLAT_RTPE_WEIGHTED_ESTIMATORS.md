# RTPE weighted-estimator engineering verification

The weighted implementation lives in `src/bioacoustic_pipeline/representation/rclat_rtpe_weighted.py`. It extends RTPE without editing implementation-v1 sources, the verified inventory or running dropout dependencies. This is implementation/numerical readiness, not a scientific protocol freeze.

## Fixed population objective

For speaker s, recording r and selected origin i, the inventory supplies

`w_i = 1 / (number_of_speakers * recordings_for_speaker * origins_for_recording)`.

Within each split all w_i are positive and sum to one. Weights are fixed metadata, never learned or silently normalised. Let O_i be the observed target-cell set and D the total output-cell count (horizons times features). Both arms use

`data_loss = sum_i w_i * sum_(j in O_i) (prediction_ij - target_ij)^2 / |O_i|`.

Ridge adds `alpha / D * sum_j ||beta_j||²`, excluding all intercepts. Equivalently each target solves a weighted least-squares problem with effective row weights `w_i * D / |O_i|` on its observed rows and penalty `alpha * ||beta_j||²`. Those effective weights are not renormalised separately for each output. At complete support they reduce to w_i. Alpha is interpreted relative to a population-weight sum of one; it is not multiplied or divided by N automatically. This is an implementation convention, not a newly selected scientific alpha.

GRU data-loss parity does not imply identical model regularisation. Its optimiser/weight-decay policy must be frozen separately. For primary single-horizon evaluation, slice the 80ms target and its support before calling weighted_mse; averaging all horizons would be a different endpoint.

## Scaler and Ridge interface

`WeightedTrainingScaler.fit(training_inputs, input_valid, origin_weights)` uses weighted per-origin moments, averaging observed history cells equally within each origin and feature. A feature missing for an entire origin contributes no support for that feature; moments are normalised over supported origins. No feature can be wholly unsupported across training. Transform never updates moments and preserves missingness separately from zero-valued normalised inputs.

Fit one common scaler on training inputs only; both estimators consume its identical transformed inputs and targets. Applying its moments to future targets does not fit new target moments. Neither evaluation inputs nor targets may update moments. Constant scales are floored at 1e-8. For this verified inventory all mel cells are observed, but missing-support contracts remain tested.

`WeightedHistoryRidge(alpha).fit(scaled_inputs, input_valid, scaled_targets, target_valid, origin_weights)` does not fit a second scaler. It concatenates ordered history values with masks. The CPU solver uses float64, square-root row weighting, weighted centring to preserve an unpenalised intercept, Cholesky solves and grouped target-support masks. Exact constant design columns are removed: with a free intercept and positive coefficient penalty their optimal coefficients are zero. Identical target masks share one factorisation.

The solver exposes maximum_normal_residual and factorisations; these are engineering diagnostics. A small residual is not predictive utility. No evaluation-driven alpha search occurs inside fit.

## Neural loss and minibatches

`weighted_mse` consumes the full population and its normalised weights. `minibatch_mse` receives global population weights plus the global indices of the batch. If indices are sampled uniformly with replacement, its estimator is

`N / B * sum_b w[index_b] * origin_loss_b`.

This has the full-objective expected gradient. Do not divide by the sum of weights present in the batch. The function cannot verify that caller-supplied indices were sampled uniformly or correctly bound to actual input rows; the scientific runner must implement and record that policy. Averaging complete equal-size disjoint batches also recovers the population objective. Unequal-size batches require their corresponding B/N factors; an unweighted mean of unequal-size batch losses is incorrect.

Weights are validated in CPU float64; neural loss arithmetic follows the prediction dtype, preserving FP32 rather than requiring XPU float64. Gradient parity has been checked on CPU. XPU execution was not certified by this preflight.

## TDD and completed preflight

Seven new tests began with a missing-module failure and then passed. They compare the production solver with an independent augmented square-root-weighted SVD least-squares reference; verify masked population and minibatch gradients, speaker/recording weighting, weighted scaler moments, invalid inputs, constant designs and FP32 loss behaviour. The full related suite has 23 passing tests. Ruff passed for module, tests and preflight runner.

The frozen engineering runner is `scripts/rclat_rtpe_weighted_preflight.py`. Its recorded plan sets numerical tolerances, resource limits, dimensions and synthetic-only status before running. It reads only the frozen inventory's identities/indices/weights; no waveform, feature cache, real acoustic target or existing model state is used.

Completed results are in `outputs/rclat_rtpe_weighted_preflight_v1`:

- Independent SVD prediction relative error: approximately 3.02e-15.
- Manual neural full-objective gradient relative error: zero in this fixture.
- Minibatch vs full-objective gradient relative error: approximately 1.88e-16.
- Full synthetic normal-equation residual: approximately 2.64e-15.
- Inputs 15,840 x 32 x 64; targets 15,840 x 3 x 64; exact inventory origin weights.
- One factorisation for shared target support; 2,048 nonconstant columns.
- Peak process RSS approximately 1.826 GiB and total elapsed approximately 3.99 seconds on this CPU run. These are measurements, not runtime guarantees for actual acoustic conditioning, another process or XPU training.
- One synthetic Ridge fit and one finite GRU batch backward check, zero optimiser updates and zero real-data model fits.

## Remaining scientific prerequisites

Freeze actual model-selection and training budgets, alpha candidates under the stated convention, stochastic batching, optimisation/regularisation, checkpoint policy, numerical/resource failure rules, primary endpoint, paired speaker-level uncertainty and decision rule. Certify XPU arithmetic if that device is chosen. The new scaler must be fitted on the frozen training partition when scientific fitting is authorised; no old P10 scaler is reused. No need to rerun closed predecessor studies.

The inference population remains the frozen sampled-origin development population. Speaker resampling does not automatically include origin-sampling, model-seed, checkpoint-selection or source-selection uncertainty. The uncertified communication bootstrap is not reused. Temporal-order and behavioural objectives stay inactive; P100/P960 and MACDE–CAVEM Stage 2 remain blocked.
