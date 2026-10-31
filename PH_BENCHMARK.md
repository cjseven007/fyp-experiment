# S005-informed pH benchmark

This additive workflow uses a quality-screened, regularly sampled pH segment to
inform a small synthetic benchmark. It does not infer a complete chemical cycle,
validate saturation, or send control commands. The existing `run_experiment.py`
workflow is unchanged. Its PCHIP reference is an estimate, not measured clean truth.

## Run

From the repository root, after installing `requirements.txt`:

```bash
python run_ph_benchmark.py --segment benchmark_inputs/S005.csv --output results/ph_benchmark_v1
python -m unittest discover -s tests -v
```

Use a new/empty output directory for each run. The input must contain timestamp,
ph_value and usable_ph, with all rows marked usable and 1-second intervals. This
initial implementation specifically checks the observed 0.01-pH recording grid.
It does not interpolate gaps. Input and output paths are explicit and portable.

## Empirical inputs versus assumptions

The runner recomputes complete 60-second windows from the input, using the existing
characterization analysis. It also exports 30/60/120-second sensitivity summaries.
The calibration JSON includes the source SHA256 for provenance.

Empirical: initial pH, median negative window slope, median positive residual SD,
median defined lag-1 residual correlation, observed recording grid. Excluding the
constant window from the residual-scale estimate and using negative slopes for
decline magnitude are explicit simulation construction choices, not unbiased
estimates of the full reactor process. Supplied characterization CSVs need not be
edited or trusted blindly: the runner recomputes them from the source segment.

Assumed: 20-minute signal length; slope magnitude multiplied by a uniform 0.75–1.25;
initial pH shifted by up to 0.04; event onset uniformly from 450–650 seconds;
gradual transition length 90–180 seconds; 90-second, 0.04-pH disturbance. These
are algorithm stress-test settings, not validated chemical parameters.

The five cases are stable, uninterrupted decline, abrupt flattening, gradual
flattening, and temporary disturbance. Only abrupt/gradual flattening have target
endpoint-like labels. The disturbance is a nuisance event and its alarms count
against endpoint specificity. A generic change detector may legitimately detect
that disturbance; distinguishing it requires additional decision logic/context.

Three observation conditions use AR(1) stationary SD multipliers 0, 1, 2 relative
to the positive residual SD median, and the observed residual lag-1 median as rho.
The AR(1) innovations have SD sigma*sqrt(1-rho²), with a stationary first value.
Signals are rounded once to the observed grid AFTER adding fluctuations. The zero
condition isolates rounding. This is a sensitivity model, not identification of
latent noise: residuals already include rounding, local curvature and process
variation. Nonzero scenarios can therefore exceed measured variability after
rounding. `synthetic_observed_diagnostics.csv` makes this mismatch inspectable.
No claim of matched instrument noise, independence, or true sensor resolution is made.

## Smoothing and selection

Reuse existing causal filters: raw; EMA spans 5/15/31/61; weighted causal SG windows
15/31/61, degree 1, oldest weight 0.2. Select one setting per family by mean
synthetic RMSE on development trials, weighting each scenario/noise condition
equally. Evaluate all families on test trials; do not retune from test outcomes.
Metrics exclude the common first 180 seconds. Transition RMSE uses onset through
transition end plus 180 seconds. RMSE on real data is not calculated.

Default: 8 development and 20 test realizations per scenario/noise condition:
120 development and 300 held-out synthetic traces. Seed namespaces are disjoint.
Shapes/timings as well as fluctuations vary between realizations. These test the
specified generator distribution, not unseen experimental conditions or independent
reactor runs. Repeating windows from S005 does not create experimental replication.

## Detection

All detectors receive ONLY causally smoothed observations. Non-overlapping 30-s
OLS slopes become available at the last timestamp in each block (29,59,...).
Normalize slopes by the source's negative-slope magnitude. Alarm timestamps are
availability times, not retrospectively backdated estimates of change location.

Baseline: absolute difference between median slopes in two adjacent 3-block
histories. Candidate thresholds 0.25/0.5/0.75/1/1.5 in normalized units.

BOCPD: exact unpruned Gaussian mean-change recursion with known observation SD and
normal mean prior. Mean prior -0.5, prior SD 1 in normalized units; hazard 0.05 per
30-second block. Observation SD is estimated separately per smoother from synthetic
development null-case slope errors with a 0.05 floor. It is a working assumption:
smoothing, correlated fluctuations and quantization can violate the likelihood.
Run length counts observations INCLUDING the current one. A reset occurs before
the current observation and uses the prior predictive; continuation branches use
their run-specific predictive distributions. The score is posterior mass at run
lengths 1 and 2. This indexing/model convention is explicit; do not substitute a
constant-hazard p(r=0) from a differently indexed recursion as an alarm score.
Thresholds 0.5/0.7/0.85/0.95/0.99; exact recursion has growing cost and is not yet a
bounded-memory SCADA implementation. First scores are necessarily high and ignored
for alarms during the common warmup.

Both methods use rising-edge alarms, a 180-s warmup and a 120-s refractory interval.
Select each threshold using development loss:

`2 * miss_fraction + false_alarms_per_trace + 0.25 * mean_detected_delay / 180`

If no targets are detected, the delay term is 0.25. This objective is an explicit
initial tradeoff, not a proven operational false-alarm requirement.

## Event scoring

One target per event scenario. Match the first alarm from onset through transition
end + 180 s. Alarms before onset, after that bound, and duplicate alarms are false.
Misses retain missing delay values, never zero delay. Gradual transitions are
intervals: export both onset-relative delay and excess delay beyond the end.
False-alarm rate uses exposure outside the matching interval after warmup; false
alarms inside it (duplicates) still count. Counts per trace are also available.
Always read recall and false alarms alongside conditional delay; delay alone can
make a detector that misses events look good. The 180-s matching tolerance is an
assumption; inspect raw alarm timestamps if a stricter tolerance is required.

## Outputs to inspect

- `source_context.png`, `channel_context.csv`: contextual channels, not an assertion
  that pH-only preparation validated them.
- `source_window_statistics.csv`, `window_sensitivity_summary.csv`: measured properties.
- `calibration.json`, `benchmark_config.json`: provenance and assumptions.
- `smoothing_tuning_summary.csv`, `detector_tuning.csv`: development selection.
- `selected_settings.json`: settings frozen before test scoring.
- `test_smoothing_summary.csv`, `test_detection_summary.csv`: stratified test results.
- `test_*_trials.csv`: individual trials and alarm times for independent aggregation.
- `traces/`: all held-out observations, clean targets, interval labels, filter outputs,
  block scores and alarms. Labels are for evaluation only.
- Five scenario PNGs: first test realization at multiplier 1 (fixed selection).
- `real_replay.csv`, `real_alarms.csv`, `real_replay.png`: exploratory real replay.
  There is no real endpoint label, hence no reported real detection accuracy.

## Validation and limitations

Tests verify causality under changes to future samples, slope units and reporting
times, Bayesian probabilities against enumeration of all short-sequence partitions,
event matching including early/duplicate alarms, reproducibility and target shapes.

This version is pH-focused. It does not fulfil a multivariate evaluation by itself,
simulate desorption, prove a chemical endpoint, or establish an autonomous switching
rule. Preserve operator actions and temperature/pressure context for interpretation.
Dropouts/freezes should be a separate quality-gate stress test, not silently passed
through these finite regularly sampled filters.

Statistical basis: Adams & MacKay (2007), Bayesian Online Changepoint Detection,
https://arxiv.org/abs/0710.3742 (run-length inference framework; implementation
conventions above are explicit); NIST median absolute deviation,
https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/mad.htm (robust
spread interpretation). Neither source validates these reactor-specific settings.
