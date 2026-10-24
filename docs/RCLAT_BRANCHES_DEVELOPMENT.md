# RCLAT branch development

These modules implement new research components. They do not replace a frozen runner, certify references, or authorise scientific fitting. The running dropout comparison and all closed outcomes remain unchanged.

## RCLAT-ASR

`bioacoustic_pipeline.representation.rclat_phonology` provides:

- `PhonologicalCTC(feature_names)`: the existing continuous Conv–BiLSTM encoder with character and phonological-feature readouts. Outputs are `(character_logits, feature_logits, output_lengths)`. Feature names come from an externally specified inventory. No hard phoneme segmentation is created.
- `joint_loss(character_loss, logits, targets, observed, lengths, weight, provenance)`: a supplied character objective plus weighted binary soft-reference cross-entropy, averaged over observed feature/frame cells. Missing targets may be NaN only where unobserved; padding may never be annotated. Weight zero returns the exact original character-loss tensor.

Active supervision requires a declared human or independently validated reference origin, independence from the acoustic model, source-use approval, and a source SHA256. These attestations are necessary metadata, not substitutes for a reference-validity audit. Model-generated labels are rejected. The target time grid must be independently supplied and audited against the encoder output grid; the loss never creates an alignment from its own predictions. No validated real phonological references were supplied in this implementation task.

The baseline character path is tested to remain bit-identical at matching initialisation. The auxiliary readout is new code, not a change to the frozen `SimpleCTC` source. The three-output interface requires a separately specified new trainer; it cannot be substituted into the live two-output runner.

## RCLAT-Communication

`bioacoustic_pipeline.diagnostics.rclat_communication` provides:

- `listener_targets`: empirical per-stimulus Request/Non-Request distributions, retaining listener identities for dependency analysis. Duplicate listener/stimulus judgments, missing stimuli and conflicting identity metadata fail. Production instructions are never used as targets.
- `canonicalize_responses`: explicit caller-supplied script-ID aliases, preserving source IDs and response values. The audited corpus uses `temp -> temperature` and `Spanish -> spanish`; these are recorded, not silently inferred.
- `fit_script_baseline`: a training-stimulus-only lexical baseline, with fixed positive smoothing and equal stimulus weighting. It never uses held-out stimuli in its estimate.
- `TemporalRequestModel`: one common C1/C2 GRU/readout architecture. C1 repeats the supplied whole-utterance summary on a fixed time grid; C2 receives the fixed-grid continuous trajectory. Both have identical parameterisation and lexical/global-summary access. C1 cannot see the fine values or time-varying masks. Returns logits, not probabilities.
- `speaker_macro_loss`: fixed-clipping binary log loss in nats, averaged over responses within stimulus, equally over stimuli within speaker, and equally over speakers. Response-rich stimuli do not dominate.
- `crossed_interval`: a paired crossed positive-weight bootstrap kernel. One Exp(1) listener weight applies to all that listener's judgments; an independent speaker weight applies to its stimuli. Stimulus response means are recomputed under listener weights, followed by the speaker-macro aggregation. Both models share weights. Caller must supply seed and replicate count; a future scientific protocol must freeze those choices and the interpretation before fitting.
- `readiness`: explicit independent-wording, source-use and methodology requirements. Even a positive status does not automatically authorise training.

The temporal model dimensions and fixed grids are configurable prototypes, not a final scientific architecture freeze. Real feature definitions, training-only scaler fitting, optimisation, selection, approved partitions, trial-history treatment and uncertainty use remain to be specified. Listener identity and production intent are not model arguments. Context/history is not invented or added by these components.

## TDD and commands

Tests were written and run before the new modules existed (red), then implemented to pass (green). Subsequent failing regressions covered malformed inventories, fractional lengths, identity mismatches, explicit aliases and the audit-command integration.

```bash
.venv/bin/python -m pytest -q \
  tests/test_rclat_phonology.py \
  tests/test_rclat_communication.py \
  tests/test_rclat_branches_preflight.py \
  tests/test_request_protocol.py \
  tests/test_rclat_simple_ctc.py \
  tests/test_rclat_ctc_learning.py \
  tests/test_rclat_dropout_ctc.py

.venv/bin/python scripts/rclat_branches_preflight.py --output /tmp/rclat_readiness_new
```

The audit only reads source metadata, judgments and review records; it processes no audio and fits no models. Existing nonempty reports are not overwritten. Original source and review packets are preserved.

The current audit found 432 stimuli, 8,856 responses, 82 listeners and 415 stimuli with listener disagreement. Independent wording remains unresolved for all432. Source reuse and final methodology remain uncertified. New scientific fitting remains blocked in both branches pending their respective prerequisites; P100/P960 and MACDE–CAVEM Stage2 remain blocked.
