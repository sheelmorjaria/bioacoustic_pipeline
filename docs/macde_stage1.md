# MACDE continuous acoustic representation pilot

MACDE models sampled acoustic trajectories without assigning biological units to frames or latent states. It has a frequency convolution branch, a twelve-feature physical branch, a causal temporal network, a 64-dimensional state and state changes at nominal 20/50/100/200 ms. Native-rate 40 ms windows advance by 10 ms. Missing spectral bands, invalid samples, silence and unsupported physical features have separate masks. Context resets at unsupported frames.

The first study evaluates future physical acoustic measurements, not latent self-prediction or communicative meaning. R0 uses current/previous physical states; R0_long uses the same full history support as MACDE. R1 uses spectral information; R2 combines both branches; R3 adds a multiscale-change objective with an otherwise identical architecture. R3 must pass a validation-side gate. All arms share eligible sources, targets and date partitions. Scalers, neural training and Ridge selection exclude held-out recordings.

The independent detail probe uses full-waveform spectral centroid, bandwidth and amplitude modulation measurements on held-out synthetic waveform families. BirdPark transfer is descriptive because only two recording groups are available. Source material has been studied previously: this is a bounded pilot with a newly frozen split, not an untouched confirmatory corpus.

The companion controls compare an observed-prefix mean/standard-deviation summary with full ordered history and per-example shuffled older history. Shuffling preserves all frame values and the newest two states. Summaries never include audio after prediction time. Discrete biological-unit comparisons require authoritative references and are outside this study.

## Reproduction

Run from the repository root in a fresh artifact namespace. Commands deliberately refuse to overwrite a differing frozen audit/specification, and training has a one-run marker.

```bash
OMP_NUM_THREADS=2 .venv/bin/python scripts/macde_stage1.py --prepare
OMP_NUM_THREADS=2 .venv/bin/python scripts/macde_stage1.py --preregister
OMP_NUM_THREADS=2 .venv/bin/python scripts/macde_continuous_controls.py --preregister
OMP_NUM_THREADS=2 .venv/bin/python scripts/macde_stage1.py --run
OMP_NUM_THREADS=2 .venv/bin/python scripts/macde_continuous_controls.py
.venv/bin/python -m pytest tests/test_macde.py tests/test_macde_bundle.py tests/test_macde_controls.py tests/test_acoustic_derivatives.py -q
```

Source audit, prospective specifications, residuals and results live in `outputs/macde_*`. Final neural checkpoints live in `checkpoints/macde_stage1_R*.pt`; trained Ridge forecast probes are evaluated in the results, while the checkpoint also contains the neural training head. Source code and source/cache checksums are pinned in the audit/specification.

## Frozen inference

```python
from bioacoustic_pipeline.embeddings.macde_bundle import load_macde

encoder = load_macde('checkpoints/macde_stage1_R2.pt', expected_sha256=checkpoint_hash)
trajectory = encoder.encode_audio(waveform, sample_rate, sample_valid=sample_support)
```

`state` and `changes` contain NaNs where complete causal history is unavailable, accompanied by support masks. Physical forecast residuals are aligned to the future observation time and appear only after that waveform is observed. They are model-relative acoustic errors, not biological surprise or boundary annotations. Inference uses frozen training scalers; it never adapts to the supplied recording.

Original event boundaries remain observational metadata. They do not define internal units, are not model targets, and must not be interpreted in extracted-clip coordinates without independently verifying the source coordinate reference.

Stage 2 remains blocked until acoustic predictive/detail criteria and adequate independent transfer evidence are established. These artifacts do not establish receiver behaviour, intended recipients, comprehension, semantics or an absence of discrete communication units.

## Frozen pilot outcome

At 40 ms, on 357 eligible positions from 13 held-out dates, normalized RMSE was R0=0.5068, R0_long=0.5189, R1=0.5110, R2=0.4836 and R3=0.4857. R0 minus R2 was 0.0233, with paired date-bootstrap 95% interval [-0.0017, 0.0572]. The primary criterion therefore failed. R2 beat the full-history Ridge control by 0.0353 [0.0126, 0.0505]. Its synthetic detail error was 0.4265 versus R0's 0.4180, within the frozen 5% preservation allowance. The multiscale auxiliary passed its training-side inclusion gate but worsened held-out error relative to R2.

The prospective companion controls gave summary=0.5677, ordered=0.5189 and scrambled=0.5382. Summary minus R2 was 0.0841 [0.0545, 0.1112]. Scrambled minus ordered was 0.0193 [-0.0091, 0.0474], so this pilot does not establish an ordering benefit in the raw-feature history control.

Only one of the two BirdPark recordings supplied eligible forecast positions (57). R2 transfer error was 1.1549 versus R0=0.6927 and R0_long=0.4496. This weak descriptive transfer result does not support cross-corpus generalization. Stage 2 remains blocked, and no receiver or semantic model was fitted.

The R2 checkpoint was reloaded with checksum verification and tested on a real held-out waveform. All 32 targeted tests passed. Original start/end sample metadata for all 96 selected bat clips is preserved in `outputs/macde_source_event_metadata.json` without assuming its coordinate relationship to the extracted waveform.
