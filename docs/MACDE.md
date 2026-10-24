# MACDE — Multiscale Acoustic Change and Detail Encoder

Repository reference, compiled 21 October 2026. This document describes the implemented method, saved experiments and their limitations. Frozen protocols and machine-readable results remain authoritative. Historical studies are preserved; this reference does not reopen them.

Related references: [CAVEM](CAVEM.md), [RCLAT](RCLAT.md), [original Stage 1 record](macde_stage1.md), [original transfer protocol](macde_transfer.md).

## Purpose and current status

MACDE learns continuous acoustic states and changes without requiring syllable or word boundaries. Its tested forecasting task predicts future physical acoustic measurements from supported past audio. Forecasting utility, acoustic preservation, biological interpretation and statistical independence are separate requirements.

| Branch | Recorded outcome |
|---|---|
| Original Stage 1 | Primary comparison failed; secondary full-history comparison positive |
| Bengalese within-bird development | Positive forecasting result on a familiar bird's held-out day |
| Frozen cross-bird transfer | Heterogeneous results; R2 lost to short-history Ridge on all three unseen birds |
| Multi-bird training | Completed; R2-multi lost to short-history Ridge in all four outer folds |
| Source/session certification | Earlier acquisition candidates did not establish certified independent groups |
| Lossless coding | Exact acoustic reconstruction demonstrated; no semantic inference |
| MACDE–CAVEM bridge | Mechanical compatibility tested with synthetic audio |
| Downstream Stage 2 | **Blocked** |

The strongest supported interpretation is conditional acoustic forecasting utility. General-purpose cross-individual superiority, independent-session transfer and communicative semantics remain unestablished.

## Acoustic input and validity

The frontend operates at the source sampling rate, with 40 ms windows advancing every 10 ms. Windows are causal: no centred padding or future audio is supplied as history. At integer sample rates, the implementation rounds the hop and uses a window of four hops.

Inputs combine:

- 64 fixed log-frequency spectral bands, with edges spanning 25–125,000 Hz. Bands unsupported by the source Nyquist frequency are masked.
- Twelve physical measurements: centroid, spread, rolloff, RMS, peak, crest, zero-crossing rate, entropy, flatness, flux, dominant frequency and dynamic coefficient of variation.
- Explicit sample, activity, spectral, physical-measurement and temporal-support masks.

The activity threshold is an absolute RMS threshold of `1e-3`. Peak-normalising quiet material to manufacture support is outside this contract. Feature standardisation is fitted on training data only. Unsupported values receive neutral numerical fills accompanied by validity indicators.

Recording boundaries and unsupported spans reset usable history. Files are never concatenated to create artificial continuity. An eligible position requires supported history and future target measurements, not merely a sufficiently long file.

The audit's 128-frame requirement covers approximately:

```text
40 ms + 127 × 10 ms = 1.31 seconds
```

Five seconds is not the original minimum. Forecast horizon and context requirements impose additional support constraints. Technical support does not establish vocal-source attribution or statistical independence.

Implementation: [MACDE encoder](../src/bioacoustic_pipeline/embeddings/macde.py), [data preparation](../src/bioacoustic_pipeline/diagnostics/macde_data.py).

## Architecture

```text
Supported waveform
       │
       ├── masked spectral bands ── spectral convolutional branch ──┐
       └── masked physical features ── physical branch ─────────────┤
                                                                  ▼
                                                        64-dimensional state
                                                                  │
                                                   causal residual temporal blocks
                                                                  │
                                            acoustic states + multiscale changes
                                                                  │
                                                   future physical-feature head
```

The spectral branch processes values and masks through two frequency convolutions, pooling and a 32-dimensional projection. The physical branch maps twelve values and twelve indicators to 32 dimensions. Their combined representation is projected to a 64-dimensional state.

Five causal residual blocks use dilations 1, 2, 4, 8 and 16, giving a nominal 32-frame state history. Change vectors compare the current state with states 2, 5, 10 and 20 frames earlier: approximately 20, 50, 100 and 200 ms. Combining the longest lag with the state history requires 52 frames of context.

The main head predicts twelve future physical measurements. A separate auxiliary head can use multiscale changes. Auxiliary objectives belong to their declared experimental arms; they are not silently added to R2. Targets are detached physical measurements rather than the model's own future latent states.

## Tasks, comparators and interpretation

| Arm | Role |
|---|---|
| R0 | Recent two-state physical-feature Ridge |
| R0_long | Full-history physical-feature Ridge |
| R1 | Spectral MACDE arm in original Stage 1 |
| R2 | Combined physical and spectral MACDE |
| R3 | Combined MACDE with auxiliary multiscale-change objective |
| R2-shuffled | Separately specified neural order-disruption control |

The original horizons are 4, 8 and 20 hops: 40, 80 and 200 ms. The primary 40 ms target avoids waveform-window overlap with the current 40 ms input window.

Comparisons must use identical forecasting positions, targets, horizons and training-side scaling. Normalized errors from different studies are not automatically comparable because the fitted normalization populations can differ.

For an error metric, a positive contrast

```text
Δ = error(comparator) − error(MACDE)
```

favours MACDE. A confidence interval conditional on recordings within one bird-day does not quantify variation across days, birds or populations.

## Original Stage 1: closed result

The original pilot used 96 selected bat clips. Source origins were retained without assuming that annotation offsets referred directly to acquired extracts. The primary held-out population contained 357 eligible positions across 13 dates.

| Arm | Held-out normalized RMSE ↓ |
|---|---:|
| R0 | 0.5068 |
| R0_long | 0.5189 |
| R1 | 0.5110 |
| R2 | 0.4836 |
| R3 | 0.4857 |

The preregistered R0-minus-R2 contrast was 0.0233, with date-bootstrap 95% CI `[-0.0017, 0.0572]`: the primary criterion failed. The secondary R0_long-minus-R2 contrast was 0.0353, CI `[0.0126, 0.0505]`.

An independent synthetic detail probe measured centroid, bandwidth and amplitude modulation. R2 error was 0.4265 versus 0.4180 for R0, within the frozen 5% allowance. This was a detail-preservation probe, not waveform reconstruction.

Companion controls found a whole-prefix summary error of 0.5677, ordered physical-history error of 0.5189 and shuffled older-history error of 0.5382. The shuffled-minus-ordered contrast was 0.0193, CI `[-0.0091, 0.0474]`, so that raw-feature order comparison was inconclusive.

A descriptive BirdPark check had only one eligible recording and 57 positions. R2 error was 1.1549 versus 0.6927 for R0 and 0.4496 for R0_long. It did not establish transfer.

Evidence: [preregistration](../outputs/macde_stage1_preregistration.json), [results](../outputs/macde_stage1_results.json), [continuous controls](../outputs/macde_continuous_controls_results.json).

## Original cross-corpus successor: no fitting

The successor considered marmosets, tree pipits and little owls. Its frozen support gate required at least four fit groups, two validation groups and three held-out groups per domain, with sufficient eligible positions.

| Domain | Fit groups / positions | Validation | Held-out |
|---|---:|---:|---:|
| Marmosets | 12 / 382 | **1 / 93** | 5 / 281 |
| Tree pipits | 6 / 684 | 2 / 228 | **2 / 228** |
| Little owls | 4 / 456 | 2 / 228 | 3 / 285 |

The full gate failed. No transfer models were fitted or scored. Grouping proxies used in a support inventory were not independently certified sessions.

Evidence: [source audit](../outputs/macde_transfer_source_audit.json), [frozen protocol](../outputs/macde_transfer_preregistration.json).

## Source acquisition and provenance

The acquisition programme separated acquired files, technical acoustic support, authoritative lineage and certified independence. Its closed inventory certified zero independent groups.

| Candidate | Closeout status | Missing evidence |
|---|---|---|
| MarmAudio | Blocked | Verified example-to-parent, offset and session crosswalk |
| CEB tree pipit | Resolvable | Authoritative metadata for retained Xeno-canto IDs |
| Petrusková tree pipit | Blocked | Blind-test crosswalk and date reconciliation |
| Zhao marmosets | Resolvable | Verified audio, subject and session relationships |
| InfantMarmosetsVox | Blocked | Session documentation and prospective sampling contract |

“Resolvable” means that a specific recovery route exists; it does not mean eligible or certified. The limitation concerns inspected materials and available metadata, not proof that the collections fundamentally lack independent recordings.

MarmAudio's six selected examples passed acoustic checks, but parent relationships remained unresolved. CEB had 35 passing files out of 38 focal recordings; metadata access failures were preserved rather than interpreted as absent metadata. Petrusková had 30 passing WAVs but unresolved recording identities and date conflicts. InfantMarmosetsVox retained explicit records for two missing annotation-referenced files.

Evidence: [acquisition closeout](../outputs/macde_acquisition_closeout.json), [decision matrix](../outputs/macde_acquisition_decision_matrix.csv), [certification ledger](../outputs/macde_source_certification_ledger.json).

### Gombe

Two audited Gombe tapes supplied technical acoustic support: Adults 3 had 3,516 eligible positions and a 12.21-second longest valid run; Adults 4 had 9,208 positions and an 8.93-second run. The combined 12,724 positions are not independently attributable chimpanzee observations: commentary and mixed sources remain relevant.

The catalogue audit retained 606 records against the publication's reported 605. Seven tape-start annotations supplied partial tape associations, but zero sample intervals were verified against catalogue identities. Recording-event and independent-group assignments therefore remained uncertified.

A cut boundary, a catalogue assignment and session independence are three separate verification tasks. Row order, dates and tape starts cannot supply missing sample coordinates.

Evidence: [candidate audit](../outputs/gombe_candidate_audit.json), [interval-linkage audit](../outputs/gombe_interval_linkage_audit.json), [linkage table](../outputs/gombe_interval_linkage_table.csv).

## BFSongRepo: corrected population inventory

BFSongRepo contains amplitude-triggered song-bout recordings from isolated Bengalese finches. Each file is a separate sequence. Bird-day containers preserve useful provenance but are not automatically independent sessions.

The latest saved inventory supersedes intermediate counts of 163 or 441 recordings. It contains **552 recordings, six bird-days and four birds**, with **764,700 technically eligible forecasting positions**.

| Bird | Recordings | Technical positions |
|---|---:|---:|
| gy6or6 | 39 | 38,103 |
| or60yw70 | 235 | 296,884 |
| bl26lb16 | 75 | 189,553 |
| gr41rd51 | 203 | 240,160 |
| Total | **552** | **764,700** |

The or60yw70 total includes an additional 111 recordings and 143,186 positions from 30 September 2012, omitted from an earlier summary. WAV/CBIN correspondence and source-date checks support format and provenance consistency, not session independence.

The multi-bird study used a bounded population of **16 selected positions per recording: 8,832 positions**, rather than fitting on all 764,700 inventory positions. Biological replication remains four birds.

Prior Bengalese corpus exposure must be disclosed. Syllable annotations were not required as forecasting targets, and no denoising was introduced.

## Within-bird development: closed positive result

The development population included four bird-days and two birds. On the protected day from the familiar bird or60yw70, evaluation used 51 recordings and 816 selected positions.

| Arm | Held-out normalized RMSE ↓ |
|---|---:|
| R0 | 0.6566 |
| R0_long | 0.6165 |
| R2 | **0.5391** |
| R2-shuffled | 0.6165 |

R0_long-minus-R2 was 0.0773, recording-bootstrap CI approximately `[0.0548, 0.1009]`. The separately trained order-disrupted arm was also worse: shuffled-minus-ordered 0.0774, CI `[0.0481, 0.1068]`. Each neural arm used its specified ordering policy during training and evaluation, preserving the newest two states.

These intervals are conditional on the evaluated bird-day. They do not establish reproducibility across independently sampled days. Evaluation on gy6or6 was negative against both Ridge baselines.

Evidence: [development results](../outputs/bfsongrepo_development_results.json), [closeout](../outputs/bfsongrepo_development_closeout.json).

## Frozen cross-individual transfer: closed mixed result

This study transferred the existing end-to-end predictor without fitting an encoder or a new prediction head on target birds.

| Unseen bird | R0 | R0_long | R2 | R0_long − R2 |
|---|---:|---:|---:|---:|
| gy6or6 | 0.8615 | 0.8939 | 0.9363 | −0.0424 |
| bl26lb16 | 0.9697 | 1.2233 | 1.0068 | +0.2164 |
| gr41rd51 | 0.9465 | 1.0207 | 0.9983 | +0.0224 |

Conditional intervals for the final column were `[-0.0653, -0.0192]`, `[0.1721, 0.2624]` and `[0.0102, 0.0351]`, respectively.

R2 beat full-history Ridge on two birds but lost to short-history R0 on **all three**. The shuffled neural model beat ordered R2 on both newly scored birds. The original within-bird order advantage was therefore not robust across the tested individuals.

Evidence: [frozen transfer results](../outputs/bfsongrepo_frozen_transfer_results.json), [closeout](../outputs/bfsongrepo_frozen_transfer_closeout.json).

## Multi-bird training: completed negative comparison

Four bird-disjoint outer folds trained on three birds and evaluated on the fourth. Training-only inner validation selected Ridge candidates. Neural arms used fixed 12-epoch budgets, width 64, Adam learning rate 0.001, weight decay 0.0001, gradient clipping at 5 and seed 0. Ordered and shuffled arms shared their architecture and training population; older-history ordering differed during training and inference.

Feature normalization and reporting scales were fitted using training birds. No held-out-bird adaptation, denoising or auxiliary objective was added. Training imbalance remained as registered. Outer folds share training individuals, so they are not four independent experiments.

| Held-out bird | R0 | R2-multi | R2-multi shuffled |
|---|---:|---:|---:|
| bl26lb16 | **1.0431** | 1.1003 | 1.1014 |
| gr41rd51 | **0.9489** | 0.9685 | 1.0468 |
| gy6or6 | **0.8463** | 0.8839 | 0.9813 |
| or60yw70 | **0.8185** | 0.9192 | 0.8995 |

R2-multi lost to R0 in every fold. The registered conditional screening rule failed. These results supersede earlier descriptions of multi-bird training as merely proposed. They do not demonstrate population-wide superiority or justify Stage 2.

The single-bird frozen reference was eligible only when its source bird was absent from the held-out target; its or60yw70 evaluation cannot count as unseen-bird transfer. Results use fold-specific training scales and should not be numerically equated with earlier study errors.

Evidence: [multibird protocol](../outputs/bfsongrepo_multibird_preregistration.json), [results](../outputs/bfsongrepo_multibird_results.json).

## Exact acoustic coding is a separate experiment

A lossless baseline covered all 552 recordings, approximately 8,058 seconds of audio.

| Codec | Encoded bytes | Fraction of raw PCM | Bits/sample | Exact reconstructions |
|---|---:|---:|---:|---:|
| FLAC | 238,261,830 | 0.4620 | 7.3919 | 552/552 |
| Predictive delta + zlib | 318,654,060 | 0.6179 | 9.8860 | 552/552 |

Raw PCM occupied 515,728,038 bytes. Exact reconstruction proves acoustic preservation for the specified sample representation. It does not demonstrate a learned semantic code or receiver interpretation.

Evidence: [lossless results](../outputs/bfsongrepo_lossless_baseline_results.json).

## Frozen inference and exported trajectories

The [bundle loader](../src/bioacoustic_pipeline/embeddings/macde_bundle.py) verifies checkpoint structure and spectral configuration, freezes parameters and runs in evaluation mode. A caller should supply a checksum obtained from a trusted provenance record.

```python
from bioacoustic_pipeline.embeddings.macde_bundle import load_macde

encoder = load_macde(
    "checkpoints/macde_stage1_R2.pt",
    expected_sha256=trusted_checkpoint_hash,
)
trajectory = encoder.encode_audio(
    waveform, sample_rate, sample_valid=sample_support,
)
```

`waveform`, `sample_rate`, `sample_support` and `trusted_checkpoint_hash` are caller-provided inputs. This example does not perform fitting.

Exports include states, state validity, multiscale changes and masks, physical forecasts, observed forecast residuals and masks, frame-end coordinates and source/activity support. Unsupported states and changes remain missing with explicit masks.

A forecast residual becomes available only when its future target has been observed. It is a model-relative error, not biological surprise. Disrupted-history or ablated checkpoints must retain their own arm policies; they are not interchangeable with the ordered continuous export.

## MACDE–CAVEM interface

The [post-call bridge](../src/bioacoustic_pipeline/contextual/macde_bridge.py) preserves source IDs, sample origins, event coordinates, masks and independently observed metadata. Its prediction cutoff is the call end. Complete-call features cannot legitimately support an onset-time prediction.

The current 64-state export creates 1,088 summary values: state mean, standard deviation, first/last state, slope, four quarter means and two summaries for each of four change scales. Summary validity is separate from numerical filling. The CAVEM transport vector has 2,186 dimensions, including validity indicators, quality measures and explicit timing/overlap flags.

Unknown caller, recipient and context remain unknown. Metadata must carry evidence and an availability coordinate no later than the prediction cutoff. Mixing source coordinate systems or incompatible encoder schemas is rejected.

The [synthetic smoke result](../outputs/macde_cavem_bridge_smoke.json) reports finite consumer states, future-audio exclusion and consumer-prefix causality. It explicitly reports `mechanical_only: true`, `model_fitting: false` and `stage2_ready: false`.

See [CAVEM's interface discussion](CAVEM.md#macdecavem-mechanical-interface) for interpretation and timing limitations.

## Research boundaries and further work

A new downstream study would require independently grounded outcomes, verified acoustic-to-event linkage, a justified dependence structure, protected evaluation and strong metadata/physical-feature comparators. Acoustic forecasting success alone does not satisfy those requirements.

The present evidence does not establish semantic understanding, animal-language translation, absence of discrete units, or the biological irrelevance of syllables. Continuous waveforms and discrete annotated structure can coexist.

RCLAT provides separate human speech experiments with known linguistic references. Their results do not alter MACDE's closed endpoints or certify animal meanings. The appropriate next investigation must be prospectively defined around a specific unresolved question, rather than modifying completed studies after inspecting results.

## Implementation and evidence map

| Material | Location |
|---|---|
| Core model | [macde.py](../src/bioacoustic_pipeline/embeddings/macde.py) |
| Frozen loader | [macde_bundle.py](../src/bioacoustic_pipeline/embeddings/macde_bundle.py) |
| Original Stage 1 runner | [macde_stage1.py](../scripts/macde_stage1.py) |
| Development runner | [bfsongrepo_development.py](../scripts/bfsongrepo_development.py) |
| Frozen transfer runner | [bfsongrepo_frozen_transfer.py](../scripts/bfsongrepo_frozen_transfer.py) |
| Multi-bird runner | [bfsongrepo_multibird.py](../scripts/bfsongrepo_multibird.py) |
| Bridge tests | [test_macde_cavem_bridge.py](../tests/test_macde_cavem_bridge.py) |

Runner links document provenance; they are not instructions to rerun closed studies. This reference was assembled from saved results and source inspection without new training or scientific scoring.
