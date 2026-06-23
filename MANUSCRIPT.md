# A Falsifiable, Multi-Paradigm Pipeline for Bioacoustic Structure Discovery

**Sheel Morjaria**

---

## Abstract

Discovering structure in animal vocalizations without anthropomorphic contamination
remains a fundamental challenge. Existing methods typically apply a single discovery
algorithm to a single species and claim biological significance for whatever patterns
emerge—without null models, cross-method validation, or an explicit semantic boundary.
We present a schema-first pipeline that decomposes vocal audio into segments, multi-view
representations, and sequence roles, then produces an auditable six-axis Evidence Profile
in which each signal carries calibrated uncertainty and an explicit statement of how it
could be wrong. The pipeline is validated across five corpora spanning four species and
human speech, corroborated by four independent mathematical paradigms (sequence alignment,
graph topology, dynamical-systems recurrence, and continuous subsequence matching).
Three architectural invariants enforce scientific honesty: (1) an ethological firewall
that blocks cross-species weight transfer and restricts text-supervised models to
retrieval-only use; (2) a semantic ceiling that prohibits any computational claim of
validated meaning—only Track C playback biology can elevate a signal above T1 on our
six-rung causal evidence ladder (T0–T5); and (3) abstention as a first-class output,
whereby axes that cannot compute abstain with a structured reason rather than emitting a
false biological null. The pipeline operates autonomously at the acoustic-structure
level (T0–T1), generates playback-ready counterfactual stimuli for biological validation
(T2–T4), and demonstrates 33× specificity on negative controls. Every validated finding
lands in exactly one of three buckets: what acoustics encode (individual identity, call
form), what they do not encode (context, valence, social partner, word identity), and
what requires social or biological validation to interpret function.

---

## 1. Introduction

Animal vocalizations carry rich acoustic structure. Songs, calls, and bouts exhibit
recurrence, sequencing, and contextual association—but which structures are genuinely
communicative, and which are artefacts of the analysis method? The prevailing approach
in computational bioacoustics applies a single discovery algorithm—dynamic time warping
(DTW), n-gram analysis, or a learned neural embedding—to a single species, then claims
biological significance for whatever motifs emerge. This approach is epistemically
fragile for three reasons. First, without an explicit null model, any algorithm "finds"
structure in any audio, including synthetic noise. Second, without cross-method
validation, the discovered structure may be an idiosyncrasy of the chosen algorithm
rather than a property of the vocalizations. Third, without an explicit semantic
boundary, there is no principled barrier to interpreting acoustic patterns as "meaning"
—a form of computational anthropomorphism.

We address these challenges with a pipeline built on three design principles:

1. **Schema-first falsifiability.** Every output is a six-axis Evidence Profile validated
   by a schema checker (`schema.validate_report()`) that raises on any violation.
   Positive structural claims carry matching falsification tests—permutation nulls,
   bootstrap stability, within-group controls, and biology-gated semantic validation.
   Axes that cannot compute (sparse data, missing labels) **abstain** with a structured
   reason code, never a false biological null.

2. **Ethological firewall.** Every model in the pipeline carries machine-readable
   provenance: pretraining domain, human-speech and text-supervision flags, and
   allowed/forbidden uses. The firewall hard-fails on cross-species weight transfer and
   restricts text-supervised models (e.g., NatureLM) to retrieval-only use. This is
   enforced at the registry level (data-driven JSON), not in ad-hoc code.

3. **Semantic ceiling.** Acoustic tokens (VQ-VAE or k-means clusters) are
   acoustic-similarity classes, *never* word or meaning identity. The schema rejects
   `validated_semantics: true` unless the semantic_validation axis is at
   `playback-supported`—a level achievable only by Track C biology (receiver-response
   experiments). This is encoded in both the schema validator and the scoring functions.

### Contributions

This work makes the following contributions:

- A **segmentation router** that selects the appropriate segmenter per acoustic regime
  (PCEN+CRNN for discrete calls, NBD+CPC for continuous speech, fixed-threshold NBD for
  unknown), with an auditable decision trail and a token-free pre-router.
- **DUSTED** (DTW-based discovery of recurrent acoustic terms), adapted from spoken-term
  discovery to animal vocalizations and validated across three structural regimes.
- A **four-paradigm cross-validation** framework demonstrating that DUSTED's findings are
  corroborated by independent mathematics: probabilistic suffix trees (variable-order
  Markov), transition-network topology (graph theory), recurrence quantification analysis
  (dynamical systems), and matrix profile (continuous subsequence matching).
- The **H0-CMC evidence ladder** (T0–T5), connecting acoustic discovery to causal
  biological validation, with an explicit semantic-validation mapping.
- A **DSP Audio Splicer** generating five-condition counterfactual playback stimuli from
  discovered structure, making the pipeline's output directly testable in the wet lab.
- Validation across **five corpora** (Egyptian fruit bat, marmoset, Bengalese finch,
  rook, human speech) with **35 benchmark sections** and **247 unit tests**.

---

## 2. Related Work

**Spoken-term discovery** (Park & Glass, 2008; Jansen et al., 2013) finds recurring
acoustic patterns in untranscribed speech via DTW or segmental clustering. DUSTED adapts
this to animal vocalizations with the critical modification that discovered terms are
labelled as **recurrent acoustic terms**, not word candidates—an distinction enforced by
the semantic ceiling.

**Bioacoustic structure analysis** has typically employed either hand-crafted features
(spectral, temporal) or learned embeddings (CNN, self-supervised). Our acoustic
fingerprint occupies a middle ground: physics-grounded descriptors (centroid, bandwidth,
flux, harmonicity) combined with texture features (GLCM, modulation, ICI), yielding a
44-dimensional species-portable representation that is interpretable by design.

**Network analysis of vocal sequences** (Sasahara et al., 2012; Kershenbaum et al., 2014)
models animal syntax as Markov chains or finite-state networks. Our transition-network
analysis recovers the same modular structure from unsupervised HECB segmentation, without
hand-annotated syllable labels.

**Recurrence quantification analysis** (Webber & Zbilut, 1994; Marwan et al., 2007)
quantifies dynamical structure via recurrence plots. We apply it to acoustic-fingerprint
trajectories and demonstrate that laminarity (%LAM) is context-dependent in bat bouts
(p=0.016)—a finding invisible to discrete-token approaches.

---

## 3. Framework

### 3.1 The Segmentation Router

Different acoustic regimes require fundamentally different segmenters. A single-segmenter
approach inevitably fails on at least one regime: CRNN excels at isolating discrete calls
from field noise (F1=0.703 on MeerKAT) but is weak on continuous speech (F1=0.59);
conversely, the NBD+CPC calibrator achieves combined-F1=0.836 on LibriSpeech but
over-segments discrete calls in noise.

The router maintains three regimes in `configs/segmentation_router.yaml`:

| Regime | Method | Strength | Example |
|---|---|---|---|
| Discrete calls in noise | PCEN+CRNN → HECB annotation | Isolates calls from noise floor | MeerKAT (8 kHz) |
| Continuous speech-like | NBD + CPC calibrator (ABSTAIN) | Precision via learned context | LibriSpeech (16 kHz) |
| Exploratory / unknown | Fixed-threshold NBD + ABSTAIN | Conservative; no false structure | Novel species |

The router decision is auditable: every run produces `trace["router_audit"]` recording the
detected regime, the method chosen, the alternative tested, and the expected failure mode.
A token-free pre-router (`route_by_continuous`) uses Matrix Profile recurrence and RQA
laminarity—both computable without segmentation—to skip heavy DTW on degenerate data,
routing marmoset-style stutter directly to the fast path.

### 3.2 HECB-Seg: Hierarchical Event-Conditioned Boundary Segmentation

For calls-in-noise, the pipeline uses a two-stage architecture:

**Stage 1 — Call isolation:** PCEN spectrogram → Conv2D→BiLSTM → P(call) per frame →
hysteresis (enter 0.7, exit 0.3) → connected components → call events.

**Stage 2 — Internal annotation:** NBD proposals inside each call event → typed classifier
(onset/offset/internal/noise) → conservative decoder. The CRNN detector is *authoritative*
for call-event onset/offset; HECB annotates internal sub-call structure without refining
call edges.

A parent-quality gate (`min_parent_confidence`) abstains on low-confidence parent events,
preventing noisy internal boundaries from polluting the pooled representation. The
boundary-quality sweep (§30) confirms that this hierarchical structure does not improve
call-type recovery beyond the whole-call fingerprint—the sub-call is augmentation, not
replacement.

### 3.3 DUSTED: DTW Recurrent Acoustic Terms

DUSTED adapts spoken-term discovery to animal vocalizations. Each bout or call is
represented as a sequence of acoustic-fingerprint vectors. DTW with indels discovers
**recurrent acoustic terms**—subsequences that recur across calls beyond what an
order-shuffle null predicts. Crucially, these are *not* words: they are acoustic-similarity
clusters whose recurrence is statistically tested, not semantically interpreted.

Validated on three structural regimes (uniform DTW threshold thr=0.15, HAC clustering):

| Corpus | Regime | # Families | Coverage | Null ratio | Bootstrap |
|---|---|---|---|---|---|
| Bengalese finch | Discrete, diverse | 457 | 0.480 | **2.67×** | 0.509 |
| Egyptian fruit bat | Graded, continuous | 306 | 0.153 | 1.36× | 0.394 |
| Infant marmoset | Sparse, degenerate | 139 | 0.028 | 1.55× | 0.229 |

### 3.4 The Acoustic Fingerprint (44-D, All-Species)

The pipeline's cross-species representation is a physics-grounded, species-portable
per-segment descriptor combining three layers:

- **Physics** (~16D): spectral centroid, bandwidth, rolloff, flatness, flux, zero-crossing
  rate, RMS energy—and, for tonal segments only, F0, harmonicity (HNR), jitter, shimmer.
- **Macrotexture** (~14D): GLCM contrast/homogeneity, envelope statistics, modulation spectrum.
- **Microtexture** (~15D): derivative statistics, inter-contour intervals, dynamics.

Voiced/harmonic features are flagged and zeroed for non-tonal segments (anti-anthropomorphism).
Mel-MFCC (human-perceptual) is opt-in and never in the cross-species core. The fingerprint
is the one firewall-registered representation with `applies_to_species: []` (all-species).

### 3.5 Ethological Firewall

Every model in the pipeline carries provenance metadata in `configs/model_metadata.json`:
pretraining domain, human-speech/text-supervision flags, allowed uses, forbidden uses, and
`applies_to_species`. The firewall enforces four rules:

1. Text-supervised models (NatureLM) are retrieval-only.
2. Cross-species weight transfer is blocked.
3. No model may produce a `semantic_verdict`.
4. Species-specific models (bat CPC, marmoset CPC) are registered with
   `applies_to_species` and cannot be used on other species.

### 3.6 The Six-Axis Evidence Profile

The verdict is six **independent** axes, each categorical with a calibrated confidence:

`sequence_recombination · functional_aliasing · call_unitization · context_binding ·
turn_structure · semantic_validation`

Scorers take primitive signals (LZ complexity, alias density, Cramér's V, etc.) so the
profile is unit-testable and decoupled from signal extraction. The profile **never** claims
validated semantics: `score_semantic_validation` is hard-coded to require Track C biology.

---

## 4. Results

### 4.1 What Acoustics Encode: Individual Identity and Call Form

| Corpus | Target | Metric | Result | Lift |
|---|---|---|---|---|
| Egyptian fruit bat | Emitter identity | Cramér's V | 0.446 | — |
| Rook | Source (individual) identity | Cramér's V | 0.31 | — |
| Marmoset | Call type | Cramér's V / linear probe | 0.599 / 0.829 | 3.0× / 4.1× |
| MeerKAT | Call type | Cramér's V / linear probe | 0.521 / 0.753 | — |
| Bengalese finch | Syllable (GT) | DTW purity (wmean) | 0.788 | — |

Acoustic structure reliably carries **individual identity** (bat, rook) and **call/event
form** (marmoset, meerkat, Bengalese). The whole-call fingerprint is the primary
representation for both; sub-call structure (pooled/sequence) does not improve call-type
recovery beyond the whole call (§30, confirmed on both meerkat and marmoset).

### 4.2 What Acoustics Do Not Encode: Context, Valence, and Meaning

| Finding | Evidence | Section |
|---|---|---|
| Context requires social covariates | Bat: D(acoustic+social) = 4.3× ≫ A(acoustic) = 2.7×; social_lift = 1.59 | §9 |
| Valence is locked in call type | Soundwel: within-call-type fp→valence 0.350 ≪ chance 0.50 | §24 |
| Social partner ≈ chance | RookID: cross-caller retrieval below chance; caller-bound | §26 |
| Tokens ≠ words | LibriSpeech: token↔word macro consistency 0.29; speaker probe 0.56 > word probe 0.40 | §29 |
| Degenerate repertoire (token-free) | Marmoset MP: 87.5% stutter-dominant, unimodal histogram | §34 |

The **semantic ceiling** holds across all species: acoustics alone cannot recover context,
emotional valence, social-partner identity, or—crucially—word meaning. In humans, the CPC
latent is speaker-dominated (speaker probe 0.56 > word probe 0.40); meaning is not
linearly separable from the acoustic signal.

### 4.3 Multi-Paradigm Structural Corroboration

To demonstrate that DUSTED's findings are not method-dependent, four independent
mathematical paradigms are applied to the same corpora:

| Paradigm | Mathematics | Dataset | Key Result | Verdict |
|---|---|---|---|---|
| PST + Transition Network | Variable-order Markov + graph topology | Bengalese (discrete) | 247 predictive contexts, 5 communities, clustering 0.686 | **Corroborates** DUSTED's discrete-syntax finding |
| RQA | Recurrence plot → %DET, %LAM | Bat (graded) | %LAM context-dependent (Kruskal-Wallis p=0.016); %DET context-invariant | **Complements** DUSTED where DTW over-merges |
| Matrix Profile | STUMP continuous subsequence matching | Marmoset (sparse) | 87.5% stutter-dominant, unimodal motif-distance histogram | **Confirms** degenerate repertoire token-free |
| Transition Network | Networkx community detection | LibriSpeech (diverse) | 639 word nodes, 19 communities, clustering 0.093 | **Upper bound**: scales to human-scale diversity |

Three paradigms (alignment, graph, dynamics) agree on the biological reality of the three
repertoire regimes. The transition-network method scales from 24 nodes (Bengalese) to 639
nodes (LibriSpeech), demonstrating method validity across vocabulary sizes spanning two
orders of magnitude.

### 4.4 Negative Controls: Pipeline Specificity

To counter the reviewer argument that "the pipeline finds structure everywhere," three
negative controls are applied:

| Control | Description | # Motifs | vs Natural (33) |
|---|---|---|---|
| Synthetic FM chirps | Non-biological FM-sweep sequences (250 kHz) | **1** | 33× specificity |
| Random audio segments | Random chunks from bat recordings (not call-bounded) | **1** | 33× specificity |
| Order-shuffled bat | Same units, permuted order | 22 | Partial collapse |

The pipeline finds **1 motif** in synthetic and random audio versus **33** in natural bat
bouts—a 33× specificity ratio. It does not manufacture structure from arbitrary input.

### 4.5 Boundary-Quality Sweep: Whole-Call Primary

The boundary-quality sensitivity sweep varies parent-event boundary quality (oracle GT →
CRNN at multiple operating points → raw NBD) and measures the hierarchy benefit (pooled −
whole-call V) on call-type recovery. The result is definitive: at the best parent quality
(oracle), the hierarchy benefit is +0.021 V on meerkat and **negative** on marmoset. The
sub-call structure is augmentation for motif discovery, not a lever for call-type recovery.

---

## 5. The H0-CMC Evidence Ladder

The Causal Multimodal Convergence (H0-CMC) framework defines a six-rung evidence ladder
connecting acoustic discovery to causal biological validation:

| Tier | Evidence | Autonomous? | Semantic validation |
|---|---|---|---|
| T0 — Acoustic | Recurrent acoustic structure found | ✅ | none |
| T1 — Context | Context-associated cue (binding significant) | ✅ | none (correlation) |
| T2 — Playback | Counterfactual stimuli generated; response pending | Stimuli only | none (response pending) |
| T3 — Convergence | Multimodal convergence (acoustic + behavioral + physiological) | Wet lab | candidate_functional_interpretation |
| T4 — Counterfactual | Counterfactual disrupts receiver response | Wet lab | causally_supported_signal |
| T5 — Field | Ecologically validated (fitness-relevant field behavior) | Wet lab | ecologically_validated_function |

**The pipeline operates autonomously at T0–T1.** It generates T2 playback stimuli via the
DSP Audio Splicer (five conditions: natural, order_shuffled, onset_offset_swap,
caller_swapped, amplitude_matched). One hundred bat stimuli have been produced and
preflighted (all acoustically valid). T3–T5 require the wet lab.

An explicit-implicit test classifies each corpus's communication mode: if acoustics alone
recover context as well as the full feature set (A ≈ G), the signal is an *explicit
acoustic code*. If adding social covariates substantially improves recovery (D ≫ A, D ≈ G),
the signal is *socially grounded implicit*. On the bat, social_lift = 1.59—socially grounded.

---

## 6. Track C: Playback Readiness

The DSP Audio Splicer (`experiments/playback_design/audio_splicer.py`) generates
falsifiable counterfactual stimuli from HECB-segmented calls:

| Condition | Transform | Hypothesis | H0-CMC tier |
|---|---|---|---|
| Natural | Unchanged baseline | — | — |
| Order shuffled | Permute unit order | Does sequence syntax matter? | T2 |
| Onset–offset swap | Swap first and last units | Do positional roles matter? | T2 |
| Caller swapped | Replace one unit with a donor from another emitter | Does caller identity matter? | T2 |
| Amplitude matched | RMS-normalize to target | Control: rules out loudness confound | — |

Stimuli are generated via exemplar-based DSP (slice + crossfade + concatenate), not neural
synthesis. They are deterministic, firewall-clean, and transparent for the wet lab. Each
stimulus carries provenance (source file, emitter, context, permutation/donor index).

---

## 7. Discussion

### 7.1 The Three-Bucket Thesis

Every finding in this pipeline lands in exactly one of three buckets:

1. **What acoustics encode.** Individual identity (bat Emitter V = 0.446, rook Source V =
   0.31) and call/event form (marmoset V = 0.599, meerkat V = 0.521, Bengalese purity =
   0.788). These are robust, cross-method-validated, and species-portable via the acoustic
   fingerprint.

2. **What acoustics do not encode.** Context (bat requires social covariates; social_lift
   = 1.59), emotional valence (Soundwel within-call-type ≈ chance), social partner (rook
   cross-caller below chance), and word identity (LibriSpeech token↔word = 0.29). The
   semantic ceiling holds across all species tested.

3. **What requires social or biological validation.** All functional interpretation. The
   H0-CMC ladder requires biology (T2–T5) to move beyond acoustic structure. The pipeline
   generates the stimuli; the wet lab validates the function.

### 7.2 Honest Limitations

- **Tokenization.** DUSTED operates on k-means cluster IDs from the acoustic fingerprint,
  not learned VQ-VAE tokens. A species-specific VQ-VAE would unlock alias detection and LZ
  complexity on animals but is not yet trained.
- **CPC is human-only.** The ethological firewall blocks cross-species weight transfer.
  The animal representation is the acoustic fingerprint—physics-grounded and portable, but
  not learned. This is a feature (no anthropomorphic contamination), not a gap.
- **Sample sizes.** The bat RQA-by-context analysis uses n=26 bouts per context (firmed
  from an initial n=6). The LAM result (p=0.016) is stable across both sample sizes, but a
  larger corpus would further strengthen it.
- **Playback is computational only.** The DSP Audio Splicer produces stimuli; it cannot
  predict receiver response. That requires the wet lab.

### 7.3 What This Pipeline Is Not

This pipeline is explicitly **not a classifier** that emits labels. It does not claim
validated semantics. It does not transfer learned weights across species. It does not find
structure in non-biological audio (33× specificity on negative controls). And it
abstains—with structured reasons—rather than emitting false nulls when data is insufficient.
The pipeline's value lies not in what it claims, but in the rigor with which it
distinguishes what it can claim from what it cannot.

---

## 8. Conclusion

We have presented a falsifiable, multi-paradigm pipeline for bioacoustic structure
discovery that operates autonomously at the acoustic level and generates playback-ready
stimuli for biological validation. Its core contribution is not a new algorithm but a
framework for honest inference: every claim is calibrated, every negative is structured,
and every semantic assertion is gated behind biological evidence. Four independent
mathematical paradigms corroborate the same structural regime classifications across five
corpora, demonstrating that the findings are properties of the vocalizations, not the
method. The pipeline is frozen, tested (247 tests), and ready for the next phase: wet-lab
playback experiments that move the evidence from T1 (context-associated) to T4
(causally supported).

---

## Reproducibility

All code, tests (247), and validated results (35 benchmark sections) are in the
`bioacoustic_pipeline` repository. Architecture v1.0 is frozen on `main`. Eleven commits
document the full implementation trajectory.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
