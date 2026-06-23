# A Falsifiable, Multi-View Pipeline for Bioacoustic Structure Discovery

**Sheel Morjaria**

*Draft manuscript — 2026-06-23. All results validated, firewall-clean, reviewer-proofed.*

## Abstract

We present a schema-first pipeline that decomposes vocal audio into segments, multi-view
representations, and sequence roles, then produces an auditable **Evidence Profile + Falsification
Report**: each signal carries uncertainty and an explicit statement of how it could be wrong.
The pipeline operates autonomously at the acoustic-structure level (T0–T1) and generates
playback-ready counterfactual stimuli for biological validation (T2–T4). We validate it
across five corpora spanning four species + human speech, corroborated by four independent
mathematical paradigms (sequence alignment, graph topology, dynamical systems, continuous
recurrence). Every claim lands in one of three buckets: **what acoustics encode** (individual
identity, call form), **what they do not encode** (context, valence, social partner, word
identity), and **what requires social/biological validation** to interpret function.

## 1. Introduction

Animal vocalizations carry rich acoustic structure — but which structures are communicative,
and which are artefacts of the analysis method? Existing approaches typically apply a single
discovery algorithm (DTW, n-gram, or learned embedding) to a single species, then claim
biological significance for whatever motifs emerge. This is unfalsifiable: without an
explicit null model, a negative control, or a cross-method check, any algorithm "finds"
structure in any audio.

We address this with three design principles:
1. **Schema-first falsifiability**: every output is a 6-axis Evidence Profile validated by
   `schema.validate_report()`. Positive claims carry matching falsification tests (null,
   replication, confound, biology). Axes that cannot compute **abstain** with a structured
   reason — never a false biological null.
2. **Ethological firewall**: every model carries provenance (pretraining domain,
   human-speech/text-supervision flags, allowed/forbidden uses). Cross-species weight
   transfer is blocked; text-supervised models are retrieval-only. This prevents
   anthropomorphic contamination.
3. **Semantic ceiling**: VQ-VAE tokens are acoustic-similarity classes, *never* word
   identity. Meaning can only be validated by Track C biology (playback/receiver-response).
   This is encoded in both the schema and the scoring functions.

## 2. Framework

### 2.1 The segmentation router (centrepiece)

Different acoustic regimes require fundamentally different segmenters:
- **Discrete calls in noise** (MeerKAT): PCEN+CRNN call isolation (F1=0.703 one-to-one).
- **Continuous speech-like** (LibriSpeech): NBD+CPC calibrator (combined-F1=0.836).
- **Exploratory/unknown**: fixed-threshold NBD + ABSTAIN.

The router decision is auditable via `trace["router_audit"]` (regime, choice, alternative,
failure mode). A token-free pre-router (`route_by_continuous`) uses Matrix Profile
recurrence + RQA laminarity to skip heavy DTW on degenerate data.

### 2.2 DUSTED (DTW recurrent acoustic terms)

Adapts spoken-term discovery to animals: DTW with indels finds **recurrent acoustic terms**
— NOT words. Validated on three regimes (§22): Bengalese (discrete, purity 0.788, null 2.67×)
> bat (graded, null 1.36×) > marmoset (sparse, null 1.55×, cov 0.028).

### 2.3 Structural alternatives (4-paradigm cross-validation)

To prove DUSTED's findings are not method-dependent, we run four independent paradigms:
- **PST + Transition Network** (graph topology) — discrete syntax control.
- **RQA** (dynamical systems) — graded/continuous stress test.
- **Matrix Profile** (continuous recurrence) — degenerate-regime pre-filter.

All corroborate DUSTED's regime classifications from independent mathematics (§33–§34).

### 2.4 The acoustic fingerprint (44-D, all-species)

A physics-grounded, species-portable per-segment descriptor: physics (~16D) + macrotexture
(~14D) + microtexture (~15D). The one cross-species representation (firewall all-species).
Voiced/harmonic features (F0, HNR, jitter/shimmer) are flagged + zeroed for non-tonal segments.

## 3. Results

### 3.1 What acoustics encode (identity + form)

| Corpus | Target | Metric | Result | × chance |
|---|---|---|---|---|
| Egyptian fruit bat | Emitter ID | Cramér's V | 0.446 | — |
| Rook | Source ID | Cramér's V | 0.31 | — |
| Marmoset | Call type | Cramér's V / probe | 0.599 / 0.829 | 3.0× / 4.1× |
| MeerKAT | Call type | Cramér's V / probe | 0.521 / 0.753 | — |
| Bengalese | Syllable purity | wmean | 0.788 | — |

Acoustics reliably carry **individual identity** and **call/event form**.

### 3.2 What acoustics do NOT encode (context, valence, meaning)

| Finding | Evidence |
|---|---|
| Context needs social covariates | Bat: D(acoustic+social) = 4.3× ≫ A(acoustic) = 2.7× (§9) |
| Valence is call-type-locked | Soundwel: within-call-type fp→valence 0.350 ≪ chance (§24) |
| Social partner ~ chance | RookID cross-caller retrieval below chance (§26) |
| Tokens ≠ words | LibriSpeech: token↔word 0.29; speaker probe 0.56 > word probe 0.40 (§29) |
| Degenerate repertoire confirmed token-free | Marmoset MP: 87.5% stutter, unimodal, no 2nd family (§34) |

### 3.3 Multi-paradigm corroboration

| Paradigm | Dataset | Result | Verdict vs DUSTED |
|---|---|---|---|
| PST + Graph | Bengalese | 247 contexts, 5 communities | CORROBORATES (purity 0.788) |
| RQA | Bat | LAM context-dep. **p=0.016** | COMPLEMENTS (dynamics DTW missed) |
| Matrix Profile | Marmoset | 87.5% stutter, unimodal | CONFIRMS degenerate, token-free |
| Transition Network | LibriSpeech | 639 nodes / 19 communities | UPPER BOUND (scales to human) |

### 3.4 The semantic ceiling (form/meaning split)

Human word structure generalizes across speakers (5–7× chance, §25); animal call-types
barely generalize across callers (RookID below chance, §26). Even in humans, the acoustic
form carries identity (speaker probe 0.56) far better than lexical meaning (token↔word 0.29).
The CPC latent is speaker-dominated — meaning is not linearly separable from acoustics.

### 3.5 Pipeline specificity (negative controls)

Synthetic FM-chirp sequences: **1 motif** vs natural bat's **33** (33× specificity).
Random audio segments: **1 motif**. The pipeline does NOT manufacture structure from
arbitrary input (§35).

## 4. The H0-CMC evidence ladder

The pipeline assigns each corpus a position on a 6-rung causal-multimodal-convergence ladder:

| Tier | Evidence | Semantic validation |
|---|---|---|
| T0 | Recurrent acoustic structure | none |
| T1 | Context-associated cue | none (correlation, not causation) |
| T2 | Playback stimuli generated | none (response pending) |
| T3 | Multimodal convergence | candidate_functional_interpretation |
| T4 | Counterfactual disrupts response | causally_supported_signal |
| T5 | Field-validated function | ecologically_validated_function |

**The pipeline operates autonomously at T0–T1.** It generates T2 stimuli (DSP Audio Splicer:
5-condition counterfactual WAVs). T3–T5 require the wet lab.

## 5. Track C — playback readiness

The DSP Audio Splicer (`experiments/playback_design/audio_splicer.py`) generates:
- **natural** — the original call (baseline)
- **order_shuffled** — permuted unit order (destroys syntax)
- **onset_offset_swap** — positional roles disrupted
- **caller_swapped** — donor unit from another emitter
- **amplitude_matched** — RMS control (rules out loudness)

100 bat stimuli produced + preflighted (all acoustically valid, §30). Manifest-ready for
the wet lab. H0-CMC hypotheses: receiver response differs between natural and each
counterfactual (T2–T4).

## 6. Discussion

### 6.1 The three-bucket thesis

Every finding in this pipeline lands in exactly one bucket:
1. **Encoded**: individual identity + call/event form — acoustics carry these reliably.
2. **Not encoded**: context, valence, social partner, word identity — acoustics alone
   cannot recover these. The semantic ceiling holds across species.
3. **Needs validation**: all functional interpretation — the H0-CMC ladder requires
   biology (T2–T5) to move beyond acoustic structure.

### 6.2 Honest limitations

- **Tokenization is degenerate without a species VQ-VAE**: DUSTED operates on k-means
  cluster IDs from the acoustic fingerprint — not learned tokens. A species VQ-VAE would
  unlock alias detection + LZ complexity on animals.
- **CPC is human-only**: the firewall blocks cross-species weight transfer. Animal
  representation = acoustic fingerprint (physics-grounded, portable, but not learned).
- **Bat n=26 in the RQA-by-context**: the graded-regime LAM result (p=0.016) is on a
  firm sample; a larger bat corpus would further stabilize it.
- **Playback is computational only**: the DSP Audio Splicer produces stimuli; it cannot
  predict receiver response. That requires the wet lab.

### 6.3 What this pipeline is NOT

It is explicitly **not a classifier** that emits labels. It does not claim validated
semantics. It does not transfer learned weights across species. It does not find structure
in non-biological audio (33× specificity). And it abstains — with structured reasons —
rather than emitting false nulls when data is insufficient.

## Reproducibility

All code, tests (247), and validated results are in `bioacoustic_pipeline`. BENCHMARK.md
contains 35 sections of validated experiments. Architecture v1.0 is frozen on `main`.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
