# Architecture: representation & segmentation stack

The canonical, frozen shape (2026-06-23). A falsifiable multi-view pipeline for bioacoustic
structure discovery — not a classifier. Each vocalisation is one **hierarchical dual-stream
event**, and the whole stack is organized around that unit. In addition to the core
verdict pipeline, the stack now includes a **DUSTED-style recurrent subsequence motif
discovery** module (§8) that runs across 5 animal corpora + LibriSpeech as a cross-species
motif-arc validation layer.

See also: [`BENCHMARK.md`](BENCHMARK.md) (the numbers), [`CLAUDE.md`](CLAUDE.md) (invariants),
[`configs/segmentation_router.yaml`](configs/segmentation_router.yaml) (the router).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.

---

## 1. The stack (data flow)

```
audio / annotations
  ↓  segmentation router (per acoustic regime)
CallEvent  (+ optional HECB InternalSegment list)        ← segmentation/hierarchical.py
  ↓
DualStreamVocalisation  (parent whole-call token)        ← representation/vocalisation.py
  ├ discrete stream:   vq_token · motif_token · fingerprint_cluster · alias_role
  ├ continuous stream: affect_mu · acoustic_fingerprint · confidence
  └ sub_segments:      [DualStreamSubToken, ...]   (HECB internal, relative positions)
  ↓
whole / sub-call / pooled / relative-sequence ablation   ← whole_sub_pooled_sequence_ablation
  ↓
calltype · caller-ID · motif · alias · cross-species diagnostics → verdict report
```

Two representation layers, deliberately distinct:

| layer | class | role |
|---|---|---|
| **event carrier** | `DualStreamSegment` (`representation/dual_stream.py`) | the typed-event **spine** — one segment = one row in the sequence motif/LZ/alias diagnostics consume. Flat by default; gains `parent_call_id` + `sub_segments` for hierarchy. |
| **analysis unit** | `DualStreamVocalisation` (`representation/vocalisation.py`) | the **call-level unit** diagnostics + ablations reason over — a whole-call parent carrying discrete + continuous streams + typed sub-children, with cross-species structural summaries. |

Both are built from the same `CallEvent`; the carrier is the lower-level typed-event
sequence, the vocalisation is the richer whole-call/hierarchy view layered on top.

---

## 2. Segmentation router (per regime)

`configs/segmentation_router.yaml` + `run_pipeline(segmentation=...)` in `orchestration.py`.

| regime | method | HECB role | F1 |
|---|---|---|---|
| discrete calls in noise | PCEN+CRNN (`segmentation/hecb.py`) | internal annotator | meerkat 0.70, marmoset 0.88 |
| continuous speech-like | NBD+CPC calibrator (`segmentation/nbd_*.py`) | optional | LibriSpeech 0.836 combined |
| exploratory / unknown | NBD + ABSTAIN | none | conservative |

`segmentation/hecb.py:hecb_segment_audio()` → `CallEvent`s; `hierarchical.py:hecb_decode()`
(snap=False, gate_durations=False by default → CRNN timing preserved; only annotates
typed internal boundaries). Falls back to NBD with a trace flag if torch/checkpoint
unavailable.

---

## 3. Segmentation objects

- **`CallEvent`** (`hierarchical.py`) — one detected vocalization: onset/offset, CRNN
  confidence, `internal_segments: list[InternalSegment]`. Produced by `hecb_decode`
  (CRNN parents) or `two_stage_segment` (NBD parents).
- **`InternalSegment`** — a typed sub-call unit (start_s/end_s/boundary_type/nbd_confidence).

---

## 4. Representation objects

- **`DualStreamSubToken`** (`vocalisation.py`) — typed internal child: relative_start/end
  (normalized within the parent), boundary_type, vq_token, fingerprint.
- **`DualStreamVocalisation`** — the parent whole-call token (discrete + continuous +
  `sub_segments`). `structural_summary()` = cross-species descriptors
  (`subcall_count`, `relative_boundary_positions`, `mean_internal_gap`).
- **`SubcallRepresentation`** (`subcall.py`) — the lighter whole/sub/pooled fingerprint
  triple used by the 3-way ablation (`whole_sub_pooled_ablation`).
- **`DualStreamSegment`** (`dual_stream.py`) — the carrier spine. `build_dual_stream`
  (flat NBD) / `build_dual_stream_from_calls` (hierarchical parent+children).

Builders: `build_vocalisation()` (vocalisation) and `build_subcall_representations()`
(subcall) consume any `CallEvent` — **segmenter-agnostic** (CRNN / oracle / NBD all feed
the same representation layer).

---

## 5. Guardrails (do not weaken)

1. **Ethological firewall** (`embeddings/firewall.py` + `configs/model_metadata.json`) —
   cross-species weight transfer blocked; text-supervised models retrieval-only; no model
   emits a `semantic_verdict`. JSON-driven registry.
2. **Semantic ceiling** — VQ tokens are acoustic-similarity classes, *never* word identity;
   functional aliases group tokens that do the same *job*, not that sound the same.
3. **Cross-species false-equivalence guardrail** (encoded structurally in `vocalisation.py`):
   `vq_token` / `fingerprint_cluster` are **species-specific** — never compared across
   species. Cross-species comparison uses `acoustic_fingerprint` (physics + macro + micro,
   the one all-species view) + `structural_summary()` only.
4. **Statistical honesty** — every positive structure claim ships with its null/permutation
   test (`diagnostics/null_models.py`); one-to-one boundary matching (no duplicate-TP inflation).
5. **Default = whole-call primary + sub-call augmentation** — never sub-call replacement
   (empirically: pooled ≥ whole > sub-call-alone across all corpora/labels tested).

---

## 6. Canonical claim (paper-safe)

Each vocalisation is represented as a **hierarchical dual-stream event**: a whole-call
parent token augmented by optional typed sub-call tokens. Whole-call representations remain
primary, while HECB-derived sub-call structure provides an auxiliary layer for internal
analysis, motif discovery, and cross-call comparison. Cross-species comparisons use acoustic
fingerprints and structural summaries rather than raw species-specific token IDs.

---

## 7. Key entry points

```python
# Segment (any regime) → CallEvents with internal segments
from bioacoustic_pipeline.segmentation.hecb import hecb_segment_audio, load_hecb
from bioacoustic_pipeline.segmentation.hierarchical import hecb_decode

# CallEvents → canonical analysis unit
from bioacoustic_pipeline.representation.vocalisation import (
    build_vocalisation, whole_sub_pooled_sequence_ablation, format_4way_table,
)

# DUSTED recurrent subsequence motif discovery (cross-species motif arc)
from bioacoustic_pipeline.diagnostics.motif_miner.subsequence_motifs import (
    CallUnits, discover_recurrent_subsequences, fixed_length_cosine_clusters,
    exact_ngram_clusters, discovery_null, bootstrap_stability, summarize_discovery,
)

# Or the full pipeline
from bioacoustic_pipeline.orchestration import run_pipeline
report, trace = run_pipeline(paths, species="marmoset", segmentation="hecb")
```

---

## 8. DUSTED recurrent subsequence motif discovery (`motif_miner/subsequence_motifs.py`)

Adapts spoken-term-discovery (DUSTED) to **recurrent acoustic terms** in animal
vocalisations — NOT words. The transferable target is recurrent motifs / sub-call
phrases / sequence templates / vocalisation-internal units. Speech word-discovery
methods are adapted, with the router/HECB structure wrapped around them.

### Design

Each vocalisation = one `CallUnits` (ordered internal sub-call units). Windows of
length L=2–6 are generated; similar windows are matched by one of three primitives;
matched windows are clustered into connected-component motif families.

| method | matching primitive | role |
|---|---|---|
| `exact_ngram_clusters` | exact cluster-ID tuple match | fragments every variant; highest precision |
| `fixed_length_cosine_clusters` | same-length windows, mean per-position cosine | catches substitutions, misses indels |
| `discover_recurrent_subsequences` | **DTW with indels** (role-aware, length-normalised) | catches substitutions + indels; richest families |

DTW cost (biologically grounded in the onset/body/offset positional-role result):

```
substitution(i, j) = (1 − cosine(fp_i, fp_j)) + role_penalty·𝟙[role_i ≠ role_j]
                   + relpos_lambda·|rel_mid_i − rel_mid_j|
indel(role)        = indel_edge if role ∈ {onset, offset} else indel_body
distance           = total_alignment_cost / max(len(a), len(b))
```

Constrained cascade (full all-pairs DTW is never run): window generation →
centroid-cosine + length-ratio prefilter (`sklearn.NearestNeighbors`) → DTW only on
surviving pairs → motif graph (edge = DTW dist < threshold) → connected components.

### Diagnostics (per motif family + corpus summary)

- **`label_purity`** — modal cluster-tuple fraction among members (GT validation when
  clusters are ground-truth labels, e.g. Bengalese syllable IDs).
- **`family_entropy`** / **`family_entropy_norm`** / **`top_family_support_fraction`** —
  diverse-inventory vs one-dominant-repeat discriminator (Bengalese = diverse;
  marmoset = one dominant; bat = graded/intermediate).
- **`calltype_enrichment`** / **`caller_enrichment`** — Cramér's V + χ² p of
  motif-presence vs the call's calltype / caller (association, NOT semantics).
- **`discovery_null`** — within-call order shuffle on total support (tests whether
  motif recurrence depends on sequence order; total support is the sensitive statistic).
- **`bootstrap_stability`** — resample calls, re-discover top-K, mean pairwise Jaccard.

### Terminology (relanguage)

Speech word discovery is adapted, but the animal target is **recurrent acoustic
structure**, NOT word recovery. "Word" is reserved for the LibriSpeech human-speech
controlled analogue (which has word GT). See the `BENCHMARK.md` Terminology section for
the full speech→animal translation table.

---

## 9. Cross-corpus motif arc (5 animal corpora + LibriSpeech)

The DUSTED module runs across 5 animal corpora with identical diagnostics, producing a
cross-species comparison of recurrent-acoustic-term structure:

| corpus | regime | inventory | key finding | §ref |
|---|---|---|---|---|
| **Bengalese finch** | discrete syllable song | diverse, high-purity | fixed-cosine wmean purity 0.895; DTW recovers `[4,5]` at 1.00 | §12 |
| **Egyptian fruit bat** | graded bout vocalisations | graded/intermediate | Emitter V ~0.46 ≫ Context V ~0.16 (2.8×); fixed-cosine headline | §13 |
| **InfantMarmosetsVox** | discrete calls (oracle HECB) | sparse, one dominant | DTW calltype V 0.51 = 4.7× exact; degenerate inventory | §11 |
| **RookID** | noisy outdoor corvid calls | degenerate (one dominant) | Event V 0.22 > Source V 0.08; motifs are call-type not individual | §18–§19 |
| **LibriSpeech** | human speech (controlled) | diverse word structure | fingerprint leads consistency + clustering; HuBERT metric-dependent | §14–§16 |

**The difference is the corpus/repertoire, not the method** — DUSTED behaves correctly in
all regimes, landing where predicted by the vocalization structure.

### Individual identity across species

The 44-D acoustic fingerprint consistently preserves individual identity:

| species | identity target | fingerprint V / accuracy | held-out control |
|---|---|---|---|
| Egyptian fruit bat | Emitter | Emitter V ~0.46 (motif families) | 70/30 held-out: V 0.375 (§17) |
| RookID rook | Source (individual) | macro-F1 0.287 (4× chance), V 0.384 | recording-date held-out (§18) |
| InfantMarmosetsVox | Caller | caller V present, sub-call augmentation modest | twin-pair held-out |

**Sub-call structure does NOT improve individual-ID accuracy** (RookID: whole-call wins
on all metrics). HECB's value is structural annotation (motifs, position roles), not
individual-ID accuracy. For *who is calling*, the whole-call fingerprint is the right tool.

---

## 10. H0-CMC framework + speaker-invariant representation science (2026-06-23)

### H0-CMC: Causal Multimodal Convergence

The pipeline's Track C validation framework, refined from H0-Unified. See
[`h0-cmc-framework.md`](h0-cmc-framework.md) for the full spec + [`cpc-policy.md`](cpc-policy.md)
for the CPC selective-use policy.

**One-line:** *Animal vocalisations are structured acoustic cues whose functional
interpretation is validated by receiver responses under controlled social contexts — not
decoded words from acoustics alone.*

**4-stage evidence ladder:** L0 (acoustic association) → L1 (contextual prediction) →
L2 (receiver response) → L3 (causal specificity). The computational pipeline operates at
T0–T1; T2–T4 (playback) is the biological next phase.

**Explicit-vs-implicit model test:** plugged directly into the bat §9 A–G ablation —
D (acoustic+social) = 4.3× ≈ G (full) = 4.7× → **socially grounded implicit communication**
(social context is the true complement to acoustics; behavior is a minor bridge).

### Speaker-invariant representation science (human-speech side)

Cross-speaker same-word retrieval (§25) + DBA consensus-template sweep (§27) prove:
- Word structure **does** generalize across speakers (CPC 5–7× chance, §25).
- The speaker/word entanglement is **non-additive** — linear centering fails (§25 Δ −0.003),
  VTLN helps NN but not the template (−0.056), HuBERT/AVES MLM fails (−0.048/−0.068).
- **Only CPC's contrastive objective** makes DBA averaging help (+0.012) — the sign flips
  across 5 representations. A speaker-invariant template requires learned non-linear
  disentanglement, never algebraic averaging of the fingerprint.

### Cross-species speaker↔caller analogy (§25 ↔ §26)

Human **words** generalize across speakers (5–7× chance); animal **call-types** barely
generalize across callers (RookID fine Event *below* chance). Animal calls are more
individually stereotyped than human words — the mechanistic root of the dominant individual
signature (bat Emitter V, RookID Source V).

### CPC selective-use policy

Animal pipeline = **fingerprint + HECB + DUSTED + Track C** (no species CPC).
CPC = **human-validation / speaker-invariant only**. Optional animal-CPC ablation on one
dataset only.

## Architecture v1.0 — Complete (2026-06-23)

The full 12-step roadmap is implemented, tested (247 tests), and committed. Here is the
complete module map:

### Core pipeline (unchanged)
- `orchestration.run_pipeline()` → audio → segment → tokenize → diagnose → verdict
- `verdict_engine/` → schema-valid 6-axis report (`build_report` self-validates)
- `embeddings/firewall.py` → ethological firewall (provenance-registered models)

### Segmentation router (the centrepiece — v1.0 frozen)
- `configs/segmentation_router.yaml` — 3 regimes + domain gate + router_audit + regime_classifier
- `orchestration._router_audit()` → `trace["router_audit"]` (regime, choice, alternative, failure mode)
- `diagnostics/repertoire_taxonomy/classifier.py` → `classify_repertoire()` (5 types) +
  `route_by_continuous()` (token-free pre-router: MP recurrence + RQA LAM → skip/rqa/full)
- `segmentation/hierarchical.py` → `hecb_decode(min_parent_confidence=)` — HECB parent gate

### Structural discovery (multi-paradigm, v1.0 frozen)
- `diagnostics/motif_miner/subsequence_motifs.py` → DUSTED (DTW recurrent terms)
- `diagnostics/structural_alternatives/` → RQA (pure numpy) + Matrix Profile (stumpy) +
  PST (pure) + Transition Network (networkx) — 4 paradigms corroborate DUSTED
- `diagnostics/repertoire_taxonomy/` → RepertoireProfile → RepertoireType classification

### H0-CMC evidence ladder
- `diagnostics/h0_cmc/evidence_tiers.py` → EvidenceTier (T0–T5) + H0CMCEvidence + classify_tier()
- `diagnostics/h0_cmc/explicit_implicit_test.py` → CommunicationMode (A vs D vs G)
- `diagnostics/h0_cmc/social_axis.py` → SocialContext + GenericContext + split_context()

### Track C playback
- `experiments/playback_design/audio_splicer.py` → 5-condition counterfactual WAVs
- `experiments/playback_design/naturalness_preflight.py` → acoustic validity screening

### Hardening
- `verdict_engine/evidence_profile.py` → AbstentionReason + abstentions trail (first-class)
- `diagnostics/channel_audit.py` → SNR/reverb/spectral QA
- `diagnostics/temporal_ecology.py` → TemporalContext + timeline extractor
- `representation/vocalisation.py` → AblationResult + summarize_ablation (whole-call-primary)

### The three validated invariants
1. **Router is the centrepiece** — regime-specific segmentation, audited per run.
2. **HECB is an annotator** — parent-quality-gated; whole-call primary confirmed (§30).
3. **Whole-call primary** — HARD invariant; pooled/sequence never beat whole on call-type V.
4. **Three representational jobs** — identity=fp; structure=HECB+DUSTED; human=CPC only.

### The three-bucket thesis (the paper's mature claim)
Every finding lands in exactly one:
- **What acoustics encode**: individual identity (bat Emitter V 0.446, rook Source V 0.31) +
  call/event form (marmoset V 0.599, meerkat V 0.521, Bengalese purity 0.788).
- **What acoustics do NOT encode**: context (bat needs social, §9), social partner (~chance),
  valence (Soundwel within-call-type ≈ chance, §24), word identity (token↔word 0.29, §29).
- **What needs social/biological validation**: bat context (D/A=1.59, socially grounded);
  all meaning (semantic_validation: none until T3+ biology).
