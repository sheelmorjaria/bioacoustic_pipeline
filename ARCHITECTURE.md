# Architecture: representation & segmentation stack

The canonical, frozen shape (2026-06-22). A falsifiable multi-view pipeline for bioacoustic
structure discovery — not a classifier. Each vocalisation is one **hierarchical dual-stream
event**, and the whole stack is organized around that unit.

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

# Or the full pipeline
from bioacoustic_pipeline.orchestration import run_pipeline
report, trace = run_pipeline(paths, species="marmoset", segmentation="hecb")
```
