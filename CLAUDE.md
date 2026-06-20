# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`bioacoustic_pipeline` is a **falsifiable, multi-view pipeline for bioacoustic
structure discovery** — it is explicitly *not* a classifier that emits labels.
It decomposes vocal audio into segments, multi-view representations, and sequence
roles, then produces an auditable **Evidence Profile + Falsification Report**:
each signal carries uncertainty and an explicit statement of how it could be
wrong. Migration of proven math out of a prototype repo; the experimental glue
was left behind.

## Commands

```bash
# Install (editable). Pick extras by what you actually use.
pip install -e ".[dev,ml]"          # dev (pytest, ruff) + torch/sklearn/onnx
pip install -e ".[foundation]"      # transformers + HF hub for AVES v2

# Lint (ruff, line-length 100, target py310; only E402 ignored)
ruff check . && ruff format --check .

# Tests (testpaths=tests, pythonpath=src configured in pyproject)
pytest -q
pytest tests/test_verdict_schema.py -q                       # one file
pytest tests/test_verdict_schema.py::TestEthologicalInvariant -q   # one class/test

# Token-stream -> verdict report (integration entry point)
python scripts/run.py --species bat --tokens-file tokens.json [--with-null]
```

`scripts/validate_track_a.py` runs **Track A** (engineering validation on
LibriSpeech): NBD boundary F1 vs wav2vec2 forced-aligned word starts, the
semantic-ceiling token→word consistency test, and the linear probe (word
identity from CPC latents). It defaults to the fixed-threshold NBD; `--calibrated`
opts into the learned NBD calibrator (refit on 400 LibriSpeech files via
`scripts/refit_nbd_calibrator.py`; default `--kind gbt`). §1 reports **two** F1s:
word-level (~0.6) and **combined / multi-resolution** (~0.84) — combined is the
blueprint's metric and reclaims word-level "false positives" that land on real
phoneme boundaries (an acoustic segmenter finds sub-word structure by design),
so it is the honest segmentation number, not word-F1. The refit default is
`--kind gbt --context cpc`: a HistGradientBoosting calibrator over 13 handcrafted
proposal features **plus 768-D CPC learned-context features** (pre/post/diff of
the CPC frame embedding around each boundary, via `CPCBackbone.embed_frames` +
`proposal_context`). That learned-context layer is what lifts combined-F1 from
~0.79 (scalars) to ~0.84 (at the blueprint's 0.85 target); `--context none` is the
scalar-only, species-portable fallback (CPC is human-only / firewall-blocked on
animals). The tuned `keep_threshold` lives inside the checkpoint
(`BoundaryCalibrator.keep_threshold`) and flows through `segment_audio`/`run_pipeline`
automatically (both build a `CPCBackbone` when the calibrator needs context);
`apply()` clamps the ABSTAIN band to `keep`. **Never feed wav2vec2 features to the
calibrator** — wav2vec2 is the GT yardstick (circular); CPC is the pipeline's own
self-supervised model. `--self-test` runs all three sections on synthetic audio with
no network. Yardstick primitives (linear probe, token→label consistency, wav2vec2
forced alignment returning word + phoneme GT) live in `diagnostics/probes.py` —
measurement instruments, not firewall-registered models.

## Architecture: schema-first, downstream-feeding-upstream

The whole codebase is organized around one invariant contract defined first:

`verdict_engine/schema.py` → `validate_report()` is the canonical output shape
and the test every upstream stage feeds. `build_report()`
(`verdict_engine/engine.py`) **self-validates**: it assembles the report then
calls `validate_report()` and raises on any violation, so no upstream module can
ever emit a malformed verdict. The verdict is **six independent axes**, not a
single label — each categorical with a calibrated confidence
(`evidence_profile.py`):

```
sequence_recombination · functional_aliasing · call_unitization
context_binding       · turn_structure        · semantic_validation
```

Axis scorers take **primitive signals** (`lz_normalized`, `alias_density`,
`alias_pvalue`, `n_hard`, `n_soft`, `n_context_bound`, …), deliberately decoupled
from how those signals are obtained. This keeps the verdict engine pure-stdlib
and unit-testable. `orchestration.synthesize_signals()` is the seam that reduces
real stage outputs into those primitives; `scripts/run.py` is a near-duplicate
CLI wrapper around it.

Data flow:

```
audio → ingestion (manifest QA + species profile)
      → segmentation (NBD proposal → calibrated boundary, with ABSTAIN)
      → embeddings (firewalled multi-view bank, cache-backed)
      → tokenization (acoustic token + affect + corpus-level alias role)
      → diagnostics (LZ, alias detector, null models, bootstrap, entropy)
      → verdict_engine (evidence profile + falsification) → schema-valid report
```

The three heavier modules worth knowing individually:

- **`segmentation/`** — `nbd_proposal` computes per-frame acoustic signals
  (rms, energy_change, spectral_flux, band_flux, rms_slope) and emits candidate
  boundaries; `nbd_calibrator` maps proposal features → P(boundary) via logistic
  regression with an **ABSTAIN band** (keep / abstain / drop) trained on human
  forced-alignment word boundaries. `nbd_evaluation` gives multi-resolution F1
  (word-level + combined, reclaiming word-FPs that land on phoneme GT).
- **`embeddings/`** — `EmbeddingBank` holds backbone adapters (CPC in-house
  human-only, BioMAE in-house bat-only ONNX, AVES foundation via `avex`). Each
  loads weights **lazily** and reports `.available`/`.load_error`; the bank
  skips anything unavailable or that fails the firewall — never silently used.
  `EmbeddingCache` is content-addressed (BLAKE2b of audio bytes + sr) so
  threshold tweaks never trigger re-encoding.
- **`diagnostics/`** — `lempel_ziv` (LZ76 → sequence_recombination axis),
  `alias_detector` (PPMI context vectors + cosine + connected components →
  functional alias groups, unsupervised), `null_models` (permutation tests with
  Phipson-Smyth p-value correction), `bootstrap` (percentile CIs + edge
  stability), `entropy` (segmentation info-gain).

## Non-negotiable invariants (do not weaken these)

1. **Ethological firewall** (`embeddings/firewall.py` +
   `configs/model_metadata.json`). Every model carries provenance
   (pretraining domain, human-speech / text-supervision flags, allowed/forbidden
   uses, `applies_to_species`). `assert_allowed()` / `assert_cross_species()`
   **hard-fail** with `ProvenanceViolation`. The registry is data-driven JSON,
   so adding/restricting a model is a config edit, not a code change. Rules it
   enforces: text-supervised models (NatureLM) are retrieval-only; cross-species
   weight transfer is blocked; no model may do `semantic_verdict`.

2. **Semantic Ceiling.** VQ-VAE tokens are acoustic-similarity classes, *never*
   word identity. Functional aliases group tokens that do the same *job*
   (distributional/contextual equivalence), not that sound the same. Don't add
   code that treats a token id as a meaning.

3. **Semantics can only be validated by Track C (biology).** The schema rejects
   `validated_semantics: true` unless the `semantic_validation` axis is at
   `playback-supported`. Computation alone can never assert meaning — this is
   encoded in both `schema.validate_report()` and `evidence_profile.py`
   (`score_semantic_validation` is hard-coded to that framing).

4. **Statistical honesty.** When a positive structure claim is made, the matching
   falsification tests (null/replication/confound/biology) are emitted alongside
   it (`falsification.py`). Use the permutation test (`null_models`) for
   significance of pooled/thresholded statistics — the file-bootstrap is biased
   downward there (see the `bootstrap.py` docstring).

## Conventions

- Optional heavy deps (`torch`, `sklearn`, `onnxruntime`, `avex`, `yaml`) are
  imported lazily inside functions/`__init__` with `# noqa: PLC0415`, guarded by
  try/except that sets a `.available`/`load_error`. Keep core/verdict/diagnostic
  logic importable without `ml`/`foundation` installed.
- Every source file carries the provenance header
  `# Copyright (c) 2025 Sheel Morjaria. All rights reserved.` — preserve it on
  new files.
- Checkpoints (`checkpoints/`, `*.pt`, `*.onnx`, `*.ckpt`), audio (`data/`),
  and the embedding `cache/` are gitignored and never committed.
- Species sample rates differ by orders of magnitude (human 16 kHz, bat 250 kHz,
  marmoset/corvid 48 kHz, dolphin 96 kHz) — NBD priors live in
  `configs/species_profiles.yaml` as initial values the calibrator refines.
- **Acoustic fingerprint** (`diagnostics/acoustic_fingerprint/`): a physics-grounded,
  species-portable per-segment descriptor (physics + macrotexture + microtexture, linear,
  ~43-D `combined`) that audits learned tokens and separates acoustic similarity from
  functional equivalence. It is **not semantic**. It enters twice: as an embeddings-bank
  view (`AcousticFingerprintBackbone`, `view_class="interpretable"`, firewall all-species
  — the one cross-species representation) and as an opt-in `run_pipeline(fingerprint=True)`
  evidence channel (`trace["acoustic_fingerprint"]`: token coherence + alias divergence +
  null p-values). To avoid anthropomorphism, the default `combined` core is universal-only;
  voiced/harmonic features (F0, HNR, jitter/shimmer) are flagged + zeroed for non-tonal
  segments, and Mel-MFCC (human-perceptual) is opt-in, never in the cross-species view.
