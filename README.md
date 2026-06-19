# bioacoustic_pipeline

A **falsifiable, multi-view pipeline for bioacoustic structure discovery.**

This is not a classifier that emits labels. It decomposes vocal audio into
acoustic segments, multi-view representations, sequence roles, and a diagnostic
**evidence profile** that generates falsifiable hypotheses for behavioral
validation. The pipeline does not infer meaning; it produces auditable signals
combined probabilistically, each carrying uncertainty and an explicit statement
of how it could be wrong.

```
Audio
  │ Corpus QA + domain manifest
  ▼
NBD acoustic boundary proposal generator  ──► species-calibrated refiner (ABSTAIN)
  │ frame-aligned segments + confidence + boundary type
  ▼
Embedding bank  (CPC · BioMAE · AVES/Perch · NatureLM-retrieval)   [firewalled]
  │ multi-view latent: acoustic token + affect + motif + functional-alias role
  ▼
Null models + bootstrap stability
  │
  ▼
Evidence model  (call-level · sequence · alias · context)
  │
  ▼
Verdict:  computational hypothesis + confidence + falsification tests
```

## Design principles

1. **No single model produces "the answer."** Each module emits an auditable
   signal; signals combine probabilistically.
2. **Falsifiable, not conclusive.** Every report states how it could be wrong
   (permutation nulls, bootstrap CIs, Track-C playback tests).
3. **Ethological firewall.** Every model carries provenance (domain, human
   speech, text supervision, allowed/forbidden uses). Text-supervised models
   (NatureLM) are retrieval-only; cross-species weight transfer is blocked; no
   model may emit a semantic verdict (biology only).
4. **Acoustic ≠ semantic.** VQ-VAE tokens are acoustic-similarity classes, not
   word identity (the Semantic Ceiling). Functional aliases group tokens that
   do the same *job*, not sound the same.

## Layout

```
src/bioacoustic_pipeline/
  ingestion/      corpus QA, metadata manifest, firewall schema parser
  segmentation/   NBD proposal generator, boundary calibrator (ABSTAIN)
  embeddings/     firewall, embedding bank, CPC/BioMAE/AVES adapters, cache
  tokenization/   acoustic token, affect vector, functional-alias role
  diagnostics/    alias detector, null models, bootstrap, information theory
  verdict_engine/ evidence profile, falsification report, output schema
  utils/          audio IO, hashing
configs/          species_profiles.yaml, model_metadata.json (firewall)
tests/            verdict-schema TDD + per-module tests
scripts/          Track A/B entry points
```

## Install (editable)

```bash
cd ~/bioacoustic_pipeline
pip install -e ".[dev,ml]"          # core + torch/sklearn/onnx
pip install -e ".[foundation]"      # transformers (AVES v2 needs `avex` in an
                                    #   isolated env — it pulls TensorFlow)
```

## Validate (Track A: engineering)

```bash
pytest -q                           # verdict schema + module tests
python scripts/validate_track_a.py  # LibriSpeech: NBD F1 + semantic ceiling
```

## Status

Migrated from the prototype repository as the **scientific instrument** for
3-track validation (A: engineering, B: bioacoustic structure, C: biological
function). The proven math (spectral flux, PPMI aliasing, permutation nulls,
calibrator) is cherry-picked; the experimental glue is intentionally left
behind in the prototype.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
