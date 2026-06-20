# bioacoustic_pipeline

A **falsifiable, multi-view pipeline for bioacoustic structure discovery.**

This is not a classifier that emits labels. It decomposes vocal audio into
acoustic segments, multi-view representations, and sequence roles, producing a
diagnostic **evidence profile** with per-axis confidence and explicit
falsification tests. The pipeline does not infer meaning; it generates auditable
hypotheses that must survive null models, bootstrap stability, and behavioral
validation.

```
Audio
  │ Corpus QA + domain manifest
  ▼
NBD acoustic boundary proposal  ──► species-calibrated refiner (CPC-context GBT)
  │ frame-aligned segments + confidence + boundary type
  ▼
Representation bank (firewalled):
  ├── Acoustic Physics Fingerprint (interpretable, all-species, ~43D)
  ├── CPC adapter (256D, human)         ├── BioMAE (112D, bat)
  ├── AVES/Perch (768D, multi-taxa)     └── β-VAE affect (16D, human)
  │ multi-view latent: acoustic token + affect vector + fingerprint cluster + alias role
  ▼
Motif Miner (token · alias-collapsed · acoustic-shape · turn-taking · affective-trajectory)
  │
  ▼
Null models + bootstrap stability
  │
  ▼
Evidence model (7 axes):
  sequence_recombination · functional_aliasing · call_unitization
  context_binding · turn_structure · acoustic_coherence · semantic_validation
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
   model may emit a semantic verdict (biology only). The acoustic fingerprint
   is the one species-agnostic (all-species) view — no training data, no leakage.
4. **Acoustic ≠ semantic.** VQ-VAE tokens are acoustic-similarity classes, not
   word identity (the Semantic Ceiling). Functional aliases group tokens that
   do the same *job*, not sound the same. The β-VAE affect vector is a
   continuous prosodic complement — disentangled, not semantic.
5. **Anti-anthropomorphism.** Voiced/harmonic features (F0, HNR, jitter,
   shimmer) are flagged sub-vectors populated only for tonal segments; Mel-MFCC
   (human-perceptual) is opt-in and excluded from the cross-species core.

## Layout

```
src/bioacoustic_pipeline/
  ingestion/        corpus QA, metadata manifest, firewall schema parser
  segmentation/     NBD proposal generator, boundary calibrator (CPC-context GBT),
                    multi-resolution F1 evaluation
  embeddings/       firewall, embedding bank (CPC · BioMAE · AVES · acoustic fingerprint · β-VAE),
                    cache, CPC encoder/trainer, β-VAE model + backbone
  tokenization/     acoustic token (VQ-VAE on CPC), affect vector, multiview latent
  diagnostics/      alias detector, null models, bootstrap, information theory (LZ, entropy),
                    acoustic_fingerprint/ (physics · macrotexture · microtexture · coherence · nulls),
                    motif_miner/ (token · alias · acoustic · turn · affective · scoring · nulls),
                    probes (linear probe, semantic ceiling, wav2vec2 forced alignment)
  verdict_engine/   evidence profile (7 axes), falsification report, output schema
  orchestration/    end-to-end pipeline: segment → tokenize → diagnose → verdict
  utils/            audio IO, hashing
configs/            species_profiles.yaml, model_metadata.json (firewall)
tests/              verdict-schema TDD + per-module tests (49 tests)
scripts/            run.py · validate_track_a.py · refit_nbd_calibrator.py · train_beta_vae.py
```

## Install (editable)

```bash
cd ~/bioacoustic_pipeline
pip install -e ".[dev,ml]"          # core + torch/sklearn/onnx
pip install -e ".[foundation]"      # transformers (AVES v2 needs `avex` in an
                                    #   isolated env — it pulls TensorFlow)
```

## Commands

### Run the pipeline (audio → verdict)
```bash
# tokens-file mode (legacy)
python scripts/run.py --species bat --tokens-file tokens.json

# audio mode (end-to-end)
python scripts/run.py --species human --audio a.wav --audio b.wav
python scripts/run.py --species human --audio-dir /data/libri
```

### Track A: engineering validation on LibriSpeech
```bash
# self-test (synthetic, no network)
python scripts/validate_track_a.py --self-test

# real validation (downloads wav2vec2 once)
python scripts/validate_track_a.py --n-files 200
python scripts/validate_track_a.py --calibrated --n-files 500 --json out.json
```

### Refit the NBD boundary calibrator
```bash
# refit on LibriSpeech (GBT + CPC learned-context, tuned on combined F1)
python scripts/refit_nbd_calibrator.py --n-files 400 --kind gbt --context cpc

# scalar-only (species-portable, no CPC needed)
python scripts/refit_nbd_calibrator.py --n-files 400 --context none
```

### Train the β-VAE affective stream
```bash
python scripts/train_beta_vae.py   # trains on LibriSpeech CPC latents → 16D affect
```

### Lint + test
```bash
ruff check . && ruff format --check .
pytest -q                           # 49 tests
```

## Track A validated results (LibriSpeech, 150 files)

| Metric | Result |
|--------|--------|
| **§1 combined/multi-resolution F1** (CPC-context GBT) | **0.836** (P=0.77, R=0.92) |
| §1 word-level F1 | 0.614 |
| §2 Semantic ceiling (token→word consistency) | 0.6% (tokens ≠ word identity) |
| §3 Linear probe (word identity from CPC latents) | 35.8% / **71× lift** over 199 classes |
| NBD calibrator combined-F1 progression | 0.67 (fixed-threshold) → 0.79 (scalar GBT) → 0.836 (CPC-context) |

The combined-F1 of 0.836 is at the blueprint's 0.85 target. The semantic ceiling
is robustly demonstrated: the word signal exists in CPC latents (71× lift) but
unsupervised VQ-VAE tokens do not recover it (0.6% consistency).

## Modules

### Acoustic Fingerprint (`diagnostics/acoustic_fingerprint/`)
A physics-grounded, species-portable per-segment descriptor (~43D universal core:
linear spectral shape + ADSR envelope + GLCM texture + modulation + microtexture).
Not semantic — audits learned tokens and separates acoustic similarity from
functional equivalence. Enters twice: as a bank view (`AcousticFingerprintBackbone`,
`view_class="interpretable"`, all-species) and as a verdict evidence channel
(`trace["acoustic_fingerprint"]`: token coherence + alias divergence + null p-values).

### Motif Miner (`diagnostics/motif_miner/`)
Discovers recurring patterns across five views: token n-grams, alias-collapsed
sequences (with **alias_gain** — does functional collapsing reveal more
structure?), acoustic-shape clusters, turn-taking patterns, and affective
trajectories. Each motif scored by NPMI + entropy reduction + null percentile.
`trace["motif_report"]` populates automatically.

### β-VAE Affective Stream (`embeddings/beta_vae.py` + `beta_vae_backbone.py`)
Trained on LibriSpeech CPC latents (256D → 16D, β=4.0 for disentanglement).
The continuous prosodic complement to the VQ-VAE syntactic token — the
dual-stream design. Human-only (firewall: `applies_to_species=["human"]`).

### Ethological Firewall (`embeddings/firewall.py` + `configs/model_metadata.json`)
JSON-driven provenance registry. Hard-fails on forbidden uses. Registered models:
human CPC/VQ-VAE/β-VAE (human-only), BioMAE (bat), AVES/Perch (multi-taxa),
acoustic fingerprint (all-species), NatureLM (retrieval-only). The one
cross-species representation is the handcrafted fingerprint (no training data →
no leakage).

## Status

Migrated from the prototype repository as the **scientific instrument** for
3-track validation:
- **Track A** (engineering): validated on LibriSpeech. §1 combined-F1 = 0.836.
  Semantic ceiling demonstrated. Calibrator refit with CPC-context GBT.
- **Track B** (bioacoustic structure): pipeline operational; awaiting species
  VQ-VAE tokenizers for bat/marmoset (segmentation + fingerprint work today;
  token diagnostics need a species tokenizer).
- **Track C** (biological function): the falsification reports are the
  experimental to-do list for wet-lab / field playback validation.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
