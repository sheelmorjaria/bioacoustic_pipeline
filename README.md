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
Domain / biological-communication gate ──► ESC-50 → NON_BIOACOUSTIC_COMMUNICATION
  │ (refuses ethological axes if out-of-domain)
  ▼
NBD boundary proposal (+ fire-on-drop)  ──► species-calibrated refiner (CPC-context GBT)
  │ frame-aligned segments + confidence + boundary type
  ▼
Sequence-sufficiency gate ──► abstains (not_evaluable) if sequences too sparse
  │
  ▼
Representation bank (firewalled):
  ├── SpectrogramBackbone (deterministic log-mel freq-profile, all-species, species presets)
  ├── Acoustic Fingerprint (interpretable physics+texture, ~43D, all-species)
  ├── CPC adapter (256D, human) · SpeciesCPCBackbone (bat/marmoset)
  ├── BioMAE (112D, bat) · AVES (768D, multi-taxa)
  └── β-VAE affect (16D, human) · Species VQ-VAE (bat, species-tokenizer)
  │ segment-representation source: explicit, configurable, compatibility-checked
  │ dual-stream: VQ-VAE syntax (discrete token) + β-VAE affect (continuous 16D)
  ▼
Diagnostics:
  ├── Context binding (token↔label MI/χ²/Cramér's V, within-speaker null, BH-FDR)
  ├── Functional aliasing (+ speaker/context/frequency-controlled nulls)
  ├── Motif Miner (token · alias-collapsed · acoustic · turn · affective)
  ├── Sequence recombination (LZ76) · Call unitization · Semantic ceiling
  └── Null models + bootstrap stability
  │
  ▼
Evidence model (8 axes + representation-dependence audit + domain gate):
  sequence_recombination · functional_aliasing · call_unitization
  context_binding · turn_structure · acoustic_coherence · motif_structure
  semantic_validation
  │ ethological axes → not_applicable when domain gate rejects
  │ ethological axes → not_evaluable when sequences too sparse
  ▼
Verdict: computational hypothesis + confidence + falsification tests
  │
  ├──► Deterministic reproducibility (Claim A: tri-run byte-identical artifacts)
  └──► Playback design (Phase 1: falsifiable stimulus variants + manifest)
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
   and spectrogram backbones are species-agnostic (all-species) views — no
   training data, no leakage.
4. **Acoustic ≠ semantic.** VQ-VAE tokens are acoustic-similarity classes, not
   word identity (the Semantic Ceiling). Functional aliases group tokens that
   do the same *job*, not sound the same.
5. **Anti-anthropomorphism.** Voiced/harmonic features (F0, HNR, jitter,
   shimmer) are flagged sub-vectors populated only for tonal segments; Mel-MFCC
   (human-perceptual) is opt-in and excluded from the cross-species core.
6. **Dual-stream separation.** VQ-VAE syntax (discrete tokens for sequence structure)
   and β-VAE affect (continuous vectors for graded state) are deliberately NOT fused.
   The `DualStreamSegment` carries both as separate typed fields so cross-stream
   diagnostics can test their relationship without losing interpretability. Late fusion only.
7. **Representation dependence is tested.** Every major diagnostic conclusion is
   evaluated under multiple representations (deterministic spectrogram +
   fingerprint + learned AVES/CPC). Conclusions that survive across
   representations are reported as representation-robust.
8. **Implementation determinism (Claim A).** The pipeline is a frozen inference
   system: same seed + same environment + same inputs → identical outputs. A
   tri-run harness proves this with byte-identical SHA-256 artifact hashes across
   three independent process runs. This is NOT a claim about biological stability
   (Claim B — bootstrap/alt-seed); it is a claim about implementation reproducibility.
9. **Domain gate.** Before any ethological diagnostic, the corpus is classified by
   domain. An environmental-sound benchmark (e.g. ESC-50) is rejected for
   bioacoustic-communication inference — biological-source clips are correctly
   labeled `BIOLOGICAL_SOUND_NON_COMMUNICATION_BENCHMARK`, never forced into
   `NON_BIOLOGICAL_SOUND`, and the ethological axes abstain to `not_applicable`.
   The result: *detects acoustic structure but refuses to infer communication*.
10. **Explicit segment-representation source.** CPC is one option, not a
    prerequisite. The VQ-VAE / β-VAE must receive the same representation they
    were trained on (CPC / BioMAE / fingerprint / spectrogram / fused-bank). The
    source is explicit, configurable, and compatibility-checked before assignment.

## Layout

```
src/bioacoustic_pipeline/
  ingestion/        corpus QA, metadata manifest, species profiles
  segmentation/     NBD proposal (+spectral_drop fire-on-drop), boundary calibrator,
                    multi-resolution F1 evaluation
  embeddings/       firewall, embedding bank, cache, CPC encoder/trainer,
                    BioMAE/AVES/β-VAE backbones, SpeciesCPCBackbone,
                    AcousticFingerprintBackbone, SpeciesAcousticTokenizer,
                    segment_representation (explicit source + compatibility check)
  tokenization/     acoustic token (VQ-VAE on CPC), species tokenizer, multiview latent,
                    FingerprintTokenizerAdapter (torch-free, all-species)
  representation/   SpectrogramBackbone (bank-integrated), DualStreamSegment (typed event)
    spectrograms/   multi-view factory (STFT + log-mel), species presets, quality,
                    shared k-means fingerprint tokenizer
    dual_stream.py  DualStreamSegment + build_dual_stream() factory
  runtime/          device manager (auto/cuda/mps/cpu), module capability registry
  diagnostics/      alias detector (+speaker-controlled nulls), null models, bootstrap,
                    lempel_ziv, entropy, sequence_sufficiency gate, context_binding
                    (MI/χ²/within-speaker null/BH-FDR), token_word calibration,
                    domain_gate (refuses ethological inference on out-of-domain corpora),
                    negative_control_report (ESC-50 + OOD),
                    probes (linear probe, semantic ceiling, forced alignment),
                    acoustic_fingerprint/ (physics · macro · micro · coherence · nulls · tokenizer),
                    motif_miner/ (token · alias · acoustic · turn · affective · nulls),
                    dual_stream_diagnostics (affect-within-token variance, token↔affect MI,
                    alias-affect equivalence, motif affect signatures)
  experiments/
    playback_design/ Phase-1 playback stimuli (exemplar concat + variants + manifest):
                    candidate_selector, motif_variants, stimulus_builder, alias_substitution,
                    affect_morphing (descriptor, not renderer), spatial_router,
                    naturalness_preflight, playback_manifest, response_schema
  verdict_engine/   evidence profile (8 axes + not_applicable + domain), falsification report,
                    output schema
  orchestration/    end-to-end pipeline: segment → tokenize → diagnose → verdict
                    (+ segment_source compatibility gate)
  hashing.py        SHA-256 + canonical JSON (determinism artifacts)
  determinism.py    set_global_determinism + reproducibility manifest
  utils/            audio IO (load + write), hashing
configs/            species_profiles.yaml, model_metadata.json (firewall),
                    pipeline_deterministic.yaml, esc50_negative_control.yaml,
                    segment_representation.yaml
tests/              171 tests (schema TDD + per-module + pipeline integration +
                    determinism + domain gate + playback design + segment representation)
scripts/            17 analysis + training + reproducibility + playback scripts (see Commands)
```

## Install (editable)

```bash
pip install -e ".[dev,ml]"          # core + torch/sklearn/onnx
pip install -e ".[foundation]"      # transformers (AVES needs `avex`; GPU recommended)
```

## Commands

### Run the pipeline (audio → verdict)
```bash
python scripts/run.py --species human --audio-dir /data/libri [--with-null]
```

### Track A: engineering validation on LibriSpeech
```bash
python scripts/validate_track_a.py --self-test              # synthetic, no network
python scripts/validate_track_a.py --calibrated --n-files 500
```

### Track B: bat (corrected, annotation-grounded)
```bash
python scripts/bat_segmentation_sweep.py --n-files 400      # plausibility-select profile
python scripts/track_b_bat_corrected.py --n-files 1500      # full corrected verdict
python scripts/bat_representation_crosscheck.py --n-bouts 2000  # 3-rep robustness
```

### Track B: marmoset (call-type labels)
```bash
python scripts/track_b_marmoset.py --rep fingerprint --n-per-type 2500  # deterministic
python scripts/track_b_marmoset.py --rep aves --n-per-type 2500        # learned cross-check (GPU)
```

### Cross-species + representation-dependence
```bash
python scripts/cross_species_fingerprint.py --n-human 500 --n-bat 4000  # shared codebook
python scripts/cross_species_aves.py --n-human 250 --n-bat 2000        # AVES 3rd representation
python scripts/control_token_matched.py --n-files 350                   # matched-size human control
python scripts/token_word_calibration.py --n-files 300                  # semantic ceiling
```

### Determinism + reproducibility (Claim A)
```bash
python scripts/deterministic_run.py --seed 1337 --n-files 20 --out-dir repro_runs/run_1
python scripts/reproducibility_tri_run.py --seed 1337 --n-files 20      # 3× byte-identical check
```

### Negative control (ESC-50 environmental sound)
```bash
python scripts/esc50_negative_control.py --source synthetic --n-files 40 \
    --report outputs/esc50_negative_control_report.json
python scripts/esc50_negative_control.py --source beans-zero --n-files 200  # real BEANS-Zero (HF)
```

### Playback design (Phase 1: falsifiable stimuli + manifest)
```bash
python scripts/playback_design.py --species synthetic --experiment motif_disruption \
    --n-files 40 --out-dir playback_out
python scripts/playback_design.py --species synthetic --experiment alias_substitution \
    --n-files 60 --out-dir playback_out
```

### Train species tokenizers
```bash
python scripts/refit_nbd_calibrator.py --n-files 400 --kind gbt --context cpc
python scripts/train_bat_vqvae.py --audio-dir /data/bat --n-files 500
python scripts/train_beta_vae.py
```

### Lint + test
```bash
ruff check . && ruff format --check .
pytest -q                           # 171 tests
```

## Validated results

### Track A (LibriSpeech, 150 files)

| Metric | Result |
|--------|--------|
| **§1 combined/multi-resolution F1** (CPC-context GBT) | **0.836** (P=0.77, R=0.92) |
| §2 Semantic ceiling (token→word consistency) | 0.6% — tokens ≠ word identity |
| §3 Linear probe (word identity from CPC latents) | 35.8% / **71× lift** over 199 classes |
| CPC latent → speaker probe | 0.56 (speaker-dominated representation) |

### Track B: Bat (Egyptian fruit bat, 23k segments, within-Emitter null)

| Representation | context_binding nmi | Cramér's V | p |
|---------------|---------------------|------------|---|
| Fingerprint (deterministic physics) | 0.027 | 0.148 | 0.007 |
| **Spectrogram** (deterministic, bat-aware) | **0.038** | **0.197** | 0.007 |
| AVES (learned multi-taxa SSL) | 0.030 | 0.162 | 0.007 |

**Representation-ROBUST: 3/3 significant, within-Emitter-controlled.** The
deterministic spectrogram is strongest — the context association is grounded in
acoustic physics.

### Track B: Marmoset (common marmoset, 12.5k calls, 5 call types)

| Representation | token↔call-type V | probe acc | p |
|---------------|-------------------|-----------|---|
| Fingerprint (deterministic) | 0.606 | 0.83 | 0.007 |
| **AVES** (learned multi-taxa SSL) | **0.698** | **0.98** | 0.007 |

**Representation-ROBUST: 2/2 significant.** Call types (Phee/Twitter/Trill/Tsik/Seep)
recovered by both deterministic and learned representations.

### Functional aliasing (representation-dependent)

| Tokenization | human alias density | bat alias density |
|-------------|--------------------|-------------------|
| Per-species VQ-VAE | 4% | 0% |
| Shared fingerprint | 0% | 17.5% |

Functional aliasing **flips sign** between representations → it is **not a species
discriminator**. This is a reviewer-safe negative result about representation dependence.

### Determinism (Claim A: implementation reproducibility)

| Artifact | Run 1 | Run 2 | Run 3 | Status |
|----------|-------|-------|-------|--------|
| verdict.json | identical | identical | identical | ✅ |
| tokens.jsonl | identical | identical | identical | ✅ |
| motifs.json | identical | identical | identical | ✅ |
| *(all 9 artifacts)* | — | — | — | **all IDENTICAL** |

Three independent subprocess runs under a fixed seed (1337), CPU, deterministic
DSP (torch-free), seeded null procedures, sorted inputs → **byte-identical** SHA-256
artifact hashes. NOT a biological stability claim (Claim B); purely implementation
reproducibility.

### Negative control: ESC-50 (BEANS-Zero subset)

| Metric | Result |
|--------|--------|
| Final verdict | **NON_BIOACOUSTIC_COMMUNICATION** |
| Pipeline outputs | NBD segments + fingerprints + tokens **computed** |
| Biological axes | functional_aliasing, motif, turn, context, semantic → **not_applicable** |
| Biological-source clips (dog, crickets, …) | `BIOLOGICAL_SOUND_NON_COMMUNICATION_BENCHMARK` |
| Non-biological clips (rain, engine, …) | `NON_BIOLOGICAL_SOUND` |

The pipeline detects acoustic structure (segments, fingerprints, tokens) but the
domain gate **refuses** ethological-communication inference — the strong negative control.

## Modules

### Sequence-Sufficiency Gate (`diagnostics/sequence_sufficiency.py`)
Prevents false biological nulls. When segmentation produces too few tokens per bout
(e.g., the bat holophrastic profile: 56% of bouts yielded zero segments), the
sequence-dependent axes abstain to `not_evaluable` instead of returning a misleading
"absent". Wired through schema → evidence-profile → engine → orchestration.

### Domain Gate (`diagnostics/domain_gate.py` + `negative_control_report.py`)
Runs **before** ethological diagnostics. Classifies the corpus domain and refuses
communication inference on out-of-domain corpora (ESC-50 → `NON_BIOACOUSTIC_COMMUNICATION`).
Three-class verdict taxonomy separates clip *source* (biological vs non-biological)
from *communication-corpus suitability*. The 5 ethological axes abstain to
`not_applicable`; the 3 acoustic axes still run. Additive schema: `not_applicable`
is a new axis level (distinct from `not_evaluable`).

### Acoustic Fingerprint (`diagnostics/acoustic_fingerprint/`)
Physics-grounded, species-portable per-segment descriptor (~43D universal core:
linear spectral shape + ADSR envelope + GLCM texture + modulation + microtexture).
Not semantic — audits learned tokens and separates acoustic similarity from
functional equivalence. Enters twice: as a bank view (`AcousticFingerprintBackbone`,
`view_class="interpretable"`, all-species) and as a verdict evidence channel.

### Spectrogram Bridge (`representation/spectrograms/`)
Multi-view time-frequency representation (linear STFT + log-mel) with species-aware
presets (bat: 2 ms / 120 kHz / 64 mel; marmoset: 10.7 ms / 48 kHz / 128 mel;
human: 32 ms / 8 kHz / 80 mel). `SpectrogramBackbone` conforms to the Backbone ABC,
gated by the firewall, cached by the bank. The bridge audits learned representations.

### Segment-Representation Source (`embeddings/segment_representation.py`)
Makes the representation that feeds VQ-VAE/β-VAE heads **explicit and checked**.
`resolve_source(species, category)` → firewall-checked `SegmentSource`.
`assert_compatible(head, source)` gates assignment (`head.input_source == source.category`).
CPC is one option; BioMAE / fingerprint / spectrogram / fused-bank are others.
Categories: `cpc | biomae | acoustic_fingerprint | spectrogram | embedding_bank_fusion`.
`run_pipeline(segment_source=...)` records `trace["segment_representation"]`.
`FingerprintTokenizerAdapter` provides torch-free all-species tokens.

### Context Binding (`diagnostics/context_binding/`)
Annotation-grounded test: is the per-segment token distribution conditioned on a
behavioral label (Context/Emitter/Addressee)? Mutual information + chi-square +
Cramér's V + permutation null + **within-speaker null** (kills the speaker confound)
+ Benjamini–Hochberg FDR.

### Speaker-Controlled Alias Null (`alias_detector.controlled_alias_tests`)
Functional aliasing must survive within-file, frequency-preserving, within-Emitter,
and within-Context shuffles. An alias result is accepted only if it survives every
null — preventing "different speakers produce different variants" from being misread
as "different tokens fill the same role".

### Playback Design — Phase 1 (`experiments/playback_design/`)
Turns diagnostic outputs into **falsifiable playback-experiment stimuli + manifest**.
Exemplar concatenation (real segment audio, no neural synthesis): motif variants
(reverse/shuffle/delete/insert/random), alias substitution, naturalness preflight
(fingerprint cosine + DTW + spectrum distance), spatial router (VBAP + labels),
affect trajectory descriptors (manifest-level, not rendered). Every output is a
hypothesis for live (Track C) validation — manifest carries explicit caveats:
no Turing-test, no grammar, no semantic-decoding claims.

### Determinism + Reproducibility (`determinism.py` + `hashing.py`)
Frozen-inference determinism. `set_global_determinism(seed)` seeds python/numpy/torch
(guarded) + deterministic-algorithm mode. `build_repro_manifest` captures git commit,
versions, device, dataset hash, NBD config, tokenizer spec. `canonical_json` =
key-order-independent serialization. Tri-run harness proves byte-identical artifacts.

### Ethological Firewall (`embeddings/firewall.py` + `configs/model_metadata.json`)
JSON-driven provenance registry. Hard-fails on forbidden uses. Registered models:
human CPC/VQ-VAE/β-VAE (human-only), bat_cpc/bat_vqvae (bat), marmoset_cpc
(marmoset, **collapsed** — cosine ~0.99 across call types), BioMAE (bat),
AVES (multi-taxa), acoustic_fingerprint + spectrogram (all-species, deterministic).

### Runtime Layer (`runtime/`)
Device manager (`resolve_device("auto")` → cuda/mps/cpu) + module capability
registry (CPU: segmentation, fingerprint, spectrogram, diagnostics, verdict;
GPU: CPC, BioMAE, AVES, VQ-VAE, β-VAE). Principle: *accelerate tensors, not
decisions.*

## Status

- **Track A** (engineering): validated on LibriSpeech. §1 combined-F1 = 0.836.
  Semantic ceiling demonstrated. CPC latent is speaker-dominated (0.56 > word 0.40).
- **Track B** (bioacoustic structure): **representation-robust context_binding**
  for both bat (3/3 representations, p=0.007) and marmoset (2/2 representations,
  p=0.007). Functional aliasing is representation-dependent (negative result).
  Species tokenizers operational (bat VQ-VAE 70% codebook; marmoset_cpc collapsed —
  fingerprint/AVES used instead).
- **Track C** (biological function): the falsification reports are the experimental
  to-do list for wet-lab / field playback validation. Phase-1 playback design
  generates the stimulus variants + manifest.
- **Determinism** (Claim A): tri-run byte-identical — the pipeline implementation
  is reproducible.
- **Negative control** (ESC-50): domain gate refuses communication inference —
  detects structure but does not overclaim.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
