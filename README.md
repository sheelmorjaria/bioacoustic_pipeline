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
  ├── Channel audit (SNR / reverb / spectral attenuation)
  ▼
Domain / biological-communication gate ──► ESC-50 → NON_BIOACOUSTIC_COMMUNICATION
  │ (refuses ethological axes if out-of-domain)
  ▼
SEGMENTATION ROUTER (the centrepiece — v1.0 frozen)
  ├── Token-free pre-router (MP recurrence + RQA LAM → skip/rqa/full)
  ├── Regime: discrete calls in noise → PCEN+CRNN (F1=0.703) → HECB internal annotation
  ├── Regime: continuous speech-like → NBD+CPC calibrator (combined-F1=0.836)
  └── Regime: exploratory/unknown → fixed-threshold NBD + ABSTAIN
  │ trace["router_audit"] (regime, choice, alternative, failure mode)
  │ HECB parent-quality gate (min_parent_confidence abstains on unreliable parents)
  │ frame-aligned segments + confidence + boundary type
  ▼
Sequence-sufficiency gate ──► abstains (not_evaluable) if sequences too sparse
  │
  ▼
Representation bank (firewalled):
  ├── Acoustic Fingerprint (interpretable physics+texture, ~43D, all-species) ← cross-species
  ├── SpectrogramBackbone (deterministic log-mel, all-species, species presets)
  ├── CPC adapter (256D, human) · SpeciesCPCBackbone (bat/marmoset)
  ├── BioMAE (112D, bat) · AVES (768D, multi-taxa)
  └── β-VAE affect (16D, human) · Species VQ-VAE (bat, species-tokenizer)
  │ segment-representation source: explicit, configurable, compatibility-checked
  ▼
Diagnostics (multi-paradigm, v1.0 frozen):
  ├── DUSTED (DTW-with-indels recurrent acoustic terms + null + bootstrap)
  ├── Structural Alternatives (4 paradigms corroborate DUSTED):
  │   ├── RQA (dynamical systems: %DET, %LAM — graded-regime lens)
  │   ├── Matrix Profile (continuous recurrence — degenerate-regime pre-filter)
  │   ├── PST (variable-order Markov — discrete-syntax control)
  │   └── Transition Network (graph topology — modularity + path length)
  ├── Repertoire Taxonomy (5 types: discrete / graded / sparse / stereotyped / human)
  ├── Context binding · Functional aliasing · LZ76 · Call unitization
  ├── Channel audit · Temporal ecology (bout structure, turn-taking, circadian)
  └── Null models + bootstrap stability + AblationResult (whole-call primary)
  │
  ▼
Evidence model (6 axes + abstention trail + domain gate):
  sequence_recombination · functional_aliasing · call_unitization
  context_binding · turn_structure · acoustic_coherence · motif_structure
  semantic_validation
  │ AbstentionReason trail (too_few_calls / context_labels / domain_rejected / ...)
  │ ethological axes → not_applicable when domain gate rejects
  │ ethological axes → not_evaluable when sequences too sparse
  ▼
Verdict: computational hypothesis + confidence + falsification tests
  │
  ├──► H0-CMC Evidence Ladder (T0 acoustic → T1 context → T2 playback → T3-T5 wet lab)
  │      explicit-implicit test (A vs D vs G → CommunicationMode)
  │      social axis (SocialContext separate from GenericContext)
  │
  ├──► DSP Audio Splicer (Track C: 5-condition counterfactual WAVs)
  │      natural · order_shuffled · onset_offset_swap · caller_swapped · amplitude_matched
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
11. **The Entanglement Ceiling — the semantic ceiling is mathematical (§36).** Caller
    identity and call form are non-separably entangled in the acoustic signal at every
    level — static spectral, dynamic temporal, linear, affine, and non-linear. Four
    disentanglement methods (LDA / LogReg projection, Z-normalization, delta-trajectory)
    all *fail*: removing caller information destroys call-type structure (up to 4× harder).
    The motor pattern *itself* is caller-specific ("the medium is the message"). Only a
    learned non-linear transform (CPC) can separate them — and the ethological firewall
    blocks that on animals. The correct response is the whole-call-primary invariant:
    use the entangled signal as-is, and let only Track-C biology (not computation)
    validate meaning. This is the paper-defining negative result.
12. **The three-bucket thesis.** Vocal communication splits cleanly into three
    epistemic buckets the pipeline treats differently: (1) **what acoustics encode** —
    caller identity + call form, recoverable by the fingerprint and HECB; (2) **what
    acoustics do not encode** — social context, affective valence, and meaning,
    which require social/biological covariates (the H0-CMC D ≫ A test) or wet-lab
    validation; (3) **what needs Track-C biology to validate** — no computational
    result, however statistically clean, can assert meaning; only playback/field
    experiments lift `semantic_validation` above `none`.

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
  representation/   SpectrogramBackbone (bank-integrated), DualStreamSegment (typed event),
                    AblationResult + summarize_ablation (whole-call-primary invariant),
                    disentangle.py (caller-identity disentanglement: LDA / LogReg / Z-norm / delta)
    spectrograms/   multi-view factory (STFT + log-mel), species presets, quality,
                    shared k-means fingerprint tokenizer
    dual_stream.py  DualStreamSegment + build_dual_stream() factory
    vocalisation.py DualStreamVocalisation + whole_sub_pooled_sequence_ablation +
                    AblationResult (winner + claim_strength)
    subcall.py       build_subcall_representations + pooled_representation
  runtime/          device manager (auto/cuda/mps/cpu), module capability registry
  diagnostics/      alias detector (+speaker-controlled nulls), null models, bootstrap,
                    lempel_ziv, entropy, sequence_sufficiency gate, context_binding
                    (MI/χ²/within-speaker null/BH-FDR), token_word calibration,
                    domain_gate (refuses ethological inference on out-of-domain corpora),
                    negative_control_report (ESC-50 + OOD),
                    probes (linear probe, semantic ceiling, forced alignment),
                    acoustic_fingerprint/ (physics · macro · micro · coherence · nulls · tokenizer),
                    motif_miner/ (token · alias · acoustic · turn · affective · nulls
                                 + subsequence_motifs: DUSTED DTW · fixed-cosine · exact-ngram
                                 + family-entropy/top-family/label-purity diagnostics),
                    dual_stream_diagnostics (affect-within-token variance, token↔affect MI,
                    alias-affect equivalence, motif affect signatures),
                    h0_cmc/ (T0→T5 evidence tiers + explicit-implicit test + social axis),
                    repertoire_taxonomy/ (5-type classifier + route_by_continuous pre-router),
                    structural_alternatives/ (RQA · Matrix Profile · PST · Transition Network),
                    channel_audit.py (SNR / reverb / spectral attenuation),
                    temporal_ecology.py (bout structure, turn-taking, circadian)
  experiments/
    playback_design/ Phase-1 playback stimuli (exemplar concat + variants + manifest):
                    candidate_selector, motif_variants, stimulus_builder, alias_substitution,
                    affect_morphing (descriptor, not renderer), spatial_router,
                    naturalness_preflight, playback_manifest, response_schema,
                    audio_splicer (DSP: 5-condition counterfactual WAVs: natural /
                    order_shuffled / onset_offset_swap / caller_swapped / amplitude_matched)
  verdict_engine/   evidence profile (8 axes + not_applicable + domain + abstentions trail),
                    falsification report, output schema, AbstentionReason vocabulary
  orchestration/    end-to-end pipeline: segment → tokenize → diagnose → verdict
                    (+ segment_source compatibility gate + _router_audit + trace["router_audit"])
  hashing.py        SHA-256 + canonical JSON (determinism artifacts)
  determinism.py    set_global_determinism + reproducibility manifest
  utils/            audio IO (load + write), hashing
configs/            species_profiles.yaml, model_metadata.json (firewall),
                    pipeline_deterministic.yaml, esc50_negative_control.yaml,
                    segment_representation.yaml
tests/              247 tests (schema TDD + per-module + pipeline integration +
                    determinism + domain gate + playback design + segment representation +
                    DUSTED subsequence motifs + repertoire taxonomy + h0_cmc +
                    structural alternatives)
scripts/            30 analysis + training + reproducibility + playback + motif scripts (see Commands)
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

### Cross-corpus DUSTED recurrent acoustic-term discovery
```bash
# Marmoset (HECB sub-call units, oracle)
python scripts/hecb_dusted_marmoset.py --n-eval 1500

# Bengalese finch (GT syllable-validated positive control)
python scripts/dusted_bengalese.py --birds 0 1 2

# Egyptian fruit bat (bout-level, emitter vs context)
python scripts/dusted_bat.py --n-bouts 2000

# Bat held-out emitter validation (70/30 split)
python scripts/bat_heldout.py --n-bouts 2000

# RookID individual-ID (whole-call vs HECB, cross-session held-out)
python scripts/rookid_individual_id.py --max-calls 3000

# RookID-DUSTED motif discovery
python scripts/rookid_dusted.py --max-calls 2000
```

### Unsupervised word-structure benchmark (LibriSpeech)
```bash
# §14: HECB vs HuBERT vs CPC vs fingerprint same-word consistency
python scripts/unsupervised_word_structure_benchmark.py --n-files 200

# Stage C: DUSTED word-term discovery (DTW, HuBERT's native metric)
python scripts/dusted_librispeech_words.py --n-files 200

# Stage D: ES-KMeans-style word-embedding clustering
python scripts/eskmeans_librispeech.py --n-files 200

# §16: HuBERT sanity ablation (layer × k × collapse × pooling grid)
python scripts/hubert_ablation.py --n-files 200
```

### Lint + test
```bash
ruff check . && ruff format --check .
pytest -q                           # 247 tests
```

### Architecture v1.0: boundary sweep + alternatives + taxonomy + playback + controls
```bash
# §30: boundary-quality sweep (whole-call primary, meerkat + marmoset)
python scripts/boundary_quality_sweep.py --n-clips 300 --n-per-type 500

# §33: DUSTED vs 4 structural alternatives (per-dataset regime)
python scripts/compare_structure_methods.py --n-files 20

# §34: targeted reanalyses (bat RQA-by-context / marmoset MP / Bengalese network)
python scripts/bat_rqa_by_context.py --per-context 30 --top-k 6
python scripts/marmoset_matrix_profile_sweep.py --n-calls 200
python scripts/bengalese_network_topology.py --n-files 100

# Step 3: repertoire taxonomy + Step 4: identity-form-context
python scripts/repertoire_taxonomy.py
python scripts/identity_form_context_disentanglement.py

# §36 Entanglement Ceiling — AVES ablation (reviewer-proofing: deep SSL control)
python scripts/entanglement_ceiling_aves.py --n-per-type 500 --min-calls-per-caller 20

# §37 Compositional-structure mining (structural prerequisites T0 + T2 stimuli)
python scripts/compositional_candidates_bat.py --n-bouts 2000 --k 24

# §38 Compositional scan — zebra finch call-type sequences (filename-only, instant)
python scripts/compositional_zebrafinch.py --age adult

# Step 6: DSP Audio Splicer (5-condition counterfactual WAVs)
python scripts/playback_splice.py --n-bouts 20 --out outputs/playback_stimuli

# §35: negative controls (synthetic chirps / random segments)
python scripts/extended_negative_controls.py --n-bouts 50

# Step 11: stereotypy vs template (cross-caller retrieval)
python scripts/stereotypy_vs_template.py --n-per-type 200

# Factorial: HECB typed vs scalar+CPC on LibriSpeech boundary F1
python scripts/hecb_calibrator_factorial.py --n-train 400 --n-test 100
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

### Cross-corpus DUSTED recurrent acoustic-term discovery (§11–§13, §18–§19)

| corpus | regime | inventory | key finding |
|---|---|---|---|
| **Bengalese finch** (§12) | discrete syllable song | diverse, high-purity | fixed-cosine wmean purity **0.895**; DTW recovers `[4,5]` at purity **1.00** |
| **Egyptian fruit bat** (§13) | graded bout vocalisations | graded/intermediate | Emitter V **~0.46** ≫ Context V **~0.16** (2.8×); fixed-cosine headline |
| **InfantMarmosetsVox** (§11) | discrete calls (oracle HECB) | sparse, one dominant | DTW calltype V **0.51** = 4.7× exact; degenerate inventory |
| **RookID** (§19) | noisy outdoor corvid calls | degenerate (one dominant) | Event V **0.22** > Source V **0.08**; motifs are call-type, not individual |

**The difference is the corpus/repertoire, not the method** — DUSTED lands where predicted
by vocalization structure. Terms = recurrent acoustic structure, NOT words.

### Bat held-out emitter validation (§17)

| metric | Train (70%) | Held-out (30%) |
|---|---|---|
| Emitter V | 0.221 | **0.375** |
| Context V | 0.085 | 0.149 |
| **ratio** | 2.60× | **2.52×** |

The emitter/caller claim **survives held-out** — not a session/file artifact.

### RookID individual-ID (§18)

| representation | Source V | macro-F1 | held-out control |
|---|---|---|---|
| **whole-call fp (44-D)** | **0.384** | **0.287** (4× chance) | recording-date held-out |
| HECB-pooled fp (44-D) | 0.321 | 0.263 | (sub-call does NOT improve ID) |

### LibriSpeech unsupervised word-structure benchmark (§14–§16)

| unit source | within/across ratio | NN acc | word-cluster V |
|---|---|---|---|
| **fingerprint (44-D)** | **0.964** | **0.147** | **0.517** |
| CPC-km(50) | 0.954 | 0.191 | 0.436 |
| HECB | 0.976 | 0.089 | 0.392 |
| HuBERT-km(50) | 1.008 (fails) | 0.052 | 0.338 (fails) |

**HuBERT is metric-dependent**: fails Levenshtein/clustering but recovers under DTW
(Stage C: word_V 0.418). 24-config ablation (§16) confirms: no HuBERT config beats
fingerprint. The interpretable fingerprint + in-house CPC lead on every test.

### Boundary-quality sweep — whole-call primary confirmed (§30)

| Corpus | Parent F1 | whole V | pooled V | gap (pooled−whole) | Verdict |
|---|---|---|---|---|---|
| MeerKAT (n=300, oracle) | 1.000 | 0.364 | 0.386 | +0.021 | hierarchy barely helps |
| Marmoset (n=2500, oracle) | — | 0.524 | 0.492 | **−0.032** | hierarchy HURTS |

Sub-call internal structure is augmentation for motif discovery, NOT a call-type-recovery
lever. `AblationResult` formalizes the whole-call-primary invariant.

### Multi-paradigm structural validation (§33–§34)

DUSTED corroborated by 4 independent mathematical paradigms:

| Paradigm | Dataset | Key result | Verdict vs DUSTED |
|---|---|---|---|
| PST + Graph | Bengalese | 247 contexts, 5 communities, clustering 0.686 | **Corroborates** |
| RQA | Bat | %LAM context-dependent (**p=0.016**); %DET invariant | **Complements** (dynamics DTW missed) |
| Matrix Profile | Marmoset | 87.5% stutter-dominant, unimodal histogram | **Confirms** degenerate, token-free |
| Transition Network | LibriSpeech | 639 nodes, 19 communities, path 1.257 | **Upper bound** (scales) |

### Cross-species repertoire taxonomy (§31)

| Corpus | Type | Discriminator |
|---|---|---|
| Bengalese finch | `discrete_syllable` | cov 0.48, null 2.67×, bootstrap 0.509 |
| Egyptian fruit bat | `graded` | cov 0.15, emitter V 0.45, social_lift 1.74 |
| Infant marmoset | `sparse_repetitive` | cov 0.028, bootstrap 0.229 |
| Rook | `individually_stereotyped` | Source V 0.31 ≫ social_other 0.04 |
| LibriSpeech | `human_lexical` | terms are NOT words (semantic ceiling) |

### Identity–form–context disentanglement (§32)

| Target | Bat | Marmoset | Rook | LibriSpeech |
|---|---|---|---|---|
| Individual ID | **strong** (V 0.45) | n/a | partial (V 0.31) | strong (probe 0.56) |
| Call/event type | weak (V 0.17) | **strong** (V 0.60, probe 0.83) | **strong** (V 0.67) | partial (token↔word 0.29) |
| Context | **needs social** (D 4.3×) | n/a | weak (V 0.04) | n/a |
| Social partner | n/a | n/a | n/a (≈chance) | n/a |

### Negative controls — pipeline specificity (§35)

| Control | # motifs | vs natural (33) | Specificity |
|---|---|---|---|
| Synthetic FM chirps | **1** | 33× | PASS |
| Random audio segments | **1** | 33× | PASS |

The pipeline does NOT manufacture structure from arbitrary input.

### The Entanglement Ceiling — semantic ceiling is mathematical (§36)

Four methods tested to separate caller identity from call form in the 44-D fingerprint:

| Method | Mathematics | Δ caller V | Δ form V | Result |
|---|---|---|---|---|
| LDA projection | Linear subspace | −0.029 | **−0.120** | FAIL (overlap) |
| LogReg projection | Linear weight | −0.005 | −0.013 | FAIL (inseparable) |
| Z-normalization | Affine | −0.019 | **−0.074** | FAIL (overlap) |
| Delta-trajectory | Non-linear | −0.105 | **−0.163** | FAIL (temporal dynamics are caller-specific) |

**All four fail.** Caller identity and call form are non-separably entangled at every level
(static spectral, dynamic temporal, linear, non-linear). Only a learned non-linear transform
(CPC) can separate them — and the firewall blocks that on animals. The whole-call-primary
invariant is the correct response: use the entangled signal as-is.

**Learned-representation control (AVES, §36).** The same linear disentanglement was repeated
on a 768-D deep SSL embedding (AVES, multi-taxa, firewall-legal for marmoset) over the same
calls. Deep learning *narrows* the ceiling (AVES form/caller ratio 2.78 vs fingerprint 2.00)
but does **not** break it: Z-normalization still sacrifices form ~3× faster than caller, and
linear LDA is a no-op (AVES encodes caller non-linearly). No firewall-compliant available
representation linearly separates identity from form. Script: `scripts/entanglement_ceiling_aves.py`.

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

### DUSTED Recurrent Subsequence Motifs (`motif_miner/subsequence_motifs.py`)
Adapts spoken-term-discovery (DUSTED) to **recurrent acoustic terms** in animal
vocalisations — NOT words. Each vocalisation = one `CallUnits` (ordered internal
sub-call units). Windows (L=2–6) are matched by three primitives: exact n-grams
(fragments), fixed-length cosine (catches substitutions), DTW with indels (catches
indels — the main method). Role-aware length-normalised DTW with centroid-cosine
prefilter so full all-pairs DTW is never run. Per-motif diagnostics: `label_purity`
(GT validation), `family_entropy` / `top_family_support_fraction` (diverse vs degenerate
discriminator), `calltype_enrichment` / `caller_enrichment` (Cramér's V, association
NOT semantics), order-shuffle null, bootstrap Jaccard. 10 unit tests. Runs across
4 animal corpora (marmoset/bat/Bengalese finch/rook) + LibriSpeech as a cross-species
motif arc (§11–§19).

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

### H0-CMC Evidence Ladder (`diagnostics/h0_cmc/`)
Six-rung causal evidence ladder (T0 acoustic → T5 field) connecting discovery
to biological validation. `EvidenceTier` + `H0CMCEvidence` dataclass +
`classify_tier()` autonomous classifier (never claims above T1 without biology).
`explicit_vs_implicit_test()` classifies signal as explicit-acoustic (A≈G) vs
socially-grounded-implicit (D≫A, D≈G). `SocialContext` / `GenericContext`
separate social covariates from generic context. Maps to `semantic_validation`
axis: T0–T2 → none, T3 → candidate, T4 → causally_supported, T5 → validated.

### Structural Alternatives (`diagnostics/structural_alternatives/`)
Four firewall-compliant paradigms that corroborate DUSTED from independent
mathematics: **RQA** (pure numpy, dynamical systems — %DET/%LAM), **Matrix
Profile** (stumpy, continuous recurrence), **PST** (pure stdlib, variable-order
Markov syntax), **Transition Network** (networkx, graph topology — clustering /
modularity / path length). Promotion rule: a decisive regime-win → router branch
(e.g. `graded → RQA`), never a wholesale DUSTED replacement.

### Repertoire Taxonomy (`diagnostics/repertoire_taxonomy/`)
Pure-stdlib classifier: `RepertoireProfile` (DUSTED stats + emitter/context V +
social_lift) → `RepertoireType` (discrete_syllable / graded / sparse_repetitive /
individually_stereotyped / socially_grounded / human_lexical). Secondary flags
(identity_dominant, socially_grounded). `route_by_continuous()` — token-free
pre-router using MP recurrence + RQA LAM to skip heavy DTW on degenerate data.

### DSP Audio Splicer (`experiments/playback_design/audio_splicer.py`)
Track C: generates counterfactual WAV stimuli from HECB/raw-NBD-segmented calls.
Two stimulus families: (1) the standard 5-condition set (natural / order_shuffled /
onset_offset_swap / caller_swapped / amplitude_matched) and (2) the **compositional**
set (`generate_compositional_counterfactuals`: natural_AB / isolated_A / isolated_B /
reversed_BA / amplitude_matched_AB) for testing whether a sequence AB is perceived as
compositional (response to AB > max(A,B) and ≠ BA). Exemplar-based DSP (slice +
crossfade + concatenate), NOT neural synthesis. Deterministic, firewall-clean.
§37 mines compositional-candidate bigrams and emits these stimuli.

### Caller-Identity Disentanglement (`representation/disentangle.py`)
Three methods to scrub caller identity from the 44-D fingerprint: `disentangle_delta()`
(non-linear differencing), `disentangle_orthogonal()` (LDA/LogReg subspace projection),
`disentangle_znorm()` (per-speaker Z-normalization). `evaluate_disentanglement()`
measures caller-V vs form-V before/after. **All four empirically fail** (§36) — this is
the **Entanglement Ceiling**: the semantic ceiling is mathematically proven, mandating
Track-C biology for meaning.

### Channel Audit + Temporal Ecology (`diagnostics/`)
`channel_audit.py`: per-recording SNR / reverb proxy / spectral attenuation / quality
tier (clean/moderate/degraded). `temporal_ecology.py`: `TemporalContext` (inter-call
interval, bout position/length, turn-taking latency, overlap flag, daily phase) +
`extract_temporal_context()` from an ordered call timeline.

## Status

- **Track A** (engineering): validated on LibriSpeech. §1 combined-F1 = 0.836.
  Semantic ceiling demonstrated. CPC latent is speaker-dominated (0.56 > word 0.40).
- **Track B** (bioacoustic structure): **representation-robust context_binding**
  for both bat (3/3 representations, p=0.007) and marmoset (2/2 representations,
  p=0.007). Functional aliasing is representation-dependent (negative result).
  Species tokenizers operational (bat VQ-VAE 70% codebook; marmoset_cpc collapsed —
  fingerprint/AVES used instead).
- **Cross-corpus DUSTED motif arc** (§11–§19): recurrent acoustic-term discovery
  validated across 4 animal corpora + LibriSpeech. Bengalese = diverse/GT-validated
  (purity 0.895); bat = graded/emitter-signatured (V ~0.46); marmoset = sparse/one
  dominant; rook = degenerate/Event-associated. The difference is the corpus, not the
  method. Bat emitter claim survives held-out (§17: V 0.375, 2.52× context).
- **Individual identity** (§18): compact 44-D fingerprints consistently preserve
  individual identity across species — bat (V ~0.46), rook (V 0.384, 4× chance
  cross-session). Sub-call structure does NOT improve individual-ID accuracy —
  HECB's value is structural annotation, not ID.
- **Word-structure benchmark** (§14–§16): interpretable fingerprint + in-house CPC
  lead on all 4 tests (consistency, DTW term discovery, clustering, HuBERT ablation).
  HuBERT is metric-dependent (fails Levenshtein/clustering, recovers under DTW).
  Conclusion locked across 24 HuBERT configs.
- **Track C** (biological function): the falsification reports are the experimental
  to-do list for wet-lab / field playback validation. Phase-1 playback design
  generates the stimulus variants + manifest. The **DSP Audio Splicer** produces
  5-condition counterfactual WAVs (100 bat stimuli preflighted); the **H0-CMC
  evidence ladder** (T0→T5) + explicit-implicit test (A vs D vs G) scaffold the
  path from computational hypothesis to validated meaning. Pipeline sits at
  **T0–T1** (acoustic + context); T2–T4 are the playback phase.
- **Determinism** (Claim A): tri-run byte-identical — the pipeline implementation
  is reproducible. Re-verified after the v1.0 freeze (9/9 SHA-256 artifact hashes
  identical across 3 independent subprocess runs).
- **Negative control** (ESC-50): domain gate refuses communication inference —
  detects structure but does not overclaim. §35 extended controls confirm the
  pipeline does not manufacture structure from arbitrary input (synthetic chirps /
  random segments → 1 motif each vs natural's 33).
- **Architecture v1.0 frozen**: segmentation router (3 regimes + token-free
  pre-router + `router_audit` trace), whole-call-primary invariant (§30,
  `AblationResult`), first-class abstention (`AbstentionReason` trail), and the
  H0-CMC framework all complete. DUSTED corroborated by **4 independent
  mathematical paradigms** (§33–§34: RQA / Matrix Profile / PST / Transition
  Network). **247 tests, ruff-clean on changed files.**
- **Semantic ceiling = mathematical** (§36): caller identity and call form are
  non-separably entangled — 4 disentanglement methods all fail. The three-bucket
  thesis (encode / do-not-encode / needs-biology) is the operating frame.

## Session 2 — representation science + Track C freeze (§21–§29, 2026-06-23)

- **Unified LS word-structure** (§21): same 600 words × 89 types for §14 + Stage C + D.
  Fingerprint leads on all aggregate metrics; CPC best same-word NN; HuBERT metric-dependent.
- **Cross-corpus DUSTED** (§22): uniform thr=0.15, HAC-corrected. Bengalese diverse / bat
  graded-emitter / marmoset sparse. Difference is the repertoire, not the method.
- **RookID Event/Comment** (§23): social_other V≈0.04 (≈chance) — partner rook not in the
  call. Source V=0.31 (individual IS in the call).
- **Soundwel valence ceiling** (§24): within-call-type fp→valence 0.350 ≪ chance → emotion
  not linearly separable from call form. Bioacoustic semantic ceiling, proven.
- **Speaker-invariant test** (§25): cross-speaker word retrieval 5–7× chance (CPC best);
  ratio≈1 (speaker-dominated); centering fails; only CPC partially disentangles.
- **Cross-caller animal analogue** (§26): RookID call-types barely generalize across callers
  (fine Event below chance) — animal calls more individually stereotyped than human words.
- **DBA consensus-template sweep** (§27): only CPC (+0.012) makes averaging help; fp (−0.040),
  VTLN (−0.056), HuBERT (−0.048), AVES (−0.068) all fail. Non-additive entanglement.
- **RookID sing-song-bout DUSTED** (§28): 1398 diverse motif families (H=0.93) but weakly
  tied to Source/variant (V<0.1). Fingerprint, not motif histogram, is the identity tool.
- **Cross-analysis** (§29): 5 summary tables — cross-speaker vs cross-caller, identity/form/
  context disentanglement, word-likeness stress test, evidence-tier status (T0–T1 ✅, T2–T4
  needs playback), playback-candidate prioritization.
- **Bat Track C A–G** (§9 extended): D (acoustic+social) = 4.3× ≈ G (4.7×); behavior adds
  +0.4× only. Social-shuffle collapses B→chance, D drops 1.1× (69% genuine). Emitter-held-out
  preserves D at 4.1× (generalizes to unseen callers).
- **H0-CMC framework**: 4-stage evidence ladder (L0 acoustic → L3 causal), 6 evidence tiers,
  explicit-vs-implicit model test, social axis separate from context. Pipeline at T0–T1;
  T2–T4 = playback phase.
- **CPC policy**: animal pipeline = fingerprint + HECB + DUSTED + Track C; CPC =
  human-validation/speaker-invariant only.

New scripts: `ls_unified_rerun`, `ls_speaker_invariant`, `ls_consensus_template`,
`ls_aves_extract`, `ls_vtln_template`, `rookid_event_comment`, `rookid_cross_caller`,
`soundwel_trackc`, `bat_trackc_ablation` (A–G + controls), `bat_trackc_cca`,
`marmoset_clusterfn_compare`. 222 tests, ruff-clean.

## Session 3: Architecture v1.0 — multi-paradigm validation + Track C (2026-06-23)

### Architecture v1.0 freeze (12 steps, all complete)
Router audit + regime pre-router (`route_by_continuous`), AblationResult, AbstentionReason
(first-class abstention trail), HECB parent gate verified, whole-call-primary invariant
(§30), `trace["router_audit"]`.

### Multi-paradigm structural validation (§33–§34)
DUSTED corroborated by 4 independent mathematical paradigms:
- **PST + Graph** (Bengalese): 247 contexts, 5 modular communities — discrete syntax confirmed.
- **RQA** (bat): LAM context-dependent (**p=0.016**) — graded dynamics DUSTED couldn't see.
- **Matrix Profile** (marmoset): 87.5% stutter-dominant, unimodal — degenerate confirmed token-free.
- **Transition Network** (LibriSpeech): 639 nodes/19 communities — scales to human diversity.

### H0-CMC evidence ladder + Track C
- **Evidence tiers** (Step 5): T0 acoustic → T5 field; maps to `semantic_validation`.
- **Explicit-implicit test** (Step 7): A vs D vs G → CommunicationMode + social axis.
- **DSP Audio Splicer** (Step 6): 5-condition counterfactual WAVs (100 bat stimuli preflighted).
- **Negative controls** (§35): synthetic chirps 1 motif vs natural's 33 → **33× specificity**.

New modules: `diagnostics/{h0_cmc, repertoire_taxonomy, structural_alternatives,
channel_audit, temporal_ecology}`, `experiments/playback_design/audio_splicer.py`.
New scripts: `boundary_quality_sweep`, `compare_structure_methods`, `bat_rqa_by_context`,
`marmoset_matrix_profile_sweep`, `bengalese_network_topology`, `repertoire_taxonomy`,
`identity_form_context_disentanglement`, `playback_splice`, `extended_negative_controls`,
`stereotypy_vs_template`, `hecb_calibrator_factorial`, `hecb_cpc_comparison`. 247 tests.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
