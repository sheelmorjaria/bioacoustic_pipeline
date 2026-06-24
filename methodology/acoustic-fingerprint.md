# Acoustic Fingerprint — physics + texture per-segment descriptor

> Personal methodology-usage reference. See `diagnostics/acoustic_fingerprint/`,
> BENCHMARK §11/§13/§36, CLAUDE.md.

## What it is

A **physics-grounded, species-portable per-segment descriptor** (~43-D universal core). Not
learned, not semantic — it audits learned tokens and separates *acoustic similarity* from
*functional equivalence*. It is the **one cross-species representation** (all-species, no
training data, no leakage) and the firewall-legal tokenization fallback for every animal.

## Structure (`diagnostics/acoustic_fingerprint/`)

| Sub-module | Dim | What |
|---|---|---|
| `physics.py` | ~16 | universal linear spectral shape (centroid, spread, slope, roll-off, flatness, crest, …) |
| `physics.voiced` | ~7 | F0 / harmonicity / HNR / jitter / shimmer — **tonal segments only** (else 0 + quality flag) |
| `macrotexture.py` | ~14 | GLCM + ADSR envelope + modulation rate/depth |
| `microtexture.py` | ~15 | derivative / dynamics / inter-contour-intervals |
| `perceptual.py` | opt-in | Mel-MFCC (human-perceptual) — **NOT** in the cross-species core |
| `coherence.py`, `nulls.py`, `similarity.py`, `report.py`, `tokenizer.py` | — | coherence stats, permutation nulls, cosine, audit report, k-means tokenizer |

`combined` (~43-D) = physics + macrotexture + microtexture, **species-neutral**. `perceptual`
and `voiced` are flagged/opt-in (anti-anthropomorphism).

## API

```python
from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp

res = afp.extract(audio, sr, species="marmoset")   # -> ExtractionResult
vec = res.combined                                  # (~43,) np.ndarray — the descriptor
# res.physics, res.macrotexture, res.microtexture, res.voiced (None if not tonal), res.quality_flags
```

Two entry points:
1. **Bank view** — `AcousticFingerprintBackbone` (`view_class="interpretable"`,
   `applies_to_species=[]` all-species). Loaded by `EmbeddingBank`; cached.
2. **Evidence channel** — `run_pipeline(fingerprint=True)` → `Segment.fingerprint` +
   `trace["acoustic_fingerprint"]` (token_coherence + alias_divergence + null p-values).

Torch-free tokenization: `FingerprintTokenizerAdapter` (see `cpc-vqvae-tokenization.md`).

## Commands

```bash
# Track B — marmoset whole-call fingerprint → call-type binding (the strongest species result)
python scripts/track_b_marmoset.py --rep fingerprint --n-per-type 2500

# Bat 3-representation robustness cross-check (fingerprint / spectrogram / AVES)
python scripts/bat_representation_crosscheck.py --n-bouts 2000

# Entanglement Ceiling proof — LDA/LogReg/Z-norm/delta on the 44-D fingerprint (§36)
python scripts/entanglement_ceiling_aves.py --n-per-type 500
```

## ⚠️ Gotchas

1. **Anti-anthropomorphism is enforced in the vector.** `voiced` (F0/HNR/jitter/shimmer) is
   populated **only for tonal segments**; `perceptual` Mel-MFCC is opt-in and excluded from
   `combined`. Do not fold them into the cross-species core.
2. **Silence / too-short → zeroed core + quality flag.** Check `quality_flags`; a zero vector
   is a skip signal, not a real descriptor.
3. **`combined` uses Hz-scale spectral features within-species** (call types differ in
   spectrum, so Hz is informative there) — fine for within-species binding, but that's why the
   cross-species *core* is kept linear/universal. Don't mix the two scopes.
4. **It is NOT semantic.** Recovering a call type (V ≈ 0.6) is an *acoustic* association, not
   meaning (semantic ceiling). The §36 Entanglement Ceiling is proven *on this 44-D vector* —
   caller identity and call form are non-separably entangled here (and on AVES too).

## Empirical headline

- **Marmoset** (strongest species result): token↔call-type nmi 0.263, **Cramér's V 0.606**,
  p=0.0066; call-type probe acc **0.829** (4.15× chance).
- **Bat**: representation-robust context_binding (fingerprint/spectrogram/AVES all
  significant, within-Emitter-controlled).
- **LibriSpeech word-structure**: fingerprint leads — word-cluster **V 0.517**, motif→word
  **V 0.481** (beats CPC/HuBERT/HECB on every test).
- **Entanglement Ceiling (§36)**: the 4 disentanglement methods all fail on this vector.

## Related
- `cpc-vqvae-tokenization.md` — `FingerprintTokenizerAdapter` (torch-free all-species tokens).
- `verdict-engine.md` — `acoustic_coherence` axis consumes fingerprint coherence.
