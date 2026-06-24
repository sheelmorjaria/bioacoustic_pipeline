# CPC / VQ-VAE Tokenization — which tokenizer is firewall-legal per species

> Personal methodology-usage reference. See `tokenization/`,
> `embeddings/{backbones,species_cpc_backbone,segment_representation}.py`,
> `configs/model_metadata.json`, CLAUDE.md.

## What it is

The discrete-token layer: CPC encodes audio → latent; a VQ-VAE quantises the latent → discrete
**acoustic-similarity tokens** (cluster ids). These tokens feed the sequence diagnostics (LZ,
alias, call_unitization, DUSTED). The hard rule: **tokens are acoustic-similarity classes,
never word/meaning identity** (semantic ceiling).

The non-obvious part of this codebase is that there are **three tokenizers**, and which one is
legal depends on species + firewall. Picking the wrong one silently firewall-blocks (or worse,
silently uses a human model on an animal).

## The three tokenizers

| Tokenizer | Module | `provenance_name` / `input_source` | Species | torch? |
|---|---|---|---|---|
| **`AcousticTokenizer`** | `tokenization/multiview.py` | `human_vqvae_large` / `cpc` | **HUMAN-ONLY** (hardwired) | yes |
| **`SpeciesAcousticTokenizer`** | `tokenization/species_acoustic_tokenizer.py` | `<species>_vqvae` / `cpc` | any species w/ CPC+VQ-VAE | yes |
| **`FingerprintTokenizerAdapter`** | `tokenization/fingerprint_tokenizer_adapter.py` | `acoustic_fingerprint` / `acoustic_fingerprint` | **all-species** | **no** |

```python
from bioacoustic_pipeline.tokenization.species_acoustic_tokenizer import SpeciesAcousticTokenizer
from bioacoustic_pipeline.tokenization.fingerprint_tokenizer_adapter import FingerprintTokenizerAdapter

# Species path (needs a registered species CPC + species VQ-VAE):
tok = SpeciesAcousticTokenizer(cpc_provenance="bat_cpc", cpc_checkpoint=...,
                               vqvae_provenance="bat_vqvae", vqvae_checkpoint=...)
# All-species torch-free path (the animal fallback):
ftok = FingerprintTokenizerAdapter()           # .tokenize(audio, sr) -> int
```

## The backbone adapters (`embeddings/`)

- `CPCBackbone` (`backbones.py`, `provenance_name="human_cpc_phonetic"`) — **human-only**.
- `SpeciesCPCBackbone` (`species_cpc_backbone.py`) — generic adapter for any registered
  species CPC; takes `provenance_name` + checkpoint.
- `BioMAEBackbone` (bat, ONNX), `AVESBackbone` (multi-taxa, `avex`) — non-CPC backbones.

## The segment-representation source contract (`embeddings/segment_representation.py`)

CPC is **one option, not a prerequisite**. The real contract is a *frozen, compatible*
segment-representation space:

```python
from bioacoustic_pipeline.embeddings.segment_representation import resolve_source, assert_compatible, SegmentSource

src  = resolve_source("marmoset", "acoustic_fingerprint")   # -> SegmentSource(category, provenance, dim, view_class)
assert_compatible(head, src)                                 # head.input_source == src.category
```

Categories: `cpc | biomae | acoustic_fingerprint | spectrogram | embedding_bank_fusion`.
`resolve_source` picks the firewall-legal provenance for a (category, species) pair;
`assert_compatible` gates assignment (the VQ-VAE/β-VAE head must receive the representation it
was trained on). `run_pipeline(segment_source=...)` records `trace["segment_representation"]`.

**Dual stream** (do not fuse): VQ-VAE tokens = discrete syntax; β-VAE affect = continuous
graded state. `DualStreamSegment` carries both as separate typed fields; late fusion only.

## Commands

```bash
# Train species tokenizers
python scripts/train_bat_vqvae.py --audio-dir /data/bat --n-files 500     # bat_vqvae (256 codes)
python scripts/train_beta_vae.py                                          # human affect
python scripts/refit_nbd_calibrator.py --n-files 400 --kind gbt --context cpc   # human CPC context

# Track B animal pipelines (pick the legal tokenizer):
python scripts/track_b_marmoset.py --rep fingerprint        # FingerprintTokenizerAdapter (default, all-species)
python scripts/track_b_bat_corrected.py --n-files 1500      # SpeciesAcousticTokenizer (bat_cpc + bat_vqvae)
```

## ⚠️ Gotchas

1. **`AcousticTokenizer` is hardwired human and firewall-blocked on animals.** Use
   `SpeciesAcousticTokenizer` (species CPC+VQ-VAE) or `FingerprintTokenizerAdapter`
   (torch-free, all-species) for any non-human species. This is the #1 tokenization trap.
2. **Animal tokenization is degenerate until a species VQ-VAE exists.** Only `human_vqvae*` +
   `bat_vqvae` are trained. So bat/marmoset discrete-token diagnostics (LZ, alias,
   call_unitization) are empty for species without a VQ-VAE — use the fingerprint tokenizer or
   accept the gap.
3. **`marmoset_cpc` is collapsed** (cosine ~0.99 across call types). It loads but is unusable
   for structure; a marmoset VQ-VAE cross-check is blocked until it is retrained. Fallback =
   fingerprint. See `ethological-firewall.md`.
4. **`input_source` must match.** A VQ-VAE trained on CPC latents declares `input_source="cpc"`;
   the fingerprint tokenizer declares `"acoustic_fingerprint"`. `assert_compatible` enforces
   this — feeding a CPC-trained VQ-VAE fingerprint vectors (or vice versa) is a silent bug the
   gate prevents.
5. **Tokens ≠ words.** The semantic ceiling: VQ tokens recover call *form* (acoustic
   similarity), never word identity. The §36 ceiling is proven on this tokenization pipeline.

## Empirical headline

- **Human**: CPC latent is speaker-dominated (speaker probe 0.56 > word probe 0.40);
  token↔word macro consistency 0.29 → semantic ceiling confirmed.
- **Bat**: `bat_vqvae` 256 codes, ~70% codebook utilization (healthy); `bat_cpc` embeds fine.
- **Marmoset**: `marmoset_cpc` collapsed → fingerprint used instead (V 0.606, probe 0.829).
- Track A calibrator's CPC-context layer (the 768-D `input_source="cpc"` features via
  `CPCBackbone.embed_frames`) is what lifts combined-F1 0.79 → **0.836** (human-only).

## Related
- `ethological-firewall.md` — provenance gates that decide which tokenizer is legal.
- `acoustic-fingerprint.md` — the all-species representation behind `FingerprintTokenizerAdapter`.
- `nbd.md` — the CPC-context calibrator path (human-only).
