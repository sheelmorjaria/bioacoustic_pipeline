# Ethological Firewall — machine-readable model provenance + hard-fail gates

> Personal methodology-usage reference. See `embeddings/firewall.py`,
> `configs/model_metadata.json`, BENCHMARK §1, CLAUDE.md.

## What it is

Every model in the pipeline carries **machine-readable provenance**: pretraining domain,
human-speech / text-supervision flags, and allowed/forbidden uses. The firewall
**hard-fails** (`ProvenanceViolation`) on forbidden uses. It is the single mechanism that
prevents anthropomorphic contamination — and (per §36) it is a *mathematical* necessity, not
just policy: because linear algebra cannot disentangle animal identity from form, the only
thing that could (a learned caller-invariant model) is exactly what the firewall withholds.

The registry is **data-driven JSON** — adding, restricting, or species-scoping a model is a
config edit, not a code change.

## The `Use` enum (`embeddings/firewall.py`)

```
EMBEDDING · PROBE · TOKENIZE · AFFECT · RETRIEVAL · ANNOTATION_SUGGESTION
VERDICT_WEIGHT · SEMANTIC_VERDICT · ANIMAL_MEANING_CLAIM
```

## API

```python
from bioacoustic_pipeline.embeddings.firewall import Use, default_firewall, ProvenanceViolation

fw = default_firewall()                              # loads configs/model_metadata.json
fw.assert_allowed("aves", Use.EMBEDDING)             # raises ProvenanceViolation if forbidden
fw.assert_cross_species("aves", "marmoset")          # raises if aves not allowed on marmoset
fw.assert_allowed("acoustic_fingerprint", Use.TOKENIZE)
fw.assert_cross_species("acoustic_fingerprint", "bat")   # all-species → ok
```

Every backbone / tokenizer adapter calls these before loading weights. `EmbeddingBank` skips
anything unavailable or that fails the firewall — a model is **never silently used**.

## Rules the registry enforces (`configs/model_metadata.json`)

1. **Text-supervised models are retrieval-only.** NatureLM etc. → `Use.RETRIEVAL` only, never
   tokenize/embed for structure claims.
2. **Cross-species weight transfer is blocked.** A model trained on species A cannot be applied
   to species B unless `applies_to_species` explicitly lists B (or is all-species `[]`).
3. **No model may emit a semantic verdict.** `Use.SEMANTIC_VERDICT` / `ANIMAL_MEANING_CLAIM`
   are forbidden for every model — meaning is biology-only (Track C).

## Registered models (current)

| Model | Domain | Species scope | Allowed uses |
|---|---|---|---|
| human CPC / VQ-VAE / β-VAE | human speech | human-only | tokenize/embed/affect (human) |
| `bat_cpc`, `bat_vqvae` (256 codes, ~70% utilized) | bat | bat | tokenize/embed (bat) |
| `marmoset_cpc` | marmoset | marmoset | embed — ⚠️ **COLLAPSED** (cosine ~0.99 across call types) |
| BioMAE (ONNX) | bat | bat | embed (bat) |
| AVES (`esp_aves2_eat_bio`) | multi-taxa animal | multi-taxa | embed (all-species, learned) |
| acoustic_fingerprint + spectrogram | n/a (deterministic) | all-species `[]` | tokenize/embed (the cross-species views) |

## Commands

```bash
# Any species tokenizer/backbone script prints its firewall check before loading:
python scripts/track_b_marmoset.py --rep aves       # "firewall: aves embedding legal for marmoset"
python scripts/entanglement_ceiling_aves.py         # asserts aves + acoustic_fingerprint for marmoset
```

## ⚠️ Gotchas

1. **`AcousticTokenizer` (multiview) is HARDWIRED human** (`human_vqvae_large`,
   `input_source="cpc"`) → firewall-blocked on animals. Use **`SpeciesAcousticTokenizer`**
   (species CPC + species VQ-VAE) or **`FingerprintTokenizerAdapter`** (torch-free, all-species)
   for non-human species. See `cpc-vqvae-tokenization.md`.
2. **`marmoset_cpc` is registered but COLLAPSED** — it loads (128-D, steps_ahead=12) but
   cosine ~0.99 across distinct call types even with correct architecture. Don't use it for
   structure; the fingerprint is the marmoset representation. A learned VQ-VAE cross-check is
   blocked until it is retrained.
3. **Adding a model = a JSON edit.** New species CPC/VQ-VAE → add a `model_metadata.json` entry
   with `applies_to_species` + allowed uses, then the existing adapters pick it up. No firewall
   code change.
4. **The firewall is vindicated by §36, not just good practice.** The Entanglement Ceiling
   proves identity/form are non-separably entangled; speaker-invariance *requires* a learned
   non-linear model, which the firewall blocks on animals. So the firewall is the correct
   response to a mathematical fact.

## Related
- `cpc-vqvae-tokenization.md` — which tokenizer is firewall-legal per species.
- `verdict-engine.md` — the `domain` block + `semantic_validation` axis consume firewall state.
- §36 Entanglement Ceiling — why the cross-species block is a mathematical necessity.
