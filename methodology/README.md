# Methodology — usage reference

Personal reference docs: how *I* use each method in this pipeline — real params, commands,
when-to-use, and the empirical gotchas hit in practice (not theory). Grounded in the actual
code + `BENCHMARK.md`. Same template per file: **what / API / params / commands / when-to-use
/ gotchas / empirical headline**.

## By pipeline layer

### Segmentation (audio → segments)
| Doc | Method | One-line |
|---|---|---|
| [nbd.md](nbd.md) | **NBD** — Neural Boundary Detection | proposal + calibrator; speech-like + species fallback. combined-F1 **0.836** |
| [hecb.md](hecb.md) | **HECB-Seg** — Hierarchical Event-Conditioned Boundary | PCEN+CRNN → NBD internal; discrete calls in noise. F1 **0.703** |

### Representation & tokenization (segments → vectors / tokens)
| Doc | Method | One-line |
|---|---|---|
| [acoustic-fingerprint.md](acoustic-fingerprint.md) | **Acoustic Fingerprint** | ~43-D physics+texture; the one cross-species, all-species view. marmoset V **0.606** |
| [cpc-vqvae-tokenization.md](cpc-vqvae-tokenization.md) | **CPC / VQ-VAE tokenization** | which tokenizer is firewall-legal per species (3 tokenizers) |

### Structure discovery (sequences → motifs / syntax)
| Doc | Method | One-line |
|---|---|---|
| [dusted.md](dusted.md) | **DUSTED** | DTW recurrent acoustic-term discovery (NOT words). Bengalese purity **0.895** |
| [structural-alternatives.md](structural-alternatives.md) | **RQA / Matrix Profile / PST / Transition Network** | 4 paradigms corroborating DUSTED (§33–§34) |

### Governance (provenance + output contract)
| Doc | Method | One-line |
|---|---|---|
| [ethological-firewall.md](ethological-firewall.md) | **Ethological Firewall** | JSON provenance registry + hard-fail gates; cross-species block is mathematically necessary (§36) |
| [verdict-engine.md](verdict-engine.md) | **Verdict Engine + Evidence Profile** | schema-first 8-axis output; self-validating; abstention first-class; semantic_validation biology-only |

## Regime routing (see `configs/segmentation_router.yaml`)

continuous speech-like → **NBD+CPC** · discrete calls in noise → **HECB** · unknown →
fixed-threshold NBD + ABSTAIN. Representation: all-species → **acoustic fingerprint**;
learned → AVES (multi-taxa) / species CPC (firewall-gated).

## Related topic docs (repo root)

- [`cpc-policy.md`](../cpc-policy.md) — CPC selective-use policy (human-only; firewall)
- [`h0-cmc-framework.md`](../h0-cmc-framework.md) — H0-CMC evidence ladder (T0–T5)

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
