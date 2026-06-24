# Methodology — usage reference

Personal reference docs: how *I* use each method in this pipeline — real params, commands,
when-to-use, and the empirical gotchas hit in practice (not theory). Grounded in the actual
code + `BENCHMARK.md`.

## Segmentation

| Doc | Method | One-line |
|---|---|---|
| [nbd.md](nbd.md) | **NBD** — Neural Boundary Detection | proposal + calibrator segmenter; speech-like audio + species fallback. combined-F1 **0.836** |
| [hecb.md](hecb.md) | **HECB-Seg** — Hierarchical Event-Conditioned Boundary | PCEN+CRNN call isolation → NBD internal segmentation; discrete calls in noise. F1 **0.703** |

**Regime routing** (see `configs/segmentation_router.yaml`): continuous speech-like → NBD+CPC;
discrete calls in noise → HECB; unknown → fixed-threshold NBD + ABSTAIN.

## Related topic docs (repo root)

- [`cpc-policy.md`](../cpc-policy.md) — CPC selective-use policy (human-only; firewall)
- [`h0-cmc-framework.md`](../h0-cmc-framework.md) — H0-CMC evidence ladder (T0–T5)

## Planned additions

Candidates for the same treatment when needed: acoustic fingerprint, DUSTED, the structural
alternatives (RQA / Matrix Profile / PST / transition network), the verdict engine + evidence
profile, the ethological firewall, CPC/VQ-VAE tokenization. Add one `.md` per method; keep the
template (what / API / params / commands / when-to-use / gotchas / empirical headline).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
