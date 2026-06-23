---
name: cpc-policy
description: CPC is human-validation/speaker-invariant only; NOT the default animal representation. Animal pipeline = fingerprint + HECB + DUSTED + Track C.
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d6b69a37-ed66-436a-aafa-b9633f25df0f
---

**CPC selective-use policy** (user directive, 2026-06-23):

**Animal production pipeline:** fingerprint + HECB + DUSTED + Track C. CPC is NOT the
default animal representation. Three downsides for animals: (1) harder to train/verify per
species; (2) hides interpretable acoustic structure; (3) current animal results are already
strong with fingerprint + HECB.

**Human validation / representation science:** CPC + fingerprint + HECB comparison. CPC is
the learned disentangler for speaker-invariant tests (§25/§27 — only CPC makes DBA template
help, Δ +0.012; all other reps fail).

**Animal CPC:** optional ablation only — pick ONE dataset (RookID or bat), compare
fingerprint vs CPC vs fingerprint+CPC. Stop if CPC doesn't clearly beat fingerprint.

**Why:** the fingerprint is interpretable, species-neutral, and already works across bats,
rooks, marmosets, birdsong, and LibriSpeech word-structure controls. CPC adds opacity
without proven animal-payoff (marmoset_cpc is degenerate; no species VQ-VAE exists).

**How to apply:** default to the 44-D fingerprint for all animal experiments. Use CPC
only in human-speech validation sections (speaker-invariance, DBA templates, cross-speaker
retrieval). See [[validation-phase-3-4]] for the §27 finding that only CPC's contrastive
objective achieves speaker-invariance.
