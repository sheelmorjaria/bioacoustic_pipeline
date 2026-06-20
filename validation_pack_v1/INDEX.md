# Validation Pack v1.0

**Pipeline frozen at git `f4f5ce3`** · generated 2026-06-20 · device: CPU

A reproducible validation bundle proving the pipeline:
1. gives identical outputs across repeated runs,
2. rejects environmental non-communication audio,
3. recovers known call-type structure in a timestamped animal dataset,
4. abstains when bat segmentation is insufficient,
5. uses bat annotations for context-binding evidence,
6. *(playback candidates deferred — generated only from structures that survive nulls).*

## Reports

| # | Report | Status | Headline result |
|---|--------|--------|-----------------|
| 1 | `reports/determinism_report.json` | ✅ PASS | 9/9 artifacts **byte-identical** across 3 independent runs (seed 1337, CPU). Claim A only — not biological stability. |
| 2 | `reports/esc50_negative_control_report.json` | ✅ PASS | **NON_BIOACOUSTIC_COMMUNICATION** (0.99). Detects structure, refuses communication inference. 24 non-bio / 16 bio-source-noncommunication / 0 candidates. |
| 3 | `reports/infant_marmoset_calltype_report.json` | ✅ PASS | **Positive control.** context_binding **strong** (0.993); token↔call-type nmi=0.267, **V=0.599, p=0.0066**; probe acc **0.816 (4.08× lift)**. Call types recovered. |
| 4 | `reports/bat_segmentation_sweep_report.json` | ✅ PASS | **Bias confirmed + fixed.** `bat_holophrastic`: 0.8 segs/bout, 43% nonzero → `not_evaluable`. `bat_call_granular`: 9.9 segs/bout, 96% nonzero → `evaluable`. Sufficiency gate prevents false null. |
| 5 | `reports/bat_context_binding_report.json` | ✅ PASS | context_binding **weak but significant** (nmi=0.046, V=0.193, **p=0.007 within-Emitter-controlled**, q=0.0066). functional_aliasing absent; motif strong. Functional-CONTEXT evidence, NOT validated meaning. |
| 6 | `reports/meerkat_calltype_report.json` | ◑ SPLIT | **Call-type recovery PASS**: token↔call-type nmi=0.319, **V=0.521, p=0.0066**, probe acc **0.753** (11 call types). **Boundary F1 GAP**: 0.079 (R=0.76, P=0.04) — fixed-threshold NBD over-segments the noisy 8 kHz field audio; needs a learned meerkat calibrator (documented gap, like the human GBT+context path). |
| 7 | `playback_candidates/` | ⏳ DEFERRED | Generated only from motifs/aliases that survive nulls — after the reports are clean. |

## Verdict summary

```
1. Determinism (Claim A):          PASS — implementation reproducible (byte-identical ×3)
2. Negative control (ESC-50):      PASS — refuses communication inference (abstains correctly)
3. Positive control (marmoset):    PASS — recovers known call-type structure (V=0.599, 4× probe lift)
4. Bat segmentation bias:          PASS — holophrastic collapses (gate→not_evaluable); granular recovers
5. Bat context binding:            PASS — token↔Context significant within-Emitter (acoustically grounded)
6. MeerKAT positive control:       SPLIT — call-type recovery PASS (V=0.521); boundary F1 GAP (0.079)
7. Playback candidates:            DEFERRED — post-validation
```

## What this proves

- **Implementation is deterministic** (Claim A, not Claim B). Same seed + env + inputs → identical outputs.
- **The evidence model abstains correctly**: ESC-50 environmental sound → `NON_BIOACOUSTIC_COMMUNICATION`
  with ethological axes `not_applicable`; insufficient bat segmentation → `not_evaluable` (no false null).
- **The pipeline recovers real structure**: marmoset call types (Phee/Twitter/Trill/Tsik/Seep) are
  recovered by the deterministic acoustic fingerprint — a positive control on a timestamped animal corpus.
- **Bat segmentation bias is fixed**: the holophrastic prior that collapsed multi-call bouts to ~1
  segment is replaced by `bat_call_granular`; the sufficiency gate hard-blocks the degenerate case.
- **Bat context binding is real but modest**: token↔Context association survives within-Emitter nulls
  (kills the speaker confound) — reported as functional-context evidence, never validated meaning.

## Scope / honesty

- Claim A (determinism) ≠ Claim B (biological stability). This pack proves A; B is separate.
- Marmoset call-type recovery is an **acoustic** association (fingerprint recovers call shapes), NOT
  semantic. Semantic ceiling holds — meaning needs Track C biology.
- Bat context_binding is **weak** (V=0.193). It is significant + within-Emitter-controlled, but modest;
  reported honestly as "weak", not inflated.
- Playback candidates are deliberately deferred until validation is clean — they will be built only
  from motifs/aliases that survive their null models.
