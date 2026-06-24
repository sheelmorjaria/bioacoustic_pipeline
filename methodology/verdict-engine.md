# Verdict Engine + Evidence Profile — the schema-first output contract

> Personal methodology-usage reference. See `verdict_engine/{schema,engine,evidence_profile,
> falsification}.py`, BENCHMARK §1/§30/§37, CLAUDE.md.

## What it is

The **canonical output** of the pipeline. Schema-first: `schema.validate_report()` defines the
legal report shape, and `engine.build_report()` **self-validates** (assembles then calls
`validate_report()`, raises on any violation) — so no upstream stage can ever emit a malformed
verdict. The verdict is **multiple independent axes**, each categorical with calibrated
confidence — never a single label.

## The eight axes (`evidence_profile.py`)

| Axis | Scorer (primitive signal) | What it asks |
|---|---|---|
| `sequence_recombination` | `score_sequence_recombination(lz_normalized)` | LZ76 compressibility — is there sequence structure? |
| `functional_aliasing` | `score_functional_aliasing(alias_density, alias_pvalue, …)` | do distinct tokens fill the same role? |
| `call_unitization` | `score_call_unitization(n_hard, n_soft, n_total_boundaries)` | hard vs soft boundary mix |
| `context_binding` | `score_context_binding(n_context_bound)` | token↔label association (NOT meaning) |
| `turn_structure` | `score_turn_structure(n_turn_bigrams, entrained)` | turn-taking/entrainment |
| `acoustic_coherence` | `score_acoustic_coherence(...)` | fingerprint coherence (§ acoustic_fingerprint) |
| `motif_structure` | `score_motif_structure(...)` | recurrent subsequence motifs (DUSTED) |
| `semantic_validation` | `score_semantic_validation(status="none")` | **hard-coded** — only Track-C biology can raise this |

Axes are deliberately decoupled from *how* their primitive signals are obtained. The
`orchestration.synthesize_signals()` seam reduces real stage outputs into those primitives;
`scripts/run.py` is a near-duplicate CLI wrapper around it.

## Abstention as a first-class output

`AbstentionReason` vocabulary — `record_abstention()` is wired into every abstention point:
`TOO_FEW_CALLS` (sufficiency gate), `CONTEXT_LABELS_INSUFFICIENT` (context), `DOMAIN_REJECTED`
(domain gate), …. The sequence-dependent axes (`sequence_recombination`,
`functional_aliasing`, `motif_structure`) abstain to `not_evaluable` when the sufficiency gate
fails; the ethological axes abstain to `not_applicable` when the domain gate rejects the corpus.
**An axis that cannot compute abstains with a structured reason — it never emits a false null.**

## API

```python
from bioacoustic_pipeline.verdict_engine import is_valid
from bioacoustic_pipeline.verdict_engine.engine import build_report, report_from_profile
from bioacoustic_pipeline.verdict_engine.evidence_profile import (
    EvidenceProfile, AxisResult, AbstentionReason, compute_evidence_profile,
)
from bioacoustic_pipeline.verdict_engine.schema import validate_report

profile = compute_evidence_profile(primitive_signals, ...)   # -> EvidenceProfile (axes + abstentions)
report   = build_report(...)                                  # self-validates; raises on schema violation
errors   = validate_report(report_dict)                       # []  <=> is_valid(report_dict)
```

`EvidenceProfile` carries `axes: dict[str, AxisResult]`, `abstentions: list`, `domain` block.
`AxisResult` = categorical level + calibrated confidence + how-it-could-be-wrong note.

## Commands

```bash
# Token-stream -> verdict report (integration entry point)
python scripts/run.py --species bat --tokens-file tokens.json [--with-null]

# A report MUST be schema-valid — verify with the TDD suite
pytest tests/test_verdict_schema.py -q
```

## ⚠️ Gotchas / invariants (do not weaken)

1. **Self-validation is non-negotiable.** `build_report()` calls `validate_report()` and raises.
   Never bypass; any new field must be added to the schema first.
2. **`semantic_validation` is hard-coded to Track-C-only.** `score_semantic_validation` cannot
   return above `none` from computation alone; the schema rejects `validated_semantics: true`
   unless the axis is at `playback-supported`. **Semantics can only be validated by biology** —
   encoded in both schema and scorer.
3. **Abstain, don't false-null.** Sparse data → `not_evaluable`; out-of-domain →
   `not_applicable`. A "0 / absent" on insufficient data is a bug, not a result (this was the
   bat §track-b-revalidation fix: the holophrastic profile produced false nulls).
4. **Axes are independent.** Don't let one axis's signal leak into another's scorer; keep the
   primitive-signal decoupling so the engine stays pure-stdlib and unit-testable.
5. **Falsification is emitted alongside positive claims** (`falsification.py`): null /
   replication / confound / biology tests. A positive structure claim without its falsification
   tests is incomplete.

## Empirical headline

- The schema is the test every upstream stage feeds — validated by `test_verdict_schema.py`
  (the ethological invariant: no computational `validated_semantics`).
- ESC-50 → `NON_BIOACOUSTIC_COMMUNICATION` with ethological axes `not_applicable` (domain gate).
- Bat corrected verdict: sequence_recombination high, context_binding significant
  (within-Emitter-controlled), functional_aliasing genuinely tested-and-absent.

## Related
- `ethological-firewall.md` — the provenance gate that feeds `domain` + cross-species blocks.
- `acoustic-fingerprint.md` / `dusted.md` — primitive-signal sources.
- `../h0-cmc-framework.md` — the T0–T5 ladder `semantic_validation` maps onto.
