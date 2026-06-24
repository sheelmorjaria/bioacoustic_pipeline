# DUSTED — DTW recurrent acoustic-term discovery (for animals, NOT words)

> Personal methodology-usage reference. See `diagnostics/motif_miner/subsequence_motifs.py`,
> BENCHMARK §11–§19/§22/§37, CLAUDE.md.

## What it is

DUSTED adapts **spoken-term discovery** to *recurrent acoustic terms* in animal vocalisations.
The critical modification vs the speech literature: discovered terms are labelled **recurrent
acoustic terms, NOT word candidates** — the semantic ceiling forbids word claims. Each
vocalisation = one `CallUnits` (an ordered sequence of internal sub-call units); windows
(L=2–6) are matched by three primitives and clustered into motif families.

## The three primitives

| Primitive | Catches | Use |
|---|---|---|
| **DTW with indels** (main) | substitutions + indels + length variation | the headline method |
| fixed-length cosine | substitutions at fixed length | clean-discrete control |
| exact n-grams | only identical sequences | lower-bound control |

DTW uses role-aware length-normalised distance + a **centroid-cosine prefilter** so full
all-pairs DTW is never run. Families are refined by HAC (`_hac_refine`).

## API

```python
from bioacoustic_pipeline.diagnostics import motif_miner as mm
from bioacoustic_pipeline.diagnostics.motif_miner.subsequence_motifs import (
    CallUnits, discover_recurrent_subsequences, fixed_length_cosine_clusters,
    exact_ngram_clusters, discovery_null,
)

calls = [CallUnits(fingerprints=..., clusters=..., calltype=ctx, caller=emitter,
                   file_id=fn, call_idx=i) for ...]
motifs = discover_recurrent_subsequences(calls, L_min=2, L_max=6, ...)   # DTW main
null   = discovery_null(calls, ...)                                      # order-shuffle null
```

`CallUnits` fields: `fingerprints` (per-unit vectors), `clusters` (per-unit token ids),
`calltype` (Context), `caller` (Emitter), `file_id`, `call_idx`.

**Per-motif diagnostics** (`SubsequenceMotif`): `label_purity` (GT validation),
`family_entropy` / `top_family_support_fraction` (diverse-vs-degenerate discriminator),
`calltype_enrichment` / `caller_enrichment` (Cramér's V — association, NOT semantics),
order-shuffle null, bootstrap Jaccard.

## Commands

```bash
# Marmoset (HECB sub-call units, oracle) — DTW ct_V 0.508 = 4.7× exact
python scripts/hecb_dusted_marmoset.py --n-eval 1500
# Bengalese finch (GT syllable-validated positive control) — purity 0.895
python scripts/dusted_bengalese.py --birds 0 1 2
# Egyptian fruit bat (bout-level, emitter vs context) — emitter V ~0.46
python scripts/dusted_bat.py --n-bouts 2000
# Bat held-out emitter (70/30)
python scripts/bat_heldout.py --n-bouts 2000
# RookID individual-ID
python scripts/rookid_dusted.py --max-calls 2000
# 3-way method comparison (exact / fixed-cosine / DTW)
python scripts/compare_structure_methods.py --n-files 20
```

## ⚠️ Gotchas

1. **Terms = recurrent acoustic structure, NOT words.** Never write "the animal uses word X";
   write "recurrent acoustic term family F (support n, purity p)". The semantic ceiling is
   encoded in the framing.
2. **Degenerate repertoires dominate the motif count.** Infant marmoset inventory is dominated
   by one same-cluster repeat (`[1,1]`/`[6,6]` ~50% of calls); order-shuffle null is
   conservative for repeats (p≈0.09). Use `family_entropy` / `top_family_support_fraction` to
   detect this — a low-entropy, high-top-fraction result is degenerate, not rich.
3. **Graded bouts (bat) over-merge under DTW.** Use fixed-length cosine as the honest headline
   there (§13); DTW's strength is discrete/intermediate repertoires (Bengalese, marmoset).
4. **Tokenization matters.** DUSTED runs on `clusters` (k-means ids from the fingerprint) or
   `fingerprints` directly. Use `FingerprintTokenizerAdapter` / `SpeciesAcousticTokenizer` —
   not the human-hardwired `AcousticTokenizer` (firewall-blocked on animals).
5. **`discovery_null` order-shuffles** — pair motifs against their own null; report the
   p-value, don't just cite motif counts.

## Empirical headline (cross-corpus motif arc, §11–§19)

| corpus | regime | key result |
|---|---|---|
| Bengalese | discrete syllable | fixed-cosine wmean purity **0.895**; DTW `[4,5]` purity 1.00 |
| Bat | graded | emitter V ~0.46 ≫ context V ~0.16 (2.8×); survives held-out (§17: V 0.375) |
| Marmoset | sparse | DTW calltype V **0.508** = 4.7× exact; degenerate inventory |
| Rook | degenerate | event V 0.22 > source V 0.08 (motifs are call-type, not individual) |

**The difference is the repertoire, not the method** — DUSTED lands where the vocalisation
structure predicts.

## Related
- `structural-alternatives.md` — 4 paradigms that corroborate DUSTED (§33–§34).
- `acoustic-fingerprint.md` — the per-unit descriptor DUSTED matches on.
- §37 compositional mining reuses DUSTED's tokenization + null machinery.
