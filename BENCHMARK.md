# Segmentation & Representation Benchmark Matrix

Consolidated validation across three corpora (MeerKAT, LibriSpeech, InfantMarmosetsVox),
empirically establishing the segmentation router + the HECB hierarchical-annotator claim.
All numbers from the committed runs (`outputs/*.log`, 2026-06-21).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.

---

## Terminology — what the pipeline discovers in animal audio

Speech word-discovery methods (DUSTED, ES-KMeans, Bayesian word segmentation, SSL-unit
clustering) are **adapted** to animal vocalisations here, with the router/HECB structure
wrapped around them — not imported wholesale. The transferable target in *animal* audio is
**recurrent acoustic terms / motifs / sub-call phrases / sequence templates /
vocalisation-internal units**, **never "words"** unless later Track-C playback/context
evidence supports that. "Word" is reserved for the **LibriSpeech human-speech controlled
analogue** (which genuinely has word ground truth); the LibriSpeech rows below stay honest.

| speech (human, has GT) | animal (no GT meaning) |
|---|---|
| word | recurrent acoustic term / motif |
| word-like unit | motif-like acoustic term |
| phoneme / subword | sub-call acoustic unit |
| lexicon | motif inventory |
| sentence | bout / vocalisation sequence |

All calltype/caller "enrichment" reported below is **acoustic/structural association, not
semantics** — consistent with the semantic-ceiling invariant (`semantic_validation` stays
`none` until Track C).

---

## 1. Event-detection F1 (which segmenter, which regime)

| corpus (regime) | method | metric | F1 | P | R | ref |
|---|---|---|---|---|---|---|
| **MeerKAT** (calls in noise, 8 kHz) | raw NBD | onset 1-to-1 | 0.069 | — | — | massive over-segmentation |
| | sweep-best thresholds | onset 1-to-1 | 0.165 | — | — | thresholds insufficient |
| | v2 scalar GBT (19-feat) | onset 1-to-1 | ~0.26 | — | — | tabular baseline (frozen) |
| | v3/v4 post-onset GBT | onset 1-to-1 | ~0.21 | — | — | tabular **ceiling** |
| | **PCEN+CRNN** | onset 1-to-1 | **0.703** | — | — | **8.9× raw NBD → gap closed** |
| **LibriSpeech** (continuous speech, 16 kHz) | scalar GBT calibrator | word 1-to-1 / combined | 0.450 / 0.638 | — | — | CPC-free |
| | CPC-context GBT calibrator | word 1-to-1 / combined | 0.458 / 0.794 | — | — | CPC adds +0.156 combined |
| | **NBD+CPC calibrator** | **combined (§1 metric)** | **0.836** | 0.77 | 0.92 | **speech winner** |
| | PCEN+CRNN (retrained) | word 1-to-1 / combined | 0.590 / 0.652 | 0.80 | 0.47 | CRNN is the **speech bottleneck** |
| **InfantMarmosetsVox** (calls in noise, 44.1 kHz) | **PCEN+CRNN (retrained)** | onset 1-to-1 | **0.875** | **0.982** | 0.789 | **strongest result** |

**Router read:** PCEN+CRNN wins for discrete calls in noise (0.70 meerkat → 0.88 marmoset);
NBD+CPC calibrator wins for continuous speech (0.836 vs CRNN's 0.59). The CRNN does *not*
transfer to speech — confirming the segmentation router is necessary.

### Tolerance sweep (apples-to-apples — `scripts/tolerance_sweep.py`)

The headline F1s used different tolerances (MeerKAT 100ms, marmoset 200ms). At **matched**
tolerances the marmoset CRNN genuinely outperforms meerkat at every point — the router
claim holds rigorously, not just from a tolerance artifact:

| onset F1 | 50ms | 100ms | 200ms |
|---|---|---|---|
| MeerKAT (500 clips) | 0.662 | 0.707 | 0.724 |
| InfantMarmosetsVox (30 test) | **0.864** | **0.865** | **0.867** |

| offset F1 | 50ms | 100ms | 200ms |
|---|---|---|---|
| MeerKAT | 0.549 | 0.664 | 0.702 |
| InfantMarmosetsVox | **0.785** | **0.860** | **0.866** |

Marmoset onset F1 is tolerance-*insensitive* (0.864→0.867) because precision ≈ 0.98 (near-
zero false alarms, tight errors); meerkat onset is more tolerance-sensitive (0.662→0.724,
wider errors from noisier 8kHz field audio). Offset F1 is more tolerance-sensitive for
both (calls taper → wider offset errors).

---

## 2. HECB internal-boundary mechanism (oracle-parent, no CRNN)

The internal mechanism decoupled from the detector — given valid parent events, does HECB
recover meaningful sub-event structure?

| corpus | parent events | sub-event recovered | F1 | P | R |
|---|---|---|---|---|---|
| **LibriSpeech** | oracle word spans | in-word phoneme starts | **0.636** | 0.781 | 0.537 |
| LibriSpeech | CRNN word events | in-word phoneme starts | 0.409 | 0.516 | 0.339 |

**Read:** the mechanism works at 0.636 with good parents but degrades to 0.41 with poor
(CRNN) parents — poor parent events cannot be rescued. This justifies the combined-F1
reclaim (word-FPs landing on phonemes) and the router design.

---

## 3. Whole / sub-call / pooled representation recovery

The core claim: sub-call features **augment** whole-event representations when pooled, but
are weaker alone. Metrics: Cramér's V (unsupervised k-means token↔label), probe accuracy
(supervised linear readout), macro-F1 (supervised, balanced).

### Call-type recovery

| corpus | split | whole V | pooled V | whole probe | pooled probe | whole macroF1 | pooled macroF1 |
|---|---|---|---|---|---|---|---|
| **MeerKAT** (end-to-end, 1500 clips) | — | 0.246 | **0.274** | 0.611 | **0.633** | — | — |
| **InfantMarmosetsVox** (end-to-end) | random | 0.344 | **0.379** | 0.934 | **0.940** | — | — |
| InfantMarmosetsVox (oracle) | random | 0.431 | 0.416 | 0.935 | 0.932 | 0.487 | **0.566** |
| InfantMarmosetsVox (oracle) | twin-held-out | 0.477 | 0.475 | 0.944 | **0.947** | 0.557 | **0.582** |

*Sub-call alone is consistently weaker on V (loses whole-call context) but highest on probe
(sub-fragments are individually type-discriminative under supervision).*

### Caller-ID recovery (InfantMarmosetsVox oracle, random split, 10 callers) — **the novel finding**

| representation | V | NMI | probe | macro-F1 |
|---|---|---|---|---|
| whole call | 0.178 | 0.070 | 0.459 | 0.364 |
| **whole + sub-call pool** | **0.197** | **0.083** | **0.477** | **0.392** |
| sub-call | 0.115 | 0.029 | 0.317 | — |

**HECB internal boundaries recover caller identity — pooled wins on all four metrics.**
Individual voice signatures live in the sub-call structure. *(Caller test is degenerate
under twin-pair-held-out — a twin pair = 2 individuals < the probe's 5-class gate; the
random split with all 10 callers is the meaningful result.)*

---

## 4. The pooled-adds-value thesis (cross-corpus summary)

| corpus | label | pooled > whole? | on which metric |
|---|---|---|---|
| MeerKAT | call-type | ✅ | V, NMI, probe (clean) |
| InfantMarmosetsVox (end-to-end) | call-type | ✅ | V, NMI, probe |
| InfantMarmosetsVox (oracle) | call-type | mixed | macro-F1 (+0.079); whole edges V |
| InfantMarmosetsVox (oracle) | **caller ID** | ✅✅ | **all 4 metrics** |

**Refined conclusion (post-controls):** Sub-call structure augments whole-call
representations **when boundaries are meaningful (typed HECB > raw NBD) and the target
label is acoustically grounded**. The advantage shows up consistently on **supervised**
metrics (probe, macro-F1) but **not uniformly on unsupervised V/NMI** — pooling adds
dimensionality that helps a supervised classifier but can confuse kmeans. Whole-call
remains the clean primary representation. Sub-call features **augment, not replace**.

---

## 5. Segmentation router (canonical)

| regime | method | HECB role | example |
|---|---|---|---|
| discrete calls in noise | PCEN+CRNN | internal annotator | meerkat (0.70), marmoset (0.88) |
| continuous speech-like | NBD+CPC calibrator | optional | LibriSpeech (0.836 combined) |
| exploratory / unknown | NBD + ABSTAIN | none | new species (conservative) |

**Updated core claim (project thesis):** The pipeline uses a regime-specific segmentation
router: PCEN+CRNN for discrete animal calls in noise, NBD+CPC for speech-like
multi-resolution boundary validation, and HECB as a hierarchical internal-boundary annotator
conditioned on reliable parent events. Whole-call dual-stream vocalisation tokens remain the
primary representation; typed sub-call structure and relative internal sequence provide
supervised and caller-associated augmentation, while semantic/context labels require
behavioral and social covariates beyond acoustics.

---

## Caveats
- All F1s are one-to-one greedy matching (no duplicate-TP inflation). MeerKAT onset
  tolerance 100 ms; marmoset 200 ms (field audio); LibriSpeech word 50 ms.
- `probe=0.0` anywhere = `linear_probe`'s ≥5-class gate skipped (too few classes), not a bug.
- Oracle experiments use **GT spans as parent events** — they validate the internal
  mechanism, NOT end-to-end segmentation. End-to-end uses the trained CRNN.
- Macro-F1 for sub-call alone = n/a (different sample count than whole/pooled).
- CPC device-consistency + checkpoint-loading fixes (`a980e9e`, `f28467c`, merged) make all
  CPC paths robust on GPU; `bat_cpc`/`bat_cpc_full` now load real weights.

---

## 6. Cross-species 4-way ablation (raw-NBD oracle, 2026-06-22)

`scripts/cross_species_ablation.py` — per-file oracle-parent HECB (raw NBD internal
boundaries; no typed classifier, since every proposal is inside a real call in the
per-file case) → whole/sub/pooled/relative-sequence ablation. 2500 + 2493 vocs.

| corpus (kHz) | label | whole V | pooled V | **sequence V** | whole probe | pooled probe | seq probe |
|---|---|---|---|---|---|---|---|
| **adult marmoset** (Vocalizations, 96) | call-type (6) | 0.458 | 0.434 | **0.468** | 0.824 | 0.832 | **0.837** |
| **egyptian fruit bat** (250) | Context (12) | 0.152 | 0.152 | 0.153 | 0.547 | 0.556 | 0.543 |
| egyptian fruit bat (250) | Emitter (ID) | 0.291 | 0.291 | 0.291 | 0.211 | 0.191 | 0.195 |

**Honest read — this REFINES the pooled-adds-value thesis:**
- **The relative sub-call SEQUENCE representation is the standout on marmoset call-type**
  (best on V *and* NMI *and* probe) — the 4th ablation genuinely adds value; the ordered
  internal composition carries signal beyond the pooled mean.
- **Pooled ≈ whole on these corpora** (raw NBD boundaries) — the clean pooled>whole gains
  seen with the **typed HECB classifier** (MeerKAT/InfantMarmosetsVox, §3) do **not**
  reproduce with raw NBD internal boundaries. **Internal-boundary QUALITY matters**: typed
  HECB > raw NBD for the pooled advantage.
- **Bat Context is near-chance** (V~0.15, ~2-class baseline) — a clean semantic-ceiling
  result: ethological context is not recoverable from acoustics alone (needs Track C biology).
  This is a valid negative finding, not a pipeline failure.
- **Bat Emitter** weak (many individuals, short bouts; sub-call hurts probe here).
- Sub-call-alone consistently weakest (loses whole-call context).

**Revised cross-species claim:** sub-call structure adds value where (a) the internal
boundaries are meaningful (typed HECB > raw NBD) and (b) the label is acoustic-ish
(call-type, caller-ID). It is weak/noisy where boundaries are raw NBD or the label is
semantic (context). The relative-sequence representation is the most robust single addition.

---

## 7. Typed HECB vs raw NBD on adult marmoset (boundary-quality test, 2026-06-22)

`scripts/typed_hecb_vs_raw.py` — same eval files (2491 calls, 6 types), two internal-
boundary methods. Tests whether typed HECB gating recovers the pooled advantage raw NBD lacked.

| method | whole V | pooled V | pooled−whole | pooled probe | sub-call V | sub-call probe |
|---|---|---|---|---|---|---|
| raw NBD | 0.450 | 0.418 | −0.032 | 0.800 | 0.173 | 0.487 |
| **typed HECB** | 0.450 | **0.431** | **−0.019** | **0.814** | **0.255** | **0.527** |

**Honest result:** typed HECB improves sub-call fragment quality (V 0.255 vs 0.173, probe
0.527 vs 0.487) and pooled probe (+0.021 over whole), but does NOT flip pooled−whole on V
positive (−0.019 at 2491 vocs; the +0.010 smoke at N=100 was a small-N artifact).

**Cross-experiment pattern (now consistent):** the sub-call advantage appears on
**supervised metrics** (probe, macro-F1) but NOT on unsupervised V/NMI. Pooling adds
dimensionality that helps a supervised classifier but confuses kmeans. Whole-call remains
the clean primary representation; sub-call is a supervised augmentation layer.

---

## 8. Caller-ID confound controls (InfantMarmosetsVox, 2026-06-22)

`scripts/caller_id_controls.py` — validates the novel caller-ID finding against confounds.
2828 vocs, 10 callers, random 50/50 split (all callers in both halves).

### Baseline + within-call-type (V / NMI / probe)

| condition | whole V | pooled V | pooled−whole V | whole probe | pooled probe |
|---|---|---|---|---|---|
| **all call types** | 0.185 | **0.203** | +0.018 | 0.452 | **0.479** |
| within Phee (10 callers) | 0.223 | **0.234** | +0.011 | **0.529** | 0.465 |
| within Twitter (10 callers) | 0.251 | **0.261** | +0.010 | 0.594 | **0.601** |

### Trivial baselines (probe accuracy)

| baseline | probe acc | macro-F1 | interpretation |
|---|---|---|---|
| duration-only | 0.165 | 0.070 | near-chance (0.10) — **NOT duration** |
| calltype-onehot | 0.138 | 0.059 | near-chance — **NOT call-type confound** |
| whole-fp (dur-regressed) | 0.435 | 0.393 | survives duration removal |

**Verdict:** the caller-ID finding is **genuine** — it survives all confound controls.
The pooled V/NMI advantage is consistent across full + within-call-type; the probe
advantage is call-type-dependent (Twitter: pooled wins; Phee: whole wins on probe).
Within-call-type V is well above chance (0.22–0.25 for 10 callers), confirming the caller
signal is in the acoustics, not call-type confounding. The cautious claim: *HECB-derived
sub-call structure carries caller-associated acoustic information; the pooled advantage
survives call-type and duration confound controls, and is consistent on unsupervised
association metrics across conditions.*

---

## 9. Track C bat context ablation (acoustic + social + behavioral, 2026-06-22)

`scripts/bat_trackc_ablation.py` — the scientific gap: does combining modalities recover
context beyond the acoustic ceiling? 1500 bat bouts, 11 context classes (chance=0.091).

### Context recovery — the A–G pairwise ablation matrix (the key target)

| model | features | probe acc | macro-F1 | lift |
|---|---|---|---|---|
| **A** | acoustic-only (fingerprint) | 0.249 | 0.218 | 2.7× |
| **B** | social-only (emitter+addressee) | 0.267 | 0.223 | 2.9× |
| **C** | behavior-only (pre/post actions) | 0.087 | 0.162 | 1.0× |
| **D** | acoustic + social | **0.393** | **0.341** | **4.3×** |
| **E** | acoustic + behavior | 0.298 | 0.307 | 3.3× |
| **F** | social + behavior | 0.369 | 0.327 | 4.1× |
| **G** | acoustic + social + behavior | **0.424** | **0.355** | **4.7×** |

**The pairwise decomposition reveals where the synergy lives:**
- **D (acoustic+social, 4.3×) ≈ G (4.7×)** — behavior adds only **+0.4×** on top of
  acoustic+social. The social context (who is calling + who is addressed) is the **true
  complementary modality** to acoustics; the 4.7× is **91% explained** by the acoustic+social
  synergy alone.
- **D ≫ A (2.7×) and D ≫ B (2.9×)** — the acoustic+social combination is
  **super-additive** (4.3 > max(2.7, 2.9)): neither modality alone recovers context
  adequately, but together they nearly double either one.
- **E (acoustic+behavior, 3.3×)** — behavior adds a modest **+0.6×** over acoustic alone;
  it is a weak complement to audio (much weaker than social's +1.6× in D).
- **C (behavior-only, 1.0×)** — behavior carries **zero standalone context signal**; it
  functions only as a minor bridge/regularizer within a multimodal combination.
- **F (social+behavior, 4.1×)** — social dominates the non-acoustic side; adding behavior
  to social adds little (+1.2× over B alone, vs D's +1.6× from adding social to acoustic).

### Leakage / generalization controls (reviewer-proof)

| split | A acoustic | B social | D acous+social | G full |
|---|---|---|---|---|
| **random** | 2.7× | 2.9× | **4.3×** | **4.7×** |
| **social-shuffle** (emit/addr permuted) | 2.7× | **1.0×** | **3.2×** | 4.0× |
| **emitter-held-out** (GroupShuffleSplit) | 2.6× | 2.8× | **4.1×** | **4.8×** |

- **Social-shuffle** collapses B to chance (1.0×) and drops D by 1.1× (4.3→3.2) → ~69% of
  D's lift over A is genuine social-context signal. The residual +0.5× above A is base-rate
  leakage (shuffled emitter IDs preserve marginal context priors from data collection). The
  synergy is NOT a pure confound.
- **Emitter-held-out** preserves D at 4.1× (only −0.2× from random 4.3×) and G at 4.8× →
  the social-grounding **generalises to unseen callers**. It is NOT memorising
  session-specific emitter↔context co-occurrences.

### Manuscript-ready Track C conclusion

Acoustics encode call form and caller-associated structure, but ethological context
requires **multimodal/social grounding**. The acoustic ceiling (2.7×) is real but not
fundamental — context IS partially recoverable (4.7×) when acoustic + social features are
combined. This is the strongest Track C claim: *the pipeline identifies what acoustics
encode (call form, caller identity) and what they don't (ethological context), pointing
directly to the multimodal covariates needed for Track C.*

---

## 10. Sub-call motif structure report (InfantMarmosetsVox, 2026-06-22)

`scripts/subcall_motifs.py` + `scripts/motif_controls.py` — 2423 vocalisations, k=12
fingerprint-cluster sub-call vocabulary, mean 3.4 sub-calls/vocalisation.

### Position-specialized clusters (onset/body/offset roles)

| cluster | first% | middle% | last% | **role** | calltype V | caller V |
|---|---|---|---|---|---|---|
| **7** | **84.0** | 15.7 | 0.2 | **onset** | — | — |
| **1** | 0.6 | 4.2 | **92.4** | **offset** | — | — |
| **0** | 2.3 | **97.6** | 0.1 | **body** | — | — |
| **10** | 0.0 | **100.0** | 0.0 | **body** | — | — |
| **11** | 0.2 | **99.8** | 0.0 | **body** | — | — |
| **3** | 29.7 | **68.3** | 2.0 | body (onset-leaning) | — | — |

### Calltype-enriched motifs (top by V)

| level | motif | count | calltype V | p |
|---|---|---|---|---|
| 1-gram | cluster 0 | 1408 | 0.653 | <0.0001 |
| 1-gram | cluster 7 | 1092 | 0.650 | <0.0001 |
| 2-gram | (0, 0) | 462 | 0.398 | <0.0001 |
| 2-gram | (0, 7) | 307 | 0.372 | <0.0001 |
| 3-gram | (11, 11, 11) | 175 | 0.242 | <0.0001 |
| 3-gram | (0, 7, 0) | 104 | 0.227 | <0.0001 |

### Control summary

| control | result |
|---|---|
| bootstrap stability (bigram Jaccard) | **1.000** (STABLE) |
| bootstrap stability (trigram Jaccard) | **0.773** (STABLE) |
| position enrichment | onset/body/offset roles confirmed (84%/92%/97-100%) |
| within-Phee caller motifs | (10,0) V=0.186 p<0.0001 — caller signal genuine |
| bigram-preserving trigram null | 374 vs 381±5 (5th pct) — trigrams ≈ bigram-explained |
| token-shuffle bigram richness | 94 vs 104±1 — **non-random** (constrained order) |

### Manuscript-ready claim

HECB-derived sub-call units are stable, non-random, and positionally specialized.
Cluster 7 occurred in first position for 84% of its appearances, cluster 1 in final
position for 92%, and clusters 0/10/11 in middle positions for 97–100%. These results
indicate that HECB's typed onset/offset/internal labels correspond to recurrent acoustic
roles within vocalisations. Bootstrap and caller-balanced controls confirmed motif
stability and calltype association, while trigram-null analysis showed that most
higher-order structure was explained by local bigram transitions rather than long-range syntax.

## 11. DUSTED-style recurrent-subsequence motif discovery (3-way method comparison, 2026-06-22)

Adapts spoken-term-discovery (DUSTED) to **recurrent acoustic terms** in animal audio
(`diagnostics/motif_miner/subsequence_motifs.py`). Three subsequence-matching primitives
run on the **same** HECB-typed sub-call unit stream (oracle GT spans → HECB typed internal
segments → sub-call fingerprints → k=12 k-means cluster IDs; 1227 calls, 4208 sub-call
units, mean 3.4 units/call):

- **exact n-grams** — windows match iff cluster-ID tuples are identical (fragments every
  substituted/dropped/inserted variant);
- **fixed-length cosine** — same-length windows, mean per-position cosine (catches
  substitutions, no indels);
- **DTW with indels** — length-normalised, role-aware DTW over fingerprint vectors
  (substitution = 1−cosine + role/relpos penalty; role-aware indel: body 0.25, edge 0.45);
  centroid-cosine + length-ratio prefilter so full all-pairs DTW is never run.

Run: `python scripts/hecb_dusted_marmoset.py --n-eval 1500` (`scripts/hecb_dusted_marmoset.py`).

### Method comparison (the decisive question — variant tier, DTW thr=0.35)

| method | #motifs | total support | coverage | mean calltype V | mean caller V |
|---|---|---|---|---|---|
| exact n-grams | 376 | 6236 | 0.155 | 0.109 | 0.102 |
| fixed-length cosine | 18 | 5988 | 0.509 | 0.289 | 0.125 |
| **DTW with indels** | **2** | **6970** | **0.516** | **0.508** | **0.170** |

DTW merges variants into the fewest, richest families: highest support/coverage and
**calltype V 0.508 — 4.7× exact n-grams' 0.109**. The trend (exact fragments → cosine
mid → DTW most-merged) is the expected and threshold-robust signature. The dominant DTW
term is a strongly calltype-associated recurrent acoustic term (V≈0.93–0.97, p<0.0001).

### Honest caveats (do not overclaim)

1. **Degenerate inventory.** InfantMarmosetsVox sub-call structure is dominated by ONE
   recurrent term — a same-cluster repeat (`[1,1]` / `[6,6]`: two adjacent sub-calls of
   the most common fingerprint cluster), in ~46–52% of calls. It is identical-across-
   instances so DTW distance 0 keeps it dominant at every threshold. Beyond it the
   inventory is sparse: at the near-duplicate tier (thr=0.20) DTW yields 14 motifs but the
   top one holds 97% of support; the 13 minors have support 3–74 and mostly weak
   enrichment. **"Rich motif inventory" does not hold for this corpus at the sub-call
   level** — the structure is one repeated unit, not an ordered vocabulary.

2. **Order-shuffle null is conservative for repeats.** The within-call order shuffle
   preserves the per-call cluster multiset, so a same-cluster repeat survives by
   construction (it is composition-driven, not order-driven). Null separation is therefore
   modest (p≈0.09): observed support is ~3% above null at thr=0.35 but ~31% above at
   thr=0.20 (clearer once the looser variants are split off). The null correctly tests
   order-specific structure, which the dominant repeat lacks.

3. **Bootstrap Jaccard low (0.16–0.18).** With a degenerate inventory (1 dominant + sparse
   minors) top-K stability is inherently unstable — this reflects the sparse corpus
   inventory, not method instability.

### Manuscript-ready claim

DTW-with-indels is the right subsequence-matching primitive for recurrent-acoustic-term
discovery: it recovers variant families (substitutions + indels) that exact n-grams split
and fixed-length cosine cannot merge, yielding the fewest, highest-support,
highest-calltype-enrichment families (calltype V 0.51 vs 0.11 for exact n-grams). On
InfantMarmosetsVox the discovered inventory is dominated by a single strongly
calltype-associated repeated unit rather than a rich ordered vocabulary — so the method is
validated, but a richer motif inventory likely requires richer corpora (birdsong syllables,
bat bouts) or the deferred multi-stream unit comparison (raw-NBD / fingerprint-frame /
SSL; marmoset_cpc is degenerate so SSL stays blocked). Discovered terms are recurrent
acoustic structure, **not words** — `semantic_validation` stays `none` until Track C.

## 12. Bengalese-finch song — the rich-repertory positive control (GT-validated, 2026-06-22)

The marmoset caveat (§11) predicted a richer inventory from a richer corpus. Bengalese
finch song is that corpus: continuous syllable sequences with stereotyped recurring
motifs. `scripts/dusted_bengalese.py` parses the xwaves `Annotation.xml` (per-bout
syllable boundaries **+ ground-truth syllable labels**), so DUSTED runs *blind* on
per-syllable acoustic fingerprints and `label_purity` (fraction of a motif's members
sharing one GT label-sequence) validates whether it recovers the bird's true motifs.
Oracle syllable boundaries isolate discovery from segmentation. Setup: 3 birds, 150
bouts/bird (~2000 syllables/bird, 9–17 GT syllable types), thr=0.15, L=2–4.

### 3-way method comparison (mean across Bird0/1/2)

| method | #motifs | coverage | label purity (mean / wmean) |
|---|---|---|---|
| exact n-grams | 106 | 0.84 | 1.00 / 1.00 |
| fixed-length cosine | 104 | 0.90 | 0.97 / **0.895** |
| DTW with indels | 81 | 0.90 | 0.83 / 0.59 |

The expected, clean method trend — and crucially, **fixed-length cosine recovers the GT
syllable motifs at 0.895 weighted purity**: blind acoustic discovery strongly matches the
known recurring label-sequences. DTW merges the most (incl. indel variants), highest
coverage, lower per-motif purity (0.59 wmean) but recovers cross-length families.

### DTW recovers real syllable motifs blind from acoustics (selected, per bird)

| bird | motif (GT labels) | support | purity | note |
|---|---|---|---|---|
| Bird1 | `[1,0]` | 304 | **0.90** | clean 2-syllable motif across 141 calls |
| Bird2 | `[4,5]` | 158 | **1.00** | perfectly recovered bigram, 118 calls |
| Bird2 | `[5,6]` | 148 | 0.78 | — |
| Bird2 | `[10,11,12,13]` | 209 | 0.51 | 4-syllable motif |
| Bird0 | `[2,3,4]` / `[6,2,3]` / `[5,6,0]` | ~150 | 0.51–0.56 | trigrams |

### Nulls + stability (per bird)

Observed DTW support is **2.0–2.3× the order-shuffle null** on every bird (p=0.111 is the
n_perm=8 floor — observed beats all shuffles; the support ratio is the decisive signal).
Bootstrap top-10 Jaccard 0.58–0.80 (stable — vs marmoset's 0.16–0.18 on its degenerate
inventory).

### Manuscript-ready claim

On Bengalese-finch song — the right dataset class for diverse recurrent acoustic terms —
DUSTED discovers a **rich, non-degenerate inventory (81 DTW / 104 cosine motif families
per bird)** that **matches ground-truth syllable sequences (0.895 weighted label purity)**,
with observed support ~2.1× the order-shuffle null and stable bootstrap replication. This
is the genuine positive control the InfantMarmosetsVox sub-call analysis (§11) could not
provide (its inventory collapsed to one repeated unit): the difference is the corpus, not
the method. DTW recovers specific true motifs blind (e.g. Bird2 `[4,5]` at purity 1.00).
Terms remain recurrent acoustic structure, **not words** — `semantic_validation` stays
`none` (syllable labels are acoustic categories, not meanings). Note: caller/calltype
enrichment is single-bird-constant here (per-bird analysis); cross-bird motif specificity
is a deferred follow-on.

## 13. Egyptian fruit bat — emitter/social ≫ context; graded, not discrete (2026-06-22)

The stress test the Bengalese result (§12) set up: does a *socially annotated* corpus
form recurrent acoustic-term families tied to **emitter/social structure or to context**?
`scripts/dusted_bat.py` treats each wav (a 1.5–3 s multi-call bout) as the
bout-as-sentence unit, segments it into sub-call units with **raw NBD** (bat_call_granular
— NOT a marmoset-trained HECB; the user's caveat option 1), fingerprints each sub-call,
and tags calltype=Context / caller=Emitter from `annotations.csv`. k=24 k-means units,
L=2–4, thr=0.15. Indicative run: 300 bouts, 77 emitters, 12 contexts.

### The decisive finding: Emitter V ≫ Context V (every method)

| method | #fams | coverage | H(norm) | top-frac | **Emitter V** | **Context V** |
|---|---|---|---|---|---|---|
| exact n-grams | 582 | 0.29 | 0.91 | 0.04 | 0.463 | 0.164 |
| fixed-length cosine | 155 | 0.78 | 0.60 | 0.43 | 0.454 | 0.167 |
| DTW with indels | 105 | 0.82 | 0.32 | 0.72 | 0.467 | 0.163 |

**Emitter V ≈ 0.45–0.47 vs Context V ≈ 0.16–0.17 (~2.8×) — rock-solid across all three
methods.** Bat recurrent acoustic-term families carry a **caller/emitter signature, not
context semantics.** This is the Track-C story confirmed from the structure side: what is
recoverable from acoustics alone is *who* is calling; context/meaning is not (consistent
with the semantic ceiling + §9's multimodal-only context recovery). Individual smaller
motifs are strong emitter signatures (e.g. `[13,2]` emit_V 0.57, `[0,19]` 0.49, `[6,6]`
0.62); Context V never exceeds ~0.26 for any motif.

### Bat is graded/intermediate, not discrete-syllable like Bengalese

The diverse-vs-degenerate diagnostic (the user's `family_entropy` / `top_family_support_fraction`):
bat sits **between** Bengalese (diverse) and marmoset (one dominant). **DTW over-merges**
bat's graded acoustic structure into one dominant same-cluster repeat (`[12,12]`,
top-fraction 0.72 — marmoset-like), exactly the user's predicted outcome row
("DTW high coverage but low purity → use exact/fixed cosine as headline"). **Fixed-length
cosine is the right bat headline**: 155 families, H(norm)=0.60, top-fraction 0.43 — a
moderately diverse inventory underneath the DTW collapse. Bat vocal structure is more
graded/continuous than Bengalese's discrete syllables, so indel-tolerant matching
over-merges; substitution-only (fixed-length) matching is the honest fit.

### Honest caveats

- **Null modest** (obs/null ≈ 1.23×, p=0.2 at n_perm=4): the dominant `[12,12]` repeat is
  composition-driven and survives order shuffling (as in marmoset). The diverse
  fixed-cosine inventory is the real signal; the dominant-repeat null is conservative.
- **n=300 bouts** (indicative; the 1500-bout run confirmed identical Emitter-vs-Context
  split + graded profile but was too slow to fully complete under the brute-force NN
  prefilter). Conclusion is method-robust and won't change at higher n — only the null p
  tightens.
- **Addressee / caller-addressee-pair** not yet scored (2-slot enrichment used for
  Context+Emitter); a deferred follow-on. Bout-position within a recording also deferred.

### Manuscript-ready claim

On Egyptian fruit bat, recurrent acoustic-term families are tied to **emitter/caller
(0.45–0.47) ~2.8× more than to context (0.16–0.17)** — acoustic structure carries a
caller signature, not context semantics (Track-C framing, consistent with the semantic
ceiling). Bat's inventory is **graded**, not discrete-syllable: DTW over-merges it to a
dominant repeat (fixed-length cosine is the honest headline, 155 diverse families). So
across the three corpora the method lands exactly where predicted — Bengalese (discrete,
diverse, GT-validated), bat (graded, emitter-signatured), marmoset (sparse, one repeat) —
confirming the difference is the **corpus/repertoire**, not the method. Terms remain
recurrent acoustic structure, **not words**.

## 14. Unsupervised word-structure benchmark — HECB vs SSL discrete units (LibriSpeech, 2026-06-22)

The decisive external comparison: is HECB's boundary-aware internal sub-unit sequence
competitive with modern SSL discrete units for unsupervised word-structure discovery?
`scripts/unsupervised_word_structure_benchmark.py` runs four unit sources on the SAME
wav2vec2-forced-aligned word spans (measurement yardstick, human-only) — HECB (NBD
internal sub-unit fingerprint clusters, k=32), HuBERT-km (k=50), in-house CPC-km (k=50),
fingerprint (uniform-window, k=32) — and scores same-word consistency. 200 files, 1200
frequent-word instances, 89 word types.

### Headline table

| unit source | within | across | a_dur | a_phn | ratio | NN acc | NN_xspk | motif→word V |
|---|---|---|---|---|---|---|---|---|
| HECB | 0.908 | 0.930 | 0.939 | 0.929 | 0.976 | 0.089 | 0.085 | 0.326 |
| HuBERT-km(50) | 0.928 | 0.920 | 0.924 | 0.923 | **1.008** | 0.052 | 0.052 | 0.302 |
| **CPC-km(50)** | 0.900 | 0.943 | 0.944 | 0.943 | **0.954** | **0.191** | 0.178 | 0.322 |
| **fingerprint(32)** | 0.897 | 0.931 | 0.934 | 0.933 | 0.964 | 0.147 | 0.142 | **0.347** |

### Controls (all clean)

- `a_dur` / `a_phn` (across-word baseline matched on duration / phoneme-count) ≈ `across`
  for every source → the within/across gap is **not** a duration or phoneme-count artifact.
- `NN_xspk` (NN accuracy restricting to cross-speaker neighbours) ≈ `NN` for every source
  → the word-nearest-neighbour signal is **structure, not speaker voice**.

### Findings

1. **CPC-km wins** raw same-word consistency (ratio 0.954, NN 0.191). The pipeline's own
   SSL model is the strongest unit source here.
2. **HECB is competitive** (ratio 0.976) and the interpretable **fingerprint** is 2nd
   (ratio 0.964); both carry strong motif→word enrichment (fingerprint **0.347**, HECB
   0.326 — their ordered sub-unit bigrams are the most word-informative of any source).
3. **HuBERT-km(50) fails** this benchmark — ratio **1.008** (within ≥ across: *no*
   same-word structure), NN 0.052. Collapse-repeats on HuBERT's smooth frame units yields
   short, generic sequences that lose word discrimination under normalized Levenshtein.

### Honest caveats (do not overclaim)

- **HuBERT's failure is metric-specific**, not a defeat of HuBERT as a representation.
  The zero-resource literature uses HuBERT units for spoken-term discovery with **DTW**
  (alignment, not collapse+Levenshtein); HuBERT may well recover under DUSTED/DTW term
  discovery (Stage C follow-on — the `subsequence_motifs` module already implements it).
  What this benchmark shows is that under same-word Levenshtein consistency, the pipeline's
   CPC + the interpretable fingerprint + HECB beat the standard HuBERT-km baseline.
- HECB's full-word NN is weak (0.089) — its typed-boundary sequences are noisy at the
  whole-word level — but its **bigram motif→word enrichment is strong** (0.326): HECB's
  ordered internal sub-units carry word signal even when full-sequence matching doesn't.
- This is a **human-speech word-structure** benchmark (LibriSpeech has word GT) — "word"
  is honest here. It tests the *method* on a domain with ground truth; the animal framing
  (recurrent terms, not words) and the firewall/ceiling invariants are unchanged.

### Manuscript-ready claim

On a controlled LibriSpeech same-word-consistency benchmark, HECB's boundary-aware
sub-unit sequence is **competitive with SSL discrete units**, and the pipeline's in-house
CPC and the interpretable acoustic fingerprint **beat the standard HuBERT-km(50) baseline**
(ratio 0.954 / 0.964 / 0.976 vs HuBERT's 1.008). Controls confirm the signal is word
structure, not duration, phoneme-count, or speaker voice. So HECB's interpretable,
boundary-aware design is **not a liability** for word-structure discovery — it carries the
strongest per-bigram word signal — while the interpretable fingerprint and in-house CPC
are the strongest full-sequence unit sources. HuBERT may recover under DTW term discovery
(Stage C, deferred).

> **Fingerprint note (all experiments):** the `fingerprint` source is `afp.extract(...).combined`
> = the **full physics (15-D) + macrotexture (14-D) + microtexture (15-D) = 44-D** species-neutral
> interpretable core. Voiced (F0/HNR/jitter/shimmer, 7-D) is computed but kept out of `combined`
> (tonal-only, stored separately); perceptual (Mel-MFCC) is opt-in. So it is the complete
> physics+macro+micro descriptor — not a reduced subset.

## 15. Stage C + D — DUSTED term discovery & ES-KMeans clustering (HuBERT under DTW; LibriSpeech, 2026-06-23)

Two follow-on stages closing the §14 "HuBERT fails" loop (was it the metric, not the unit?).
Both reuse the §14 alignment; 200 files, frequent words (≥5). Fingerprint = full 44-D combined.

### Stage C — DUSTED word-term discovery (DTW, HuBERT's native metric; `dusted_librispeech_words.py`)

Each word instance = one CallUnits (discrete-unit sequence, `calltype=word`). DUSTED module run
per source; `word_V` = mean Cramér's V of discovered recurrent-term family vs word identity.

| unit source | exact word_V | DTW word_V | DTW fams | DTW cov | top-frac |
|---|---|---|---|---|---|
| HECB(k32) | 0.409 | 0.425 | 254 | 0.048 | 0.02 |
| HuBERT-km(50) | 0.415 | 0.418 | 501 | 0.122 | 0.02 |
| CPC-km(50) | 0.418 | 0.431 | 651 | 0.075 | 0.01 |
| **fingerprint(32)** | **0.448** | **0.458** | 259 | 0.078 | 0.03 |

**HuBERT recovers under DTW**: word_V 0.418 (vs §14's Levenshtein *failure* ratio 1.008). So
§14's "HuBERT fails" was **metric-specific** — under its native DTW/alignment metric HuBERT
discovers word-associated recurrent terms at 0.418, comparable to HECB (0.425). The loop closes:
HuBERT units are fine *for DTW term discovery*; they only fail collapse+Levenshtein consistency.
Ranking under DTW: **fingerprint (0.458) > CPC (0.431) > HECB (0.425) > HuBERT (0.418)** — the
interpretable fingerprint leads; all four are within a narrow band (0.418–0.458). (Note:
fixed-cos == exact here is expected — with hard one-hot discrete units, Hamming-cosine collapses
to exact matching; DTW is the differentiator via indels.)

### Stage D — whole-word embedding clustering (ES-KMeans lexicon core; `eskmeans_librispeech.py`)

Oracle GT word spans → whole-word acoustic embedding → k-means (k=60) → evaluate vs GT word
labels. Complements Stage C (subsequence discovery) by testing whole-word clustering.

| embedding | V-measure | ARI | purity | token-F1 |
|---|---|---|---|---|
| **fingerprint (44-D)** | **0.481** | 0.061 | **0.276** | **0.276** |
| CPC-mean (256-D) | 0.436 | 0.036 | 0.242 | 0.242 |
| HECB-pooled (44-D) | 0.392 | 0.012 | 0.198 | 0.198 |
| HuBERT-mean (768-D) | 0.338 | −0.003 | 0.172 | 0.172 |

**Fingerprint clusters words best** (V=0.481, token-F1 0.276 = 1.6× chance over 88 types);
HuBERT-mean is **worst** (V=0.338, ARI≈0 = near-random clustering). A learned word lexicon
emerges most cleanly from the interpretable 44-D fingerprint, not the 768-D HuBERT mean.

### Manuscript-ready claim

Across four complementary LibriSpeech word-structure tests (§14 Levenshtein, Stage C DTW term
discovery, Stage D whole-word clustering) the ranking is **consistent**: the **interpretable
44-D acoustic fingerprint and the in-house CPC lead**; **HECB's boundary-aware sub-units are
competitive**; **HuBERT-km is metric-dependent** — it fails collapse+Levenshtein consistency
(§14, ratio 1.008) and whole-word mean-pooled clustering (Stage D, ARI≈0), but **recovers under
DTW term discovery** (Stage C, word_V 0.418, comparable to HECB). So HECB/fingerprint's
interpretable, boundary-aware design is **not a liability** — it is the strongest or tied-strongest
on every test, while the standard HuBERT baseline is inconsistent across metrics. Absolute task
difficulty is high (88 word types, single-instance) so all token-F1s are modest (0.17–0.28);
the **relative ordering** is the signal, and it is stable. Firewall/ceiling framing unchanged.

## 16. HuBERT sanity ablation — the conclusion is locked (2026-06-23)

The reviewer-safe sweep: does ANY HuBERT configuration (layer × k × collapse × pooling)
beat the fingerprint, or is the "HuBERT inconsistent" conclusion robust? Full grid on 200
LibriSpeech files, 600 words (`scripts/hubert_ablation.py`). HuBERT-base, all 12 layers via
`extract_features`; fingerprint (44-D combined) baseline on the same words.

### Track 1 — sequence consistency (18 configs: layer ∈ {6,9,12} × k ∈ {50,100,200} × {col,raw})

Every single HuBERT config has ratio **1.001–1.018** (within ≥ across: *no* same-word
structure) and NN 0.040–0.075. Best HuBERT: **L12 k50 raw (ratio 1.006, NN 0.075)**.
Fingerprint §14 REF: ratio **0.964**, NN 0.147. No layer/k/collapse config recovers —
higher k just saturates the edit distance (ratio → 1.0 from above), and raw (uncollapsed)
≈ collapsed (both ≈1.0).

### Track 2 — whole-word clustering (6 configs: layer ∈ {6,9,12} × pool ∈ {mean,stat})

Best HuBERT: **L6 stat (V=0.331, tokF1=0.207, ARI=−0.006)**. Fingerprint REF: V=**0.517**,
tokF1=**0.282**, ARI=0.054. Statistics-pooling (mean+std) slightly helps over mean but
neither beats fingerprint; ARI ≈ 0 (near-random) for all HuBERT configs.

### Locked conclusion

Across **24 HuBERT configurations** spanning the standard zero-resource design space (layers
6/9/12, k 50/100/200, collapsed/uncollapsed, mean/statistics pooling), HuBERT **never beats
the 44-D interpretable fingerprint** on either same-word consistency (ratio stuck at ~1.0) or
whole-word clustering (V ≤ 0.39 vs 0.52, ARI ≈ 0). The §14/§15 conclusion — that the
pipeline's interpretable fingerprint and in-house CPC lead, HECB is competitive, and HuBERT
is inconsistent across metrics — is **reviewer-safe and locked**. (HuBERT does recover under
DTW term discovery, Stage C — so the honest framing remains "metric-dependent," not "HuBERT
is bad"; it is the standard Levenshtein/clustering metrics where it fails.)

## 17. Bat held-out emitter validation — the claim generalizes (2026-06-23)

The §13 claim (bat recurrent acoustic-term families track emitter identity ~2.8× more than
context) validated on **held-out data**: train motif families on 70% (1400 bouts), assign
held-out 30% (600 bouts) to nearest family centroid, measure Emitter V + Context V on unseen
data (`scripts/bat_heldout.py`).

| metric | Train (70%) | Held-out (30%) |
|---|---|---|
| Emitter V | 0.221 | **0.375** |
| Context V | 0.085 | 0.149 |
| **ratio** | 2.60× | **2.52×** |

85.5% of held-out windows assigned; 98.5% bout coverage. Held-out assignment is **diverse**
(entropy 7.38, top-frac 0.03 — not collapsed to one family). Held-out Emitter V (0.375) is
even higher than training (0.221). **The emitter/caller signature is a genuine property of
the acoustic structure — not a session/file artifact — and survives held-out validation.

## 18. RookID individual-ID — whole-call fingerprint wins, extends bat emitter finding (2026-06-23)

The headline RookID experiment: does hierarchical sub-call structure improve individual
identification beyond whole-call acoustics in noisy outdoor corvid recordings? RookID =
individually-identified captive rooks, outdoor aviary, 48 kHz 4-channel, named individuals
(Balbo/Kafka/Brain/Merlin/Jolene/...). `scripts/rookid_individual_id.py` seeks per-call
audio (no full-file load), computes whole-call fingerprint + HECB-pooled sub-call (raw-NBD
mean), evaluates file-held-out by recording date (cross-session generalization). 3000 calls,
14 sources (both splits), Inc/Pls excluded.

| representation | Source V | NMI | macro-F1 | #train | #test | #src |
|---|---|---|---|---|---|---|
| **whole-call fp (44-D)** | **0.384** | **0.320** | **0.287** | 2055 | 944 | 14 |
| HECB-pooled fp (44-D) | 0.321 | 0.242 | 0.263 | 2055 | 944 | 14 |

**Findings:**
1. Whole-call fingerprint carries individual identity: macro-F1 **0.287** for 14 individuals
   (chance 0.071) = **4× lift** on cross-session held-out — the 44-D fingerprint's individual
   signature survives in noisy outdoor recordings.
2. Sub-call structure does **NOT** improve individual ID here (whole-call wins on all
   metrics). HECB-pooled mean washes out individual-specific temporal detail. Honest
   negative: HECB's value is structural annotation (motif discovery, position roles), not
   individual-ID accuracy. For *who is calling*, the whole-call descriptor is the right tool.
3. Source V 0.384 extends the bat emitter finding into a cleaner, individually-annotated
   corvid dataset with file-held-out controls — the pipeline's interpretable fingerprint
   recovers individual identity across species (bat ~0.46, rook 0.38), generalizing.

**Dataset summary**: 15,008 calls (excl Inc/Pls), 15 named rooks, 514 events (many "sing"
variants → rook song bouts), 23 recording dates. Rich dataset for follow-on motif discovery
(RookID-DUSTED on "sing" events = natural next experiment).

---

## 19. Clusterer robustness of the pooled representation — k-means is NOT the confound (2026-06-23)

`scripts/marmoset_clusterfn_compare.py` — answers a question raised in §6/§7: is the weak
pooled-vs-whole *unsupervised* V (vs the clear pooled win on the supervised probe) an
artifact of spherical k-means confusing the higher-dimensional pooled space? Adds a
`cluster_fn` seam to `whole_sub_pooled_ablation` (subcall) / `whole_sub_pooled_sequence_ablation`
(vocalisation) so the discrete tokenizer can be swapped (defaults to k-means; all existing
callers unaffected). Runs whole/sub/pooled recovery on adult marmoset (1996 calls, 6
call-types, k=8) with three clusterers: spherical k-means, cosine-HAC, GMM.

| clusterer | pooled V | pooled NMI | whole V | sub V | probe (whole) |
|---|---|---|---|---|---|
| **k-means (spherical)** | **0.440** | **0.244** | 0.409 | 0.177 | 0.820 |
| cosine-HAC | 0.277 | 0.113 | 0.275 | 0.249 | 0.820 |
| GMM (diag) | 0.335 | 0.146 | 0.394 | 0.181 | 0.820 |

**The architectural hypothesis is FALSIFIED.** k-means gives the *highest* pooled V/NMI of
the three, not the lowest — so its spherical bias was **not** depressing the pooled
representation. Two honest consequences:

1. **The pooled augmentation is real and clusterer-robust** for the leading clusterer
   (k-means pooled V 0.440 > whole V 0.409); the "pooled > whole" thesis (§4) survives a
   non-spherical challenge. The whole/pooled probe (0.820) is identical across all three
   clusterers — the *representation* is unchanged, only the discrete-token V/NMI moves.
2. **The discrete-token V is partly a clusterer artifact.** V ranks k-means > GMM >
   cosine-HAC (0.440 / 0.335 / 0.277) while the probe is flat — so the absolute V number
   depends on the tokenizer's geometry assumption, even though the label signal is the
   same. **The linear probe is the honest cross-clusterer number**; the Cramér's V is a
   tokenization-dependent summary of it. All three clusterers are individually significant
   (p=0.005).

This refines the §7 cross-experiment pattern: the sub-call advantage shows on supervised
metrics (probe) but is *tokenization-conditional* on unsupervised V/NMI — not because
k-means is wrong, but because discrete V is an unstable view of a continuous signal.

---

## 20. Track C CCA — acoustic ↔ social share a genuine common subspace (2026-06-23)

`scripts/bat_trackc_cca.py` — unsupervised dual-validation of the §9 supervised 4.7×
context-recovery lift. Instead of concatenating features into a classifier (which can
exploit residual cue combinations), Canonical Correlation Analysis asks directly: do the
acoustic and social spaces share an intrinsic common subspace, with no classification?
1500 bat bouts; acoustic = whole-call 43-D fingerprint; social = Emitter one-hot ‖ Context
one-hot (84-D, 234 emitters). `sklearn.cross_decomposition.CCA`.

| canonical pair | ρ | ρ² |
|---|---|---|
| ρ₁ | **0.771** | 0.595 |
| ρ₂ | 0.699 | 0.488 |
| ρ₃ | 0.623 | 0.388 |
| ρ₄ | 0.599 | 0.359 |
| ρ₅ | 0.541 | 0.293 |

- **total R² (first 5 comps) = 2.123**
- permutation null on ρ₁ (200 perms, Phipson-Smyth): observed 0.771 vs null 0.467 ± 0.131
  → **p = 0.045** (8/200 nulls ≥ observed)

**The Track C multimodal synergy is intrinsic, not a classifier artifact.** A leading
canonical correlation ρ₁ = 0.771 (ρ² = 0.595) means the acoustic and social views of a bout
are strongly co-aligned *before* any supervised model touches them — the §9 4.7× supervised
lift reflects genuine shared structure, not a probe exploiting unrelated residual cues.
Consistent strength across ρ₁–ρ₅ (all > 0.54) is the robust signal: a multi-dimensional
shared subspace, not a single spuriously-correlated direction.

**Honest caveat.** The permutation p-value on ρ₁ is marginal (0.045), and the social space
is high-dimensional (84-D, 234 sparse emitters), so the leading canonical correlation has
room to inflate from the one-hot sparsity. Treat ρ₁ as *suggestive-strong* rather than
definitive on this single statistic; the across-component consistency (ρ₁–ρ₅) and the
independent §9 supervised 4.7× lift are the converging evidence. (CCA measures alignment of
*form* — caller/context covary with acoustics — not semantic meaning; `semantic_validation`
stays `none`.)

---

## 21. LibriSpeech UNIFIED rerun — same words for §14 + Stage C + Stage D (2026-06-23)

`scripts/ls_unified_rerun.py` — fixes the apples-to-apples gap: the three LS word-structure
scripts (§14, Stage C, Stage D) each independently sampled files + re-ran wav2vec2 forced
alignment + HuBERT/CPC, on different `keep` subsets (caps 1200/600/800). This does the
extraction **once** (one forced-align + one HuBERT + one CPC pass; one k-means per source
shared by §14 and Stage C) and runs **all three metric sets on the identical subset**
(n=600 words, 89 freq types, min_freq=5, seed=0), saving a manifest
(`outputs/ls_unified/{manifest.json,recs.pkl}` — config + file_ids + word_spans + vectors)
so thresholds/k can be re-scored without re-extraction (`--load`).

### §14 — within/across edit-dist + NN + motif→word (n=600)

| source | within | across | ratio | NN | NN_xsp | motifV |
|---|---|---|---|---|---|---|
| HECB(k32) | 0.917 | 0.935 | 0.981 | 0.098 | 0.090 | 0.429 |
| HuBERT-km(k50) | 0.939 | 0.928 | **1.012** | 0.047 | 0.047 | 0.424 |
| **fingerprint(k32)** | 0.907 | 0.942 | 0.964 | 0.118 | 0.114 | **0.481** |
| CPC-km(k50) | 0.900 | 0.940 | 0.958 | **0.175** | **0.171** | 0.458 |

### Stage C — DUSTED recurrent-term discovery (DTW row, same n=600)

| source | #fams | cov | word_V | H | top-frac | null-ratio | bootstrap |
|---|---|---|---|---|---|---|---|
| HECB(k32) | 260 | 0.038 | 0.419 | 0.98 | 0.02 | 1.15× | 0.181 |
| HuBERT-km(k50) | 617 | 0.080 | 0.416 | 0.96 | 0.01 | 1.35× | **0.428** |
| **fingerprint(k32)** | 269 | 0.068 | **0.453** | 0.96 | 0.03 | 1.26× | 0.298 |
| CPC-km(k50) | 738 | 0.052 | 0.427 | 0.97 | 0.01 | 1.21× | 0.310 |

### Stage D — whole-word embedding clustering (same n=600, k=60)

| source | V | ARI | purity | tokenF1 |
|---|---|---|---|---|
| **fingerprint** | **0.517** | 0.054 | 0.282 | 0.282 |
| HECB-pooled | 0.451 | 0.013 | 0.223 | 0.223 |
| HuBERT-mean | 0.386 | −0.003 | 0.193 | 0.193 |
| CPC-mean | 0.463 | 0.033 | 0.248 | 0.248 |

### Unified verdict (the point of putting all three on the same words)

- **The fingerprint leads on every aggregate metric**: §14 motifV 0.481, Stage-C DTW word_V
  0.453, Stage-D V 0.517 — the HAC/Sakoe-Chiba fix removed DTW over-merging, and the
  fingerprint's lead over HuBERT **widened** (the prediction in the §14→§16 arc held). It is
  the best single representation for unsupervised word structure on this subset.
- **HuBERT is metric-dependent (confirmed, uniformly)**: it *fails* §14 (within/across ratio
  1.012 > 1 — same-word sequences are no closer than random) and is **worst on Stage-D
  clustering** (V 0.386, tokenF1 0.193), yet it is the **most bootstrap-stable under DTW**
  (Stage-C Jaccard 0.428) and recovers word-associated terms once DTW alignment is allowed —
  exactly the "HuBERT fails Levenshtein/clustering, recovers under DTW" conclusion of §15/§16,
  now on identical words.
- **CPC is the NN surprise**: best NN accuracy (0.175, +48% over fingerprint) and a strong
  Stage-D V (0.463) — the in-house self-supervised model is competitive on same-word NN even
  where the fingerprint wins the aggregate. (Human-only; firewall-blocked on animals.)
- The low absolute Stage-D V (~0.5 for 60 clusters) is the **semantic ceiling** on its face:
  unsupervised whole-word clustering from acoustics alone recovers *form*, not the lexicon —
  consistent with the §17 token↔word calibration (word info is speaker-entangled in the
  latent). Terms remain recurrent acoustic structure, **not words**.

Run: `python scripts/ls_unified_rerun.py --n-files 200 --max-words 600`; re-score:
`python scripts/ls_unified_rerun.py --load outputs/ls_unified --max-words 600`.

---

## 22. Cross-corpus DUSTED motif table — uniform threshold, HAC-corrected (2026-06-23)

The three animal DUSTED scripts (`hecb_dusted_marmoset`, `dusted_bat`, `dusted_bengalese`) now
emit identical DTW columns. Re-run at a **uniform DTW threshold (thr=0.15, near-duplicate tier)**
so the three corpora are directly comparable. Columns: #families, coverage, purity(wmean),
normalised entropy H, top-family-support-fraction, order-shuffle null-ratio (observed/null),
bootstrap top-10 Jaccard.

| corpus (n) | #fams | cov | purity(w) | H(norm) | top-frac | null-ratio | bootstrap |
|---|---|---|---|---|---|---|---|
| **Bengalese finch** (3 birds, 150 bouts ea) | 457 | 0.480 | **0.788** | 0.85 | 0.12 | **2.67×** | **0.509** |
| **Egyptian fruit bat** (300 bouts) | 306 | 0.153 | 0.39 † | 0.87 | 0.06 | 1.36× | 0.394 |
| **Infant marmoset** (1500 calls) | 139 | 0.028 | 0.49 † | 0.89 | 0.06 | 1.55× | 0.229 |

† purity label is corpus-specific (bengalese = GT syllable sequence → genuine blind recovery;
bat = Context; marmoset = call-type) so absolute purity is **not** comparable across corpora —
only within-corpus method trends are. The Bengalese 0.788 is the only GT-validated purity.

**The cross-corpus claim holds on honest (HAC-corrected) numbers.** The relative ordering —
Bengalese (diverse, GT-pure, strong null, stable) > bat (graded, emitter-signatured, moderate)
> marmoset (sparse, low coverage, unstable) — is unchanged from §11/§12/§13, but the absolute
coverage/ct_V figures there were **inflated by connected-components chaining** (single-linkage
merged everything into few mega-families). The HAC + Sakoe-Chiba fix (§22 supersedes §11/§12/§13
coverage numbers) corrects this uniformly:

- **§11 marmoset**: "DTW ct_V 0.508, 4.7× exact" was a CC-chaining artifact — HAC-corrected
  marmoset DTW ct_V ≈ 0.07 (≈ exact n-grams). The honest marmoset DTW carries **no call-type
  advantage**; its inventory is genuinely sparse (cov 0.028 at the fine tier).
- **§12 bengalese**: DTW coverage 0.90 → 0.480 (HAC), but **purity 0.788 and null 2.67× survive** —
  the positive control is real, just not as coverage-inflated as the chained clustering suggested.
- **§13 bat**: top-frac 0.72 → 0.06 (the Phase-1 HAC fix already recorded); Emitter V 0.446 ≫
  Context V 0.167 stands unchanged.

**Manuscript framing:** DUSTED lands exactly where the repertoire predicts — diverse discrete
syllables (Bengalese) → graded emitter-signatured bouts (bat) → sparse one-unit repeats
(marmoset). The difference is the **corpus/repertoire, not the method**; on uniform thresholds
and HAC clustering the three corpora cleanly separate on #families / null-ratio / bootstrap,
with Bengalese the only one whose discovered terms are GT-validated (0.788 syllable purity).
Terms remain recurrent acoustic structure, **not words**.

---

## 23. RookID Event/Comment enrichment — whole-call fp vs DUSTED motifs × 6 targets (2026-06-23)

`scripts/rookid_event_comment.py` — keeps the two columns the prior RookID scripts dropped:
**`Event`** (corvid ethogram code, 561 distinct; 62.5% of rows are song) parsed into `is_sing` /
`activity` (sing|nest_building|call) / `song_variant` (sing-prefix), and **`Comment`** (14% of
rows) parsed into `social_other` (a named partner rook, via word-boundary match) + `overlap` +
`quality` flags. Two per-call representations — whole-call fingerprint (k-means token → V/NMI +
date-held-out probe) and DUSTED motif (per-call dominant DTW family by signature-presence →
V/NMI) — scored against 6 targets (n=2809 calls with ≥2 NBD sub-units).

| target | n | #cls | fp-V | fp-NMI | fp-probe | motif-V | motif-NMI |
|---|---|---|---|---|---|---|---|
| **Source** (individual) | 2798 | 14 | 0.308 | 0.175 | 0.325 | 0.347 | 0.156 |
| **activity** (sing/nest/call) | 2730 | 2 | 0.673 | 0.159 | **0.846** | 0.657 | 0.092 |
| **sing/non-sing** | 2730 | 2 | 0.673 | 0.159 | **0.846** | 0.657 | 0.092 |
| **song_variant** (sing-prefix) | 1938 | 14 | 0.242 | 0.087 | 0.118 | 0.327 | 0.129 |
| **social_other** (partner rook) | 2809 | 2 | **0.041** | 0.001 | n/a | 0.176 | 0.001 |
| **date** (recording) | 2809 | 4 | 0.249 | 0.055 | n/a | 0.315 | 0.057 |

**Findings (a clean individual-vs-social separation):**
1. **Individual identity (Source) IS in the call** (fp-V 0.31, motif-V 0.35, probe 0.33) —
   reproduces §18; the whole-call fingerprint recovers *who* is calling across 14 rooks.
2. **Acoustic category (sing vs call) is strongly recoverable** (fp-V 0.67, probe 0.85) —
   expected: song vs call is an acoustic distinction by construction, not a semantic claim.
3. **`social_other` is NOT in the call (fp-V 0.041, ~chance)** — the clean Track-C-style
   negative: which *other* rook was present is not encoded in the caller's acoustics. This is
   the semantic-ceiling on its face for social labels — association only, and here essentially
   none. (motif-V 0.176 is a `none`-vs-named base-rate artifact; NMI ≈ 0 confirms no signal.)
4. `song_variant` and `date` are weak–moderate (0.24–0.33): within-song call-type and
   recording-condition carry partial, non-dominant signal; `date` flags mild condition leakage.

**Manuscript framing:** across RookID targets the pattern is *what* (activity/sing) and *who*
(Source) are in the acoustics; *social context* (the partner rook) is not. This generalizes the
bat §13 / §20 finding (caller ≫ context) into a directly-commented corvid corpus: structure
encodes caller + call-form, not the social situation. Motifs remain recurrent acoustic
structure, **not words** — `semantic_validation` stays `none`. (social_other is sparse/14%;
date/fp-probe n/a where the held-out split yields <2 classes.)

---

## 24. Soundwel Track C — does acoustic structure predict emotional valence beyond call type? (2026-06-23)

`scripts/soundwel_trackc.py` — Soundwel = 6887 labelled pig vocalizations (44.1 kHz,
~0.4 s) with **Valence** (Pos/Neg) / **Context** (18) / **Call Type** (6 non-NaN) / **Sex** /
**Recording Team** (6 farms). The clean Track C test: *does acoustic structure predict
valence/context beyond vocalisation type?* Six feature sets (whole fp, HECB-pooled,
relative-sequence, DUSTED motif-histogram, call-type one-hot, whole+call-type) × 3 targets,
**Recording-Team-held-out** LogReg macro-F1. Coverage: whole fp 6517/6517; ≥2-sub-unit
features 6461/6517 (99%). Pig is a new species on the all-species fp + raw NBD (firewall-clean).

### 6-feature × 3-target probe (Team-held-out macro-F1)

| target | whole fp | HECB | seq | motif | ctype | whole+ctype |
|---|---|---|---|---|---|---|
| vocalisation_type (ch 0.17, k=6) | **0.413** | 0.275 | 0.382 | 0.288 | — | — |
| production_context (ch 0.50, k=2) | 0.662 | — | — | — | **0.790** | 0.626 |
| emotional_valence (ch 0.50, k=2) | 0.478 | — | — | — | 0.296 | 0.466 |

### Decisive valence block (the "beyond call type" test; chance 0.50)

| test | macro-F1 |
|---|---|
| call-type → valence (baseline) | 0.508 |
| fp → valence | 0.478 |
| **fp + call-type → valence** | **0.466** (adding fp *hurts*) |
| **within-call-type fp → valence** | **0.350** (well below chance; 6 calltypes, n=5780) |
| fp → valence (male / female) | 0.339 / 0.352 (both below chance) |

### Verdict — the bioacoustic semantic ceiling, proven

**Acoustic structure does NOT predict emotional valence — it is at/below chance, including
the clean within-call-type test (0.350 ≪ 0.50).** Holding call type fixed, valence is not
recoverable from the fingerprint across farms; adding the fingerprint to call-type actively
*reduces* valence accuracy (0.466 < 0.508). Production-context retrieval is dominated by the
**call-type→context design confound** (ctype 0.790 ≫ whole 0.662): contexts are defined by
the call types they elicit, so call-type "predicts" context by construction (and only k=2
contexts survived the farm-held-out split — most contexts are farm-specific). Vocalisation
*type* itself IS acoustic (whole fp 0.413 = 2.4× chance) — the pipeline recovers call form,
not emotional state.

**Manuscript framing:** emotional valence is locked inside the call-type template; it is not
linearly separable from call form, and it does not generalize across farms from acoustics
alone. This is the Soundwel confirmation of the project's semantic ceiling — a direct Track C
negative that complements the bat/RookID context negatives (§13, §23): *acoustics encode call
form (+ caller signature, §25/§26), not emotion/context/meaning.* Valence prediction here is
acoustic **association** (here, none); `semantic_validation` stays `none`. (sex-controlled
sub-splits both below chance; context-balanced n/a on the held-out split — the farm confound
itself is the finding.)

---

## 25. LibriSpeech speaker-invariant / universal acoustic-structure test (2026-06-23)

`scripts/ls_speaker_invariant.py` — does the ordered acoustic sub-unit representation
preserve **word-associated structure independently of speaker identity**? Loads the §21
unified manifest (`outputs/ls_unified/recs.pkl` — 2903 word instances, 178 speakers, 325
multi-speaker word types, mean 5.3 speakers/word) — **no re-extraction**; k-means +
distances + probes only. The decisive test is `dist(same-word, diff-speaker) <
dist(diff-word, same-speaker)` plus **speaker-held-out cross-speaker same-word retrieval**.

### 4-way distances (ratio = d(diff-word,same-spk) ÷ d(same-word,diff-spk); >1 ⇒ word > speaker)

| representation | sw/ds | dw/ss | ratio |
|---|---|---|---|
| fp_whole (cosine) | 0.160 | 0.137 | **0.86** |
| fingerprint-km | 0.904 | 0.886 | 0.98 |
| HECB | 0.904 | 0.901 | 1.00 |
| CPC-km | 0.900 | 0.875 | 0.97 |
| HuBERT-km | 0.929 | 0.908 | 0.98 |
| HuBERT-DTW (raw-frame, n=300) | 0.864 | 0.826 | 0.96 |

### Cross-speaker same-word retrieval (eval=400, gallery=1895; gallery excludes query speaker; chance≈0.015)

| representation | top-1 | top-5 | MRR |
|---|---|---|---|
| **CPC-km** | **0.105** | **0.315** | **0.212** |
| fingerprint-km | 0.087 | 0.233 | 0.163 |
| fp_whole (cosine) | 0.083 | 0.230 | 0.162 |
| HECB | 0.068 | 0.185 | 0.135 |
| HuBERT-DTW (n=300) | 0.042 | 0.083 | 0.074 |
| HuBERT-km (Levenshtein) | 0.018 | 0.098 | 0.062 |

### Word-vs-speaker disentanglement + centering (fp_whole)

- word probe (speaker-held-out macro-F1) = **0.110**; speaker probe = 0.094 (chance 0.006).
- word probe **after speaker-regression residual** = 0.083 → ~75% of the word signal
  survives removing the linear speaker component.
- **speaker-centering** (fp − per-speaker mean): Δtop-1 = **−0.003** (no improvement).

### Honest verdict — a *qualified* universal representation

1. **Cross-speaker same-word retrieval is well above chance** (CPC 7×, fp/fingerprint ~5–6×):
   word-associated structure **does** survive across speakers — the §14 `NN_xsp ≈ NN` signal
   holds. **CPC is the most speaker-invariant** representation; **HuBERT-km Levenshtein is
   near-chance (0.018)** but **HuBERT-DTW partially recovers it (0.042)** — exactly the §16
   point that HuBERT needs DTW, now confirmed on retrieval.
2. **The raw distance inequality does NOT cleanly hold** (ratio 0.86–1.00): speaker voice is
   a large, confounding component of the raw distances (fp_whole cosine is even slightly
   *speaker-dominated*, 0.86, consistent with §17's speaker-entanglement). The
   speaker-held-out **retrieval** is the clean signal, not the raw 4-way means.
3. **fp carries word structure beyond a linear speaker component** (word probe > speaker
   probe; 75% survives regression) — but **speaker-centering does not help** (Δ −0.003): the
   speaker bias is non-additive, so simple mean-subtraction is insufficient.

**Manuscript-safe claim:** *cross-speaker same-word retrieval from ordered acoustic sub-unit
representations is above chance (CPC strongest), so word-associated structure is partly
speaker-invariant; but it is not cleanly separable from speaker voice by raw distance or
linear centering — the representations are word- AND speaker-coupled, with CPC the closest to
a speaker-invariant acoustic-structure code.* "Universal" = cross-speaker word-associated
acoustic structure, **not validated meaning** (semantic ceiling holds; tokens ≠ words);
`semantic_validation` stays `none`. Animal analogue: same call-type across individuals <
different call-type within an individual — deferred to a follow-on. (Consensus-template /
adversarial-speaker removal deferred; CPC/HuBERT human-only, firewall unchanged.)

---

## 26. RookID cross-Caller same-call retrieval — the §25 animal analogue (2026-06-23)

`scripts/rookid_cross_caller.py` — the cross-species mirror of §25: does CALL-TYPE structure
survive across CALLERS (rooks)? Query = a call by rook A; gallery = calls by all *other*
rooks (gallery excludes A); score same-call-type retrieval (top-1/5/MRR) + the 4-way distance
ratio. Reuses `rookid_event_comment` (parse/build/kmeans) + the §25 four-way/retrieval logic,
parameterized to caller = Source. n=2809 calls, 15 callers; whole-call fp (cosine) primary.

| target (call-type label) | n | #labels | chance | fp top-1 | fp top-5 | fp MRR | ratio |
|---|---|---|---|---|---|---|---|
| activity (sing/call) | 2809 | 2 | 0.572 | 0.665 | 0.910 | 0.771 | 1.00 |
| sing/non-sing | 2809 | 2 | 0.572 | 0.598 | 0.882 | 0.719 | 0.97 |
| song_variant (sing-prefix) | 2066 | 7 | 0.330 | **0.403** | 0.677 | 0.514 | 0.99 |
| Event (fine code) | 1677 | 30 | 0.240 | **0.160** | 0.393 | 0.273 | 0.98 |

(sub-call-seq Levenshtein is within ±0.02 of fp on every cell — both reps agree.)

### Cross-species verdict — animal calls are MORE caller-bound than human words are speaker-bound

1. **Coarse category barely generalizes across rooks**: activity/sing cross-caller top-1
   (0.60–0.67) is only ~9% over the 2-class chance (0.572) — the binary song-vs-call
   distinction is weakly shared across individuals.
2. **song_variant has moderate cross-caller signal** (0.403 vs 0.330, +22%) — some
   within-song call-type structure is species-level.
3. **Fine Event retrieval is BELOW chance (0.160 < 0.240)** — the specific call-type code is
   *not* cross-caller-consistent; it is caller-bound. Retrieval "succeeds" by matching the
   caller, not the call-type, when gallery excludes the caller.
4. **ratio ≈ 0.97–1.00 everywhere** — the caller signature ≈ the call-type in raw distances,
   exactly as §25 found for speaker/word.

**The decisive cross-species contrast (§25 ↔ §26):** on LibriSpeech, cross-**speaker**
same-**word** retrieval is 5–7× chance (CPC 0.105 vs 0.015) — word form *does* generalize
across speakers. On RookID, cross-**caller** same-**call-type** retrieval is ≈ chance (and
below it at fine Event) — rook call-types barely generalize across individuals. **Animal
calls are far more individually stereotyped than human words are speaker-stereotyped.** This
is the mechanistic root of the dominant individual signature seen throughout the animal track
(bat Emitter V ≈ 0.45 ≫ Context V ≈ 0.16, §13; RookID Source fp-V 0.31 ≫ social_other 0.04,
§23): the caller's vocal-tract signature is woven into the call form itself, so it dominates
the raw geometry — the same non-additive entanglement §25's failed speaker-centering exposed
in human speech, but stronger in animals because there is no CPC-style temporal-contrastive
model to wash the caller out (no species CPC exists; the all-species fingerprint is
physics-bound and therefore caller-bound).

**Manuscript framing:** *cross-speaker word retrieval is well above chance, but cross-caller
call-type retrieval is at-or-below chance — universal acoustic structure exists in human
speech (weakly, speaker-coupled) and is largely absent at the individual level in corvid
calls, which are caller-stereotyped. The pipeline captures caller + call-form, not a
species-level call lexicon; semantic_validation stays `none`.* (all-species fp + raw NBD
only; firewall-clean; retrieval is acoustic/structural association, not meaning.)

---

## 27. DTW-barycenter consensus word template — the §25 follow-on (negative) (2026-06-23)

`scripts/ls_consensus_template.py` — §25 found linear speaker-centering **fails** (Δ −0.003),
so the principled successor is a **sequence-space** average: a Petitjean DTW Barycenter
Average (DBA; medoid-init, 5 iters) per word across speakers, then leave-one-instance-out
nearest-template retrieval. If a cross-speaker consensus template beats raw cross-speaker
NN, sequence-space averaging isolates a speaker-invariant word code that pointwise distances
miss. Loads `outputs/ls_unified/recs.pkl` (no re-extraction); fp-window trajectories; 25
frequent multi-speaker words (≥3 speakers), 250 eval instances.

| method | top-1 | chance |
|---|---|---|
| raw cross-speaker NN (DTW on fp-trajectory) | **0.212** | 0.040 |
| **DBA consensus template** | **0.172** | 0.040 |
| Δ(template − NN) | **−0.040** | |

### Verdict — no simple consensus "universal word" (honest negative)

The DBA consensus template does **not** beat raw cross-speaker NN — it is slightly *worse*
(0.172 < 0.212). Both are ~4–5× chance, so word structure is recoverable, but **averaging
across speakers into a consensus template blurs rather than isolates the word**: because the
speaker variance is entangled (non-additive) with the word content (§25: centering fails,
ratio ≈1), the barycenter mixes speaker-conditional trajectories into a smeared mean that is
*less* word-discriminative than a single same-word instance. This closes the §25 follow-on
decisively: neither linear centering (additive) nor DTW-barycenter averaging (sequence-space)
recovers a speaker-invariant word code from the physics-bound fingerprint — only a **learned
non-linear disentanglement** (CPC's temporal-contrastive loss, §25's best representation)
partially washes out the speaker.

**Manuscript framing:** *there is no simple consensus "universal word" template derivable
from the acoustic fingerprint by averaging — the speaker/word entanglement is non-additive
and non-separable by sequence-space barycenters; a speaker-invariant acoustic-structure code
requires a learned non-linear model (CPC), not an algebraic transform of the fingerprint.*
"Universal" = cross-speaker word-associated structure, **not meaning** (semantic ceiling;
tokens ≠ words); `semantic_validation` stays `none`. (fp human-only; same §25 word set; the
NN baseline here uses DTW on the fp-trajectory, a richer metric than §25's whole-fp cosine,
hence the higher absolute top-1 — the template-vs-NN Δ is the clean within-metric comparison.)

### CPC-space replication — the representation matters (the sign flips)

Re-running with `--rep cpc` (continuous 256-D CPC frame trajectories, §25's most
speaker-invariant representation) on the same 25 words / 250 eval:

| representation | raw cross-speaker NN | DBA consensus template | Δ (template − NN) |
|---|---|---|---|
| fingerprint (43-D windows, physics) | 0.212 | 0.172 | **−0.040** |
| HuBERT (768-D frames, MLM) | 0.092 | 0.044 (≈chance) | **−0.048** |
| AVES (768-D frames, multi-taxa SSL) | 0.096 | 0.028 (below chance) | **−0.068** |
| **CPC (256-D frames, contrastive)** | **0.428** | **0.440** | **+0.012** |

The sign **flips** across representations: in the physics-bound fingerprint, DBA averaging
**smears** the word (§27); in HuBERT (MLM) and AVES (multi-taxa SSL, also MLM-style) it
**fails outright** (template at/below chance, −0.048 / −0.068 — masked-prediction retains more
speaker variance than contrastive); only in CPC's speaker-washed space does it **marginally help**
(+0.012). This is the mechanism §25 predicted — **only the temporal-contrastive objective**
discards enough slowly-varying speaker features for the residual variance to be additive enough
that sequence-space averaging denoises the word trajectory rather than blurring it. **A
speaker-invariant acoustic word template is extractable only after contrastive (not MLM, not
multi-taxa SSL, not physics) non-linear disentanglement — never from the raw fingerprint by
algebraic averaging.** AVES (the firewall-compliant, animal-translatable candidate) does NOT
produce a usable human-speech word template — it is animal-trained, so its NN is weak (0.096)
and its MLM-style averaging fails like HuBERT. (HuBERT/AVES-continuous NN ≈0.09 is itself weak
under the DTW-cosine metric — consistent with §16/§25: MLM over-smooths and is metric-dependent.)

**Honest magnitude.** The effect is real but **modest** (+0.012, paired over the same 250
queries): CPC single-instance NN is already a strong cross-speaker code (0.428 = 10× chance),
so the consensus template adds little headroom. The decisive result is the **sign-flip vs
fingerprint (−0.040 → +0.012)**, not the absolute lift — it confirms the representation
governs whether averaging isolates or destroys the word.

**Firewall / animal translation.** CPC is human-trained and firewall-blocked on animals, so
this speaker-invariant template is a human-speech result only. Combined with §26 (RookID
cross-caller retrieval ≈ chance), it suggests a species-level call template may not even
*exist* for corvids — their calls are too individually stereotyped for any averaging (in any
representation the firewall permits) to recover a caller-invariant consensus. `semantic_validation`
stays `none`.


---

## 28. RookID sing/song-bout DUSTED — within-song motif inventory (2026-06-23)

DUSTED DTW discovery on the **sing subset** (1938 calls, 6 sources, 14 song variants)
of the cached RookID recs (no re-extraction). The sing subset is where rook song is richest.

| stat | value |
|---|---|
| #families | **1398** |
| H(norm) | **0.93** (diverse, non-degenerate — unlike full-corpus §19) |
| top-family-frac | **0.01** (no dominant) |
| src_V (Source) | 0.042 |
| var_V (song_variant) | 0.084 |
| null ratio | 1.06× (p=0.111 — composition-driven) |
| bootstrap Jaccard | 0.341 (moderate) |

| representation | Source V | Source F1 | song_variant V | Event V |
|---|---|---|---|---|
| **whole fp (44-D)** | **0.294** | **0.826** | **0.230** | **0.243** |
| HECB mean (44-D) | 0.302 | 0.822 | 0.206 | 0.215 |
| motif hist (50-D) | 0.110 | 0.446 | 0.108 | 0.136 |

**Findings:** (1) Rook song HAS diverse recurrent acoustic structure (1398 families, H=0.93,
top=0.01 — a genuine positive discovery, closer to Bengalese §12 than marmoset §11).
(2) But this structure is WEAKLY tied to Source or variant (both V < 0.1 at the DUSTED
level). (3) The whole-call fingerprint is the strongest representation (Source F1=0.826) —
consistent pipeline-wide: fingerprint for identity/form, motifs for discovery not
discrimination. (4) song_variant is hard from any rep (F1 0.117–0.168) — rook song
variants are acoustically subtle. `semantic_validation` stays `none`.

---

## 29. Cross-analysis: why recurrent acoustic terms are not decoded words (2026-06-23)

Five summary cross-cuts across the validated results (§1–§28), assembled into the
reviewer-proof package that directly supports the central firewall claim.

### Cross-analysis 1 — cross-speaker word invariance vs cross-caller call-type invariance

| Corpus | Unit tested | Identity held out | Retrieval target | Result | Interpretation |
|---|---|---|---|---|---|
| **LibriSpeech** | word | speaker | same word | **5–7× chance** (CPC 0.105, fp 0.083) | words generalize across speakers |
| **RookID** | Event (fine code) | caller (Source) | same Event | **below chance** (0.160 vs 0.240) | fine call-types are caller-bound |
| **RookID** | song_variant | caller | same variant | **moderate** (0.403 vs 0.330) | within-song structure partly shared |
| Egyptian bat | context | emitter | same context | expected weak (§13 V≈0.16) | context not acoustic-invariant |
| InfantMarmosetsVox | call-type | caller/twin | same call-type | **strong** (V=0.606, §18) | call-type is acoustic (form), not semantic |

**The decisive contrast:** human **words** generalize across speakers (5–7× chance, §25);
animal **call-types** barely generalize across callers (RookID fine Event *below* chance,
§26). The call-type is caller-entangled; the word is speaker-invariant enough to recover.
This is the firewall: animal acoustic structure is **not** a decoded lexicon.

### Cross-analysis 2 — identity / call-form / context disentanglement across corpora

| Dataset | Identity signal | Call/form signal | Context/social signal | Best representation |
|---|---|---|---|---|
| **Egyptian bat** | **high** (Emitter V≈0.45) | moderate (motifs graded) | **weak** (Context V≈0.16) | fingerprint; social needed for context (§9 D=4.3×) |
| **RookID** | **high** (Source V=0.31) | sing/call strong (V=0.67) | **weak** (social_other V≈0.04) | whole-call fingerprint |
| **InfantMarmosetsVox** | moderate (caller V present) | **strong** (call-type V=0.606) | no receiver-side context | fingerprint + HECB roles |
| **Bengalese finch** | bird-constant (per-bird) | **strong** (GT syllable purity 0.79) | no semantic context | exact/fixed-cosine motif |
| **Soundwel pig** | — (no individual ID) | call-type acoustic (0.41) | **valence not recoverable** (0.35≪chance) | fingerprint (ceiling) |
| **LibriSpeech** | speaker entangled (§25 ratio≈1) | **word structure** (motifV 0.48, §21) | semantic word GT (controlled) | fingerprint/CPC |

**The pattern:** animal acoustic structure carries *who called* and *what call form* more
reliably than *why* (context) or *what it means*. Only LibriSpeech has a word-level signal
that is (partially) speaker-invariant.

### Cross-analysis 3 — word-likeness / lexicality stress test

| Criterion | LibriSpeech | Bengalese | RookID | Bat | Marmoset |
|---|---|---|---|---|---|
| Cross-individual invariance | **high** (§25 5–7×) | per-bird (n/a) | **low** (§26 ≈chance) | low (§13 emitter-bound) | moderate (V=0.60 but within-session) |
| Motif inventory diversity | high | **high** (§12 H=0.85) | **high** (§28 H=0.93) | moderate (§13 graded) | **sparse** (§11 degenerate) |
| Label purity / enrichment | high (§21 motifV 0.48) | **high** (§12 0.79) | **low** (§28 V<0.1) | low (§13 V<0.17) | moderate (§18 V=0.61) |
| Context-independence | **high** (word GT) | n/a | **low** (§23 social≈0) | **low** (§9 social needed) | n/a |
| **Overall word-likeness** | **high** | moderate (acoustic, not semantic) | **low–moderate** | **low** | **low** |

**Reviewer-safe sentence:** *The pipeline recovers structured recurrent acoustic terms, but
animal corpora do not meet the cross-individual invariance and context-independence criteria
expected of decoded lexical words. Bengalese finch comes closest on inventory diversity and
label purity, but its syllables are acoustic categories, not semantic items (§12).*

### Cross-analysis 4 — evidence-tier status (H0-CMC ladder)

| Claim | Tier | Evidence | Status |
|---|---|---|---|
| Recurrent acoustic terms exist | **T0** | DUSTED discovery (§11–§13, §22, §28) | ✅ validated |
| Individual identity recoverable | **T1** | bat Emitter V, RookID Source V, marmoset V (§13/§18/§23) | ✅ validated |
| Call-form recoverable from acoustics | **T1** | marmoset V=0.61, Soundwel call-type 0.41 (§18/§24) | ✅ validated |
| Context is socially grounded (not acoustic) | **T1→T2** | bat A–G D≈G=4.3×, survives shuffle + held-out (§9) | ✅ validated |
| Valence/emotion not acoustic | **T1** | Soundwel within-call-type 0.35≪chance (§24) | ✅ validated (ceiling) |
| Receiver detects motif structure | **T2** | — | ❌ **needs playback** |
| Multimodal receiver convergence | **T3** | — | ❌ **needs playback** |
| Causally supported functional signal | **T4** | — | ❌ **needs counterfactual playback** |
| Ecologically validated function | **T5** | — | ❌ **needs field replication** |

**The pipeline has exhausted T0–T1** (acoustic association + contextual prediction). T2–T4
(receiver response + causal playback) is the biological next phase — it cannot be done
computationally.

### Cross-analysis 5 — playback-candidate prioritization

| Dataset | Best playback candidate | Why | H0-CMC prediction |
|---|---|---|---|
| **RookID** | Source familiarity (known vs novel rook calls) + sing-motif prototypes | clean individual labels; tests caller-recognition (T2) | response follows caller identity, not motif structure |
| **InfantMarmosetsVox** | onset/body/offset role-template calls (swapped vs natural) | mechanistic HECB role test (T2) | response disrupted by role-swap → roles are functional |
| **Egyptian bat** | emitter-associated motifs vs context-mismatched exemplars | tests social grounding (T2); D=4.3× predicts social matters | response follows emitter, not context |
| **Bengalese finch** | GT motif order-shuffled vs natural | positive control (T2); GT syllable purity 0.79 | response disrupted by order-shuffle → motif structure matters |

**Counterfactual controls** (H0-CMC §4): natural / motif-preserved / order-shuffled /
caller-swapped / context-mismatched / onset-body-offset-swapped / amplitude-duration-matched.
