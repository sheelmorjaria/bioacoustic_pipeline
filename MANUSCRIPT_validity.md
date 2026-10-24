# Validity-Aware Bioacoustic Inference: Task-Specific Prediction and Limits of Recurrent-Unit Discovery

*Draft v0.2 — submission shape. All numerical tables are
artifact-generated (`scripts/build_manuscript_tables.py` →
`outputs/manuscript_tables.md`); figures regenerate from artifacts
(`outputs/cross_corpus_forecasting.png`); the 43-claim evidence ledger
(`outputs/claim_to_artifact.md`, `scripts/build_claim_ledger.py`) is the
comprehensive supplement. Status note: the seventy-file source-reference
panel review is pending and is described as such wherever relevant.*

## Abstract

Bioacoustic pipelines increasingly claim to discover structure in animal
vocalisations — recurring units, "word-like" fragments, context-coded
signals. We report a complete, falsifiable evaluation programme on
Egyptian fruit bat vocalisations (~70k annotated bouts, with marmoset,
chiffchaff and annotated Bengalese finch material as replication and
reference corpora) that deliberately separates four properties analyses
conflate: acoustic predictability, recurring components, contextual
association, and communicative interpretation. A release contract gates
every scientific run on declared validity masks, feature-schema
versions, source manifests and protected splits (six real failure modes
pinned as regression fixtures; a runtime feature-contract gate added
after a caught 44-dimensional violation). Immediate acoustic history
improved prospective 40 ms forecasting in all three tested corpora
(bat +0.044, marmoset +0.029, chiffchaff +0.018); additional 40–200 ms
history improved it in bat and chiffchaff but was not established in
the marmoset extension; static fingerprint acoustics classified
production context above chance for never-seen callers (0.628,
caller-clustered CI [0.543, 0.648]); and one exploratory context
association (envelope-CV rate, C12>C11) survived prespecified
confirmation on independent recording groups (Θ=+0.936, one-sided
p=0.038). In contrast, every recurrent-unit discovery procedure —
whole-call families, variable-length sub-call fragments,
individual-conditioned repertoires, natural-repetition transfer on
annotated song — closed without establishing a transferable inventory,
each with a measured operating limitation (selection-before-threshold
ceiling; control-calibrated specificity conflicts; candidate-ranking
failures). We retain the asymmetry as the contribution: predictive
value and unit discovery are separable; explicit abstention and
preserved correction histories change what a pipeline can responsibly
claim; and a hypothesis ladder (whole-event organisation → reusable
components → component reuse → combinatorial organisation → receiver
evidence) keeps "sentence-like" and "word-like" as structural roles
defined by evidence, never semantic labels. Forecast residuals yield a
conservative acoustic-deviation stream but not a superior boundary
detector — a predictive residual can be informative without being a
good segmentation cue.

**Keywords:** bioacoustics; validity; forecasting; recurrence;
individual recognition; context classification; falsifiability;
preregistration

**Central argument.** A bioacoustic pipeline can produce useful predictive
measurements while failing to establish reusable acoustic units. Reliable
inference requires separate validation of source support, representation,
prediction, recurrence, and human interpretation — with explicit
abstention and preserved correction history.

---

**Figures**

- F1 `outputs/cross_corpus_forecasting.png` — procedure portability of the
  forecasting comparison across bat / marmoset / chiffchaff: immediate and
  older-history effects with uncertainty, eligible-support fraction and
  independent evaluation groups per corpus (artifact-generated).

**Tables** (all artifact-generated via `scripts/build_manuscript_tables.py`;
sign conventions explicit and self-tested)

- T1 task-specific model comparisons (8 rows)
- T2 prespecified acoustic/context effects (6 rows)
- T3 discovery and validation outcomes (8 rows)
- Supplement: the 43-claim evidence ledger with cohort, representation,
  masks, model, aggregation, uncertainty and correction-history fields,
  plus the falsification-status footer distinguishing prespecified,
  exploratory-to-confirmed, retained-scoped, closed-under-tested-procedure,
  descriptive, rejected and quarantined records.

**Reproducibility.** `python scripts/build_manuscript_tables.py --self-test`
validates sign conventions; `python scripts/build_claim_ledger.py`
regenerates the supplement; `pytest -q` (578 passing) covers the release
contract, feature-schema and derivative contracts, silence-aware
segmentation, and every study's machinery.

---

## 1. Introduction

Acoustic similarity, predictability, recurring components and communicative
interpretation are four different properties that bioacoustic analyses
frequently conflate. A vocalisation can be highly predictable without
containing reusable units; a recurring component can exist without being
used by receivers; a receiver response can exist without validating any
particular computational unit inventory. This paper reports a complete,
falsifiable evaluation programme on Egyptian fruit bat vocalisations
(*Rousettus aegyptiacus*, ~70k annotated bouts; supplemented by four
bird-ID corpora and annotated Bengalese finch song as external reference
material) that deliberately separates these questions — and reports both
its retained findings and its diagnosed failures.

Three design commitments organise everything that follows. First, a
**release contract**: every scientific run must declare its validity
masks, feature-schema version, source manifest and evaluation split, and
refuses to proceed otherwise (six real failure modes are pinned as
regression fixtures: padding-as-observation, partial support, mask
substitution, feature-layout drift, duplicate sources, evaluation
leakage). Second, **falsification-first analysis**: every positive
structure claim ships with its matching null or control, and prespecified
abstention is a valid outcome. Third, a **claim-to-artifact ledger**: all
28 results trace to machine-generated artifacts with explicit sign
conventions, uncertainty scopes and correction histories; quarantined
predecessors are preserved, never silently replaced.

The programme's honest summary: the pipeline yields reliable task-specific
prediction (fingerprint identification across species, pooled temporal
modelling, prospective acoustic forecasting, one independently confirmed
context-associated acoustic effect) while every procedure attempted for
recurring-unit discovery — whole-call families, variable-length
sub-call fragments, individual repertoires, natural-repetition transfer —
closed without establishing a transferable unit inventory, each with a
documented operating limitation rather than a complete causal account. We
argue this asymmetry is itself the methodological contribution:
predictive value and unit discovery are separable, and conflating them
produces claims neither the statistics nor the sources support.

## 2. Methods: measurement and provenance

**Sources and identities.** The bat corpus provides per-bout annotations
(emitter, addressee, production context). Two identity levels are kept
distinct throughout: the file/recording-group level determines partitioning
(70/30 discovery-validation by block hash; later studies 40/20/40
discovery/calibration/validation); the bout level identifies search hosts.
Cross-species material (chiffchaff, pipit, little owl — colour-ring or
song-structure IDs with documented provenance caveats; marmoset
InfantMarmosetsVox; Bengalese finch with Koumura-Okanoja syllable
annotations) enters only as external reference or portability tests.

**Validity as four distinct concepts.** Source validity, detected
activity, feature support and task eligibility are never interchanged.
The acoustic fingerprint's absolute energy gate (rms < 10⁻³ zeroes the
descriptor core) and peak-relative activity checks *disagree* on
low-gain recordings — an integration failure that produced distance-zero
"recurrence" matches on windows containing one valid frame plus
thirty-nine gated frames. Those candidates were re-adjudicated
(preservation of historical distances; rejection as evidence) and future
searches require `source_valid ∧ activity ∧ representation_support` at
the window level before nearest-neighbour selection.

**Representation.** The 44-D acoustic fingerprint (physics +
macrotexture + microtexture; species-portable, firewall-cleared) is the
default for identification and retrieval; a 12-feature supported subset
with declared physical units and gap-safe derivatives serves trajectory
and dynamics analyses. Feature-layout versioning is enforced
(schema v2; a v1 name-index drift — reading macro's `mod_depth` as the
envelope CV — was caught by its own downstream study and pinned by a
regression test against the live extractor). Shared encoders (CPC
variants) enter only under ethological-firewall provenance; the
caller-supervised multi-head checkpoint is a restricted instrument
(probe/retrieval only), never evidence for discovering caller structure.

**Experimental comparisons.** All studies prespecify endpoints, splits,
and decision rules before evaluation; exploratory results are labelled as
such, and confirmations run on independent recording groups. Controls are
matched for search opportunity (equal gallery sizes, duration-matched
random-fragment references), and thresholds are frozen before validation.
Where calibration was the study (§Results 3), the calibration sample
identities, quotas and seed are frozen in the artifact.

## 3. Results 1: task-specific predictive value

Table 1 (artifact-generated). Four findings summarise:

1. **Fingerprint task-defaults.** Across four species, the whole-call
   fingerprint identifies callers at 0.23–0.83 balanced accuracy against
   chance 0.03–0.10, dominating species CPC (0.08–0.36) with fusion
   adding little beyond the better arm.
2. **Pooling beats personalisation** for discrete temporal prediction:
   caller-specific transition models did not beat the pooled PST on
   validity-aware reanalysis, and pooled temporal memory beyond order-1
   is marginal (A₁=0.926 vs A_longer=0.070 per-token NLL).
3. **Recent history improves prospective forecasting**: state+derivative
   input reduces aggregate nRMSE by 0.044 [CI −0.051, −0.039] versus
   state alone at the 40 ms prospective horizon (1,116 held-out files,
   33,729 prediction instances, paired file-level bootstrap) — and the
   derivative parameterisation is informationally equivalent to
   state+previous-state (+0.0007), establishing the gain as *history*
   value, not differentiation. A **procedure replication on
   InfantMarmosetsVox** (scaler and Ridge refit on marmoset training
   material under the identical fitting policy; recording-group date-block
   splits; 203 calls, 28 recording groups, 2,196 held-out prediction
   instances, 10 callers) reproduces the full structure: Δ_history =
   +0.029 [CI +0.007, +0.042 on the improvement scale; 98.9% of
   replicates], state+previous-state equivalent (+0.0002), derivative
   input alone worse (0.525 vs 0.369). The gain differs in magnitude from
   the bat corpus without supporting an intrinsic species difference
   (recording conditions, call composition and feature support also
   differ); the replication tests generality of the predictive result,
   not the intra-call unit hypothesis.
4. **Vocal-target continuation** improved its prespecified
   similarity-margin endpoint by +0.047 [0.039, 0.056] on unseen callers
   under matched budgets. The completed marmoset multi-head evaluation
   (single prespecified run) shows a solid population history gain
   (G_pop=+0.171 [0.149, 0.205]) but **no caller-head advantage**: the
   interaction T_y averages −0.039 (5/8 callers negative) and own-head
   benefit D_y is heterogeneous (−0.18..+0.37); two held-out callers
   remain unavailable rather than zero-filled. Scores are contrastive
   continuation scores, not perplexity; the result is conditional
   prediction evidence with the correct label supplying the head.

## 4. Results 2: context-associated dynamics

The prespecified primary endpoint — median signed centroid rate,
C12−C11 within 67 supported caller-recipient pairs (21 callers) — is
**null** (Θ=+1,310 Hz/s, CI [−6,396, +11,795], p=0.84). The secondary
family (11 features × 2 summaries, multiplicity-flagged) produced one
nominal survivor: dyn_cv |rate| (p=0.0015 exploratory). Its
**prespecified one-sided confirmation on 11,013 bouts drawn from
cache-external recording groups** (91 supported caller-recipient pairs,
21 callers; support manifest frozen before scoring; bouts and recording
groups are distinct units throughout) confirmed the direction — Θ=+0.936 [0.463, 1.874], one-sided
p=0.0375 — with the caveats that the effect halved relative to
exploration and callers overlap the exploratory set (recording groups did
not). This is the programme's single exploratory-to-confirmation result,
and its study history differs by design from the prespecified findings
above. Interpretation scope: context-associated envelope-dynamics
differences within supported caller-recipient material — not motor
physiology, not meaning.

**Context prediction and population-dependent fusion.** In the
held-out-caller evaluation, the static acoustic classifier achieved
balanced accuracy of 0.628, with reported caller-clustered confidence
interval [0.543, 0.648], compared with a 0.500 baseline. Performance
varied across the six evaluated callers. Adding the prespecified dynamics
features did not establish additional benefit: the paired BA difference
was −0.0009 [−0.0161, 0.0131]. For familiar callers evaluated on new
recordings, adding static acoustics and dynamics to the social-identity
baseline improved BA by 0.0364 [0.0256, 0.0469]. This fusion advantage
did not transfer to held-out callers: the combined predictor achieved BA
0.498 and predicted C11 for 97.5% of bouts under the declared
unknown-social-category handling. The results establish
context-associated predictive value in the evaluated acoustic
representation and population-dependent performance of the fitted fusion
model. They do not identify receiver-used cues, communicative meanings
or universally context-discriminative acoustic features.

## 5. Results 3: discovery does not guarantee transfer

Table 3. Five studies, five documented closures:

- **Whole-call families** (frozen exemplar-radius): 1,160 discovery
  families form; transfer to independent recording blocks is 0.001–0.004.
  A separate equal-gallery comparison retains a caller-associated
  matching advantage (0.207 vs 0.095, caller-clustered CI [0.097,
  0.167]) as a scoped result — not confirmed caller-specificity.
- **Variable-length sub-call discovery** (control-calibrated radius):
  discovery families recover 1.25–4.2% on held-out recordings against a
  duration-matched random-fragment maximum of 14.6%.
- **Individual-conditioned repertoire** (caller 215; 4,190 events, 376
  recording blocks; three frozen partitions): three of sixty-four
  whole-call families slightly exceed their reference (0.07 pp, no
  uncertainty attached); the dominant recurrent fragment (74 accepted
  matches across 50 discovery blocks) recovers 6.3% held-out against a
  same-caller random-fragment reference of 6.2% mean / 32.7% max.
- **Planted-fragment sensitivity**: supplied-query matching and
  localisation are validated *for the tested exact feature-trajectory
  insertions* (92.4% recovery, one-frame median boundary error, 3.0%
  decoy absorption) — scope excludes the waveform path, natural
  variation and discovery.
- **Natural repetition on annotated Bengalese song** (procedure
  portability; per-bird re-calibration): zero accepted matches across
  eighteen natural class-queries; blind discovery families gap material.
  The annotation-informed acceptance calibration then **abstained for
  all three birds** under prespecified constraints (background
  acceptance ≤ 0.10, class-balanced recall ≥ 0.25): recall never exceeds
  ~0.10 anywhere on the threshold grid because the single-best-interval
  search selects non-class intervals *before* thresholding —
  R̄(τ) ≤ Σ w_q c_q is a threshold-independent ceiling.

**Candidate availability and ranking intervention.** We retained up to
ten distinct local alignment candidates per query–host pair, including
the original raw-cost optimum. Oracle annotated-interval coverage
increased from raw top-1 rates of 0.319, 0.042 and 0.167 to 0.868,
0.729 and 0.875 for Birds 0–2, respectively. This demonstrates
substantial selection headroom within the evaluated candidate sets,
while leaving candidate-generation misses and operational discrimination
unresolved. A prespecified background-relative ranking score, evaluated
on the same candidate sets with a frozen bank of development background
trajectories, did not improve top-1 selection. Both the baseline and
intervention abstained under the requirements of background acceptance
no greater than 0.10 and class-balanced recovery of at least 0.25. The
result rejects this ranking intervention under the tested conditions; it
does not establish a transferable unit inventory or identify a
successful replacement matcher.

These closures are **under the tested procedures**, with demonstrated
failures (the selection-before-threshold ceiling; the
specificity-vs-natural-variation calibration conflict; the
random-fragment calibration's draw sensitivity) distinguished from
unresolved causes (relative contributions of representation, recording
distribution and candidate selection to failed transfer). No claim of
absent units, failed matchers generally, or biological absence follows.

## 6. Results 4: audit and abstention

The programme's correction record is part of its results: 72.89%
recorded-silence contamination quarantined and re-analysed; the
one-valid-frame recurrence failure diagnosed to an absolute-vs-relative
gate mismatch; feature-layout drift caught by its own downstream study;
the blinded 40-pair source review — completed, locked, then audited —
found its own package limitations (shared exemplar anchor, overlapping
hosts, two cross-condition clips, waveform-axis scaling), and is
reported as a **completed descriptive visual review with post-lock
dependence and rendering limitations; no observed candidate-control
differentiation on the recorded binary endpoint** (not equivalence). An
observed zero difference is not an equivalence claim, and clip-group
counts are provenance counts, not effective sample sizes. Abstention
outcomes — the calibration study above, and the scaling study's
unpassable-by-construction positive control — are reported as results.

## 7. Discussion

**Corrected synthesis (review-adopted).** Graded vocalisations can be
analysed as continuous acoustic dynamics without assuming a discrete
biological vocabulary. In the evaluated corpora, recent acoustic history
improved prediction, while the tested caller-personalisation procedures
did not establish a useful incremental advantage over their shared
comparators. These findings concern fitted predictors and do not
establish identical individual dynamics or the absence of caller-specific
organisation. Separate analyses demonstrated context-associated acoustic
variation and context-classification performance, but neither result
established reusable components or communicative meaning. Continuous
representations avoid a particular quantisation choice; they do not
remove requirements for valid source support, natural-occurrence
evaluation, appropriate controls and protected generalisation tests. A
future analysis could ask whether a prespecified contextual variable
changes the history-to-future mapping beyond acoustic-level differences.
Any functional or compositional interpretation would require
independently grounded receiver evidence. The defensible frontier is
not "shared graded grammar has been established" but "predictive
acoustic dynamics have been established, and their contextual
organisation can be tested separately from component reuse and receiver
interpretation."

Three restraint clauses bind the results below. First, the pooled-PST
and multi-head outcomes are model comparisons under the specified
representation, training and evaluation procedures — the tested
personalisation not helping is not "the grammar is shared", and a small
T_y is not proof that caller identity has no relationship with temporal
organisation; conversely, "discrete-token analysis cannot see the graded
grammar" is a sensitivity hypothesis, not a finding — neither absence
nor hidden grammar is inferred from the same nondetection. D_y is a
difference in predictive performance between heads, not a measured
manifold displacement, and licenses no caller-mean-subtraction
disentanglement claim. Second, scale has three coordinates — feature
support, history extent, forecast horizon: a 10 ms hop is not a 10 ms
observation window is not a 10 ms biological memory, so the appropriate
statement is "no tested caller-personalisation benefit at this
representation and evaluation scale". Third, CPC contrasts (density
ratios under a candidate distribution), PST held-out-loss improvements
and nRMSE reductions are distinct quantities — their differences are not
automatically mutual-information amounts or information decompositions,
and the interpretable held-out predictive endpoints are not replaced by
information estimators for the sake of sounding fundamental.

What is retained: task-default representations with quantified scoping;
pooled temporal modelling; a small, real prospective-history forecasting
gain; one independently confirmed context-associated acoustic effect
(dyn_cv), the strongest current candidate for receiver testing. What
failed and why it matters: every discovery procedure closed without a
transferable unit inventory, and the failures are *informative* — they
locate limits (selection before thresholding; calibration interacting
with near-identical background; enrichment criteria that random
fragments from a structured repertoire can satisfy) that more model
capacity does not address. What remains unvalidated: any claim of
reusable intra-call units, receiver-perceived categories, or meaning —
all reserved for biological evidence. The immediate receiver question is
precise: whether receivers detect or use the documented variation in
relative envelope dynamics (dyn_cv) under an appropriate social
situation; the computational side of that hand-off is specified
(`RECEIVER_HANDOFF.md`). Measurement validation continues in parallel:
the seventy-file source-reference panel (score-blind representative +
challenge samples with 0.78–0.92 mask disagreement) remains the pending
independent assessment of activity-mask and feature-support coverage;
until completed, mask assessment is relative to its existing proposal
reference.

## 7c. Predictive profile: horizon decay and order-specificity

Two prespecified measurements complete the programme's characterisation
of the acoustic process as a predictive object (validity-repaired:
alpha selected on train-side validation only; decay evaluated on
COMMON source positions eligible at all four horizons, so the curve
reflects horizon rather than changing populations). **Horizon decay**:
history-dependent predictive gain decreases monotonically over the
tested 40–320 ms horizons — +0.0524 [+0.0456, +0.0585] at 40 ms,
+0.0382 at 80 ms, +0.0278 at 160 ms, +0.0147 [+0.0093, +0.0201] at
320 ms (15,085 identical evaluation positions per horizon; all CIs
excluding zero) — while remaining positive at 320 ms: a decaying
predictive timescale over this range, with no upper boundary
identified and no biological memory constant claimed.
**Order/adjacency dependence**: replacing the true history states with
random same-trajectory states (marginal occupancy preserved, temporal
adjacency destroyed) raises loss from 0.788 to 1.644 — **Δ_order =
+0.8566 [0.7534, 0.9582]**: prospective acoustic predictability
depends on temporal arrangement as well as state occupancy. (Local
continuity is itself one form of temporal order and may be a major
reason the true history wins; what is established is adjacency
dependence, not a specific sequential rule.) Together: short-timescale
acoustic prediction depends on temporal arrangement as well as state
occupancy, without requiring a discrete-unit inventory — ordered
temporal structure demonstrated directly in continuous acoustic space.
An earlier developmental run with evaluation-side alpha selection is
superseded by this repaired analysis.

## 7d. Continuous-state decomposition: occupancy, drift, diffusion

Treating the acoustic trajectory as a continuous-state (analog-valued)
object completes the context picture in the language the review
proposed — an operational stochastic-dynamical description (dx = v_C(x)
dt + B_C(x) dW as a model of trajectories, not a claim about the
production system). On common acoustic support (18,083 evaluation
positions; k-NN displacement fields with split-half noise
references): **diffusion is exactly equal** (local dispersion ratio
C12/C11 = 1.001 [0.967, 1.039]); **trajectory geometry is
indistinguishable** (path length 135.6 vs 138.6 per second; turning
angle 1.319 vs 1.320 rad; dwell 0.242 vs 0.243); and **drift differs
by 1.27× [1.22, 1.32] the split-half sampling noise of a single
context's own field** — a small, CI-consistent excess whose
magnitude sits far below the occupancy difference that drives context
classification. Three operationalisations now converge — cross-context
transfer costs (0.4–1.3% nRMSE), aggregate forecast surprise (no
detected difference), and displacement fields (1.27× sampling noise)
— on one conclusion: **C11/C12 differ mainly in where trajectories
occur, with substantially shared local dynamics.** This is the
analog-valued counterpart of everything the token-discrete analyses
could not establish, obtained without inventing a vocabulary.

## 7e. Feature mechanics and cross-caller model transfer

Completing the predictive roadmap: **feature ablation** (prespecified
groups removed, never zeroed) shows all three acoustic dimensions
carry temporal predictive signal — spectral-shape +0.0260 [+0.0142,
+0.0425] across six features, amplitude +0.0265 [+0.0225, +0.0314]
across three, temporal-change +0.0116 [+0.0099, +0.0134] across three
— with amplitude the densest carrier per feature (≈0.0088 vs
0.0043/0.0039), consistent with the per-feature forecasting gains.
The temporal signal is distributed, not concentrated in one modality.
**Cross-caller model transfer** (coefficients and scaler fully frozen;
five callers held out): loss rises only from 0.787 to 0.801 —
**Δ_transfer = +0.0141 (~1.8% relative)** — the local predictive
relationship is largely caller-general even without refitting,
doubly consistent with the marmoset no-caller-head-advantage result.
The roadmap's future-behaviour endpoint remains untested and honestly
so: the corpus carries per-bout context labels but no independently
timestamped behavioural streams; a next-bout-context proxy is
constructible but is context-sequence prediction, not behaviour, and
is reserved.

## 7f. Architectural consequence: a neural predictive-state model does not beat Ridge

The seven predictive findings motivated a validity-aware multi-timescale
model (VPSE: 64-D encoder, 128-D GRU, per-horizon Gaussian heads;
context as a read-only global head; no caller or context conditioning
of the dynamics core). Run against the frozen ladder on identical
protected positions, the v1 model (multi-horizon Gaussian NLL core;
auxiliary objectives constructed but not yet active; 8 epochs, one
seed) was **worse than Ridge older-history at every horizon by
0.43–0.51 nRMSE** and did not beat persistence at 160/320 ms. By the
prespecified criterion the linear forecaster is retained, and the null
strengthens the programme's headline: the temporal order, decay and
transfer structure measured across six corpora is captured by a shared
linear model. The follow-up burden is explicit — a fully-instrumented
VPSE (auxiliary losses active, multi-seed, declared budget) must beat
0.880/0.955/0.972/0.976 on these blocks before the architecture earns
its complexity.

## 7g. Cross-corpus model transfer: procedure ports everywhere; frozen maps are regime-dependent

The procedure/model distinction is now measured directly (prespecified
four-pair panel; frozen scaler and coefficients; identical target
positions; diagnostic only, never a production default). The
bat-trained map **beats each target's own refit Ridge on chiffchaff
(0.679 vs 0.770) and little owl (0.455 vs 0.647)** — the expected
large-source sample-size effect on a shared coordinate geometry — but
**fails at persistence level on pipit** (0.892 vs refit 0.662), and
chiffchaff→pipit fails hardest (1.318), with pipit's target refit
requiring α=1000: the frozen-scaler failure pattern of coordinate
distribution shift, not species-specific dynamics. Summary:
**procedure portability holds in all five independently partitioned
corpora; model portability is regime-dependent**, bounded by the
physical coordinates' corpus-specific distributions.

## 7h. Human reference battery: the same diagnostics in a system known to have discrete units

A positive-control regime closes the interpretation question the animal
results raise: what do these analyses look like in a corpus where
discrete words and sentence composition independently exist? The full
battery (forecasting ladder, order control, recurrence recovery, masked
reconstruction) was run unchanged on LibriSpeech dev-clean (400
utterances, 6 speakers; speaker-blocked 70/30; the same 12-feature
contract with the human species tag). Provenance is stated prominently:
LibriSpeech ships utterance transcripts but no word timestamps, so the
recurrence study is utterance-level (same- vs different-speaker) and
the reconstruction strata are energy-onset proxies — declared acoustic
proxies, not lexical ground truth. All comparisons are structural;
raw nRMSE magnitudes are not comparable across corpora.

Known speech shows **the same structural predictive profile as the
animal corpora**: history gain at every horizon (Δ_recent/L_pers 6.7% →
17.7% across 40→320 ms), older-history gain decaying even more steeply
than bat's (13.2% → 0.5% relative), and a large order penalty
(L_shuffled/L_true = 7.4×, vs ≈2.1× in bat). Conversely, where the
reference says units exist, the label-blind machinery does not recover
them: the matcher's top candidates land in voiced regions of
*different* speakers' utterances (0/20 same-speaker at the utterance
level), and the whole-event summary G fails to aid masked
reconstruction in speech as well (Δ_G = −0.034; bidirectional local
context does beat the blend, more so where an energy onset sits inside
the mask — local constraint strengthening where smoothness breaks).

This calibrates both directions of the thesis. **Continuous predictive
organisation does not imply discrete units** — the same predictive
diagnostics are measurable in a system that demonstrably contains
them. **And unit existence cannot be established or refuted by these
continuous diagnostics at all** — it requires unit-specific reference
evidence, and the G-style summaries that fail on animal corpora fail
here too, where composition is known: the failure is in the summary,
not evidence about compositional structure in either system.

## 7i. Natural-unit reference benchmark: discovery sensitivity measured directly against annotated units

The Human Reference Battery (§7h) exposed a gap: unit-existence questions
need unit-specific reference evidence. The Bengalese corpus provides it —
manual note boundaries and class labels for three birds — so the frozen
label-blind discovery procedure was scored, once and preregistered, on
four properties against that reference (labels hidden during discovery;
same bout strata; frozen radius, seeding and family construction). **All
four discovery-side scores fail in all three birds**: known-query
boundary localisation 0.12–0.23 (validation-bout clustered CIs well
below the 0.7 rule; an any-overlap readout rules out boundary-sprawl as
the cause), identity matching 0.12–0.23 rank / near-zero radius
acceptance, family completeness 0.0 with members landing 98.7–100% in
unannotated gaps, and contextual reuse uninterpretable at these recovery
rates. Meanwhile the reference side establishes the target structure
exists: annotated elements recur with on average 2.6/2.3 distinct
successor classes in Birds 0/1 (the H3 premise), while Bird2's
near-deterministic repertoire (1.5) makes contextual reuse untestable
there. Per the preregistered clause, this is **a limitation of the
tested procedure — localisation/path scoring first, representation/metric
discrimination second, acceptance calibration third — not absence of
biological units**.

The benchmark also produced a correction: its reconciliation could not
reproduce the earlier development-strata localisation diagnostic
(0.83–0.97) and exposed a cross-bout coordinate-frame bug in that
diagnostic's host construction. Corrected values (0.53/0.25/0.50) agree
with the benchmark; original values and chronology are preserved in the
artifact. One prior interpretation ("acceptance calibration" as the
 Birds 1/2 limitation) is superseded by "localisation/path scoring".

## 7j. Word-recovery on aligned speech: the matcher is not a natural-unit detector

With word timestamps available (public MFA alignments, Zenodo 2619474,
CC-BY-4.0; machine alignments with automated QA and a deterministic
human-validation sample pending), the unit-recovery question was asked
where units are beyond doubt. The frozen label-blind matcher — the same
fingerprint representation and top-K DTW candidate mechanism deployed on
the animal corpora — was set to retrieve 30 frequent/mid-frequency word
types (5 median-duration query tokens each) across 566 held-out
utterances from unseen speakers, scored only after the search, with
duration- and frequency-matched different-word controls. **Same-word
recovery is absent: cross-speaker Recall@1 = 0.020 [0.00, 0.047], below
the matched control (0.089), not rescued by same-speaker search (0.067);
Recall@10 (0.193) does not exceed the control either.** When a correct
top-1 does occur, boundaries are window-scale accurate (IoU 0.73, onset
10 ms) — the deficit is candidate selection/ranking, the mirror image of
the Bengalese localisation-first decomposition. This calibrates the
central inference: the animal unit-discovery nulls say at least as much
about the tested procedure as about the animal corpora, because the
procedure fails in the regime where units indisputably exist. The
preregistered follow-up is the expert-annotated zebra-finch benchmark
with a fixed decision tree (good recovery → Bengalese failure is
corpus/procedure-interaction specific; poor recovery again → the
fingerprint+DTW method is not validated as a natural-unit detector across
two independently annotated bird corpora; no threshold or representation
changes during the benchmark); perceptual-similarity data answer a
different question (acoustic distance vs bird-perceived similarity), and
motif/chunk sequencing tests are interpretable only after lower-level
recovery works.

## 7k. Boundary decomposition: the matcher is near-ceiling; localisation is the failure

The word-recovery null (§7j) motivated the architecture test the review
prescribed: separate "where might a unit boundary be?" (NBD) from "are
these candidate units the same kind?" (representation), instead of asking
free-start DTW to do both. On the same annotated Bengalese bouts, two
stages were preregistered. **Stage B (oracle-boundary ceiling): given
TRUE annotated syllable spans, the frozen fingerprint with fixed-endpoint
DTW retrieves same-class syllables at Recall@1 = 0.953–0.994 (chance
0.22–0.28; segment-clustered CIs; cross-bout only)** — the
representation/matcher is not the bottleneck. **Stage A (raw
fixed-threshold NBD): proposals land near 0.66–0.74 of true boundaries at
30 ms tolerance (median error 20 ms) with precision 0.10–0.20 — 76–91%
of proposals fall in inter-syllable gaps**, so the rule "NBD closes the
localisation gap" (F1 ≥ 0.6, R_proposal ≥ 0.8) does not fire. The
three-ceilings reading: R_oracle − R_proposal ≈ 0.25–0.33 is
segmentation/proposal loss, and the bulk of the oracle-to-historical gap
is selection loss — precisely where a typed boundary filter
(onset/offset/internal/noise roles, trained within the target corpus) or
a non-circular animal calibrator must earn precision. This reframes the
programme's nulls: "fingerprint+DTW is not a natural-unit detector"
becomes "the fingerprint matcher is near-ceiling; the free-start search
was the failure" — a reinterpretation hypothesis for the speech null as
well (its correct top-1s had IoU 0.73, onset 10 ms; §7j's speech result
is consistent with, but does not yet independently confirm, this
interpretation, because the word-timestamp human validation is still
pending). **The main architectural lesson, stated as a correction of the
earlier narrative: the 44-D fingerprint is not the reason natural-unit
recovery failed in Bengalese song — under true boundaries, same-class
retrieval is near ceiling; the dominant losses arise from blind boundary
proposal and proposal selection.** The D2 correction (§7i) is part of
this revision: it removes the earlier, unsupported belief that
localisation was already largely solved; the corrected story —
representation capable, blind localisation and selection not yet capable
enough — is stronger methodologically because it yields a concrete,
falsifiable next step rather than another architecture search. The
revised architecture — parent → NBD proposals → typed selection →
candidate units → fingerprint (→ CPC where allowed) → retrieval — now
has every link measured except selection, which is the subject of the
proposal–selection study that follows. An NBD proposal is an
acoustic-change candidate, not a syllable, motor primitive, or word-like
unit (Semantic Ceiling intact).

## 7l. Proposal–selection decomposition: high-recall proposals solve the ceiling; typing evidence is the open problem; end-to-end recovery jumps to 0.52–0.67

The intervention implied by §7k's diagnosis was preregistered and run on
the frozen split. **The single prospective high-recall NBD regime
(lower floors, fire-on-drop enabled — syllable offsets are drop-type —
shorter debounce) raises proposal recall to 0.918–0.929** (from
0.66–0.74), satisfying the ≥0.9 ceiling that §7k identified as binding.
**The typed 4-way selector does not earn precision on the 13 proposal
features** (kept precision 0.10–0.19 ≈ the raw non-gap fraction; typed ≈
scalar capacity-matched baseline, ΔF1 ≤ 0.01; development 4-way accuracy
0.23–0.37): distinguishing unit-edge acoustic change from internal/gap
change is the measured open problem. **Nevertheless, end-to-end unit
recovery — selected segments → usable (IoU ≥ 0.5) → fingerprint
retrieval — reaches coverage-weighted 0.521–0.665** (coverage 0.61–0.73
× retrieval 0.857–0.916, near the oracle 0.953–0.994), versus ~0.02–0.23
for the historical free-start procedure: a 4–30× improvement from
moving localisation upstream, before any new selector capability. The
formal "pipeline works" rule (kept F1 ≥ 0.6) does not fire, and the
decomposition shows exactly why: retrieval is near-oracle, proposal
recall is solved, and the residual loss is segment-IoU coverage —
boundary-typing evidence. That motivates the LibriSpeech word-edge
typing benchmark (known word→phoneme hierarchy; labels {word-onset,
word-offset, internal-phoneme, gap}; CPC context arm permitted on
humans under the firewall), whose architecture — not weights,
thresholds, or duration priors — is the transferable quantity.

## 7m. Word-edge typing on speech: the selection problem is now located, not solved

The architecture-calibration benchmark the review prescribed — LibriSpeech
as the corpus where the unit hierarchy is known, training on fit speakers
and scoring prospectively on eval speakers, with nothing but
architecture/metrics/discipline eligible to transfer — yields four
calibration facts. **First, the composition: 66% of raw high-recall NBD
proposals on speech fall INSIDE words (internal-phoneme), 26% on word
edges, 8% in gaps** — "real acoustic change is not unit boundary" is now
a measured quantity, and it dominates. **Second, the proposal ceiling is
regime-dependent: recall against word edges is 0.713 on speech under the
Bengalese-frozen high-recall regime (0.92 on Bengalese), below the 0.9
floor.** Third, the scalar typed selector raises edge purity by +0.074
(0.256 → 0.330) but its word-boundary F1 (0.359) does not beat the
keep-everything reference (0.377) — the 13 NBD scalars do not separate
word edges from internal-phoneme changes in either system. Fourth, the
fixed [0.10, 2.0] s duration-prior decoder HURTS on speech (F1 0.203;
short function words abstained); the CPC-context arm — the one place
learned local evidence is firewall-permitted on humans — could not be
run for lack of a local human CPC checkpoint and is recorded as the
open arm. By the review's own decision framing, this mixed-to-negative
result bounds how strongly the animal nulls can be interpreted until
the typing evidence matures, and it locates exactly what is missing: a
speech-appropriate proposal regime, boundary-typing evidence beyond the
scalars, and corpus-calibrated rather than fixed duration priors.

**Locked interpretations (review, 2026-10-20).** The unit-recovery
section now reads as a three-stage decomposition: proposal generation is
repairable but corpus-specific (the Bengalese high-recall regime must
not be assumed to transfer); the weak link is the 13-dimensional
proposal feature set, which fails to type edges in both systems; the
decoder result shows that a fixed prior mismatched to natural
duration variation can hurt — not that decoders are bad; and the
CPC-context arm remains an open architectural test, not a failed one.
Summary form: **the discovery pipeline's main limitation is now
localized to segmentation evidence rather than unit representation. In
Bengalese song, oracle-boundary retrieval is near ceiling and a
high-recall proposal regime recovers most annotated boundaries, but
current scalar/typed proposal features do not reliably distinguish true
unit edges from internal or gap-related acoustic changes. LibriSpeech
shows the same general phenomenon: most acoustic-change proposals
correspond to within-word phonetic structure rather than lexical edges.
The unresolved problem is not whether the representation can compare
natural units, but whether the boundary layer can assign the correct
structural role to acoustic changes.** If typing is revisited, exactly
one richer feature family is tested at a time (local trajectory context;
proposal-centered fingerprint differences; human CPC context for
LibriSpeech; Bengalese-specific learned context trained only on
Bengalese development data) with the proposal set frozen — measuring
typing evidence, not proposal changes.

## 7n. Zebra-finch external benchmark: the pattern replicates; the branch closes

The frozen-order portability test (Lee et al. 2026 data: three juvenile
males; expert cluster reference `az` prespecified; recording-level
splits; zero retuning) answers the refined question in two of three
birds: **oracle-boundary same-class retrieval 0.707/0.809 against chance
~0.25, high-recall NBD proposal recall 0.910/0.913 — the regime
transfers within the animal domain, corpus-specific only against speech
— and boundary typing again weak (kept F1 ≈ 0.49, purity earned but
below the 0.6 rule).** The third bird fails at the oracle level
(Recall@1 0.291 ≈ chance, Recall@10 0.671), and the prespecified
reference-quality QA locates the likely cause there: **inter-expert
label agreement on these juvenile vocalizations is itself low (az↔kl
0.21–0.24; az↔mb 0.003–0.14)** — the "expert class" reference is weakly
consistent, bounding what any acoustic clustering can score against one
expert's labels. The preregistered decision tree therefore closes
neither cleanly: not poor-again-across-the-board, and the one failure
sits at the reference boundary. Per the frozen order, no threshold or
representation changes were made during the benchmark, and the
unit-recovery branch stops here with a four-point final form: the
matcher is near-ceiling under true boundaries; high-recall proposals
repair animal-domain recall; boundary typing is the cross-system weak
link; and reference consistency is itself a measurable bottleneck in
animal unit benchmarks.

## 7o. The representation question resolved at both ends: within-bird syllable identity yes, cross-speaker word identity no; contrast parameterisation is not the bottleneck

Two preregistered studies complete the boundary-representation arc.
**First, the speech-side oracle comparison** (true MFA word spans,
fixed-endpoint DTW, same types/tokens/controls as the free-start
benchmark): cross-speaker same-word Recall@1 = 0.193 [0.133, 0.254],
statistically indistinguishable from the duration- and
frequency-matched different-word control (0.185 [0.092, 0.277]) though
far above prevalence (0.011). The fingerprint's retrieval excellence is
therefore scope-bound: within-individual avian syllables (0.95–0.99;
zebra finch 0.71–0.81) yes, cross-speaker word identity no — and the
speech free-start null is accordingly not purely a localisation failure;
representation/metric insufficiency contributes for cross-speaker word
identity. **Second, the M0–M3 ablation on Bengalese** (frozen proposals;
M0/M1 freeze-asserted against the completed TCN study): explicit
multiscale signed contrasts (20/50/100 ms; signed, normalised,
dispersion; validity-gated with eligibility flags) add ~nothing beyond
the TCN's implicit learning (ΔF1 CIs top out at +0.025), and their
combination with the TCN adds nothing beyond either. The +0.05 rule
fires for no arm; per-role accounting locates the untouched loss in
gap/noise rejection (kept-gap counts 90–123 per bird under every arm).
A prespecified-as-post-hoc Bird2 diagnostic explains the earlier
F1-versus-recovery divergence: the TCN's +0.163 end-to-end gain is a
paired-edge-coverage effect (0.551 → 0.796; segment IoU≥0.5 0.637 →
0.805) driven by onset/offset recall while precision stays ~0.14 —
boundary F1 is not the most informative primary endpoint for the
unit-recovery objective, a prespecification lesson for any successor
study. The open problem is now precisely precision-oriented gap/noise
rejection at candidate boundaries; none of the four tested
representation families moves it.

## 7p. Consolidation: the gap/noise limitation is in the measurements; metric–objective alignment becomes a result

Three closing analyses freeze the boundary-representation arc.
**Statistical qualification of the speech oracle** (post hoc addendum on
the frozen per-query checkpoints): the two-sample query-cluster
bootstrap of the arm difference gives ΔRecall@1 = +0.009 [−0.106,
+0.119] — the interval spans zero and bounds any same-word advantage
below practical use. Per the review's wording, the study establishes
that the tested representation and matcher **did not demonstrate
cross-speaker word-specific discrimination beyond the matched control**
— not that the representation contains no word information; the speech
limitation is accordingly two separable problems (oracle-boundary
discrimination and blind selection) whose relative contributions are
not identified. The M0–M3 failure is likewise a failure of the
prespecified criterion, not proof that every contrast-based
representation is ineffective. **Descriptive analysis of retained gap
proposals** (frozen M0 selector reproduced exactly): gap and true-edge
score distributions nearly coincide — gap medians sit 0.02–0.04 below
edge medians, 37–45% of gap proposals score above the edge median, the
top predicted decile shows an observed edge rate of only ~0.23, and no
threshold separates them (rejecting 90% of gaps would cost ~88% of edge
recall). **The retained gap proposals are acoustically indistinguishable
from true unit boundaries under the current 13-D measurements — the
limitation is in the measurements, not the decision rule** (descriptive;
no threshold adopted). Finally, the Bird 2 divergence between boundary
F1 (+0.024) and both paired-edge coverage (0.551 → 0.796) and end-to-end
recovery (+0.163) is elevated to a central methodological result —
explicitly post hoc: a correctly measured performance metric can be
poorly aligned with the scientific objective, motivating a unit-recovery
primary endpoint (R_unit: usable-segment recovery with prespecified
overlap and held-out same-class retrieval, paired-edge coverage reported
separately) for any successor preregistration. The programme's staged
limitations table (forecasting reliable; avian syllable retrieval strong
within individuals; cross-speaker word retrieval not demonstrated beyond
control; proposals high-recall; typing failing its criterion across
four representation families; segment formation improving where F1 does
not; background rejection the persistent limitation; reference
consistency a material confound; transferable unit discovery and
composition not established) is the central figure of the companion
methodological paper: proposals → selected boundaries → paired segments
→ unit retrieval, each transition with its measured performance,
positive-control behaviour and principal failure mode.

## 8. Hierarchy of structural hypotheses (terminology contract)

Following the review, whole calls may be treated as *sentence-like
acoustic candidates* and recurrent internal fragments as *word-like
acoustic candidates* — as STRUCTURAL ROLES defined by evidence, never as
semantic labels:

| level | operational object | evidence that would warrant the role | programme status |
|---|---|---|---|
| H1 whole-event organisation | independently delimited vocal event; aggregated valid-frame features | reproducible whole-event structure across independent recordings | **partially supported**: whole-event acoustics support identification and context prediction (0.63 held-out BA) and predictive structure; failed family transfer means stable whole-event FAMILIES are not established |
| H2 reusable internal components | source-supported variable-length fragments recurring across distinct whole vocalisations | recurrence + source support + boundary stability + held-out recovery + specificity | **not established** under the tested discovery procedures (selection-before-threshold limitation measured) |
| H3 component reuse | fragments with DIFFERENT neighbouring material, not whole-call duplicates | recurrence across different surroundings | **not established** — oracle@10 coverage of 0.73–0.88 demonstrates candidate availability/localisation HEADROOM, not component reuse |
| H4 combinatorial organisation | arrangement contributes beyond unordered presence | ordered > unordered on protected data | **weak positional/order evidence only**: equal-thirds ordered advantage (+0.006, CI>0); withheld-combination generalisation failed |
| H5 functional composition | receiver responses follow a prespecified component rule | substitution/novel-combination experiments | **untested** (Track C, collaborator-led; RECEIVER_HANDOFF.md) |

A masked within-vocalization reconstruction study operationalised the
predictive form of the sentence-like hypothesis directly (does the rest
of a vocalisation constrain a hidden central 70 ms interval beyond its
local neighbourhood?): the simple boundary blend beat every learned
arm including the whole-call-context model (Δ_whole-call = −0.80
[−1.63, −0.06]; Δ_C ≈ 0), so **no vocalisation-level constraint of
this kind was demonstrated** on this material — an informative negative
for the H1→H2 progression under this operationalisation. A two-corpus
portability test (frozen comparison, stopped per design) replicated
the pattern with an informative twist: in Bengalese song (discrete
syllables; 3 evaluation blocks) the blend again won (0.802 vs 0.953
bidirectional; Δ_G negative); in chiffchaff (continuously voiced song;
1,466 eval positions across 172 blocks) learned bidirectional context
beat the blend (+0.109 [+0.087, +0.152]) — but the whole-call summary
added exactly nothing beyond it (Δ_G = −0.0004 [−0.0035, +0.0041]).
Across all three regimes, G never improved reconstruction beyond
bidirectional local context: the sentence-like-parent constraint
hypothesis is unsupported in this operationalisation everywhere, while
whether learned local context beats simple smoothness varies with
regime and training abundance.

The ladder H1 ⇏ H2 ⇏ H3 ⇏ H4 ⇏ H5 is strictly progressive; no level
implies the next, and only H5 would make the language analogy
biologically substantive. Terminology discipline: "sentence-like" and
"word-like" always carry "acoustic candidate"; feature-block names
(macrotexture/microtexture) never define temporal scale — scale is set
by explicit support and lag definitions (the history studies' 40 ms
window and 40–200 ms lags). A block-ablation study (physics/macro/micro
× whole-call/within-call) is specified as a possible future experiment
testing which feature family helps which level, not built into any
definition.

**Forecast-surprise as a boundary proposal signal.** A validated
immediate-history forecaster was used to generate an auditable
acoustic-deviation stream from the residual between predicted and
observed future states. We tested whether this signal improved
localisation of annotated Bengalese syllable onsets relative to
energy-change and spectral-flux baselines. Forecast surprise was
consistently more conservative, producing fewer false proposals per
second, but recovered substantially fewer annotated onsets and did not
improve the combined detector. We therefore retain forecast surprise as
a review-prioritisation and QA signal rather than a natural-boundary
detector. The result does not alter the previously closed
recurrent-unit discovery branch and does not imply biological unit
boundaries. More generally: a predictive residual can be informative
without being a good segmentation cue.

## 9. Outlook and application roadmap

The retained predictors provide building blocks for assisted context
annotation, acoustic-deviation screening and experimental stimulus
assessment. The current evaluations do not establish call-end
anticipation, physiological stress detection, transferable field
behaviour tagging or independent manipulation of caller identity and
acoustic dynamics. Each application requires validation against its own
target and operating conditions.

| priority | application | current status |
|---|---|---|
| 1 | auditable context-annotation assistance on eligible, in-domain recordings | closest to the validated classification endpoint; pilot shipped (`outputs/context_review_pilot/`: declared abstain band, uncalibrated scores, reviewer confirmation required) |
| 2 | acoustic-deviation review queue ("flagged for review", never "stress detected") | plausible use of residuals; alarm burden unvalidated |
| 3 | offline/shadow-mode call-end prediction | new boundary endpoint; observed-offset baseline first |
| 4 | controlled receiver assay | collaborator-led; independently validated stimuli; specific behavioural hypothesis (the dyn_cv envelope-dynamics question) |

One future estimand is specified now for completeness — not a result
obtained: label-conditioned acoustic prediction asks whether a
prespecified contextual variable changes the history-to-future mapping
beyond context-specific offsets (shared A H_t + b vs context-level
A H_t + b_C vs context-dependent A_C H_t + b_C; the scientifically
relevant contrast is L(context-level) − L(context-dependent)). This is
distinct from P(C | audio): context-classifiable audio could still lack
a context-dependent temporal rule. It would be label-conditioned
prediction, not an audio-only classifier, and would choose ONE
independently annotated variable and ONE primary comparison — not an
unrestricted search over caller, recipient, arousal, call type and
recording condition until something interacts.

**Older-history and context-dependent forecasting.** Adding acoustic
observations at lags of 40–200 ms reduced held-out nRMSE from 0.7103 to
0.6942 relative to the two-state baseline, an improvement of 0.0161
[reported block-clustered bootstrap CI 0.0119, 0.0198]. In contrast,
adding context-by-older-history terms beyond context-level and
context-by-recent terms increased loss from 0.6954 to 0.6988, giving a
primary improvement-scale contrast of −0.0034 [−0.0042, −0.0025]. The
findings support the predictive value of the additional acoustic
history, while the tested context-dependent extension did not
generalise beneficially. The cause of its poorer performance is not
isolated by this comparison. Results apply to continuously supported
spans of at least 280 ms within the ten frozen dyadic pairs; they do not
establish order dependence, identical context-specific dynamics, grammar
or communicative meaning. The shared longer-history arm is retained as a
research predictor within these applicability conditions only. A
subsequent one-shot avian portability study (chiffchaff; foreground-only;
6,762 clips, 100% with the required contiguous support; 626,498 held-out
positions across 172 recordings and 8 callers; the identical contract
with species-specific refitting and the runtime feature-contract gate
enforced) replicated the older-history benefit — Δ_older = +0.0171
[+0.0140, +0.0198], matching the bat magnitude — with immediate history
also helping (+0.0180 [+0.0135, +0.0205]) and both arms beating
persistence. Across the three tested corpora: **immediate recent history improved
prospective acoustic prediction in all three; additional 40–200 ms
history produced a further improvement in the evaluated bat and
chiffchaff material, while that incremental benefit was not established
in the marmoset extension.** The chiffchaff effect (+0.0171 [0.0140,
0.0198]) being close to the bat's (+0.0161) is descriptively
interesting and licenses no shared-memory-process inference; the
marmoset interval still accommodates a bat-sized effect and its
eligible population was much sparser and more concentrated. The support
audit appears beside the forecasting result (figure
`cross_corpus_forecasting.png` — labelled **procedure portability**,
not cross-species memory: immediate and older-history effects with
uncertainty, eligible-support fraction, and independent evaluation
groups per corpus) so that effect size and data support are read
together without support availability becoming a causal explanation
for effect differences.

**Marmoset replication of the older-history contrast.** We tested
whether observations at lags of 40–200 ms improved 40 ms-ahead
forecasting beyond the two-state baseline, using the supported
twelve-feature representation and the original fitting policy. Among
24,695 audited calls lasting at least 0.6 seconds, 1,917 contained the
required contiguous support. The final comparison used identical
held-out positions for both arms: 7,492 positions across 24 evaluation
recordings. The two-state and longer-history predictors achieved nRMSE
of 1.736 and 1.785, respectively. The improvement-scale contrast was
−0.0496 [reported bootstrap interval −0.1104, +0.0188], so the positive
older-history benefit observed in bats was not established in this
marmoset analysis. The interval includes zero and the original bat point
estimate; the result therefore does not demonstrate harm from older
history or a between-species difference. It is conditional on the
selected supported material and does not alter the earlier positive
result for immediate recent history. The extension is recorded as not
promoted for marmosets — an operational decision under uncertainty. A
runtime feature-contract gate (width, names, schema version at fitting
and scoring entry) now enforces the representation this study's
correction history motivated.

**Parallel global and local predictive views.** We combined the frozen
context classifier and shared acoustic forecaster without conditioning
one on the other. Each vocalization was represented by global
acoustic/context measurements, local prospective predictions, and
model-relative forecast residuals. Static acoustics retained
context-classification value, whereas context-dependent forecasting
did not improve local prediction under the tested interaction model.
Consistently, median forecast surprise did not show a detected C11–C12
difference. Together these results support complementary global and
local predictive descriptions rather than evidence for distinct
context-specific local transition rules. Concretely, audio branches
into G (global event characterisation and context classification),
F (shared local continuation) and R (model-relative deviation) — with
context-conditioned forecasting explicitly closed. Each bout can now emit one auditable
record combining context score with abstention, forecast surprise
(median and q90), whole-call static features, validity support, source
interval, caller/recipient provenance and model versions. The one
prespecified joint question — does the shared forecaster's residual
differ by context? — was tested as Θ_A = median(A|C12) − median(A|C11)
under the single frozen shared forecaster within the ten supported
dyadic pairs: **−0.0152 [−0.0430, +0.0097]** — no detected difference
(the interval spans zero and permits modest differences in either
direction; no equivalence margin was prespecified, so equality is not
established). The combined statement: C11/C12 are distinguishable from
global/static acoustics, but neither the tested context-conditioned
forecasting model nor the forecast-residual comparison established
context-specific local predictive dynamics. These are two COMPLEMENTARY
tests on shared corpus, representation and related machinery — not
independent corroborations. A single statistic was prespecified; no
residual-feature search was opened. A direct occupancy–dynamics
decomposition then tested the sharper hypothesis (different state
distributions with a shared local law): state distributions differ
(energy distance 0.0167, recording-group permutation p=0.001) **on 98%
common support**, and cross-context forecaster transfer on that common
support costs 0.4–0.9% nRMSE — Γ₁₁ = +0.0039 [−0.0013, +0.0091] lies
inside a prospective 1% equivalence margin, while Γ₁₂ = +0.0093
[+0.0053, +0.0129] has a point estimate inside but a CI edge beyond
it. The honest verdict: **different acoustic occupancy on heavily
overlapping support; practical predictive equivalence established in
one transfer direction, unresolved in the other** — leaning toward
shared local dynamics without forcing the conclusion, and with the
matched-transition diagnostic (cross/within ratio 1.014) and the
absence of residual-vector context signal both consistent with that
reading. A preregistered directional test on four never-analysed
caller-recipient pairs (24,513 common-support positions; δ_min = 0.005
declared prospectively from a half-margin rationale) left the
asymmetry hypothesis unresolved — Δ_asym = +0.0042 [−0.0053,
+0.0117], CI spanning both zero and δ_min — and, notably, both
transfer directions exceeded the equivalence margin on these groups,
so the earlier Γ₁₁ equivalence is itself population-unstable. The
defensible summary across both populations: small transfer costs
(0.4–1.3% nRMSE) in both directions, with no stable direction
difference — shared local dynamics remains the lean, formally
unestablished.

Two reasoning corrections are recorded with this framing. First, the
no-lexicon safeguard holds because **acoustic forecasting success does
not establish discrete units** — not because continuous success implies
their absence; a signal can have both predictable continuous dynamics
and discrete organisation, and the discovery studies simply did not
establish the latter under the tested procedures. Second, an
anticipatory controller requires P(call ends within H ms | audio now) —
a different target from the 40 ms feature horizon measured here; a
validity-aware evaluation scoring continuous spans can omit exactly the
vocal-to-nonvocal transitions such a detector must handle, and any
deployed trigger's timeline (window support, buffering, inference,
output delay) counts against the nominal horizon.

## 10. Data, code and provenance

All results regenerate from versioned artifacts via
`scripts/build_claim_ledger.py` and `scripts/build_manuscript_tables.py`;
the ledger's claims (43 at this draft) carry cohort/representation/mask/model/
aggregation/uncertainty/correction fields, and the falsification-status
footer distinguishes prespecified, exploratory-to-confirmed,
retained-scoped, closed-under-tested-procedure, descriptive, rejected
and quarantined records. Checkpoints and caches are content-addressed
and gitignored; corpora citations in the supplement.

## Acknowledgements

The advisory review process that shaped this programme operated as an
independent methodological auditor; its corrections — including
narrowed attributions, reconciled estimands and re-adjudicated
candidates — are preserved verbatim in the correction-history fields of
the supplement.

## Pending at this draft

- The seventy-file source-reference panel review (representative and
  challenge subsets; measurement-validation debt described as pending
  in §2 and §10 — not as completed validation).
- The MFA alignment human-validation sample (100 words,
  `outputs/mfa_alignment_review/review.csv`) — §7j's numbers carry
  "MFA-reference, human-validation pending" until this is done.
- Figure polish beyond F1 (per-study panels as needed by the venue).
- Venue-specific formatting; corpora citations to be completed in the
  supplement.
- Author contributions and ethics statements (Track C work remains
  prospective; no animal experimentation was conducted for this paper).
