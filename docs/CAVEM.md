# CAVEM — Contextual Acoustic and Vocal Event Model

Repository reference, compiled 21 October 2026. This document distinguishes the implemented framework, completed endpoint-specific studies and proposed applications. Saved protocols and results remain authoritative; this documentation does not reopen completed experiments.

Related references: [MACDE](MACDE.md), [RCLAT](RCLAT.md).

## Purpose and evidential status

CAVEM models acoustic events in their temporal and observed social context. Its central question is whether acoustic information or event history improves prediction beyond simpler current-event and metadata baselines.

An acoustic representation is not a meaning label. An annotated recipient is not proof of intended address, and a subsequent vocalisation is not automatically a communicative response. Each endpoint needs independent reference evidence and an explicit observation contract.

| Component or study | Recorded status |
|---|---|
| Causal event-processing implementation | Implemented |
| Bat masked-acoustic reconstruction | Recurrent model worse than tested simple baselines |
| Bat operational recipient-response analysis | No supported incremental acoustic or history benefit |
| Marmoset next-annotated-caller prediction | Primary linear-history contrast failed; recurrent secondary contrast positive |
| Elephant receiver-grounding control | Added treatment did not improve prediction; acoustic linkage unresolved |
| Wild marmoset response-type task | Insufficient label support; abstained |
| MACDE acoustic interface | Mechanically implemented and tested |
| MACDE downstream Stage 2 | **Blocked** |

The next-caller finding is conditional on annotation sequences. It does not establish receiver-response prediction, acoustic semantic decoding or independent-session transfer.

## Conceptual levels

| Level | Representation | Required evidence |
|---|---|---|
| Acoustic | Validity-masked trajectories within a recording or event | Verified waveform support and coordinates |
| Event history | Observed callers, timing and context across events | Reliable event identity, chronology and metadata |
| Biological outcome | Subsequent vocal, behavioural or physiological measurements | Independently observed and appropriately timed references |

Agreement among outcomes can be informative but does not define meaning by itself. Isolated clips cannot be assembled into invented conversations. Missing identities and unobserved outcomes remain missing.

MACDE can provide the first level; CAVEM can consume acoustic summaries and observed event history. Their mechanical compatibility does not prove that the combination improves any biological endpoint.

## Core implementation

Source: [contextual/cavem.py](../src/bioacoustic_pipeline/contextual/cavem.py).

### Event representation

`EventRecord` carries:

| Field | Meaning |
|---|---|
| `acoustic` | A time-by-feature acoustic array |
| `valid` | Explicit valid-frame mask |
| `dt_prev_ms` | Optional elapsed-time value, requiring a declared coordinate convention |
| `duration_ms` | Event duration |
| `caller` | Observed caller identity or unknown |
| `recipient` | Observed recipient annotation or unknown |
| `context` | Observed context or unknown |
| `provenance` | Source evidence and coordinate information |

The generic event-input helper averages supported acoustic frames and adds two elapsed-time features and a log-duration feature. For an acoustic dimension F, that prototype input has F+3 dimensions. With no supported frames, a numerical zero fill does not create an observed acoustic measurement.

The prototype causal GRU maintains a chronological state and predicts a binary endpoint. Training-side normalization must remain fixed at evaluation. Its small default architecture and generic split helpers are engineering components, not universal scientific protocols; study-specific runners define the actual models and evaluation units.

### Temporal availability

A prediction may use only information available at its declared cutoff. Complete-call duration or acoustics are available after the call, not automatically at its onset. Recurrent state must reset at recording boundaries and other declared discontinuities.

The generic elapsed-time transform clips negative values and caps large gaps. It therefore cannot preserve overlap by itself. Consumers requiring overlap information must retain signed timing and explicit overlap indicators, as the MACDE bridge does.

**Legacy timing limitation:** comments and calculations in the generic response helpers do not use a uniform onset-to-onset versus previous-end convention. A new outcome study must audit and freeze its coordinate convention rather than assuming that helper names certify it. The endpoint definitions in historical study runners must be reported as implemented, without silently reinterpreting them as new prospective tasks.

Likewise, a later event is not proof that the entire intervening recording was monitored. Negative response labels require an observation contract; missing follow-up must not be converted into verified nonresponse.

## Comparator logic

A contextual model should demonstrate incremental value against strong alternatives.

| Comparator | Information or purpose |
|---|---|
| Metadata only | Observed identities and context |
| Current-event acoustics | Interpretable physical measurements |
| Explicit history summaries | Counts, recencies, timing and composition |
| Recurrent history | Ordered event sequence |
| Combined acoustic/history model | Additional predictive information under a specified contract |

The primary contrast should compare against a competitively specified baseline, not merely a weak high-dimensional model. A positive secondary contrast cannot replace a failed preregistered primary endpoint.

When loss is lower-is-better, `loss(baseline) − loss(augmented)` is positive for an improvement. Uncertainty must reflect repeated recordings, dates, individuals and listeners as appropriate. Group names are not certificates of biological independence.

## Bat acoustic reconstruction: completed negative result

The initial local bat annotation tables did not support constructing the intended multi-event interactions: none of 91,080 inspected local annotation files had more than one annotated vocalisation in that representation. The fallback used valid cached acoustic bouts for masked-feature reconstruction. This finding concerns that inspected table/release representation; later source tables supported a different operational history analysis.

The reconstruction task compared persistence, interpolation and a self-supervised recurrent model using identical masks and eligible cases.

| Model | Stored normalized error ↓ | Conditional interval |
|---|---:|---|
| Persistence | 2,259.7293 | [2,083.3784, 2,430.5521] |
| Interpolation/blend | **1,701.6076** | [1,591.5853, 1,807.9918] |
| CAVEM self-supervised GRU | 4,585.3261 | [4,464.0256, 4,703.6659] |

These values reproduce the saved artifact's scale; they are not percentages and are not directly comparable with MACDE errors. Evaluation included 832 masked cases across 476 groups.

The recurrent model was worse than both simple alternatives. Interpolation can use surrounding context and is a retrospective reconstruction comparator, not a causal forecasting system. This experiment did not establish interaction modelling or semantics.

Evidence: [results](../outputs/cavem_v1_bat.json), [runner](../scripts/cavem_v1_bat.py).

## Bat event-history recipient analysis: completed unsupported increment

A separate analysis used original file-information and annotation tables. Operational streams grouped treatment and channel and split at gaps exceeding 30 seconds. These streams are analysis conventions, not certified independent sessions.

The risk set required known, distinct caller and annotated recipient identities plus cached acoustics. It contained 2,149 events, with 66 positives; 618 events were evaluated. Exclusions included unknown identities, absent acoustic support and insufficient operational follow-up.

### Exact endpoint and availability limitation

The implemented positive label asks whether the annotated recipient appears as an emitter within five seconds **after the index file-start recording timestamp**. Follow-up eligibility uses the presence of a later file timestamp within six seconds.

That eligibility rule is a follow-up proxy, not evidence of continuous five-second monitoring. Negatives are operational labels, not independently verified absence of a receiver response. Moreover, complete-event acoustics and duration are not necessarily available at the index file start. The result must not be relabelled as a validated onset-time prospective response forecast.

A future prospective analysis needs verified onset/offset coordinates, feature availability and continuously observed risk windows. This documentation preserves the historical analysis rather than changing its endpoint after observing results.

### Models and results

| Arm | Input | Evaluation log loss ↓ |
|---|---|---:|
| M0 | Observed metadata and timing/duration descriptors | **0.1339** |
| M1 | M0 + acoustic means and standard deviations | 0.1391 |
| M2s | M1 + explicit preceding-history summaries | 0.1408 |

The acoustic cache supplies 24 summary values: mean and standard deviation of twelve physical measurements. The linear history arm adds preceding 30-second history summaries.

The M0-minus-M1 gain was −0.0052, CI `[-0.0174, 0.0070]`. M1-minus-M2s was −0.0018, CI `[-0.0099, 0.0081]`; a caller-cluster sensitivity interval also included zero. Recurrent extension was deferred after the linear-history gate failed.

No incremental acoustic or history utility was supported. Recipient annotations, temporal adjacency and operational stream labels do not establish intent or causal turn-taking.

Evidence: [results](../outputs/cavem_event_history_bat.json), [runner](../scripts/cavem_event_history_bat.py).

## Marmoset next-annotated-caller study

This study used Grijseels recording annotations to predict the caller of the next annotated event. It used **no audio and no MACDE features**.

### Population and causal target

The task conditions on an annotated sequence, not exhaustive observation of every natural call. Candidate animals come from recording metadata, including available animals with zero observed calls. Author-selected paired/trio versions were retained without pooling alternative versions; solo and flagged recordings were excluded according to the protocol.

Exact-onset ties were excluded and reset history instead of being arbitrarily ordered. Different-onset overlapping calls remained eligible. Terminal events lacking a next target were excluded. Prediction uses current-onset information; later offsets and durations are not inputs.

Histories include at most 32 events and never cross recording or tie boundaries. Caller indicators, elapsed-time information and availability indicators form the sequential input. There is no maximum-gap claim that the events constitute complete natural exchanges.

Date-group partitioning produced 7,476 training, 4,233 validation and 3,235 held-out targets. Held-out evaluation spanned nine dates and 29 sessions, with recurring animals. This is not unseen-individual evaluation.

### Models

| Arm | Definition |
|---|---|
| M0 | Training caller frequencies with smoothing and available-animal mask |
| M1 | First-order current-to-next-caller table, conditioned on animal configuration |
| M2 | M1 plus linear summaries of the available history |
| M3 | M1 plus recurrent processing of the same bounded event history |

M2 and M3 have matched access to history information, but not identical model capacity. Validation selected training duration under the frozen policy, followed by the specified refit. The primary test was M1 versus M2; recurrent comparisons were secondary.

### Results

| Arm | Event-weighted log loss ↓ | Accuracy ↑ | Session-macro log loss ↓ |
|---|---:|---:|---:|
| M0 | 0.6793 | 0.6853 | 0.6698 |
| M1 | 0.7586 | 0.5901 | 0.7389 |
| M2 | 0.7538 | 0.6334 | 0.7173 |
| M3 | **0.5886** | **0.7218** | **0.5851** |

The primary M1-minus-M2 contrast was 0.0048, session-bootstrap CI `[-0.1121, 0.0929]`: **the primary criterion failed**.

The secondary M2-minus-M3 contrast was 0.1652, CI `[0.0749, 0.2735]`. Date- and current-caller-cluster sensitivity intervals also favoured M3. This secondary analysis supports predictive value from the tested recurrent representation over tested compressed summaries, conditional on this annotated task. It was not multiplicity-adjusted evidence for a broad family of hypotheses.

M3's lower observed loss than M0 is also useful descriptively, but should not be substituted for the registered primary comparison. Annotation completeness, next natural vocal event, recipient response and acoustic semantics were not established.

Evidence: [dataset audit](../outputs/grijseels_dataset_audit.json), [preregistration](../outputs/grijseels_next_caller_preregistration.json), [results](../outputs/grijseels_next_caller_results.json), [runner](../scripts/grijseels_next_caller.py).

## Independent receiver-grounding investigations

### Elephant playback outcomes

An adjacent analysis used independently observed post-playback vigilance, with 34 trials across seven family groups. The outcome was vigilance duration within the defined 60-second post-playback period.

| Predictor | Out-of-fold MSE ↓ |
|---|---:|
| Metadata only | **479.0194** |
| Metadata + treatment | 483.1870 |

The improvement was −4.1676, group-bootstrap CI `[-32.1846, 12.6834]`. Added treatment did not demonstrate predictive benefit.

The added variable was **treatment**, not acoustic features. A verified playback-stimulus-to-waveform-feature crosswalk remains missing, so this is not a MACDE–receiver experiment. Leave-group-out models share training data; their uncertainty should not be presented as independent experimental replications.

Evidence: [results](../outputs/elephant_receiver_condition_results.json), [source-code inspection](../outputs/elephant_playback_code_inspection.json).

### Wild marmoset response-type support

A strict exchange audit retained 178 pairs, with 100 assigned to fit, 74 held out and four unassigned. The original seven-class response-type endpoint lacked support: held-out Chirp was absent from fit, while some rare fit labels were absent from held-out data.

The study abstained. It did not merge labels, repair partitions after inspection or infer missing identities to make fitting possible. Source role labels are not independently grounded communicative meanings. Selected exchange pairs also do not establish exhaustive nonresponse observation or independent nonvocal receiver behaviour.

Evidence: [exchange eligibility](../outputs/wild_marmoset_exchange_eligibility_audit.json), [label support](../outputs/wild_marmoset_response_type_label_support_audit.json).

## MACDE–CAVEM mechanical interface

Source: [macde_bridge.py](../src/bioacoustic_pipeline/contextual/macde_bridge.py). Tests: [bridge tests](../tests/test_macde_cavem_bridge.py).

```text
Verified waveform and source coordinates
                  │
          frozen ordered MACDE
                  │
    supported trajectory within observed event
                  │
      masked post-call acoustic summaries
                  │
     observed identities/context + event timing
                  │
            CAVEM event sequence
                  │
    separately specified downstream prediction
```

### Typed source and metadata contract

| Object | Required content |
|---|---|
| `AudioSource` | Explicit source ID, coordinate reference, native sample rate, sample origin, waveform and optional sample mask |
| `VocalEvent` | Source-linked start/end samples and independently observed metadata |
| `ObservedMetadata` | Value, evidence and sample coordinate at which it became available |
| `AcousticExport` | Event coordinates, encoder identity, trajectories, summary values, masks and quality |

The checkpoint loader requires a trusted SHA-256 and accepts the ordered combined export policy. The model must be frozen and in evaluation mode. Source IDs, coordinate systems, bounds and integral sample coordinates are checked.

Prediction is post-call only: the cutoff equals the event end. The encoder receives source audio only through that cutoff, preserving valid preceding history. Exported event frames must lie fully inside the event. Changes whose earlier state predates the event are masked.

Metadata may be used only if supported by independent evidence and available by the cutoff. The acoustic model does not invent callers, recipients or context. Unknown values remain explicit.

### Acoustic summaries and transport

For a 64-dimensional state, the export includes:

- State mean, standard deviation, first state, last state and slope.
- Four temporal-quarter state means.
- Mean and mean absolute change at each of four change lags.

This is 17 × 64 = **1,088 summary values**, each with a validity indicator. The synthetic smoke test's 1,024 supported summaries is a supported subset, not the total dimension.

The finite CAVEM transport vector has **2,186 dimensions**: filled summary values, summary-validity indicators, seven quality measures and three timing/overlap indicators. Explicit signed gaps and overlap flags supplement elapsed-time features that otherwise clip negative values.

A transport row marked numerically valid means it is safe for the consumer to process. It does not mean every acoustic summary was supported; the embedded masks and `acoustic_supported` flag retain that distinction.

Sequences reject incompatible source IDs, coordinate references, sample rates, encoder identities or summary schemas. Completion times must increase. The bridge does not certify that independently observed event boundaries or metadata are scientifically correct; those remain source-audit responsibilities.

### What the tests establish

The [saved smoke result](../outputs/macde_cavem_bridge_smoke.json) uses synthetic tone audio and reports:

| Check | Result |
|---|---|
| Finite consumer states | Passed |
| Future-audio exclusion | Passed |
| Consumer-prefix causality | Passed |
| Scientific fitting | Not performed |
| Mechanical-only flag | True |
| Stage 2 ready | **False** |

These are interoperability safeguards. They do not demonstrate better next-caller prediction, recipient-response prediction or communicative interpretation.

A post-call representation cannot prospectively predict a response that already began during that call without a separately declared endpoint handling that overlap. Likewise, bridge timing must not be passed into legacy response helpers without verifying compatible conventions.

## Prospective integration comparison

A future study could compare four frozen input policies under the same outcome, eligibility and protected evaluation:

| Arm | Information |
|---|---|
| C0 | Observed metadata/context only |
| C1 | C0 plus interpretable acoustic descriptors |
| C2 | C0 plus frozen MACDE representations |
| C3 | C0 plus descriptors and frozen MACDE representations |

A useful primary increment would be `loss(C1) − loss(C3)`, accompanied by comparison against C0 so that improvement over a weak acoustic baseline is not mistaken for biological usefulness.

This table is a proposal, not an executed study or permission to fit. A valid protocol must specify source linkage, risk-window coverage, feature availability, target validity, dependence-aware partitions, model selection and uncertainty before scoring.

Sequential training with a frozen acoustic encoder offers clearer attribution than immediate joint optimisation. Joint learning would require separate justification and adequate reference data.

## Interpretation and limits

The current programme supports a working causal event architecture and an endpoint-specific recurrent next-caller result. It also contains failed primary comparisons, unsupported acoustic increments and source-reference abstentions. All belong in the research record.

The following distinctions remain essential:

```text
mechanical interoperability ≠ downstream predictive utility
next annotated caller ≠ verified receiver response
annotated recipient ≠ inferred communicative intention
behavioural association ≠ causal communicative effect
acoustic reconstruction ≠ semantic understanding
```

Claims about communicatively relevant acoustic information require independent outcomes and controls for identity, context and recording domain. Stronger causal claims require suitable interventions and receiver measurements. Neither continuous acoustic processing nor discrete event modelling establishes that animal signals contain or lack human-like words.

MACDE's within-bird forecasting result does not unlock CAVEM Stage 2; its completed cross-bird and multi-bird limitations remain part of the evidence. RCLAT's human linguistic references and proposed listener-interpretation work address separate questions and do not supply animal semantic ground truth.

## Reproducibility and use

Each historical runner and result linked above defines its own endpoint and population. Preserve source coordinates, explicit missing values, frozen splits and failed eligibility decisions when reusing components. Do not substitute generic helper defaults for a study protocol.

This reference was assembled from saved artifacts and implementation inspection. No model fitting, new outcome scoring or changes to frozen study records were performed for documentation.
