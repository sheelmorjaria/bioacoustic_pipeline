# MACDE transfer study v1

This is a new frozen acoustic study, separate from the closed Stage 1 pilot. It uses locally available infant-marmoset, pipit and little-owl recordings. These corpora have already appeared elsewhere in the research programme: protocol independence does not imply fresh-data confirmation. Chiffchaff was excluded before preregistration because its files covered only two recording days. Previously evaluated BirdPark clips are not reused as protected transfer evidence.

## Design

The source audit uses the existing native-rate frontend, physical feature schema, spectral bands and absolute activity threshold. Sources are chosen in deterministic hash order, without inspecting labels or prediction scores. The search is bounded at 256 files per corpus; at most 48 files and two files per group are retained. A selected 128-frame window must contain at least one eligible position with 52 supported past/current frames and 20 supported future frames. Prediction itself targets physical measurements 40 ms ahead, with no overlapping source/target waveform samples. No gaps are concatenated.

Marmoset recording dates group all twins together. Bird filename year/identity proxies are grouped conservatively across the release's original train/test labels. These proxies do not independently establish recording independence or unseen-animal generalization. A deterministic group hash allocates fit, validation and held-out sets. Every corpus must provide at least four fit groups, two validation groups and three held-out groups, and 40 positions in each partition. Failure ends the study at its support audit; no model fitting or adaptive source substitutions follow.

Physical-only, spectral-only, combined and combined-with-disrupted-history neural arms share the MACDE architecture, parameter count and training budget. Unused streams have values and masks zeroed. Order disruption jointly permutes the oldest 50 frames and their masks within each 52-frame input, preserving the newest two states and the future target. The disrupted arm is trained and evaluated under this policy; comparisons therefore do not merely scramble a trained ordered model at inference.

Each source domain supplies its own training scalers, model parameters, validation epoch selection and final fit+validation refit. Recent-two-state and full-history physical Ridge baselines receive the same source partitions and eligible targets. Every trained source predictor is evaluated on its own domain and the other two domains, without target-domain parameter fitting or input adaptation.

For reporting, all predictors evaluated on one target domain use the same physical target standard deviations, estimated from that domain's fit+validation observations. These reporting scales never enter the source predictor. Raw per-feature RMSE is also retained. This separates prediction from reporting normalization; errors across different target domains should not be pooled casually.

The preregistered criterion requires combined representations to beat the recent-state physical Ridge baseline with an upper paired-bootstrap confidence bound below zero in all six directed transfers. All directions are reported regardless of sign. The order comparison and within-domain comparisons are secondary. Bootstrap intervals condition on fitted models, use recording proxies as clusters and have no multiplicity adjustment. No outcome unlocks Stage 2 automatically.

## Reproduction and provenance

```bash
OMP_NUM_THREADS=2 .venv/bin/python scripts/macde_transfer.py --preregister
OMP_NUM_THREADS=2 .venv/bin/python scripts/macde_transfer.py --prepare
# Only if every source-support gate passes:
OMP_NUM_THREADS=2 .venv/bin/python scripts/macde_transfer.py --run
```

Specifications, audit, results and paired residuals use the `outputs/macde_transfer_*` namespace; preparation uses `cache/macde_transfer.npz`. The specification pins implementation hashes before source extraction. The audit pins source and cache checksums. Fitting has a one-run marker and refuses a second attempt. Original pilot outputs remain unchanged.

Checkpoints in `checkpoints/macde_transfer_<source>_<arm>.pt` record the source scalers, arm policy and specification hash. These study checkpoints require the stream masking/history policy in the transfer runner; the Stage 1 inference wrapper is not an interchangeable loader for physical-only or disrupted arms.

Tests cover grouping, original release-partition merging, paired frame/mask disruption, fixed newest states, stream ablation and the unchanged core causality/validity contracts. The endpoint is acoustic forecasting. It does not establish receiver behaviour, semantic content, biological causality, or the presence/absence of discrete units.

## Frozen support-audit outcome

The prespecified gate failed. No models were fitted, no checkpoints were created, and no transfer or ordering claims can be made from this study.

| Domain | Fit groups / positions | Validation groups / positions | Held-out groups / positions |
|---|---:|---:|---:|
| Infant marmosets | 12 / 382 | **1 / 93** | 5 / 281 |
| Pipits | 6 / 684 | 2 / 228 | **2 / 228** |
| Little owls | 4 / 456 | 2 / 228 | 3 / 285 |

Requirements were at least 4 fit, 2 validation and 3 held-out groups, with at least 40 positions per partition, in every domain. The audit retained 66 files overall (29 marmoset, 20 pipit, 17 owl); all came from the fixed source search and eligibility rules. No unfavorable model scores informed this decision because fitting never started.

This is an operational support limitation under the frozen sampling budget and grouping rules, not evidence against transfer learning or temporal organization. A successor study would need a separately frozen source plan with greater independent recording coverage. The current study remains closed without relaxed groups, changed partitions or adaptive replacements. Stage 2 remains blocked.
