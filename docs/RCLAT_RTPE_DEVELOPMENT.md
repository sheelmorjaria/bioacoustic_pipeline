# RCLAT Temporal Prediction Engine: implementation v1

Status: software implemented; synthetic integration smoke completed. RTPE-1 scientific protocol is not frozen and no real-data RTPE model was fitted. Previous study outcomes and progression blocks are unchanged.

## Implemented contracts

`src/bioacoustic_pipeline/representation/rclat_rtpe.py` provides:

- `TemporalPredictionEngine`: a unidirectional GRU over causal masked moving means at explicit frame scales. Multiple horizon-specific acoustic outputs share continuous states. Optional symbol logits permit a separately supervised linguistic decoder. No bidirectional layers, temporal normalisation over future frames, or inferred linguistic boundaries are used.
- `forecast_windows`: fixed histories from one recording with independently observed future feature cells as targets. Every frame from history start through the furthest target must have some observed support. Fully unsupported gaps exclude the window; partial feature missingness retains its explicit mask. Time is never compressed across gaps.
- `TrainingScaler`: training-only observed-cell statistics; transform never adapts to evaluation observations.
- `HistoryRidge`: the same full ordered feature history and masks, with training-only standardisation and separate masked Ridge fits per target cell. Alpha is explicit, not selected from evaluation results. It already has access to order: a GRU advantage would show utility beyond a linear ordered-history model, not information unavailable to all non-GRU models.
- `masked_mse`: computes loss only on finite, observed target cells. Missing targets do not enter arithmetic or gradients. Scientific aggregation over recordings and individuals must be frozen separately; this cell loss is not a group-level endpoint.
- `permute_history`: applies a caller-specified bijection to feature frames and masks together. It preserves the constituent observations; it does not certify naturalness, continuity, equivalent prediction difficulty, or artifact matching.
- `order_loss`: auxiliary loss requires explicit approval metadata and a control-audit hash. Approval is an external responsibility; a boolean flag does not itself establish scientific validity.
- `BehavioralReadout`: unavailable without explicit independent-observation, source-use, transfer and reference-hash metadata. The default engine has no active behavioural head. No current CAVEM or request-interpretation gate is satisfied by this API.
- `validate_group_split`: rejects recording or individual identifiers spanning train, validation and test partitions. The caller must supply independently valid identities; this checker cannot discover mislabeled or related individuals.

Input is floating B,T,F with matching boolean validity. A model call consumes supported, unpadded windows. There is no streaming state cache, waveform loader, implicit padding, resampling, source selection, or target alignment. Horizon/scale/history units are frames; real studies must freeze frame cadence and waveform support.

For future-acoustic forecasting, only the last position in each window is scored, against the actual observations at its specified future offsets. Earlier prefix outputs are useful for causality tests, but are not automatically a separately evaluated information-accrual analysis. Optional linguistic logits are not a complete validated CTC trainer. Existing label inventories, independently checked lengths and decoder policies still require a separate protocol. Decoder prediction time is not physical information-availability time.

## TDD and integration verification

New tests initially failed because the module was absent. After implementation the initial six contracts passed. A further training-scaler test failed before its class was added; all eight tests then passed. The tests exercise exact prefix causality, missing-value invariance, gap-safe future targets, masked gradients, known linear history recovery by Ridge, permutation/mask correspondence, auxiliary/behavioural gates and individual/recording split leakage. Ruff passed for the library, tests and new runner.

`scripts/rclat_rtpe_smoke.py --output <new-directory>` is CPU-only and accepts no real-data source. It creates ten independently identified synthetic AR trajectories, splits them 6/2/2 into training/validation/test, fits training-only scaling and Ridge, and runs exactly 40 GRU updates. Outputs retain all loss values, per-recording scores, synthetic model state and hashes. Reversed history is a plumbing control only; it changes recency and is not an artifact-matched order experiment. No winner, uncertainty interval or scientific pass criterion is declared.

Example:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/rclat_rtpe_smoke.py --output outputs/rtpe_smoke_new
```

The executed smoke is retained at `outputs/rclat_rtpe_implementation_v1`.

## Required scientific freeze before RTPE-1

1. Authorised source population; independently valid recording/individual groups; deterministic partitions and hashes. Existing development exposure must be disclosed. No existing protected final-test source is implicitly authorised.
2. Feature definitions and causal waveform support. An encoder can be mathematically causal while upstream centred windows or full-recording normalisation leak future audio. Feature timestamps must reflect the latest consumed waveform sample. Targets must come from actual future waveform observations, not another predictor's generated labels.
3. Frame cadence, all history/scale/horizon values, target dimensions, support masks, eligibility, target scaling, clipping and weighting. Use identical eligible origins and target cells across arms.
4. Fixed T0 Ridge and T1 GRU search budgets, training-only/validation-only selection, optimiser, stopping budget, seeds, checkpoint rules, numerical and resource checks. The synthetic smoke settings are software fixtures, not proposed scientific defaults.
5. A control audit for T2/T3: preserve masks and valid observations, distinguish recency changes from arrangement effects, quantify boundary artifacts, and supply matched controls. A permutation classifier alone cannot establish meaningful sequence organisation.
6. A primary held-out group-level metric and paired comparison, sufficient independent-group support, and an appropriate uncertainty procedure. The communication crossed bootstrap remains uncertified and is not silently reused here.
7. Linguistic and behavioural targets require their own independent references and downstream protocols. Better acoustic forecasting cannot pass an ASR, semantic or behavioural gate.

No ASR dropout dependency, frozen predecessor source, model checkpoint or scientific result was modified by this implementation. P100/P960 and MACDE–CAVEM Stage 2 remain blocked.
