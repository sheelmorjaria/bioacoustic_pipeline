# NBD — Neural Boundary Detection (proposal + calibrator segmentation)

> Personal methodology-usage reference. Practical: real params, commands, when-to-use,
> and the empirical gotchas hit in *this* pipeline. See `BENCHMARK.md` §1/§30, `CLAUDE.md`,
> and `src/bioacoustic_pipeline/segmentation/nbd_{proposal,calibrator,evaluation}.py`.

## What it is

NBD is the pipeline's **fundamental segmenter**. It is a two-stage proposal-and-refine
boundary detector:

1. **Proposal** (`nbd_proposal.py`): per-frame acoustic signals → candidate boundaries at
   signal peaks above a floor, merged within a debounce window.
2. **Calibrator** (`nbd_calibrator.py`): maps each candidate's features → `P(boundary)` via a
   learned classifier, with an **ABSTAIN band** (keep / abstain / drop).

NBD is *not* a neural net in the heavy sense — "Neural" is historical; the proposal core is
deterministic DSP. The "learned" part is the calibrator (logistic / GBT) over handcrafted +
optional CPC-context features.

## Where it sits in the router

The segmentation router (`configs/segmentation_router.yaml`) picks the regime:

| Regime | Segmenter | Use when |
|---|---|---|
| continuous speech-like | **NBD + CPC calibrator** | human speech (LibriSpeech); combined-F1 **0.836** |
| discrete calls in noise | PCEN + CRNN (HECB Stage 1) | meerkat/marmoset isolated calls |
| exploratory / unknown | **fixed-threshold NBD + ABSTAIN** | new species, no calibrator |

NBD is the fallback for unknown species and the headline method for speech-like audio. For
discrete calls use **HECB** (see `hecb.md`) — NBD over-segments in noise.

## Proposal stage — API

```python
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals

sig   = compute_signals(audio, sr, hop_ms=10.0)                 # -> NBDSignals
props = generate_proposals(sig, floor_energy=0.05, floor_flux=0.03, debounce_ms=30.0)
# props: list[Proposal], each .time_ms (ms) and .duration_to_next_ms
```

**Per-frame signals** (`NBDSignals`): `rms`, `energy_change`, `spectral_flux`, `band_flux`,
`rms_slope` (all at `hop_ms` resolution, default 10 ms).

**Peak logic**: energy-change peaks above `floor_energy` ∪ spectral-flux peaks above
`floor_flux`; peaks within `debounce_ms` are merged (stronger combined signal wins). Segment
boundaries = consecutive proposal times (first proposal = audio start, last = audio end).

> **Name mapping gotcha.** `species_profiles.yaml` uses keys `energy_threshold` /
> `spectral_threshold`; `generate_proposals` uses `floor_energy` / `floor_flux`. They are the
> same two knobs. `energy_* → floor_energy`, `spectral_* → floor_flux`.

## Calibrator stage — API

```python
from bioacoustic_pipeline.segmentation.nbd_calibrator import BoundaryCalibrator, proposal_features
cal = BoundaryCalibrator()
cal.fit(proposals, labels, context=cpc_context_or_None)   # context = CPC frame-embed matrix
keep = cal.keep_threshold                                  # tuned by refit; apply() clamps ABSTAIN to keep
```

- **Features per proposal = 13 scalars**: 8 point features + 5 local-context
  (`rms_ratio`, flux/energy contrast, `flux_asym`, `duration_to_next`).
- **+ 768-D CPC learned-context** (`proposal_context` over `CPCBackbone.embed_frames`):
  pre/post/diff of the CPC frame embedding around each boundary. This is the layer that lifts
  combined-F1 from ~0.79 (scalars) to **0.836**.
- **ABSTAIN band**: `P ≥ keep_threshold → emit`; `abstain ≤ P < keep → ABSTAIN` (uncertain);
  `P < abstain → drop`. `keep_threshold` is tuned on combined-F1 and stored in the checkpoint;
  `apply()` clamps the ABSTAIN band to `keep`.

## Species proposal profiles (`configs/species_profiles.yaml`)

These are **initial priors**, refined per-corpus by the calibrator. `floor_energy` /
`floor_flux` / `debounce_ms` / `sample_rate`:

| species | sr | floor_energy | floor_flux | debounce_ms | note |
|---|---|---|---|---|---|
| human | 16k | 0.15 | 0.05 | 150 | word/pause onsets; phonemes handled by calibrator |
| marmoset | 48k | 0.15 | 0.05 | 30 | combinatorial — rapid syllables |
| bat (yaml) | 250k | 0.20 | **0.50** | **200** | ⚠️ HOLOPHRASTIC — see gotcha |
| meerkat | 8k | 0.08 | 0.04 | 40 | broadband low-sr; tight debounce |
| dolphin | 96k | 0.15 | 0.08 | 80 | tonal whistles |
| corvid | 48k | 0.15 | 0.05 | 60 | provisional |

## Commands

```bash
# Refit the human calibrator on LibriSpeech (default: gbt + CPC context)
python scripts/refit_nbd_calibrator.py --n-files 400 --kind gbt --context cpc

# Track A validation (word/multi-resolution F1 vs wav2vec2 forced alignment)
python scripts/validate_track_a.py --calibrated --n-files 500
python scripts/validate_track_a.py --self-test                      # synthetic, no network

# Plausibility-select a species NBD profile (sweep floors/debounce)
python scripts/bat_segmentation_sweep.py --n-files 400
```

In code: `segment_audio(cpc=...)` / `run_pipeline(cpc=...)` build CPC context at inference
when the calibrator needs it (`run_pipeline` auto-constructs `CPCBackbone`). Train/inference
proposal params are unified via `orchestration.CALIBRATOR_PROPOSAL` = `(0.04, 0.02, 25.0)`.

## ⚠️ Gotchas (all hit in this project)

1. **NEVER feed wav2vec2 features to the calibrator.** wav2vec2 is the Track-A ground-truth
   yardstick (`probes.forced_align`) — using it in the calibrator is circular. CPC is the
   pipeline's *own* SSL model → fair. Human CPC only; firewall-blocked on animals.
2. **The yaml `bat` profile (0.20/0.50/200 ms) is HOLOPHRASTIC and INVALID.** It collapses
   multi-call bouts to ~0–1 segments → false nulls (this is why the old
   `outputs/track_b_bat_5000.json` is invalid; see BENCHMARK §bat-revalidation). Use the
   **`bat_call_granular`** profile instead: `floor_energy=0.05, floor_flux=0.05,
   debounce_ms=25.0` (plausibility-selected via `bat_segmentation_sweep.py`). The default
   `_PROPOSAL` in `dusted_bat.py` / `compositional_candidates_bat.py` is already this.
3. **CPC context is human-only.** On animals the firewall blocks CPC → use `--context none`
   (scalar-only, species-portable). The acoustic fingerprint is the interpretable complement.
4. **NBD over-segments noisy field audio.** MeerKAT (8 kHz) showed a boundary-F1 gap (~0.079,
   over-segmentation) — that gap is what HECB's CRNN closes for discrete calls.
5. **§1 reports TWO F1s.** Word-level (~0.6) and **combined/multi-resolution (~0.836)**.
   Combined is the honest number — it reclaims word-FPs that land on real phoneme boundaries
   (an acoustic segmenter finds sub-word structure by design). Cite combined.

## Empirical headline

Track A (LibriSpeech, 150 files), combined/multi-resolution F1:

```
fixed-threshold 0.67 → scalar-GBT 0.79 → GBT+CPC-context 0.836  (P=0.77, R=0.92)
```

CPC's win is **precision**. At the blueprint's 0.85 target.

## Related

- **HECB** (`hecb.md`) — two-stage segmentation built *on top of* NBD (event-conditioned
  internal segmentation). Use HECB for discrete calls in noise; NBD+calibrator for speech-like.
- Acoustic fingerprint, CPC, firewall — see CLAUDE.md.
