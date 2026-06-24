# HECB-Seg — Hierarchical Event-Conditioned Boundary Segmentation

> Personal methodology-usage reference. Practical: stages, API, when-to-use, and the
> empirical regime limits hit in *this* pipeline. See `BENCHMARK.md` §30, the
> segmentation-roadmap / hecb-seg-architecture memory notes, and
> `src/bioacoustic_pipeline/segmentation/{hecb.py,hierarchical.py}`.

## What it is

HECB is the pipeline's **two-stage segmenter for discrete calls in noise**. It builds *on top
of* NBD (see `nbd.md`): a learned detector isolates whole calls, then NBD segments each call
*internally* — conditioned on the parent event being real. Hence **Event-Conditioned**.

The intuition: in noisy field audio (meerkat, marmoset isolated calls), raw NBD over-segments
— it fires on noise between calls. HECB's CRNN first says "this region is a call," and only
*inside* that region does NBD look for sub-call structure. A **parent-quality gate**
(`min_parent_confidence`) abstains on unreliable parents rather than segmenting noise.

## The five stages (design)

| Stage | What | Status |
|---|---|---|
| 1 | **PCEN + CRNN call isolation** — `P(call)` per frame → hysteresis → `CallEvent`s | ✅ implemented |
| 2 | **Event-conditioned NBD proposals** — NBD *inside* each CRNN event | ✅ implemented |
| 3 | **Typed boundary evidence** — NBD(13) + CRNN-context(9) + patch(8) + fingerprint(2) = 32-D | ✅ implemented |
| 4 | **Typed proposal classifier** — onset / offset / internal / noise | roadmap |
| 5 | **Semi-Markov decoder** — sequence-coherent boundary selection | roadmap |

`hierarchical.py` implements the Stage-1→2 data structures (`CallEvent`,
`InternalSegment`, `two_stage_segment`). `hecb.py` is the runtime inference path
(`run_pipeline(segmentation="hecb")`): PCEN → CRNN → typed evidence.

## API

```python
from bioacoustic_pipeline.segmentation.hecb import load_hecb, hecb_segment_audio, HECBDSPConfig
from bioacoustic_pipeline.segmentation.hierarchical import (
    CallEvent, InternalSegment, two_stage_segment, proba_to_call_events, segment_call_internally,
)

model, dsp = load_hecb(species, device="cpu")        # loads the species CRNN (lazy, checkpoint-backed)
events: list[CallEvent] = hecb_segment_audio(audio, sr, model, dsp, ...)
# each CallEvent has .internal_segments: list[InternalSegment] (the sub-call units)
```

Key pieces:
- **`HECBDSPConfig`** — PCEN/mel/CRNN shape (`n_mels`, `hidden`, …).
- **`compute_pcen(audio, sr, n_fft, hop, n_mels)`** — per-channel-energy-normalization
  spectrogram (suppresses noise gain before the CRNN).
- **`_crnn_proba(model, audio, dsp, device)`** — Stage 1: Conv2D→BiLSTM→`P(call)` per frame.
- **`_proba_to_events(proba, dsp, min_ms=40.0)`** — hysteresis + connected components → events.
- **`_extract_typed_evidence(...)`** — Stage 3: the 32-D typed vector per proposal
  (`NBD13 + CRNN-ctx9 + patch8 + fp2`).
- **`two_stage_segment(...)`** — end-to-end Stage 1→2; the `min_parent_confidence` gate drops
  low-confidence parents.

`InternalSegment` = one sub-call unit (NBD-derived) inside a `CallEvent`. `CallEvent` carries
the CRNN onset/offset + `internal_segments`.

## Where it sits in the router

| Regime | Segmenter | Use when |
|---|---|---|
| **discrete calls in noise** | **PCEN + CRNN (HECB)** | meerkat, marmoset isolated calls |
| continuous speech-like | NBD + CPC calibrator | human speech (use NBD, not HECB) |
| exploratory / unknown | fixed-threshold NBD + ABSTAIN | new species |

**HECB is for discrete calls in noise. It is the wrong tool for graded bout vocalizations**
(bat) — see gotcha.

## Commands

```bash
# Train / evaluate the PCEN+CRNN frame detector
python scripts/train_pcen_crnn_detector.py
python scripts/train_pcen_cnn_calibrator.py

# Boundary-quality sweep (HECB decode accepts ANY parent events: oracle/CRNN/degraded/NBD)
python scripts/boundary_quality_sweep.py --n-clips 300 --n-per-type 500

# Factorial: HECB-typed vs scalar+CPC × {no-ABSTAIN, ABSTAIN+tuned} on LibriSpeech
python scripts/hecb_calibrator_factorial.py --n-train 400 --n-test 100
```

## ⚠️ Gotchas (all hit in this project)

1. **Regime mismatch on graded bouts.** The CRNN is trained on *discrete calls in noise*
   (F1=0.703 one-to-one on MeerKAT). Bat *bouts* are graded/continuous/overlapping — the wrong
   regime. Do **not** use HECB on bat; use raw NBD `bat_call_granular` (see `nbd.md`). For
   Bengalese, prefer the GT syllable annotations over blind HECB for syntax work.
2. **No species-trained HECB/CPC for most animals.** HECB's strongest form (HECB+CPC) needs
   CPC context; CPC is human-only and firewall-blocked on animals. So "HECB on animal X"
   degrades to a cross-domain CRNN + fixed NBD unless a species CRNN is trained.
3. **CRNN is self-sufficient — NBD fusion adds nothing on its home regime.** On discrete
   calls, PCEN+CRNN alone gives F1=0.703; fusing NBD does not improve it. NBD's value inside
   HECB is *internal* sub-call annotation, not call isolation.
4. **CRNN fails on continuous speech.** LibriSpeech F1=0.312 with the CRNN — it is built for
   isolated calls, not running speech. On speech, NBD+CPC calibrator wins (0.836).
5. **Whole-call primary (§30).** Sub-call internal structure does NOT improve call-type
   recovery beyond the whole-call fingerprint (meerkat +0.021 V; marmoset **−0.032 V**). HECB's
   value is structural *annotation* (positional roles, compositional prerequisites §37), not a
   call-type-recovery lever.
6. **Stages 4–5 are roadmap.** Today HECB delivers CallEvents + typed-evidence proposals +
   internal segments. The typed classifier (onset/offset/internal/noise) and semi-Markov
   decoder are not yet wired.

## Empirical headline

- **PCEN + CRNN: F1 = 0.703 (one-to-one) on MeerKAT** — the discrete-call-in-noise regime,
  ~8.9× raw NBD. Closes the over-segmentation gap NBD shows on noisy 8 kHz field audio.
- **LibriSpeech F1 = 0.312** with the CRNN → confirms CRNN is discrete-call-only; speech-like
  audio belongs to NBD+CPC (0.836).
- **Whole-call primary** confirmed on both meerkat and marmoset (§30) — internal segmentation
  is augmentation for motif discovery, not a classification lever.

## Related

- **NBD** (`nbd.md`) — the proposal+calibrator HECB builds on; the speech-like / fallback regime.
- Compositional-candidate mining (`scripts/compositional_candidates_bat.py`, §37) — uses
  HECB/raw-NBD internal units to mine structural prerequisites (T0) + emit T2 stimuli.
- Segmentation router — `configs/segmentation_router.yaml` + `orchestration._router_audit`.
