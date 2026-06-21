# HECB-Seg: hierarchical internal-boundary annotator + pipeline integration

## Review around these claims
- **HECB is an internal-boundary annotator, not a replacement for CRNN event detection.**
  PCEN+CRNN is the authoritative call-event detector; HECB annotates typed internal
  sub-call structure. Snapping is OFF by default — the CRNN's threshold-crossing
  onset/offset (~21 ms MAE) is already more precise than any 20 ms-quantized NBD proposal.
- **`run_pipeline(segmentation="hecb")` preserves CRNN timing and adds typed internal
  boundaries.** Each call becomes a whole-call `Segment` carrying `.internal_segments`.
- **Fallback to NBD is explicit and traceable.** If torch or the species' `checkpoints/hecb/`
  is unavailable, `run_pipeline` falls back to NBD and records it in `trace["hecb"]`.
- **`DualStreamSegment` hierarchy remains backward compatible.** New `parent_call_id` /
  `sub_segments` default empty; `build_dual_stream` behavior unchanged.
- **Whole/sub/pooled ablation is now standardized** as `representation.subcall` (a reusable,
  tested module); `hecb_downstream.py` is a thin wrapper.

## Commits
1. `59fca37` — HECB-Seg as hierarchical internal-boundary annotator (decoder + scripts + LS mechanistic/router validation)
2. `b8df64a` — standardize whole/sub-call/pooled ablation as a representation layer
3. `a81c492` — thread InternalSegments into DualStreamSegment + `segmentation="hecb"`

## Empirical results (honest, falsifiable)
- **Segmentation (MeerKAT, 2000/500, 50 ep):** no-snap HECB preserves CRNN timing exactly
  (identical onset/offset F1 + MAE) + 213 calls / 793 internal boundaries. Aggressive AND
  conservative snapping both coarsen MAE (3-point sweep).
- **Downstream (MeerKAT, 1500 clips):** pooled (whole+sub) V=0.274 > whole V=0.246 >
  sub-call-alone V=0.142. Sub-call features augment whole-call recovery when pooled.
- **LibriSpeech mechanistic (400/100):** oracle-event HECB recovers in-word phonemes
  F1=0.636 (P=0.78 R=0.54) — mechanistic justification for combined/multi-resolution F1.
- **LibriSpeech router (400/100, 40 ep):** CRNN word F1=0.59 ≪ NBD+CPC 0.836; HECB-on-CRNN
  internal 0.409 ≪ oracle 0.636 — CRNN is the speech bottleneck, not the internal mechanism.
  Segmentation router (NBD+calibrator for speech, PCEN+CRNN+HECB for calls-in-noise) necessary.

## Verification
- **199 tests pass** (183 prior + 11 subcall + 5 hecb-pipeline). `src/` ruff-clean.

## Caveat (NOT in scope for this PR)
A pre-existing latent bug was surfaced (not introduced here): on **GPU boxes** the human
CPC-calibrator path hits a CUDA/CPU mismatch in `CPCBackbone.embed_frames`. The existing
suite passes because it avoids that path; the new HECB pipeline tests use `calibrator=False`
to sidestep it. **Tracked as a separate follow-up** (`fix/cpcbackbone-device-consistency`),
not folded into this PR so the review stays focused on the HECB architectural change.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
