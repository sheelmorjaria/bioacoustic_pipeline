# fix: CPCBackbone device-consistency + GPU enablement

## What
- `CPCBackbone.__init__`: after loading the checkpoint, `.to("cuda" if available else "cpu")`
  — CPC now runs on the GPU when present (was CPU-only even on GPU boxes).
- `CPCBackbone.embed_frames`: move the input tensor to the model's device, and `.cpu()`
  before `.numpy()` (a CUDA tensor can't be `.numpy()`-ed — the GPU move surfaced this).

## Why (honest)
A standalone `embed_frames` call did **not** reproduce a crash on this CUDA box (model +
input were both CPU, so it worked). The motivation is **defensive hardening + GPU
enablement**, not a confirmed reproducible bug. The conv2d CUDA/CPU failure observed in
the HECB work was the CRNN detector (fixed in the HECB PR), not CPC. This PR makes CPC
robust to any device configuration and faster on GPU machines.

## Test
`tests/test_cpcbackbone_device.py` — 3 gated smoke tests (embed_frames shape/dtype, embed
mean-pool, short-audio zero-pad). 186 tests pass (183 + 3). src/ ruff-clean.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
