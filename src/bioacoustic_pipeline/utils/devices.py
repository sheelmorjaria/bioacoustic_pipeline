"""Device resolution: cuda -> xpu -> cpu, one place.

The repo historically inlined ``"cuda" if torch.cuda.is_available() else "cpu"``
(``TrainingConfig``, ``SpeciesCPCBackbone``, scripts). On Intel XPU machines
that silently runs everything on CPU — the accelerator exists but the idiom
never asks about it. Every new code path resolves its device here; the two
existing inline sites we touch for §45 were migrated; ``CPCBackbone`` (human,
checkpoint-gated) is deliberately left alone until its checkpoint returns.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations


def resolve_device(explicit: str | None = None) -> str:
    """Pick a torch device string: explicit > cuda > xpu > cpu.

    Explicit requests are VALIDATED here — fail fast at startup (a clear
    RuntimeError), not mid-training when the first tensor hits a missing
    device. Torch is imported lazily so core/verdict/diagnostic code stays
    importable without the ``ml`` extras; without torch the answer is
    always ``"cpu"``.
    """
    if explicit:
        _validate_explicit(explicit)
        return explicit
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.xpu.is_available():
            return "xpu"
    except AttributeError:  # non-XPU torch build
        pass
    return "cpu"


def _validate_explicit(explicit: str) -> None:
    backend = explicit.split(":")[0]
    if backend == "cpu":
        return
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            f"device '{explicit}' requested but torch is not installed"
        ) from exc
    available = False
    if backend == "cuda":
        available = torch.cuda.is_available()
    elif backend == "xpu":
        available = getattr(torch.xpu, "is_available", lambda: False)()
    if not available:
        raise RuntimeError(
            f"device '{explicit}' requested but backend '{backend}' is not "
            f"available (torch {torch.__version__}) — refusing to silently "
            f"fall back to CPU"
        )
