"""Central device manager — auto/cuda/mps/cpu resolution.

No module does ad-hoc ``device = "cuda"``. Every torch-backed module asks the runtime.

Policy: ``resolve_device("auto")`` picks CUDA if available, MPS next, CPU last.
Each device config carries AMP flag, dtype, worker count, pin_memory — the data
GPU-bound modules need to batch efficiently.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceConfig:
    """Resolved runtime device + execution parameters."""

    device: str  # "cuda" | "mps" | "cpu"  (string for JSON-serialisability)
    use_amp: bool
    dtype: str  # "float16" | "float32"
    num_workers: int
    pin_memory: bool

    @property
    def is_gpu(self) -> bool:
        return self.device in ("cuda", "mps")

    def as_dict(self) -> dict:
        return {
            "device": self.device,
            "use_amp": self.use_amp,
            "dtype": self.dtype,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "is_gpu": self.is_gpu,
        }


def resolve_device(prefer: str = "auto") -> DeviceConfig:
    """Resolve the runtime device.

    Args:
        prefer: "auto" (default; CUDA > MPS > CPU), "cuda", "mps", or "cpu".

    Returns:
        A :class:`DeviceConfig`. Raises ``RuntimeError`` if a specific device is
        requested but unavailable.
    """
    if prefer == "cpu":
        return _cpu_config()

    if prefer in ("auto", "cuda", "mps"):
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            return _cpu_config()

        if prefer in ("auto", "cuda") and torch.cuda.is_available():
            return _cuda_config()

        if prefer in ("auto", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return _mps_config()

        if prefer in ("cuda", "mps"):
            raise RuntimeError(f"{prefer!r} requested but not available")

    return _cpu_config()


def _cuda_config() -> DeviceConfig:
    return DeviceConfig(
        device="cuda",
        use_amp=True,
        dtype="float16",
        num_workers=max(4, (os.cpu_count() or 4) // 2),
        pin_memory=True,
    )


def _mps_config() -> DeviceConfig:
    return DeviceConfig(
        device="mps",
        use_amp=False,
        dtype="float32",
        num_workers=max(2, (os.cpu_count() or 4) // 2),
        pin_memory=False,
    )


def _cpu_config() -> DeviceConfig:
    return DeviceConfig(
        device="cpu",
        use_amp=False,
        dtype="float32",
        num_workers=max(2, (os.cpu_count() or 4) // 2),
        pin_memory=False,
    )
