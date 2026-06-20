"""Runtime layer — device management + module capability registry.

Usage:
    from bioacoustic_pipeline.runtime import resolve_device, module_plan
    cfg = resolve_device("auto")
    plan = module_plan(cfg.device)

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from bioacoustic_pipeline.runtime.capabilities import (
    CAPABILITIES,
    ModuleCapability,
    module_device,
    module_plan,
)
from bioacoustic_pipeline.runtime.device import DeviceConfig, resolve_device

__all__ = [
    "DeviceConfig",
    "resolve_device",
    "CAPABILITIES",
    "ModuleCapability",
    "module_device",
    "module_plan",
]
