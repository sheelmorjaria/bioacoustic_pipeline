"""Module capability registry — which pipeline stages use GPU vs CPU.

Principle: ``Accelerate tensors, not decisions.`` GPU-bound stages are neural
encoders (CPC, BioMAE, AVES) + batched tensor math. CPU-bound stages are
segmentation, DSP, statistics, null models, and verdicts.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleCapability:
    """Whether a pipeline module can/should use GPU."""

    name: str
    can_use_gpu: bool
    preferred_device: str  # "cpu" | "accelerator"
    reason: str


CAPABILITIES: dict[str, ModuleCapability] = {
    # --- CPU-only stages ---
    "corpus_qa": ModuleCapability("corpus_qa", False, "cpu", "I/O and metadata dominated"),
    "nbd": ModuleCapability("nbd", False, "cpu", "DSP segmentation; deterministic"),
    "spectrogram_backbone": ModuleCapability("spectrogram_backbone", False, "cpu", "Deterministic DSP control representation; keep reproducible"),
    "acoustic_fingerprint": ModuleCapability("acoustic_fingerprint", False, "cpu", "Physics features and summary statistics"),
    "functional_alias": ModuleCapability("functional_alias", False, "cpu", "Sparse/count-based statistics and graph clustering"),
    "motif_miner": ModuleCapability("motif_miner", False, "cpu", "Sequence statistics and nulls"),
    "context_binding": ModuleCapability("context_binding", False, "cpu", "Contingency table + permutation null"),
    "sequence_sufficiency": ModuleCapability("sequence_sufficiency", False, "cpu", "Count-based gate"),
    "verdict_engine": ModuleCapability("verdict_engine", False, "cpu", "Schema validation + report assembly"),
    # --- GPU-capable stages ---
    "cpc": ModuleCapability("cpc", True, "accelerator", "Torch neural encoder"),
    "biomae": ModuleCapability("biomae", True, "accelerator", "Torch neural encoder"),
    "aves": ModuleCapability("aves", True, "accelerator", "Transformer embedding backbone"),
    "vqvae": ModuleCapability("vqvae", True, "accelerator", "Batch tokenization benefits from GPU"),
    "beta_vae": ModuleCapability("beta_vae", True, "accelerator", "Affect vector extraction"),
}


def module_device(module_name: str, runtime_device: str) -> str:
    """Resolve the device for a module given the runtime config.

    Returns "cuda"/"mps" if the module is GPU-capable AND the runtime selected a GPU;
    otherwise "cpu".
    """
    cap = CAPABILITIES.get(module_name)
    if cap is None:
        return "cpu"
    if cap.can_use_gpu and runtime_device in ("cuda", "mps"):
        return runtime_device
    return "cpu"


def module_plan(runtime_device: str) -> dict[str, str]:
    """Full module→device plan for embedding in reports."""
    return {name: module_device(name, runtime_device) for name in CAPABILITIES}
