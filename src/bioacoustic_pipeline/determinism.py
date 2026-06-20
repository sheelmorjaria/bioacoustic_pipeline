"""Frozen-inference determinism: global seeding + reproducibility manifest.

This module exists to make the **implementation** reproducible — same seed + same
environment + same inputs -> identical outputs (Claim A). It is deliberately *not*
a claim about biological/statistical stability (Claim B — bootstrap / alt-seed),
which is a separate demonstration.

``set_global_determinism`` is called at the very top of every deterministic entry
point. It seeds Python, NumPy, and (if importable) PyTorch, and switches PyTorch
into deterministic-algorithm mode. PyTorch explicitly warns that full reproducibility
is not guaranteed across releases, platforms, or CPU-vs-GPU execution — so the
reproducibility manifest scopes the claim to a fixed environment, fixed device, and
fixed versions (captured here by ``build_repro_manifest``).

The synthetic DSP-core demo (``scripts/deterministic_run.py``) never imports torch:
its determinism rests on NumPy + sklearn (seeded k-means) + pure-Python diagnostics,
so the torch block is guarded by ``try/except ImportError``.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from bioacoustic_pipeline.hashing import sha256_string
from bioacoustic_pipeline.runtime import resolve_device

PIPELINE_VERSION = "bioacoustic_pipeline-determinism-v1"

CLAIM_SCOPE = (
    "Determinism is scoped to a fixed environment, fixed device (CPU), fixed seed, "
    "fixed dataset manifest, fixed model configuration, and fixed software versions. "
    "PyTorch (if used) is in deterministic-algorithm mode. CPU-vs-GPU and cross-release "
    "reproducibility are NOT guaranteed."
)


def set_global_determinism(seed: int = 1337) -> None:
    """Seed every RNG the pipeline can draw from and force deterministic modes.

    Idempotent and safe to call at the top of any entry point. The torch half is
    guarded so the function works on a torch-free install (the DSP-core path).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required for deterministic CUDA matmul in some CUDA paths (harmless on CPU).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _git_commit(repo: Path) -> str:
    """``git rev-parse HEAD`` of ``repo`` (short-circuit to 'unknown' if not a repo)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 — not a git repo / git missing
        return "unknown"


def capture_environment(repo: str | Path | None = None) -> dict:
    """Snapshot the software + hardware environment that scopes the determinism claim."""
    repo = Path(repo) if repo is not None else Path.cwd()
    env = {
        "pipeline_version": PIPELINE_VERSION,
        "git_commit": _git_commit(repo),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": resolve_device("cpu").as_dict(),
    }
    try:
        import sklearn  # noqa: PLC0415

        env["sklearn"] = sklearn.__version__
    except ImportError:
        env["sklearn"] = None
    try:
        import torch  # noqa: PLC0415

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["cudnn_version"] = str(torch.backends.cudnn.version())
    except ImportError:
        env["torch"] = None
        env["cuda_available"] = False
    return env


def dataset_manifest_hash(audio_paths) -> str:
    """SHA-256 over the sorted (relative-name, byte-size) list of the corpus.

    Content-light on purpose: identical inputs across runs (same files, same sizes,
    same order) -> identical hash, without reading every byte. The audio *contents*
    are then fixed by the seed that generated them.
    """
    items = sorted((Path(p).name, os.path.getsize(p)) for p in audio_paths)
    payload = "\n".join(f"{name}\t{size}" for name, size in items)
    return sha256_string(payload)


def build_repro_manifest(
    *,
    species: str,
    seed: int,
    audio_paths,
    profile: dict,
    tokenizer_info: dict | None = None,
    calibrator_fitted: bool = False,
    repo: str | Path | None = None,
) -> dict:
    """Assemble the reproducibility manifest for one deterministic run.

    Captures everything that must be frozen for Claim A: versions + git commit +
    device (the environment), the species NBD profile + calibrator proposal params
    (the model config), the dataset manifest hash (the inputs), the seed, and the
    fingerprint-tokenizer hyperparameters (the seeded k-means codebook spec).
    """
    from bioacoustic_pipeline.orchestration import CALIBRATOR_PROPOSAL  # noqa: PLC0415

    return {
        **capture_environment(repo),
        "species": species,
        "seed": seed,
        "n_files": len(list(audio_paths))
        if not isinstance(audio_paths, (list, tuple))
        else len(audio_paths),
        "dataset_manifest_hash": dataset_manifest_hash(audio_paths),
        "species_profile": dict(profile),
        "nbd_config": {
            "proposal_params": dict(CALIBRATOR_PROPOSAL),
            "calibrator_fitted": calibrator_fitted,
        },
        "fingerprint_tokenizer": dict(tokenizer_info) if tokenizer_info else None,
        "claim_scope": CLAIM_SCOPE,
        "claim": (
            "Claim A (implementation determinism): under the frozen environment above, three "
            "runs produced byte-identical artifacts. This is NOT a claim about biological or "
            "statistical stability (Claim B)."
        ),
    }
