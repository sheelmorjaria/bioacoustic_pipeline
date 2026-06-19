"""Multi-view latent layer: acoustic token + affect + functional-alias role.

Per segment this emits the discrete acoustic token (VQ-VAE on CPC latents) and
the continuous affect vector. The functional-alias ROLE is NOT a per-segment
encoding — it is assigned at corpus level by ``diagnostics.alias_detector``
(two tokens doing the same job), so it is a slot filled downstream.

Design principle: the VQ-VAE token is an ACOUSTIC-SIMILARITY class, never word
identity (the Semantic Ceiling). The alias role is the safe, unsupervised
functional layer.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class MultiViewLatent:
    """One segment's multi-view representation."""

    acoustic_token: int
    affect_vector: np.ndarray | None = None
    alias_role: int | None = None  # filled corpus-level by diagnostics
    metadata: dict = field(default_factory=dict)


class AcousticTokenizer:
    """VQ-VAE acoustic tokenizer on CPC latents (human). Guarded on checkpoints."""

    provenance_name = "human_vqvae_large"

    def __init__(self, cpc_checkpoint: str | Path | None = None,
                 vqvae_checkpoint: str | Path | None = None):
        self._cpc_ckpt = cpc_checkpoint or str(
            REPO_ROOT / "checkpoints" / "human_cpc_phonetic" / "best_model.pt"
        )
        self._vqvae_ckpt = vqvae_checkpoint or str(
            REPO_ROOT / "checkpoints" / "human_vqvae_large" / "best_model.pt"
        )
        self._cpc = None
        self._vq = None
        self._norm_mean = None
        self._norm_std = None
        self._err = None
        try:
            import torch  # noqa: PLC0415
            from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig  # noqa: PLC0415
            from bioacoustic_pipeline.embeddings.cpc_trainer import CPCModel  # noqa: PLC0415
            from bioacoustic_pipeline.tokenization.syntactic_vqvae import (  # noqa: PLC0415
                SyntacticVQVAE,
            )

            enc = EncoderConfig(
                sample_rate=16000, frame_size_ms=10, hidden_dim=256,
                num_channels=(128, 256, 256), kernel_sizes=(5, 5, 3), strides=(2, 2, 1),
            )
            self._cpc = CPCModel(encoder_config=enc, ar_config={}, steps_ahead=2)
            self._cpc.load_state_dict(
                torch.load(self._cpc_ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
            )
            self._cpc.eval()
            ckpt = torch.load(self._vqvae_ckpt, map_location="cpu", weights_only=False)
            cfg = ckpt["config"]
            self._vq = SyntacticVQVAE(
                cfg["input_dim"], cfg["codebook_size"], cfg["codebook_dim"], cfg["hidden_dim"]
            )
            self._vq.load_state_dict(ckpt["model_state_dict"])
            self._vq.eval()
            self._norm_mean = ckpt["norm_mean"]
            self._norm_std = ckpt["norm_std"]
        except Exception as exc:  # noqa: BLE001
            self._err = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self._vq is not None

    @property
    def load_error(self) -> str | None:
        return self._err

    def tokenize(self, audio: np.ndarray, sr: int) -> int:
        """CPC-encode -> mean-pool 256D -> normalize -> VQ-VAE token."""
        import torch  # noqa: PLC0415

        frame = max(int(sr * 0.01), 1)
        n_frames = len(audio) // frame
        if n_frames < 2:
            return -1
        frames = audio[: n_frames * frame].reshape(1, n_frames, frame)
        with torch.no_grad():
            z, _, _ = self._cpc(torch.from_numpy(frames.astype(np.float32)))
        latent = z.squeeze(0).mean(dim=0).numpy()
        normed = (latent - self._norm_mean) / self._norm_std
        with torch.no_grad():
            return int(
                self._vq.tokenize(torch.from_numpy(normed.astype(np.float32)).unsqueeze(0)).item()
            )
