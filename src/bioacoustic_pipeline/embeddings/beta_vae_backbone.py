"""β-VAE backbone — CPC latent → 16-D affect vector (the affective dual stream).

The affective complement to the VQ-VAE syntactic token: where VQ-VAE collapses to
discrete acoustic-similarity classes, the β-VAE learns a continuous disentangled
affective/prosodic latent. Human-only (trained on LibriSpeech CPC latents).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bioacoustic_pipeline.embeddings.backbones import Backbone

REPO_ROOT = Path(__file__).resolve().parents[3]


class BetaVAEBackbone(Backbone):
    """Segment audio → CPC mean-pool → normalize → β-VAE → 16-D affect vector."""

    name = "beta_vae"
    provenance_name = "human_beta_vae"
    embed_dim = 16
    view_class = "learned"

    def __init__(self, checkpoint: str | Path | None = None):
        self._vae_ckpt = checkpoint or str(
            REPO_ROOT / "checkpoints" / "human_beta_vae" / "best_model.pt"
        )
        self._vae = None
        self._cpc = None
        self._norm_mean = None
        self._norm_std = None
        self._err = None
        try:
            import torch  # noqa: PLC0415

            from bioacoustic_pipeline.embeddings.beta_vae import BetaVAE  # noqa: PLC0415
            from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig  # noqa: PLC0415
            from bioacoustic_pipeline.embeddings.cpc_trainer import CPCModel  # noqa: PLC0415

            # β-VAE
            ckpt = torch.load(self._vae_ckpt, map_location="cpu", weights_only=False)
            cfg = ckpt["config"]
            self._vae = BetaVAE(
                input_dim=cfg["input_dim"], latent_dim=cfg["latent_dim"],
                hidden_dim=cfg["hidden_dim"], beta=cfg["beta"],
            )
            self._vae.load_state_dict(ckpt["model_state_dict"])
            self._vae.eval()
            self._norm_mean = ckpt["norm_mean"]
            self._norm_std = ckpt["norm_std"]
            self.embed_dim = cfg["latent_dim"]
            # CPC encoder (cached — loaded once, reused per segment)
            enc = EncoderConfig(
                sample_rate=16000, frame_size_ms=10, hidden_dim=256,
                num_channels=(128, 256, 256), kernel_sizes=(5, 5, 3), strides=(2, 2, 1),
            )
            self._cpc = CPCModel(encoder_config=enc, ar_config={}, steps_ahead=2)
            self._cpc.load_state_dict(
                torch.load(
                    str(REPO_ROOT / "checkpoints" / "human_cpc_phonetic" / "best_model.pt"),
                    map_location="cpu", weights_only=False,
                )["model_state_dict"]
            )
            self._cpc.eval()
        except Exception as exc:  # noqa: BLE001
            self._err = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self._vae is not None

    @property
    def load_error(self) -> str | None:
        return self._err

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Segment audio → CPC mean-pool 256D → normalize → β-VAE → 16-D affect."""
        import torch  # noqa: PLC0415

        frame = max(int(sr * 0.01), 1)
        n_frames = len(audio) // frame
        if n_frames < 2:
            return np.zeros(self.embed_dim, dtype=np.float32)
        frames = audio[: n_frames * frame].reshape(1, n_frames, frame)
        with torch.no_grad():
            z_latent, _, _ = self._cpc(torch.from_numpy(frames.astype(np.float32)))
            latent = z_latent.squeeze(0).mean(dim=0)
            normed = (latent - torch.from_numpy(self._norm_mean)) / torch.from_numpy(self._norm_std)
            mu, _ = self._vae.encode(normed.unsqueeze(0))
        return mu.squeeze(0).numpy().astype(np.float32)

    def affect_from_latent(self, latent: np.ndarray) -> np.ndarray:
        """Direct: precomputed 256-D CPC latent → 16-D affect vector."""
        import torch  # noqa: PLC0415

        with torch.no_grad():
            normed = (torch.from_numpy(latent.astype(np.float32))
                      - torch.from_numpy(self._norm_mean)) / torch.from_numpy(self._norm_std)
            mu, _ = self._vae.encode(normed.reshape(1, -1))
            return mu.reshape(-1).numpy().astype(np.float32)
