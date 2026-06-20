"""Species acoustic tokenizer — VQ-VAE on a species' own CPC latents.

Same contract as ``AcousticTokenizer`` (CPC-encode -> mean-pool -> normalize ->
VQ-VAE token) but parameterized by the species' OWN CPC + VQ-VAE checkpoints, so
a non-human species gets discrete acoustic-similarity tokens without any
cross-species weight transfer. ``AcousticTokenizer`` is hardwired to the human
pair and firewall-blocked on animals; this adapter is the legal species path.

Both the CPC backbone provenance (e.g. ``bat_cpc``) and the VQ-VAE provenance
(e.g. ``bat_vqvae``) MUST be registered in the ethological firewall with the
target species in ``applies_to_species``; the consumer (run_pipeline / scripts)
checks the firewall before tokenizing, exactly as for ``AcousticTokenizer``.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


class SpeciesAcousticTokenizer:
    """VQ-VAE tokenizer over a species CPC backbone."""

    def __init__(
        self,
        species: str,
        cpc_provenance: str,
        vqvae_provenance: str,
        *,
        cpc_checkpoint: str | Path | None = None,
        vqvae_checkpoint: str | Path | None = None,
    ):
        self.species = species
        self.cpc_provenance = cpc_provenance
        self.provenance_name = vqvae_provenance  # what the firewall gates tokenize on
        self._vqvae_ckpt = vqvae_checkpoint or str(
            REPO_ROOT / "checkpoints" / vqvae_provenance / "best_model.pt"
        )
        self._cpc = None
        self._vq = None
        self._norm_mean = None
        self._norm_std = None
        self._err = None
        try:
            import torch  # noqa: PLC0415

            from bioacoustic_pipeline.embeddings.species_cpc_backbone import (  # noqa: PLC0415
                SpeciesCPCBackbone,
            )
            from bioacoustic_pipeline.tokenization.syntactic_vqvae import (  # noqa: PLC0415
                SyntacticVQVAE,
            )

            self._cpc = SpeciesCPCBackbone(
                name=cpc_provenance, provenance_name=cpc_provenance, checkpoint=cpc_checkpoint
            )
            if not self._cpc.available:
                raise RuntimeError(f"CPC {cpc_provenance!r} unavailable: {self._cpc.load_error}")
            ckpt = torch.load(self._vqvae_ckpt, map_location="cpu", weights_only=False)
            cfg = ckpt["config"]
            self._vq = SyntacticVQVAE(
                cfg["input_dim"], cfg["codebook_size"], cfg["codebook_dim"], cfg["hidden_dim"]
            )
            self._vq.load_state_dict(ckpt["model_state_dict"])
            self._vq.eval()
            self._norm_mean = np.asarray(ckpt["norm_mean"], dtype=np.float32).reshape(-1)
            self._norm_std = np.asarray(ckpt["norm_std"], dtype=np.float32).reshape(-1)
            if self._norm_mean.shape[0] != cfg["input_dim"]:
                raise RuntimeError(
                    f"VQ-VAE norm dim {self._norm_mean.shape[0]} != input_dim {cfg['input_dim']} "
                    f"— CPC/VQ-VAE provenance mismatch"
                )
        except Exception as exc:  # noqa: BLE001
            self._err = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self._vq is not None and self._cpc is not None and self._cpc.available

    @property
    def load_error(self) -> str | None:
        return self._err

    def tokenize(self, audio: np.ndarray, sr: int) -> int:
        """Species-CPC-encode -> mean-pool -> normalize -> VQ-VAE token (-1 if too short)."""
        import torch  # noqa: PLC0415

        latent = self._cpc.embed(audio, sr)
        if latent.shape[0] != self._norm_mean.shape[0]:
            return -1
        normed = (latent - self._norm_mean) / self._norm_std
        with torch.no_grad():
            return int(
                self._vq.tokenize(torch.from_numpy(normed.astype(np.float32)).unsqueeze(0)).item()
            )
