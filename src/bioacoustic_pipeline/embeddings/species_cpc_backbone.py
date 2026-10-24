"""Species CPC backbone — generic adapter for any species' CPC checkpoint.

Same contract as ``CPCBackbone`` (human) but parameterized by checkpoint path +
provenance, so bat_cpc / marmoset_cpc / any future species CPC can be used without
a dedicated subclass. Each must be registered in the firewall with the correct
``applies_to_species``.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bioacoustic_pipeline.embeddings.backbones import Backbone
from bioacoustic_pipeline.utils.devices import resolve_device

REPO_ROOT = Path(__file__).resolve().parents[3]


class SpeciesCPCBackbone(Backbone):
    """CPC backbone for any species — checkpoint + provenance passed at init."""

    view_class = "learned"

    def __init__(
        self,
        name: str,
        provenance_name: str,
        checkpoint: str | Path | None = None,
        embed_dim: int = 256,
    ):
        self.name = name
        self.provenance_name = provenance_name
        self.embed_dim = embed_dim
        self._checkpoint = checkpoint or str(REPO_ROOT / "checkpoints" / name / "best_model.pt")
        self._model = None
        self._err = None
        try:
            import torch  # noqa: PLC0415

            from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig  # noqa: PLC0415
            from bioacoustic_pipeline.embeddings.cpc_trainer import CPCModel  # noqa: PLC0415

            # Auto-detect full conv architecture from checkpoint weights
            ckpt = torch.load(self._checkpoint, map_location="cpu", weights_only=False)
            sd = ckpt["model_state_dict"]
            # Conv1d weights: (out_channels, in_channels, kernel_size) at indices 0, 3, 6
            num_channels = []
            kernel_sizes = []
            for idx in (0, 3, 6):
                key = f"encoder.encoder.{idx}.weight"
                if key in sd and sd[key].dim() == 3:
                    num_channels.append(sd[key].shape[0])
                    kernel_sizes.append(sd[key].shape[2])
            if len(num_channels) != 3:
                raise ValueError(
                    f"could not detect 3 conv layers from checkpoint (found {len(num_channels)})"
                )
            # Final projection (encoder.encoder.8) gives the embedding dim
            final_key = "encoder.encoder.8.weight"
            hidden_dim = sd[final_key].shape[0] if final_key in sd else num_channels[-1]
            self.embed_dim = hidden_dim
            enc = EncoderConfig(
                sample_rate=16000,
                frame_size_ms=10,
                hidden_dim=hidden_dim,
                num_channels=tuple(num_channels),
                kernel_sizes=tuple(kernel_sizes),
                strides=(2, 2, 1),  # strides don't affect weight shapes — any valid combo loads
            )
            self._model = CPCModel(encoder_config=enc, ar_config={}, steps_ahead=2)
            # Tolerate extra training-only predictor heads (e.g. predictors.2..5 from a
            # checkpoint trained with steps_ahead > 2) — they're unused by the encoder-only
            # embed/embed_frames path. But FAIL if any model key is MISSING from the checkpoint
            # (that would silently leave random weights on a real layer).
            model_keys = set(self._model.state_dict().keys())
            missing = model_keys - set(sd.keys())
            if missing:
                raise ValueError(
                    f"checkpoint missing required model keys (would leave random weights): "
                    f"{sorted(missing)[:5]}"
                )
            self._model.load_state_dict(sd, strict=False)
            self._model.eval()
            # Use the available accelerator + keep model/input on the same device
            # (mirrors CPCBackbone: avoids device input-vs-weight mismatches).
            # resolve_device() also sees XPU — the old cuda-else-cpu idiom ran
            # CPU silently on Intel-GPU machines.
            self._model = self._model.to(resolve_device())
        except Exception as exc:  # noqa: BLE001
            self._err = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._err

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import torch  # noqa: PLC0415

        frame = max(int(sr * 0.01), 1)
        n_frames = len(audio) // frame
        if n_frames < 2:
            return np.zeros(self.embed_dim, dtype=np.float32)
        frames = audio[: n_frames * frame].reshape(1, n_frames, frame)
        with torch.no_grad():
            x = torch.from_numpy(frames.astype(np.float32))
            x = x.to(next(self._model.parameters()).device)
            z, _, _ = self._model(x)
        return z.squeeze(0).mean(dim=0).cpu().numpy().astype(np.float32)

    def embed_frames(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Per-frame CPC latents (n_frames, embed_dim) at 10 ms hop."""
        import torch  # noqa: PLC0415

        frame = max(int(sr * 0.01), 1)
        n_frames = len(audio) // frame
        if n_frames < 2:
            return np.zeros((2, self.embed_dim), dtype=np.float32)
        frames = audio[: n_frames * frame].reshape(1, n_frames, frame)
        with torch.no_grad():
            x = torch.from_numpy(frames.astype(np.float32))
            x = x.to(next(self._model.parameters()).device)
            z, _, _ = self._model(x)
        return z.squeeze(0).cpu().numpy().astype(np.float32)
