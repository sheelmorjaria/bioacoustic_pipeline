"""Backbone adapters: audio segment -> fixed-length embedding.

CPC (in-house, human-only) + BioMAE (in-house, bat-only) + AVES (foundation,
via `avex`). Each adapter loads its weights lazily and reports availability —
if a checkpoint / optional dep is missing it is skipped, never silently used.
Provenance names match configs/model_metadata.json (the firewall registry).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

# backbones.py -> embeddings -> bioacoustic_pipeline -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]


class Backbone(ABC):
    """audio -> embedding; reports whether it loaded."""

    name: str
    provenance_name: str  # key in configs/model_metadata.json
    embed_dim: int

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @property
    @abstractmethod
    def load_error(self) -> str | None: ...

    @abstractmethod
    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray: ...


class CPCBackbone(Backbone):
    name = "cpc"
    provenance_name = "human_cpc_phonetic"
    embed_dim = 256

    def __init__(self, checkpoint: str | Path | None = None):
        self._checkpoint = checkpoint or str(REPO_ROOT / "checkpoints" / "human_cpc_phonetic" / "best_model.pt")
        self._model = None
        self._err = None
        try:
            import torch  # noqa: PLC0415
            from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig  # noqa: PLC0415
            from bioacoustic_pipeline.embeddings.cpc_trainer import CPCModel  # noqa: PLC0415

            enc = EncoderConfig(
                sample_rate=16000, frame_size_ms=10, hidden_dim=256,
                num_channels=(128, 256, 256), kernel_sizes=(5, 5, 3), strides=(2, 2, 1),
            )
            m = CPCModel(encoder_config=enc, ar_config={}, steps_ahead=2)
            ckpt = torch.load(self._checkpoint, map_location="cpu", weights_only=False)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            self._model = m
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
            z, _, _ = self._model(torch.from_numpy(frames.astype(np.float32)))
        return z.squeeze(0).mean(dim=0).numpy().astype(np.float32)


class BioMAEBackbone(Backbone):
    name = "biomae"
    provenance_name = "biomae"
    embed_dim = 112

    def __init__(self, checkpoint: str | Path | None = None):
        self._checkpoint = checkpoint or str(REPO_ROOT / "checkpoints" / "onnx" / "biomae.onnx")
        self._sess = None
        self._in_name = None
        self._err = None
        try:
            import onnxruntime as ort  # noqa: PLC0415
            from bioacoustic_pipeline.embeddings.bio_spectrogram import (  # noqa: PLC0415
                SpectrogramConfig,
                UltrasonicSpectrogram,
            )

            self._sess = ort.InferenceSession(
                str(self._checkpoint), providers=["CPUExecutionProvider"]
            )
            self._in_name = self._sess.get_inputs()[0].name
            self._spec = UltrasonicSpectrogram(
                config=SpectrogramConfig(sample_rate=16000, n_fft=1024, hop_length=512)
            )
        except Exception as exc:  # noqa: BLE001
            self._err = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self._sess is not None

    @property
    def load_error(self) -> str | None:
        return self._err

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import torch  # noqa: PLC0415
        from scipy.ndimage import zoom  # noqa: PLC0415

        wav = torch.FloatTensor(audio).unsqueeze(0)
        with torch.no_grad():
            spec = self._spec(wav, return_db=True).squeeze(0).numpy()
        spec = zoom(spec, (40 / max(spec.shape[0], 1), 512 / max(spec.shape[1], 1)), order=1)
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)
        feats = self._sess.run(
            None, {self._in_name: spec.astype(np.float32).reshape(1, 1, 40, 512)}
        )[0].flatten()
        if len(feats) < self.embed_dim:
            feats = np.pad(feats, (0, self.embed_dim - len(feats)))
        return feats[: self.embed_dim].astype(np.float32)


class AVESBackbone(Backbone):
    """Foundation AVES v2 (esp-aves2-eat-bio) via `avex`. Gated on the dep."""

    name = "aves"
    provenance_name = "aves"
    embed_dim = 768

    def __init__(self, hf_repo: str = "EarthSpeciesProject/esp-aves2-eat-bio"):
        self._hf_repo = hf_repo
        self._encoder = None
        self._err = None
        try:
            import avex  # noqa: F401, PLC0415
        except ImportError as exc:
            self._err = (
                "avex not installed (pip install avex; pulls TensorFlow + torch "
                f"upgrades — use an isolated env). {exc}"
            )
            return
        try:
            from avex.models import build_model_from_config  # noqa: PLC0415

            self._encoder = build_model_from_config(hf_repo=self._hf_repo)
        except Exception as exc:  # noqa: BLE001
            self._err = (
                f"avex importable but load failed ({type(exc).__name__}: {exc}); "
                "verify the avex loader API for this checkpoint."
            )

    @property
    def available(self) -> bool:
        return self._encoder is not None

    @property
    def load_error(self) -> str | None:
        return self._err

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return np.asarray(self._encoder.encode(audio, sr)).reshape(-1).astype(np.float32)
