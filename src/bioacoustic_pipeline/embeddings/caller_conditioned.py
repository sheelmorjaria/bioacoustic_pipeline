"""Caller-conditioned CPC — Arm B of §45 (supervised metric learning).

Honest framing (encoded in the firewall registry entry ``bat_cpc_caller``):
injecting caller IDs into training makes this a SUPERVISED speaker-embedding
model, not a self-supervised representation. It can never be *evidence* for
caller-structure claims — it was optimized to produce exactly that structure
(circularity) — so it is registered ``allowed_use: [probe, retrieval]`` and
``contains_caller_supervision: true``; the EmbeddingBank hard-fails it for
EMBEDDING, and verdict_weight/tokenize are forbidden regardless of registry
edits.

Design (see §45 plan):

- Conditioning enters PRE-AR: ``z' = z + E[caller]``; the AR context runs over
  ``z'``. The InfoNCE pool/positives stay the RAW latents ``z`` — the
  self-supervised term cannot be trivially solved by matching caller vectors.
- Parent semantics preserved, including the known quirk that the per-step
  Linear predictor heads receive no gradient (queries are the raw AR
  context); documented here, deliberately not "fixed".
- Caller term: window anchors ``u_i = normalize(mean_t c_t)``; one positive
  sampled among same-caller DIFFERENT-file windows (same-file pairs are
  masked out — they share channel, not just caller); in-batch cross-caller
  windows are the hard negatives. ``L = L_cpc + λ_c · L_caller``.
- UNK callers: ``caller_ids=None`` adds nothing — held-out callers degrade
  gracefully to the unconditioned path (designed, not a bug).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from bioacoustic_pipeline.embeddings.backbones import Backbone
from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig
from bioacoustic_pipeline.embeddings.cpc_trainer import CPCModel, CPCTrainer, TrainingConfig
from bioacoustic_pipeline.utils.devices import resolve_device

REPO_ROOT = Path(__file__).resolve().parents[3]


class CallerConditionedCPC(CPCModel):
    """CPCModel with a caller embedding injected before the AR context net."""

    def __init__(
        self,
        encoder_config: Optional[EncoderConfig] = None,
        ar_config: Optional[dict] = None,
        steps_ahead: int = 5,
        n_callers: int = 0,
    ):
        super().__init__(
            encoder_config=encoder_config, ar_config=ar_config, steps_ahead=steps_ahead
        )
        hidden = encoder_config.hidden_dim if encoder_config is not None else 128
        self.caller_embedding = torch.nn.Embedding(n_callers, hidden) if n_callers > 0 else None
        if self.caller_embedding is not None:
            torch.nn.init.normal_(self.caller_embedding.weight, std=0.02)

    def forward(
        self,
        audio_frames: torch.Tensor,
        caller_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """(B, T, frame) -> (raw z_latent, conditioned context, predictions).

        ``caller_ids=None`` (or an uninitialized embedding) reproduces the
        parent forward exactly.
        """
        batch_size, seq_len, frame_size = audio_frames.shape
        audio_flat = audio_frames.reshape(-1, 1, frame_size)
        z_flat = self.encoder(audio_flat)
        z_latent = z_flat.reshape(batch_size, seq_len, -1, z_flat.shape[-1]).mean(dim=2)

        z_for_ar = z_latent
        if caller_ids is not None and self.caller_embedding is not None:
            # One-hot matmul instead of nn.Embedding's fused lookup: identical
            # semantics (row sum), but the backward is a plain GEMM — the
            # EmbeddingBackward kernel faults on XPU with real batches
            # ("negative dimension -1", torch 2.14.1+xpu).
            n = self.caller_embedding.num_embeddings
            one_hot = F.one_hot(caller_ids.clamp(min=0), num_classes=n).to(z_latent.dtype)
            z_for_ar = z_latent + (one_hot @ self.caller_embedding.weight).unsqueeze(1)

        context = self.ar_model(z_for_ar)
        predictions = [predictor(context) for predictor in self.predictors]
        return z_latent, context, predictions

    def caller_contrastive_loss(
        self,
        context: torch.Tensor,
        caller_ids: torch.Tensor,
        file_ids: torch.Tensor,
        *,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """Window-level caller term with same-file masking (see module docs).

        Vectorized via ``gather``/``where`` only — no boolean advanced
        indexing in the autograd path (an XPU backward bug bit the original
        ``nonzero``-indexed version; this formulation is backend-safe).
        """
        anchors = F.normalize(context.mean(dim=1), dim=-1)
        sims = anchors @ anchors.t() / temperature
        caller_ids = caller_ids.to(anchors.device)
        file_ids = file_ids.to(anchors.device)
        same_caller = caller_ids.unsqueeze(0) == caller_ids.unsqueeze(1)
        diff_file = file_ids.unsqueeze(0) != file_ids.unsqueeze(1)
        pos_mask = same_caller & diff_file  # genuine: same caller, other file
        neg_ok = ~same_caller  # hard negatives: in-batch cross-caller
        n = sims.shape[0]
        rand = torch.rand(n, n, device=sims.device)
        pos_idx = (rand * pos_mask).argmax(dim=1)  # one sampled positive/anchor
        has_pos = pos_mask.any(dim=1) & neg_ok.any(dim=1)
        s_pos = sims.gather(1, pos_idx.unsqueeze(1)).squeeze(1)
        neg_inf = torch.finfo(sims.dtype).min
        neg_sims = torch.where(neg_ok, sims, torch.full_like(sims, neg_inf))
        logits = torch.cat([s_pos.unsqueeze(1), neg_sims], dim=1)  # (n, n+1)
        targets = torch.zeros(n, dtype=torch.long, device=sims.device)
        per_row = F.cross_entropy(logits, targets, reduction="none")
        denom = has_pos.sum().clamp(min=1)
        return (per_row * has_pos).sum() / denom


class CallerCPCTrainer(CPCTrainer):
    """CPCTrainer over labeled 5-tuple batches: L_cpc + λ_c·L_caller."""

    def __init__(
        self,
        model: CallerConditionedCPC,
        config: TrainingConfig,
        train_dataset=None,
        val_dataset=None,
        lambda_caller: float = 0.5,
        caller_temperature: float = 0.1,
        n_callers: int = 0,
    ):
        if not isinstance(model, CallerConditionedCPC):
            raise TypeError("CallerCPCTrainer needs a CallerConditionedCPC")
        super().__init__(model, config, train_dataset=train_dataset, val_dataset=val_dataset)
        self.lambda_caller = lambda_caller
        self.caller_temperature = caller_temperature
        if n_callers and model.caller_embedding is not None:
            if n_callers > model.caller_embedding.num_embeddings:
                raise ValueError(
                    f"batch carries caller ids up to {n_callers} but the model "
                    f"embedding table holds {model.caller_embedding.num_embeddings}"
                )

    def _combined(self, batch):
        context, future, caller_ids, _context_code, file_ids = batch
        context = context.to(self.device)
        future = future.to(self.device)
        caller_ids = caller_ids.to(self.device)
        file_ids = file_ids.to(self.device)
        audio = torch.cat([context, future], dim=1)
        z_latent, ar_context, predictions = self.model(audio, caller_ids=caller_ids)
        cpc = self.model.compute_loss(
            z_latent,
            ar_context,
            predictions,
            self.config.temperature,
            n_negatives=self.config.negative_samples,
        )
        caller = self.model.caller_contrastive_loss(
            ar_context, caller_ids, file_ids, temperature=self.caller_temperature
        )
        return cpc, caller

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        totals = {"cpc": 0.0, "caller": 0.0}
        n = 0
        for batch in self.train_loader:
            cpc, caller = self._combined(batch)
            loss = cpc + self.lambda_caller * caller
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
            totals["cpc"] += float(cpc)
            totals["caller"] += float(caller)
            n += 1
        avg = lambda k: totals[k] / n if n else 0.0  # noqa: E731
        return {
            "train_loss": avg("cpc") + self.lambda_caller * avg("caller"),
            "cpc_loss": avg("cpc"),
            "caller_loss": avg("caller"),
        }

    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        totals = {"cpc": 0.0, "caller": 0.0}
        n = 0
        with torch.no_grad():
            for batch in self.val_loader:
                cpc, caller = self._combined(batch)
                totals["cpc"] += float(cpc)
                totals["caller"] += float(caller)
                n += 1
        avg = lambda k: totals[k] / n if n else 0.0  # noqa: E731
        return {
            "val_loss": avg("cpc") + self.lambda_caller * avg("caller"),
            "cpc_loss": avg("cpc"),
            "caller_loss": avg("caller"),
        }


def _detect_architecture(sd: dict) -> Tuple[EncoderConfig, int, int]:
    """Conv shapes -> EncoderConfig + steps_ahead + n_callers (checkpoint probe)."""
    channels, kernels = [], []
    for idx in (0, 3, 6):
        key = f"encoder.encoder.{idx}.weight"
        if key not in sd:
            raise ValueError(f"checkpoint missing {key} — cannot auto-detect architecture")
        w = sd[key]
        channels.append(int(w.shape[0]))
        kernels.append(int(w.shape[2]))
    hidden = (
        int(sd["encoder.encoder.8.weight"].shape[0])
        if "encoder.encoder.8.weight" in sd
        else channels[-1]
    )
    steps_ahead = len({k.split(".")[1] for k in sd if k.startswith("predictors.")})
    n_callers = (
        int(sd["caller_embedding.weight"].shape[0]) if "caller_embedding.weight" in sd else 0
    )
    return (
        EncoderConfig(
            sample_rate=16000,
            frame_size_ms=10,
            hidden_dim=hidden,
            num_channels=tuple(channels),
            kernel_sizes=tuple(kernels),
            strides=(2, 2, 1),
        ),
        steps_ahead,
        n_callers,
    )


class CallerConditionedBackbone(Backbone):
    """Restricted-use adapter for the caller-conditioned checkpoint.

    ``embed`` pools the conditioned CONTEXT (mean over time) with
    ``caller_ids=None`` — the UNK path — so held-out callers degrade to the
    unconditioned representation by design.
    """

    view_class = "learned"

    def __init__(
        self,
        name="bat_cpc_caller",
        provenance_name="bat_cpc_caller",
        checkpoint=None,
        embed_dim=256,
    ):
        self.name = name
        self.provenance_name = provenance_name
        self._model = None
        self._err: str | None = None
        self.embed_dim = embed_dim
        ckpt = (
            Path(checkpoint) if checkpoint else REPO_ROOT / "checkpoints" / name / "best_model.pt"
        )
        try:
            import torch  # noqa: PLC0415  (module already torch-bound; mirror SpeciesCPCBackbone)

            sd_all = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = sd_all["model_state_dict"]
            enc, steps_ahead, n_callers = _detect_architecture(sd)
            self.embed_dim = enc.hidden_dim
            model = CallerConditionedCPC(
                encoder_config=enc, steps_ahead=max(steps_ahead, 2), n_callers=n_callers
            )
            model_keys = set(model.state_dict().keys())
            missing = model_keys - set(sd.keys())
            if missing:
                raise ValueError(
                    f"checkpoint missing required model keys (would leave random "
                    f"weights): {sorted(missing)[:5]}"
                )
            model.load_state_dict(sd, strict=False)
            model.eval()
            self._model = model.to(resolve_device())
        except Exception as exc:  # noqa: BLE001
            self._err = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._err

    def _frames(self, audio: np.ndarray, sr: int):
        frame = max(int(sr * 0.01), 1)
        audio = np.asarray(audio, dtype=np.float64)
        n = len(audio) // frame
        if n < 2:
            return None
        x = audio[: n * frame].reshape(n, frame)
        peak = np.abs(x).max()
        if peak > 0:
            x = x / peak
        return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import torch  # noqa: PLC0415

        if self._model is None:
            return np.zeros(self.embed_dim, dtype=np.float32)
        x = self._frames(audio, sr)
        if x is None:
            return np.zeros(self.embed_dim, dtype=np.float32)
        with torch.no_grad():
            x = x.to(next(self._model.parameters()).device)
            _, context, _ = self._model(x, caller_ids=None)
            return context.squeeze(0).mean(dim=0).cpu().numpy().astype(np.float32)

    def embed_frames(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import torch  # noqa: PLC0415

        if self._model is None:
            return np.zeros((2, self.embed_dim), dtype=np.float32)
        x = self._frames(audio, sr)
        if x is None:
            return np.zeros((2, self.embed_dim), dtype=np.float32)
        with torch.no_grad():
            x = x.to(next(self._model.parameters()).device)
            _, context, _ = self._model(x, caller_ids=None)
            return context.squeeze(0).cpu().numpy().astype(np.float32)
