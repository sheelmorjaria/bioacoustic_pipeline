"""Shared-encoder multi-head CPC — heads-only caller conditioning (§46).

Design (Track-C instrument): encoder + AR context are SHARED and
unconditioned — the "grammar" is population-level. Caller specificity lives
ONLY in per-caller linear prediction heads W_k^(y) routed by caller id —
the "translation" from context to future is caller-specific. This is what
makes cross-model perplexity well-posed: donor head y_B evaluated on
recipient trajectory (context_A, z_A) asks "how well does B's translation
predict A's futures under the SHARED grammar" — a caller-mismatch measure
that pre-AR conditioning (§45's bat_cpc_caller) cannot express.

Two hard facts encoded here:

1. The parent ``CPCModel.compute_loss`` uses the RAW context as queries —
   prediction heads get NO gradient (the documented §45 quirk). A heads-only
   architecture is therefore INERT under the parent loss, so this class
   carries ``head_routing_loss``: per-caller subsets of the batch, each
   evaluated with its OWN head's outputs as queries, reusing the parent
   InfoNCE on each subset. Parent semantics are untouched.

2. Caller labels define the heads => ``contains_caller_supervision``-style
   provenance applies on registration (probe/retrieval/playback-stimulus
   uses; never verdict evidence — circular by construction). UNK / unseen
   callers route to a population head ("unk") that always exists.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig
from bioacoustic_pipeline.embeddings.cpc_trainer import CPCModel

UNK_HEAD = "unk"


class SharedEncoderMultiHeadCPC(CPCModel):
    """CPCModel with per-caller prediction heads and a shared grammar."""

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

        def _head_stack() -> nn.ModuleList:
            stack = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(steps_ahead)])
            for lin in stack:
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            return stack

        self.heads = nn.ModuleDict()
        for caller in range(n_callers):
            self.heads[str(caller)] = _head_stack()
        self.heads[UNK_HEAD] = _head_stack()  # population head: UNK/unseen
        self._n_callers = n_callers

    def _routed_heads(self, caller_ids: Optional[torch.Tensor]):
        key = UNK_HEAD
        if caller_ids is not None and len(caller_ids) > 0:
            ids = {int(c) for c in caller_ids.tolist()}
            if ids.issubset(set(range(self._n_callers))):
                key = None  # mixed batch: route per-row in predict()
        return key

    def forward(
        self,
        audio_frames: torch.Tensor,
        caller_ids: Optional[torch.Tensor] = None,
    ):
        """(B, T, frame) -> (z_latent, context, routed predictions).

        z/context are IDENTICAL across callers (shared path); only the
        prediction heads route. A single-caller batch routes wholesale; a
        mixed batch routes per-row; UNK/unseen ids fall back to the
        population head.
        """
        z_latent, context, _ = CPCModel.forward(self, audio_frames)
        device = context.device
        per_row = torch.full((context.shape[0],), -1, dtype=torch.long, device=device)
        if caller_ids is not None:
            for i, c in enumerate(caller_ids.tolist()):
                per_row[i] = int(c) if 0 <= int(c) < self._n_callers else -1
        predictions = []
        for k in range(self.steps_ahead):
            out = torch.empty_like(context)
            for group_key in {UNK_HEAD} | {str(c) for c in set(per_row.tolist()) if c >= 0}:
                rows = torch.where(per_row == (-1 if group_key == UNK_HEAD else int(group_key)))[0]
                if len(rows) == 0:
                    continue
                out[rows] = self.heads[group_key][k](context[rows])
            predictions.append(out)
        return z_latent, context, predictions

    def head_routing_loss(
        self,
        audio_frames: torch.Tensor,
        caller_ids: torch.Tensor,
        file_ids: torch.Tensor,
        *,
        temperature: float = 0.07,
        n_negatives: Optional[int] = None,
    ) -> torch.Tensor:
        """InfoNCE with per-caller HEAD OUTPUTS as queries.

        Groups the batch by caller; each group's queries are its own head's
        predictions (so heads receive gradient — the parent loss would leave
        them inert); positives are future-aligned latents exactly as in the
        parent implementation, reused per group.
        """
        z_latent, context, _ = CPCModel.forward(self, audio_frames)
        caller_ids = caller_ids.to(context.device)
        total, n_groups = 0.0, 0
        for caller in sorted(set(caller_ids.tolist())):
            rows = torch.where(caller_ids == caller)[0]
            if len(rows) < 2:  # InfoNCE needs in-batch candidates
                continue
            key = str(caller) if 0 <= caller < self._n_callers else UNK_HEAD
            queries = torch.cat(
                [self.heads[key][k](context[rows]) for k in range(self.steps_ahead)],
                dim=0,
            )
            # The UNK population head co-trains on EVERY group (same
            # evidence): held-out recipients route to it for their
            # own-baseline term in cross_perplexity, so it must be a trained
            # population translation, not a random init.
            unk_queries = torch.cat(
                [self.heads[UNK_HEAD][k](context[rows]) for k in range(self.steps_ahead)],
                dim=0,
            )
            z_rep = torch.cat([z_latent[rows]] * self.steps_ahead, dim=0)
            dummy_preds = [None] * self.steps_ahead  # parent loop consumes len(predictions)
            loss = super().compute_loss(
                z_rep, queries, dummy_preds, temperature, n_negatives=n_negatives
            )
            unk_loss = super().compute_loss(
                z_rep, unk_queries, dummy_preds, temperature, n_negatives=n_negatives
            )
            total = total + (loss + unk_loss) * len(rows)
            n_groups += len(rows)
        if n_groups == 0:
            return torch.zeros((), device=context.device)
        return total / n_groups

    def cross_perplexity(
        self,
        audio_frames: torch.Tensor,
        donor_caller: int,
        recipient_caller: Optional[int] = None,
        *,
        temperature: float = 0.07,
        n_negatives: Optional[int] = None,
    ):
        """Delta L (cross - own) under the shared grammar, per batch.

        The donor's head evaluated on the recipient's trajectory vs the
        recipient's own head on the same trajectory. Positive values = the
        donor's translation predicts the recipient's futures WORSE — the
        caller-mismatch signal for Track-C stimulus selection.
        """
        z_latent, context, _ = CPCModel.forward(self, audio_frames)
        donor_key = str(donor_caller) if 0 <= donor_caller < self._n_callers else UNK_HEAD
        own_key = (
            str(recipient_caller)
            if recipient_caller is not None and 0 <= recipient_caller < self._n_callers
            else UNK_HEAD
        )
        z_rep = torch.cat([z_latent] * self.steps_ahead, dim=0)

        def _loss_for(key):
            queries = torch.cat(
                [self.heads[key][k](context) for k in range(self.steps_ahead)], dim=0
            )
            dummy_preds = [None] * self.steps_ahead
            return super().compute_loss(
                z_rep, queries, dummy_preds, temperature, n_negatives=n_negatives
            )

        return float(_loss_for(donor_key) - _loss_for(own_key))


class MultiHeadCPCTrainer:
    """Trainer for SharedEncoderMultiHeadCPC over labeled 5-tuple batches.

    Total loss = grammar term (parent InfoNCE on the raw shared context —
    trains encoder+AR exactly like the self-supervised Arm A) + λ·head term
    (``head_routing_loss`` — trains the per-caller heads and the shared
    translation path). Follows CPCTrainer's optimizer/scheduler/checkpoint
    conventions by composition (CPCTrainer's 2-tuple unpack cannot host the
    labeled batches).
    """

    def __init__(
        self,
        model: SharedEncoderMultiHeadCPC,
        config,
        train_dataset=None,
        val_dataset=None,
        lambda_heads: float = 0.5,
        temperature: float | None = None,
    ):
        import torch  # noqa: PLC0415

        if not isinstance(model, SharedEncoderMultiHeadCPC):
            raise TypeError("MultiHeadCPCTrainer needs a SharedEncoderMultiHeadCPC")
        self.model = model.to(config.device)
        self.config = config
        self.lambda_heads = lambda_heads
        self.temperature = temperature if temperature is not None else config.temperature

        from torch.utils.data import DataLoader

        nw = getattr(config, "num_workers", 0)
        self.train_loader = (
            DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=nw,
                persistent_workers=nw > 0,
            )
            if train_dataset is not None
            else None
        )
        self.val_loader = (
            DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=nw,
                persistent_workers=nw > 0,
            )
            if val_dataset is not None
            else None
        )
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        self.current_epoch = 0
        self.best_loss = float("inf")
        from pathlib import Path  # noqa: PLC0415

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _combined(self, batch):
        import torch  # noqa: PLC0415

        context, future, caller_ids, _context_code, file_ids = batch
        context = context.to(self.config.device)
        future = future.to(self.config.device)
        caller_ids = caller_ids.to(self.config.device)
        file_ids = file_ids.to(self.config.device)
        audio = torch.cat([context, future], dim=1)
        z_latent, ar_context, _ = self.model(audio, caller_ids=caller_ids)
        grammar = self.model.compute_loss(
            z_latent,
            ar_context,
            [None] * self.model.steps_ahead,
            self.temperature,
            n_negatives=self.config.negative_samples,
        )
        heads = self.model.head_routing_loss(
            audio,
            caller_ids,
            file_ids,
            temperature=self.temperature,
            n_negatives=self.config.negative_samples,
        )
        return grammar, heads

    def train_epoch(self):
        import torch  # noqa: PLC0415

        self.model.train()
        totals = {"grammar": 0.0, "heads": 0.0}
        n = 0
        for batch in self.train_loader:
            grammar, heads = self._combined(batch)
            loss = grammar + self.lambda_heads * heads
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
            totals["grammar"] += float(grammar)
            totals["heads"] += float(heads)
            n += 1
        avg = lambda k: totals[k] / n if n else 0.0  # noqa: E731
        return {
            "train_loss": avg("grammar") + self.lambda_heads * avg("heads"),
            "grammar_loss": avg("grammar"),
            "head_loss": avg("heads"),
        }

    def validate(self):
        import torch  # noqa: PLC0415

        if self.val_loader is None:
            return {}
        self.model.eval()
        totals = {"grammar": 0.0, "heads": 0.0}
        n = 0
        with torch.no_grad():
            for batch in self.val_loader:
                grammar, heads = self._combined(batch)
                totals["grammar"] += float(grammar)
                totals["heads"] += float(heads)
                n += 1
        avg = lambda k: totals[k] / n if n else 0.0  # noqa: E731
        return {
            "val_loss": avg("grammar") + self.lambda_heads * avg("heads"),
            "grammar_loss": avg("grammar"),
            "head_loss": avg("heads"),
        }

    def train(self):

        history = {"train_loss": [], "val_loss": []}
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch()
            history["train_loss"].append(train_metrics["train_loss"])
            val_metrics = self.validate()
            if val_metrics:
                history["val_loss"].append(val_metrics["val_loss"])
            self.scheduler.step()
            if val_metrics and val_metrics["val_loss"] < self.best_loss:
                self.best_loss = val_metrics["val_loss"]
                self.save_checkpoint(epoch, best=True)
            elif (epoch % max(self.config.save_every, 1)) == 0:
                self.save_checkpoint(epoch)
            if not val_metrics and (epoch % max(self.config.save_every, 1)) == 0:
                self.save_checkpoint(epoch)
        return history

    def save_checkpoint(self, epoch: int, best: bool = False):
        import torch  # noqa: PLC0415

        from pathlib import Path  # noqa: PLC0415

        out = Path(self.config.checkpoint_dir)
        from bioacoustic_pipeline.embeddings.cpc_trainer import LOSS_VERSION  # noqa: PLC0415

        payload = {
            "epoch": epoch,
            "loss_version": LOSS_VERSION,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
            "n_callers": self.model._n_callers,
        }
        torch.save(payload, out / ("best_model.pt" if best else f"checkpoint_epoch_{epoch}.pt"))

    def load_checkpoint(self, path):
        import torch  # noqa: PLC0415

        ckpt = torch.load(path, map_location=self.config.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.current_epoch = ckpt["epoch"] + 1
        self.best_loss = ckpt["best_loss"]


@torch.no_grad()
def per_frame_head_loss(
    model: SharedEncoderMultiHeadCPC,
    audio_frames: torch.Tensor,
    *,
    head_key: str,
    temperature: float = 0.07,
    n_negatives: Optional[int] = None,
    neg_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-anchor-frame InfoNCE under one head, aligned to context frames.

    Returns ``(T - steps_ahead,)`` — position t is the average over horizons
    k of -log softmax(positive vs shared sampled negatives), with the head's
    k-step prediction as query and the true future latent as positive
    (parent semantics, per-frame reduction).
    """
    z_latent, context, _ = CPCModel.forward(model, audio_frames)
    B, T, hidden = context.shape
    K = model.steps_ahead
    device = context.device
    pool = torch.nn.functional.normalize(z_latent.reshape(-1, hidden), dim=1)
    frame_losses = []
    for k in range(K):
        query = torch.nn.functional.normalize(model.heads[head_key][k](context), dim=1)
        pos = torch.nn.functional.normalize(z_latent, dim=1)
        anchors = T - K  # valid t for every horizon
        q = query[:, :anchors, :]
        positive = torch.cat([pos[:, k + 1 + t, :].unsqueeze(1) for t in range(anchors)], dim=1)
        pos_logit = (q * positive).sum(dim=2, keepdim=True) / temperature
        if neg_idx is None:
            cap = n_negatives if n_negatives is not None else 10
            neg_idx = torch.randint(0, pool.shape[0], (cap,), device=device)
        negs = pool[neg_idx]
        neg_logits = torch.matmul(q, negs.t()) / temperature
        logits = torch.cat([pos_logit, neg_logits], dim=2)
        labels = torch.zeros(B, anchors, dtype=torch.long, device=device)
        flat = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
        ).reshape(B, anchors)
        # valid-anchor reduction over the BATCH axis, per frame: a fully
        # masked anchor would otherwise read as a falsely 'perfect'
        # prediction in the ΔL landscape
        valid = torch.isfinite(logits[:, :, 1:]).any(dim=2).to(flat.dtype)
        per_anchor = (flat * valid).sum(dim=0) / valid.sum(dim=0).clamp(min=1.0)
        frame_losses.append(per_anchor)  # (anchors,)
    return torch.stack(frame_losses).mean(dim=0)  # (T - K,)


@torch.no_grad()
def cross_perplexity_series(
    model: SharedEncoderMultiHeadCPC,
    audio_frames: torch.Tensor,
    *,
    donor_caller: int,
    recipient_caller: Optional[int] = None,
    temperature: float = 0.07,
    n_negatives: Optional[int] = None,
    neg_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Temporal ΔL_CPC series: donor head minus own head, per context frame.

    Positive values localize WHERE the donor's translation fails to predict
    the recipient's futures under the shared grammar — the mismatch spikes
    Phase 3's donor selection filters.
    """
    donor_key = str(donor_caller) if 0 <= donor_caller < model._n_callers else UNK_HEAD
    own_key = (
        str(recipient_caller)
        if recipient_caller is not None and 0 <= recipient_caller < model._n_callers
        else UNK_HEAD
    )
    # SHARED negative draw: donor and own heads are compared on identical
    # evidence (same pool samples), so delta isolates the translation change
    cap = n_negatives if n_negatives is not None else 10
    if neg_idx is None:
        z0, _, _ = CPCModel.forward(model, audio_frames)
        flat = z0.reshape(-1, z0.shape[-1]).shape[0]
        neg_idx = torch.randint(0, flat, (cap,), device=z0.device)
    cross = per_frame_head_loss(
        model,
        audio_frames,
        head_key=donor_key,
        temperature=temperature,
        n_negatives=n_negatives,
        neg_idx=neg_idx,
    )
    own = per_frame_head_loss(
        model,
        audio_frames,
        head_key=own_key,
        temperature=temperature,
        n_negatives=n_negatives,
        neg_idx=neg_idx,
    )
    return cross - own
