"""Frozen-contract RTPE execution primitives; scientific entrypoint is separately pinned."""

from datetime import datetime, timezone
import hashlib
import json
import resource
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .rclat_rtpe_protocol import paired_speaker_summary, select_minimum
from ..representation.rclat_rtpe_pool import PoolTemporalPredictionEngine
from ..representation.rclat_rtpe_weighted import (
    WeightedHistoryRidge,
    WeightedTrainingScaler,
    minibatch_mse,
)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def save(path, value):
    with Path(path).open("xb") as f:
        torch.save(value, f)


class SelectionLedger:
    def __init__(self, *, updates, interval, alphas):
        if updates <= 0 or interval <= 0 or updates % interval or not alphas:
            raise ValueError("Complete fixed validation schedule required")
        self.updates = updates
        self.interval = interval
        self.alphas = list(alphas)
        self.completed = 0
        self.validation = []

    def advance(self, step):
        if step != self.completed + 1 or step > self.updates:
            raise ValueError("Updates must be sequential and within frozen budget")
        self.completed = step

    def validate(self, step, score):
        if (
            step != self.completed
            or step % self.interval
            or step in [s for s, _ in self.validation]
        ):
            raise ValueError("Validation must follow the registered completed update")
        select_minimum([(step, score)])
        self.validation.append((step, score))

    def lock(self, ridge_scores):
        expected = list(range(self.interval, self.updates + 1, self.interval))
        if (
            self.completed != self.updates
            or [s for s, _ in self.validation] != expected
            or [a for a, _ in ridge_scores] != self.alphas
        ):
            raise ValueError("Selection/evaluation sealed until every fit and update completes")
        return dict(
            ridge_alpha=select_minimum(ridge_scores),
            gru_step=select_minimum(self.validation),
            completed_updates=self.completed,
            validation_steps=expected,
        )


def group_scores(losses, ids):
    a = torch.as_tensor(losses, dtype=torch.float64)
    if (
        a.ndim != 2
        or len(a) != len(ids)
        or not len(a)
        or not torch.isfinite(a).all()
        or (a < 0).any()
        or len({(r["uid"], r["origin"]) for r in ids}) != len(ids)
    ):
        raise ValueError("Unique finite nonnegative per-origin horizon losses required")
    records = {}
    owners = {}
    for row, loss in zip(ids, a):
        uid, speaker = row["uid"], row["speaker"]
        if uid in owners and owners[uid] != speaker:
            raise ValueError("Recording identity conflict")
        owners[uid] = speaker
        records.setdefault(uid, []).append(loss)
    rec = {uid: torch.stack(values).mean(0) for uid, values in records.items()}
    speakers = {}
    for uid, loss in rec.items():
        speakers.setdefault(owners[uid], []).append(loss)
    sp = {s: torch.stack(values).mean(0) for s, values in speakers.items()}
    return dict(
        macro=torch.stack(list(sp.values())).mean(0).tolist(),
        speakers={s: v.tolist() for s, v in sp.items()},
        recordings={uid: v.tolist() for uid, v in rec.items()},
    )


def load_dataset(records, *, split, history, horizons, selection):
    if split == "test" and (not selection or not selection.get("completed_updates")):
        raise ValueError("Internal evaluation caches remain sealed until selection is locked")
    rows = [r for r in records if r["split"] == split]
    if not rows:
        raise ValueError("Empty source partition")
    xs = []
    vs = []
    ys = []
    ms = []
    weights = []
    ids = []
    for r in rows:
        if sha(r["cache_path"]) != r["cache_sha256"]:
            raise ValueError("Cache hash mismatch: " + r["uid"])
        with np.load(r["cache_path"], allow_pickle=False) as f:
            mel = f["mel"]
            valid = f["mel_valid"]
        if (
            mel.ndim != 2
            or valid.shape != mel.shape
            or valid.dtype != np.dtype(bool)
            or not np.isfinite(mel[valid]).all()
        ):
            raise ValueError("Invalid source feature support")
        origins = r["selected_origins"]
        target_indices = np.array(origins)[:, None] + np.array(horizons)[None]
        if (
            hashlib.sha256(valid[target_indices].tobytes()).hexdigest()
            != r["selected_target_mask_sha256"]
        ):
            raise ValueError("Frozen target support mismatch")
        for t in origins:
            if (
                t - history + 1 < 0
                or t + max(horizons) >= len(mel)
                or not valid[t - history + 1 : t + max(horizons) + 1].any(1).all()
            ):
                raise ValueError("Frozen origin no longer eligible")
            xs.append(mel[t - history + 1 : t + 1])
            vs.append(valid[t - history + 1 : t + 1])
            ys.append(mel[np.array(horizons) + t])
            ms.append(valid[np.array(horizons) + t])
            weights.append(r["origin_weight_within_split"])
            ids.append(dict(uid=r["uid"], speaker=r["speaker"], origin=t))
    return dict(
        x=torch.from_numpy(np.stack(xs)),
        v=torch.from_numpy(np.stack(vs)),
        y=torch.from_numpy(np.stack(ys)),
        m=torch.from_numpy(np.stack(ms)),
        weights=torch.tensor(weights, dtype=torch.float64),
        ids=ids,
    )


def _finite(tensor, label):
    if not torch.isfinite(tensor).all():
        raise ValueError("Nonfinite " + label)


def _evaluate(predict, data, check_limits):
    values = []
    with torch.no_grad():
        for start in range(0, len(data["x"]), 128):
            check_limits()
            value = (
                predict(data["x"][start : start + 128], data["v"][start : start + 128])
                .detach()
                .cpu()
            )
            _finite(value, "forecast")
            values.append(value)
    predictions = torch.cat(values)
    mask = data["m"]
    counts = mask.sum(-1)
    if not (counts > 0).all():
        raise ValueError("Each evaluated horizon requires observed targets")
    _finite(data["y"][mask], "observed targets")
    errors = (
        torch.where(mask, predictions.double(), 0) - torch.where(mask, data["y"].double(), 0)
    ).square()
    per_origin = errors.sum(-1) / counts
    return group_scores(per_origin, data["ids"]), predictions, per_origin


def execute_comparison(
    spec,
    train,
    validation,
    evaluation_loader,
    directory,
    *,
    device,
    protocol_sha256,
    execution_sha256="synthetic-test",
    started_at=None,
    integrity_check=None,
):
    """Run one new attempt; failures never resume, retry or unlock evaluation.

    Production entrypoint fixes spec, device and data inventory through its
    execution manifest. Tests supply explicitly synthetic fixtures on CPU.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic() if started_at is None else started_at
    context = dict(phase="initialise", step=0, indices=[])
    write(
        root / "STARTED.json",
        dict(
            protocol_sha256=protocol_sha256,
            execution_sha256=execution_sha256,
            device=device,
            started_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )

    def limits():
        if time.monotonic() - started > spec["runtime"]["max_walltime_seconds"]:
            raise TimeoutError("Frozen walltime limit")
        if (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            > spec["runtime"]["peak_host_RSS_GiB"]
        ):
            raise MemoryError("Frozen host RSS limit")

    try:
        torch.set_num_threads(2)
        torch.manual_seed(spec["gru"]["seed"])
        if device.startswith("xpu"):
            torch.xpu.set_device(device)
            torch.xpu.manual_seed_all(spec["gru"]["seed"])
        limits()
        ledger = SelectionLedger(
            updates=spec["gru"]["updates"],
            interval=spec["gru"]["validation_every"],
            alphas=spec["ridge"]["alpha_candidates"],
        )
        if spec["gru"]["checkpoint_every"] != spec["gru"]["validation_every"]:
            raise ValueError("Frozen checkpoint and validation schedules must agree")
        primary = spec["horizons"].index(spec["primary_horizon"])
        context["phase"] = "training-only scaling"
        scaler = WeightedTrainingScaler().fit(train["x"], train["v"], train["weights"])

        def scaled(data):
            return {
                **data,
                "x": scaler.transform(data["x"], data["v"]),
                "y": scaler.transform(data["y"], data["m"]),
            }

        train = scaled(train)
        validation = scaled(validation)
        save(root / "scaler.pt", dict(mean=scaler.mean, scale=scaler.scale))
        write(root / "train_origin_ids.json", train["ids"])
        write(root / "validation_origin_ids.json", validation["ids"])
        ridge_scores = []
        ridge_files = {}
        checkpoint_hashes = {}
        for candidate, alpha in enumerate(spec["ridge"]["alpha_candidates"]):
            context.update(phase="ridge", alpha=alpha)
            limits()
            if integrity_check:
                integrity_check()
            ridge = WeightedHistoryRidge(alpha).fit(
                train["x"], train["v"], train["y"], train["m"], train["weights"]
            )
            limits()
            if ridge.maximum_normal_residual > spec["runtime"]["post_fit_ridge_residual_max"]:
                raise ValueError("Frozen Ridge residual threshold")
            score, _, losses = _evaluate(ridge.predict, validation, limits)
            ridge_scores.append((alpha, score["macro"][primary]))
            state = {
                k: getattr(ridge, k)
                for k in [
                    "alpha",
                    "history_shape",
                    "output_shape",
                    "active_columns",
                    "coefficients",
                    "intercept",
                    "factorizations",
                    "maximum_normal_residual",
                ]
            }
            path = root / f"ridge_{candidate:02d}.pt"
            save(path, state)
            ridge_files[alpha] = path
            checkpoint_hashes[str(path)] = sha(path)
            write(
                root / f"ridge_{candidate:02d}_validation.json",
                dict(
                    alpha=alpha,
                    score=score,
                    per_origin_losses=losses.tolist(),
                    normal_residual=ridge.maximum_normal_residual,
                ),
            )
            del ridge
        # Fixed fresh initialisation independent of any preprocessing/solver RNG.
        torch.manual_seed(spec["gru"]["seed"])
        model = PoolTemporalPredictionEngine(
            train["x"].shape[-1],
            hidden=spec["gru"]["hidden"],
            scales=(1, 4, 16),
            horizons=tuple(spec["horizons"]),
        ).to(device)
        for p in model.order.parameters():
            p.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=spec["gru"]["learning_rate"],
            betas=tuple(spec["gru"]["betas"]),
            eps=spec["gru"]["eps"],
            weight_decay=spec["gru"]["weight_decay"],
        )
        generator = torch.Generator(device="cpu").manual_seed(2026102301)

        def neural_predict(x, v):
            return model(x.to(device), v.to(device))["forecast"][:, -1]

        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        save(root / "initial_model.pt", initial)
        context.update(phase="gru", alpha=None)
        with (root / "steps.jsonl").open("x", buffering=1) as journal:
            for step in range(1, spec["gru"]["updates"] + 1):
                limits()
                model.train()
                indices = torch.randint(
                    len(train["x"]), (spec["gru"]["batch_origins"],), generator=generator
                )
                context.update(step=step, indices=indices.tolist())
                optimizer.zero_grad(set_to_none=True)
                prediction = neural_predict(train["x"][indices], train["v"][indices])
                _finite(prediction, "training forecasts")
                loss = minibatch_mse(
                    prediction,
                    train["y"][indices].to(device),
                    train["m"][indices].to(device),
                    train["weights"],
                    indices,
                )
                _finite(loss, "loss")
                loss.backward()
                for p in model.parameters():
                    if p.requires_grad:
                        if p.grad is None:
                            raise ValueError("Missing active parameter gradient")
                        _finite(p.grad, "gradient before clipping")
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    spec["gru"]["gradient_clip_norm"],
                )
                _finite(norm, "gradient clip norm")
                optimizer.step()
                for p in model.parameters():
                    _finite(p, "parameter after update")
                for state in optimizer.state.values():
                    for value in state.values():
                        if isinstance(value, torch.Tensor):
                            _finite(value, "optimizer state")
                ledger.advance(step)
                limits()
                journal.write(
                    json.dumps(
                        dict(
                            step=step,
                            indices=indices.tolist(),
                            loss=float(loss.detach().cpu()),
                            gradient_norm=float(norm.detach().cpu()),
                            learning_rate=optimizer.param_groups[0]["lr"],
                            elapsed_seconds=time.monotonic() - started,
                            host_peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                            / 2**20,
                        )
                    )
                    + "\n"
                )
                if step % spec["gru"]["validation_every"] == 0:
                    if integrity_check:
                        integrity_check()
                    model.eval()
                    score, _, losses = _evaluate(neural_predict, validation, limits)
                    ledger.validate(step, score["macro"][primary])
                    checkpoint = dict(
                        step=step,
                        model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        optimizer=optimizer.state_dict(),
                        sampler_rng=generator.get_state(),
                        cpu_rng=torch.get_rng_state(),
                        xpu_rng=torch.xpu.get_rng_state(device)
                        if device.startswith("xpu")
                        else None,
                        protocol_sha256=protocol_sha256,
                        execution_sha256=execution_sha256,
                    )
                    path = root / f"gru_{step:06d}.pt"
                    save(path, checkpoint)
                    checkpoint_hashes[str(path)] = sha(path)
                    write(
                        root / f"gru_{step:06d}_validation.json",
                        dict(step=step, score=score, per_origin_losses=losses.tolist()),
                    )
                    print(
                        f"RTPE update {step}/{spec['gru']['updates']}; validation80ms={score['macro'][primary]:.6g}",
                        flush=True,
                    )
        limits()
        if integrity_check:
            integrity_check()
        selection = ledger.lock(ridge_scores)
        selection.update(
            protocol_sha256=protocol_sha256,
            execution_sha256=execution_sha256,
            ridge_checkpoint=str(ridge_files[selection["ridge_alpha"]]),
            gru_checkpoint=str(root / f"gru_{selection['gru_step']:06d}.pt"),
            checkpoint_hashes=checkpoint_hashes,
        )
        write(root / "selection.json", selection)
        context.update(phase="sealed evaluation opened after complete selection")
        for path, digest in checkpoint_hashes.items():
            if sha(path) != digest:
                raise ValueError("Checkpoint integrity failure")
        evaluation = scaled(evaluation_loader(selection))
        write(root / "evaluation_origin_ids.json", evaluation["ids"])
        state = torch.load(selection["ridge_checkpoint"], map_location="cpu", weights_only=True)
        ridge = WeightedHistoryRidge(state["alpha"])
        for k, v in state.items():
            setattr(ridge, k, v)
        rs, rp, rl = _evaluate(ridge.predict, evaluation, limits)
        checkpoint = torch.load(selection["gru_checkpoint"], map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        gs, gp, gl = _evaluate(neural_predict, evaluation, limits)
        final_step = spec["gru"]["updates"]
        if selection["gru_step"] == final_step:
            fs, fp, fl = gs, gp, gl
        else:
            checkpoint = torch.load(
                root / f"gru_{final_step:06d}.pt", map_location="cpu", weights_only=True
            )
            model.load_state_dict(checkpoint["model"])
            model.eval()
            fs, fp, fl = _evaluate(neural_predict, evaluation, limits)
        paired = paired_speaker_summary(
            {s: v[primary] for s, v in rs["speakers"].items()},
            {s: v[primary] for s, v in gs["speakers"].items()},
            seed=spec["uncertainty"]["seed"],
            replicates=spec["uncertainty"]["replicates"],
        )
        np.savez_compressed(
            root / "evaluation_forecasts.npz",
            ridge=rp.numpy(),
            gru_selected=gp.numpy(),
            gru_final=fp.numpy(),
            targets=evaluation["y"].numpy(),
            observed=evaluation["m"].numpy(),
        )
        write(
            root / "evaluation_losses.json",
            dict(ridge=rl.tolist(), gru_selected=gl.tolist(), gru_final=fl.tolist()),
        )
        limits()
        result = dict(
            completed=True,
            completed_updates=ledger.completed,
            selection=selection,
            primary=paired,
            ridge=rs,
            gru_selected=gs,
            gru_final=fs,
            final_gru_step=final_step,
            elapsed_seconds=time.monotonic() - started,
            protocol_sha256=protocol_sha256,
            execution_sha256=execution_sha256,
            development_utility_gate="PASS" if paired["development_utility_passed"] else "FAIL",
            P100_authorized=False,
            P960_authorized=False,
            stage2_ready=False,
        )
        if integrity_check:
            integrity_check()
        write(root / "result.json", result)
        write(
            root / "artifact_manifest.json", {str(p): sha(p) for p in root.iterdir() if p.is_file()}
        )
        return result
    except Exception:
        write(
            root / "FAILURE.json",
            dict(
                context=context,
                traceback=traceback.format_exc(),
                protocol_sha256=protocol_sha256,
                execution_sha256=execution_sha256,
                gate="UNEVALUATED_INCOMPLETE",
                automatic_retry=False,
            ),
        )
        raise
