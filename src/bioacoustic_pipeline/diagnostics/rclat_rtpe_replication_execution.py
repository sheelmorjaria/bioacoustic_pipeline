"""Separately versioned staged replication; all seed fits precede any evaluation."""

from datetime import datetime, timezone
import resource
import time
import traceback
import json
from pathlib import Path

import numpy as np
import torch
from .rclat_rtpe_execution import SelectionLedger, _evaluate, _finite, sha, write, save
from ..representation.rclat_rtpe_pool import PoolTemporalPredictionEngine
from ..representation.rclat_rtpe_weighted import (
    WeightedHistoryRidge,
    WeightedTrainingScaler,
    minibatch_mse,
)


def train_seed(
    spec,
    train,
    validation,
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
        generator = torch.Generator(device="cpu").manual_seed(spec["gru"]["sampler_seed"])

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
        context.update(phase="training complete; evaluation globally sealed")
        for path, digest in checkpoint_hashes.items():
            if sha(path) != digest:
                raise ValueError("Checkpoint integrity failure")
        write(
            root / "TRAINING_COMPLETE.json",
            dict(
                completed_updates=ledger.completed,
                elapsed_seconds=time.monotonic() - started,
                selection_sha256=sha(root / "selection.json"),
                scaler_sha256=sha(root / "scaler.pt"),
            ),
        )
        return selection
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


def assert_global_lock(lock_path):
    from .rclat_rtpe_replication import unlock_replication

    path = Path(lock_path)
    if not path.is_file():
        raise ValueError("Evaluation sealed until the global five-seed selection lock exists")
    lock = json.loads(path.read_text())
    selections = {}
    for seed, record in lock["trainings"].items():
        directory = Path(record["path"])
        if (
            sha(directory / "selection.json") != record["selection_sha256"]
            or sha(directory / "TRAINING_COMPLETE.json") != record["complete_sha256"]
        ):
            raise ValueError("Global selection artifact integrity failed")
        selections[seed] = json.loads((directory / "selection.json").read_text())
    unlock_replication(
        selections, pairs=lock["pairs"], updates=lock["updates"], interval=lock["interval"]
    )
    return lock


def evaluate_seed(spec, directory, evaluation_loader, lock_path, *, device, integrity_check=None):
    from .rclat_rtpe_protocol import paired_speaker_summary

    root = Path(directory)
    if (root / "EVALUATION_STARTED.json").exists() or (root / "result.json").exists():
        raise FileExistsError("Existing evaluation attempt preserved; no retry")
    lock = assert_global_lock(lock_path)
    if str(root) not in [v["path"] for v in lock["trainings"].values()]:
        raise ValueError("Seed is not part of the locked study")
    if integrity_check:
        integrity_check()
    selection = json.loads((root / "selection.json").read_text())
    training = json.loads((root / "TRAINING_COMPLETE.json").read_text())
    write(root / "EVALUATION_STARTED.json", dict(global_lock_sha256=sha(lock_path)))
    started = time.monotonic()

    def limits():
        if (
            training["elapsed_seconds"] + time.monotonic() - started
            > spec["runtime"]["max_walltime_seconds"]
        ):
            raise TimeoutError("Frozen per-seed active walltime")
        if (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            > spec["runtime"]["peak_host_RSS_GiB"]
        ):
            raise MemoryError("Frozen RSS limit")

    try:
        for path, digest in selection["checkpoint_hashes"].items():
            if sha(path) != digest:
                raise ValueError("Checkpoint changed before evaluation")
        raw = evaluation_loader()
        moments = torch.load(root / "scaler.pt", map_location="cpu", weights_only=True)
        scaler = WeightedTrainingScaler()
        scaler.mean = moments["mean"]
        scaler.scale = moments["scale"]
        data = {
            **raw,
            "x": scaler.transform(raw["x"], raw["v"]),
            "y": scaler.transform(raw["y"], raw["m"]),
        }
        write(root / "evaluation_origin_ids.json", data["ids"])
        state = torch.load(selection["ridge_checkpoint"], map_location="cpu", weights_only=True)
        ridge = WeightedHistoryRidge(state["alpha"])
        for key, value in state.items():
            setattr(ridge, key, value)
        rs, rp, rl = _evaluate(ridge.predict, data, limits)
        model = PoolTemporalPredictionEngine(
            data["x"].shape[-1],
            hidden=spec["gru"]["hidden"],
            scales=(1, 4, 16),
            horizons=tuple(spec["horizons"]),
        ).to(device)

        def predict(x, v):
            return model(x.to(device), v.to(device))["forecast"][:, -1]

        checkpoint = torch.load(selection["gru_checkpoint"], map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        gs, gp, gl = _evaluate(predict, data, limits)
        final = spec["gru"]["updates"]
        if selection["gru_step"] == final:
            fs, fp, fl = gs, gp, gl
        else:
            checkpoint = torch.load(
                root / f"gru_{final:06d}.pt", map_location="cpu", weights_only=True
            )
            model.load_state_dict(checkpoint["model"])
            model.eval()
            fs, fp, fl = _evaluate(predict, data, limits)
        primary = spec["horizons"].index(spec["primary_horizon"])
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
            targets=data["y"].numpy(),
            observed=data["m"].numpy(),
        )
        write(
            root / "evaluation_losses.json",
            dict(ridge=rl.tolist(), gru_selected=gl.tolist(), gru_final=fl.tolist()),
        )
        if integrity_check:
            integrity_check()
        result = dict(
            completed=True,
            completed_updates=selection["completed_updates"],
            initialization_seed=spec["gru"]["seed"],
            sampler_seed=spec["gru"]["sampler_seed"],
            selection=selection,
            primary=paired,
            ridge=rs,
            gru_selected=gs,
            gru_final=fs,
            final_gru_step=final,
            active_elapsed_seconds=training["elapsed_seconds"] + time.monotonic() - started,
            development_utility_gate="PASS" if paired["development_utility_passed"] else "FAIL",
        )
        limits()
        write(root / "result.json", result)
        write(
            root / "artifact_manifest.json", {str(p): sha(p) for p in root.iterdir() if p.is_file()}
        )
        return result
    except Exception:
        write(
            root / "EVALUATION_FAILURE.json",
            dict(traceback=traceback.format_exc(), automatic_retry=False),
        )
        raise
