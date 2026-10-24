"""Fixed synthetic sequence trainability probe; no speech-study modification."""

import argparse
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import rclat_ctc_learning_diagnostic as core

OUT = Path("outputs/rclat_ctc_synthetic_diagnostic_v1")
CHECKPOINTS = Path("checkpoints/rclat_ctc_synthetic_diagnostic_v1")
# Reuse frozen instrumentation, with new artifact roots and supplied synthetic batches.
core.OUT = OUT
core.CHECKPOINTS = CHECKPOINTS


def verify():
    plan = json.loads((OUT / "execution_plan.json").read_text())
    design = json.loads((OUT / "design.json").read_text())
    for mapping in [plan["dependencies"], design["dependencies"]]:
        for path, digest in mapping.items():
            if core.sha(path) != digest:
                raise ValueError("Dependency changed: " + path)
    core.predecessor.check()
    cohort = json.loads((OUT / "cohort.json").read_text())
    records = cohort["records"]
    messages = set()
    splits = {"training": [], "validation": []}
    for row in records:
        if row["text"] in messages:
            raise ValueError("Repeated fixture message")
        messages.add(row["text"])
        if core.sha(row["fixture_path"]) != row["sha256"]:
            raise ValueError("Fixture hash changed")
        with np.load(row["fixture_path"], allow_pickle=False) as f:
            x = f["features"]
            target = f["target_ids"]
            mask = f["valid"]
            expected = core.encode_text(row["text"]).numpy()
            if (
                x.shape != (row["native_frames"], 64)
                or x.dtype != np.float32
                or mask.shape != x.shape
                or mask.dtype != np.bool_
                or not mask.all()
                or not np.isfinite(x).all()
                or not np.array_equal(expected, target)
            ):
                raise ValueError("Synthetic fixture schema/target mismatch")
            recovered = "".join(
                "ABCD"[int(x[a["start_frame"] : a["stop_frame_exclusive"], :4].mean(0).argmax())]
                for a in row["intervals"]
            )
            if recovered != row["text"]:
                raise ValueError("Known feature code mismatch")
        splits[row["split"]].append({**row, "transcript": row["text"], "speaker": "synthetic"})
    if len(splits["training"]) != 64 or len(splits["validation"]) != 32:
        raise ValueError("Population changed")
    return plan, design, splits


def batcher(device):
    def batch(rows, where=device):
        features = []
        masks = []
        native = []
        targets = []
        for row in rows:
            with np.load(row["fixture_path"], allow_pickle=False) as f:
                features.append(torch.from_numpy(f["features"].copy()))
                masks.append(torch.from_numpy(f["valid"].copy()))
                native.append(int(f["native_length"]))
                targets.append(torch.from_numpy(f["target_ids"].copy()))
        return (
            pad_sequence(features, batch_first=True).to(where),
            pad_sequence(masks, batch_first=True).to(where),
            torch.tensor(native),
        ), targets

    return batch


def run(action, device):
    plan, design, cohort = verify()
    if device != plan["device"]:
        raise ValueError("Device differs from frozen synthetic plan")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)
    torch.xpu.set_device(device)
    name = "preflight" if action == "preflight" else "SYNTHETIC"
    core.write(
        OUT / f"{name}_STARTED.json",
        dict(
            action=action,
            device=device,
            execution_plan_sha256=core.sha(OUT / "execution_plan.json"),
            requested_cache_capacity=16384,
            model_inputs="features/masks/lengthsonly;no generatorintervals",
            stage2_ready=False,
        ),
    )
    batch = batcher(device)
    if action == "preflight":
        return core.preflight(device, plan, cohort, batch, name)
    if not json.loads((OUT / "preflight_result.json").read_text())["passed"]:
        raise RuntimeError("Device parity prerequisite missing/failed")
    model = core.build()
    digest = core.predecessor.initial_digest(model)
    prior = json.loads(Path("outputs/rclat_p10_corrected_v3/mel_initial_state.json").read_text())
    if digest != prior["full_state_sha256"]:
        raise ValueError("Fresh seed0initialisation differs")
    core.write(
        OUT / "initial_state.json", dict(full_state_sha256=digest, checkpoint_loaded=False, seed=0)
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    core.check_parameters_and_optimizer(model, optimizer)
    history = {}
    start = time.monotonic()
    step = 0
    with (OUT / "SYNTHETIC_journal.jsonl").open("x", buffering=1) as journal:

        def record():
            history[step] = core.evaluate(model, batch, cohort, "SYNTHETIC", step, journal)
            exact = {}
            for split in ["training", "validation"]:
                predictions = history[step][split]["predictions"]
                exact[split] = sum(r["reference"] == r["hypothesis"] for r in predictions) / len(
                    predictions
                )
            path = CHECKPOINTS / f"step_{step:04}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise RuntimeError("Existing checkpoint;no overwrite")
            rng = np.random.get_state()
            torch.save(
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    step=step,
                    torch_rng=torch.get_rng_state(),
                    xpu_rng=torch.xpu.get_rng_state_all(),
                    python_rng=random.getstate(),
                    numpy_rng=[rng[0], rng[1].tolist(), rng[2], rng[3], rng[4]],
                    execution_plan_sha256=core.sha(OUT / "execution_plan.json"),
                    diagnostic_only=True,
                    resume_permitted=False,
                ),
                path,
            )
            core.write(
                OUT / f"step_{step:04}.json",
                dict(
                    step=step,
                    training_metrics=history[step]["training"]["metrics"],
                    validation_metrics=history[step]["validation"]["metrics"],
                    exact_message_recovery=exact,
                    checkpoint=str(path),
                    checkpoint_sha256=core.sha(path),
                ),
            )

        record()
        for epoch in range(1, 126):
            rows = core.predecessor.original.epoch_order(cohort["training"], epoch)
            for pos in range(0, 64, 4):
                if time.monotonic() - start > plan["max_seconds"]:
                    raise TimeoutError("Fixed diagnostic walltime bound")
                step += 1
                model.train()
                group = rows[pos : pos + 4]
                journal.write(
                    json.dumps(
                        dict(event="begin", epoch=epoch, step=step, uids=[r["uid"] for r in group])
                    )
                    + "\n"
                )
                for g in optimizer.param_groups:
                    g["lr"] = core.predecessor.original.learning_rate(step, 21120)
                optimizer.zero_grad(set_to_none=True)
                inputs, targets = batch(group)
                logits, lengths = model(*inputs)
                core.assert_lengths(logits, lengths, inputs[2], targets)
                del inputs
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Nonfinite logits")
                loss = core.ctc_objective(logits, lengths, targets)
                loss.backward()
                checked = core.check_gradients(model)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("Nonfinite gradient norm")
                optimizer.step()
                checks = core.check_parameters_and_optimizer(model, optimizer)
                journal.write(
                    json.dumps(
                        dict(
                            event="step_complete",
                            epoch=epoch,
                            step=step,
                            loss=float(loss.detach()),
                            gradient_norm=float(norm),
                            gradients_checked=checked,
                            **checks,
                            **core.predecessor.telemetry(device),
                        )
                    )
                    + "\n"
                )
                del logits, lengths, loss
                if step in design["evaluation_steps"]:
                    record()
        if step != 2000 or set(history) != set(design["evaluation_steps"]):
            raise ValueError("Fixed execution coverage mismatch")
        final = history[2000]
        core.write(
            OUT / "result.json",
            dict(
                completed=True,
                updates=step,
                epochs=125,
                elapsed_seconds=time.monotonic() - start,
                training_fit_indicator_passed=final["training"]["metrics"]["pooled_cer"] < 0.10
                and final["training"]["metrics"]["nonempty_fraction"] >= 0.95,
                final_training_metrics=final["training"]["metrics"],
                final_validation_metrics=final["validation"]["metrics"],
                model_selection=False,
                real_speech_learning_established=False,
                P100_authorized=False,
                P960_authorized=False,
                stage2_ready=False,
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "run"])
    parser.add_argument("--device", choices=["xpu:0", "xpu:1"], default="xpu:0")
    a = parser.parse_args()
    try:
        run(a.action, a.device)
    except Exception:
        path = OUT / f"{a.action}_FAILURE.json"
        if not path.exists():
            core.write(
                path,
                dict(traceback=traceback.format_exc(), automatic_retry=False, stage2_ready=False),
            )
        raise
