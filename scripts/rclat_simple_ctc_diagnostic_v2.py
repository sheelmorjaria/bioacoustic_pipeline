"""Small encoder control; frozen D1 procedure with capped LR and new encoder."""

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import platform
import random
import time
import traceback
import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import rclat_p10_c3_train as predecessor
from bioacoustic_pipeline.diagnostics.frozen_acoustic_transfer import frozen_scale
from bioacoustic_pipeline.diagnostics.rclat import normalize_text
from bioacoustic_pipeline.diagnostics.rclat_ctc_learning import (
    assert_lengths,
    conditioned_posterior,
    edit_counts,
    frame_summary,
)
from bioacoustic_pipeline.representation.rclat_ctc import (
    LABELS,
    ctc_objective,
    decode_logits,
    encode_text,
)
from bioacoustic_pipeline.representation.rclat_simple_ctc import SimpleCTC
from rclat_ctc_lr_diagnostic import learning_rate as capped_rate
from bioacoustic_pipeline.representation.rclat_numerics import (
    check_gradients,
    check_parameters_and_optimizer,
)

OUT = Path("outputs/rclat_simple_ctc_diagnostic_v2")
CHECKPOINTS = Path("checkpoints/rclat_simple_ctc_diagnostic_v2")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(2**20), b""):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2)


def build():
    return SimpleCTC()


def prepare(action, device):
    plan = json.loads((OUT / "execution_plan.json").read_text())
    for path, digest in plan["dependencies"].items():
        if sha(path) != digest:
            raise ValueError("Execution dependency changed: " + path)
    spec, _ = predecessor.check()
    design = json.loads((OUT / "design.json").read_text())
    for path, digest in design["dependencies"].items():
        if sha(path) != digest:
            raise ValueError("Design dependency changed: " + path)
    cohort = json.loads((OUT / "cohort.json").read_text())
    for row in cohort["training"] + cohort["validation"]:
        if (
            sha(row["cache_path"]) != row["cache_sha256"]
            or sha(row["path"]) != row["source_sha256"]
        ):
            raise ValueError("Selected source/cache changed: " + row["uid"])
    if action == "d0":
        for entry in design["D0"]["checkpoints"].values():
            if sha(entry["path"]) != entry["sha256"]:
                raise ValueError("Scientific checkpoint changed")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)
    torch.xpu.set_device(device)
    scaler = json.loads(Path("outputs/rclat_p10/mel_scaler.json").read_text())

    def batch(rows, where=device):
        arrays = [
            frozen_scale(*predecessor.original.sequence(r["uid"], "mel"), scaler) for r in rows
        ]
        values = pad_sequence([torch.from_numpy(x) for x, m in arrays], batch_first=True).to(where)
        masks = pad_sequence([torch.from_numpy(m) for x, m in arrays], batch_first=True).to(where)
        native = torch.tensor([len(x) for x, m in arrays])
        targets = [encode_text(r["transcript"]) for r in rows]
        for row, length, target in zip(rows, native, targets):
            if (
                int(length) != row["actual_feature_frames"]
                or len(target) != row["actual_target_length"]
            ):
                raise ValueError("Actual length differs from fixed cohort")
        return (values, masks, native), targets

    name = f"preflight_xpu{device.split(':')[-1]}" if action == "preflight" else action.upper()
    write(
        OUT / f"{name}_STARTED.json",
        dict(
            action=action,
            device=device,
            plan_sha256=sha(OUT / "execution_plan.json"),
            python=platform.python_version(),
            torch=torch.__version__,
            device_properties=str(torch.xpu.get_device_properties(device)),
            requested_cache_capacity=os.environ.get("ONEDNN_PRIMITIVE_CACHE_CAPACITY"),
            scientific_P10_C3_reopened=False,
        ),
    )
    return plan, design, cohort, batch, name


def preflight(device, plan, cohort, batch, name):
    rows = cohort["training"][:4]
    cpu = build()
    gpu = copy.deepcopy(cpu).to(device)
    inputs, targets = batch(rows, "cpu")
    yc, lc = cpu(*inputs)
    assert_lengths(yc, lc, inputs[2], targets)
    loss_c = ctc_objective(yc, lc, targets)
    loss_c.backward()
    yg, lg = gpu(inputs[0].to(device), inputs[1].to(device), inputs[2])
    assert_lengths(yg, lg, inputs[2], targets)
    loss_g = ctc_objective(yg, lg, targets)
    loss_g.backward()
    check_gradients(gpu)
    numerator = denominator = 0.0
    for p, q in zip(cpu.parameters(), gpu.parameters()):
        a = p.grad.double()
        b = q.grad.detach().cpu().double()
        numerator += float((a - b).square().sum())
        denominator += float(a.square().sum())
    gradient = (numerator / max(denominator, 1e-300)) ** 0.5
    logits = float(
        (yc.detach().double() - yg.detach().cpu().double()).norm() / yc.detach().double().norm()
    )
    loss = abs(float(loss_c.detach()) - float(loss_g.detach())) / max(
        abs(float(loss_c.detach())), 1e-30
    )
    passed = max(gradient, logits, loss) <= plan["parity_tolerance"]
    write(
        OUT / f"{name}_result.json",
        dict(
            passed=passed,
            gradient_relative_L2=gradient,
            logit_relative_L2=logits,
            loss_relative=loss,
            training_updates=0,
            uids=[r["uid"] for r in rows],
        ),
    )
    if not passed:
        raise FloatingPointError("Fresh device parity prerequisite failed")


def evaluate(model, batch, cohort, stage, checkpoint, log):
    model.eval()
    out = OUT / stage / f"step_{checkpoint:04}"
    result = {}
    for split in ["training", "validation"]:
        rows = cohort[split]
        predictions = []
        chars = Counter()
        words = Counter()
        ratios = []
        summaries = []
        for pos in range(0, len(rows), 4):
            group = rows[pos : pos + 4]
            inputs, targets = batch(group)
            with torch.no_grad():
                logits, lengths = model(*inputs)
            assert_lengths(logits, lengths, inputs[2], targets)
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Evaluation logits")
            cpu = logits.detach().cpu()
            native = inputs[2]
            del inputs, logits
            for row, x, length, target, native_length in zip(group, cpu, lengths, targets, native):
                x = x[: int(length)].contiguous()
                p, gamma, stats = conditioned_posterior(x, target)
                posterior = frame_summary(p, gamma, LABELS.index("E"))
                hypothesis = decode_logits(x, len(x))
                reference = normalize_text(row["transcript"])
                hyp = normalize_text(hypothesis)
                rc = reference.replace(" ", "")
                hc = hyp.replace(" ", "")
                char = edit_counts(rc, hc)
                word = edit_counts(reference.split(), hyp.split())
                path = out / split / f"{row['uid']}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    raise RuntimeError("Raw output exists; no overwrite")
                np.savez_compressed(
                    path,
                    logits=x.numpy(),
                    target_ids=target.numpy(),
                    conditioned_occupancy=gamma,
                    native_length=int(native_length),
                    output_length=int(length),
                )
                prediction = dict(
                    uid=row["uid"],
                    speaker=row["speaker"],
                    reference=reference,
                    hypothesis=hyp,
                    reference_characters=len(rc),
                    reference_words=len(reference.split()),
                    character_errors=char["errors"],
                    word_errors=word["errors"],
                    character_decomposition=char,
                    word_decomposition=word,
                    cer=char["errors"] / len(rc),
                    wer=word["errors"] / len(reference.split()),
                    hypothesis_reference_character_length_ratio=len(hc) / len(rc),
                    native_length=int(native_length),
                    output_length=int(length),
                    target_length=len(target),
                    ctc_minimum_frames=len(target) + int((target[1:] == target[:-1]).sum()),
                    frame_statistics=posterior,
                    conditioned_statistics=stats,
                    raw_path=str(path),
                    raw_sha256=sha(path),
                )
                predictions.append(prediction)
                chars.update(hc)
                words.update(hyp.split())
                ratios.append(len(hc) / len(rc))
                summaries.append(posterior)
            del cpu
            log.write(
                json.dumps(
                    dict(
                        event="evaluation_batch_complete",
                        stage=stage,
                        checkpoint=checkpoint,
                        split=split,
                        batch=pos // 4 + 1,
                    )
                )
                + "\n"
            )
        frames = sum(r["valid_frames"] for r in summaries)
        weighted = {
            key: sum(r[key] * r["valid_frames"] for r in summaries) / frames
            for key in [
                "mean_blank_probability",
                "blank_argmax_fraction",
                "mean_E_probability",
                "E_argmax_fraction",
                "conditioned_mean_blank_occupancy",
                "conditioned_mean_E_occupancy",
            ]
        }
        metrics = predecessor.original.aggregate(predictions)
        unit_decomposition = {
            unit: {
                component: float(
                    np.mean(
                        [
                            sum(
                                r[f"{unit}_decomposition"][component]
                                for r in predictions
                                if r["speaker"] == s
                            )
                            / sum(
                                r[
                                    "reference_characters"
                                    if unit == "character"
                                    else "reference_words"
                                ]
                                for r in predictions
                                if r["speaker"] == s
                            )
                            for s in sorted({r["speaker"] for r in predictions})
                        ]
                    )
                )
                for component in ["substitutions", "deletions", "insertions"]
            }
            for unit in ["character", "word"]
        }
        result[split] = dict(
            metrics=metrics,
            speaker_macro_decomposition=unit_decomposition,
            frame_weighted_statistics=weighted,
            valid_frames=frames,
            character_distribution=chars.most_common(),
            top_words=words.most_common(30),
            single_character_word_fraction=sum(v for k, v in words.items() if len(k) == 1)
            / max(1, sum(words.values())),
            hypothesis_reference_character_length_ratio_quantiles=np.quantile(
                ratios, [0, 0.25, 0.5, 0.75, 1]
            ).tolist(),
            predictions=predictions,
        )
    write(out / "evaluation.json", result)
    print(
        json.dumps(
            dict(
                event="evaluation_complete",
                stage=stage,
                checkpoint=checkpoint,
                training_CER=result["training"]["metrics"]["speaker_macro_cer"],
                validation_CER=result["validation"]["metrics"]["speaker_macro_cer"],
            )
        ),
        flush=True,
    )
    return result


def run(action, device):
    plan, design, cohort, batch, name = prepare(action, device)
    if action == "preflight":
        return preflight(device, plan, cohort, batch, name)
    prerequisite = OUT / f"preflight_xpu{device.split(':')[-1]}_result.json"
    if not json.loads(prerequisite.read_text())["passed"]:
        raise RuntimeError("Device preflight not passed")
    started = time.monotonic()
    with (OUT / f"{name}_journal.jsonl").open("x", buffering=1) as journal:
        if action == "d0":
            history = {}
            for epoch, entry in design["D0"]["checkpoints"].items():
                if time.monotonic() - started > plan["D0_max_seconds"]:
                    raise TimeoutError("D0 bound")
                checkpoint = torch.load(entry["path"], map_location="cpu", weights_only=True)
                model = build()
                model.load_state_dict(checkpoint["model"], strict=True)
                model = model.to(device)
                history[epoch] = evaluate(model, batch, cohort, "D0", int(epoch), journal)
                del model, checkpoint
            write(
                OUT / "D0_result.json",
                dict(
                    completed=True,
                    checkpoints=list(history),
                    elapsed_seconds=time.monotonic() - started,
                    model_updates=0,
                    CTC_logit_gradients_used_on_CPU_only=True,
                    stage2_ready=False,
                ),
            )
            return
        if not json.loads((OUT / "D0_result.json").read_text())["completed"]:
            raise RuntimeError("D0 must complete before D1")
        model = build()
        reference = json.loads(
            Path("outputs/rclat_simple_ctc_diagnostic_v2/initial_state_reference.json").read_text()
        )
        digest = predecessor.initial_digest(model)
        if digest != reference["full_state_sha256"]:
            raise ValueError("Fresh initialization differs from frozen simple-encoder seed0 reference")
        write(
            OUT / "D1_initial_state.json",
            dict(full_state_sha256=digest, checkpoint_loaded=False, seed=0),
        )
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        )
        check_parameters_and_optimizer(model, optimizer)
        history = {}

        def record(step):
            history[step] = evaluate(model, batch, cohort, "D1", step, journal)
            folder = CHECKPOINTS / "D1"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"step_{step:04}.pt"
            if path.exists():
                raise RuntimeError("Checkpoint exists")
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
                    execution_plan_sha256=sha(OUT / "execution_plan.json"),
                    diagnostic_only=True,
                    resume_permitted=False,
                ),
                path,
            )
            write(
                OUT / f"D1_step_{step:04}.json",
                dict(
                    step=step,
                    training_metrics=history[step]["training"]["metrics"],
                    validation_metrics=history[step]["validation"]["metrics"],
                    checkpoint=str(path),
                    checkpoint_sha256=sha(path),
                ),
            )

        record(0)
        step = 0
        for epoch in range(1, 126):
            rows = predecessor.original.epoch_order(cohort["training"], epoch)
            for pos in range(0, len(rows), 4):
                if time.monotonic() - started > plan["D1_max_seconds"]:
                    raise TimeoutError("D1 bound")
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
                    g["lr"] = capped_rate(step, "capped250")
                optimizer.zero_grad(set_to_none=True)
                inputs, targets = batch(group)
                logits, lengths = model(*inputs)
                assert_lengths(logits, lengths, inputs[2], targets)
                del inputs
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Training logits")
                loss = ctc_objective(logits, lengths, targets)
                loss.backward()
                checked = check_gradients(model)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("Gradient norm")
                optimizer.step()
                state = check_parameters_and_optimizer(model, optimizer)
                journal.write(
                    json.dumps(
                        dict(
                            event="step_complete",
                            epoch=epoch,
                            step=step,
                            loss=float(loss.detach()),
                            gradient_norm=float(norm),
                            gradients_checked=checked,
                            **state,
                            **predecessor.telemetry(device),
                        )
                    )
                    + "\n"
                )
                del logits, lengths, loss
                if step in design["D1"]["evaluation_steps"]:
                    record(step)
        assert step == 2000 and set(history) == set(design["D1"]["evaluation_steps"])
        final = history[2000]["training"]["metrics"]
        write(
            OUT / "D1_result.json",
            dict(
                completed=True,
                updates=step,
                epochs=125,
                training_fit_indicator_passed=final["speaker_macro_cer"] < 0.10
                and final["nonempty_fraction"] >= 0.95,
                final_training_metrics=final,
                final_validation_metrics=history[2000]["validation"]["metrics"],
                elapsed_seconds=time.monotonic() - started,
                model_selection=False,
                P100_authorized=False,
                P960_authorized=False,
                stage2_ready=False,
            ),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["preflight", "d1"])
    p.add_argument("--device", choices=["xpu:0", "xpu:1"], required=True)
    a = p.parse_args()
    OUT = OUT / a.device.replace(":", "_")
    CHECKPOINTS = CHECKPOINTS / a.device.replace(":", "_")
    try:
        run(a.action, a.device)
    except Exception:
        path = OUT / f"{a.action}_{a.device.replace(':', '_')}_FAILURE.json"
        if not path.exists():
            write(
                path,
                dict(traceback=traceback.format_exc(), automatic_retry=False, stage2_ready=False),
            )
        raise
