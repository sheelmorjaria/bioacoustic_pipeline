"""Durable, process-bounded operator stage diagnostics; no production guard edits."""

import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

from .rclat_mask_operator import reference_selection


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


class StageJournal:
    def __init__(self, path):
        self.file = Path(path).open("x", buffering=1)
        self.sequence = 0

    def event(self, stage, event, **fields):
        self.sequence += 1
        self.file.write(
            json.dumps(
                dict(sequence=self.sequence, stage=stage, event=event, **fields), allow_nan=False
            )
            + "\n"
        )
        self.file.flush()
        os.fsync(self.file.fileno())

    def step(self, name, operation):
        self.event(name, "started")
        start = time.monotonic()
        try:
            value = operation()
            if isinstance(value, torch.Tensor) and value.device.type == "xpu":
                torch.xpu.synchronize(value.device)
            info = (
                dict(
                    shape=list(value.shape),
                    stride=list(value.stride()),
                    storage_offset=value.storage_offset(),
                    dtype=str(value.dtype),
                    device=str(value.device),
                )
                if isinstance(value, torch.Tensor)
                else {}
            )
            self.event(name, "completed", elapsed_seconds=time.monotonic() - start, **info)
            return value
        except Exception:
            self.event(name, "failed", traceback=traceback.format_exc())
            raise

    def close(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()


def run_bounded(command, log_path, *, timeout, cleanup=3):
    start = time.monotonic()
    timed_out = False
    unreaped = False
    with Path(log_path).open("xb") as output:
        process = subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            try:
                process.wait(timeout=cleanup)
            except subprocess.TimeoutExpired:
                unreaped = True
        return dict(
            pid=process.pid,
            returncode=process.returncode,
            timed_out=timed_out,
            unreaped_after_kill=unreaped,
            elapsed_seconds=time.monotonic() - start,
        )


class StageMismatch(ValueError):
    def __init__(self, stage):
        super().__init__("Round-trip discrepancy: " + stage)
        self.stage = stage


def worker_case(values, observed, layout, operation, directory, *, device="xpu:0"):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    write(root / "STARTED.json", dict(layout=layout, operation=operation, device=device))
    journal = StageJournal(root / "stages.jsonl")
    started = time.monotonic()
    operation_started = False
    result = dict(
        layout=layout, operation=operation, status="EXCEPTION", input_roundtrip_verified=False
    )

    def snapshot(stage, tensor, expected, strict=True):
        actual = journal.step(stage, lambda: tensor.detach().cpu().clone())
        with (root / (stage + ".pt")).open("xb") as f:
            torch.save(dict(actual=actual, expected=expected), f)
            f.flush()
            os.fsync(f.fileno())
        equal = bool(
            np.array_equal(actual.numpy(), expected.detach().cpu().numpy(), equal_nan=True)
        )
        journal.event(
            stage,
            "comparison",
            exact_equal=equal,
            actual_elements=actual.numel(),
            expected_elements=expected.numel(),
        )
        if not equal and strict:
            raise StageMismatch(stage)
        return actual, equal

    try:
        if str(device).startswith("xpu"):
            journal.step("device.initialization", lambda: torch.xpu.init())
            journal.step("device.initial_synchronize", lambda: torch.xpu.synchronize(device))
        if layout not in ["contiguous", "strided", "offset7"] or operation not in [
            "transfer",
            "index",
            "and",
            "where",
        ]:
            raise ValueError("Undeclared layout or operation")
        reference = reference_selection(values, observed)
        selected = reference.pop("selected")
        write(root / "reference.json", reference)
        with (root / "CPU_sources.pt").open("xb") as f:
            torch.save(dict(values=values, observed=observed), f)
        dense = journal.step("transfer.values_H2D", lambda: values.to(device))
        snapshot("transfer.values_roundtrip", dense, values)
        dense_mask = journal.step("transfer.mask_H2D", lambda: observed.to(device))
        snapshot("transfer.mask_roundtrip", dense_mask, observed)
        if layout == "contiguous":
            x, m = dense, dense_mask
        else:
            shape = (
                (*values.shape[:-1], values.shape[-1] * 2)
                if layout == "strided"
                else (values.numel() + 7,)
            )
            backing = journal.step(
                "layout.values_allocate_fill",
                lambda: torch.full(shape, 17.0, dtype=values.dtype, device=device),
            )
            snapshot(
                "layout.values_initial_roundtrip",
                backing,
                torch.full(shape, 17.0, dtype=values.dtype),
            )
            mask_backing = journal.step(
                "layout.mask_allocate_fill",
                lambda: torch.zeros(shape, dtype=torch.bool, device=device),
            )
            snapshot(
                "layout.mask_initial_roundtrip", mask_backing, torch.zeros(shape, dtype=torch.bool)
            )
            if layout == "strided":
                x = journal.step("layout.values_view", lambda: backing[..., ::2])
                m = journal.step("layout.mask_view", lambda: mask_backing[..., ::2])
                expected_backing = torch.full(shape, 17.0, dtype=values.dtype)
                expected_backing[..., ::2] = values
                expected_mask = torch.zeros(shape, dtype=torch.bool)
                expected_mask[..., ::2] = observed
            else:
                x = journal.step("layout.values_view", lambda: backing[7:].reshape(values.shape))
                m = journal.step("layout.mask_view", lambda: mask_backing[7:].reshape(values.shape))
                expected_backing = torch.full(shape, 17.0, dtype=values.dtype)
                expected_backing[7:] = values.flatten()
                expected_mask = torch.zeros(shape, dtype=torch.bool)
                expected_mask[7:] = observed.flatten()
            journal.step("layout.values_copy", lambda: x.copy_(dense))
            snapshot("layout.values_after_copy_roundtrip", backing, expected_backing)
            journal.step("layout.mask_copy", lambda: m.copy_(dense_mask))
            snapshot("layout.mask_after_copy_roundtrip", mask_backing, expected_mask)
        snapshot("pre_operation.values_roundtrip", x, values)
        snapshot("pre_operation.mask_roundtrip", m, observed)
        result["input_roundtrip_verified"] = True
        operation_started = operation != "transfer"
        correct = True
        if operation == "index":
            indexed = journal.step("boolean_index", lambda: x[m])
            host, correct = snapshot(
                "boolean_index.result_roundtrip", indexed, torch.from_numpy(selected), strict=False
            )
            result.update(
                expected_selected_count=len(selected),
                actual_selected_count=host.numel(),
                selected_values_exact=correct,
            )
        elif operation in ["and", "where"]:
            journal.step("candidate_" + operation, lambda: None)
            int_mask = journal.step("predicate.mask_to_int64", lambda: m.to(torch.int64))
            flat_mask = journal.step(
                "predicate.mask_reshape", lambda: int_mask.reshape(len(values), -1)
            )
            counts = journal.step("predicate.observed_count_sum", lambda: flat_mask.sum(1))
            _, counts_equal = snapshot(
                "predicate.observed_counts_roundtrip",
                counts,
                torch.tensor(reference["observed_counts"]),
                strict=False,
            )
            positive = journal.step("predicate.counts_positive", lambda: counts > 0)
            nonempty = journal.step("predicate.all_origins_supported", lambda: positive.all())
            if operation == "and":
                finite = journal.step("AND.finite_flags", lambda: torch.isfinite(x))
                nonfinite = journal.step("AND.invert_finite_flags", lambda: ~finite)
                bad = journal.step("AND.mask_nonfinite", lambda: m & nonfinite)
                bad_int = journal.step("AND.nonfinite_to_int64", lambda: bad.to(torch.int64))
                bad_flat = journal.step(
                    "AND.nonfinite_reshape", lambda: bad_int.reshape(len(values), -1)
                )
                bad_counts = journal.step("AND.nonfinite_count_sum", lambda: bad_flat.sum(1))
                _, bad_equal = snapshot(
                    "AND.bad_counts_roundtrip",
                    bad_counts,
                    torch.tensor(reference["observed_nonfinite_counts"]),
                    strict=False,
                )
                zero = journal.step("AND.counts_equal_zero", lambda: bad_counts == 0)
                all_finite = journal.step("AND.all_observed_finite", lambda: zero.all())
                correct = counts_equal and bad_equal
            else:
                zeros = journal.step("WHERE.zeros", lambda: torch.zeros_like(x))
                masked = journal.step("WHERE.masked_values", lambda: torch.where(m, x, zeros))
                finite = journal.step("WHERE.finite_flags", lambda: torch.isfinite(masked))
                all_finite = journal.step("WHERE.all_observed_finite", lambda: finite.all())
                correct = counts_equal
            decision = journal.step("predicate.final_and", lambda: nonempty & all_finite)
            verdict = journal.step("predicate.decision_roundtrip", lambda: decision.detach().cpu())
            result["verdict"] = bool(verdict)
            result["reference_verdict"] = reference["valid"]
            correct = correct and result["verdict"] == reference["valid"]
        snapshot("post_operation.values_roundtrip", x, values)
        snapshot("post_operation.mask_roundtrip", m, observed)
        result["status"] = "PASS" if correct else "OPERATOR_MISMATCH"
    except StageMismatch as error:
        result.update(
            status="POST_OPERATION_MISMATCH" if operation_started else "PRE_OPERATION_MISMATCH",
            first_mismatch_stage=error.stage,
            traceback=traceback.format_exc(),
        )
    except Exception:
        result.update(status="EXCEPTION", traceback=traceback.format_exc())
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        journal.close()
        write(root / "result.json", result)
    return result
