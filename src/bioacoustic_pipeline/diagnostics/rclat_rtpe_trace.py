"""Observational numerical tracing; never changes tensors or replaces model operators."""

import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes


def _write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


class CaseRecorder:
    def __init__(self, directory, *, seed, case):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=False)
        self.seed = seed
        self.case = case
        self.sequence = 0
        self.first_nonfinite = None
        self.journal = (self.root / "events.jsonl").open("x", buffering=1)
        _write(self.root / "STARTED.json", dict(seed=seed, case=case))

    def event(self, **fields):
        self.sequence += 1
        row = dict(sequence=self.sequence, seed=self.seed, case=self.case, **fields)
        self.journal.write(json.dumps(row, allow_nan=False) + "\n")
        return row

    def tensor(self, name, tensor, *, observed=None, operation=None, operand_shapes=None):
        with _disable_current_modes():
            host = tensor.detach().cpu().clone()
            flags = torch.isfinite(host)
            device_flags = torch.isfinite(tensor.detach()).cpu()
            mask = torch.ones_like(flags) if observed is None else observed.detach().cpu()
            if mask.dtype != torch.bool or mask.shape != host.shape:
                raise ValueError("Observation mask must match recorded tensor")
            bad = ~flags
            finite = host[flags]
            info = dict(
                name=name,
                operation=operation,
                operand_shapes=operand_shapes,
                shape=list(host.shape),
                dtype=str(host.dtype),
                device=str(tensor.device),
                host_nonfinite=int(bad.sum()),
                device_nonfinite=int((~device_flags).sum()),
                finiteness_flag_disagreements=int((flags != device_flags).sum()),
                observed_nonfinite=int((bad & mask).sum()),
                expected_masked_nonfinite=int((bad & ~mask).sum()),
                finite_max_abs=float(
                    finite.to(torch.uint8).max()
                    if finite.dtype == torch.bool
                    else finite.abs().max()
                )
                if finite.numel()
                else None,
                tensor_sha256=hashlib.sha256(
                    host.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            )
            row = self.event(**info)
            unexpected = info["observed_nonfinite"] > 0 or bool(
                ((flags != device_flags) & mask).any()
            )
            if unexpected and self.first_nonfinite is None:
                self.first_nonfinite = row
                with (self.root / "first_nonfinite.pt").open("xb") as f:
                    torch.save(
                        dict(
                            tensor=host,
                            observed=mask,
                            device_finite_flags=device_flags,
                            metadata=row,
                        ),
                        f,
                    )
                    f.flush()
                    os.fsync(f.fileno())
            return info

    def finish(self, status, details):
        if (self.root / "result.json").exists():
            raise FileExistsError("Completed case cannot be overwritten")
        result = dict(
            seed=self.seed,
            case=self.case,
            status=status,
            first_unexpected_nonfinite=self.first_nonfinite,
            events_recorded=self.sequence,
            **details,
        )
        self.event(name="case.finished", status=status)
        self.journal.flush()
        os.fsync(self.journal.fileno())
        _write(self.root / "result.json", result)
        self.journal.close()
        return result


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


class NumericalTrace(TorchDispatchMode):
    """Observe dispatched operator outputs; omit uninitialised allocation buffers.

    Tensor copies/finiteness checks synchronise the device and may change the
    symptom or dispatch behaviour. A passing trace is not a corrective result.
    """

    def __init__(self, recorder, phase):
        super().__init__()
        self.recorder = recorder
        self.phase = phase

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        name = str(func)
        if name.startswith(("aten.empty", "aten.new_empty", "aten.resize_")):
            self.recorder.event(
                name=self.phase + ".allocation",
                operation=name,
                uninitialised_output_not_checked=True,
            )
            return result
        with _disable_current_modes():
            inputs = [list(t.shape) for t in _tensors(args)]
            outputs = list(_tensors(result))
            for i, tensor in enumerate(outputs):
                self.recorder.tensor(
                    f"{self.phase}.output{i}", tensor, operation=name, operand_shapes=inputs
                )
            if not outputs:
                self.recorder.event(
                    name=self.phase + ".scalar", operation=name, scalar_result=repr(result)[:128]
                )
        return result


def attach_checkpoint_hooks(model, capture):
    """Observer hooks always return None, regardless of the logger's return value."""

    def encoder_input(_module, args):
        capture("encoder_input", args[0])

    def encoder_output(_module, _args, output):
        capture("GRU_sequence", output[0])
        capture("GRU_hidden", output[1])

    def forecast_input(_module, args):
        capture("forecast_head_input", args[0])

    def forecast_output(_module, _args, output):
        capture("forecast_head_output", output)

    return [
        model.encoder.register_forward_pre_hook(encoder_input),
        model.encoder.register_forward_hook(encoder_output),
        model.forecast.register_forward_pre_hook(forecast_input),
        model.forecast.register_forward_hook(forecast_output),
    ]
