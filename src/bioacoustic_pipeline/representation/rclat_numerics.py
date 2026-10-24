"""Explicit finite-state guards for separately versioned RCLAT executions."""

from collections import defaultdict

import torch


def require_finite(named_tensors, label):
    """Batch device-local checks without one host synchronisation per parameter."""
    groups = defaultdict(list)
    for name, tensor in named_tensors:
        groups[str(tensor.device)].append((name, tensor))
    checked = 0
    for rows in groups.values():
        flags = torch.stack([torch.isfinite(tensor).all() for _, tensor in rows])
        checked += len(rows)
        if not bool(flags.all()):
            good = flags.detach().cpu().tolist()
            bad = [name for (name, _), valid in zip(rows, good) if not valid]
            raise FloatingPointError(f"Nonfinite {label}: {bad}")
    return checked


def check_gradients(model):
    rows = list(model.named_parameters())
    missing = [name for name, parameter in rows if parameter.grad is None]
    if missing:
        raise FloatingPointError(f"Missing parameter gradients: {missing}")
    return require_finite([(name, parameter.grad) for name, parameter in rows], "gradients")


def check_parameters_and_optimizer(model, optimizer):
    parameters = require_finite(model.named_parameters(), "parameters")
    rows = []
    names = {id(p): name for name, p in model.named_parameters()}
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if torch.is_tensor(value):
                rows.append((f"{names[id(parameter)]}.{key}", value))
    return {
        "parameters_checked": parameters,
        "optimizer_tensors_checked": require_finite(rows, "optimizer state"),
    }


def gradient_norm_float64(model):
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            value = parameter.grad.detach().cpu().double()
            squared += float((value * value).sum())
    return squared**0.5
