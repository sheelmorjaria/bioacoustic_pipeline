"""Operator-only validity predicates, not a replacement of frozen production guards."""

import numpy as np
import torch


def _schema(values, observed):
    if (
        values.ndim < 2
        or values.shape != observed.shape
        or not values.is_floating_point()
        or observed.dtype != torch.bool
        or values.device != observed.device
        or values.shape[0] == 0
        or values.numel() == 0
    ):
        raise ValueError("Matching nonempty floating tensors and boolean support required")


def reference_selection(values, observed):
    """Independent NumPy host reference; observed nonfinite and empty rows reject."""
    _schema(values, observed)
    a = values.detach().cpu().numpy()
    m = observed.detach().cpu().numpy()
    counts = m.reshape(len(a), -1).sum(1)
    bad = (m & ~np.isfinite(a)).reshape(len(a), -1).sum(1)
    return dict(
        expected_count=int(counts.sum()),
        observed_counts=counts.tolist(),
        observed_nonfinite_counts=bad.tolist(),
        valid=bool((counts > 0).all() and (bad == 0).all()),
        selected=a[m].copy(),
    )


def shape_preserving_checks(values, observed):
    """Two diagnostic-only paths; neither performs boolean indexing/gathering.

    Each origin needs observed support. Masked NaN/Inf are allowed, observed
    NaN/Inf rejected. Independent predicate paths are compared with CPU truth.
    """
    _schema(values, observed)
    counts = observed.to(torch.int64).reshape(len(values), -1).sum(1)
    bad = observed & ~torch.isfinite(values)
    bad_counts = bad.to(torch.int64).reshape(len(values), -1).sum(1)
    nonempty = (counts > 0).all()
    and_valid = nonempty & (bad_counts == 0).all()
    masked = torch.where(observed, values, torch.zeros_like(values))
    where_valid = nonempty & torch.isfinite(masked).all()
    return dict(
        and_valid=bool(and_valid.detach().cpu()),
        where_valid=bool(where_valid.detach().cpu()),
        observed_counts=counts.detach().cpu().tolist(),
        observed_nonfinite_counts=bad_counts.detach().cpu().tolist(),
    )


def layout_copy(values, observed, layout, device):
    """Construct layouts on destination, preserving intended strides/storage offset."""
    _schema(values, observed)
    if layout == "contiguous":
        return values.to(device).contiguous(), observed.to(device).contiguous()
    if layout == "strided":
        shape = (*values.shape[:-1], values.shape[-1] * 2)
        a = torch.full(shape, 17.0, dtype=values.dtype, device=device)
        m = torch.zeros(shape, dtype=torch.bool, device=device)
        a[..., ::2].copy_(values.to(device))
        m[..., ::2].copy_(observed.to(device))
        return a[..., ::2], m[..., ::2]
    if layout == "offset7":
        a = torch.full((values.numel() + 7,), 17.0, dtype=values.dtype, device=device)
        m = torch.zeros(values.numel() + 7, dtype=torch.bool, device=device)
        x = a[7:].reshape(values.shape)
        v = m[7:].reshape(values.shape)
        x.copy_(values.to(device))
        v.copy_(observed.to(device))
        return x, v
    raise ValueError("Unknown prospective layout")
