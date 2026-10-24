"""Retrospective/prospective CTC instrumentation; no training policy changes."""

import numpy as np
import torch
from torch.nn import functional as F


def assert_lengths(logits, lengths, native_lengths, targets):
    if logits.ndim != 3 or len(lengths) != len(native_lengths) or len(targets) != len(lengths):
        raise ValueError("Batch length schema mismatch")
    expected = (native_lengths.cpu() + 3) // 4
    if not torch.equal(lengths.cpu(), expected) or logits.shape[1] != int(expected.max()):
        raise ValueError("Declared CTC lengths differ from actual 4x output")
    for length, target in zip(lengths, targets):
        minimum = len(target) + int((target[1:] == target[:-1]).sum())
        if not len(target) or int(length) < minimum:
            raise ValueError("CTC path infeasible")


def conditioned_posterior(logits, target):
    """CPU FP64 occupancy from CTC logit gradients; validates without clipping."""
    x = torch.as_tensor(logits, dtype=torch.float64, device="cpu").detach().requires_grad_(True)
    y = torch.as_tensor(target, dtype=torch.long, device="cpu")
    if x.ndim != 2 or not torch.isfinite(x).all() or not len(y):
        raise ValueError("Invalid posterior input")
    minimum = len(y) + int((y[1:] == y[:-1]).sum())
    if len(x) < minimum or (y <= 0).any() or (y >= x.shape[1]).any():
        raise ValueError("Invalid/infeasible posterior target")
    logp = x.log_softmax(-1)
    loss = F.ctc_loss(
        logp[:, None, :],
        y,
        torch.tensor([len(x)]),
        torch.tensor([len(y)]),
        blank=0,
        reduction="sum",
        zero_infinity=False,
    )
    gradient = torch.autograd.grad(loss, x)[0]
    p = logp.exp().detach()
    occupancy = (p - gradient).detach()
    tol = 1e-7
    support = torch.zeros(x.shape[1], dtype=torch.bool)
    support[0] = True
    support[y] = True
    if (
        not torch.isfinite(loss)
        or not torch.isfinite(occupancy).all()
        or float(occupancy.min()) < -tol
        or float(occupancy.max()) > 1 + tol
        or not torch.allclose(
            occupancy.sum(-1), torch.ones(len(x), dtype=torch.float64), atol=tol, rtol=0
        )
        or (not support.all() and float(occupancy[:, ~support].abs().max()) > tol)
    ):
        raise FloatingPointError("Invalid conditioned CTC posterior; no sanitisation")
    return (
        p.numpy(),
        occupancy.numpy(),
        dict(
            ctc_log_probability=-float(loss.detach()),
            occupancy_min=float(occupancy.min()),
            occupancy_max=float(occupancy.max()),
            occupancy_row_sum_max_error=float((occupancy.sum(-1) - 1).abs().max()),
        ),
    )


def frame_summary(p, occupancy, e_index):
    nonblank = p[:, 1:]
    mass = nonblank.sum(-1)
    entropy = None
    if np.all(mass > 0):
        q = nonblank / mass[:, None]
        positive = q > 0
        terms = np.zeros_like(q)
        terms[positive] = -q[positive] * np.log2(q[positive])
        entropy = float(terms.sum(-1).mean())
    return dict(
        valid_frames=len(p),
        mean_blank_probability=float(p[:, 0].mean()),
        blank_argmax_fraction=float((p.argmax(-1) == 0).mean()),
        mean_E_probability=float(p[:, e_index].mean()),
        E_argmax_fraction=float((p.argmax(-1) == e_index).mean()),
        mean_nonblank_entropy_bits=entropy,
        conditioned_mean_blank_occupancy=float(occupancy[:, 0].mean()),
        conditioned_mean_E_occupancy=float(occupancy[:, e_index].mean()),
        blank_probability_histogram=np.histogram(p[:, 0], bins=np.linspace(0, 1, 11))[0].tolist(),
        E_probability_histogram=np.histogram(p[:, e_index], bins=np.linspace(0, 1, 11))[0].tolist(),
    )


def edit_counts(reference, hypothesis):
    """Minimum edits; tie preference diagonal, deletion, insertion."""
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, a in enumerate(reference, 1):
        row = [(i, 0, i, 0)]
        for j, b in enumerate(hypothesis, 1):
            d, s, de, ins = previous[j - 1]
            best = (d + (a != b), s + (a != b), de, ins)
            d, s, de, ins = previous[j]
            candidate = (d + 1, s, de + 1, ins)
            if candidate[0] < best[0]:
                best = candidate
            d, s, de, ins = row[-1]
            candidate = (d + 1, s, de, ins + 1)
            if candidate[0] < best[0]:
                best = candidate
            row.append(best)
        previous = row
    d, s, de, ins = previous[-1]
    return dict(errors=d, substitutions=s, deletions=de, insertions=ins)
