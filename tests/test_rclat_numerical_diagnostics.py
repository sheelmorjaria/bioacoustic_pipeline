"""Check that diagnostics distinguish norm overflow and invalid gradients."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path("scripts").resolve()))
from rclat_p10_numerical_diagnostic import tensor_stats
from rclat_p10_attention_diagnostic import attention


def test_float64_norm_distinguishes_float32_overflow():
    values = torch.tensor([1e30, -1e30], dtype=torch.float32)
    assert not torch.isfinite(torch.linalg.vector_norm(values))
    report = tensor_stats(values)
    assert report["nonfinite"] == 0
    assert isinstance(report["l2_float64"], float)
    assert report["l2_float64"] > 1e30


def test_nonfinite_elements_not_hidden_by_finite_summary():
    report = tensor_stats(torch.tensor([3.0, float("nan"), float("inf")]))
    assert report["nonfinite"] == 2
    assert report["finite_max_abs"] == 3.0
    assert isinstance(report["l2_float64"], str)


def test_attention_candidate_has_same_cpu_forward_and_vjp():
    torch.manual_seed(0)
    q = torch.randn(2, 7, 8)
    w = torch.randn(24, 8) * 0.1
    b = torch.randn(24) * 0.1
    valid = torch.arange(7)[None] < torch.tensor([4, 7])[:, None]
    upstream = torch.randn(2, 7, 8)
    results = []
    for variant in ["native", "composite_softmax"]:
        leaf, weight, bias = [v.clone().requires_grad_() for v in [q, w, b]]
        out, _ = attention(leaf, weight, bias, valid, variant)
        out.backward(upstream)
        results.append([out.detach(), leaf.grad, weight.grad, bias.grad])
    for reference, candidate in zip(*results):
        torch.testing.assert_close(reference, candidate, atol=2e-6, rtol=2e-5)
