"""Independent tiny CTC path enumeration and decoder/length fixtures."""

from itertools import product, groupby
import numpy as np
import pytest
import torch
from bioacoustic_pipeline.diagnostics.rclat_ctc_learning import (
    conditioned_posterior,
    frame_summary,
    assert_lengths,
    edit_counts,
)
from bioacoustic_pipeline.representation.rclat_ctc import LABELS, decode_logits
from bioacoustic_pipeline.representation.rclat_scale_v4 import ScaleCTC


def collapsed(path):
    return [k for k, _ in groupby(path) if k != 0]


@pytest.mark.parametrize("target,frames", [([1], 3), ([1, 1], 3), ([1, 2], 4)])
def test_posterior_matches_all_enumerated_paths(target, frames):
    torch.set_num_threads(2)
    x = torch.tensor(np.random.default_rng(7).normal(size=(frames, 3)), dtype=torch.float64)
    probs = x.softmax(-1).numpy()
    mass = 0.0
    expected = np.zeros_like(probs)
    for path in product(range(3), repeat=frames):
        if collapsed(path) != target:
            continue
        weight = float(np.prod([probs[t, k] for t, k in enumerate(path)]))
        mass += weight
        for t, k in enumerate(path):
            expected[t, k] += weight
    expected /= mass
    p, gamma, stats = conditioned_posterior(x, target)
    np.testing.assert_allclose(p, probs, atol=1e-12)
    np.testing.assert_allclose(gamma, expected, atol=1e-9)
    assert stats["ctc_log_probability"] == pytest.approx(np.log(mass), abs=1e-9)
    if target == [1]:
        np.testing.assert_allclose(gamma[:, 2], 0, atol=1e-9)


@pytest.mark.parametrize(
    "symbols,expected",
    [(["-", "-"], ""), (["A", "A"], "A"), (["A", "-", "A"], "AA"), (["A", "|", "|", "B"], "A B")],
)
def test_decoder_independent_fixtures(symbols, expected):
    x = torch.full((len(symbols), len(LABELS)), -20.0)
    for t, s in enumerate(symbols):
        x[t, LABELS.index(s)] = 20
    assert decode_logits(x, len(x)) == expected


def test_unequal_padded_batch_actual_output_lengths():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    model = ScaleCTC(4, width=16, layers=2, heads=2, expansion=32, kernel=3).eval()
    x = torch.randn(2, 53, 4)
    x[1, 37:] = float("nan")
    m = torch.ones_like(x, dtype=torch.bool)
    with torch.no_grad():
        y, lengths = model(x, m, torch.tensor([53, 37]))
    assert y.shape[1] == 14 and lengths.tolist() == [14, 10]
    assert_lengths(y, lengths, torch.tensor([53, 37]), [torch.tensor([2, 2]), torch.tensor([3])])
    with pytest.raises(ValueError, match="actual 4x"):
        assert_lengths(
            y, lengths - 1, torch.tensor([53, 37]), [torch.tensor([2]), torch.tensor([3])]
        )


def test_infeasible_repeated_target_is_rejected():
    with pytest.raises(ValueError, match="infeasible"):
        conditioned_posterior(np.ones((2, 3)), [1, 1])


def test_frame_statistics_distinguish_free_and_conditioned_blank():
    p, gamma, _ = conditioned_posterior(np.array([[8.0, 0.0, -1.0]] * 3), [1])
    stats = frame_summary(p, gamma, 1)
    assert stats["blank_argmax_fraction"] == 1
    assert stats["conditioned_mean_blank_occupancy"] < stats["mean_blank_probability"]
    assert sum(stats["blank_probability_histogram"]) == 3


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("AB", "AB", (0, 0, 0)),
        ("AB", "AC", (1, 0, 0)),
        ("AB", "A", (0, 1, 0)),
        ("A", "AB", (0, 0, 1)),
    ],
)
def test_edit_decomposition(a, b, expected):
    r = edit_counts(a, b)
    assert tuple(r[k] for k in ["substitutions", "deletions", "insertions"]) == expected
