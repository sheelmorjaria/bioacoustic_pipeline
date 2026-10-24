import numpy as np
import pytest
import torch
from bioacoustic_pipeline.representation.rclat_ctc import (
    CausalCTC,
    ctc_objective,
    encode_text,
    acoustic_features,
)


def test_causal_encoder_cannot_use_future_frames():
    torch.manual_seed(0)
    model = CausalCTC("mel", 64).eval()
    x = torch.randn(1, 30, 64)
    mask = torch.ones_like(x, dtype=torch.bool)
    with torch.no_grad():
        full, _ = model(x, mask)
        prefix, _ = model(x[:, :12], mask[:, :12])
    torch.testing.assert_close(full[:, :12], prefix, atol=1e-6, rtol=1e-6)


def test_padding_does_not_change_logits_or_final_state():
    model = CausalCTC("physical", 12).eval()
    x = torch.randn(2, 20, 12)
    mask = torch.ones_like(x, dtype=torch.bool)
    with torch.no_grad():
        logits, state = model(x, mask, torch.tensor([9, 20]))
        single, hidden = model(x[:1, :9], mask[:1, :9])
    torch.testing.assert_close(logits[:1, :9], single, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(state[:, :1], hidden, atol=1e-6, rtol=1e-6)


def test_invalid_ctc_alignment_is_rejected_not_silently_zeroed():
    logits = torch.zeros(1, 2, 29, requires_grad=True)
    with pytest.raises(ValueError, match="align"):
        ctc_objective(logits, torch.tensor([2]), [encode_text("AA")])
    loss = ctc_objective(
        torch.zeros(1, 5, 29, requires_grad=True), torch.tensor([5]), [encode_text("AA")]
    )
    assert torch.isfinite(loss)
    loss.backward()


def test_frontend_uses_only_complete_past_windows_and_preserves_silence():
    sr = 16000
    x = 0.1 * np.sin(2 * np.pi * 2000 * np.arange(sr) / sr)
    prefix = acoustic_features(x[:8000], sr)
    full = acoustic_features(x, sr)
    for key in ["mel", "physical", "spectral"]:
        np.testing.assert_allclose(prefix[key], full[key][: len(prefix[key])], equal_nan=True)
    silent = acoustic_features(np.zeros(1600), sr)
    assert len(silent["physical"]) == 7
    assert not silent["physical_valid"].any() and silent["mel_valid"].all()


def test_ctc_can_learn_a_tiny_independent_synthetic_sequence():
    from bioacoustic_pipeline.representation.rclat_ctc import decode_logits

    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(0)
        model = CausalCTC("mel", 4, width=16)
        features = torch.eye(4)[torch.tensor([0] * 6 + [1] * 6 + [2] * 6)][None]
        valid = torch.ones_like(features, dtype=torch.bool)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        target = [encode_text("A B")]
        first = None
        for _ in range(2000):
            optimizer.zero_grad()
            logits, _ = model(features, valid)
            loss = ctc_objective(logits, torch.tensor([18]), target)
            if first is None:
                first = float(loss.detach())
            loss.backward()
            optimizer.step()
        assert float(loss.detach()) < first * 0.1
        assert decode_logits(model(features, valid)[0][0], 18) == "A B"
    finally:
        torch.set_num_threads(old_threads)


def test_decoder_initialization_and_capacity_match_across_frontends():
    models = [
        CausalCTC(a, d)
        for a, d in [("mel", 64), ("physical", 12), ("combined", 76), ("pretrained", 768)]
    ]
    reference = models[0]
    for model in models[1:]:
        for key, value in reference.temporal.state_dict().items():
            torch.testing.assert_close(value, model.temporal.state_dict()[key], rtol=0, atol=0)
        for key, value in reference.head.state_dict().items():
            torch.testing.assert_close(value, model.head.state_dict()[key], rtol=0, atol=0)


def test_scaler_excludes_unsupported_values_and_does_not_adapt_at_inference():
    from bioacoustic_pipeline.representation.rclat_ctc import fit_scaler
    from bioacoustic_pipeline.diagnostics.frozen_acoustic_transfer import frozen_scale

    training = np.array([[1.0, 999.0], [3.0, -999.0]])
    mask = np.array([[True, False], [True, False]])
    scaler = fit_scaler([(training, mask)])
    assert scaler["mean"] == [2.0, 0.0]
    assert scaler["support"] == [True, False]
    values, valid = frozen_scale(np.array([[102.0, 4.0]]), np.ones((1, 2), bool), scaler)
    np.testing.assert_allclose(values, [[100.0, 0.0]])
    assert valid.tolist() == [[True, False]]
    assert scaler["mean"] == [2.0, 0.0]
