"""Mechanical integration only;these tests assert no biological effects."""
import numpy as np
import pytest

from bioacoustic_pipeline.contextual.macde_bridge import (
    AcousticBridge, AudioSource, VocalEvent, ObservedMetadata, to_cavem_sequence,
)


class Encoder:
    """Deterministic causal fixture;real encoder masking is tested separately."""
    stage2_ready = True  # cannot unlock biological evaluation through this adapter

    def encode_audio(self, audio, sample_rate, sample_valid=None):
        ends = np.arange(4, len(audio)+1, 2)
        state = np.stack([audio[ends-1], audio[ends-2]], 1)
        valid = np.ones(len(ends), bool)
        if sample_valid is not None:
            valid = np.array([sample_valid[t-4:t].all() for t in ends])
        change_valid = np.tile(valid[:, None], (1, 4))
        return {'state': state, 'state_valid': valid, 'changes': np.tile(state[:, None], (1, 4, 1)),
                'change_valid': change_valid, 'frame_end_seconds': ends/sample_rate,
                'source_valid': valid, 'detected_activity': valid}


def source(audio=None, origin=100):
    return AudioSource('recording-A', 'recording-A:native-samples', 100., origin,
                       np.arange(80, dtype=float) if audio is None else audio)


def event(start=100, end=160, **kwargs):
    return VocalEvent('event1', 'recording-A', start, end, **kwargs)


def test_prefix_causality_coordinates_and_unknown_identity():
    bridge = AcousticBridge(Encoder(), encoder_id='synthetic-fixture')
    original = source()
    out = bridge.export_event(original, event())
    audio = original.audio.copy()
    audio[60:] = -999
    altered = bridge.export_event(source(audio), event())
    np.testing.assert_allclose(out.summary, altered.summary, equal_nan=True)
    assert (out.trajectory['frame_end_sample'] <= 160).all()
    assert (out.trajectory['frame_start_sample'] >= 100).all()
    assert out.caller is None and out.recipient is None
    assert out.source_origin_sample == 100
    assert out.coordinate_reference == 'recording-A:native-samples'
    assert out.metadata()['stage2_ready'] is False


def test_post_call_timing_and_future_metadata_are_enforced():
    bridge = AcousticBridge(Encoder(), encoder_id='fixture')
    with pytest.raises(ValueError, match='post-call'):
        bridge.export_event(source(), event(), prediction_sample=100)
    with pytest.raises(ValueError, match='available'):
        bridge.export_event(source(), event(caller=ObservedMetadata('A', 170, 'independent annotation')))
    out = bridge.export_event(source(), event(caller=ObservedMetadata('Unk', 150, 'original unknown annotation')))
    assert out.caller == 'Unk' and out.recipient is None


def test_missing_features_are_masked_and_never_filled_as_observed():
    bridge = AcousticBridge(Encoder(), encoder_id='fixture')
    src = source()
    src = AudioSource(src.source_id, src.coordinate_reference, src.sample_rate,
                      src.sample_origin, src.audio, np.zeros(len(src.audio), bool))
    out = bridge.export_event(src, event())
    assert not out.summary_valid.any() and np.isnan(out.summary).all()
    consumer = to_cavem_sequence([out])[0]
    assert np.isfinite(consumer.acoustic).all()
    assert consumer.caller is None
    assert not consumer.provenance['acoustic_supported']


def test_change_support_does_not_reach_before_event():
    bridge = AcousticBridge(Encoder(), encoder_id='fixture')
    out = bridge.export_event(source(), event(start=145, end=160))
    assert not out.trajectory['change_valid'][:, -1].any()
    assert out.summary.shape == out.summary_valid.shape


def test_sequence_preserves_overlap_and_rejects_artificial_concatenation():
    bridge = AcousticBridge(Encoder(), encoder_id='fixture')
    a = bridge.export_event(source(), event(start=100, end=140))
    b = bridge.export_event(source(), VocalEvent('event2', 'recording-A', 130, 160))
    seq = to_cavem_sequence([a, b])
    assert seq[1].dt_prev_ms == -100.
    assert seq[1].provenance['overlap_with_previous']
    assert seq[1].duration_ms == 300.
    with pytest.raises(ValueError, match='completion'):
        to_cavem_sequence([a, a])
    other = AudioSource('recording-B', 'recording-B:samples', 100., 100, np.arange(80, dtype=float))
    c = bridge.export_event(other, VocalEvent('c', 'recording-B', 100, 150))
    with pytest.raises(ValueError, match='recording'):
        to_cavem_sequence([a, c])


def test_cavem_accepts_export_and_preserves_prefix_state():
    from bioacoustic_pipeline.contextual.cavem import CausalGRU
    bridge = AcousticBridge(Encoder(), encoder_id='fixture')
    a = bridge.export_event(source(), event(start=100, end=140))
    b = bridge.export_event(source(), VocalEvent('event2', 'recording-A', 140, 170))
    seq = to_cavem_sequence([a, b])
    model = CausalGRU(event_dim=seq[0].acoustic.shape[1], hidden=4)
    np.testing.assert_allclose(model.causal_states(seq[:1])[0], model.causal_states(seq)[0], atol=1e-6)


def test_checkpoint_policy_and_integrity_are_checked(tmp_path):
    import hashlib
    import torch
    path = tmp_path / 'shuffled.pt'
    torch.save({'arm_policy': 'combined_disrupted'}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='checksum'):
        AcousticBridge.from_checkpoint(path, expected_sha256='incorrect')
    with pytest.raises(ValueError, match='ordered'):
        AcousticBridge.from_checkpoint(path, expected_sha256=digest)
