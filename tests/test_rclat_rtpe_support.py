import numpy as np
import pytest

from bioacoustic_pipeline.diagnostics.rclat_rtpe_support import (
    partition_metadata,
    timing_support,
)


def rows():
    return [
        dict(
            uid=f"{i}-10-1",
            speaker=str(i),
            samples=16000,
            feature_frames=97,
            source_sha256=f"{i:064x}",
            decoded_pcm16_sha256=f"{i + 100:064x}",
            path=f"a/{i}.flac",
        )
        for i in range(12)
    ]


def test_partition_is_deterministic_speaker_disjoint_and_order_invariant():
    a = partition_metadata(rows(), train_speakers=6, validation_speakers=3)
    b = partition_metadata(list(reversed(rows())), train_speakers=6, validation_speakers=3)
    assert a == b
    assert {k: sum(r["split"] == k for r in a) for k in ("train", "validation", "test")} == dict(
        train=6, validation=3, test=3
    )
    assert all("transcript" not in r for r in a)


def test_duplicates_and_identity_conflicts_cannot_cross_groups():
    r = rows()
    r[1]["decoded_pcm16_sha256"] = r[0]["decoded_pcm16_sha256"]
    with pytest.raises(ValueError, match="Duplicate"):
        partition_metadata(r, train_speakers=6, validation_speakers=3)
    r = rows()
    r[0]["speaker"] = "someone_else"
    with pytest.raises(ValueError, match="identity"):
        partition_metadata(r, train_speakers=6, validation_speakers=3)


def test_frame_end_availability_and_future_waveform_nonoverlap():
    r = timing_support(16000, history=32, horizons=(4, 8, 16))
    assert r["frames"] == 97 and r["common_origins_upper_bound"] == 50
    assert r["first_origin_available_sample_exclusive"] == 5600
    assert r["last_origin_available_sample_exclusive"] == 13440
    assert r["target_start_minus_origin_end_samples"] == [0, 640, 1920]
    assert r["latest_target_end_sample_exclusive"] <= 16000
    with pytest.raises(ValueError, match="overlap"):
        timing_support(16000, history=32, horizons=(1,))
    assert timing_support(640, history=32, horizons=(4,))["common_origins_upper_bound"] == 0


def test_existing_logmel_is_invariant_to_unavailable_future_samples():
    from bioacoustic_pipeline.representation.rclat_ctc import acoustic_features

    rng = np.random.default_rng(77)
    x = rng.normal(0, 0.1, 2240)
    prefix = acoustic_features(x[:1600], 16000)
    changed = x.copy()
    changed[1600:] = rng.normal(0, 10, len(x) - 1600)
    full = acoustic_features(changed, 16000)
    np.testing.assert_array_equal(prefix["mel"], full["mel"][: len(prefix["mel"])])
    np.testing.assert_array_equal(
        prefix["frame_end_samples"], np.arange(len(prefix["mel"])) * 160 + 640
    )
