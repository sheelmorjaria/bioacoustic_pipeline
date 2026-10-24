import hashlib
import io
import zipfile
import numpy as np
import pytest
import soundfile as sf
from bioacoustic_pipeline.diagnostics.rclat_vctk_ingestion import waveform_support
from bioacoustic_pipeline.diagnostics.rclat_vctk_transfer_data import build_source_arrays


def fixture(tmp_path):
    stream = io.BytesIO()
    sf.write(
        stream,
        np.random.default_rng(23).normal(0, 0.05, 48000),
        48000,
        format="FLAC",
        subtype="PCM_16",
    )
    payload = stream.getvalue()
    info = waveform_support(payload)
    name = "root/p225/p225_001_mic2.flac"
    z = tmp_path / "source.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr(name, payload)
    row = dict(
        info,
        speaker="p225",
        event="p225_001",
        member=name,
        selected_origins=[32, 40, 60, 80],
        source_bytes_sha256=hashlib.sha256(payload).hexdigest(),
    )
    return z, row


def test_reconstructs_exact_selected_histories_and_masks(tmp_path):
    z, row = fixture(tmp_path)
    data, ids = build_source_arrays(z, [row])
    assert data["x"].shape == (4, 32, 64) and data["v"].all()
    assert data["y"].shape == (4, 3, 64) and not data["m"][:, 0].any() and data["m"][:, 1:].all()
    assert [r["origin"] for r in ids] == [32, 40, 60, 80]


def test_changed_source_or_unsupported_origin_cannot_be_replaced(tmp_path):
    z, row = fixture(tmp_path)
    row["source_bytes_sha256"] = "bad"
    with pytest.raises(ValueError, match="source"):
        build_source_arrays(z, [row])
    z, row = fixture(tmp_path)
    row["selected_origins"] = [31, 40, 60, 80]
    with pytest.raises(ValueError, match="support"):
        build_source_arrays(z, [row])
