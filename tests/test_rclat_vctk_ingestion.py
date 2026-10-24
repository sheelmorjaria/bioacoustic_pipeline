import hashlib
import io
import json
import zipfile
import numpy as np
import pytest
import soundfile as sf
from bioacoustic_pipeline.diagnostics.rclat_vctk_ingestion import (
    audit_archive,
    download_stream,
    safe_url,
)


def archive(tmp_path, files):
    path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, payload in files.items():
            z.writestr(name, payload)
    return path, hashlib.md5(path.read_bytes()).hexdigest()


def audio(seed=1, rate=48000, channels=1):
    x = np.random.default_rng(seed).normal(0, 0.05, (48000, channels))
    b = io.BytesIO()
    sf.write(b, x, rate, format="FLAC", subtype="PCM_16")
    return b.getvalue()


def test_synthetic_support_duplicates_channels_and_reproducibility(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("checkpoint access forbidden"))
    same = audio()
    p, md5 = archive(
        tmp_path,
        {
            "root/p225/p225_001_mic2.flac": same,
            "root/p225/p225_002_mic2.flac": same,
            "root/p225/p225_003_mic1.flac": audio(3),
            "root/p226/p226_001_mic2.flac": audio(2),
            "root/p226/p226_002_mic2.flac": audio(4, 16000),
            "root/p226/p226_003_mic2.flac": audio(5, 48000, 2),
        },
    )
    a = audit_archive(p, tmp_path / "a", expected_md5=md5)
    b = audit_archive(p, tmp_path / "b", expected_md5=md5)
    assert a["counts"] == b["counts"]
    assert a["counts"]["retained_recordings"] == 2 and a["counts"]["retained_speakers"] == 2
    rows = [json.loads(s) for s in (tmp_path / "a" / "recordings.jsonl").read_text().splitlines()]
    assert {s["reason"] for s in rows} == {
        "eligible",
        "within_speaker_duplicate",
        "mic2_absent",
        "unsupported_sample_rate",
        "unsupported_channels",
    }
    assert all(s["eligible_origin_count"] == 49 for s in rows if s["reason"] == "eligible")
    with pytest.raises(FileExistsError):
        audit_archive(p, tmp_path / "a", expected_md5=md5)


def test_cross_speaker_duplicate_withheld(tmp_path):
    payload = audio()
    p, md5 = archive(
        tmp_path, {"root/p225/p225_001_mic2.flac": payload, "root/p226/p226_001_mic2.flac": payload}
    )
    result = audit_archive(p, tmp_path / "audit", expected_md5=md5)
    assert result["counts"]["retained_recordings"] == 0
    assert result["counts"]["exclusion_reasons"]["cross_speaker_duplicate"] == 2


def test_checksum_and_unsafe_archive_fail_before_decoding(tmp_path):
    p, md5 = archive(tmp_path, {"../escape": b"bad"})
    with pytest.raises(ValueError, match="checksum"):
        audit_archive(p, tmp_path / "wrong", expected_md5="wrong")
    with pytest.raises(ValueError, match="Unsafe"):
        audit_archive(p, tmp_path / "unsafe", expected_md5=md5)
    assert (tmp_path / "wrong" / "FAILURE.json").exists() and (
        tmp_path / "unsafe" / "FAILURE.json"
    ).exists()


def test_download_receipt_and_partial_failure_preservation(tmp_path):
    class Response(io.BytesIO):
        headers = {"Content-Length": "3"}

        def geturl(self):
            return "https://datashare.ed.ac.uk/bitstreams/fake/download"

    payload = b"zip"
    dest = tmp_path / "a.zip"
    receipt = download_stream(
        Response(payload), dest, expected_md5=hashlib.md5(payload).hexdigest(), max_bytes=10
    )
    assert dest.read_bytes() == payload and receipt["bytes"] == 3
    with pytest.raises(FileExistsError):
        download_stream(Response(payload), dest, expected_md5="wrong", max_bytes=10)
    with pytest.raises(ValueError, match="checksum"):
        download_stream(Response(payload), tmp_path / "bad.zip", expected_md5="wrong", max_bytes=10)
    assert (tmp_path / "bad.zip.partial").exists()
    with pytest.raises(ValueError):
        safe_url("https://evil.example/archive")
    with pytest.raises(ValueError):
        safe_url("http://datashare.ed.ac.uk/archive")


def test_invalid_and_short_waveforms_and_exact_coefficients(tmp_path):
    from bioacoustic_pipeline.diagnostics.rclat_vctk_ingestion import waveform_support

    bad = io.BytesIO()
    sf.write(bad, np.array([np.nan] * 1000), 48000, format="WAV", subtype="FLOAT")
    assert waveform_support(bad.getvalue())["reason"] == "invalid_waveform"
    short = io.BytesIO()
    sf.write(short, np.zeros(1000), 48000, format="FLAC", subtype="PCM_16")
    assert waveform_support(short.getvalue())["reason"] == "insufficient_duration"
    p, md5 = archive(tmp_path, {"root/p225/p225_001_mic2.flac": audio()})
    coeff = tmp_path / "bad.npz"
    np.savez(coeff, taps=np.ones(193))
    with pytest.raises(AssertionError, match="coefficients"):
        audit_archive(p, tmp_path / "coeff", expected_md5=md5, coefficient_path=coeff)


def test_ambiguous_event_and_member_budget_prevent_decoding(tmp_path):
    payload = audio()
    p, md5 = archive(
        tmp_path,
        {"first/p225/p225_001_mic2.flac": payload, "second/p225/p225_001_mic2.flac": payload},
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        audit_archive(p, tmp_path / "ambiguous", expected_md5=md5)
    with pytest.raises(MemoryError, match="memory budget"):
        audit_archive(p, tmp_path / "budget", expected_md5=md5, max_member_bytes=10)
