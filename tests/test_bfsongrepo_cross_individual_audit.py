"""Source audit tests: native coordinates, metadata conflicts and no identity inference."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def audit(monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('support_audit', scripts/'bfsongrepo_cross_individual_audit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_audio():
    n = 64000
    x = (3000*np.sin(2*np.pi*2000*np.arange(n)/32000)).astype(np.int16)
    channels = np.stack([x, np.zeros(n, np.int16)], axis=1).astype('>i2').tobytes()
    rec = f'ADFREQ = 32000\nChans = 2\nSamples = {n}\nFile created: Sat, Apr 21, 2012, 08:00:00\n'.encode()
    return channels, rec


def test_native_decoding_preserves_identity_and_support(audit):
    raw, rec = fixture_audio()
    row, mask = audit.audit_recording('bl26lb16/042112/song.cbin', raw, rec, 'bl26lb16', '042112')
    assert row['rec_date_agrees_with_archive']
    assert row['duration_seconds'] == 2
    assert row['session_id'] is None and row['partition'] is None
    assert row['full_recording_support']['runs_at_least128_frames'] > 0
    assert len(mask) == 197


def test_date_conflict_is_preserved(audit):
    raw, rec = fixture_audio()
    row, _ = audit.audit_recording('bl26lb16/042012/song.cbin', raw, rec, 'bl26lb16', '042012')
    assert not row['rec_date_agrees_with_archive']
    assert row['documented_date'] == '2012-04-20'
    assert row['file_created_from_rec'].startswith('2012-04-21')


def test_archive_identity_mismatch_rejected(audit):
    raw, rec = fixture_audio()
    with pytest.raises(ValueError, match='Unexpected bird/day'):
        audit.audit_recording('bl26lb16/042112/song.cbin', raw, rec, 'gr41rd51', '042112')
