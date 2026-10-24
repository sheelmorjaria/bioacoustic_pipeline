"""Acquisition integrity checks require neither network access nor model fits."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def audit_module():
    path = Path(__file__).resolve().parents[1] / 'scripts/macde_acquisition_audit.py'
    spec = importlib.util.spec_from_file_location('macde_acquisition_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_integrity_is_checked_and_recorded(tmp_path, monkeypatch):
    module = audit_module()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    folder = tmp_path / 'candidate'
    folder.mkdir()
    audio = folder / 'example.wav'
    audio.write_bytes(b'example')
    md5 = hashlib.md5(audio.read_bytes()).hexdigest()
    (folder / 'record.json').write_text(json.dumps({
        'metadata': {'license': {'id': 'cc-by-4.0'}},
        'files': [{'key': audio.name, 'size': 7, 'checksum': 'md5:'+md5,
                   'links': {'self': 'https://example.org/data'}}]}))
    result = module.verify(folder)
    assert result[0]['sha256'] == hashlib.sha256(b'example').hexdigest()
    assert result[0]['repository_manifest_sha256'] == module.digest(folder / 'record.json')
    audio.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Checksum/size mismatch'):
        module.verify(folder)
