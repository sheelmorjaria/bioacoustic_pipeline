import importlib.util
import json

import pytest


def module():
    spec = importlib.util.spec_from_file_location("rtpe_entry", "scripts/rclat_rtpe_train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_execution_manifest_rejects_changed_sources_and_protocol(tmp_path, monkeypatch):
    m = module()
    protocol = tmp_path / "protocol.json"
    dep = tmp_path / "source.py"
    dep.write_text("original")
    protocol.write_text(
        json.dumps(
            dict(
                revision=2,
                status="LOCALLY_FROZEN_SCIENTIFIC_SPECIFICATION_NOT_EXTERNAL_REGISTRATION",
                dependencies={},
                preflight_artifacts={},
            )
        )
    )
    manifest = dict(
        protocol_sha256=m.sha(protocol),
        dependencies={str(dep): m.sha(dep)},
        environment=dict(
            python=m.platform.python_version(), torch=m.torch.__version__, numpy=m.np.__version__
        ),
    )
    (tmp_path / "execution_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(m, "PROTOCOL", protocol)
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "EXPECTED_PROTOCOL_SHA256", m.sha(protocol))
    m.bindings()
    dep.write_text("changed")
    with pytest.raises(ValueError, match="dependency changed"):
        m.bindings()
    dep.write_text("original")
    protocol.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        m.bindings()


def test_existing_attempt_is_rejected_before_any_xpu_or_source_access(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "bindings", lambda: ({}, {}))
    (tmp_path / "RUN_REQUESTED.json").write_text("{}")
    with pytest.raises(RuntimeError, match="no retry/resume"):
        m.run()


def test_verified_manifest_cannot_be_changed_during_execution(tmp_path, monkeypatch):
    m = module()
    p = tmp_path / "protocol.json"
    p.write_text("{}")
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "PROTOCOL", p)
    monkeypatch.setattr(m, "frozen", lambda: {})
    e = dict(
        protocol_sha256=m.sha(p),
        dependencies={},
        device="xpu:0",
        environment=dict(
            python=m.platform.python_version(), torch=m.torch.__version__, numpy=m.np.__version__
        ),
    )
    path = tmp_path / "execution_manifest.json"
    path.write_text(json.dumps(e))
    (tmp_path / "input_verification.json").write_text(
        json.dumps(dict(protocol_sha256=m.sha(p), execution_sha256=m.sha(path)))
    )
    m.bindings()
    e["device"] = "xpu:1"
    path.write_text(json.dumps(e))
    with pytest.raises(ValueError, match="manifest"):
        m.bindings()
