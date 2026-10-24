import importlib.util
import json
import pytest


def load():
    spec = importlib.util.spec_from_file_location(
        "source_corrected", "scripts/rclat_rtpe_vctk_support_corrected.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_predecessor_permission_does_not_authorise_successor(tmp_path):
    m = load()
    m.ROOT = tmp_path
    with pytest.raises(ValueError, match="Fresh"):
        m.authorize()
    (tmp_path / "protocol.json").write_text("{}")
    (tmp_path / "execution_manifest.json").write_text("{}")
    (tmp_path / "SOURCE_AUTHORIZATION.json").write_text(
        json.dumps(
            dict(support_audit_authorized=True, protocol_sha256="old", execution_sha256="old")
        )
    )
    with pytest.raises(ValueError, match="Fresh"):
        m.authorize()


def test_corrected_runner_exposes_no_forecast_or_acquisition():
    m = load()
    for name in ["predict", "train", "evaluate", "acquire", "resolve_archive"]:
        assert not hasattr(m, name)
