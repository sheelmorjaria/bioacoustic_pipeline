import importlib.util
import pytest


def test_unapproved_scientific_execution_cannot_start(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "zero_shot", "scripts/rclat_rtpe_vctk_zero_shot.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.ROOT = tmp_path
    with pytest.raises(ValueError, match="authorisation"):
        m.execute()
    assert not (tmp_path / "RUN_REQUESTED.json").exists()
    assert not hasattr(m, "fit") and not hasattr(m, "train")
