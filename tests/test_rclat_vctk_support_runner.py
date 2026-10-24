import importlib.util
import io
import json
from pathlib import Path
import pytest


def runner():
    spec = importlib.util.spec_from_file_location(
        "source_runner", "scripts/rclat_rtpe_vctk_support_runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_authorization_cannot_enable_forecasting(tmp_path):
    module = runner()
    module.ROOT = tmp_path
    for action in ["acquisition", "support_audit"]:
        with pytest.raises(ValueError, match="authorisation"):
            module.authorize(action)
    assert (
        not hasattr(module, "predict")
        and not hasattr(module, "train")
        and not hasattr(module, "evaluate")
    )
    source = Path("scripts/rclat_rtpe_vctk_support_runner.py").read_text()
    assert "torch.load" not in source and "load_state_dict" not in source


def test_resolver_accepts_only_official_matching_archive_metadata(tmp_path):
    module = runner()
    url = "https://datashare.ed.ac.uk/server/api/core/bitstreams/archive/content"
    bundles = {
        "_embedded": {
            "bundles": [
                {
                    "name": "ORIGINAL",
                    "_links": {"bitstreams": {"href": "https://datashare.ed.ac.uk/meta"}},
                }
            ]
        }
    }
    archive = {
        "name": "fixture.zip",
        "sizeBytes": 3,
        "checkSum": {"checkSumAlgorithm": "MD5", "value": "abc"},
        "_links": {"content": {"href": url}},
    }

    class Reply(io.BytesIO):
        def geturl(self):
            return "https://datashare.ed.ac.uk/meta"

    class Opener:
        def __init__(self):
            self.calls = 0

        def open(self, url, timeout):
            self.calls += 1
            return Reply(
                json.dumps(
                    bundles if self.calls == 1 else {"_embedded": {"bitstreams": [archive]}}
                ).encode()
            )

    policy = {
        "source": {"archive_name": "fixture.zip", "published_md5": "abc"},
        "acquisition": {"maximum_archive_bytes": 10},
    }
    assert module.resolve_archive(Opener(), policy, tmp_path)["official_archive_url"] == url
    other = tmp_path / "bad"
    other.mkdir()
    archive["checkSum"]["value"] = "wrong"
    with pytest.raises(ValueError, match="checksum"):
        module.resolve_archive(Opener(), policy, other)
