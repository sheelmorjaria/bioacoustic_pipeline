import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rclat_grounding_matching_audit import describe, best_matching  # noqa: E402


def test_joint_bounds_and_identity():
    a = {"uid": "a", "speaker": "s", "transcript": "AAAAAAAAAA", "duration_seconds": 10}
    b = {"uid": "b", "speaker": "s", "transcript": "BBBBBBBBBBB", "duration_seconds": 11}
    assert describe(a, b)["eligible"]
    assert not describe(a, {**b, "duration_seconds": 11.1})["eligible"]
    assert not describe(a, {**b, "speaker": "other"})["eligible"]
    assert not describe(a, {**b, "transcript": a["transcript"]})["eligible"]


def test_matching_requires_all_nodes():
    edges = {("a", "b"): {"cost": 0.1}, ("c", "d"): {"cost": 0.2}}
    assert best_matching(["a", "b", "c", "d"], edges) == [("a", "b"), ("c", "d")]
    assert best_matching(["a", "b", "c", "d"], {("a", "b"): {"cost": 0.1}}) is None
