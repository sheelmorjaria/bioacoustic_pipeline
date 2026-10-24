import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rclat_grounding_source_support import disjoint_pairs  # noqa: E402


def test_cardinality_precedes_cost_and_no_reuse():
    edges = [
        {"uids": ["a", "b"], "cost": 0},
        {"uids": ["a", "c"], "cost": 0.1},
        {"uids": ["b", "d"], "cost": 0.1},
    ]
    assert disjoint_pairs(["a", "b", "c", "d"], edges) == [("a", "c"), ("b", "d")]
    assert disjoint_pairs(["a", "b"], []) == []
