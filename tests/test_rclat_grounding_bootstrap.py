import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rclat_bilstm_matched_summary import paired_interval  # noqa: E402


def test_cluster_interval_constant_and_seed_reproducible():
    x = paired_interval([0.2] * 24, 20261022, 10000)
    assert abs(x["point_estimate"] - 0.2) < 1e-12
    assert all(abs(v - 0.2) < 1e-12 for v in x["percentile95"])
    values = list(range(24))
    assert paired_interval(values, 7, 200) == paired_interval(values, 7, 200)
