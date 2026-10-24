import hashlib
import numpy as np
import pytest
from bioacoustic_pipeline.diagnostics.rclat_vctk_transfer_sampling import select_inventory


def rows():
    values = []
    for speaker in ["p225", "s5"]:
        for i in range(5):
            origins = np.arange(32, 64, dtype="<i8")
            values.append(
                dict(
                    speaker=speaker,
                    event=f"{speaker}_{i:03d}",
                    reason="eligible",
                    eligible_origin_count=32,
                    eligible_origin_sha256=hashlib.sha256(origins.tobytes()).hexdigest(),
                    first_eligible_origin=32,
                    last_eligible_origin=63,
                    member=f"{speaker}/{speaker}_{i:03d}_mic2.flac",
                )
            )
    return values


def test_selection_is_deterministic_balanced_and_blind_to_scores():
    a = rows()
    first = select_inventory(a, salt="fixture", recordings=3, origins=4)
    for row in a:
        row["forecast_loss"] = -1000
    second = select_inventory(list(reversed(a)), salt="fixture", recordings=3, origins=4)
    assert first == second and len(first) == 6
    assert all(len(r["selected_origins"]) == 4 for r in first)
    assert all(r["origin_weight"] == 1 / 24 for r in first)
    assert sum(r["recording_weight"] for r in first) == pytest.approx(1)


def test_insufficient_support_or_changed_origin_hash_abstains():
    with pytest.raises(ValueError, match="support"):
        select_inventory(rows(), salt="x", recordings=6, origins=4)
    a = rows()
    a[0]["eligible_origin_sha256"] = "bad"
    with pytest.raises(ValueError, match="hash"):
        select_inventory(a, salt="x", recordings=3, origins=4)
    a = rows()
    a.append(a[0].copy())
    with pytest.raises(ValueError, match="Duplicate"):
        select_inventory(a, salt="x", recordings=3, origins=4)
