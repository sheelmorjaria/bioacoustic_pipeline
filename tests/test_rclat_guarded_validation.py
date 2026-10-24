from bioacoustic_pipeline.diagnostics.rclat_guarded_validation import fault_schedule, fault_fixture


def test_fault_schedule_covers_roles_positions_layouts_and_support_contracts():
    rows = fault_schedule()
    assert len(rows) == 288 and len({r["name"] for r in rows}) == 288
    assert {r["batch"] for r in rows} == {4, 128}
    assert {r["layout"] for r in rows} == {"contiguous", "strided", "offset7"}
    assert {r["location"] for r in rows if "location" in r} == {"first", "middle", "last"}
    assert sum(r["expected_accept"] for r in rows) == 108


def test_generated_faults_have_independent_finite_observed_semantics():
    import numpy as np

    for row in fault_schedule():
        data = fault_fixture(row)
        mask = data["mask"].numpy()
        observed = mask.reshape(len(mask), -1)
        if row["family"] == "input":
            values = data["values"].numpy()
            accept = bool(mask.any(-1).all() and np.isfinite(np.where(mask, values, 0)).all())
        else:
            p = data["prediction"].numpy()
            y = data["target"].numpy()
            accept = bool(
                observed.any(1).all()
                and np.isfinite(np.where(mask, p, 0)).all()
                and np.isfinite(np.where(mask, y, 0)).all()
            )
        assert accept == row["expected_accept"], row
