import pytest

from bioacoustic_pipeline.diagnostics.rclat_bootstrap_calibration import (
    simulate,
    population_difference,
    fast_crossed_interval,
    wilson_interval,
)
from bioacoustic_pipeline.diagnostics.rclat_communication import crossed_interval


def scenario(**overrides):
    return {
        "speaker_sd": 0.8,
        "listener_sd": 0.8,
        "alternative": False,
        "uneven": False,
        **overrides,
    }


def test_accelerated_bootstrap_matches_existing_reference_kernel():
    d = simulate(scenario(), seed=9, speakers=3, listeners=6, min_raters=2, max_raters=4)
    a = crossed_interval(d["stimuli"], d["responses"], d["c1"], d["c2"], seed=17, replicates=61)
    b = fast_crossed_interval(
        d["stimuli"], d["responses"], d["c1"], d["c2"], seed=17, replicates=61
    )
    assert b["difference"] == pytest.approx(a["difference"], abs=1e-12)
    assert b["interval95"] == pytest.approx(a["interval95"], abs=1e-12)


def test_symmetric_null_and_nonzero_alternative_have_known_population_targets():
    assert population_difference(scenario()) == pytest.approx(0, abs=1e-12)
    assert population_difference(scenario(alternative=True)) > 0
    assert population_difference(scenario(), nodes=80) == pytest.approx(
        population_difference(scenario(), nodes=128), abs=1e-9
    )


def test_simulation_is_reproducible_with_unique_observed_listener_edges():
    a = simulate(scenario(uneven=True), seed=8, speakers=3, listeners=6, min_raters=2, max_raters=5)
    assert a == simulate(
        scenario(uneven=True), seed=8, speakers=3, listeners=6, min_raters=2, max_raters=5
    )
    edges = {(r["source_filename"], r["listener_id"]) for r in a["responses"]}
    assert len(edges) == len(a["responses"])
    assert len({r["speaker_id"] for r in a["stimuli"]}) == 3


def test_wilson_interval_preserves_monte_carlo_uncertainty():
    lo, hi = wilson_interval(190, 200)
    assert lo < 0.95 < hi and hi < 1
    lo, hi = wilson_interval(0, 200)
    assert lo == pytest.approx(0) and hi > 0
