import copy
import numpy as np
import pytest
from bioacoustic_pipeline.diagnostics.rclat_rtpe_replication import (
    seed_spec,
    replication_summary,
    unlock_replication,
)

PAIRS = [
    (101, 2026102311),
    (211, 2026102312),
    (307, 2026102313),
    (401, 2026102314),
    (503, 2026102315),
]


def baseline():
    return dict(
        gru=dict(seed=17, sampling="old", updates=5000, batch_origins=128, validation_every=250),
        uncertainty=dict(seed=2026102302, replicates=200),
        ridge=dict(alpha_candidates=[0.1, 1]),
    )


def test_seed_variants_only_change_registered_stochastic_streams():
    base = baseline()
    original = copy.deepcopy(base)
    spec = seed_spec(base, initialization_seed=101, sampler_seed=2026102311)
    assert base == original
    assert spec["gru"]["seed"] == 101 and spec["gru"]["sampler_seed"] == 2026102311
    assert spec["gru"]["updates"] == 5000 and spec["ridge"] == base["ridge"]
    with pytest.raises(ValueError):
        seed_spec(base, initialization_seed=True, sampler_seed=1)


def test_global_unlock_requires_all_five_completed_and_selected_without_seed_dropping():
    complete = {
        str(seed): dict(
            completed_updates=5000, gru_step=1750, validation_steps=list(range(250, 5001, 250))
        )
        for seed, _ in PAIRS
    }
    unlock_replication(complete, pairs=PAIRS, updates=5000, interval=250)
    with pytest.raises(ValueError):
        unlock_replication(
            dict(list(complete.items())[:-1]), pairs=PAIRS, updates=5000, interval=250
        )
    complete["101"]["completed_updates"] = 4999
    with pytest.raises(ValueError):
        unlock_replication(complete, pairs=PAIRS, updates=5000, interval=250)


def test_consistency_rule_uses_every_seed_and_keeps_axes_separate():
    ridge = {str(i): 2.0 for i in range(4)}
    runs = {
        str(seed): {str(i): 1.0 + k * 0.01 + i * 0.02 for i in range(4)}
        for k, (seed, _) in enumerate(PAIRS)
    }
    result = replication_summary(ridge, runs, pairs=PAIRS, bootstrap_seed=9, replicates=200)
    assert result["finite_seed_replication_gate"] == "PASS"
    assert result["seeds_passed_original_criterion"] == 5
    assert result["between_seed_macro_sample_SD"] > 0
    assert result["between_speaker_seed_mean_sample_SD"] > 0
    assert result["seed_speaker_interaction_RMS"] < 1e-12
    assert "joint_interval95" not in result
    runs["503"] = {str(i): 3.0 for i in range(4)}
    result = replication_summary(ridge, runs, pairs=PAIRS, bootstrap_seed=9, replicates=200)
    assert result["finite_seed_replication_gate"] == "FAIL"
    assert result["seeds_passed_original_criterion"] == 4
    assert result["mean_difference"] > 0


def test_no_missing_seeds_speakers_or_nonfinite_cells_can_be_silently_removed():
    ridge = {"a": 2.0, "b": 2.0}
    runs = {str(s): {"a": 1.0, "b": 1.0} for s, _ in PAIRS}
    bad = copy.deepcopy(runs)
    del bad["101"]["b"]
    with pytest.raises(ValueError):
        replication_summary(ridge, bad, pairs=PAIRS, bootstrap_seed=9, replicates=20)
    bad = copy.deepcopy(runs)
    bad["101"]["b"] = np.nan
    with pytest.raises(ValueError):
        replication_summary(ridge, bad, pairs=PAIRS, bootstrap_seed=9, replicates=20)
    bad = copy.deepcopy(runs)
    del bad["503"]
    with pytest.raises(ValueError):
        replication_summary(ridge, bad, pairs=PAIRS, bootstrap_seed=9, replicates=20)
