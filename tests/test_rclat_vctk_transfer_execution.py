import json
import numpy as np
import pytest
from bioacoustic_pipeline.diagnostics.rclat_vctk_transfer_execution import (
    speaker_scores,
    run_inference_only,
    score_predictions,
)

SEEDS = [101, 211, 307, 401, 503]


def fixture():
    ids = [
        dict(speaker=s, event=f"{s}_{i // 2}", origin=i % 2)
        for s in ["p225", "s5"]
        for i in range(4)
    ]
    target = np.zeros((8, 3, 64), dtype=np.float32)
    mask = np.ones_like(target, dtype=bool)
    mask[:, 0] = False

    def predict(seed):
        ridge = np.ones_like(target)
        gru = np.full_like(target, 0.2)
        ridge[:, 0] = np.nan
        gru[:, 0] = np.inf
        return dict(ridge=ridge, gru=gru, targets=target, observed=mask)

    return ids, predict


def test_hierarchical_mean_not_flat_origin_mean():
    ids = [
        dict(speaker="a", event="a1"),
        dict(speaker="a", event="a1"),
        dict(speaker="a", event="a2"),
        dict(speaker="b", event="b1"),
    ]
    scores = speaker_scores(np.array([0.0, 4.0, 10.0, 0.0]), ids)
    assert scores == {"a": 6.0, "b": 0.0}


def test_all_five_lock_and_disabled_40ms_and_bootstrap(tmp_path):
    ids, predict = fixture()
    with pytest.raises(ValueError, match="locked"):
        score_predictions(tmp_path / "missing", SEEDS, ids, bootstrap_seed=17, replicates=100)
    result = run_inference_only(
        tmp_path / "run", SEEDS, ids, predict, bootstrap_seed=17, replicates=100
    )
    assert result["all_five_transfer_gate"] == "PASS"
    assert len(result["per_seed"]) == 5 and result["mean_primary_difference"] == pytest.approx(0.96)
    for row in result["per_seed"].values():
        np.testing.assert_allclose(row["80ms"]["interval95"], [0.96, 0.96], atol=1e-7)
    with pytest.raises(FileExistsError):
        run_inference_only(tmp_path / "run", SEEDS, ids, predict, bootstrap_seed=17)


def test_incomplete_prediction_failure_preserved_no_scoring_or_retry(tmp_path):
    ids, predict = fixture()

    def failure(seed):
        if seed == 307:
            raise RuntimeError("injected")
        return predict(seed)

    root = tmp_path / "run"
    with pytest.raises(RuntimeError):
        run_inference_only(root, SEEDS, ids, failure, bootstrap_seed=17)
    assert (
        (root / "FAILURE.json").exists()
        and not (root / "prediction_lock.json").exists()
        and not (root / "result.json").exists()
    )
    assert json.loads((root / "FAILURE.json").read_text())["automatic_retry"] is False
    with pytest.raises(FileExistsError):
        run_inference_only(root, SEEDS, ids, predict, bootstrap_seed=17)


def test_nonfinite_retained_or_unsupported_target_rejected(tmp_path):
    ids, predict = fixture()

    def invalid(seed):
        out = predict(seed)
        out["gru"][0, 1, 0] = np.nan
        return out

    with pytest.raises(ValueError, match="finite"):
        run_inference_only(tmp_path / "invalid", SEEDS, ids, invalid, bootstrap_seed=17)

    def unsupported(seed):
        out = predict(seed)
        out["observed"][0, 2] = False
        return out

    with pytest.raises(ValueError, match="support"):
        run_inference_only(tmp_path / "empty", SEEDS, ids, unsupported, bootstrap_seed=17)


def test_prediction_hash_tamper_blocks_scoring(tmp_path):
    ids, predict = fixture()
    root = tmp_path / "run"
    run_inference_only(root, SEEDS, ids, predict, bootstrap_seed=17, replicates=100)
    with (root / "seed_101_predictions.npz").open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        score_predictions(root, SEEDS, ids, bootstrap_seed=17, replicates=100)
