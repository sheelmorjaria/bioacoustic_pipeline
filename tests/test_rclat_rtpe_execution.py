import hashlib

import numpy as np
import pytest
import torch

from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import (
    SelectionLedger,
    group_scores,
    load_dataset,
    execute_comparison,
)


def test_selection_requires_all_updates_candidates_and_validation_points():
    ledger = SelectionLedger(updates=6, interval=2, alphas=[0.1, 1.0])
    with pytest.raises(ValueError):
        ledger.lock([(0.1, 1), (1.0, 1)])
    with pytest.raises(ValueError):
        ledger.advance(2)
    for step in range(1, 7):
        ledger.advance(step)
        if step % 2 == 0:
            ledger.validate(step, 0.5 if step < 6 else 0.6)
    with pytest.raises(ValueError):
        ledger.lock([(0.1, 1)])
    selected = ledger.lock([(0.1, 1), (1.0, 1)])
    assert selected["gru_step"] == 2 and selected["ridge_alpha"] == 0.1
    assert selected["completed_updates"] == 6


def test_group_scores_equal_recordings_and_speakers_not_origins():
    ids = [
        dict(uid="a", speaker="A", origin=0),
        dict(uid="a", speaker="A", origin=1),
        dict(uid="b", speaker="A", origin=0),
        dict(uid="c", speaker="B", origin=0),
    ]
    r = group_scores(torch.tensor([[1.0], [3.0], [6.0], [4.0]]), ids)
    assert r["macro"] == [4.0] and r["speakers"] == {"A": [4.0], "B": [4.0]}
    with pytest.raises(ValueError):
        group_scores(torch.full((4, 1), float("nan")), ids)


def test_evaluation_loader_stays_sealed_before_selection(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Attempted sealed cache access")

    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError, match="sealed"):
        load_dataset([], split="test", history=4, horizons=[4], selection=None)


def dataset(seed, speakers):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(8, 4, 2, generator=gen)
    y = x[:, -1:, :].clone()
    ids = [dict(uid=f"r{i // 2}", speaker=speakers[i // 4], origin=i % 2) for i in range(8)]
    return dict(
        x=x,
        v=torch.ones_like(x, dtype=torch.bool),
        y=y,
        m=torch.ones_like(y, dtype=torch.bool),
        weights=torch.full((8,), 1 / 8, dtype=torch.float64),
        ids=ids,
    )


def test_synthetic_end_to_end_completion_lock_and_failure_preservation(tmp_path):
    spec = dict(
        horizons=[4],
        primary_horizon=4,
        ridge=dict(alpha_candidates=[0.1, 1.0]),
        gru=dict(
            updates=6,
            validation_every=2,
            checkpoint_every=2,
            hidden=4,
            seed=17,
            batch_origins=4,
            learning_rate=0.001,
            betas=[0.9, 0.999],
            eps=1e-8,
            weight_decay=0.0001,
            gradient_clip_norm=5.0,
        ),
        uncertainty=dict(seed=4, replicates=20),
        runtime=dict(
            max_walltime_seconds=60, peak_host_RSS_GiB=8, post_fit_ridge_residual_max=1e-8
        ),
    )
    train = dataset(1, ["trainA", "trainB"])
    val = dataset(2, ["valA", "valB"])
    test = dataset(3, ["testA", "testB"])
    opened = []

    def evaluation(selection):
        assert selection["completed_updates"] == 6
        assert (tmp_path / "run" / "selection.json").is_file()
        opened.append(True)
        return test

    result = execute_comparison(
        spec,
        train,
        val,
        evaluation,
        tmp_path / "run",
        device="cpu",
        protocol_sha256="synthetic-fixture",
    )
    assert len(opened) == 1 and result["completed_updates"] == 6
    assert result["final_gru_step"] == 6
    assert len(list((tmp_path / "run").glob("gru_*.pt"))) == 3
    assert len((tmp_path / "run" / "steps.jsonl").read_text().splitlines()) == 6
    assert (tmp_path / "run" / "result.json").is_file()
    with pytest.raises(FileExistsError):
        execute_comparison(
            spec,
            train,
            val,
            evaluation,
            tmp_path / "run",
            device="cpu",
            protocol_sha256="synthetic-fixture",
        )
    bad = dataset(1, ["trainA", "trainB"])
    bad["y"][0] = float("nan")
    with pytest.raises(ValueError):
        execute_comparison(
            spec,
            bad,
            val,
            lambda _: pytest.fail("Evaluation leaked"),
            tmp_path / "failed",
            device="cpu",
            protocol_sha256="synthetic-fixture",
        )
    assert (tmp_path / "failed" / "FAILURE.json").is_file()
    assert not (tmp_path / "failed" / "selection.json").exists()


def test_loader_rechecks_cache_hashes_and_exact_target_masks(tmp_path):
    path = tmp_path / "cache.npz"
    mel = np.arange(80, dtype=np.float32).reshape(40, 2)
    mask = np.ones_like(mel, dtype=bool)
    np.savez(path, mel=mel, mel_valid=mask)
    origins = [4, 6]
    targetmask = mask[np.array(origins)[:, None] + np.array([4])[None]]
    row = dict(
        uid="a",
        speaker="s",
        split="train",
        cache_path=str(path),
        cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        selected_origins=origins,
        origin_weight_within_split=0.5,
        selected_target_mask_sha256=hashlib.sha256(targetmask.tobytes()).hexdigest(),
    )
    data = load_dataset([row], split="train", history=4, horizons=[4], selection=None)
    assert data["x"].shape == (2, 4, 2)
    assert torch.equal(data["y"][:, 0], torch.from_numpy(mel[np.array(origins) + 4]))
    row["cache_sha256"] = "bad"
    with pytest.raises(ValueError, match="hash"):
        load_dataset([row], split="train", history=4, horizons=[4], selection=None)


def test_resource_failure_cannot_open_evaluation_or_be_relabelled(tmp_path):
    spec = dict(
        horizons=[4],
        primary_horizon=4,
        ridge=dict(alpha_candidates=[0.1]),
        gru=dict(updates=2, validation_every=2, checkpoint_every=2, seed=17),
        runtime=dict(max_walltime_seconds=0, peak_host_RSS_GiB=8),
    )
    data = dataset(1, ["A", "B"])
    with pytest.raises(TimeoutError):
        execute_comparison(
            spec,
            data,
            data,
            lambda _: pytest.fail("Evaluation leaked"),
            tmp_path / "limited",
            device="cpu",
            protocol_sha256="synthetic",
        )
    assert (tmp_path / "limited" / "FAILURE.json").is_file()
    assert not (tmp_path / "limited" / "result.json").exists()
