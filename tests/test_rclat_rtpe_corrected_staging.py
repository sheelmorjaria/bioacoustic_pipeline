import json

import pytest
import torch
from bioacoustic_pipeline.diagnostics.rclat_rtpe_replication_corrected import (
    train_seed,
    evaluate_seed,
)
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha

PAIRS = [
    (101, 2026102311),
    (211, 2026102312),
    (307, 2026102313),
    (401, 2026102314),
    (503, 2026102315),
]


def data(seed, prefix):
    x = torch.randn(8, 4, 2, generator=torch.Generator().manual_seed(seed))
    return dict(
        x=x,
        v=torch.ones_like(x, dtype=torch.bool),
        y=x[:, -1:].clone(),
        m=torch.ones(8, 1, 2, dtype=torch.bool),
        weights=torch.full((8,), 0.125, dtype=torch.float64),
        ids=[
            dict(uid=f"{prefix}{i // 2}", speaker=f"{prefix}{i // 4}", origin=i % 2)
            for i in range(8)
        ],
    )


def spec(seed, sampler):
    return dict(
        horizons=[4],
        primary_horizon=4,
        ridge=dict(alpha_candidates=[0.1]),
        gru=dict(
            seed=seed,
            sampler_seed=sampler,
            updates=2,
            batch_origins=4,
            hidden=4,
            validation_every=2,
            checkpoint_every=2,
            learning_rate=0.001,
            betas=[0.9, 0.999],
            eps=1e-8,
            weight_decay=0.0001,
            gradient_clip_norm=5.0,
        ),
        uncertainty=dict(seed=9, replicates=20),
        runtime=dict(
            max_walltime_seconds=60, peak_host_RSS_GiB=8, post_fit_ridge_residual_max=1e-8
        ),
    )


def test_all_five_training_stages_precede_any_evaluation(tmp_path):
    lock = tmp_path / "global_lock.json"
    opened = []

    def evaluation():
        opened.append(True)
        return data(3, "test")

    with pytest.raises(ValueError, match="sealed"):
        evaluate_seed(spec(*PAIRS[0]), tmp_path / "seed101", evaluation, lock, device="cpu")
    assert not opened
    trainings = {}
    for seed, sampler in PAIRS:
        directory = tmp_path / f"seed{seed}"
        train_seed(
            spec(seed, sampler),
            data(1, "train"),
            data(2, "val"),
            directory,
            device="cpu",
            protocol_sha256="synthetic",
        )
        assert (directory / "TRAINING_COMPLETE.json").is_file()
        assert not (directory / "result.json").exists()
        trainings[str(seed)] = dict(
            path=str(directory),
            selection_sha256=sha(directory / "selection.json"),
            complete_sha256=sha(directory / "TRAINING_COMPLETE.json"),
        )
    lock.write_text(json.dumps(dict(pairs=PAIRS, updates=2, interval=2, trainings=trainings)))
    result = evaluate_seed(spec(*PAIRS[0]), tmp_path / "seed101", evaluation, lock, device="cpu")
    assert len(opened) == 1 and result["completed_updates"] == 2
    assert (tmp_path / "seed101" / "result.json").exists()
    with pytest.raises(FileExistsError):
        evaluate_seed(spec(*PAIRS[0]), tmp_path / "seed101", evaluation, lock, device="cpu")


def test_sampler_stream_is_taken_from_seed_spec(tmp_path):
    directory = tmp_path / "seed"
    train_seed(
        spec(101, 55),
        data(1, "train"),
        data(2, "val"),
        directory,
        device="cpu",
        protocol_sha256="synthetic",
    )
    records = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
    gen = torch.Generator().manual_seed(55)
    assert records[0]["indices"] == torch.randint(8, (4,), generator=gen).tolist()


def test_failure_is_preserved_without_completion_or_retry(tmp_path):
    directory = tmp_path / "failed"
    broken = spec(*PAIRS[0])
    broken["runtime"]["max_walltime_seconds"] = -1
    with pytest.raises(TimeoutError):
        train_seed(
            broken,
            data(1, "train"),
            data(2, "val"),
            directory,
            device="cpu",
            protocol_sha256="synthetic",
        )
    failure = json.loads((directory / "FAILURE.json").read_text())
    assert failure["automatic_retry"] is False
    assert not (directory / "TRAINING_COMPLETE.json").exists()
    assert not (directory / "selection.json").exists()
    with pytest.raises(FileExistsError):
        train_seed(
            broken,
            data(1, "train"),
            data(2, "val"),
            directory,
            device="cpu",
            protocol_sha256="synthetic",
        )
