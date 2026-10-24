import json
import sys

import pytest

from bioacoustic_pipeline.diagnostics.rclat_mask_stages import (
    StageJournal,
    run_bounded,
    worker_case,
)


def test_each_stage_start_and_completion_is_durable_and_failures_are_localised(tmp_path):
    log = StageJournal(tmp_path / "stages.jsonl")
    assert log.step("transfer", lambda: 3) == 3
    with pytest.raises(RuntimeError):
        log.step("index", lambda: (_ for _ in ()).throw(RuntimeError("failure")))
    log.close()
    rows = [json.loads(line) for line in (tmp_path / "stages.jsonl").read_text().splitlines()]
    assert [(r["stage"], r["event"]) for r in rows] == [
        ("transfer", "started"),
        ("transfer", "completed"),
        ("index", "started"),
        ("index", "failed"),
    ]


def test_supervisor_stops_only_its_own_stalled_worker(tmp_path):
    result = run_bounded(
        [sys.executable, "-c", 'import time; print("started",flush=True); time.sleep(30)'],
        tmp_path / "console.log",
        timeout=0.3,
        cleanup=2,
    )
    assert result["timed_out"] and result["returncode"] is not None
    assert result["elapsed_seconds"] < 3
    completed = run_bounded(
        [sys.executable, "-c", 'print("complete")'], tmp_path / "other.log", timeout=2, cleanup=1
    )
    assert not completed["timed_out"] and completed["returncode"] == 0


def test_cpu_worker_validates_pre_operation_values_and_keeps_modes_separate(tmp_path):
    import torch

    x = torch.arange(768, dtype=torch.float32).reshape(4, 3, 64)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask.flatten()[0] = False
    x.flatten()[0] = float("inf")
    for operation in ["transfer", "index", "and", "where"]:
        directory = tmp_path / operation
        result = worker_case(x, mask, "strided", operation, directory, device="cpu")
        assert result["status"] == "PASS", result
        rows = [json.loads(line) for line in (directory / "stages.jsonl").read_text().splitlines()]
        stages = [r["stage"] for r in rows]
        assert (
            "pre_operation.values_roundtrip" in stages and "pre_operation.mask_roundtrip" in stages
        )
        if operation != "index":
            assert "boolean_index" not in stages
        if operation == "index":
            assert "candidate_and" not in stages and "candidate_where" not in stages
