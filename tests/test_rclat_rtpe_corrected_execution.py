import ast
import inspect
import importlib.util
import json
import pytest
import torch
from bioacoustic_pipeline.diagnostics import rclat_rtpe_replication_corrected as corrected
from bioacoustic_pipeline.diagnostics import rclat_rtpe_replication_execution as original


def test_only_guard_bindings_change_training_and_evaluation():
    for name in ["train_seed", "evaluate_seed", "assert_global_lock"]:
        assert ast.dump(ast.parse(inspect.getsource(getattr(corrected, name)))) == ast.dump(
            ast.parse(inspect.getsource(getattr(original, name)))
        )
    assert corrected.PoolTemporalPredictionEngine.__name__ == "GuardedPoolTemporalPredictionEngine"
    assert corrected.minibatch_mse.__name__ == "guarded_minibatch_mse"


def test_execute_requires_pinned_explicit_authorisation(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "corrected_runner", "scripts/rclat_rtpe_replication_corrected.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.ROOT = tmp_path
    for name in ["protocol.json", "execution_manifest.json"]:
        (tmp_path / name).write_text("{}")
    with pytest.raises(ValueError, match="authorisation"):
        runner.require_authorization()
    (tmp_path / "AUTHORIZATION.json").write_text(
        json.dumps(
            dict(
                authorized=True,
                protocol_sha256="wrong",
                execution_sha256="wrong",
                action="execute_five_seed_scientific_replication",
            )
        )
    )
    with pytest.raises(ValueError, match="authorisation"):
        runner.require_authorization()


def test_guarded_preflight_rejects_observed_invalid_and_accepts_masked(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "corrected_runner", "scripts/rclat_rtpe_replication_corrected.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    model = corrected.PoolTemporalPredictionEngine(2, hidden=4, scales=(1, 4, 16), horizons=(4,))
    x = torch.ones(2, 4, 2)
    v = torch.ones_like(x, dtype=torch.bool)
    y = torch.ones(2, 1, 2)
    m = torch.ones_like(y, dtype=torch.bool)
    w = torch.tensor([0.5, 0.5], dtype=torch.float64)
    idx = torch.arange(2)
    y[0, 0, 0] = float("nan")
    m[0, 0, 0] = False
    assert all(torch.isfinite(z).all() for z in runner.run_grad(model, x, v, y, m, w, idx))
    m[0, 0, 0] = True
    with pytest.raises(ValueError, match="targets"):
        runner.run_grad(model, x, v, y, m, w, idx)
