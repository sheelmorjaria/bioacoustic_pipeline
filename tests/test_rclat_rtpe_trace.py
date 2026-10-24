import json

import pytest
import torch

from bioacoustic_pipeline.diagnostics.rclat_rtpe_trace import CaseRecorder, NumericalTrace


def test_expected_missing_values_distinguished_from_observed_nonfinite(tmp_path):
    r = CaseRecorder(tmp_path / "case", seed=101, case=2)
    x = torch.tensor([1.0, float("nan"), float("inf")])
    m = torch.tensor([True, False, True])
    info = r.tensor("fixture.targets", x, observed=m)
    assert info["host_nonfinite"] == 2 and info["observed_nonfinite"] == 1
    assert info["expected_masked_nonfinite"] == 1
    r.finish("FAILED", {"reason": "fixture"})
    result = json.loads((tmp_path / "case" / "result.json").read_text())
    assert result["seed"] == 101 and result["case"] == 2
    with pytest.raises(FileExistsError):
        r.finish("PASSED", {})


def test_operation_trace_identifies_nonfinite_math_without_changing_gradients(tmp_path):
    r = CaseRecorder(tmp_path / "case", seed=1, case=0)
    a = torch.tensor([1.0, 2.0], requires_grad=True)
    with NumericalTrace(r, "forward"):
        loss = (a.square() * 3).sum()
    loss.backward()
    torch.testing.assert_close(a.grad, torch.tensor([6.0, 12.0]), rtol=0, atol=0)
    with NumericalTrace(r, "probe"):
        _ = a / 0
    r.finish("FAILED", {})
    rows = [
        json.loads(line) for line in (tmp_path / "case" / "events.jsonl").read_text().splitlines()
    ]
    bad = [row for row in rows if row.get("operation") and row.get("host_nonfinite", 0)]
    assert any("div" in row["operation"] for row in bad)
    assert all(row["seed"] == 1 and row["case"] == 0 for row in rows)
    assert (tmp_path / "case" / "first_nonfinite.pt").is_file()


def test_tensor_logging_preserves_model_output_and_input_gradients(tmp_path):
    torch.manual_seed(9)
    model = torch.nn.Linear(3, 2).double()
    a = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    original = model(a).square().sum()
    gradient = torch.autograd.grad(original, a)[0]
    r = CaseRecorder(tmp_path / "case", seed=9, case=0)
    b = a.detach().clone().requires_grad_()
    with NumericalTrace(r, "forward"):
        output = model(b)
    r.tensor("forecast", output)
    with NumericalTrace(r, "backward"):
        output.square().sum().backward()
    r.finish("PASSED", {})
    torch.testing.assert_close(original, output.square().sum(), rtol=0, atol=0)
    torch.testing.assert_close(gradient, b.grad, rtol=0, atol=0)


def test_checkpoint_hooks_ignore_callback_return_and_preserve_model_values():
    from bioacoustic_pipeline.diagnostics.rclat_rtpe_trace import attach_checkpoint_hooks
    from bioacoustic_pipeline.representation.rclat_rtpe_pool import PoolTemporalPredictionEngine

    torch.manual_seed(17)
    model = PoolTemporalPredictionEngine(3, hidden=8, scales=(1, 4), horizons=(4, 8))
    x = torch.randn(2, 8, 3)
    m = torch.ones_like(x, dtype=torch.bool)
    expected = model(x, m)["forecast"]
    names = []

    def callback(name, tensor):
        names.append(name)
        return {"must_not_replace_outputs": True}

    handles = attach_checkpoint_hooks(model, callback)
    actual = model(x, m)["forecast"]
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert names == [
        "encoder_input",
        "GRU_sequence",
        "GRU_hidden",
        "forecast_head_input",
        "forecast_head_output",
    ]


def test_diagnostic_case_executes_with_real_observer_wiring_on_cpu(tmp_path):
    import importlib.util
    from types import SimpleNamespace

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    entry = load("trace_entry", "scripts/rclat_rtpe_numerical_trace.py")
    helper = load("reference", "scripts/rclat_rtpe_protocol_prepare_v2.py")
    torch.set_num_threads(2)
    torch.manual_seed(31)
    x = torch.randn(4, 32, 64)
    v = torch.ones_like(x, dtype=torch.bool)
    y = torch.randn(4, 3, 64)
    m = torch.ones_like(y, dtype=torch.bool)
    v[:, 10, 2] = False
    x[~v] = float("nan")
    m[0, 0, 2] = False
    y[~m] = float("nan")
    data = {
        "x_0": x.numpy(),
        "valid_0": v.numpy(),
        "y_0": y.numpy(),
        "observed_0": m.numpy(),
        "population_weights": torch.full((8,), 0.125, dtype=torch.float64).numpy(),
        "indices_0": torch.arange(4).numpy(),
    }
    recorder = CaseRecorder(tmp_path / "case", seed=101, case=0)
    status, details = entry.case_run(
        101, 0, data, SimpleNamespace(run_grad=helper.run_grad), recorder, device="cpu"
    )
    recorder.finish(status, details)
    assert status == "PASSED_INSTRUMENTED", details
    assert (tmp_path / "case" / "checkpoints.pt").is_file()
