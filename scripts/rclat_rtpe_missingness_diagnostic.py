"""Bounded attribution of XPU input-gradient nonfiniteness; no updates."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def main():
    root = Path("outputs/rclat_rtpe_xpu_missingness_v1")
    root.mkdir(exist_ok=False)
    variants = [
        ("nan_inputs_nan_targets", False, False),
        ("zero_inputs_nan_targets", True, False),
        ("nan_inputs_zero_targets", False, True),
        ("zero_inputs_zero_targets", True, True),
    ]
    (root / "plan.json").write_text(
        json.dumps(
            dict(
                variants=variants,
                devices=[0, 1],
                optimizer_updates=0,
                original_failure_preserved=True,
            ),
            indent=2,
        )
    )
    spec = importlib.util.spec_from_file_location("prep", "scripts/rclat_rtpe_protocol_prepare.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    with np.load(
        "outputs/rclat_rtpe1_protocol_v1/training_numerical_fixtures.npz", allow_pickle=False
    ) as f:
        d = {k: f[k] for k in f.files}
    torch.set_num_threads(2)
    rows = []
    for device in [0, 1]:
        for name, zero_x, zero_y in variants:
            x = torch.from_numpy(d["x_2"]).clone()
            v = torch.from_numpy(d["valid_2"])
            y = torch.from_numpy(d["y_2"]).clone()
            m = torch.from_numpy(d["observed_2"])
            if zero_x:
                x[~v] = 0
            if zero_y:
                y[~m] = 0
            w = torch.from_numpy(d["population_weights"])
            indices = torch.from_numpy(d["indices_2"])
            torch.manual_seed(17)
            model = prep.TemporalPredictionEngine(
                64, hidden=64, scales=(1, 4, 16), horizons=(4, 8, 16)
            )
            gpu = copy.deepcopy(model).to(f"xpu:{device}")
            a = prep.run_grad(model, x, v, y, m, w, indices)
            b = prep.run_grad(
                gpu,
                x.to(f"xpu:{device}"),
                v.to(f"xpu:{device}"),
                y.to(f"xpu:{device}"),
                m.to(f"xpu:{device}"),
                w,
                indices,
            )
            gradient = b[-1]
            nonfinite = ~torch.isfinite(gradient)
            row = dict(
                device=device,
                variant=name,
                input_gradient_nonfinite=int(nonfinite.sum()),
                nonfinite_on_observed_inputs=int((nonfinite & v).sum()),
                input_gradient_relative_error=None
                if nonfinite.any()
                else float((a[-1].double() - gradient.double()).norm() / a[-1].double().norm()),
                missing_input_nonfinite=int((nonfinite & ~v).sum()),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
            del gpu
            torch.xpu.empty_cache()
    (root / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
