"""Dataset-free FP32 softmax backward comparison for this installed XPU runtime.

Random tensors only; no model fitting. Refuses to overwrite its output receipt.
"""

import argparse
import json
from pathlib import Path

import torch


def main(device):
    path = Path(f"outputs/rclat_p10_numerical/random_softmax_{device.replace(':', '_')}.json")
    if path.exists():
        raise RuntimeError("Reproducer output already exists")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.xpu.set_device(device)
    generator = torch.Generator().manual_seed(0)
    reports = []
    for length in [299, 320, 333, 334, 335, 336, 352, 384]:
        scores = 0.5 * torch.randn(4, 4, length, length, generator=generator)
        upstream = 0.01 * torch.randn(scores.shape, generator=generator)
        lengths = torch.tensor(
            [length, round(length * 0.76), round(length * 0.65), round(length * 0.83)]
        )
        valid = torch.arange(length)[None] < lengths[:, None]
        scores = scores.masked_fill(~valid[:, None, None, :], -10000)
        reference = None
        for target, mode in [("cpu", "native"), (device, "native"), (device, "composite")]:
            leaf = scores.to(target).clone().requires_grad_()
            if mode == "native":
                probabilities = leaf.softmax(-1)
            else:
                numerator = (leaf - leaf.amax(-1, keepdim=True)).exp()
                probabilities = numerator / numerator.sum(-1, keepdim=True)
            probabilities.backward(upstream.to(target))
            observed = leaf.grad.detach().cpu().double()
            if reference is None:
                reference = observed
            error = float(
                torch.linalg.vector_norm(observed - reference) / torch.linalg.vector_norm(reference)
            )
            reports.append(
                {
                    "shape": list(scores.shape),
                    "device": target,
                    "mode": mode,
                    "nonfinite_gradient_values": int((~torch.isfinite(observed)).sum()),
                    "relative_l2_gradient_error_vs_cpu": error
                    if torch.isfinite(torch.tensor(error))
                    else str(error),
                    "gradient_max_abs": float(observed.abs().max()),
                    "gradient_row_sum_max_abs": float(observed.sum(-1).abs().max()),
                }
            )
        print(json.dumps(reports[-2:]), flush=True)
    data = {
        "scope": "Random synthetic tensors only; no source audio, model weights or optimization",
        "torch": torch.__version__,
        "device": device,
        "device_properties": str(torch.xpu.get_device_properties(device)),
        "seed": 0,
        "dtype": "float32",
        "reports": reports,
    }
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--device", default="xpu:0")
    main(parser.parse_args().device)
