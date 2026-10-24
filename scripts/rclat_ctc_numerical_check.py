"""Fixed-input CPU/XPU numerical and device-transfer gradient verification."""

import copy
import json
from pathlib import Path

import torch

from bioacoustic_pipeline.representation.rclat_ctc import CausalCTC, ctc_objective, encode_text


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    checks = []
    for arm, dimension in [("mel", 64), ("physical", 12), ("combined", 76), ("pretrained", 768)]:
        torch.manual_seed(123)
        reference = CausalCTC(arm, dimension)
        x = torch.randn(2, 40, dimension)
        mask = torch.ones_like(x, dtype=torch.bool)
        lengths = torch.tensor([35, 40])
        targets = [encode_text("A B"), encode_text("THE")]
        logits, _ = reference(x, mask, lengths)
        loss = ctc_objective(logits, lengths, targets)
        loss.backward()
        for index in range(2):
            model = copy.deepcopy(reference).to(f"xpu:{index}")
            model.zero_grad()
            actual, _ = model(x.to(f"xpu:{index}"), mask.to(f"xpu:{index}"), lengths)
            measured = ctc_objective(actual, lengths, targets)
            measured.backward()
            torch.testing.assert_close(actual.cpu(), logits, atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(measured, loss, atol=2e-5, rtol=2e-4)
            largest = 0.0
            for name, parameter in model.named_parameters():
                expected = dict(reference.named_parameters())[name].grad
                if parameter.grad is None or expected is None:
                    raise ValueError("Disconnected gradient")
                torch.testing.assert_close(parameter.grad.cpu(), expected, atol=2e-5, rtol=2e-4)
                largest = max(largest, float((parameter.grad.cpu() - expected).abs().max()))
            checks.append(
                {
                    "arm": arm,
                    "device": f"xpu:{index}",
                    "max_logit_absolute_difference": float(
                        (actual.cpu() - logits).detach().abs().max()
                    ),
                    "loss_absolute_difference": float((measured - loss).detach().abs()),
                    "max_gradient_absolute_difference": largest,
                    "all_parameter_gradients_connected": True,
                }
            )
    output = Path("outputs/rclat_ctc_numerical_check.json")
    if output.exists():
        raise RuntimeError("Verification record already exists")
    output.write_text(
        json.dumps(
            {
                "checks": checks,
                "atol": 2e-5,
                "rtol": 2e-4,
                "precision": "float32",
                "seed": 123,
                "torch": torch.__version__,
                "strict_deterministic_algorithms": True,
                "bitwise_CPU_XPU_equivalence_claimed": False,
                "scientific_training_performed": False,
            },
            indent=2,
        )
        + "\n"
    )
    print("All eight fixed-input CPU/XPU comparisons passed, including every parameter gradient.")


if __name__ == "__main__":
    main()
