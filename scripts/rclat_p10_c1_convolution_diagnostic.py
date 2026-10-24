"""Training-only read-only checkpoint/batch convolution failure probe; never optimizes."""

import json
import sys
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
import rclat_p10_train as original
from bioacoustic_pipeline.diagnostics.frozen_acoustic_transfer import frozen_scale
from bioacoustic_pipeline.representation.rclat_scale_v4 import ScaleCTC

arm = sys.argv[1]
device = sys.argv[2]
out = Path("outputs/rclat_p10_c1_convolution_diagnostic")
out.mkdir(exist_ok=True)
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
checkpoint = Path(f"checkpoints/rclat_p10_corrected_v1/{arm}/epoch_11.pt")
payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
model = ScaleCTC(
    64 if arm == "mel" else 76,
    width=256,
    layers=12,
    heads=4,
    expansion=1024,
    kernel=15,
    dropout=0.0,
)
# Infer frontend dimensions directly from the checkpoint schema.
if model.input_dim != payload["model"]["projection.weight"].shape[1] // 2:
    model = ScaleCTC(
        payload["model"]["projection.weight"].shape[1] // 2,
        width=256,
        layers=12,
        heads=4,
        expansion=1024,
        kernel=15,
        dropout=0.0,
    )
model.load_state_dict(payload["model"])
model.to(device).train()
journal = Path(f"outputs/rclat_p10_corrected_v1/{arm}_steps.jsonl")
context = json.loads(journal.read_text().splitlines()[-1])
scaler = json.loads(Path(f"outputs/rclat_p10_corrected_v1/{arm}_scaler.json").read_text())
arrays = [frozen_scale(*original.sequence(uid, arm), scaler) for uid in context["uids"]]
lengths = torch.tensor([len(x) for x, m in arrays])
values = pad_sequence([torch.from_numpy(x) for x, m in arrays], batch_first=True).to(device)
masks = pad_sequence([torch.from_numpy(m) for x, m in arrays], batch_first=True).to(device)
record = dict(
    arm=arm,
    device=device,
    context=context,
    checkpoint_sha256=original.protocol.sha(checkpoint),
    checkpoint_epoch=payload["epoch"],
    checkpoint_note="Epoch 11; six later updates are not checkpointed",
    input_shape=list(values.shape),
    lengths=lengths.tolist(),
    convolutions=[],
)


def hook(module, args):
    x = args[0]
    record["convolutions"].append(
        dict(
            shape=list(x.shape),
            stride=list(x.stride()),
            contiguous=x.is_contiguous(),
            finite=bool(torch.isfinite(x).all()),
        )
    )


for block in model.blocks:
    block.depth.register_forward_pre_hook(hook)
try:
    logits, output_lengths = model(values, masks, lengths)
    loss = logits.square().mean()
    loss.backward()
    if device.startswith("xpu"):
        torch.xpu.synchronize()
    record.update(
        ok=True,
        logits_finite=bool(torch.isfinite(logits).all()),
        gradients_finite=all(
            bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None
        ),
        output_lengths=output_lengths.tolist(),
    )
except Exception as exc:
    record.update(ok=False, error=repr(exc))
path = out / f"batch_{arm}_{device.replace(':', '')}.json"
path.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
