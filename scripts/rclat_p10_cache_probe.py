"""Fresh-process installed-cache-control behaviour probe; synthetic only."""

import json
import os
import torch

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
device = "xpu:0"
x = torch.randn(4, 256, 417, device=device, requires_grad=True)
w = torch.randn(256, 1, 15, device=device, requires_grad=True)
for i in range(3):
    y = torch.nn.functional.conv1d(x, w, padding=7, groups=256)
    y.square().mean().backward()
    x.grad = None
    w.grad = None
    torch.xpu.synchronize()
print(
    json.dumps(
        dict(
            capacity_requested=os.environ.get("ONEDNN_PRIMITIVE_CACHE_CAPACITY"),
            verbose=os.environ.get("ONEDNN_VERBOSE"),
            device=device,
            torch=torch.__version__,
        )
    ),
    flush=True,
)
