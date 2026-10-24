import json
import sys
import torch

arm = int(sys.argv[1])
torch.set_num_threads(2)
torch.manual_seed(0)
rows = []
for deterministic in (True, False):
    torch.use_deterministic_algorithms(deterministic)
    for length in (416, 417, 418, 420):
        x = torch.randn(4, 256, length, requires_grad=True)
        w = torch.randn(256, 1, 15, requires_grad=True)
        ref = torch.nn.functional.conv1d(x, w, padding=7, groups=256)
        ref.square().mean().backward()
        item = dict(device=arm, deterministic=deterministic, length=length)
        try:
            xx = x.detach().to(f"xpu:{arm}").requires_grad_()
            ww = w.detach().to(f"xpu:{arm}").requires_grad_()
            yy = torch.nn.functional.conv1d(xx, ww, padding=7, groups=256)
            yy.square().mean().backward()
            torch.xpu.synchronize(arm)
            item.update(
                ok=True,
                forward_rel=float((yy.cpu() - ref).norm() / ref.norm()),
                input_gradient_rel=float((xx.grad.cpu() - x.grad).norm() / x.grad.norm()),
                weight_gradient_rel=float((ww.grad.cpu() - w.grad).norm() / w.grad.norm()),
            )
        except Exception as e:
            item.update(ok=False, error=str(e))
        rows.append(item)
        print(json.dumps(item), flush=True)
json.dump(
    dict(torch=torch.__version__, results=rows),
    open(f"outputs/rclat_p10_c1_convolution_diagnostic/synthetic_xpu{arm}.json", "w"),
    indent=2,
)
