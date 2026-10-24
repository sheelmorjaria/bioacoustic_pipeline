import torch
from bioacoustic_pipeline.diagnostics.rclat_guard_execution import optimizer, state_vector


def test_fixed_adamw_policy_and_state_readout():
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    o = optimizer([p])
    assert o.param_groups[0]["lr"] == 0.001
    assert o.param_groups[0]["weight_decay"] == 0.0001
    p.grad = torch.ones_like(p)
    o.step()
    assert state_vector(o, "exp_avg").shape == (2,)
    assert torch.equal(state_vector(o, "exp_avg"), torch.tensor([0.1, 0.1]))
    assert state_vector(o, "step").tolist() == [1.0]
