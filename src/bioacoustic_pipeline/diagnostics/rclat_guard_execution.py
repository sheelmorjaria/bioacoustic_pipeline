"""Fixed engineering AdamW sequence and CPU-readable optimiser state."""

import torch


def optimizer(parameters):
    return torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.0001, foreach=False, fused=False)


def state_vector(opt, key):
    values = [state[key].detach().cpu().flatten() for state in opt.state.values() if key in state]
    return torch.cat(values) if values else torch.empty(0)
