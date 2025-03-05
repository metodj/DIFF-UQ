import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def postprocess_la_adm(output):
    return torch.split(output, 3, dim=1)[0]  # get rid of CFG channels


def preprocess_la_adm(x, y, betas, num_timesteps, device, dtype=torch.float32):

    x = x.to(device, dtype=dtype, non_blocking=True)
    y = y.to(device, non_blocking=True)
    t = torch.randint(low=0, high=num_timesteps, size=(x.shape[0],), device=device)
    e = torch.randn_like(x, device=device, dtype=dtype)
    b = betas.to(device, dtype=dtype)
    a = (1 - b).cumprod(dim=0)[t]
    xt = x * a[:, None, None, None].sqrt() + e * (1.0 - a[:, None, None, None]).sqrt()

    return (t, xt, y), e
