import torch
import torch.nn as nn
import torch.nn.functional as F
import geotorch

import numpy as np
import math

from functools import partial

class CustomLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, *,
                 bias: bool = True,
                 weight_init=None,
                 device=None,
                 dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self._weight_size = self.weight.size()
        if weight_init is None:
            self.weight.data = self.weight.data.flatten()
        else:
            self.weight.data = weight_init.flatten()

    def forward(self, input):
        return F.linear(input, self.weight.view(self._weight_size), self.bias)

class InnerGW_base(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class InnerGW_linear(InnerGW_base):
    def __init__(self, p: int, q: int, *,
                 weight_init=None,
                 device=None,
                 scale=None,
                 trainable_scale=False):
        super().__init__()
        self.P = CustomLinear(p, q, bias=False, weight_init=weight_init).to(device)
        
        if scale is None:
            scale = np.sqrt(min(p, q))
        self.scale = nn.Parameter(torch.tensor(scale, device=device), requires_grad=trainable_scale)

        geotorch.sphere(self.P, "weight")

    def forward(self, x, y):
        x, y = x.flatten(1), y.flatten(1)
        Px = self.scale * self.P(x)
        return F.mse_loss(Px, y)

    @property
    def matrix(self):
        return self.scale * self.P.weight.data.view(self.P._weight_size).T