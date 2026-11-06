
import torch
import torch.nn as nn
import torch.nn.functional as F
import geotorch
import numpy as np


class CustomLinear(nn.Linear):

    def __init__(self, in_features, out_features, *, bias=True, weight_init=None, device=None, dtype=None, clipping_value=None):
        super().__init__(in_features, out_features, bias, device, dtype)

        self.p = in_features
        self.q = out_features
        self._weight_size = self.weight.size()

        if weight_init is None:
            self.weight.data = self.weight.data.flatten()
        else:
            self.weight.data = weight_init.flatten()

        if clipping_value is not None:
            w = self.weight.data
            w = w.clamp(-clipping_value / 2, clipping_value / 2)
            self.weight.data = w

    def forward(self, input):
        return F.linear(input, self.weight.view(-1, input.shape[1]), self.bias)


class CostBase(nn.Module):

    def __init__(self):
        super().__init__()


class InnerGWLinear(CostBase):

    def __init__(self, p, q, *, weight_init=None, device=None, scale=None, trainable_scale=False):
        super().__init__()

        self.P = CustomLinear(p, q, bias=False, weight_init=weight_init).to(device)

        if scale is None:
            scale = np.sqrt(min(p, q))

        self.scale = nn.Parameter(torch.tensor(scale, device=device), requires_grad=trainable_scale)

        geotorch.sphere(self.P, 'weight')

    def forward(self, x, y):
        x, y = x.flatten(1), y.flatten(1)
        Px = self.scale * self.P(x)
        return F.mse_loss(Px, y)

    def apply_P(self, x):
        return self.scale * self.P(x)

    @property
    def matrix(self):
        return self.scale * self.P.weight.data.view(self.P._weight_size).T
