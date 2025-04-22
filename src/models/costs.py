import torch
import torch.nn as nn
import torch.nn.functional as F
import geotorch

import numpy as np
import math
from math import sqrt

from functools import partial
import random
import torch.nn.utils.parametrize as parametrize

class CustomLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, *,
                 bias: bool = True,
                 weight_init=None,
                 device=None,
                 dtype=None,
                 clipping_value=None):
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
            w = w.clamp(-clipping_value/2, clipping_value/2)
            self.weight.data = w

    def forward(self, input):
        #return F.linear(input, self.weight.view(self._weight_size), self.bias)
        return F.linear(input, self.weight.view(-1, input.shape[1]), self.bias)
        


class CostBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class InnerGW_linear(CostBase):
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
        
        #Px_norm = torch.norm(Px, dim=1)
        #y_norm = torch.norm(y, dim=1)
        #Discussion about this: https://discuss.pytorch.org/t/dot-product-batch-wise/9746/11
        #return -(Px*y).sum(-1) 

    @property
    def matrix(self):
        return self.scale * self.P.weight.data.view(self.P._weight_size).T
        
class EmbeddedGW_linear(CostBase):
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
        r1 = -sqrt(1/scale)
        r2 = -r1
        self.b = nn.Parameter((r1 - r2) * torch.rand(bs, q, device=device) + r2, requires_grad=True)
        geotorch.sphere(self.P, "weight")

    def forward(self, x, y):
        x, y = x.flatten(1), y.flatten(1)
        Px_b = self.scale * self.P(x) - self.b
        
        return F.mse_loss(Px_b, y) 
        
        #Px_norm = torch.norm(Px, dim=1)
        #y_norm = torch.norm(y, dim=1)
        #Discussion about this: https://discuss.pytorch.org/t/dot-product-batch-wise/9746/11
        #return -(Px*y).sum(-1) 

    @property
    def matrix(self):
        return self.scale * self.P.weight.data.view(self.P._weight_size).T

#Obtained from: https://pytorch.org/tutorials/intermediate/parametrizations.html
class Skew(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        
    def forward(self, X):
        X = X.view(self.p, self.p)
        A = X.triu(1)
        out = A - A.transpose(-1, -2)
        return out.view(self.p, self.p)

class CayleyMap(nn.Module):
    def __init__(self, p, q, diag_list, device='cpu'):
        super().__init__()
        self.p = p
        self.q = q

        self.register_buffer("Id", torch.eye(p, device=device))
        
        D = torch.eye(q, p).T#torch.diag(torch.tensor(diag_list)).to(torch.float32) @ torch.eye(p, q)
        D = D.to(device)
        self.register_buffer("D", D)

    def forward(self, X):
        # (I + X)(I - X)^{-1}
        X = X.view(self.p, self.p)
        out = torch.linalg.solve(self.Id - X, self.Id + X)
        out = out @ self.D
        
        return out.view(-1)

        
class CayleyCost(CostBase):
    def __init__(self, p: int, q: int, neg_eigvals=None, *,
                 
                 weight_init=None,
                 device=None,
                 scale=None,
                 trainable_scale=False):
        super().__init__()

        if neg_eigvals is None:
            neg_eigvals = 0
            
        self.p = p
        self.q = q
        
        self.P = CustomLinear(p, p, bias=False, weight_init=weight_init).to(device)

        diag_list = [-1]*neg_eigvals + [1]*(p - neg_eigvals)
        
        random.shuffle(diag_list)
        
        parametrize.register_parametrization(self.P, "weight", Skew(p), unsafe=True)
        parametrize.register_parametrization(self.P, "weight", CayleyMap(p, q, diag_list, device), unsafe=True)

        #matrix = self.P.weight.view(p,q)

        #assert self.verify_orthogonality(matrix, p, q) 
        
        if scale is None:
            scale = np.sqrt(min(p, q))
            
        self.scale = nn.Parameter(torch.tensor(scale, device=device), requires_grad=trainable_scale)

        
    def verify_orthogonality(self, matrix, p, q, tol=1e-5):
        matrix = matrix.reshape(p, q)
        identity_approx = matrix.T @ matrix
        identity = torch.eye(matrix.size(1), device=matrix.device, dtype=matrix.dtype)
    
        return torch.allclose(identity_approx, identity, atol=tol)
        
    def forward(self, x, y):
        x, y = x.flatten(1), y.flatten(1)
        #Px = self.scale * self.P(x) 
        Px =  self.P(x) 
        
        #assert self.verify_orthogonality(self.P.weight, self.p, self.q) 
        
        return F.mse_loss(Px, y) 
        

    @property
    def matrix(self):
        return self.scale * self.P.weight.data.view(self.P._weight_size).T
        
class SquaredGW_cost(CostBase):
    def __init__(self, p: int, q: int, *,
                 device=None,
                ):
        super().__init__()
        
        self.P = CustomLinear(p, q, bias=False, weight_init=None).to(device)

    def forward(self, x, y):
        x, y = x.flatten(1), y.flatten(1)
        Px = self.P(x).flatten(1)
        Px_norm = torch.norm(Px, dim=1)**2
        xy_norm = torch.norm(x, dim=1)**2 * torch.norm(y, dim=1)**2
        return F.mse_loss(Px, y) #- xy_norm.mean() - Px_norm.mean() 

    @property
    def matrix(self):
        return self.P.weight.data.view(self.P._weight_size).T

class CostModel(CostBase):
    def __init__(self, p: int, q: int, *,
                 weight_init=None,
                 device=None):
        
        super().__init__()

        self.scale = np.sqrt(min(p, q))
        self.clipping_value = None#np.sqrt(p * q)
        self.A = CustomLinear(p, q, bias=False, weight_init=weight_init, clipping_value=self.clipping_value).to(device)
        
        #geotorch.sphere(self.P, "weight")

    def forward(self, x, y):
        
        if self.clipping_value is not None:
            w = self.A.weight.data
            w = w.clamp(-self.clipping_value/2, self.clipping_value/2)
            self.A.weight.data = w
            
        out = -4*torch.norm(x, dim=-1)**2 * torch.norm(y, dim=-1)**2 - 32 * self.A(x) @ y.T
        
        return out

    @property
    def matrix(self):
        return self.scale * self.A.weight.data.view(self.A._weight_size).T