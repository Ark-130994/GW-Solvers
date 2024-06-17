import numpy as np
import torch.nn as nn
import torch

from ..utils import nwise

__all__ = ["mlp"]


def mlp(input_size, output_size=1, *,
        hidden_size=None, num_layers=4,
        layer_sizes=None):
    if hidden_size is None:
        hidden_size = input_size
    if layer_sizes is None:
        layer_sizes = [hidden_size] * (num_layers - 1) + [output_size]
    modules: list[nn.Module] = [nn.Linear(input_size, layer_sizes[0])]

    for in_size, out_size in nwise(layer_sizes):
        modules.append(nn.LeakyReLU())
        modules.append(nn.Dropout(.1))
        modules.append(nn.Linear(in_size, out_size))

    return nn.Sequential(*modules)

class mlp_class(nn.Module):
    def __init__(self, input_size, output_size=1, *,
        hidden_size=None, num_layers=4,
        layer_sizes=None):
        super().__init__()
        
        if hidden_size is None:
            hidden_size = input_size
        if layer_sizes is None:
            layer_sizes = [hidden_size] * (num_layers - 1) + [output_size]
        modules: list[nn.Module] = [nn.Linear(input_size, layer_sizes[0])]

        for in_size, out_size in nwise(layer_sizes):
            modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(.1))
            modules.append(nn.Linear(in_size, out_size))

        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.model(x), 0, torch.ones_like(x)

class mlp_vae1(nn.Module):
    def __init__(self, input_size, output_size, latent_dim, hidden_size, num_layers=4, layers_size=None, device='cuda'):
        super().__init__()
        self.mlp = mlp(input_size, latent_dim, hidden_size=512, num_layers=4)
        
        self.mean_fc = nn.Linear(latent_dim, output_size)
        self.logvar_fc = nn.Linear(latent_dim, output_size)
        
        #self.z = nn.Parameter(torch.randn(128, 4, output_size, device=device))
        
    def forward(self, x):
        
        
        h1 = self.mlp(x)
        h_mean = self.mean_fc(h1)
        h_logvar = self.logvar_fc(h1)

        eps = torch.randn_like(h_logvar)
        h = h_mean + torch.exp(0.5 * h_logvar)*eps 
        return h, h_mean, h_logvar
    
class mlp_vae2(nn.Module):
    def __init__(self, input_size, output_size, latent_dim, hidden_size, num_layers=4, layers_size=None, device='cuda'):
        super().__init__()
        self.mlp = mlp(input_size, latent_dim, hidden_size=512, num_layers=4)
        
        self.mean_fc = nn.Linear(latent_dim, output_size)
        self.logvar_fc = nn.Linear(latent_dim, output_size)
        
        #self.z = nn.Parameter(torch.randn(128, 4, output_size, device=device))
        
    def forward(self, x, P, z):
        Px_train = x @ P
        Pxz_train = torch.cat([Px_train[:, None].repeat(1, 4, 1), z], dim=2)
        
        h1 = self.mlp(Pxz_train)
        h_mean = self.mean_fc(h1)
        h_logvar = self.logvar_fc(h1)

        eps = torch.randn_like(h_logvar)
        h = h_mean + torch.exp(0.5 * h_logvar)*eps 
        return h, h_mean, h_logvar
        