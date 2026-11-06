
import torch.nn as nn
from itertools import pairwise as nwise


def MLP(input_size, output_size=1, *, hidden_sizes):

    layer_sizes = list(hidden_sizes) + [output_size]
    modules: list[nn.Module] = [nn.Linear(input_size, layer_sizes[0])]

    for in_size, out_size in nwise(layer_sizes):
        modules.append(nn.LeakyReLU())
        modules.append(nn.Dropout(.1))
        modules.append(nn.Linear(in_size, out_size))

    return nn.Sequential(*modules)
