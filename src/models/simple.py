import numpy as np
import torch.nn as nn
import torch

from ..utils import nwise


from ott.solvers import quadratic
import jax
import functools

import jax
import jax.numpy as jnp
from typing import Any, Callable, Optional, Sequence, Tuple, Union, List
import flax.linen as ln
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import frozen_dict
from flax.training import train_state
from jax.nn import initializers
import abc
import torch.nn.functional as F

__all__ = ["mlp"]


def mlp(input_size, output_size=1, *,
        hidden_sizes):
    
    layer_sizes = hidden_sizes + [output_size]
    modules: list[nn.Module] = [nn.Linear(input_size, layer_sizes[0])]

    for in_size, out_size in nwise(layer_sizes):
        modules.append(nn.LeakyReLU())
        modules.append(nn.Dropout(.1))
        modules.append(nn.Linear(in_size, out_size))

    return nn.Sequential(*modules)


def fcnn(input_size, output_size=1, *,
        hidden_sizes):
    
    layer_sizes = hidden_sizes + [output_size]
    modules: list[nn.Module] = [nn.Linear(input_size, layer_sizes[0])]

    for in_size, out_size in nwise(layer_sizes):
        modules.append(nn.LeakyReLU())
        modules.append(nn.Linear(in_size, out_size))

    return nn.Sequential(*modules)

def mlp_old(input_size, output_size=1, *,
        hidden_size=None, num_layers=4,
        layer_sizes=None):
    if hidden_size is None:
        hidden_size = input_size
    if layer_sizes is None:
        layer_sizes = [hidden_sizes] * (num_layers - 1) + [output_size]
    modules: list[nn.Module] = [nn.Linear(input_size, layer_sizes[0])]

    for in_size, out_size in nwise(layer_sizes):
        modules.append(nn.LeakyReLU())
        modules.append(nn.Dropout(.1))
        modules.append(nn.Linear(in_size, out_size))

    return nn.Sequential(*modules)

def mlp2(input_size, output_size=1, *,
         hidden_size=None, num_layers=4,
         layer_sizes=None):
    if hidden_size is None:
        hidden_size = input_size
    if layer_sizes is None:
        layer_sizes = [hidden_size] * (num_layers - 1) + [output_size]
    modules: list[nn.Module] = [nn.Linear(input_size, layer_sizes[0])]

    for in_size, out_size in nwise(layer_sizes):
        modules.append(nn.ReLU())
        modules.append(nn.Linear(in_size, out_size))

    return nn.Sequential(*modules)

class mlp_class(nn.Module):
    def __init__(self, input_size, output_size=1, *,
        hidden_size=None, num_layers=4,
        layer_sizes=None, residual=False):
        super().__init__()

        self.residual = residual
        
        if hidden_size is None:
            hidden_size = input_size
        if layer_sizes is None:
            layer_sizes = [hidden_size] * (num_layers - 1) + [output_size]
        modules: list[nn.Module] = [nn.Linear(input_size, layer_sizes[0])]

        for in_size, out_size in nwise(layer_sizes):
            modules.append(nn.ReLU())
            modules.append(nn.Linear(in_size, out_size))

        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        out = self.model(x)            
        
        if self.residual is True:
            return x + out
        
        return out

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

PotentialValueFn_t = Callable[[jnp.ndarray], jnp.ndarray]
PotentialGradientFn_t = Callable[[jnp.ndarray], jnp.ndarray]

class NeuralTrainState(train_state.TrainState):
  """Adds information about the model's value and gradient to the state.

  This extends :class:`~flax.training.train_state.TrainState` to include
  the potential methods from :class:`~ott.solvers.nn.models.ModelBase`
  used during training.

  Args:
    potential_value_fn: the potential's value function
    potential_gradient_fn: the potential's gradient function
  """
  potential_value_fn: Callable[
      [frozen_dict.FrozenDict[str, jnp.ndarray], Optional[PotentialValueFn_t]],
      PotentialValueFn_t] = struct.field(pytree_node=False)
  potential_gradient_fn: Callable[[frozen_dict.FrozenDict[str, jnp.ndarray]],
                                  PotentialGradientFn_t] = struct.field(
                                      pytree_node=False
                                  )
class ModelBase(abc.ABC, ln.Module):

  @property
  @abc.abstractmethod
  def is_potential(self) -> bool:
    """Indicates if the module implements a potential value or a vector field.

    Returns:
      ``True`` if the module defines a potential, ``False`` if it defines a
       vector field.
    """

  def potential_value_fn(
        self,
        params: frozen_dict.FrozenDict[str, jnp.ndarray],
        other_potential_value_fn: Optional[PotentialValueFn_t] = None,
    ) -> PotentialValueFn_t:
      
      if self.is_potential:
        return lambda x: self.apply({"params": params}, x)

      assert other_potential_value_fn is not None, \
          "The value of the gradient-based potential depends " \
          "on the value of the other potential."

      def value_fn(x: jnp.ndarray) -> jnp.ndarray:
        squeeze = x.ndim == 1
        if squeeze:
          x = jnp.expand_dims(x, 0)
        grad_g_x = jax.lax.stop_gradient(self.apply({"params": params}, x))
        value = -other_potential_value_fn(grad_g_x) + \
            jax.vmap(jnp.dot)(grad_g_x, x)
        return value.squeeze(0) if squeeze else value

      return value_fn


  def potential_gradient_fn(
        self,
        params: frozen_dict.FrozenDict[str, jnp.ndarray],
    ) -> PotentialGradientFn_t:
      
      if self.is_potential:
        return jax.vmap(jax.grad(self.potential_value_fn(params)))
      return lambda x: self.apply({"params": params}, x)


  def create_train_state(
        self,
        rng:jax.Array,#: jax.random.PRNGKeyArray,
        optimizer: optax.OptState,
        input: Union[int, Tuple[int, ...]],
        **kwargs: Any,
    ) -> NeuralTrainState:

      params = self.init(rng, jnp.ones(input))["params"]

      return NeuralTrainState.create(
          apply_fn=self.apply,
          params=params,
          tx=optimizer,
          potential_value_fn=self.potential_value_fn,
          potential_gradient_fn=self.potential_gradient_fn,
          **kwargs,
      )


class Block(ln.Module):

  dims: List
  out_dim: int = 32
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = ln.silu

  @ln.compact
  def __call__(self, x):
    #print(self.dims)
    for i, dim in enumerate(self.dims):#range(self.num_layers):
      x = ln.Dense(dim, name=f"fc{i}")(x)
      x = self.act_fn(x)
    return ln.Dense(self.out_dim, name="fc_final")(x)

class mlp_jax(ModelBase):
  hidden_dims: List
  out_dim: int
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = ln.silu

  @property
  def is_potential(self) -> bool:
    return True

  @ln.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    z = x
    z = Block(dims=self.hidden_dims, out_dim=self.out_dim, act_fn=self.act_fn)(z)
    return z

class FCNN(nn.Module):
    def __init__(self, dim_init, hidden_layer, dim_final):
        super(generator_x_y, self).__init__()
        
        self.lin1 = nn.Linear(dim_init, hidden_layer)
        self.lin2 = nn.Linear(hidden_layer, hidden_layer)
        self.lin3 = nn.Linear(hidden_layer, hidden_layer)
        self.lin_end = nn.Linear(hidden_layer, dim_final)
        
    def forward(self, inp):
        out = Fn.leaky_relu(self.lin1(inp))
        out = Fn.leaky_relu(self.lin2(out))
        out = Fn.leaky_relu(self.lin3(out))
        out = self.lin_end(out)
        
        return out 