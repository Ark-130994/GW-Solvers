
from typing import Callable, List

import jax
import jax.numpy as jnp
import flax.linen as ln
from flax.training import train_state


class Block(ln.Module):
    dims: List
    out_dim: int = 32
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = ln.silu

    @ln.compact
    def __call__(self, x):
        for i, dim in enumerate(self.dims):
            x = ln.Dense(dim, name=f'fc{i}')(x)
            x = self.act_fn(x)
        return ln.Dense(self.out_dim, name='fc_final')(x)


class MLPJax(ln.Module):
    hidden_dims: List
    out_dim: int
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = ln.silu

    @ln.compact
    def __call__(self, x):
        z = Block(dims=self.hidden_dims, out_dim=self.out_dim, act_fn=self.act_fn)(x)
        return z

    def create_train_state(self, rng, optimizer, source_dim):
        params = self.init(rng, jnp.ones(source_dim))['params']
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)
