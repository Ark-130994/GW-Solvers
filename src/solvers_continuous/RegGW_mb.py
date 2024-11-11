from ott.solvers import quadratic
import jax
import functools
from tqdm.auto import trange
import jax
import jax.numpy as jnp
from typing import Any, Callable, Optional, Sequence, Tuple, Union, List
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import frozen_dict
from flax.training import train_state
from jax.nn import initializers
import abc

import wandb

import jax.tree_util as jtu
from src.metrics import compute_metrics

from ott.geometry import costs, pointcloud

from ott.tools import sinkhorn_divergence

from ott import utils
import torch
import numpy as np


def _get_step_fn():

    def loss_fn(params, apply_fn, batch, cost_fn, eps_fit, eps_reg, lamb) :
      mapped_samples = apply_fn({"params": params}, batch["source"])

      #print(mapped_samples.shape)
      val_fitting_loss = fitting_loss(mapped_samples, batch["target"], eps_fit)
      val_regularizer = regularizer(batch["source"], mapped_samples, cost_fn, epsilon=eps_reg )
      val_tot_loss = (val_fitting_loss + lamb * val_regularizer)
        
      return val_tot_loss#, loss_logs

    @functools.partial(jax.jit, static_argnums=5)
    def step_fn( state_neural_net, train_batch, cost_fn, eps_fit=0.01, eps_reg=0.001, lamb=1.0):

      grad_fn = jax.value_and_grad(loss_fn, argnums=0)
      value, grads = grad_fn(state_neural_net.params, state_neural_net.apply_fn, train_batch, cost_fn, eps_fit, eps_reg, lamb)

      return state_neural_net.apply_gradients(grads=grads), value

    return step_fn

@jax.jit
def fitting_loss(x, y, epsilon_fitting):
    out = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud, x, y, epsilon=epsilon_fitting, static_b=True
    )
    return out.divergence

@jax.jit
def regularizer(
    source,
    target,
    cost_fn = None,
    epsilon = None,
    relative_epsilon = None,
    scale_cost= 1.0,
    **kwargs
):
    
    #cost_fn = costs.SqEuclidean() if cost_fn is None else cost_fn
    n = source.shape[0]
    geom_xx = pointcloud.PointCloud(
        x=source,
        y=source,
        cost_fn=cost_fn,
        epsilon=epsilon,
        relative_epsilon=relative_epsilon,
        scale_cost=scale_cost,
    )

    geom_yy = pointcloud.PointCloud(
        x=target,
        y=target,
        cost_fn=cost_fn,
        epsilon=epsilon,
        relative_epsilon=relative_epsilon,
        scale_cost=scale_cost,
    )

    source_cost = geom_xx.cost_matrix 
    target_cost = geom_yy.cost_matrix  

    gt_displacement_cost = jnp.mean(jax.vmap(costs.SqEuclidean())(source_cost, target_cost))/n

    out = quadratic.solve(geom_xx, geom_yy)

    loss = gt_displacement_cost - out.reg_gw_cost
      
    return loss

class RegGW_mb:
    def __init__(self, mover_model, mover_optimizer, source_dim, cost_fn, eps_fit=0.01, eps_reg=0.001, lamb=1, rng_seed=0, toy_type=False):
        self.mover_model = mover_model
        self.mover_optimizer = mover_optimizer

        self.eps_fit = eps_fit
        self.eps_reg = eps_reg

        self.lamb = lamb
        self.cost_fn = cost_fn

        rng = jax.random.PRNGKey(0)
        self.state_neural_net = self.mover_model.create_train_state(rng, self.mover_optimizer, source_dim)

        self.step_fn = _get_step_fn()

    def train_epoch_toy(self, source_sampler,  target_sampler, n_samples, n_epochs, epoch, wandb_report):
        x_train = source_sampler.sample((n_samples,))
        y_train = target_sampler.sample((n_samples,))
        
        x_train_jnp, y_train_jnp = jnp.array(x_train.cpu().numpy()), jnp.array(y_train.cpu().numpy())
        x_train_jnp, y_train_jnp = jtu.tree_map(jnp.asarray, x_train_jnp), jtu.tree_map(jnp.asarray, y_train_jnp)
                    
        train_batch = {'source': x_train_jnp, 'target':y_train_jnp}
                    
        self.state_neural_net, mover_loss = self.step_fn(self.state_neural_net, train_batch, self.cost_fn, eps_fit=self.eps_fit, eps_reg=self.eps_reg, lamb=self.lamb)

        if wandb_report:
            loss_metrics = {"train/mover_loss": np.asarray(mover_loss),
                            "train/step": epoch}

            wandb.log(loss_metrics)
    def train_epoch(self, sampler_source, sampler_target, n_samples, n_epochs, epoch, wandb_report):
        
        x_train, _ = sampler_source.sample(n_samples)
        y_train, _ = sampler_target.sample(n_samples)
        
        x_train_jnp, y_train_jnp = jnp.array(x_train.cpu().numpy()), jnp.array(y_train.cpu().numpy())
        x_train_jnp, y_train_jnp = jtu.tree_map(jnp.asarray, x_train_jnp), jtu.tree_map(jnp.asarray, y_train_jnp)
                    
        train_batch = {'source': x_train_jnp, 'target':y_train_jnp}
                    
        self.state_neural_net, mover_loss = self.step_fn(self.state_neural_net, train_batch, self.cost_fn, eps_fit=self.eps_fit, eps_reg=self.eps_reg, lamb=self.lamb)

        if wandb_report:
            loss_metrics = {"train/mover_loss": np.asarray(mover_loss),
                            "train/step": epoch}

            wandb.log(loss_metrics)

    def valid_step(self, sampler_source, sampler_target, n_samples, metric_names, target_vectors, n_eval):
        metrics_dict = {metric_name:[] for metric_name in metric_names}
        
        sampler_source.reset_sampler()

        with torch.no_grad():
        
            for _ in trange(n_eval, leave=False, desc="Evaluation"):
                    
                if sampler_target is None:
                    x, y, labels = sampler_source.sample(n_samples)
                else:
                    sampler_target.reset_sampler()
                    x, labels = sampler_source.sample(n_samples)
                    y, _      = sampler_target.sample(n_samples)
                    
                x_jnp = jnp.array(x.cpu().numpy())
                y_sampled = self.state_neural_net.apply_fn({"params":self.state_neural_net.params}, x_jnp)
                        
                y_sampled = torch.tensor(np.asarray(y_sampled)).to(torch.float32)
                        
                metrics_dict = compute_metrics(x, y, y_sampled, labels, target_vectors, metrics_dict)
            
        return metrics_dict
