import collections
import types
from functools import partial
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type, Union
import torch
import diffrax

from tqdm import tqdm
from tqdm.auto import trange

from ott.neural.methods.flows import dynamics

import jax
import jax.numpy as jnp
from ott.geometry import costs, geometry, pointcloud

from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein
from ott.neural.networks.layers import time_encoder

import optax

import jax
import jax.numpy as jnp
from tqdm import tqdm_notebook as tqdm


import wandb
from ott.solvers import utils as solver_utils
import numpy as np
import jax.tree_util as jtu
import jax
import jax.numpy as jnp
import numpy as np
import scipy
import sklearn.preprocessing as pp
from ott.geometry.pointcloud import PointCloud
from ott.solvers.linear import acceleration, sinkhorn
from ott.solvers.quadratic import gromov_wasserstein
from ott.tools.sinkhorn_divergence import sinkhorn_divergence
import scanpy as sc
import functools
from ott.neural.networks.layers import time_encoder
from ott.neural.networks.velocity_field import VelocityField
from ott import utils
from src.metrics import compute_metrics

import matplotlib.pyplot as plt

#def report_wandb_fn(metrics_dict, metrics_names, epoch):
#
#    for key in metrics_dict.keys():
#        for metric_name in metrics_names:
#            wandb.log({f'{key}/{metric_name}':metrics_dict[key][-1][metric_name]['mean'],
#                       f'{key}/repeat':epoch})

def report_wandb_fn(metrics_dict, metrics_names, epoch, prefix):
    for metric_name in metrics_names:
        wandb.log({f'{prefix}/{metric_name}':metrics_dict[metric_name][-1]}, step=epoch)

def sample_conditional_indices(rng: jax.Array, tmat: jnp.ndarray, *, k_samples_per_x: int = 1) -> Tuple[jnp.ndarray, jnp.ndarray]:
    
      assert k_samples_per_x > 0, "Number of samples per source must be positive."
      n, m = tmat.shape
    
      src_marginals = tmat.sum(axis=1)
      rng, rng_ixs = jax.random.split(rng, 2)
      indices = jax.random.choice(rng_ixs, a=n, p=src_marginals, shape=(n,))
      tmat = tmat[indices]
    
      rngs = jax.random.split(rng, n)
      tgt_ixs = jax.vmap(
          lambda rng, row: jax.random.choice(rng, a=m, p=row, shape=(k_samples_per_x,)),
          in_axes=[0, 0],
      )(rngs, tmat)  # (m, k)
    
      src_ixs = jnp.repeat(indices[:, None], k_samples_per_x, axis=1)  # (n, k)
      return src_ixs, tgt_ixs
    
def _multivariate_normal(rng: jax.Array, shape: Tuple[int, ...], dim: int, mean: float = 0.0, cov: float = 1.0) -> jnp.ndarray:
      mean = jnp.full(dim, fill_value=mean)
      cov = jnp.diag(jnp.full(dim, fill_value=cov))
      return jax.random.multivariate_normal(rng, mean=mean, cov=cov, shape=shape)
    
class GENOT:
    def __init__(
        self,
        neural_net,
        flow,
        ot_solver,
        
        input_dim: int,
        output_dim: int,
        k_latent_per_x: int = 1,
        cost_fn = costs.Cosine(),#costs.SqEuclidean(),
        scale_cost = 1.0,
        seed: int = 0,
        **kwargs,
        ) -> None:


        self.flow = flow
        self.ot_solver = ot_solver

        self.rng = jax.random.PRNGKey(seed)
        self.seed = seed

        self.neural_net = neural_net
        self.state_neural_net = None
        self.optimizer = optax.adamw(learning_rate=1e-4, weight_decay=1e-10)
        self.latent_fn = functools.partial(_multivariate_normal, dim=output_dim)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k_latent_per_x = k_latent_per_x

        self.cost_fn = cost_fn
        self.scale_cost = scale_cost

        self.time_sampler = solver_utils.uniform_sampler

        #Inititalizing
        self.state_neural_net = self.neural_net.create_train_state(self.rng, self.optimizer, self.output_dim, self.input_dim)
        self.step_fn = self._get_step_fn()
                
        self.match_fn = self._get_gromov_match_fn(self.ot_solver, cost_fn=self.cost_fn, scale_cost=self.scale_cost,
                                                  k_samples_per_x=self.k_latent_per_x)

    def __call__(self, source_train, target_train):
        
        train_batch = {}

        self.rng, rng_time, rng_latent, rng_step_fn = jax.random.split(self.rng, 4)
        
        n_samples = len(source_train) * self.k_latent_per_x

        train_batch["source"] = source_train
        train_batch["target"] = target_train
       
        train_batch["time"] = self.time_sampler(rng_time, n_samples)
        train_batch["latent"] = self.latent_fn(rng_latent, shape=(len(source_train), self.k_latent_per_x))
       
        self.state_neural_net = self.step_fn(rng_step_fn, self.state_neural_net, train_batch)

    def _get_gromov_match_fn(self, ot_solver, cost_fn, scale_cost, k_samples_per_x):
        
        @partial(jax.jit, static_argnames=[ "ot_solver", "cost_fn", "scale_cost", "k_samples_per_x"])
        def match_pairs(key, x, y, ot_solver, cost_fn, scale_cost, k_samples_per_x) :

            geom_xx = pointcloud.PointCloud(x=x[..., 0:], y=x[..., 0:], cost_fn=cost_fn, scale_cost=scale_cost)
            geom_yy = pointcloud.PointCloud(x=y[..., 0:], y=y[..., 0:], cost_fn=cost_fn, scale_cost=scale_cost)

            geom_xy = None
            prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy, geom_xy, fused_penalty=0.0, tau_a=1.0, tau_b=1.0)
            out = ot_solver(prob).matrix

            return out#

        return jax.tree_util.Partial(match_pairs, ot_solver=ot_solver, cost_fn=cost_fn, scale_cost=scale_cost, k_samples_per_x=k_samples_per_x, )

    def _get_step_fn(self) -> Callable:

        def loss_fn(
          params: jnp.ndarray, apply_fn: Callable, time: jnp.ndarray, source: jnp.ndarray,
          target: jnp.ndarray, latent: jnp.ndarray,
          source_conditions: Optional[jnp.ndarray], rng: jax.Array) -> jnp.ndarray:
            
            rng_flow, rng_dropout = jax.random.split(rng, 2)
            x_t = self.flow.compute_xt(rng_flow, time, latent, target)
            if source_conditions is None:
              cond = source
            else:
              cond = jnp.concatenate([source, source_conditions], axis=-1)
    
            v_t = apply_fn({"params": params},
                                    time,
                                    x_t,
                                    cond,
                                    rngs={"dropout": rng_dropout})
            u_t = self.flow.compute_ut(time, latent, target)
    
            return jnp.mean((v_t - u_t) ** 2)
            
        @jax.jit
        def step_fn(rng: jax.Array, state_neural_net, batch):
            
            rng_match, rng_latent, rng_grad = jax.random.split(rng, 3)

            tmat = self.match_fn(rng_match, batch["source"], batch["target"])
            inds_source, inds_target = sample_conditional_indices(rng=rng, tmat=tmat, k_samples_per_x=self.k_latent_per_x)

            source_batch, target_batch = batch["source"][inds_source], batch["target"][inds_target]
            rng_latent = jax.random.split(rng_latent, (len(target_batch)))

            batch["source"] = jnp.reshape(source_batch, (len(source_batch), -1))
            batch["target"] = jnp.reshape(target_batch, (len(source_batch), -1))
            batch["latent"] = jnp.reshape(batch['latent'], (len(source_batch), -1))

            grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
            loss, grads = grad_fn(state_neural_net.params,state_neural_net.apply_fn, batch['time'], batch['source'], 
                                  batch['target'], batch['latent'], None, rng_grad)

            return state_neural_net.apply_gradients(grads=grads)

        return step_fn


    def transport(self, source, condition=None, t0=0.0, t1=1.0, rng=None, **kwargs: Any):
    
        def vf(t: jnp.ndarray, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
          params = self.state_neural_net.params
          return self.state_neural_net.apply_fn({"params": params}, t, x, cond)
    
        def solve_ode(x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
          ode_term = diffrax.ODETerm(vf)
          sol = diffrax.diffeqsolve(
              ode_term,
              t0=t0,
              t1=t1,
              y0=x,
              args=cond,
              **kwargs,
          )
          return sol.ys[0]
    
        kwargs.setdefault("dt0", None)
        kwargs.setdefault("solver", diffrax.Tsit5())
        kwargs.setdefault(
            "stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5)
        )
    
        rng = utils.default_prng_key(rng)
        latent = self.latent_fn(rng, (len(source),))
    
        if condition is not None:
          source = jnp.concatenate([source, condition], axis=-1)
    
        return jax.jit(jax.vmap(solve_ode))(latent, source)

class FlowGW_mb:
    
    def __init__(self, model, source_dim, target_dim, eps, cost_fn, seed):
        self.model = model
        self.eps = eps

        self.source_dim = source_dim
        self.target_dim = target_dim
 
        self.cost_fn = cost_fn

        linear_ot_solver = sinkhorn.Sinkhorn(momentum=acceleration.Momentum(value=1.0, start=25))
        solver = gromov_wasserstein.GromovWasserstein(epsilon=self.eps, linear_ot_solver=linear_ot_solver)
        self.genot_fgw = GENOT(
                               self.model,
                               flow = dynamics.ConstantNoiseFlow(0.0),
                               ot_solver=solver,
                               scale_cost="mean",
                               cost_fn=self.cost_fn,
                               input_dim=source_dim,
                               output_dim=target_dim,
                               k_latent_per_x=1,
                               seed=seed
                              )

    def train_epoch(self, sampler, n_samples, n_iters, epoch, wandb_report):
        
        x_train, y_train, labels_train = sampler.sample(n_samples)   
        x_train_jnp, y_train_jnp = np.asarray(x_train.cpu().numpy()), np.asarray(y_train.cpu().numpy())
        x_train_jnp, y_train_jnp = jtu.tree_map(jnp.asarray, x_train_jnp), jtu.tree_map(jnp.asarray, y_train_jnp)

        self.genot_fgw(x_train_jnp, y_train_jnp)
        
    def valid_step(self, sampler, n_samples, metric_names, target_vectors, n_eval):
            
        metrics_dict = {metric_name:[] for metric_name in metric_names}
        
        for _ in trange(n_eval, leave=False, desc="Evaluation"):
            x, y, labels = sampler.sample(n_samples)
            x_jnp = jnp.array(x.cpu().numpy())
            y_sampled = self.genot_fgw.transport(x_jnp)
                    
            y_sampled = torch.tensor(np.asarray(y_sampled)).to(torch.float32)
                    
            metrics_dict = compute_metrics(x, y, y_sampled, labels, target_vectors, metrics_dict)
            
        return metrics_dict


