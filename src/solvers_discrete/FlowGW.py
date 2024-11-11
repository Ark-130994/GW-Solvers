import collections
import types
from functools import partial
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type, Union
import torch
import diffrax

from tqdm import tqdm
from ott.neural.methods.flows import dynamics

import jax
import jax.numpy as jnp
from ott.geometry import costs, geometry, pointcloud

from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein

import optax

import jax
import jax.numpy as jnp
from tqdm import tqdm_notebook as tqdm
from tqdm.auto import trange

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
    
      src_ixs = jnp.repeat(indices[:, None], k_samples_per_x, axis=1) 
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
        fused_dim: int,
        iterations: int,
        k_latent_per_x: int = 1,
        fused_penalty: int = (1.0-0.3)/0.3,
        cost_fn = costs.Cosine(),
        lr = 1e-4,
        scale_cost = 1.0,
        seed: int = 0,
        **kwargs,
        ) -> None:


        self.flow = flow
        self.ot_solver = ot_solver

        self.rng = jax.random.PRNGKey(seed)
        self.seed = seed
        self.iterations = iterations

        self.neural_net = neural_net
        self.state_neural_net = None
        self.optimizer = optax.adamw(learning_rate=lr, weight_decay=1e-10)
        self.latent_fn = functools.partial(_multivariate_normal, dim=output_dim)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fused_dim = fused_dim
        self.fused_penalty = fused_penalty
        if self.fused_dim == 0:
            self.fused_penalty = 0
            
        self.k_latent_per_x = k_latent_per_x

        self.cost_fn = cost_fn
        self.scale_cost = scale_cost

        self.time_sampler = solver_utils.uniform_sampler

        #Inititalizing
        self.state_neural_net = self.neural_net.create_train_state(self.rng, self.optimizer, self.output_dim, self.input_dim)
        self.step_fn = self._get_step_fn()
                
        self.match_fn = self._get_gromov_match_fn(self.ot_solver, cost_fn=self.cost_fn, scale_cost=self.scale_cost,
                                                  k_samples_per_x=self.k_latent_per_x, fused_dim=self.fused_dim, fused_penalty=self.fused_penalty)

    def __call__(self, source_train, target_train):
        
        train_batch = {}

        self.rng, rng_time, rng_latent, rng_step_fn = jax.random.split(self.rng, 4)
        
        n_samples = len(source_train) * self.k_latent_per_x

        train_batch["source"] = source_train
        train_batch["target"] = target_train
       
        train_batch["time"] = self.time_sampler(rng_time, n_samples)
        train_batch["latent"] = self.latent_fn(rng_latent, shape=(len(source_train), self.k_latent_per_x))
       
        self.state_neural_net = self.step_fn(rng_step_fn, self.state_neural_net, train_batch)

    def _get_gromov_match_fn(self, ot_solver, cost_fn, scale_cost, k_samples_per_x, fused_dim, fused_penalty):
        
        @partial(jax.jit, static_argnames=[ "ot_solver", "cost_fn", "scale_cost", "k_samples_per_x", "fused_dim", "fused_penalty"])
        def match_pairs(key, x, y, ot_solver, cost_fn, scale_cost, k_samples_per_x, fused_dim, fused_penalty) :

            geom_xx = pointcloud.PointCloud(x=x[..., fused_dim:], y=x[..., fused_dim:], cost_fn=cost_fn, scale_cost=scale_cost)
            geom_yy = pointcloud.PointCloud(x=y[..., fused_dim:], y=y[..., fused_dim:], cost_fn=cost_fn, scale_cost=scale_cost)

            if fused_dim > 0:
                geom_xy = pointcloud.PointCloud(x=x[..., :fused_dim], y=y[..., :fused_dim], cost_fn=cost_fn, scale_cost=scale_cost)
            else:
                geom_xy = None
                
            prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy, geom_xy, fused_penalty=fused_penalty, tau_a=1.0, tau_b=1.0)
            out = ot_solver(prob).matrix

            return out

        return jax.tree_util.Partial(match_pairs, ot_solver=ot_solver, cost_fn=cost_fn, scale_cost=scale_cost, k_samples_per_x=k_samples_per_x, fused_dim=fused_dim, fused_penalty=fused_penalty)


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

class FlowGW:
    
    def __init__(self, eps, embed_dim, n_freq, n_layers, cost_fn, lr, toy_type=None):
        self.eps       = eps
        self.embed_dim = embed_dim
        self.n_freq    = n_freq
        self.n_layers  = n_layers
        self.cost_fn   = cost_fn
        self.toy_type  = toy_type
        self.lr        = lr

    def solve(self, x_dict, y_dict, labels_dict, target_vectors, wandb_report=False, maxiters=200, report_every=20):
    #def solve(self, x_dict, y_dict, labels_dict, target_vectors, fused_dim=0, wandb_report=False, maxiters=200, report_every=20):

        x_train, y_train, labels_train = x_dict['train'], y_dict['train'], labels_dict['train']
        x_train_jnp, y_train_jnp       = np.asarray(x_train.cpu().numpy()), np.asarray(y_train.cpu().numpy())
        x_train_jnp, y_train_jnp       = jtu.tree_map(jnp.asarray, x_train_jnp), jtu.tree_map(jnp.asarray, y_train_jnp)
        
        x_test, y_test, labels_test    = x_dict['test'], y_dict['test'], labels_dict['test']
        if x_test != None:
            x_test_jnp                     = np.asarray(x_test.cpu().numpy())
            x_test_jnp                     = jtu.tree_map(jnp.asarray, x_test_jnp)

        else:
            x_test_jnp = None
            y_test_jnp = None

        source_dim = x_train.shape[1]
        target_dim = y_train.shape[1]

        metric_names = ['Top@1', 'Top@5', 'Top@10', 'cossim_gt', 'inner_gw', 'foscttm']

        metrics_dict_train = {metric_name:[] for metric_name in metric_names}
        metrics_dict_test = {metric_name:[] for metric_name in metric_names}

        neural_net = VelocityField(
                hidden_dims=[self.embed_dim]*self.n_layers,
                time_dims=[self.embed_dim, self.embed_dim],
                output_dims=[self.embed_dim, self.embed_dim, self.embed_dim] + [target_dim],
                condition_dims=[self.embed_dim, self.embed_dim, self.embed_dim],
                time_encoder=functools.partial(time_encoder.cyclical_time_encoder, n_freqs=self.n_freq),
            )
        
        linear_ot_solver = sinkhorn.Sinkhorn(momentum=acceleration.Momentum(value=1.0, start=25))
        solver = gromov_wasserstein.GromovWasserstein(epsilon=self.eps, linear_ot_solver=linear_ot_solver)
        genot_fgw = GENOT(
                    neural_net,
                    flow = dynamics.ConstantNoiseFlow(0.0),
                    ot_solver=solver,
                    scale_cost="mean",
                    input_dim=source_dim,
                    output_dim=target_dim,
                    cost_fn=self.cost_fn,
                    lr=self.lr,
                    fused_dim=0,
                    #fused_penalty=(1.0-0.3)/0.3,
                    iterations=maxiters,
                    k_latent_per_x=1,
        )

        for it in tqdm(range(maxiters)):
            genot_fgw(x_train_jnp, y_train_jnp)

            if ((it) % report_every == 0 and it!=0) or it == maxiters-1:

                if self.toy_type is None:

                    if wandb_report:
                        y_sampled = np.asarray(genot_fgw.transport(x_train_jnp, rng=jax.random.PRNGKey(0)))
                        y_sampled_test = np.asarray(genot_fgw.transport(x_test_jnp, rng=jax.random.PRNGKey(0)))
                        
                        y_sampled = torch.tensor(y_sampled).to(torch.float32)
                        y_sampled_test = torch.tensor(y_sampled_test).to(torch.float32)
                        
                        metrics_dict_train = compute_metrics(x_train, y_train, y_sampled, labels_train.cpu(), target_vectors.cpu(), metrics_dict_train)
                        report_wandb_fn(metrics_dict_train, metric_names, it, 'train')
                        
                        metrics_dict_test = compute_metrics(x_test, y_test, y_sampled_test, labels_test.cpu(), target_vectors.cpu(), metrics_dict_test)
                        report_wandb_fn(metrics_dict_test, metric_names, it, 'test')
                        
                    else:
                        pass
                else:
                    
                    y_sampled_np = genot_fgw.transport(x_train_jnp, rng=jax.random.PRNGKey(0))
                    y_sampled_np = np.asarray(y_sampled_np)
                
                    fig = plt.figure(figsize=(8, 8))
                    
                    if self.toy_type == 'toy_2d_3d':
                        ax = fig.add_subplot(projection='3d')
                       
                    if self.toy_type == 'toy_3d_2d':
                        ax = fig.add_subplot(projection=None)
                        
                    ax.scatter(*y_sampled_np.T, c=labels_train.cpu().numpy(),  cmap="Spectral", alpha=0.8)
                    ax.set_title('FlowGW')
                    plt.show()
                    
        return genot_fgw

    def fit(self, x_dict, y_dict, labels_dict, target_vectors, wandb_report=False, max_iters=200, report_every=10):
        
        self.x_dict, self.y_dict, self.labels_dict = x_dict, y_dict, labels_dict
        
        model = self.solve(self.x_dict, self.y_dict, self.labels_dict, target_vectors, wandb_report, max_iters, report_every)
        
        self.model = model
        
    def valid_step(self, sampler_source, sampler_target, n_samples, metric_names, target_vectors, n_eval):
            
        metrics_dict = {metric_name:[] for metric_name in metric_names}
        
        with torch.no_grad():
        
            sampler_source.reset_sampler()

            for _ in trange(n_eval, leave=False, desc="Evaluation"):
                
                if sampler_target is None:
                    x, y, labels = sampler_source.sample(n_samples)
                else:
                    sampler_target.reset_sampler()
                    x, labels = sampler_source.sample(n_samples)
                    y, _      = sampler_target.sample(n_samples)
                    
                x, y, labels = x.cpu(), y.cpu(), labels.cpu()
                x_jnp        = np.asarray(x.numpy())
                y_sampled    = self.model.transport(x_jnp)
                y_sampled    = torch.tensor(np.asarray(y_sampled)).to(torch.float32)

                metrics_dict = compute_metrics(x, y, y_sampled, labels, target_vectors, metrics_dict)
            
            return metrics_dict


