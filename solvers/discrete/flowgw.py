
import functools
import numpy as np
import torch
import jax
import jax.numpy as jnp
import diffrax
import optax

from functools import partial
from ott import utils
from ott.geometry import costs, pointcloud
from ott.problems.quadratic import quadratic_problem
from ott.solvers import utils as solver_utils
from ott.solvers.linear import acceleration, sinkhorn
from ott.solvers.quadratic import gromov_wasserstein
from ott.neural.methods.flows import dynamics
from ott.neural.networks.layers import time_encoder
from ott.neural.networks.velocity_field import VelocityField


def sample_conditional_indices(rng, tmat, k_samples_per_x=1):
    assert k_samples_per_x > 0, 'Number of samples per source must be positive.'
    n, m = tmat.shape

    src_marginals = tmat.sum(axis=1)
    rng, rng_ixs = jax.random.split(rng, 2)
    indices = jax.random.choice(rng_ixs, a=n, p=src_marginals, shape=(n,))
    tmat = tmat[indices]

    rngs = jax.random.split(rng, n)
    tgt_ixs = jax.vmap(lambda rng, row: jax.random.choice(rng, a=m, p=row, shape=(k_samples_per_x,)), in_axes=[0, 0])(rngs, tmat)
    src_ixs = jnp.repeat(indices[:, None], k_samples_per_x, axis=1)

    return src_ixs, tgt_ixs


def _multivariate_normal(rng, shape, dim, mean=0.0, cov=1.0):
    mean = jnp.full(dim, fill_value=mean)
    cov = jnp.diag(jnp.full(dim, fill_value=cov))
    return jax.random.multivariate_normal(rng, mean=mean, cov=cov, shape=shape)


class GENOT:

    def __init__(self, neural_net, flow, ot_solver, input_dim, output_dim, fused_dim, iterations, k_latent_per_x=1,
                 fused_penalty=(1.0-0.3)/0.3, cost_fn=None, lr=1e-4, scale_cost=1.0, seed=0):

        self.flow = flow
        self.ot_solver = ot_solver
        self.rng = jax.random.PRNGKey(seed)
        self.seed = seed
        self.iterations = iterations

        self.neural_net = neural_net
        self.optimizer = optax.adamw(learning_rate=lr, weight_decay=1e-10)
        self.latent_fn = functools.partial(_multivariate_normal, dim=output_dim)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fused_dim = fused_dim
        self.fused_penalty = fused_penalty if fused_dim > 0 else 0
        self.k_latent_per_x = k_latent_per_x

        self.cost_fn = cost_fn if cost_fn is not None else costs.Cosine()
        self.scale_cost = scale_cost
        self.time_sampler = solver_utils.uniform_sampler

        self.state_neural_net = self.neural_net.create_train_state(self.rng, self.optimizer, self.output_dim, self.input_dim)
        self.step_fn = self._get_step_fn()
        self.match_fn = self._get_gromov_match_fn(self.ot_solver, self.cost_fn, self.scale_cost, self.k_latent_per_x, self.fused_dim, self.fused_penalty)
        self.transport_fn = self._get_transport_fn()

    def __call__(self, source_train, target_train):
        self.rng, rng_time, rng_latent, rng_step_fn = jax.random.split(self.rng, 4)

        n_samples = len(source_train) * self.k_latent_per_x
        train_batch = {
            'source': source_train,
            'target': target_train,
            'time': self.time_sampler(rng_time, n_samples),
            'latent': self.latent_fn(rng_latent, shape=(len(source_train), self.k_latent_per_x)),
        }

        self.state_neural_net, loss = self.step_fn(rng_step_fn, self.state_neural_net, train_batch)
        return loss

    def _get_gromov_match_fn(self, ot_solver, cost_fn, scale_cost, k_samples_per_x, fused_dim, fused_penalty):

        @partial(jax.jit, static_argnames=['ot_solver', 'cost_fn', 'scale_cost', 'k_samples_per_x', 'fused_dim', 'fused_penalty'])
        def match_pairs(key, x, y, ot_solver, cost_fn, scale_cost, k_samples_per_x, fused_dim, fused_penalty):
            geom_xx = pointcloud.PointCloud(x=x[..., fused_dim:], y=x[..., fused_dim:], cost_fn=cost_fn, scale_cost=scale_cost)
            geom_yy = pointcloud.PointCloud(x=y[..., fused_dim:], y=y[..., fused_dim:], cost_fn=cost_fn, scale_cost=scale_cost)

            if fused_dim > 0:
                geom_xy = pointcloud.PointCloud(x=x[..., :fused_dim], y=y[..., :fused_dim], cost_fn=cost_fn, scale_cost=scale_cost)
            else:
                geom_xy = None

            prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy, geom_xy, fused_penalty=fused_penalty, tau_a=1.0, tau_b=1.0)
            return ot_solver(prob).matrix

        return jax.tree_util.Partial(match_pairs, ot_solver=ot_solver, cost_fn=cost_fn, scale_cost=scale_cost,
                                     k_samples_per_x=k_samples_per_x, fused_dim=fused_dim, fused_penalty=fused_penalty)

    def _get_step_fn(self):

        def loss_fn(params, apply_fn, time, source, target, latent, source_conditions, rng):
            rng_flow, rng_dropout = jax.random.split(rng, 2)
            x_t = self.flow.compute_xt(rng_flow, time, latent, target)

            if source_conditions is None:
                cond = source
            else:
                cond = jnp.concatenate([source, source_conditions], axis=-1)

            v_t = apply_fn({'params': params}, time, x_t, cond, rngs={'dropout': rng_dropout})
            u_t = self.flow.compute_ut(time, latent, target)

            return jnp.mean((v_t - u_t)**2)

        @jax.jit
        def step_fn(rng, state_neural_net, batch):
            rng_match, rng_latent, rng_grad = jax.random.split(rng, 3)

            tmat = self.match_fn(rng_match, batch['source'], batch['target'])
            inds_source, inds_target = sample_conditional_indices(rng=rng, tmat=tmat, k_samples_per_x=self.k_latent_per_x)

            source_batch, target_batch = batch['source'][inds_source], batch['target'][inds_target]
            rng_latent = jax.random.split(rng_latent, len(target_batch))

            batch['source'] = jnp.reshape(source_batch, (len(source_batch), -1))
            batch['target'] = jnp.reshape(target_batch, (len(source_batch), -1))
            batch['latent'] = jnp.reshape(batch['latent'], (len(source_batch), -1))

            grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
            loss, grads = grad_fn(state_neural_net.params, state_neural_net.apply_fn, batch['time'],
                                  batch['source'], batch['target'], batch['latent'], None, rng_grad)

            return state_neural_net.apply_gradients(grads=grads), loss

        return step_fn

    def _get_transport_fn(self, t0=0.0, t1=1.0, **kwargs):
        kwargs.setdefault('dt0', None)
        kwargs.setdefault('solver', diffrax.Tsit5())
        kwargs.setdefault('stepsize_controller', diffrax.PIDController(rtol=1e-5, atol=1e-5))

        def vf(t, x, args):
            params, cond = args
            return self.state_neural_net.apply_fn({'params': params}, t, x, cond)

        def solve_ode(x, cond, params):
            sol = diffrax.diffeqsolve(diffrax.ODETerm(vf), t0=t0, t1=t1, y0=x, args=(params, cond), **kwargs)
            return sol.ys[0]

        return jax.jit(jax.vmap(solve_ode, in_axes=(0, 0, None)))

    def transport(self, source, condition=None, t0=0.0, t1=1.0, rng=None, **kwargs):
        rng = utils.default_prng_key(rng)
        latent = self.latent_fn(rng, (len(source),))

        if condition is not None:
            source = jnp.concatenate([source, condition], axis=-1)

        transport_fn = self.transport_fn if t0 == 0.0 and t1 == 1.0 and not kwargs else self._get_transport_fn(t0, t1, **kwargs)
        return transport_fn(latent, source, self.state_neural_net.params)


class FlowGW:

    def __init__(self, eps=0.01, embed_dim=128, n_freq=10, n_layers=3, cost_fn='cosine', lr=1e-4, n_epochs=200, seed=0, device='cpu'):
        jax.config.update('jax_enable_x64', False)

        self.eps       = eps
        self.embed_dim = embed_dim
        self.n_freq    = n_freq
        self.n_layers  = n_layers
        self.cost_fn   = costs.Cosine() if cost_fn == 'cosine' else cost_fn
        self.lr        = lr
        self.n_epochs  = n_epochs
        self.seed      = seed
        self.device    = self.get_device(device)

    def get_device(self, device):
        if device == 'cpu':
            return jax.devices('cpu')[0]

        if device == 'cuda' or device.startswith('cuda:'):
            try:
                devices = jax.devices('gpu')
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError('CUDA was requested but the JAX GPU backend is not available.') from exc

            idx = int(device.split(':')[1]) if ':' in device else 0
            if idx < 0 or idx >= len(devices):
                raise RuntimeError(f'CUDA device {idx} is not available.')
            return devices[idx]

        raise ValueError(f'Unknown device: {device}')

    def setup(self, source_train, target_train):
        x = source_train.detach().cpu().numpy() if torch.is_tensor(source_train) else np.asarray(source_train)
        y = target_train.detach().cpu().numpy() if torch.is_tensor(target_train) else np.asarray(target_train)

        with jax.default_device(self.device):
            self.x_train = jax.device_put(jnp.asarray(x, dtype=jnp.float32), self.device)
            self.y_train = jax.device_put(jnp.asarray(y, dtype=jnp.float32), self.device)

            source_dim = self.x_train.shape[1]
            target_dim = self.y_train.shape[1]

            neural_net = VelocityField(
                hidden_dims=[self.embed_dim] * self.n_layers,
                time_dims=[self.embed_dim, self.embed_dim],
                output_dims=[self.embed_dim, self.embed_dim, self.embed_dim] + [target_dim],
                condition_dims=[self.embed_dim, self.embed_dim, self.embed_dim],
                time_encoder=functools.partial(time_encoder.cyclical_time_encoder, n_freqs=self.n_freq),
            )

            linear_ot_solver = sinkhorn.Sinkhorn(momentum=acceleration.Momentum(value=1.0, start=25))
            solver = gromov_wasserstein.GromovWasserstein(epsilon=self.eps, linear_ot_solver=linear_ot_solver)

            self.model = GENOT(
                neural_net,
                flow=dynamics.ConstantNoiseFlow(0.0),
                ot_solver=solver,
                scale_cost='mean',
                input_dim=source_dim,
                output_dim=target_dim,
                cost_fn=self.cost_fn,
                lr=self.lr,
                fused_dim=0,
                iterations=self.n_epochs,
                k_latent_per_x=1,
                seed=self.seed,
            )

    def solve(self):
        with jax.default_device(self.device):
            loss = self.model(self.x_train, self.y_train)
        return {'flow_loss': loss}

    def continuous_approximation(self):
        pass

    def predict(self, source):
        x = source.detach().cpu().numpy() if torch.is_tensor(source) else np.asarray(source)

        with jax.default_device(self.device):
            x = jax.device_put(jnp.asarray(x, dtype=jnp.float32), self.device)
            prediction = self.model.transport(x, rng=jax.random.PRNGKey(0))

        prediction = np.asarray(prediction)
        return torch.as_tensor(prediction, dtype=source.dtype, device=source.device) if torch.is_tensor(source) else torch.as_tensor(prediction)

    def get_checkpoint(self):
        return {
            'params': jax.device_get(self.model.state_neural_net.params),
            'opt_state': jax.device_get(self.model.state_neural_net.opt_state),
            'rng': jax.device_get(self.model.rng),
        }
