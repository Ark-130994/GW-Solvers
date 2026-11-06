
import functools
import jax
import jax.numpy as jnp
import numpy as np
import torch

from ott.solvers import quadratic
from ott.geometry import costs, pointcloud
from ott.tools import sinkhorn_divergence


def _get_step_fn():

    def loss_fn(params, apply_fn, batch, cost_fn, eps_fit, eps_reg, lamb):
        mapped_samples = apply_fn({'params': params}, batch['source'])

        val_fitting_loss = fitting_loss(mapped_samples, batch['target'], eps_fit)
        val_regularizer = regularizer(batch['source'], mapped_samples, cost_fn, epsilon=eps_reg)
        val_tot_loss = val_fitting_loss + lamb * val_regularizer

        return val_tot_loss

    @functools.partial(jax.jit, static_argnums=5)
    def step_fn(state_neural_net, train_batch, cost_fn, eps_fit=0.01, eps_reg=0.001, lamb=1.0):
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
def regularizer(source, target, cost_fn=None, epsilon=None, relative_epsilon=None, scale_cost=1.0, **kwargs):

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

    gt_displacement_cost = jnp.mean(jax.vmap(costs.SqEuclidean())(source_cost, target_cost)) / n

    out = quadratic.solve(geom_xx, geom_yy)
    loss = gt_displacement_cost - out.reg_gw_cost

    return loss


class RegGW:

    def __init__(self, mover_model, mover_optimizer, source_dim, cost_fn, n_samples=256,
                 eps_fit=0.01, eps_reg=0.001, lamb=1.0, rng_seed=0, device='cpu'):

        self.mover_model     = mover_model
        self.mover_optimizer = mover_optimizer

        self.eps_fit   = eps_fit
        self.eps_reg   = eps_reg
        self.lamb      = lamb
        self.cost_fn   = cost_fn
        self.n_samples = n_samples
        self.rng_seed  = rng_seed
        self.source_dim = source_dim
        self.device    = self.get_device(device)

        self.state_neural_net = None
        self.step_fn = _get_step_fn()

    def get_device(self, device):
        if device == 'cpu':
            return jax.devices('cpu')[0]

        if device == 'cuda' or device.startswith('cuda:'):
            try:
                devices = jax.devices('gpu')
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError('JAX GPU backend is not available.') from exc

            idx = int(device.split(':')[1]) if ':' in device else 0
            if idx < 0 or idx >= len(devices):
                raise RuntimeError(f'CUDA device {idx} is not available.')
            return devices[idx]

        raise ValueError(f'Unknown device: {device}')

    def setup(self, source_sampler, target_sampler):
        self.source_sampler = source_sampler
        self.target_sampler = target_sampler

        with jax.default_device(self.device):
            self.rng = jax.random.PRNGKey(self.rng_seed)
            self.state_neural_net = self.mover_model.create_train_state(self.rng, self.mover_optimizer, self.source_dim)

            apply_fn = self.state_neural_net.apply_fn
            self.predict_fn = jax.jit(lambda params, x: apply_fn({'params': params}, x))

    def to_jax(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return jax.device_put(jnp.asarray(x), self.device)

    def train(self):

        x_train = self.source_sampler.sample(self.n_samples)
        y_train = self.target_sampler.sample(self.n_samples)

        if isinstance(x_train, (tuple, list)):
            x_train = x_train[0]
        if isinstance(y_train, (tuple, list)):
            y_train = y_train[0]

        with jax.default_device(self.device):
            train_batch = {
                'source': self.to_jax(x_train),
                'target': self.to_jax(y_train),
            }

            self.state_neural_net, mover_loss = self.step_fn(
                self.state_neural_net, train_batch, self.cost_fn,
                eps_fit=self.eps_fit, eps_reg=self.eps_reg, lamb=self.lamb
            )

        return {'mover_loss': mover_loss}

    def predict(self, source):
        with jax.default_device(self.device):
            x = self.to_jax(source)
            prediction = self.predict_fn(self.state_neural_net.params, x)

        prediction = np.asarray(prediction)

        if torch.is_tensor(source):
            return torch.as_tensor(prediction, dtype=source.dtype, device=source.device)

        return torch.as_tensor(prediction)

    def get_checkpoint(self):
        return {
            'params': jax.device_get(self.state_neural_net.params),
            'opt_state': jax.device_get(self.state_neural_net.opt_state),
        }
