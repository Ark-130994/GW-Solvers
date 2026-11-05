
import numpy as np
import torch
import jax

import jax.numpy as jnp
from sklearn.neural_network import MLPRegressor
from ott.geometry.costs import CostFn
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn


@jax.tree_util.register_pytree_node_class
class GWCost_IP(CostFn):
    def __init__(self):
        super().__init__()

    def pairwise(self, x: jnp.ndarray, y: jnp.ndarray):
        return -jnp.vdot(x, y)


def prox_lasso(x, l1reg=None, scaling=1.0):
    if l1reg is None:
        l1reg = 1.0

    if isinstance(l1reg, (float, int)):
        l1reg = jax.tree_util.tree_map(lambda y: l1reg * jnp.ones_like(y), x)

    return jax.tree_util.tree_map(lambda u, v: jnp.sign(u) * jax.nn.relu(jnp.abs(u) - v * scaling), x, l1reg)


def update_m(x, y, method='exact', solver_state=None, reg=0.5):
    py = solver_state.apply(y.T, axis=1)
    m  = py @ x

    if method == 'l1_reg':
        m = prox_lasso(m, l1reg=reg)
    elif method != 'exact':
        raise ValueError(f'Unknown matrix update method: {method}')

    return m


def update_geometry(x, y, m, geom_epsilon, geom_batch_size=None):
    return PointCloud(x=x @ m.T, y=y, epsilon=geom_epsilon, batch_size=geom_batch_size, cost_fn=GWCost_IP())


@jax.jit
def entropic_pred(x, y, y_emb, g, b, eps):
    logits  = (x @ y_emb.T + g[None, :]) / eps + jnp.log(b)[None, :]
    weights = jax.nn.softmax(logits, axis=1)
    return weights @ y


class StructuredGW:

    def __init__(self, m_init=None, method_m='exact', eps=1e-4, tol=1e-3, experiment_type='embedding', device='cpu', inference_type='mlp'):
        
        jax.config.update('jax_enable_x64', True)
        self.m_init          = m_init
        self.method_m        = method_m
        self.eps             = eps
        self.tol             = tol
        self.experiment_type = experiment_type
        self.device          = self.get_device(device)
        self.inference_type  = inference_type
        

        if self.inference_type not in ['mlp', 'entropic']:
            raise ValueError(f'Unknown inference type: {self.inference_type}')

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
        self.x = source_train.detach().cpu().double().numpy() if torch.is_tensor(source_train) else np.asarray(source_train, dtype=np.float64)
        self.y = target_train.detach().cpu().double().numpy() if torch.is_tensor(target_train) else np.asarray(target_train, dtype=np.float64)

        with jax.default_device(self.device):
            self.x_train = jax.device_put(jnp.asarray(self.x, dtype=jnp.float64), self.device)
            self.y_train = jax.device_put(jnp.asarray(self.y, dtype=jnp.float64), self.device)

            if self.m_init is None:
                self.m = jnp.ones((self.y.shape[1], self.x.shape[1]), dtype=jnp.float64) / max(self.y.shape[1], self.x.shape[1])
            else:
                self.m = jax.device_put(jnp.asarray(self.m_init, dtype=jnp.float64), self.device)

            self.a = jnp.ones(len(self.x), dtype=jnp.float64) / len(self.x)
            self.b = jnp.ones(len(self.y), dtype=jnp.float64) / len(self.y)

            self.ot_solver = sinkhorn.Sinkhorn(threshold=self.tol, max_iterations=1000, norm_error=2, lse_mode=True)

            geom = update_geometry(self.x_train, self.y_train, self.m, self.eps)
            prob = linear_problem.LinearProblem(geom, a=self.a, b=self.b)

            rng = jax.random.PRNGKey(0)
            initializer = self.ot_solver.create_initializer()
            self.f, self.g = initializer(prob, None, None, lse_mode=True, rng=rng)

        self.solver_state     = None
        self.continuous_model = None
        self.y_emb            = None

        x      = self.x_train
        y      = self.y_train
        a      = self.a
        b      = self.b
        eps    = self.eps
        method = self.method_m
        solver = self.ot_solver

        def structured_gw_step(m, f, g):
            geom   = update_geometry(x, y, m, eps)
            prob   = linear_problem.LinearProblem(geom, a=a, b=b)
            state1 = solver(prob, (f, g))

            m = update_m(x, y, method=method, solver_state=state1, reg=eps)

            geom   = update_geometry(x, y, m, eps)
            prob   = linear_problem.LinearProblem(geom, a=a, b=b)
            state2 = solver(prob, (state1.f, state1.g))

            return m, state2, state1.n_iters, state1.converged

        self._step = jax.jit(structured_gw_step, device=self.device)

    def solve(self):
        self.m, self.solver_state, n_iters_1, converged_1 = self._step(self.m, self.f, self.g)
        self.f, self.g = self.solver_state.f, self.solver_state.g

        return {
            'ot_cost': float(np.asarray(self.solver_state.reg_ot_cost)),
            'sinkhorn_iters_1': int(np.asarray(n_iters_1)),
            'sinkhorn_iters_2': int(np.asarray(self.solver_state.n_iters)),
            'sinkhorn_converged_1': int(np.asarray(converged_1)),
            'sinkhorn_converged_2': int(np.asarray(self.solver_state.converged)),
        }

    @property
    def coupling(self):
        if self.solver_state is None:
            return None
        return self.solver_state.matrix

    def continuous_approximation(self):
        if self.solver_state is None:
            raise RuntimeError('StructuredGW has not been trained.')

        if self.inference_type == 'mlp':
            y_emb  = self.y_train @ self.m
            y_pred = entropic_pred(self.x_train, self.y_train, y_emb, self.g, self.b, self.eps)

            self.continuous_model = MLPRegressor(hidden_layer_sizes=256, random_state=1, max_iter=500)
            self.continuous_model.fit(self.x, np.asarray(y_pred))
            self.y_emb = None

        elif self.inference_type == 'entropic':
            self.y_emb = self.y_train @ self.m
            self.continuous_model = None

    def predict(self, source):
        if self.inference_type == 'mlp':
            if self.continuous_model is None:
                raise RuntimeError('Continuous model has not been fitted.')

            prediction = self.continuous_model.predict(source.detach().cpu().numpy())

        elif self.inference_type == 'entropic':
            if self.y_emb is None:
                raise RuntimeError('Entropic inference has not been initialized.')

            x = jax.device_put(jnp.asarray(source.detach().cpu().double().numpy()), self.device)
            eps_inference = 0.5 #max(self.eps, 0.05)
            prediction = np.asarray(entropic_pred(x, self.y_train, self.y_emb, self.g, self.b, eps_inference))

        return torch.as_tensor(prediction, dtype=source.dtype, device=source.device)

    def get_checkpoint(self):
        return {
            'coupling': np.asarray(self.coupling),
            'm': np.asarray(self.m),
            'f': np.asarray(self.f),
            'g': np.asarray(self.g),
            'b': np.asarray(self.b),
            'continuous_model': self.continuous_model,
            'inference_type': self.inference_type,
            'target_train': self.y if self.inference_type == 'entropic' else None,
        }
