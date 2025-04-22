import torch
from tqdm.auto import trange
from tqdm import tqdm_notebook as tqdm
from src.metrics import compute_metrics_2
import wandb
import random
import numpy as np
import jax
from functools import partial

import jax.numpy as jnp
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from tqdm.notebook import tqdm
from ott.geometry.costs import CostFn

from ott.geometry.pointcloud import PointCloud

from typing import Optional
from typing import Tuple

import jax
import jax.numpy as jnp

from jaxopt._src import tree_util
from typing import Any

from jax import tree_util as tu
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

tree_map = tu.tree_map

def prox_lasso(x: Any,
               l1reg: Optional[Any] = None,
               scaling: float = 1.0) -> Any:

    if l1reg is None:
        l1reg = 1.0

    if type(l1reg) == float:
        l1reg = tree_util.tree_map(lambda y: l1reg*jnp.ones_like(y), x)

    def fun(u, v): return jnp.sign(u) * jax.nn.relu(jnp.abs(u) - v * scaling)
    return tree_util.tree_map(fun, x, l1reg)


def update_M(X, Y, method="exact", solver_state=None, reg=0.5):
    if method == "exact":
        PY = (solver_state.matrix@Y).T#solver_state.apply(Y.T, axis=1)
        M = PY.dot(X)
    elif method == "l1_reg":
        PY = (solver_state.matrix@Y).T#solver_state.apply(Y.T, axis=1)
        M = PY.dot(X)
        M = prox_lasso(M, l1reg=reg)
    #elif method == "l12_reg":
    #    PY = solver_state.matrix@Y#solver_state.apply(Y.T, axis=1)
    #    M = PY.dot(X)
    #    M = L21columns().prox(M, gamma=kwargs["l12_reg"])

    return M
def report_wandb_fn(metrics_dict, metrics_names, epoch, prefix):
    for metric_name in metrics_names:
        wandb.log({f'{prefix}/{metric_name}':metrics_dict[metric_name][-1]}, step=epoch)

@jax.jit
def f_eps(x, Y, g, beta, eps):
    a = (g[None, :] + x.dot(Y.T)) / eps
    return - eps * jax.nn.logsumexp(a=a, b=beta)

def beta_tilde_fn(Y, g, beta, eps):
    @jax.jit
    def beta_tilde_sample(x):
        return - jax.grad(f_eps, argnums=2)(x, Y, g, beta, eps)

    return beta_tilde_sample

@jax.jit
def beta_tilde_vec(x, Y, g, beta, eps):
    fun = beta_tilde_fn(Y, g, beta, eps)
    return jax.jit(jax.vmap(fun))(x)

def entropic_pred(X, Y, M, g, beta, eps):
    betas = beta_tilde_vec(X, Y.dot(M), g, beta, eps)
    return betas.dot(Y)

@jax.tree_util.register_pytree_node_class
class GWCost_IP(CostFn):
    def __init__(self):
        super().__init__()

    def pairwise(self, x: jnp.ndarray, y: jnp.ndarray):
        return - jnp.vdot(x, y)

#@partial(jax.jit, static_argnums=(8,))
def generate_preds(X_train, Y_train, X_test, M, g, beta, eps, batch_size):
    X_train1 = X_train[:(X_train.shape[0] // batch_size) * batch_size, :]
    X_train1 = X_train1.reshape((X_train1.shape[0] // batch_size, batch_size, X_train1.shape[1]))
    X_train2 = X_train[(X_train.shape[0] // batch_size) * batch_size:]
    genes_test_pred = jnp.zeros(shape=(X_train1.shape[0], batch_size, X_test.shape[1]))
    
    for ix in range(X_train1.shape):
        betas = beta_tilde_vec(X_train[ix], Y_train.dot(M), g, beta, eps)
        new_pred_test = betas.dot(X_test)
        genes_test_pred = genes_test_pred.at[i].set(new_pred_test)

    genes_test_pred = genes_test_pred.reshape(X_train1.shape[0] * batch_size, X_test.shape[1])
    last_pred_test = beta_tilde_vec(X_train2, Y_train.dot(M), g, beta, eps).dot(X_test)
    genes_test_pred = jnp.concatenate((genes_test_pred, last_pred_test))
    return genes_test_pred
    
@partial(jax.jit, static_argnums=(4,))
def update_geometry(X, Y, M, geom_epsilon, geom_batch_size,):
    z_x = X.dot(M.T)
    z_y = Y
    cost_fn = GWCost_IP()
    return PointCloud(x=z_x, y=z_y, epsilon=geom_epsilon, batch_size=geom_batch_size, cost_fn=cost_fn)

class StructuredGW():
    def __init__(self, M_init, method_M, eps=1e-4, tol=1e-3, toy_type=False):
        self.M_init = M_init
        self.method_M = 'exact'
        self.eps = eps
        self.tol = tol
        self.toy_type = toy_type
        
    #def solve(X, Y, labels, target_vectors, M=None, eps=1e-4, method_M='exact', threshold=1e-3, max_iter=1000):
    def solve(self, x_dict, y_dict, labels_dict, target_vectors, wandb_report=False, maxiter=200, report_every=20):
        
        x_train, y_train, labels_train = x_dict['train'], y_dict['train'], labels_dict['train']
        x_test, y_test, labels_test = x_dict['test'], y_dict['test'], labels_dict['test']
        
        x_train_jnp, y_train_jnp = np.asarray(x_train.cpu().numpy()), np.asarray(y_train.cpu().numpy())
        
        if self.M_init is None:
            M = jnp.ones((y_train.shape[1], x_train.shape[1])) / jnp.maximum(y_train.shape[1], x_train.shape[1])
        else:
            M = self.M_init

        a = jnp.ones(x_train.shape[0]) / x_train.shape[0]
        b = jnp.ones(y_train.shape[0]) / y_train.shape[0]

        solver = sinkhorn.Sinkhorn(
            threshold=self.tol,
            max_iterations=1000,
            norm_error=2,
            lse_mode=True,
        )

        rng = jax.random.PRNGKey(0)
        initializer = solver.create_initializer()
        geom = update_geometry(X=x_train_jnp, Y=y_train_jnp, M=M, geom_batch_size=None, geom_epsilon=self.eps)
        prob = linear_problem.LinearProblem(geom, a=a, b=b)
        init_dual_a, init_dual_b = initializer(prob, *(None, None), lse_mode=True, rng=rng)

        metric_names = ['Top@1', 'Top@5', 'Top@10', 'cossim_gt', 'inner_gw', 'foscttm']
        #metric_names = ['Top@1', 'Top@5', 'Top@10', 'cossim_gt', 'inner_gw', 'foscttm', 'distortion', 'mmd', 'bw_uvp', 'sinkhorn_divergence']
        

        #report_keys = ['train_bary', 'train_entropic']
        metrics_dict_train = {metric_name:[] for metric_name in metric_names}
        metrics_dict_test = {metric_name:[] for metric_name in metric_names}

        #costs = - jnp.ones(max_iter)

        for it in tqdm(range(maxiter)):

            solver_state = solver(prob, (init_dual_a, init_dual_b))

            init_dual_a, init_dual_b = solver_state.f, solver_state.g


            M = update_M(x_train_jnp, y_train_jnp, method=self.method_M, solver_state=solver_state, reg=self.eps)

            geom = update_geometry(X=x_train_jnp, Y=y_train_jnp, M=M, geom_epsilon=geom.epsilon,
                                       geom_batch_size=geom.batch_size)

            prob = linear_problem.LinearProblem(geom, a=a, b=b)

            solver_state = solver(prob, (init_dual_a, init_dual_b))
            init_dual_a, init_dual_b = solver_state.f, solver_state.g  
            coupling_matrix = solver_state.matrix

            if (it % report_every == 0 and it != 0) or it == maxiter-1:
                #y_sampled1 = (coupling_matrix @ Y) / coupling_matrix.sum(axis=1, keepdims=True)
                #y_sampled1 = torch.tensor(np.asarray(y_sampled1)).to(torch.float32)
                
                #y_sampled_train = entropic_pred(x_train.cpu().numpy(), y_train.cpu().numpy(), M, solver_state.g, b, eps)
                #y_sampled_train = torch.tensor(np.asarray(y_sampled_train)).to(torch.float32)
#
                #metrics_train_dict = {metric:[] for metric in metric_names}
                #metrics_train_dict = compute_metrics(x_train, y_train, y_sampled, torch.tensor(labels), target_vectors, metrics_train_dict2)
                
                continuous_solver = MLPRegressor(hidden_layer_sizes=256, random_state=1, max_iter=500)
                
                if self.toy_type is False:
                    if wandb_report:
                        y_sampled_train = entropic_pred(x_train.cpu().numpy(), y_train.cpu().numpy(), M, solver_state.g, b, self.eps)
                        y_sampled_train = torch.tensor(np.asarray(y_sampled_train)).to(torch.float32)
                        metrics_dict_train = compute_metrics_2(x_train.cpu(), y_train.cpu(), y_sampled_train.cpu(), labels_train.cpu(), target_vectors, metrics_dict_train)
                        report_wandb_fn(metrics_dict_train, metric_names, it, 'train')

                        continuous_solver.fit(x_train.cpu().numpy(), y_sampled_train.cpu().numpy())
                        y_sampled_test = continuous_solver.predict(x_test.cpu().numpy())
                        y_sampled_test = torch.tensor(y_sampled_test).to(torch.float32)
    #
                        metrics_dict_test = compute_metrics_2(x_test.cpu(), y_test.cpu(), y_sampled_test.cpu(), labels_test.cpu(), target_vectors, metrics_dict_test)
                        report_wandb_fn(metrics_dict_test, metric_names, it, 'test')
                else:
                    T = np.asarray(solver_state.matrix)
                    y_sampled_np = (T @ y_train.cpu().numpy())/T.sum(axis=1, keepdims=True)
                    #y_sampled = torch.tensor(y_sampled).to(torch.float32)
                
                    fig = plt.figure(figsize=(8, 8))
                    
                    if self.toy_type == 'toy_2d_3d':
                        ax = fig.add_subplot(projection='3d')
                       
                    if self.toy_type == 'toy_3d_2d':
                        ax = fig.add_subplot(projection=None)
                        
                    ax.scatter(*y_sampled_np.T, c=labels_train.cpu().numpy(),  cmap="Spectral")
                    ax.set_title('StructuredGW')
                    
                    plt.show()


        return coupling_matrix, continuous_solver, solver_state.g, M, b
            #y_pred = (coupling_matrix @ Y) / coupling_matrix.sum(axis=1, keepdims=True)
        
    def fit(self, x_dict, y_dict, labels_dict, target_vectors, wandb_report, max_iters, report_every):

        self.x_dict, self.y_dict, self.labels_dict = x_dict, y_dict, labels_dict
        #self.coupling, self.g, self.M, self.b = train(self.x_train, self.y_train, labels, target_vectors, None, max_iter=max_iters, eps=self.eps)
        
        coupling, continuous_solver, g, M, b  = self.solve(self.x_dict, self.y_dict, self.labels_dict, target_vectors, wandb_report, max_iters, report_every)
        
        self.coupling = coupling
        self.continuous_solver = continuous_solver
        self.g = g
        self.b = b
        
    def valid_step(self, sampler_source, sampler_target, n_samples, metric_names, target_vectors, n_eval):
            
        metrics_dict = {metric_name:[] for metric_name in metric_names}
        
        with torch.no_grad():
        
            #sampler_source.reset_sampler()

            for _ in trange(n_eval, leave=False, desc="Evaluation"):
                
                if sampler_target is None:
                    x, y, labels = sampler_source.sample(n_samples)
                else:
                    #sampler_target.reset_sampler()
                    x, labels = sampler_source.sample(n_samples)
                    y, _      = sampler_target.sample(n_samples)
                    
                x, y, labels = x.cpu(), y.cpu(), labels.cpu()
                y_sampled = self.continuous_solver.predict(x.numpy())

                y_sampled = torch.tensor(y_sampled).to(torch.float32)

                metrics_dict = compute_metrics(x, y, y_sampled, labels, target_vectors, metrics_dict)
            
            return metrics_dict

    def valid_step_2(self, source_samples, target_samples, n_samples, metric_names, target_vectors, n_eval):
            
        metrics_dict = {metric_name:[] for metric_name in metric_names}
        x = source_samples[:]
        y = target_samples[:]
        
        #print(metrics_dict)
        with torch.no_grad():

                    
            x, y, labels = x.cpu(), y.cpu(), labels.cpu()
            y_sampled = self.continuous_solver.predict(x.numpy())
            y_sampled = torch.tensor(y_sampled).to(torch.float32)
            metrics_dict = compute_metrics_2(x, y, y_sampled, labels, target_vectors, metrics_dict, valid_step)
            
        return metrics_dict
    