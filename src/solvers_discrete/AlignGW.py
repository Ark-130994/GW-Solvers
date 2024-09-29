import numpy as np
import ot
import torch
import scipy as sp
from scipy.stats import describe
from time import time
import matplotlib.pyplot as plt
from src.metrics import compute_metrics
from tqdm.auto import trange
import wandb
from src.utils import cosine_similarity
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from tqdm import tqdm_notebook as tqdm

from ot import bregman
from ot.gromov import gwggrad, gwloss

from src.models.general_solvers import sinkhorn_knopp



def report_wandb_fn(metrics_dict, metrics_names, epoch, prefix):
    for metric_name in metrics_names:
        wandb.log({f'{prefix}/{metric_name}':metrics_dict[metric_name][-1]}, step=epoch)
            

class AlignGW():

    def __init__(self, metric, normalize_dists, loss_fun, eps, tol, toy_type=None):
        
        self.metric = metric
        self.normalize_dists = normalize_dists
        self.loss_fun = loss_fun
        self.eps = eps
        self.tol = tol
        self.ot_warm = True
        self.toy_type = toy_type

    def compute_distances(self, x, y):
        
        Cx = sp.spatial.distance.cdist(x, x, metric=self.metric)
        Cy = sp.spatial.distance.cdist(y, y, metric=self.metric)
            
        if self.normalize_dists == 'max':
            Cx /= Cx.max()
            Cy /= Cy.max()
        elif self.normalize_dists == 'mean':
            Cx /= Cx.mean()
            Cy /= Cy.mean()
        elif self.normalize_dists == 'median':
            Cx /= torch.median(Cx)
            Cy /= torch.median(Cy)

        self.Cx, self.Cy = Cx, Cy

    def compute_gamma_entropy(self, G):

        Prod = G * (np.log(G) - 1)
        ent = np.nan_to_num(Prod).sum()

        return ent

    def init_matrix(self, Cx, Cy, T, p, q, loss_fun='square_loss'):

        if loss_fun == 'square_loss':
            def f1(a):
                return (a**2) / 2#in POT this is only a**2, the same for below

            def f2(b):
                return (b**2) / 2

            def h1(a):
                return a

            def h2(b):
                return b
            
        elif loss_fun == 'kl_loss':
            def f1(a):
                return a * np.log(a + 1e-15) - a

            def f2(b):
                return b

            def h1(a):
                return a

            def h2(b):
                return np.log(b + 1e-15)

        constC1 = np.dot(np.dot(f1(Cx), p.reshape(-1, 1)),
                    np.ones(len(q)).reshape(1, -1))
        constC2 = np.dot(np.ones(len(p)).reshape(-1, 1),
                    np.dot(q.reshape(1, -1), f2(Cy).T))
        
        constC = constC1 + constC2
        hCx = h1(Cx)
        hCy = h2(Cy)
        
        return constC, hCx, hCy

    def solve(self, x_dict, y_dict, labels_dict, target_vectors, wandb_report=False, maxiter=200, report_every=20):

        x_train, y_train, labels_train = x_dict['train'], y_dict['train'], labels_dict['train']
        x_test, y_test, labels_test = x_dict['test'], y_dict['test'], labels_dict['test']
        
        x_train_np, y_train_np, labels_train_np = x_train.cpu().numpy(), y_train.cpu().numpy(), labels_train.cpu().numpy() 
        
        p = ot.unif(x_train_np.shape[0])
        q = ot.unif(y_train_np.shape[0])
        
        self.compute_distances(x_train_np, y_train_np)
        metric_names = ['Top@1', 'Top@5', 'Top@10', 'cossim_gt', 'inner_gw', 'foscttm']

        Cx = self.Cx 
        Cy = self.Cy 
        T = np.outer(p, q)  
        
        constC, hCx, hCy = self.init_matrix(Cx, Cy, T, p, q, self.loss_fun)
        
        err = 1
        metrics_dict_train = {metric_name:[] for metric_name in metric_names}
        metrics_dict_test = {metric_name:[] for metric_name in metric_names}
        #print(self.eps)
        #while (err > self.tol and it <= maxiter):
        for it in tqdm(range(maxiter)):
            
            Tprev = T
            tens = gwggrad(constC, hCx, hCy, T)

            if self.ot_warm and it > 0:
                #T, log = bregman.sinkhorn_knopp(p, q, tens, self.eps, init_u=log['u'], init_v=log['v'], log=True)
                T, log = sinkhorn_knopp(p, q, tens, self.eps, init_u=log['u'], init_v=log['v'], log=True)
                
            elif self.ot_warm:
                
                T, log = bregman.sinkhorn(p, q, tens, self.eps, log=True)
            else:
                T = bregman.sinkhorn(p, q, tens, self.eps)
                
            if (it) % report_every == 0 or it == maxiter-1:
                
                #dist = gwloss(constC, hCx, hCy, T)
              
                err = np.linalg.norm(T - Tprev)
                if err < self.tol:
                    print(f'Converged after {it}...')
                    
                continuous_solver = MLPRegressor(hidden_layer_sizes=256, random_state=1, max_iter=500)
                
                if self.toy_type is None:

                    if wandb_report:
                        y_sampled_train_np = (T @ y_train_np)/T.sum(axis=1, keepdims=True)
                        y_sampled_train = torch.tensor(y_sampled_train_np).to(torch.float32)
                        metrics_dict_train = compute_metrics(x_train.cpu(), y_train.cpu(), y_sampled_train.cpu(), labels_train.cpu(), target_vectors, metrics_dict_train)
                        report_wandb_fn(metrics_dict_train, metric_names, it, 'train')

                        continuous_solver.fit(x_train.cpu().numpy(), y_sampled_train.cpu().numpy())
                        y_sampled_test = continuous_solver.predict(x_test.cpu().numpy())
                        y_sampled_test = torch.tensor(y_sampled_test).to(torch.float32)
    #
                        metrics_dict_test = compute_metrics(x_test.cpu(), y_test.cpu(), y_sampled_test.cpu(), labels_test.cpu(), target_vectors, metrics_dict_test)
                        report_wandb_fn(metrics_dict_test, metric_names, it, 'test')
                else:
                    
                    y_sampled_np = (T @ y_train_np)/T.sum(axis=1, keepdims=True)
                
                    fig = plt.figure(figsize=(8, 8))
                    
                    if self.toy_type == 'toy_2d_3d':
                        ax = fig.add_subplot(projection='3d')
                       
                    if self.toy_type == 'toy_3d_2d':
                        ax = fig.add_subplot(projection=None)
                        
                    ax.scatter(*y_sampled_np.T, c=labels_train.cpu().numpy(),  cmap="Spectral")
                    plt.show()
            
        return T, continuous_solver
        
    def fit(self, x_dict, y_dict, labels_dict, target_vectors, wandb_report, max_iters, report_every):
        
        self.x_dict, self.y_dict, self.labels_dict = x_dict, y_dict, labels_dict
                    
        coupling, continuous_solver  = self.solve(self.x_dict, self.y_dict, self.labels_dict, target_vectors, wandb_report, max_iters, report_every)
        
        self.coupling = coupling
        self.continuous_solver = continuous_solver
    
    def valid_step(self, sampler, n_samples, metric_names, target_vectors, n_eval):
            
        metrics_dict = {metric_name:[] for metric_name in metric_names}
        
        with torch.no_grad():
        
            sampler.reset_sampler()

            for _ in trange(n_eval, leave=False, desc="Evaluation"):
                x, y, labels = sampler.sample(n_samples)
                x, y, labels = x.cpu(), y.cpu(), labels.cpu()
                y_sampled = self.continuous_solver.predict(x.numpy())

                y_sampled = torch.tensor(y_sampled).to(torch.float32)

                metrics_dict = compute_metrics(x, y, y_sampled, labels, target_vectors, metrics_dict)
            
            return metrics_dict