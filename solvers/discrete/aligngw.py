
import numpy as np
import torch

from scipy.spatial.distance import cdist
from sklearn.neural_network import MLPRegressor
from ot.gromov import gwggrad, gwloss
from ot import bregman


class AlignGW:

    def __init__(self, cost_name='euclidean', normalize_dists='max', loss_fun='square_loss', eps=0.01, ot_warm=True, approximation='argmax', device='cpu'):

        self.cost_name       = cost_name
        self.normalize_dists = normalize_dists
        self.loss_fun        = loss_fun
        self.eps             = eps
        self.ot_warm         = ot_warm
        self.approximation   = approximation
        self.device          = torch.device(device)

        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available.')

    def compute_distances(self, x, y):

        Cx = cdist(x, x, metric=self.cost_name)
        Cy = cdist(y, y, metric=self.cost_name)

        if self.normalize_dists == 'max':
            Cx /= Cx.max()
            Cy /= Cy.max()
        elif self.normalize_dists == 'mean':
            Cx /= (Cx.mean() + 1e-10)
            Cy /= (Cy.mean() + 1e-10)
        elif self.normalize_dists == 'median':
            Cx /= np.median(Cx)
            Cy /= np.median(Cy)

        return Cx, Cy

    def init_matrix(self, Cx, Cy, p, q):

        if self.loss_fun == 'square_loss':
            f1  = Cx**2 / 2
            f2  = Cy**2 / 2
            hCx = Cx
            hCy = Cy

        elif self.loss_fun == 'kl_loss':
            f1  = Cx * torch.log(Cx + 1e-15) - Cx
            f2  = Cy
            hCx = Cx
            hCy = torch.log(Cy + 1e-15)

        else:
            raise ValueError(f'Unknown loss function: {self.loss_fun}')

        constC1 = (f1 @ p).reshape(-1, 1)
        constC2 = (q @ f2.T).reshape(1, -1)
        constC  = constC1 + constC2

        return constC, hCx, hCy

    def setup(self, source_train, target_train):

        self.x = source_train.detach().cpu().numpy() if torch.is_tensor(source_train) else np.asarray(source_train)
        self.y = target_train.detach().cpu().numpy() if torch.is_tensor(target_train) else np.asarray(target_train)

        Cx, Cy = self.compute_distances(self.x, self.y)
        self.Cx = torch.as_tensor(Cx, dtype=torch.float64, device=self.device)
        self.Cy = torch.as_tensor(Cy, dtype=torch.float64, device=self.device)

        self.p = torch.full((len(self.x),), 1.0 / len(self.x), dtype=torch.float64, device=self.device)
        self.q = torch.full((len(self.y),), 1.0 / len(self.y), dtype=torch.float64, device=self.device)

        self.constC, self.hCx, self.hCy = self.init_matrix(self.Cx, self.Cy, self.p, self.q)

        self.T = torch.outer(self.p, self.q)
        self.warmstart = None
        self.continuous_model = None

    @torch.no_grad()
    def solve(self):

        Tprev = self.T.clone()
        tens  = gwggrad(self.constC, self.hCx, self.hCy, self.T)

        if self.ot_warm:
            self.T, log = bregman.sinkhorn_knopp(self.p, self.q, tens, self.eps, warmstart=self.warmstart, log=True)
            self.warmstart = (torch.log(log['u']), torch.log(log['v']))
        else:
            self.T = bregman.sinkhorn(self.p, self.q, tens, self.eps)

        return {
            'gw_loss': gwloss(self.constC, self.hCx, self.hCy, self.T).item(),
            'coupling_change': torch.linalg.norm(self.T - Tprev).item(),
        }

    @torch.no_grad()
    def continuous_approximation(self):

        if self.approximation == 'argmax':
            indices = torch.argmax(self.T, dim=1).cpu().numpy()
            y_pred  = self.y[indices]

        elif self.approximation == 'barycentric':
            y = torch.as_tensor(self.y, dtype=self.T.dtype, device=self.device)
            y_pred = ((self.T @ y) / self.T.sum(dim=1, keepdim=True)).cpu().numpy()

        else:
            raise ValueError(f'Unknown approximation method: {self.approximation}')

        self.continuous_model = MLPRegressor(hidden_layer_sizes=256, random_state=1, max_iter=500)
        self.continuous_model.fit(self.x, y_pred)

    def predict(self, source):

        if self.continuous_model is None:
            raise RuntimeError('Continuous model has not been fitted.')

        prediction = self.continuous_model.predict(source.detach().cpu().numpy())
        return torch.as_tensor(prediction, dtype=source.dtype, device=source.device)

    def get_checkpoint(self):
        return {'T': self.T.detach().cpu().numpy(), 'continuous_model': self.continuous_model}
