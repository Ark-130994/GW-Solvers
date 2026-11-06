
from tqdm.auto import trange

import flax.linen as ln
import optax

from ott.geometry import costs
from solvers.continuous.reggw import RegGW
from solvers.continuous.models.mlp_jax import MLPJax


import torch

from solvers.continuous.cyclegw import CycleGW
from solvers.continuous.models.fcnn import FCNN


from solvers.continuous.flowgw import FlowGW


class ContinuousSolver:

    def __init__(self, config):

        self.config       = config
        self.solver_cfg   = config['solver']
        self.name         = self.solver_cfg['solver_name']
        self.n_epochs     = self.solver_cfg['n_epochs']
        self.report_every = self.solver_cfg['report_every']
        self.history      = []
        self.model        = None

        if self.report_every <= 0:
            raise ValueError('report_every must be positive.')

        dataset_cfg = config['dataset']

        if config['experiment_type'] == 'toy':
            _, source_name, target_name = dataset_cfg['toy_type'].split('_')
            source_dim = int(source_name[:-1])
            target_dim = int(target_name[:-1])
        else:
            source_dim = dataset_cfg['source_dim']
            target_dim = dataset_cfg['target_dim']

        
        if self.name == 'RegGW':

            mover_model = MLPJax(
                hidden_dims=self.solver_cfg['hidden_sizes_mlp'],
                out_dim=target_dim,
                act_fn=ln.relu,
            )

            mover_optimizer = optax.adam(learning_rate=self.solver_cfg['mover_lr'])

            cost_name = self.solver_cfg['cost_discrete']
            if cost_name == 'cosine':
                cost_fn = costs.Cosine()
            elif cost_name == 'euclidean':
                cost_fn = costs.SqEuclidean()
            else:
                raise ValueError(f'Unknown cost: {cost_name}')

            self.model = RegGW(
                mover_model=mover_model,
                mover_optimizer=mover_optimizer,
                source_dim=source_dim,
                cost_fn=cost_fn,
                n_samples=dataset_cfg.get('batch_size_train', dataset_cfg['n_train_samples']),
                eps_fit=self.solver_cfg['eps_fit'],
                eps_reg=self.solver_cfg['eps_reg'],
                lamb=self.solver_cfg['lamb'],
                rng_seed=self.solver_cfg.get('seed', 0),
                device=config['device'],
            )


        elif self.name == 'CycleGW':

            hidden_sizes = self.solver_cfg['hidden_sizes_mlp']
            device = config['device']

            F_model = FCNN(source_dim, target_dim, hidden_sizes=hidden_sizes).to(device)
            G_model = FCNN(target_dim, source_dim, hidden_sizes=hidden_sizes).to(device)

            F_optimizer = torch.optim.Adam(F_model.parameters(), lr=self.solver_cfg['f_lr'])
            G_optimizer = torch.optim.Adam(G_model.parameters(), lr=self.solver_cfg['g_lr'])

            models = {'F_model': F_model, 'G_model': G_model}
            optimizers = {'F_model': F_optimizer, 'G_model': G_optimizer}

            self.model = CycleGW(
                models=models,
                optimizers=optimizers,
                eps=self.solver_cfg['eps'],
                sigmas=self.solver_cfg['sigmas'],
                reg=self.solver_cfg['reg'],
                kernel_type=self.solver_cfg['kernel_type'],
                take_median=self.solver_cfg['take_median'],
                n_samples=dataset_cfg.get('batch_size_train', dataset_cfg['n_train_samples']),
            )

        
        elif self.name == 'FlowGW':

            self.model = FlowGW(
                n_samples=dataset_cfg.get('batch_size_train', dataset_cfg['n_train_samples']),
                eps=self.solver_cfg['eps'],
                embed_dim=self.solver_cfg['embed_dim'],
                n_freq=self.solver_cfg['n_freq'],
                n_layers=self.solver_cfg['n_layers'],
                cost_fn=self.solver_cfg['cost_fn'],
                lr=self.solver_cfg['lr'],
                n_epochs=self.n_epochs,
                seed=self.solver_cfg.get('seed', 0),
                device=config['device'],
            )


        else:
            raise ValueError(f'Unknown continuous solver: {self.name}')


    def fit(self, dataset, offline_logger=None):

        source_sampler = dataset['source_sampler']
        target_sampler = dataset['target_sampler']

        if self.model is None:
            raise ValueError(f'Unknown continuous solver: {self.name}')

        self.model.setup(source_sampler, target_sampler)
        self.history = []

        for epoch in trange(self.n_epochs, desc=self.name):
            row = self.model.train()
            row = {} if row is None else row
            row['step'] = epoch + 1
            self.history.append(row)

            if (epoch + 1) % self.report_every == 0 or epoch == self.n_epochs - 1:
                for item in self.history:
                    for key, value in item.items():
                        if key != 'step' and hasattr(value, 'item'):
                            item[key] = value.item()

                report_dataset = {**dataset, 'target_train_pred': self.model.predict(dataset['source_train']), 'target_test_pred': self.model.predict(dataset['source_test'])}

                if offline_logger is not None:
                    offline_logger(step=epoch + 1, model=self.model, history=self.history, dataset=report_dataset)

        return self.model
