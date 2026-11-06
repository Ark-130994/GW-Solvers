
import copy
import torch
from tqdm.auto import trange

from solvers.neural_innergw.neuralgw import NeuralGW
from solvers.neural_innergw.models.mlp_torch import MLP
from solvers.neural_innergw.models.p_init import estimate_gaussian_params, compute_optimal_P
from solvers.neural_innergw.models.cost_models import InnerGWLinear


class EarlyStoppingTop1:

    def __init__(self, config):
        self.metric_split = config.get('metric_split', 'test')
        self.metric_name  = config.get('metric_name', 'Top@1')
        self.min_delta    = config.get('min_delta', 1e-3)

        self.degradation_patience = config.get('degradation_patience', 5)
        self.plateau_patience     = config.get('plateau_patience', 3)
        self.good_threshold      = config.get('good_threshold', 0.95)

        self.best_score      = float('-inf')
        self.best_step       = None
        self.best_checkpoint = None

        self.degradation_count   = 0
        self.good_plateau_count  = 0
        self.stop_reason         = None

    def step(self, step, metrics, model):

        score = float(metrics[self.metric_name])
        improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score      = score
            self.best_step       = step
            self.best_checkpoint = copy.deepcopy(model.get_checkpoint())
            self.degradation_count  = 0
            self.good_plateau_count = 0

        else:
            if score < self.best_score - self.min_delta:
                self.degradation_count += 1
            else:
                self.degradation_count = 0

            if score >= self.good_threshold:
                self.good_plateau_count += 1
            else:
                self.good_plateau_count = 0

        if self.degradation_count >= self.degradation_patience:
            self.stop_reason = f'{self.metric_name} degraded for {self.degradation_patience} reports.'

        elif self.best_score >= self.good_threshold and self.good_plateau_count >= self.plateau_patience:
            self.stop_reason = f'{self.metric_name} plateaued above {self.good_threshold} for {self.plateau_patience} reports.'

        return self.stop_reason is not None


class NeuralInnerGWSolver:

    def __init__(self, config):

        self.config        = config
        self.solver_cfg    = config['solver']
        self.name          = self.solver_cfg['solver_name']
        self.n_epochs      = self.solver_cfg['n_epochs']
        self.report_every  = self.solver_cfg['report_every']
        self.device        = config['device']

        self.init_P        = self.solver_cfg.get('init_P', False)
        self.epochs_init_P = self.solver_cfg.get('epochs_init_P', 0)

        self.model         = None
        self.history       = []
        self.stop_reason   = None

        if self.name != 'NeuralGW':
            raise ValueError(f'Unknown solver: {self.name}')

        if self.report_every <= 0:
            raise ValueError('report_every must be positive.')

        if self.init_P and not 0 < self.epochs_init_P < self.n_epochs:
            raise ValueError('epochs_init_P must be between 1 and n_epochs - 1.')

        early_cfg = self.solver_cfg.get('early_stopping', {})
        self.early_stopper = EarlyStoppingTop1(early_cfg) if early_cfg.get('enabled', False) else None

        if self.early_stopper is not None and config['experiment_type'] != 'embedding':
            raise ValueError('Top@1 early stopping requires embedding evaluation.')

    def _make_trainable_cost(self):

        cost_model = InnerGWLinear(self.source_dim, self.target_dim, device=self.device)
        cost_optimizer = torch.optim.Adam(cost_model.parameters(), lr=self.solver_cfg['cost_lr'])

        self.model.set_trainable_cost(cost_model, cost_optimizer)

    def _initialize_P(self, dataset):

        cfg = self.config['dataset']
        n_samples = min(cfg.get('init_P_samples', 10000), cfg['n_train_samples'])
        batch_size = cfg.get('batch_size_train', cfg['n_train_samples'])

        m_s, cov_s = estimate_gaussian_params(dataset['source_sampler'], n_samples, batch_size)
        m_t, cov_t = estimate_gaussian_params(dataset['target_sampler'], n_samples, batch_size)

        for sampler in (dataset['source_sampler'], dataset['target_sampler']):
            if hasattr(sampler, 'reset_sampler'):
                sampler.reset_sampler()

        P = compute_optimal_P(m_s, m_t, cov_s, cov_t, self.source_dim, self.target_dim)

        self.model.set_fixed_P(P)

    def setup(self, dataset):

        self.source_dim = dataset['source_train'].shape[-1]
        self.target_dim = dataset['target_train'].shape[-1]

        mover_model  = MLP(self.source_dim, self.target_dim, hidden_sizes=self.solver_cfg['hidden_sizes_mover_mlp']).to(self.device)
        critic_model = MLP(self.target_dim, hidden_sizes=self.solver_cfg['hidden_sizes_critic_mlp']).to(self.device)

        mover_optimizer  = torch.optim.Adam(mover_model.parameters(), lr=self.solver_cfg['mover_lr'], weight_decay=1e-10)
        critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=self.solver_cfg['critic_lr'], weight_decay=1e-10)

        models = {'mover': mover_model, 'critic': critic_model, 'cost': None}
        optimizers = {'mover': mover_optimizer, 'critic': critic_optimizer, 'cost': None}
        n_iters = {key: self.solver_cfg[f'{key}_iters'] for key in ('cost', 'mover', 'critic')}

        if min(n_iters.values()) < 1:
            raise ValueError('cost_iters, mover_iters and critic_iters must be positive.')

        self.model = NeuralGW(
            models=models,
            optimizers=optimizers,
            n_samples=self.config['dataset'].get('batch_size_train', self.config['dataset']['n_train_samples']),
            n_iters=n_iters,
            reg=self.solver_cfg.get('reg_critic'),
            lambda_distortion=self.solver_cfg.get('lambda_distortion', 1e-3),
        )

        self.model.setup(dataset['source_sampler'], dataset['target_sampler'])

        if self.init_P:
            self._initialize_P(dataset)
            print(f'Initialized fixed P for {self.epochs_init_P} training iterations.')
        else:
            self._make_trainable_cost()

    def restore_checkpoint(self, checkpoint):

        if checkpoint['cost'] is not None:
            if self.model.cost_optimizer is None:
                self._make_trainable_cost()

            self.model.cost_model.load_state_dict(checkpoint['cost'])
            self.model.cost_optimizer.load_state_dict(checkpoint['cost_optimizer'])

        else:
            self.model.set_fixed_P(checkpoint['fixed_P'])

        self.model.mover_model.load_state_dict(checkpoint['mover'])
        self.model.critic_model.load_state_dict(checkpoint['critic'])

        self.model.mover_optimizer.load_state_dict(checkpoint['mover_optimizer'])
        self.model.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

    def fit(self, dataset, offline_logger=None):

        self.setup(dataset)
        self.history = []

        for epoch in trange(self.n_epochs, desc=self.name):

            step = epoch + 1

            # The first epochs_init_P iterations use fixed P.
            # Switch BEFORE iteration epochs_init_P + 1.
            if self.init_P and epoch == self.epochs_init_P:
                self._make_trainable_cost()
                print(f'Iteration {step}: switching from fixed P to trainable cost.')

            row = self.model.train()
            row['step'] = step
            self.history.append(row)

            if step % self.report_every == 0 or step == self.n_epochs:

                for item in self.history:
                    for key, value in item.items():
                        if key != 'step' and hasattr(value, 'item'):
                            item[key] = value.item()

                report_dataset = {
                    **dataset,
                    'target_train_pred': self.model.predict(dataset['source_train']),
                    'target_test_pred': self.model.predict(dataset['source_test']),
                }

                if offline_logger is not None:
                    offline_logger(step=step, model=self.model, history=self.history, dataset=report_dataset)

                # Early stopping is enabled only after P initialization.
                if self.early_stopper is not None and (not self.init_P or step > self.epochs_init_P):

                    if offline_logger is None:
                        raise ValueError('Early stopping requires an offline logger.')

                    metrics = offline_logger.logger.metric_history[self.early_stopper.metric_split][-1]
                    should_stop = self.early_stopper.step(step, metrics, self.model)

                    if self.early_stopper.best_step == step:
                        path = offline_logger.ckpt_dir / 'best.pt'
                        torch.save({
                            'step': step,
                            'score': self.early_stopper.best_score,
                            'model': self.early_stopper.best_checkpoint,
                            'history': self.history,
                        }, path)

                    if should_stop:
                        self.stop_reason = self.early_stopper.stop_reason
                        print(f'Early stopping at iteration {step}: {self.stop_reason}')
                        break

        if self.early_stopper is not None and self.early_stopper.best_checkpoint is not None:
            self.restore_checkpoint(self.early_stopper.best_checkpoint)
            print(f"Restored best model from iteration {self.early_stopper.best_step}.")

        return self.model
