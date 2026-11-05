
from tqdm.auto import trange
from solvers.discrete.aligngw import AlignGW
from solvers.discrete.structuredgw import StructuredGW
from solvers.discrete.flowgw import FlowGW


class DiscreteSolver:

    def __init__(self, config):

        self.config       = config
        self.solver_cfg   = config['solver']
        self.name         = self.solver_cfg['solver_name']
        self.n_epochs     = self.solver_cfg['n_epochs']
        self.report_every = self.solver_cfg['report_every']
        self.history      = []

        if self.report_every <= 0:
            raise ValueError('report_every must be positive.')

        if self.name == 'AlignGW':
            self.model = AlignGW(
                cost_name=self.solver_cfg['cost_name'],
                normalize_dists=self.solver_cfg['normalize_dists'],
                loss_fun=self.solver_cfg['loss_fun'],
                eps=self.solver_cfg['eps'],
                ot_warm=self.solver_cfg['ot_warm'],
                approximation=self.solver_cfg.get('approximation', 'argmax'),
                device=config['device'],
            )
        
        elif self.name == 'StructuredGW':
            self.model = StructuredGW(
                m_init=self.solver_cfg.get('m_init'),
                method_m=self.solver_cfg.get('method_m', 'exact'),
                eps=self.solver_cfg['eps'],
                tol=self.solver_cfg['tol'],
                experiment_type=config['experiment_type'],
                device=config['device'],
                inference_type=self.solver_cfg.get('inference_type', 'mlp'),
            )

        elif self.name == 'FlowGW':
            self.model = FlowGW(
                eps=self.solver_cfg['eps'],
                embed_dim=self.solver_cfg['embed_dim'],
                n_freq=self.solver_cfg['n_freq'],
                n_layers=self.solver_cfg['n_layers'],
                cost_fn=self.solver_cfg.get('cost_fn', 'cosine'),
                lr=self.solver_cfg['lr'],
                n_epochs=self.n_epochs,
                seed=config['dataset'].get('seed', 0),
                device=config['device'],
            )

        else:
            raise ValueError(f'Unknown discrete solver: {self.name}')

    def fit(self, dataset, offline_logger=None):

        source_train = dataset['source_train']
        target_train = dataset['target_train']
        source_test  = dataset['source_test']

        self.model.setup(source_train, target_train)
        self.history = []

        
        for epoch in trange(self.n_epochs, desc=self.name):
            row = self.model.solve()
            row = {} if row is None else row
            row['step'] = epoch + 1
            self.history.append(row)

            if (epoch + 1) % self.report_every == 0 or epoch == self.n_epochs - 1:
                if self.name == 'FlowGW':
                    for row in self.history:
                        if not isinstance(row['flow_loss'], float):
                            row['flow_loss'] = float(row['flow_loss'])

                self.model.continuous_approximation()
                report_dataset = {**dataset, 'target_train_pred': self.model.predict(source_train), 'target_test_pred': self.model.predict(source_test)}

                if offline_logger is not None:
                    offline_logger(step=epoch + 1, model=self.model, history=self.history, dataset=report_dataset)

        return self.model
