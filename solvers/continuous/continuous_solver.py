
from tqdm.auto import trange


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

        # Continuous solver initialization will be added here.

    def fit(self, dataset, offline_logger=None):

        source_sampler = dataset['source_sampler']
        target_sampler = dataset['target_sampler']

        if self.model is None:
            raise ValueError(f'Unknown continuous solver: {self.name}')

        self.model.setup(source_sampler, target_sampler)
        self.history = []

        for epoch in trange(self.n_epochs, desc=self.name):
            row = self.model.solve()
            row = {} if row is None else row
            row['step'] = epoch + 1
            self.history.append(row)

            if (epoch + 1) % self.report_every == 0 or epoch == self.n_epochs - 1:
                for item in self.history:
                    for key, value in item.items():
                        if key != 'step' and hasattr(value, 'item'):
                            item[key] = value.item()

                report_dataset = {**dataset, 'target_train_pred': self.model.predict(dataset['source_train'][:2048]), 'target_test_pred': self.model.predict(dataset['source_test'])}

                if offline_logger is not None:
                    offline_logger(step=epoch + 1, model=self.model, history=self.history, dataset=report_dataset)

        return self.model
