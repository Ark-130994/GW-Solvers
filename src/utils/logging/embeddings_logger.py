
import csv
import math
import matplotlib.pyplot as plt

from pathlib import Path
from numbers import Real
from src.metrics import build_index, compute_metrics


class EmbeddingsOfflineLogger:
    def __init__(self, save_dir, config):
        self.save_dir = Path(save_dir)
        self.plot_dir = self.save_dir / 'plots'
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_cfg    = config.get('metrics', {})
        self.metric_history = {'train': [], 'test': []}
        self.indices        = None

        self.is_cycle_gw = config['solver']['solver_name'] == 'CycleGW'

    def save_csv(self, name, history):
        if not history:
            return

        fields = list(dict.fromkeys(key for row in history for key in row))

        with open(self.save_dir / name, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)

    
    def plot_history(self, history, filename, ncols=None):
        metric_names = list(dict.fromkeys(key for row in history for key, value in row.items() if key != 'step' and isinstance(value, Real)))

        if not metric_names:
            return

        ncols = ncols or len(metric_names)
        nrows = math.ceil(len(metric_names) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3 * nrows), squeeze=False)
        colors = plt.get_cmap('tab20').colors

        for i, name in enumerate(metric_names):
            ax = axes.flat[i]
            rows = [row for row in history if name in row and isinstance(row[name], Real)]
            ax.plot([row['step'] for row in rows], [row[name] for row in rows], color=colors[i % len(colors)])
            ax.set_title(name)
            ax.set_xlabel('Epoch')
            ax.grid(alpha=0.3)

        for ax in list(axes.flat)[len(metric_names):]:
            ax.set_visible(False)

        fig.tight_layout()
        fig.savefig(self.plot_dir / filename, dpi=150)
        plt.close(fig)


    def evaluate(self, step, dataset, split):
        metrics = compute_metrics(
            source=dataset[f'source_{split}'],
            target=dataset[f'target_{split}'],
            prediction=dataset[f'target_{split}_pred'],
            labels=dataset[f'labels_{split}'],
            labels_red=dataset[f'labels_{split}_red'],
            index_full=self.indices['full'],
            index_red=self.indices['red'],
            split=split,
            top_n=self.metrics_cfg.get('top_n', (1, 5, 10)),
            mmd_sigma=self.metrics_cfg.get('mmd_sigma', 10.0),
            sinkhorn_blur=self.metrics_cfg.get('sinkhorn_blur', 0.01),
            distortion_cost=self.metrics_cfg.get('distortion_cost', 'cosine'),
        )

        self.metric_history[split].append({'step': step, **metrics})
        self.save_csv(f'metrics_{split}.csv', self.metric_history[split])
        self.plot_history(self.metric_history[split], f'metrics_{split}.png', ncols=3)

    
    def __call__(self, step, model, history, dataset):
        self.save_csv('losses.csv', history)
        self.plot_history(history, 'losses.png')

        if self.indices is None:
            self.indices = {
                'full': build_index(dataset['target_full']),
                'red': build_index(dataset['target_reduced']),
            }

            if self.is_cycle_gw:
                self.indices['source_full'] = build_index(dataset['source_full'])
                self.indices['source_red']  = build_index(dataset['source_reduced'])

        for split in ('train', 'test'):
            if f'target_{split}_pred' in dataset:
                if self.is_cycle_gw:
                    self.evaluate_cycle(step, model, dataset, split, 'direct')
                    self.evaluate_cycle(step, model, dataset, split, 'inverse')
                else:
                    self.evaluate(step, dataset, split)

                
    def evaluate_cycle(self, step, model, dataset, split, direction):

        if direction == 'direct':
            source     = dataset[f'source_{split}']
            target     = dataset[f'target_{split}']
            prediction = dataset[f'target_{split}_pred']
            labels     = dataset[f'labels_{split}']
            labels_red = dataset[f'labels_{split}_red']
            index_full = self.indices['full']
            index_red  = self.indices['red']

        else:
            source     = dataset[f'target_{split}']
            target     = dataset[f'source_{split}']
            prediction = model.predict_inverse(source)
            labels     = dataset['labels_target_train'] if split == 'train' else dataset['labels_test']
            labels_red = labels
            index_full = self.indices['source_full']
            index_red  = self.indices['source_red']

        metrics = compute_metrics(
            source=source,
            target=target,
            prediction=prediction,
            labels=labels,
            labels_red=labels_red,
            index_full=index_full,
            index_red=index_red,
            split=split,
            top_n=self.metrics_cfg.get('top_n', (1, 5, 10)),
            mmd_sigma=self.metrics_cfg.get('mmd_sigma', 10.0),
            sinkhorn_blur=self.metrics_cfg.get('sinkhorn_blur', 0.01),
            distortion_cost=self.metrics_cfg.get('distortion_cost', 'cosine'),
        )

        name = f'{split}_{direction}'
        self.metric_history.setdefault(name, []).append({'step': step, **metrics})
        self.save_csv(f'metrics_{name}.csv', self.metric_history[name])
        self.plot_history(self.metric_history[name], f'metrics_{name}.png', ncols=3)

