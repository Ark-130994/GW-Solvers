
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from numbers import Real


class ToyOfflineLogger:
    def __init__(self, save_dir):
        self.plot_dir = Path(save_dir) / 'plots'
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def to_numpy(self, x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    def plot_history(self, history):
        metric_names = list(dict.fromkeys(key for row in history for key, value in row.items() if key != 'step' and isinstance(value, Real)))

        if not metric_names:
            return

        fig, axes = plt.subplots(1, len(metric_names), figsize=(4.5 * len(metric_names), 3), squeeze=False)
        colors = plt.get_cmap('tab10').colors

        for i, (ax, name) in enumerate(zip(axes.flat, metric_names)):
            rows = [row for row in history if name in row and isinstance(row[name], Real)]
            ax.plot([row['step'] for row in rows], [row[name] for row in rows], color=colors[i % len(colors)])
            ax.set_xlabel('Epoch')
            ax.set_ylabel(name)
            ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.plot_dir / 'losses.png', dpi=150)
        plt.close(fig)

    def plot_samples(self, step, dataset):
        source_labels = self.to_numpy(dataset['source_test_labels'])
        target_labels = self.to_numpy(dataset['target_test_labels'])

        n_clusters = int(max(source_labels.max(), target_labels.max())) + 1
        cmap = plt.get_cmap('Spectral', n_clusters)

        samples = [
            ('Source', dataset['source_test'], source_labels),
            ('Target', dataset['target_test'], target_labels),
            ('Prediction', dataset['target_test_pred'], source_labels),
        ]

        fig = plt.figure(figsize=(13, 4))

        for i, (title, values, labels) in enumerate(samples, start=1):
            values = self.to_numpy(values)
            projection = '3d' if values.shape[1] == 3 else None
            ax = fig.add_subplot(1, 3, i, projection=projection)
            ax.scatter(*values.T, c=labels, cmap=cmap, vmin=-0.5, vmax=n_clusters-0.5, s=8, alpha=0.7)
            ax.set_title(title)

        fig.tight_layout()
        fig.savefig(self.plot_dir / f'test_{step:06d}.png', dpi=150)
        plt.close(fig)

    def __call__(self, step, model, history, dataset):
        self.plot_history(history)
        
        if model.__class__.__name__ == 'CycleGW':
            self.plot_cycle_samples(step, model, dataset)
        else:
            self.plot_samples(step, dataset)



    def plot_cycle_samples(self, step, model, dataset):

        def to_numpy(x):
            return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

        source = to_numpy(dataset['source_test'])
        target = to_numpy(dataset['target_test'])

        source_pred = to_numpy(dataset['target_test_pred'])
        target_pred = to_numpy(model.predict_inverse(dataset['target_test']))

        labels_source = to_numpy(dataset['source_test_labels'])
        labels_target = to_numpy(dataset['target_test_labels'])

        n_clusters = int(max(labels_source.max(), labels_target.max())) + 1

        panels = [
            ('Source X', source, labels_source),
            ('Target Y', target, labels_target),
            ('F(X)', source_pred, labels_source),
            ('G(Y)', target_pred, labels_target),
        ]

        fig = plt.figure(figsize=(10, 9))

        for i, (title, values, labels) in enumerate(panels, 1):
            ax = fig.add_subplot(2, 2, i, projection='3d' if values.shape[1] == 3 else None)
            ax.scatter(*values.T, c=labels, cmap=plt.get_cmap('Spectral', n_clusters),
                    vmin=-0.5, vmax=n_clusters - 0.5, s=8, alpha=0.7)
            ax.set_title(title)

        fig.tight_layout()
        fig.savefig(self.plot_dir / f'test_{step:06d}.png', dpi=150)
        plt.close(fig)
