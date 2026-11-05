
import shutil
import torch
from pathlib import Path
from src.data import get_samplers_toy, load_vectors, get_samplers_embeddings_discrete, get_samplers_embeddings_continuous

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_data(config):
    solver_type     = config['solver_type']
    experiment_type = config['experiment_type']

    if solver_type not in ['continuous', 'discrete']:
        raise ValueError(f'Unknown solver type: {solver_type}')

    if experiment_type == 'toy':
        return get_samplers_toy(config)

    elif experiment_type == 'embedding':
        source_vectors, target_vectors = load_vectors(config)

        if solver_type == 'discrete':
            return get_samplers_embeddings_discrete(config, source_vectors, target_vectors)
        elif solver_type == 'continuous':
            return get_samplers_embeddings_continuous(config, source_vectors, target_vectors, paired=config['dataset']['paired'])

    raise ValueError(f'Unknown experiment type: {experiment_type}')


def get_attempt_dir(config, results_root='results'):
    results_root = Path(results_root).expanduser()
    if not results_root.is_absolute():
        results_root = PROJECT_ROOT / results_root

    run_root = results_root / config['experiment_type'] / config['solver_type'] / config['solver']['solver_name']

    if config['experiment_type'] == 'embedding':
        dataset_cfg = config['dataset']
        experiment_name = f"{dataset_cfg['dataset_name']}_{dataset_cfg['emb_type_source']}_{dataset_cfg['source_dim']}_{dataset_cfg['target_dim']}"
        setting_name    = f"alpha_{dataset_cfg['alpha']}_seed_{dataset_cfg['seed']}"
        run_root = run_root / experiment_name / setting_name

    run_root.mkdir(parents=True, exist_ok=True)

    for idx in range(10000):
        attempt_dir = run_root / f'{idx:03d}'
        try:
            attempt_dir.mkdir()
        except FileExistsError:
            continue

        (attempt_dir / 'plots').mkdir()
        (attempt_dir / 'ckpts').mkdir()
        return attempt_dir

    raise RuntimeError(f'No free attempt directory in {run_root}')


def initialize_output(config, config_path, results_root='results'):
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    attempt_dir = get_attempt_dir(config, results_root)
    shutil.copyfile(config_path, attempt_dir / 'config.yaml')
    print(f'Results directory: {attempt_dir}', flush=True)
    return attempt_dir


def save_checkpoint(path, model, history, step):
    checkpoint = {'step': step, 'history': history, **model.get_checkpoint()}
    torch.save(checkpoint, path)


class ReportCallback:
    def __init__(self, logger, attempt_dir, n_epochs):
        self.logger   = logger
        self.ckpt_dir = Path(attempt_dir) / 'ckpts'
        self.n_epochs = n_epochs

    def __call__(self, step, model, history, dataset):
        checkpoint_path = self.ckpt_dir / f'epoch_{step:06d}.pt'
        save_checkpoint(checkpoint_path, model, history, step)

        if step == self.n_epochs:
            shutil.copyfile(checkpoint_path, self.ckpt_dir / 'last.pt')

        if self.logger is not None:
            self.logger(step=step, model=model, history=history, dataset=dataset)
