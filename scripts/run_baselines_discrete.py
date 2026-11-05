
import argparse
import yaml
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_base import load_data, initialize_output, ReportCallback
from solvers.discrete.discrete_solver import DiscreteSolver
from src.utils.logging.toy_logger import ToyOfflineLogger
from src.utils.logging.embeddings_logger import EmbeddingsOfflineLogger


def get_tensors_toy(config, source_sampler, target_sampler):
    n_train_samples = config['dataset']['n_train_samples']
    n_test_samples  = config['dataset']['n_test_samples']

    source_train = source_sampler.sample(n_train_samples)
    target_train = target_sampler.sample(n_train_samples)

    source_test, source_test_labels = source_sampler.sample_with_labels(n_test_samples)
    target_test, target_test_labels = target_sampler.sample_with_labels(n_test_samples)

    return {
        'source_train': source_train,
        'target_train': target_train,
        'source_test': source_test,
        'target_test': target_test,
        'source_test_labels': source_test_labels,
        'target_test_labels': target_test_labels,
    }


def get_tensors_embeddings(config, data):
    source_full, target_full, source_reduced, target_reduced, source_sampler, target_sampler, test_sampler = data

    n_test_samples = config['dataset']['n_test_samples']
    n_eval         = config['dataset']['n_eval']

    source_train, labels_train, labels_train_red = source_sampler.sample(source_sampler.loader.batch_size)
    target_train = target_sampler.sample(target_sampler.loader.batch_size)[0]

    test_batches    = [test_sampler.sample(n_test_samples) for _ in range(n_eval)]
    source_test     = torch.cat([batch[0] for batch in test_batches], dim=0)
    target_test     = torch.cat([batch[1] for batch in test_batches], dim=0)
    labels_test     = torch.cat([batch[2] for batch in test_batches], dim=0)
    labels_test_red = torch.cat([batch[3] for batch in test_batches], dim=0)

    return {
        'source_train': source_train,
        'target_train': target_train,
        'source_test': source_test,
        'target_test': target_test,
        'labels_train': labels_train,
        'labels_train_red': labels_train_red,
        'labels_test': labels_test,
        'labels_test_red': labels_test_red,
        'source_full': source_full,
        'target_full': target_full,
        'source_reduced': source_reduced,
        'target_reduced': target_reduced,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    attempt_dir = initialize_output(config, args.config)
    data = load_data(config)

    if config['experiment_type'] == 'toy':
        source_sampler, target_sampler = data
        dataset = get_tensors_toy(config, source_sampler, target_sampler)
        logger = ToyOfflineLogger(attempt_dir)

    elif config['experiment_type'] == 'embedding':
        dataset = get_tensors_embeddings(config, data)
        logger = EmbeddingsOfflineLogger(attempt_dir, config)

    else:
        raise ValueError(f"Unknown experiment type: {config['experiment_type']}")

    offline_logger = ReportCallback(logger, attempt_dir, config['solver']['n_epochs'])

    solver = DiscreteSolver(config)
    model  = solver.fit(dataset, offline_logger=offline_logger)


if __name__ == '__main__':
    main()
