
import argparse
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_base import load_data, initialize_output, ReportCallback
from solvers.continuous.continuous_solver import ContinuousSolver
from src.utils.logging.toy_logger import ToyOfflineLogger
from src.utils.logging.embeddings_logger import EmbeddingsOfflineLogger


def get_samples_toy(config, source_sampler, target_sampler):

    n_train_samples = config['dataset']['n_train_samples']
    n_test_samples  = config['dataset']['n_test_samples']

    source_train = source_sampler.sample(n_train_samples)
    target_train = target_sampler.sample(n_train_samples)

    source_test, source_test_labels = source_sampler.sample_with_labels(n_test_samples)
    target_test, target_test_labels = target_sampler.sample_with_labels(n_test_samples)

    return {
        'source_sampler': source_sampler,
        'target_sampler': target_sampler,
        'source_train': source_train,
        'target_train': target_train,
        'source_test': source_test,
        'target_test': target_test,
        'source_test_labels': source_test_labels,
        'target_test_labels': target_test_labels,
    }


def get_samples_embeddings(config, data):

    source_full, target_full, source_sampler, target_sampler, test_sampler = data

    source_train, labels_train        = source_sampler.loader.dataset.tensors
    target_train, _                   = target_sampler.loader.dataset.tensors
    source_test, target_test, labels_test = test_sampler.loader.dataset.tensors

    return {
        'source_sampler': source_sampler,
        'target_sampler': target_sampler,
        'source_train': source_train,
        'target_train': target_train,
        'source_test': source_test,
        'target_test': target_test,
        'labels_train': labels_train,
        'labels_train_red': labels_train,
        'labels_test': labels_test,
        'labels_test_red': labels_test,
        'source_full': source_full,
        'target_full': target_full,
        'source_reduced': source_full,
        'target_reduced': target_full,
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
        dataset = get_samples_toy(config, source_sampler, target_sampler)
        logger = ToyOfflineLogger(attempt_dir)

    elif config['experiment_type'] == 'embedding':
        dataset = get_samples_embeddings(config, data)
        logger = EmbeddingsOfflineLogger(attempt_dir, config)

    else:
        raise ValueError(f"Unknown experiment type: {config['experiment_type']}")

    offline_logger = ReportCallback(logger, attempt_dir, config['solver']['n_epochs'])

    solver = ContinuousSolver(config)
    solver.fit(dataset, offline_logger=offline_logger)


if __name__ == '__main__':
    main()
