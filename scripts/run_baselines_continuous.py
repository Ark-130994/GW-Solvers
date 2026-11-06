
import argparse
import sys
import yaml
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_base import load_data, initialize_output, ReportCallback
from solvers.continuous.continuous_solver import ContinuousSolver
from src.utils.logging.toy_logger import ToyOfflineLogger
from src.utils.logging.embeddings_logger import EmbeddingsOfflineLogger


def get_samplers_toy(config, source_sampler, target_sampler):

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




def get_samplers_embeddings(config, data):

    source_full, target_full, source_sampler, target_sampler, test_sampler = data

    batch_size_train = config['dataset']['batch_size_train']
    batch_size_test  = config['dataset']['batch_size_test']
    n_test_samples   = config['dataset']['n_test_samples'] * config['dataset']['n_eval']

    source_train, target_train, labels_train, labels_target_train = [], [], [], []

    for i in range(0, 2048, batch_size_train):
        n = min(batch_size_train, 2048 - i)

        source, labels_source = source_sampler.sample(n)
        target, labels_target = target_sampler.sample(n)

        source_train.append(source)
        target_train.append(target)
        labels_train.append(labels_source)
        labels_target_train.append(labels_target)

    source_sampler.reset_sampler()
    target_sampler.reset_sampler()

    source_test, target_test, labels_test = [], [], []

    for i in range(0, n_test_samples, batch_size_test):
        n = min(batch_size_test, n_test_samples - i)

        source, target, labels = test_sampler.sample(n)

        source_test.append(source)
        target_test.append(target)
        labels_test.append(labels)

    source_train        = torch.cat(source_train)
    target_train        = torch.cat(target_train)
    labels_train        = torch.cat(labels_train)
    labels_target_train = torch.cat(labels_target_train)

    source_test = torch.cat(source_test)
    target_test = torch.cat(target_test)
    labels_test = torch.cat(labels_test)

    return {
        'source_sampler': source_sampler,
        'target_sampler': target_sampler,
        'source_train': source_train,
        'target_train': target_train,
        'source_test': source_test,
        'target_test': target_test,
        'labels_train': labels_train,
        'labels_target_train': labels_target_train,
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
        dataset = get_samplers_toy(config, source_sampler, target_sampler)
        logger = ToyOfflineLogger(attempt_dir)

    elif config['experiment_type'] == 'embedding':
        dataset = get_samplers_embeddings(config, data)
        logger = EmbeddingsOfflineLogger(attempt_dir, config)

    else:
        raise ValueError(f"Unknown experiment type: {config['experiment_type']}")

    offline_logger = ReportCallback(logger, attempt_dir, config['solver']['n_epochs'])

    solver = ContinuousSolver(config)
    solver.fit(dataset, offline_logger=offline_logger)


if __name__ == '__main__':
    main()
