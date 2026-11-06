
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_base import load_data, initialize_output, ReportCallback
from scripts.run_baselines_continuous import get_samplers_toy, get_samplers_embeddings
from solvers.neural_innergw.neural_innergw_solver import NeuralInnerGWSolver
from src.utils.logging.toy_logger import ToyOfflineLogger
from src.utils.logging.embeddings_logger import EmbeddingsOfflineLogger


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f'Invalid configuration: {args.config}')

    
    if config['solver_type'] != 'neural_innergw':
        raise ValueError('NeuralGW requires solver_type: neural_innergw.')


    attempt_dir = initialize_output(config, args.config)
    
    sampling_config = {**config, 'solver_type': 'continuous'}
    data = load_data(sampling_config)


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

    solver = NeuralInnerGWSolver(config)
    solver.fit(dataset, offline_logger=offline_logger)


if __name__ == '__main__':
    main()
