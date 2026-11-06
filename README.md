
# GWBench: A benchmark for continuous Gromov-Wasserstein Solvers

Implementations of Gromov–Wasserstein (GW) solvers with a shared experiment pipeline for comparing alignment methods. Experiments support **toy** distributions in 2D and 3D and **embedding** alignment (e.g., Twitter-GloVe, 100 → 50 dimensions).

## Discrete GW experiments

The discrete pipeline includes **AlignGW**, **StructuredGW**, and **FlowGW**. Although these solvers fit GW on finite samples, the pipeline also supports **out-of-sample inference**, allowing predictions for unseen data.

Run a toy experiment from the repository root, using a Python environment with the project dependencies installed:

```bash
python scripts/run_baselines_discrete.py --config=configs/toy/flowgw.yaml
```

For an embedding experiment:

```bash
python scripts/run_baselines_discrete.py --config=configs/embeddings/flowgw.yaml
```

## Continuous GW experiments

The continuous baseline pipeline includes **RegGW**, **CycleGW**, and **FlowGW** (minibatch). These methods train on sampled minibatches rather than fitting a GW coupling to a fixed dataset.

Run a toy experiment:

```bash
python scripts/run_baselines_continuous.py --config=configs/continuous/flowgw_toy.yaml
```

For an embedding experiment:

```bash
python scripts/run_baselines_continuous.py --config=configs/continuous/flowgw_embeddings.yaml
```

## Neural inner-GW experiments

The proposed methods are **NeuralGW** and **NeuralGW_P**. The latter starts with a fixed, Gaussian-initialized cost matrix P before switching to a trainable cost model. Embedding experiments also support configurable early stopping.

Run a toy experiment:

```bash
python scripts/run_neural_innergw.py --config=configs/continuous/neuralgw_toy.yaml
```

For an embedding experiment:

```bash
python scripts/run_neural_innergw.py --config=configs/continuous/neuralgw_embeddings.yaml
```

## Configuration and results

Choose a configuration under `configs/` and edit its `dataset` and `solver` sections to select the data, method, number of training iterations, and reporting frequency. For GPU experiments, set `device: cuda`.

Each run creates a numbered directory under:

```text
results/<experiment_type>/<solver_type>/<solver_name>/...
```

The pipeline saves a copy of the configuration, model checkpoints (`ckpts/`), plots (`plots/`), and evaluation metrics. Embedding experiments report alignment and distributional metrics on training and test samples.

> **Note:** “Discrete” refers to the GW solver pipeline; the toy and embedding inputs are real-valued vectors.
