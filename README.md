# GW-Solvers

Implementations of Gromov–Wasserstein (GW) solvers with a shared experiment pipeline for comparing alignment methods.

## Discrete GW experiments

The discrete solver pipeline currently includes **AlignGW**, **StructuredGW**, and **FlowGW**. Experiments support two data settings:

- **Toy:** Synthetic distributions in 2D and 3D.
- **Embeddings:** Alignment between embedding spaces (e.g., Twitter-GloVe, 100 → 50 dimensions).

### Run an experiment

From the repository root, using a Python environment with the project dependencies installed:

```bash
python scripts/run_baselines_discrete.py --config=configs/toy/flowgw.yaml
```

For an embedding experiment:

```bash
python scripts/run_baselines_discrete.py --config=configs/embeddings/flowgw.yaml
```

Choose a configuration under `configs/` and edit its `dataset` and `solver` sections to select the data, method, training iterations, and reporting frequency. For GPU experiments, set `device: cuda`.

### Results

Each run creates a numbered directory under:

```text
results/<experiment_type>/discrete/<solver_name>/...
```

The pipeline saves a copy of the configuration, model checkpoints (`ckpts/`), plots (`plots/`), and evaluation metrics. Embedding experiments report alignment and distributional metrics on training and test samples.

> **Note:** “Discrete” here refers to the GW solver pipeline; the toy and embedding inputs are real-valued vectors.