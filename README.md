# Neural Gromov-Wasserstein solver (NeuralGW)

This repository contains a ```Pytorch``` implementation of the NeuralGW solver from our paper **Uncovering Challenges of Solving the Continuous Gromov-Wasserstein Problem**.

## Requirements

We use the ```faiss```, ```moscot``` and ```gensim``` libraries.

## Experiments

We provide the ```notebooks/GWSolvers_NeuralGW.ipynb``` notebook to reproduce the experiments in the paper for ```glove/twitter``` and ```bone_marrow``` datasets. The datasets will be downloaded automatically or they can be located in a folder called ```datasets```. 

To train on different dimensions of ```glove/twitter``` change the ```SOURCE_DIM``` and ```TARGET_DIM``` parameters. The value ```ALPHA``` is used to change the correlatedness of the source and target samples as indicated in the paper.

The ```src/solvers_continuous/NeuralGW.py``` file contains our implementation of the NeuralGW solver. To modify the hyperparameters and optimizers of the models use the ```src/training.py``` file.


## License

[MIT](https://choosealicense.com/licenses/mit/)