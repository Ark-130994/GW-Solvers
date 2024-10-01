# Gromov-Wasserstein Solvers

This repository contains a the implementations for our paper **Uncovering Challenges of Solving the Continuous Gromov-Wasserstein Problem**.

## Requirements

For the solver implementations we used Pytorch, Python Optimal Transport (POT), Optax and Optimal Transport Tools (OTT) packages. Additionally we request the use of ```faiss```, ```moscot``` and ```gensim``` libraries.

## Experiments

We provide the osurce code for the solvers, we split them into two main categories, continuous and discrete solvers, as continuous we consider the ones that allow batch training, in this category we consider NeuralGW, FlowGW (bvatched) and RegGW (batched). On the side of discrete solvers we have AlignGW, StructuredGW, FlowGW and RegGW.

To run the experiments we included two notebooks for each category of solvers: ```GWSolvers_continuous.ipynb``` and ```GWSolvers_discrete.ipynb```, they include their corresponding set of parameters, you can find the specific parameters we used in the paper. Both GloVe and bone marrow datasets can be used.

To train on different dimensions of ```glove/twitter``` change the ```SOURCE_DIM``` and ```TARGET_DIM``` parameters. The value ```ALPHA``` is used to change the correlatedness of the source and target samples as indicated in the paper.

## License

[MIT](https://choosealicense.com/licenses/mit/)
