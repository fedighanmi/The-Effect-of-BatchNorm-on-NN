# The-Effect-of-BatchNorm-on-NN

This repository studies the effect of **Batch Normalization** on neural network training and performance using a **fixed neural network architecture**.  
The goal is to isolate and analyze the impact of BatchNorm by keeping all other components of the training pipeline unchanged.

## Objective

The main objective of this project is to understand how Batch Normalization affects:
- Training stability
- Convergence behavior
- Final classification accuracy
  
This is specifically for a relatively small number of epochs.

## Datasets

The experiments are conducted on two widely used image classification datasets:

- **CIFAR-10** – natural image classification with 10 object classes
- **MNIST** – handwritten digit classification

Each dataset is analyzed independently.

## Repository Structure


- `01_Poster_CIFAR10.ipynb`  
  Notebook used to train, test and analyse on the **CIFAR-10** dataset.
  
- `01_Poster_MNIST.ipynb`  
  Notebook used to also train, test and analyse on the **MNIST** dataset.

- `utils.py`  
  Pool of functions for training loops, evaluation, plotting, and visualization.

- `requirements.txt`  
  List of required Python dependencies.


