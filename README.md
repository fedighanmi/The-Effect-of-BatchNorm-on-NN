# The-Effect-of-BatchNorm-on-NN

This repository studies the effect of **Batch Normalization** on neural network training and performance using an **identical neural network architecture** between experiments. The goal is to isolate and analyze the impact of BatchNorm by keeping all other components of the training unchanged.

## Objective

The main objective of this project is to understand how Batch Normalization affects:
- Training stability
- Convergence behavior
- Stress testing behavior
  

## Datasets

The experiments are conducted on two widely used image classification datasets:

- **CIFAR-10** – natural image classification with 10 object classes
- **MNIST** – handwritten digit classification

Each dataset is analyzed independently.

## Repository Structure

- `notebooks`  
  folder containing the jupyter notebooks used to train, test and analyse on the **CIFAR-10** and **MNIST** dataset.

- `notebooks/utils.py`  
  Pool of functions for training loops, evaluation, plotting, and visualization.

- `requirements.txt`  
  List of required Python dependencies.

## Results Summary
- `POSTER_1_RESULTS.pdf`
- `POSTER_2_RESULTS.pdf`

Contains a summary of findings and presentation of results.

