# LR-XFL-Logical-Reasoning-based-Explainable-Federated-Learning
Open-source code for AAAI-24 paper "LR-XFL: Logical Reasoning-based Explainable Federated Learning"

# README for LR-XFL

## Overview
This is the code for AAAI-24 Paper No. 7418. This repository contains LR-XFL model implementations and the experimental codes that produced the results documented in the paper.

## Directory Structure

- `entropy_lens/`: Core modules of entropy-based network.
  - `models/`: Model definitions and implementations.
  - `logic/`: Logic-related modules
  - `nn/`: Neural network modules
  
- `experiments/`: Scripts and data related to different experimental setups.
  - `result_plot.py`: Script to plot results.
  - `data/`: datasets used in experiments.
  - `mnist.py`: Scripts to run LR-XFL, and the baseline FedAvg-Logic on MNIST(Even/Odd) dataset.
  - `cub.py`: Scripts to run LR-XFL, and the baseline FedAvg-Logic on CUB dataset.
  - `vdem.py`: Scripts to run LR-XFL, and the baseline FedAvg-Logic on V-Dem dataset.
  - `mimic.py`: Scripts to run LR-XFL, and the baseline FedAvg-Logic on MIMIC-II dataset.
  - `mnist_tree.py`: Scripts to run the baseline distributed decision tree (DDT) on MNIST(Even/Odd) dataset.
  - `cub_tree.py`: Scripts to run the baseline distributed decision tree (DDT) on CUB dataset.
  - `vdem_tree.py`: Scripts to run the baseline distributed decision tree (DDT) on V-Dem dataset.
  - `mimic_tree.py`: Scripts to run the baseline distributed decision tree (DDT) on MIMIC-II dataset.
  
- `results/`: Directory to store output and results from the project.

