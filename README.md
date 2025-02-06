## Personal Note
I'm really thankful to Ms. Yanci Zhang for introducing such an interesting take on AI. The concept of FL and how it can link to `wisdom of the crowd' really amazed me. 

A huge thanks to my teammates for accompanying me and putting up with me. I know I can be unbearable sometimes and a bit of an unrealistic perfectionist (lol). After the presentation, I still remember when teacher Thuy said our team had great chemistry and that no one felt left behind. That was the best compliment I've received for this assignment.

A special thanks also to teacher Ha Quang Thuy for guiding us, listening attentively to our presentation, and providing such a detailed review of our report — along with awarding us the highest score in the class! ^^ I know for sure there were still a lot of shortcomings, but the comments teacher Thuy gave us were really motivating.


# LR-XFL: Logical Reasoning-based Explainable Federated Learning

## Overview

This is the open-source code for AAAI-24 paper "LR-XFL: Logical Reasoning-based Explainable Federated Learning". This repository contains LR-XFL model implementations and the experimental codes that produced the results documented in the paper.

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

## Acknowledgement

The local client model is built based on the entropy-lens ([https://github.com/pietrobarbiero/entropy-lens](https://github.com/pietrobarbiero/entropy-lens)). We hereby greatly thank the authors of entropy-lens for their clear code and novel research.



