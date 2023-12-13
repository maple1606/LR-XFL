from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree, DecisionTreeClassifier
from typing import List
import numpy as np
import torch
from collections import defaultdict
import os
import pandas as pd
import numpy as np
import time
import copy
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset, ConcatDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything, utilities
from torch import stack, squeeze
from sklearn.metrics import f1_score, accuracy_score
import random
from experiments.data.load_datasets import load_mnist, add_noise
from experiments.data.data_sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, mnist_noniid_per_class
from entropy_lens.logic.metrics import test_explanation, complexity
import logging
import warnings
from sympy import simplify_logic

warnings.filterwarnings('ignore')

# logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR) # %% md
# logging.getLogger('lightning').setLevel(0)
utilities.distributed.log.setLevel(logging.ERROR)

## Import MIMIC-II dataset

# choose noise type from ['global', 'client', 'none']
# 'global' means adding noise for all clients, 'client' means selecting part of clients to add noise
noise_type = 'non'
# a client randomly generates a number between (0, 1), if the number > 0.8, it takes noise as input
client_noise_threshold = 0.6
client_noise_n = [1] * int(10 - 10 * client_noise_threshold) + [0] * int(10 * client_noise_threshold)
random.shuffle(client_noise_n)

x, y, concept_names = load_mnist()

dataset = TensorDataset(x, y)
train_size = int(len(dataset) * 0.9)
val_size = (len(dataset) - train_size) // 2
test_size = len(dataset) - train_size - val_size
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size=len(train_data))
val_loader = DataLoader(val_data, batch_size=len(val_data))
test_loader = DataLoader(test_data, batch_size=len(test_data))
n_concepts = next(iter(train_loader))[0].shape[1]
n_classes = 2
print(concept_names)
print(n_concepts)
print(n_classes)


# sample the data into different data silos
# sample = 'iid'
sample = 'non-iid'
unequal = True
per_class = True
num_users = 10
if sample == 'iid':
    # Sample IID user data from Mnist
    user_groups_train = mnist_iid(train_data, num_users)
else:
    # Sample Non-IID user data from Mnist
    if unequal:
        # Chose uneuqal splits for every user
        if per_class:
            user_groups_train = mnist_noniid_per_class(train_data, num_users)
        else:
            user_groups_train = mnist_noniid_unequal(train_data, num_users)
    else:
        # Chose euqal splits for every user
        user_groups_train = mnist_noniid(train_data, num_users)

# %% md

## 5-fold cross-validation with explainer network

base_dir = f'./results/MNIST/explainer'
os.makedirs(base_dir, exist_ok=True)

n_seeds = 1
max_epoch = 10
results_list = []
explanations = {i: [] for i in range(n_classes)}

# Constructing the filename
filename = (
    f"../results/"
    f"test_"
    f"MNIST_"
    f"FL_"
    f"tree_"
    f"ClientNum_{num_users}_"
    f"{sample}_"
    f"RuleGenAccT_non_"
    f"NType_{noise_type}.txt"
)

train_loader = DataLoader(train_data, batch_size=len(train_data))
val_loader = DataLoader(val_data, batch_size=len(val_data))
test_loader = DataLoader(test_data, batch_size=len(test_data))
n_concepts = next(iter(train_loader))[0].shape[1]
n_classes = 2

with open(filename, 'w') as file:
    file.write('---------------------------\n')

# class_weight = {i: 1 for i in range(n_classes)}
train_loader_users, val_loader_users, test_loader_users = [], [], []

# Create local decision trees for each user
local_decision_trees = [DecisionTreeClassifier() for _ in range(num_users)]

# Evaluate rules on validation set and select the best rule
best_accuracy = 0
best_user = -1


for user in range(num_users):
    # generate a number between (0, 1), if the number > noise threshold, the client receives noisy data only
    train_size_user = int(len(user_groups_train[user]) * 0.9)
    val_size_user = (len(user_groups_train[user]) - train_size_user) // 2
    test_size_user = len(user_groups_train[user]) - train_size_user - val_size_user
    train_data_user, val_data_user, test_data_user = random_split(
        Subset(train_data, list(user_groups_train[user])), [train_size_user, val_size_user, test_size_user])
    # Extract inputs and targets as NumPy arrays and concatenate all batches
    train_data_user_X, train_data_user_y = [], []
    for inputs, targets in train_data_user:
        train_data_user_X.append(inputs.numpy())
        train_data_user_y.append(int(torch.argmax(targets, dim=0)))

    missing_classes = set(range(n_classes)) - set(np.unique(train_data_user_y))

    if len(missing_classes) != 0:
        # Create a synthetic example for the missing class
        # Modify this to match the structure of your actual data

        for missing_class in missing_classes:
            # Create a synthetic example for the missing class
            # Modify this to match the structure of your actual data
            synthetic_example_x = np.mean(train_data_user_X, axis=0).reshape(1, -1)
            synthetic_example_y = np.array([missing_class])

            train_data_user_X = np.vstack([train_data_user_X, synthetic_example_x])
            train_data_user_y = np.hstack([train_data_user_y, synthetic_example_y])

    local_decision_trees[user].fit(train_data_user_X, train_data_user_y)


def tree_rules(tree, feature_names):
    tree_ = tree.tree_
    rules_dict = {i: [] for i in range(n_classes)}

    def recurse(node, rule):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]

            # Create left and right rules based on the threshold conditions
            rule_left, rule_right = list(rule), list(rule)
            # if threshold < 0.7:
            #     rule_left += [f"~{name}"]
            # if threshold >= 0.3:
            #     rule_right += [f"{name}"]
            rule_left += [f"~{name}"]
            rule_right += [f"{name}"]

            # Recurse on both children
            recurse(tree_.children_left[node], rule_left)
            recurse(tree_.children_right[node], rule_right)
        else:
            class_label = np.argmax(tree_.value[node])
            rules_dict[class_label].append(" & ".join(rule))

    recurse(0, [])
    return rules_dict


def global_decision_tree_select(X, y):
    predictions_per_client = defaultdict(list)

    user_score = [0 for _ in range(num_users)]
    # Collect predictions from each client's decision tree
    for user, tree in enumerate(local_decision_trees):
        predictions = tree.predict(X)
        for i, prediction in enumerate(predictions):
            if prediction == y[i]:
                user_score[user] += 1
            predictions_per_client[i].append((user, prediction))
    winning_user = np.argmax(user_score)
    return winning_user


# Evaluate the global decision tree
X_val, y_val = [], []
for inputs, targets in val_data:
    X_val.append(inputs.numpy())
    y_val.append(int(torch.argmax(targets, dim=0)))

X_test, y_test = [], []
for inputs, targets in test_data:
    X_test.append(inputs.numpy())
    y_test.append(int(torch.argmax(targets, dim=0)))

winning_users = global_decision_tree_select(X_val, y_val)
global_predictions = local_decision_trees[winning_users].predict(X_test)
global_score = accuracy_score(y_test, global_predictions)


global_explanation_class = {}  # Dictionary to store rules for each class

with open(filename, 'a') as file:
    rules_by_class = tree_rules(local_decision_trees[winning_users], concept_names)
    for class_label in range(n_classes):  # Assuming there are 200 classes, numbered from 0 to 199
        class_rules = rules_by_class[class_label]
        formatted_rules = " | ".join(f"({rule})" for rule in class_rules if rule)
        formatted_rules = str(simplify_logic(formatted_rules, 'dnf', force=True))
        print(f"Rules for class {class_label + 1}:")
        print(formatted_rules)
        file.write(f"Rules for class {class_label + 1}:\n")
        file.write(f"{formatted_rules}\n")
        global_explanation_class[class_label] = formatted_rules



def replace_names(explanation: str, concept_names: List[str]) -> str:
    """
    Replace names of concepts in a formula.

    :param explanation: formula
    :param concept_names: new concept names
    :return: Formula with renamed concepts
    """
    feature_abbreviations = [f'feature{i:010}' for i in range(len(concept_names))]
    mapping = []
    for f_abbr, f_name in zip(feature_abbreviations, concept_names):
        mapping.append((f_name, f_abbr))

    for k, v in mapping:
        explanation = explanation.replace(k, v)

    return explanation

global_explanation_accuracy = 0
global_explanation_fidelity = 0
for target_class in range(n_classes):
    # test the accuracy of the aggregated logic
    global_explanation_class_name = replace_names(global_explanation_class[target_class], concept_names)
    global_explanation_accuracy_class, global_y_formula_class = test_explanation(global_explanation_class_name,
                                                                                 test_data.dataset.tensors[0][
                                                                                     test_data.indices],
                                                                                 test_data.dataset.tensors[1][
                                                                                     test_data.indices],
                                                                                 target_class)
    if global_y_formula_class is not None:
        global_explanation_fidelity_class = accuracy_score(global_predictions == target_class,
                                                           global_y_formula_class)
        global_explanation_fidelity += global_explanation_fidelity_class
    global_explanation_accuracy += global_explanation_accuracy_class

print('---------------------------')
print("Winning users:", winning_users)
print("Global Decision Tree acc:", global_score)
print('Global rule mean accuracy', global_explanation_accuracy / n_classes)
print('Global rule mean fidelity', global_explanation_fidelity / n_classes)
print('---------------------------')

with open(filename, 'a') as file:
    file.write('---------------------------\n')
    file.write(f"Winning users: {winning_users}\n")
    file.write(f"Global Decision Tree acc: {global_score}\n")
    file.write(f'Global rule mean accuracy: {global_explanation_accuracy / n_classes}\n')
    file.write(f'Global rule mean fidelity: {global_explanation_fidelity / n_classes}\n')
    file.write('---------------------------\n')



