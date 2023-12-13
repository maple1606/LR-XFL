from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree, DecisionTreeClassifier
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset, ConcatDataset
import random
from entropy_lens.logic.metrics import test_explanation
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter

from experiments.data.load_datasets import load_cub, load_cub2, add_noise
from experiments.data.data_sampling import cub_iid, cub_noniid_per_class, cub_noniid
from collections import defaultdict
from sympy import simplify_logic


# choose noise type from ['global', 'client', 'none']
# 'global' means adding noise for all clients, 'client' means selecting part of clients to add noise
noise_type = 'non'
# a client randomly generates a number between (0, 1), if the number > 0.8, it takes noise as input
client_noise_threshold = 0.2
client_noise_n = [1] * int(10 - 10 * client_noise_threshold) + [0] * int(10 * client_noise_threshold)
random.shuffle(client_noise_n)

x, y, concept_names = load_cub2()
dataset = TensorDataset(x, y)
train_size = int(len(dataset) * 0.9)
val_size = (len(dataset) - train_size) // 2
test_size = len(dataset) - train_size - val_size
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

if noise_type == 'global':
    # x_noise, y_noise = add_noise('cub', noise_type='label', noise_ratio=0.05)
    # the feature noise is so like data augmentation, the results even get better
    x_noise, y_noise = add_noise('cub', noise_type='feature', noise_ratio=0.8, noise_degree=0.1)
    dataset_noise = TensorDataset(x_noise, y_noise)
    train_data = ConcatDataset([train_data, dataset_noise])
elif noise_type == 'client':
    # noise_ratio is fixed 1 because the new noise dataset should stay the same size as the original dataset
    # noise_degree suggests how much noise will be integrated into the noise dataset (between 0-1)
    x_noise, y_noise = add_noise('cub', noise_type='label', noise_ratio=1, noise_degree=0.8)
    dataset_noise = TensorDataset(x_noise, y_noise)
    train_data_noise, _, _ = random_split(dataset_noise, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size=len(train_data))
val_loader = DataLoader(val_data, batch_size=len(val_data))
test_loader = DataLoader(test_data, batch_size=len(test_data))
n_concepts = next(iter(train_loader))[0].shape[1]
n_classes = 200

# print(concept_names)
# print(n_concepts)
# print(n_classes)

# sample the data into different data silos
sample = 'iid'
# sample = 'non-iid'
per_class = True
num_users = 10
if sample == 'iid':
    # Sample IID user data from CUB
    user_groups_train = cub_iid(train_data, num_users)
else:
    # Sample Non-IID user data from CUB based on the attribute
    # user_groups_train = cub_noniid(train_data, num_users)
    user_groups_train = cub_noniid_per_class(train_data, num_users)


n_seeds = 1
max_epoch = 10
results_list = []
explanations = {i: [] for i in range(n_classes)}

# Constructing the filename
filename = (
    f"../results/"
    f"CUB_"
    f"FL_"
    f"tree_"
    f"ClientNum_{num_users}_"
    f"{sample}_"
    f"RuleGenAccT_non_"
    f"NType_{noise_type}.txt"
    f"NRatio_{round(1-client_noise_threshold, 2)}.txt"
)

with open(filename, 'w') as file:
    file.write('---------------------------\n')


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

    if noise_type == 'client' and client_noise_n[user] > client_noise_threshold:
        train_data_user, val_data_user, test_data_user = random_split(
            Subset(train_data_noise, list(user_groups_train[user])), [train_size_user, val_size_user, test_size_user])

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
            synthetic_example_x = np.zeros((1, len(train_data_user_X[0])))
            # synthetic_example_x = np.mean(train_data_user_X, axis=0).reshape(1, -1)
            synthetic_example_y = np.array([missing_class])

            train_data_user_X = np.vstack([train_data_user_X, synthetic_example_x])
            train_data_user_y = np.hstack([train_data_user_y, synthetic_example_y])

    local_decision_trees[user].fit(train_data_user_X, train_data_user_y)


def tree_rules(tree, feature_names):
    tree_ = tree.tree_
    rules_dict = {i: [] for i in range(200)}  # Assuming 200 classes

    def recurse(node, rule):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]

            # Create left and right rules based on the threshold conditions
            rule_left, rule_right = list(rule), list(rule)
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


def replace_names(mapping_flag, explanation: str, concept_names: List[str]) -> str:
    """
    Replace names of concepts in a formula.

    :param explanation: formula
    :param concept_names: new concept names
    :return: Formula with renamed concepts
    """
    feature_abbreviations = [f'feature{i:010}' for i in range(len(concept_names))]
    mapping = []
    for f_abbr, f_name in zip(feature_abbreviations, concept_names):
        if mapping_flag:
            mapping.append((f_name, f_abbr))
        else:
            mapping.append((f_abbr, f_name))

    for k, v in mapping:
        explanation = explanation.replace(k, v)

    return explanation

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
        print(f"Rules for class {class_label + 1}:")
        print(formatted_rules)
        file.write(f"Rules for class {class_label + 1}:\n")
        file.write(" | ".join(f"({rule})" for rule in class_rules if rule) + "\n")
        global_explanation_class[class_label] = formatted_rules
    #     class_rules = rules_by_class[class_label]
    #     formatted_rules = " | ".join(f"({rule})" for rule in class_rules if rule)
    #     new_rule = replace_names(1, formatted_rules, concept_names)
    #     formatted_rules = str(simplify_logic(new_rule, 'dnf', force=True))
    #     new_rule = replace_names(0, formatted_rules, concept_names)
    #     print(f"Rules for class {class_label + 1}:")
    #     print(new_rule)
    #     file.write(f"Rules for class {class_label + 1}:\n")
    #     file.write(f"{new_rule}\n")
    #     global_explanation_class[class_label] = new_rule




global_explanation_accuracy = 0
global_explanation_fidelity = 0
for target_class in range(n_classes):
    # test the accuracy of the aggregated logic
    global_explanation_class_name = replace_names(1, global_explanation_class[target_class], concept_names)
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



