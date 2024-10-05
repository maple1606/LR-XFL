import random
import sys
sys.path.append('../')

import os
import pandas as pd
import numpy as np
import time
import torch
import copy
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset, ConcatDataset
import random
from torch import stack, squeeze
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything, utilities
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from entropy_lens.models.explainer import Explainer
from entropy_lens.logic.metrics import formula_consistency
from experiments.data.load_datasets import load_vDem, add_noise
from experiments.data.data_sampling import vdem_iid, vdem_noniid_per_class, vdem_noniid
from experiments.local_training import local_train
from experiments.utils import average_weights, average_weights_class, weighted_weights, max_weights, max_weights_class
from experiments.global_logic_aggregate import _global_aggregate_explanations, client_selection_class
from entropy_lens.logic.utils import replace_names
from entropy_lens.logic.metrics import test_explanation, complexity
import logging
import warnings

warnings.filterwarnings('ignore')

# logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR) # %% md
# logging.getLogger('lightning').setLevel(0)
utilities.distributed.log.setLevel(logging.ERROR)

## Import CUB dataset

# %%

num_users = 10

# choose noise type from ['global', 'client', 'none']
# 'global' means adding noise for all clients, 'client' means selecting part of clients to add noise
noise_type = 'non'
# a client randomly generates a number between (0, 1), if the number > 0.8, it takes noise as input
client_noise_threshold = 0.8
client_noise_n = [1] * int(num_users - num_users * client_noise_threshold) + [0] * int(num_users * client_noise_threshold)
random.shuffle(client_noise_n)

x, c, y, concept_names = load_vDem()
dataset = TensorDataset(c, y)
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
n_classes = 2
print(concept_names)
print(n_concepts)
print(n_classes)

# sample the data into different data silos
sample = 'iid'
# sample = 'non-iid'
per_class = True
if sample == 'iid':
    # Sample IID user data from CUB
    user_groups_train = vdem_iid(train_data, num_users)
else:
    # Sample Non-IID user data from CUB based on the attribute
    # user_groups_train = vdem_noniid_per_class(train_data, num_users)
    user_groups_train = vdem_noniid(train_data, num_users)
print([len(user_groups_train[i]) for i in range(len(user_groups_train))])


## 5-fold cross-validation with explainer network

base_dir = f'./results/vdem/explainer'
os.makedirs(base_dir, exist_ok=True)

n_seeds = 1
max_epoch = 20
results_list = []
explanations = {i: [] for i in range(n_classes)}

# Constructing the filename
filename = (
    f"results/"
    
    f"VDem_"
    f"test_test_test_"
    f"FL_"
    f"ClientNum_{num_users}_"
    f"{sample}_"
    f"RuleGenAccT_0.7_"
    f"NType_{noise_type}_"
    f"NRatio_{round(1-client_noise_threshold, 2)}_"
    f"NDegree_0.8_"
    f"Epochs_{max_epoch}_"
    f"TopkExp_3_"
    f"Connector_auto_"
    f"BeamS_non_"
    f"Aggr_weighted.txt"
)

with open(filename, 'w') as file:
    file.write('---------------------------\n')

for seed in range(n_seeds):
    seed_everything(seed)
    print(f'Seed [{seed + 1}/{n_seeds}]')
    # record global model parameters and global model performance for each epoch
    global_model_results_list = []
    global_model_parameters_list = []
    # train_data_users, val_data_users, test_data_users = [], [], []
    train_loader_users, val_loader_users, test_loader_users = [], [], []
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
        train_loader_user = DataLoader(train_data_user, batch_size=len(train_data_user))
        val_loader_user = DataLoader(val_data_user, batch_size=len(val_data_user))
        test_loader_user = DataLoader(test_data_user, batch_size=len(test_data_user))
        train_loader_users.append(train_loader_user)
        val_loader_users.append(val_loader_user)
        test_loader_users.append(test_loader_user)

    global_model = Explainer(n_concepts=n_concepts, n_classes=n_classes, l1=1e-5, lr=0.01,
                        explainer_hidden=[20])
    global_trainer = Trainer(max_epochs=0, gpus=1, auto_lr_find=True, deterministic=True,
                             check_val_every_n_epoch=1, default_root_dir=base_dir,
                             weights_save_path=base_dir)
    global_weights = global_model.state_dict()
    local_models = {}
    local_weights = {}
    local_concept_mask, local_results, local_explanation_f = {}, {}, {}
    for epoch in range(max_epoch):
        # train local model first
        global_connector_class = [[] for _ in range(n_classes)]
        users_for_train = [i for i in range(num_users)]
        for user_id in users_for_train:
            local_weights[user_id], local_concept_mask[user_id], local_results[user_id], local_explanation_f[user_id] \
                = local_train(user_id, epochs=20, train_loader=train_loader_users[user_id],
                              val_loader=val_loader_users[user_id], test_loader=test_loader_users[user_id],
                              n_classes=n_classes, n_concepts=n_concepts, concept_names=concept_names, base_dir=base_dir,
                              results_list=results_list, explanations=explanations, model=copy.deepcopy(global_model),
                              topk_explanations=50, verbose=False, logic_generation_threshold=0.7)
            if local_explanation_f[user_id] is not None:
                for f in local_explanation_f[user_id]:
                    if f['explanation_connector'] is not None:
                        global_connector_class[f['target_class']].append(f['explanation_connector'])
        test_dataset = TensorDataset(test_data.dataset.tensors[0][test_data.indices])
        test_loader = DataLoader(test_dataset, batch_size=32) 

        global_y_test_out = global_trainer.predict(global_model, dataloaders=test_loader)
        max_size = max([t.size(0) for t in global_y_test_out])
        padded_tensors = [F.pad(t, (0, 0, 0, max_size - t.size(0))) for t in global_y_test_out]
        global_y_test_out = squeeze(stack(padded_tensors, dim=0), dim=1)

        """user_to_engage_class refers to the the clients holding the top accuracy;
        local_explanations_accuracy_class includes rule -> (rule, highest acc, clients holding the top accuracy);
        local_explanations_support_class includes rule -> (numbers of supporting clients, total acc, supporting client list)"""
        user_to_engage_class, local_explanations_accuracy_class, local_explanations_support_class = \
            client_selection_class(n_classes, num_users, local_explanation_f)

        # select users from above to aggregate global logic
        users_to_aggregate = set([])
        users_aggregation_weight = {u: 0 for u in range(num_users)}
        global_explanation_accuracy = 0
        global_explanation_fidelity = 0
        for target_class in range(n_classes):
            if len(global_connector_class[target_class]) != 0:
                counts = Counter(global_connector_class[target_class])
                global_connector = max(counts, key=counts.get)
            else:
                global_connector = 'AND'
            # global_connector = 'OR'
            '''
            local_explanations_accuracy, local_explanations_support_class, topk_explanations, target_class, x, y,
            concept_names, user engagement scale: large for engaging all users with aggregated logic,
            small for only engaging the top accurate users for a given logic
            '''
            # topk_explanations, target_class, x, y, concept_names,
            global_explanation_class, global_accuracy_class, user_to_engage_class[target_class] = _global_aggregate_explanations(
                local_explanations_accuracy_class[target_class],
                local_explanations_support_class[target_class],
                topk_explanations=3, target_class=target_class,
                x=val_data.dataset.tensors[0][val_data.indices],
                y=val_data.dataset.tensors[1][val_data.indices],
                concept_names=concept_names, user_engagement_scale='large', connector=global_connector, beam_width=1)

            # test the accuracy of the aggregated logic
            global_explanation_accuracy_class, global_y_formula_class = test_explanation(global_explanation_class,
                                                                    test_data.dataset.tensors[0][test_data.indices],
                                                                    test_data.dataset.tensors[1][test_data.indices],
                                                                    target_class)
            if global_y_formula_class is not None:
                # Get the minimum length of the two arrays
                min_len = min(len(global_y_test_out), len(global_y_formula_class))

                # Trim both arrays to the same length
                global_y_test_out_trimmed = global_y_test_out[:min_len].argmax(dim=1)  # Convert to class labels
                global_y_formula_class_trimmed = global_y_formula_class[:min_len]

                if isinstance(global_y_formula_class_trimmed, np.ndarray):
                    global_y_formula_class_trimmed = torch.tensor(global_y_formula_class_trimmed)

                    # Convert to binary labels (0 or 1) for both outputs and formula
                    global_y_test_binary = (global_y_test_out_trimmed == target_class).to(torch.int).flatten()  # Binary: 1 if equal to target_class, else 0
                    global_y_formula_binary = (global_y_formula_class_trimmed == target_class).to(torch.int).flatten()

                # Check the shapes of both arrays to ensure they are the same
                print(f"global_y_test_binary shape: {global_y_test_binary.shape}")
                print(f"global_y_formula_binary shape: {global_y_formula_binary.shape}")

                # Ensure both arrays have the same length
                min_len = min(global_y_test_binary.shape[0], global_y_formula_binary.shape[0])
                global_y_test_binary = global_y_test_binary[:min_len]
                global_y_formula_binary = global_y_formula_binary[:min_len]

                # Now compute the accuracy score
                global_explanation_fidelity_class = accuracy_score(global_y_test_binary.cpu(), global_y_formula_binary.cpu())

                # Print the accuracy result
                print(f"Global Explanation Fidelity Class Accuracy: {global_explanation_fidelity_class}")

                global_y_formula_binary = (global_y_formula_class_trimmed == target_class).to(torch.int).flatten()

                # Compute accuracy score with trimmed arrays
                global_explanation_fidelity_class = accuracy_score(global_y_test_binary, global_y_formula_binary)
                global_explanation_fidelity += global_explanation_fidelity_class
            global_explanation_accuracy += global_explanation_accuracy_class
            if concept_names is not None and global_explanation_class is not None:
                global_explanation_class = replace_names(global_explanation_class, concept_names)
            explanations[target_class].append(global_explanation_class)

            if not global_explanation_class:
                # print('No global explanation')
                continue
            else:
                print('------------------')
                print('Class:', target_class)
                print('Global explanation:', global_explanation_class)
                print('Explanation accuracy:', global_explanation_accuracy_class)
                print('Explanation fidelity:', global_explanation_fidelity_class)
                with open(filename, 'a') as file:
                    file.write('------------------\n')
                    file.write(f'Class: {target_class}\n')
                    file.write(f'Global explanation: {global_explanation_class}\n')
                    file.write(f'Explanation accuracy: {global_explanation_accuracy_class}\n')
                    file.write(f'Explanation fidelity: {global_explanation_fidelity_class}\n')

            for u in range(num_users):
                # record how many times a client appears in the aggregation list for all classes
                if u in user_to_engage_class[target_class]:
                    print("* ", end='')
                    with open(filename, 'a') as file:
                        file.write('* ')
                    users_aggregation_weight[u] += 1
                if not local_explanation_f[u]:
                    continue
                local_explanation_class = local_explanation_f[u][target_class]['explanation']
                local_explanation_accuracy_class, _ = test_explanation(local_explanation_class,
                                                                       test_data.dataset.tensors[0][test_data.indices],
                                                                       test_data.dataset.tensors[1][test_data.indices],
                                                                       target_class)
                if concept_names is not None and local_explanation_class is not None:
                    local_explanation_class = replace_names(local_explanation_class, concept_names)
                print('User ID:', u)
                print('Local explanation:', local_explanation_class)
                print('Explanation accuracy:', local_explanation_accuracy_class)
                with open(filename, 'a') as file:  # 'a' will append to the existing file
                    file.write(f'User ID: {u}\n')
                    file.write(f'Local explanation: {local_explanation_class}\n')
                    file.write(f'Explanation accuracy: {local_explanation_accuracy_class}\n')
            print('------------------')
            # to get the union of the selected users based on different classes for model updates
            users_to_aggregate = users_to_aggregate.union(user_to_engage_class[target_class])

        if users_to_aggregate is not None and len(users_to_aggregate) != 0:
            # For avg all parameters of all users
            # global_weights = average_weights(local_weights)

            # For avg all parameters using the union of valid users
            # global_weights = average_weights({k: v for k, v in local_weights.items() if k in users_to_aggregate})

            # For weighted all parameters using the aggregation weights
            global_weights = weighted_weights(local_weights, users_aggregation_weight)

            # For avg parameters class by class based on the valid users per class
            # global_weights = average_weights_class(local_weights, user_to_engage_class)

            # For max parameters class by class based on the valid users per class, simulating OR operation
            # global_weights = max_weights_class(local_weights, user_to_engage_class)
        else:
            global_weights = average_weights(local_weights)
        # update global weights
        global_model_parameters_list.append(global_weights)
        global_model.load_state_dict(global_weights)

        # model.freeze()
        # global model validation
        global_model_validation_results = global_trainer.test(copy.deepcopy(global_model), dataloaders=val_loader)
        global_model_results_list.append(global_model_validation_results)
        valid_data = []
        for item in test_loader.dataset:
            if isinstance(item, tuple) and len(item) == 2:
                valid_data.append(item)
        if valid_data:
            x_data, y_data = zip(*valid_data)  
            x_tensor = torch.stack(x_data)  
            y_tensor = torch.stack(y_data) 

            valid_dataset = TensorDataset(x_tensor, y_tensor)
            valid_test_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
            # decide if early stop using the validation results
            if len(global_model_results_list) > 1:
                if global_model_results_list[-1][0]['test_acc_epoch'] <= global_model_results_list[-2][0]['test_acc_epoch']:
                    print('Global model has reached best performance.')

                    global_model_results = global_trainer.test(copy.deepcopy(global_model), dataloaders=valid_test_loader)
                    print('---------------------------')
                    print('Global model accuracy:', global_model_results[0]['test_acc_epoch'])
                    print('Global rule mean accuracy', global_explanation_accuracy / n_classes)
                    print('Global rule mean fidelity', global_explanation_fidelity / n_classes)
                    print('---------------------------')
                    with open(filename, 'a') as file:  # 'a' will append to the existing file
                        file.write('---------------------------\n')
                        file.write('Global model has reached best performance in the last epoch.\n')
                        file.write(f'Global model accuracy: {global_model_results[0]["test_acc_epoch"]}\n')
                        file.write(f'Global rule mean accuracy: {global_explanation_accuracy / n_classes}\n')
                        file.write(f'Global rule mean fidelity: {global_explanation_fidelity / n_classes}\n')
                        file.write('---------------------------\n')
                    # break
            global_model_results = global_trainer.test(copy.deepcopy(global_model), dataloaders=valid_test_loader)
            print('---------------------------')
            print('Global model accuracy:', global_model_results[0]['test_acc_epoch'])
            print('Global rule mean accuracy', global_explanation_accuracy/n_classes)
            print('Global rule mean fidelity', global_explanation_fidelity/n_classes)
            print('---------------------------')
            with open(filename, 'a') as file:  # 'a' will append to the existing file
                file.write('---------------------------\n')
                file.write(f'Global model accuracy: {global_model_results[0]["test_acc_epoch"]}\n')
                file.write(f'Global rule mean accuracy: {global_explanation_accuracy / n_classes}\n')
                file.write(f'Global rule mean fidelity: {global_explanation_fidelity / n_classes}\n')
                file.write('---------------------------\n')

if explanations != None:
    consistencies = []
    for j in range(n_classes):
        consistencies.append(formula_consistency(explanations[j]))
    explanation_consistency = np.mean(consistencies)

    results_df = pd.DataFrame(results_list)
    results_df['explanation_consistency'] = explanation_consistency
    results_df.to_csv(os.path.join(base_dir, 'results_aware_cub.csv'))
    # results_df
    # results_df.mean()
    # results_df.sem()
else:
    print("No valid explanations")

