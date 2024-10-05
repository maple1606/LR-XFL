import numpy as np
from torchvision import datasets, transforms
import torch
import random


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = len(dataset) // num_users  # Items per user
    dict_users = {}
    all_idxs = np.arange(len(dataset))

    np.random.shuffle(all_idxs)  
    for i in range(num_users):
        if i < num_users - 1:
            dict_users[i] = set(all_idxs[i * num_items: (i + 1) * num_items])
        else:
            dict_users[i] = set(all_idxs[i * num_items:])
    
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, int(len(dataset)/200)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype=int) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = torch.argmax(dataset.dataset.tensors[1], dim=1)[dataset.indices].numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 20, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid_per_class(dataset, num_users):
    assert num_users <= 10
    lst = list(range(10))
    part_size = len(lst) // num_users  # Integer division to get the size of each part
    remaining = len(lst) % num_users

    user_shard = [lst[i * part_size: (i + 1) * part_size] + ([lst[-remaining+i]] if i < remaining else [])
                  for i in range(num_users)]

    input_class = torch.argmax(dataset.dataset.tensors[0], dim=1)[dataset.indices].numpy()
    dict_users = {}
    for i in range(num_users):
        dict_users[i] = np.where([input_class[j] in user_shard[i] for j in range(len(input_class))])[0]
    return dict_users



def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def cub_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cub_noniid_per_class(dataset, num_users):
    lst = list(range(108))
    part_size = len(lst) // num_users  # Integer division to get the size of each part
    remaining = len(lst) % num_users

    user_shard = [lst[i * part_size: (i + 1) * part_size] + ([lst[-remaining+i]] if i < remaining else [])
                  for i in range(num_users)]

    tensor = dataset.dataset.tensors[0][dataset.indices]
    result = []

    for row in tensor:
        max_val = row.max().item()  # Get the max value of this row
        indices_of_max_vals = (row == max_val).nonzero(as_tuple=True)[0].numpy()  # Get indices of max values
        selected_index = np.random.choice(indices_of_max_vals)  # Randomly select one of these indices
        result.append(selected_index)

    input_class = torch.tensor(result)

    # input_class = torch.argmax(dataset.dataset.tensors[0], dim=1)[dataset.indices].numpy()
    dict_users = {}
    for i in range(num_users):
        dict_users[i] = np.where([input_class[j] in user_shard[i] for j in range(len(input_class))])[0]
    return dict_users


def cub_noniid(dataset, num_users):
    lst = list(range(200))
    part_size = len(lst) // num_users
    remaining = len(lst) % num_users

    user_shard = [lst[i * part_size: (i + 1) * part_size] + ([lst[-remaining + i]] if i < remaining else [])
                  for i in range(num_users)]

    tensor = dataset.dataset.tensors[0][dataset.indices]
    training_labels = dataset.dataset.tensors[1][dataset.indices]
    result = []

    for row in training_labels:
        result.append(torch.argmax(row))

    input_class = torch.tensor(result)

    # input_class = torch.argmax(dataset.dataset.tensors[0], dim=1)[dataset.indices].numpy()
    dict_users = {}
    for i in range(num_users):
        dict_users[i] = []
        for idx, j in enumerate(input_class):
            if j in user_shard[i]:
                dict_users[i].append(idx)
    return dict_users


def mimic_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def generate_random_numbers(target_sum, num_values):
    values = [0] + sorted(random.sample(range(1, target_sum), num_values - 1)) + [target_sum]
    return [values[i+1] - values[i] for i in range(num_values)]


def mimic_noniid(dataset, num_users):
    lst = list(range(90))
    num_users = int(num_users/2)
    part_size = len(lst) // num_users
    remaining = len(lst) % num_users

    user_shard = [lst[i * part_size: (i + 1) * part_size] + ([lst[-remaining + i]] if i < remaining else [])
                  for i in range(num_users)]

    tensor = dataset.dataset.tensors[0][dataset.indices]
    training_labels = dataset.dataset.tensors[1][dataset.indices]
    result = []

    for row in tensor:
        max_val = row.max().item()
        indices_of_max_vals = (row == max_val).nonzero(as_tuple=True)[0].numpy()
        selected_index = np.random.choice(indices_of_max_vals)
        result.append(selected_index)

    input_class = torch.tensor(result)

    # input_class = torch.argmax(dataset.dataset.tensors[0], dim=1)[dataset.indices].numpy()
    dict_users = {}
    for i in range(num_users):
        dict_users[2 * i] = []
        dict_users[2 * i + 1] = []
        for idx, j in enumerate(result):
            if j in user_shard[i]:
                if training_labels[idx][0].item() == 1.:
                    dict_users[2 * i].append(idx)
                elif training_labels[idx][1].item() == 1.:
                    dict_users[2 * i + 1].append(idx)

    return dict_users


def mimic_noniid_per_class(dataset, num_users):
    lst = list(range(90))
    part_size = len(lst) // num_users  # Integer division to get the size of each part
    remaining = len(lst) % num_users

    user_shard = [lst[i * part_size: (i + 1) * part_size] + ([lst[-remaining+i]] if i < remaining else [])
                  for i in range(num_users)]

    tensor = dataset.dataset.tensors[0][dataset.indices]
    result = []

    for row in tensor:
        max_val = row.max().item()  # Get the max value of this row
        indices_of_max_vals = (row == max_val).nonzero(as_tuple=True)[0].numpy()  # Get indices of max values
        selected_index = np.random.choice(indices_of_max_vals)  # Randomly select one of these indices
        result.append(selected_index)

    input_class = torch.tensor(result)

    # input_class = torch.argmax(dataset.dataset.tensors[0], dim=1)[dataset.indices].numpy()
    dict_users = {}
    for i in range(num_users):
        dict_users[i] = np.where([input_class[j] in user_shard[i] for j in range(len(input_class))])[0]
    return dict_users


def vdem_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def vdem_noniid_per_class(dataset, num_users):
    lst = list(range(14))
    # random.shuffle(lst)
    part_size = len(lst) // num_users  # Integer division to get the size of each part
    remaining = len(lst) % num_users

    user_shard = [lst[i * part_size: (i + 1) * part_size] + ([lst[-remaining+i]] if i < remaining else [])
                  for i in range(num_users)]

    tensor = dataset.dataset.tensors[0][dataset.indices]
    result = []

    for row in tensor:
        max_val = row.max().item()  # Get the max value of this row
        indices_of_max_vals = (row == max_val).nonzero(as_tuple=True)[0].numpy()  # Get indices of max values
        selected_index = np.random.choice(indices_of_max_vals)  # Randomly select one of these indices
        result.append(selected_index)

    input_class = torch.tensor(result)

    # input_class = torch.argmax(dataset.dataset.tensors[0], dim=1)[dataset.indices].numpy()
    dict_users = {}
    for i in range(num_users):
        dict_users[i] = np.where([input_class[j] in user_shard[i] for j in range(len(input_class))])[0]
    return dict_users


def vdem_noniid(dataset, num_users):
    lst = list(range(14))
    num_users = int(num_users/2)
    part_size = len(lst) // num_users
    remaining = len(lst) % num_users

    user_shard = [lst[i * part_size: (i + 1) * part_size] + ([lst[-remaining + i]] if i < remaining else [])
                  for i in range(num_users)]

    tensor = dataset.dataset.tensors[0][dataset.indices]
    training_labels = dataset.dataset.tensors[1][dataset.indices]
    result = []

    for row in tensor:
        max_val = row.max().item()
        indices_of_max_vals = (row == max_val).nonzero(as_tuple=True)[0].numpy()
        selected_index = np.random.choice(indices_of_max_vals)
        result.append(selected_index)

    input_class = torch.tensor(result)

    # input_class = torch.argmax(dataset.dataset.tensors[0], dim=1)[dataset.indices].numpy()
    dict_users = {}
    for i in range(num_users):
        dict_users[2 * i] = []
        dict_users[2 * i + 1] = []
        for idx, j in enumerate(result):
            if j in user_shard[i]:
                if training_labels[idx][0].item() == 1.:
                    dict_users[2 * i].append(idx)
                elif training_labels[idx][1].item() == 1.:
                    dict_users[2 * i + 1].append(idx)

    return dict_users






