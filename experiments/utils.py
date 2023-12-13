import copy
import torch


def average_weights(w):
    """
    Returns the average of the weights.
    """
    key_list = list(w.keys())
    w_avg = copy.deepcopy(w[key_list[0]])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[key_list[i]][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_weights_class(w, user_to_engage_class):
    """
    Returns the average of the weights (model 0 separately avg for classes).
    """
    key_list = list(w.keys())
    w_avg = copy.deepcopy(w[key_list[0]])
    for key in w_avg.keys():
        if key == 'model.0.weight' or key == 'model.0.bias':
            for c in range(len(user_to_engage_class)):
                stacked_tensor = torch.stack([w[user][key][c] for user in list(user_to_engage_class[c])])
                w_avg[key][c] = torch.mean(stacked_tensor, dim=0)[0]
        else:
            stacked_tensor = torch.stack([w[key_list[i]][key] for i in range(0, len(w))])
            w_avg[key] = torch.mean(stacked_tensor, dim=0)
    return w_avg


def weighted_weights(w, users_aggregation_weight):
    """
    Returns the weighted average of the weights.
    """
    key_list = list(w.keys())

    agg_sum = sum(users_aggregation_weight.values())
    if agg_sum == 0:
        users_aggregation_weight_norm = {key: 1 / len(users_aggregation_weight) for key, _ in users_aggregation_weight.items()}
    else:
        users_aggregation_weight_norm = {key: value / agg_sum for key, value in users_aggregation_weight.items()}

    w_avg = copy.deepcopy(w[key_list[0]])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * users_aggregation_weight_norm[key_list[0]]
        for i in range(1, len(w)):
            w_avg[key] += w[key_list[i]][key] * users_aggregation_weight_norm[key_list[i]]
    return w_avg

def max_weights(w):
    """
    Returns the average of the weights.
    """
    key_list = list(w.keys())
    w_avg = copy.deepcopy(w[key_list[0]])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[key_list[i]][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def max_weights_class(w, user_to_engage_class):
    """
    Returns the average of the weights (model 0 separately avg for classes).
    """
    key_list = list(w.keys())
    w_max = copy.deepcopy(w[key_list[0]])
    for key in w_max.keys():
        if key == 'model.0.weight' or key == 'model.0.bias':
            for c in range(len(user_to_engage_class)):
                stacked_tensor = torch.stack([w[user][key][c] for user in list(user_to_engage_class[c])])
                w_max[key][c] = torch.max(stacked_tensor, dim=0)[0]
        else:
            stacked_tensor = torch.stack([w[key_list[i]][key] for i in range(0, len(w))])
            w_max[key] = torch.max(stacked_tensor, dim=0)[0]
    return w_max

