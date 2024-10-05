import copy
from typing import List

import torch
import re
from sympy import simplify_logic

from entropy_lens.logic.metrics import test_explanation
from entropy_lens.logic.utils import replace_names
from entropy_lens.nn.concepts import Conceptizator
from entropy_lens.nn.logic import EntropyLinear


def explain_class(model: torch.nn.Module, x, y1h, x_val: torch.Tensor, y_val1h: torch.Tensor,
                  target_class: int, max_minterm_complexity: int = None, topk_explanations: int = 3,
                  max_accuracy: bool = False, concept_names: List = None) -> [str, str]:
    """
    Generate a local explanation for a single sample.

    :param model: pytorch model
    :param x: input samples to extract logic formulas.
    :param y1h: target labels to extract logic formulas (MUST be one-hot encoded).
    :param x_val: input samples to validate logic formulas.
    :param y_val1h: target labels to validate logic formulas (MUST be one-hot encoded).
    :param target_class: target class.
    :param max_minterm_complexity: maximum number of concepts per logic formula (per sample).
    :param topk_explanations: number of local explanations to be combined.
    :param max_accuracy: if True a formula is simplified only if the simplified formula gets 100% accuracy.
    :param concept_names: list containing the names of the input concepts.
    :return: Global explanation
    """
    x_correct, y_correct1h = _get_correct_data(x, y1h, model, target_class)
    if x_correct is None:
        return None, None, None

    activation = 'identity_bool'
    feature_names = [f'feature{j:010}' for j in range(x_correct.size(1))]
    conceptizator = Conceptizator(activation)
    y_correct = conceptizator(y_correct1h[:, target_class])
    y_val = conceptizator(y_val1h[:, target_class])

    class_explanation = ''
    class_explanation_raw = ''
    # use a n*n matrix to record concept co-occurrence
    class_concept_co_occ = {}
    class_concept_co_occ['positive'] = torch.zeros([len(feature_names), len(feature_names)])
    class_concept_co_occ['negative'] = torch.zeros([len(feature_names), len(feature_names)])


    for layer_id, module in enumerate(model.children()):
        if isinstance(module, EntropyLinear):
            local_explanations_accuracies = {}
            local_explanations_raw = {}

            # look at the "positive" rows of the truth table only
            positive_samples = torch.nonzero(y_correct)
            for positive_sample in positive_samples:
                # The original code is "x_correct, y_correct1h", but since in the simplify code, the
                # name is c_validation, I just pass the validation data instead. It makes sense.
                local_explanation, local_explanation_raw, acc, acc_raw = _local_explanation(module, feature_names, positive_sample,
                                                                              local_explanations_raw,
                                                                              x_val, y_val1h,
                                                                              target_class, max_accuracy,
                                                                              max_minterm_complexity, class_concept_co_occ)

                # test explanation accuracy
                if local_explanation_raw and local_explanation_raw not in local_explanations_accuracies:
                        local_explanations_raw[local_explanation_raw] = local_explanation
                        local_explanations_accuracies[local_explanation_raw] = (local_explanation, acc)

            # To decide whether to use 'AND' or 'OR' to connect rules
            # Get the maximum row sum value
            max_neg_row_sum = class_concept_co_occ['negative'].sum(dim=1).max().item()
            pattern = r'feature\d+'
            rules_length = 0
            # Iterate through each key in the dictionary
            for key in local_explanations_raw:
                features_in_key = re.findall(pattern, key)
                rules_length += len(features_in_key)
                break
            rules_length *= len(positive_samples)


            connector = 'AND'
            if rules_length == 0:
                # Handle the case where no valid rules are generated
                print("No valid rules generated.")
                return None, None, None
            else:
                if torch.trace(class_concept_co_occ['positive']) / torch.sum(class_concept_co_occ['positive']) > 0.9 or max_neg_row_sum / rules_length > 0.8:
                    connector = 'OR'
            # connector = 'OR'

            # aggregate local explanations and replace concept names in the final formula
            aggregated_explanation, best_acc = _aggregate_explanations(local_explanations_accuracies,
                                                                       topk_explanations,
                                                                       target_class, x_val, y_val1h, connector)
            class_explanation_raw = str(aggregated_explanation)
            class_explanation = class_explanation_raw

            # modify the code so that the explanation does not change to concept_names at client slide
            # if concept_names is not None:
            #     class_explanation = replace_names(class_explanation, concept_names)

            break

    return class_explanation[1:-1], class_explanation_raw, connector


def _simplify_formula(explanation: str, x: torch.Tensor, y: torch.Tensor, target_class: int, max_accuracy: bool) -> str:
    """
    Simplify formula to a simpler one that is still coherent.

    :param explanation: local formula to be simplified.
    :param x: input data.
    :param y: target labels (1D, categorical NOT one-hot encoded).
    :param target_class: target class
    :param max_accuracy: drop  term only if it gets max accuracy
    :return: Simplified formula
    """

    base_accuracy, _ = test_explanation(explanation, x, y, target_class)
    raw_accuracy = base_accuracy
    for term in explanation.split(' & '):
        explanation_simplified = copy.deepcopy(explanation)

        if explanation_simplified.endswith(f'{term}'):
            explanation_simplified = explanation_simplified.replace(f' & {term}', '')
        else:
            explanation_simplified = explanation_simplified.replace(f'{term} & ', '')

        if explanation_simplified:
            accuracy, preds = test_explanation(explanation_simplified, x, y, target_class)
            if (max_accuracy and accuracy == 1.) or (not max_accuracy and accuracy >= base_accuracy):
                explanation = copy.deepcopy(explanation_simplified)
                base_accuracy = accuracy

    return explanation, raw_accuracy, base_accuracy


def _aggregate_explanations(local_explanations_accuracy, topk_explanations, target_class, x, y, symbol):
    """
    Sort explanations by accuracy and then aggregate explanations which increase the accuracy of the aggregated formula.

    :param local_explanations_accuracy: dictionary of explanations and related accuracies.
    :param topk_explanations: limits the number of explanations to be aggregated.
    :param target_class: target class.
    :param x: observations in validation set.
    :param y: labels in validation set.
    :return:
    """
    if len(local_explanations_accuracy) == 0:
        return '', 0

    else:
        # get the topk most accurate local explanations
        local_explanations_sorted = sorted(local_explanations_accuracy.items(), key=lambda x: -x[1][1])[:topk_explanations]
        explanations = []
        best_accuracy = 0
        best_explanation = ''
        for explanation_raw, (explanation, accuracy) in local_explanations_sorted:
            # modified the original code logic to save time
            if explanation in explanations:
                continue
            explanations.append(explanation)

            # aggregate example-level explanations
            if symbol == 'AND':
                aggregated_explanation = ' & '.join(explanations)
            else:
                aggregated_explanation = ' | '.join(explanations)
            if aggregated_explanation in ['', 'False', 'True', '(False)', '(True)']:
                continue
            accuracy, _ = test_explanation(aggregated_explanation, x, y, target_class)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # aggregated_explanation_simplified = simplify_logic(aggregated_explanation, 'dnf', force=True)
                aggregated_explanation_simplified = aggregated_explanation
                aggregated_explanation_simplified = f'({aggregated_explanation_simplified})'
                best_explanation = aggregated_explanation_simplified
            else:
                # wrong original code to put explanations as [best_explanations] under if
                explanations = explanations[:-1]

    return best_explanation, best_accuracy


def _local_explanation(module, feature_names, neuron_id, neuron_explanations_raw,
                       c_validation, y_target, target_class, max_accuracy, max_minterm_complexity, class_concept_co_occ):
    # explanation is the conjunction of non-pruned features
    explanation_raw = ''
    if max_minterm_complexity:
        concepts_to_retain = torch.argsort(module.alpha[target_class], descending=True)[:max_minterm_complexity]
    else:
        non_pruned_concepts = module.concept_mask[target_class]
        concepts_sorted = torch.argsort(module.alpha[target_class])
        concepts_to_retain = concepts_sorted[non_pruned_concepts[concepts_sorted]]

    for j in concepts_to_retain:
        if feature_names[j] not in ['()', '']:
            if explanation_raw:
                explanation_raw += ' & '
            if module.conceptizator.concepts[0][neuron_id, j] > module.conceptizator.threshold:
                # if non_pruned_neurons[j] > 0:
                for k in concepts_to_retain:
                    if module.conceptizator.concepts[0][neuron_id, k] > module.conceptizator.threshold:
                        class_concept_co_occ['positive'][j][k] += 1
                    else:
                        class_concept_co_occ['negative'][j][k] += 1
                explanation_raw += feature_names[j]
            else:
                explanation_raw += f'~{feature_names[j]}'

    explanation_raw = str(explanation_raw)
    if explanation_raw in ['', 'False', 'True', '(False)', '(True)']:
        return None, None, None, None

    simplify = True
    # TODO: why if explanation_raw appeared once, it won't be simplified again
    if explanation_raw in neuron_explanations_raw:
        return None, None, None, None
    elif simplify:
        explanation, raw_acc, simplify_acc = _simplify_formula(explanation_raw, c_validation, y_target, target_class, max_accuracy)
    else:
        explanation = explanation_raw
        raw_acc, _ = test_explanation(explanation, c_validation, y_target, target_class)
        simplify_acc = raw_acc

    if explanation in ['', 'False', 'True', '(False)', '(True)']:
        return None, None, None, None

    return explanation, explanation_raw, simplify_acc, raw_acc


def _get_correct_data(x, y, model, target_class):
    x_target = x[y[:, target_class] == 1]
    y_target = y[y[:, target_class] == 1]

    # get model's predictions
    # Add a sigmoid function here, since model(x_target) is not within (0, 1)
    preds = torch.sigmoid(model(x_target).squeeze(-1))

    # identify samples correctly classified of the target class
    # TODO: the preds is not in the range of (0, 1), so why need to constrain it larger than 0.5, the forward codes are in logic line 37
    correct_mask = y_target[:, target_class].eq(preds[:, target_class]>0.5)
    # if sum(correct_mask) < 2:
    #     return None, None

    x_target_correct = x_target[correct_mask]
    y_target_correct = y_target[correct_mask]

    # collapse samples having the same boolean values and class label different from the target class
    x_reduced_opposite = x[y[:, target_class] != 1]
    y_reduced_opposite = y[y[:, target_class] != 1]
    preds_opposite = torch.sigmoid(model(x_reduced_opposite).squeeze(-1))

    # identify samples correctly classified of the opposite class
    correct_mask = y_reduced_opposite[:, target_class].eq(preds_opposite[:, target_class]>0.5)
    # if sum(correct_mask) < 2:
    #     return None, None

    x_reduced_opposite_correct = x_reduced_opposite[correct_mask]
    y_reduced_opposite_correct = y_reduced_opposite[correct_mask]

    # select the subset of samples belonging to the target class
    x_validation = torch.cat([x_reduced_opposite_correct, x_target_correct], dim=0)
    y_validation = torch.cat([y_reduced_opposite_correct, y_target_correct], dim=0)

    model.eval()
    model(x_validation)
    return x_validation, y_validation
