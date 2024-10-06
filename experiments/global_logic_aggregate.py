import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from entropy_lens.logic.nn.entropy import explain_class
from entropy_lens.logic.metrics import test_explanation, complexity
from entropy_lens.nn.functional.loss import entropy_logic_loss
from entropy_lens.nn.logic import EntropyLinear
from sympy import simplify_logic
from entropy_lens.logic.utils import replace_names


def client_selection_class(n_classes, num_users, local_explanation_f):
    print(n_classes)
    print(num_users)
    # if all(value is None for value in local_explanation.values()):
    user_to_engage_class = {}
    # local_explanation_accuracy = {}
    local_explanations_accuracy_class = {}
    local_explanations_support_class = {}

    for m in range(len(local_explanation_f)):
        print(f"Client {m} explanations: {local_explanation_f[m]}")


    for target_class in range(n_classes):
        local_explanations_accuracy = {}
        # {explanation: total support users, total support accuracy, support users}
        local_explanations_support = {}
        users_to_keep = set([])
        users_to_disgard = set([])
        for m in range(num_users):
            if not local_explanation_f[m]:
                continue
            if not local_explanation_f[m][target_class]:
                continue
            local_explanation = local_explanation_f[m][target_class]['explanation']
            if local_explanation:
                users_to_keep.add(m)
                if local_explanation in local_explanations_accuracy:
                    list(local_explanations_support[local_explanation])[2].append(m)
                    local_explanations_support[local_explanation] = (
                        list(local_explanations_support[local_explanation])[0] + 1,
                        list(local_explanations_support[local_explanation])[1] + local_explanation_f[m][target_class][
                            'explanation_accuracy'],
                        list(local_explanations_support[local_explanation])[2])
                    if local_explanation_f[m][target_class]['explanation_accuracy'] > \
                            list(local_explanations_accuracy[local_explanation])[1]:
                        users_to_disgard.update(list(local_explanations_accuracy[local_explanation])[2])
                        local_explanations_accuracy[local_explanation] = (
                        local_explanation, local_explanation_f[m][target_class]['explanation_accuracy'], [m])
                    elif local_explanation_f[m][target_class]['explanation_accuracy'] == \
                            list(local_explanations_accuracy[local_explanation])[1]:
                        list(local_explanations_accuracy[local_explanation])[2].append(m)
                    else:
                        users_to_disgard.add(m)
                else:
                    local_explanations_accuracy[local_explanation] = (
                    local_explanation, local_explanation_f[m][target_class]['explanation_accuracy'], [m])
                    local_explanations_support[local_explanation] = (
                    1, local_explanation_f[m][target_class]['explanation_accuracy'], [m])
        local_explanations_accuracy_class[target_class] = local_explanations_accuracy
        local_explanations_support_class[target_class] = local_explanations_support
        if users_to_keep:
            user_to_engage_class[target_class] = users_to_keep - users_to_disgard
        else:
            user_to_engage_class[target_class] = set([m for m in range(num_users)])

    return user_to_engage_class, local_explanations_accuracy_class, local_explanations_support_class


def _global_aggregate_explanations(local_explanations_accuracy, local_explanations_support, topk_explanations,
                                   target_class, x, y, concept_names, user_engagement_scale, connector, beam_width):
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
        return None, 0, set([i for i in range(topk_explanations)])

    else:
        users_to_keep = set()
        # get the topk most accurate local explanations
        local_explanations_sorted = sorted(local_explanations_accuracy.items(), key=lambda x: -x[1][1])[:topk_explanations]

        # Initialize the beam with an empty sequence and 0 accuracy
        beam = [(0, [])]

        # The maximum number of steps to take
        max_steps = len(local_explanations_sorted)

        for step in range(max_steps):
            # For each sequence in the beam, expand it with each possible next explanation
            next_beam = []
            for accuracy, sequence in beam:
                for explanation_raw, (explanation, accuracy, user_id) in local_explanations_sorted:
                    if explanation in sequence:
                        continue

                    # Create a new sequence by adding the explanation
                    new_sequence = sequence + [explanation]

                    # Aggregate the explanations in the new sequence
                    if connector == 'AND':
                        aggregated_explanation = ' & '.join(new_sequence)
                    else:
                        aggregated_explanation = ' | '.join(new_sequence)

                    if aggregated_explanation in ['', 'False', 'True', '(False)', '(True)']:
                        continue

                    # Test the accuracy of the new sequence
                    new_accuracy, _ = test_explanation(aggregated_explanation, x, y, target_class)

                    # Store the new sequence and its accuracy
                    next_beam.append((new_accuracy, new_sequence))

            # Select the top sequences by accuracy up to the beam width
            beam = sorted(next_beam, key=lambda x: -x[0])[:beam_width]

        # Select the sequence with the highest accuracy from the final beam
        best_accuracy, best_sequence = max(beam, key=lambda x: x[0])

        # Process the best sequence
        aggregated_explanation = ' & '.join(best_sequence) if connector == 'AND' else ' | '.join(best_sequence)
        aggregated_explanation_simplified = simplify_logic(aggregated_explanation, 'dnf', force=True)
        aggregated_explanation_simplified = f'({aggregated_explanation_simplified})'
        best_explanation = aggregated_explanation_simplified

        # Update users_to_keep based on the best_sequence
        for explanation in best_sequence:
            if user_engagement_scale == 'large':
                users_to_keep.update(list(local_explanations_support[explanation])[2])
            else:
                users_to_keep.update(user_id)

    return best_explanation, best_accuracy, users_to_keep