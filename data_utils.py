from collections import defaultdict

import numpy as np


def expand_by_nan(arr, size):
    """Append nan elements to the array to the desired size 

    Args:
        arr: input array
        size (int): desired size

    Returns:
        expanded array
    """
    result = np.copy(arr)
    result = result[-size:]
    n = len(result)
    if (n < size):
        k = size - n
        result = np.hstack((np.array([np.nan] * k), result))
    assert len(result) == size
    return result


def subsamples(sample, size):
    result = []
    for i in range(0, len(sample) - size + 1):
        result.append(sample[i:i + size])
    return result


def split_history(user_history, size, target_index=0):
    """ Divide user_history into fixed-size intervals

    Args:
        user_history: history for some user for all features (including target)
        user_history = [target_history, feature_1_history, feature_2_history, ...]
        size (int): interval size
        target_index (int, optional): index for target_history. Defaults to 0.

    Returns:
        dataset (X, y) consisting of fixed-size vectors to train a model
    """

    inputs = []
    labels = []

    features_with_subsamples = []
    n_features = len(user_history)

    target_subsamples = subsamples(user_history[target_index], size)
    count_of_subsamples = len(target_subsamples)
    assert np.size(user_history[
                       target_index]) >= size, "To split data, the history length must be equal to or greater than the desired interval size"

    features_with_subsamples.append(target_subsamples)
    for f in user_history[1:]:
        features_with_subsamples.append(subsamples(f, size))

    for i in range(count_of_subsamples):
        X = []
        X.append(target_subsamples[i][:-1])
        for j in range(1, n_features):
            X.append(features_with_subsamples[j][i])
        inputs.append(np.concatenate(X))
        labels.append(target_subsamples[i][-1])
    return inputs, labels


def create_test_dataset(clients_history_np, history_size):
    to_test_dataset = defaultdict(list)
    for client_history in clients_history_np:
        client_history_len = np.size(client_history[0])
        if (client_history_len <= 2):
            continue
        F = []
        # the last element ([-1]) is nan, choose [-2] to test
        previous_targets = client_history[0][:-2]
        F.append(expand_by_nan(previous_targets, history_size - 1))
        for f in client_history[1:]:
            truncated_f = f[:-1]
            F.append(expand_by_nan(truncated_f, history_size))
        target_to_predict = client_history[0][-2]
        F = np.hstack(F)
        to_test_dataset[client_history_len].append((target_to_predict, F))

    print('to_test_dataset len = ', len(to_test_dataset))
    return to_test_dataset


def create_train_dataset(clients_history_np, history_size):
    progres_bar_delta = 1000

    TARGET_INDEX = 0
    X = []
    y = []

    i = 0
    print('Progress to prepare datasets... ')
    print('.' * (len(clients_history_np) // progres_bar_delta))
    for client_history in clients_history_np:

        if i % progres_bar_delta == 0:
            print('.', end='')
        i += 1

        client_history_len = np.size(client_history[0]) - 2
        if (client_history_len < history_size):
            continue
        # don't consider the last two values from each client history
        # have used them for validation and making the prediction
        F = []
        for feature_history in client_history:
            F.append(feature_history[:-2])
        inputs, labels = split_history(F, history_size)
        X.append(np.stack(inputs))
        y.append(labels)
    print()
    print('X size = ', len(X))
    print('y size = ', len(y))

    y = np.concatenate(y)
    X = np.vstack(X)
    print('X_set shape = ', X.shape)
    print('y_set shape = ', y.shape)
    return X, y


def create_weights(y_train, largest_class_weight_coef):
    classes = np.unique(y_train, axis=0)
    print(classes)
    classes.sort()
    class_samples = np.bincount(y_train)
    total_samples = class_samples.sum()
    n_classes = len(class_samples)
    weights = total_samples / (n_classes * class_samples * 1.0)
    class_weight_dict = {key: value for (key, value) in zip(classes, weights)}
    class_weight_dict[classes[1]] = class_weight_dict[classes[1]
                                    ] * largest_class_weight_coef
    sample_weights = [class_weight_dict[y] for y in y_train]
    return sample_weights
