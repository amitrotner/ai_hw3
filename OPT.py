import numpy as np
import pandas as pd
from math import sqrt
from itertools import combinations
from sklearn.metrics import accuracy_score


def euclidean_dist(x, y, indices):
    dist = 0
    for i in indices:
        dist += (x[i] - y[i]) ** 2
    return sqrt(dist)


def get_combinations():
    for j in range(1, 9):
        for combination in list(combinations([i for i in range(8)], j)):
            yield list(combination)


if __name__ == "__main__":
    k = 9

    # Loading train data
    train_data = pd.read_csv("train.csv", sep=',', header=0)
    train_features = train_data.values[:, 0:8]
    train_outcome = train_data.values[:, 8]

    # Finding max and min value for every feature in the training data
    min_list = [min(train_features[:, i]) for i in range(8)]
    max_list = [max(train_features[:, i]) for i in range(8)]

    # Normalize training data by min and max value
    for i in range(8):
        for j in range(len(train_features[:, i])):
            train_features[j, i] = (train_features[j, i] - min_list[i]) / (max_list[i] - min_list[i])

    # Loading test data
    test_data = pd.read_csv("test.csv", sep=',', header=0)
    test_features = test_data.values[:, 0:8]
    test_real_outcome = test_data.values[:, 8]
    test_outcome = test_data.values[:, 8]

    # Normalize test data by min and max value of train data
    for i in range(8):
        for j in range(len(test_features[:, i])):
            test_features[j, i] = (test_features[j, i] - min_list[i]) / (max_list[i] - min_list[i])

    # Looping over all combinations of features
    accuracy = 0
    best_comb = []
    for comb in get_combinations():
        # Finding KNN by euclidean distance and changing value in test outcome according to the majority vote
        for i in range(len(test_features[:, 0])):
            arr_dist = []
            for j in range(len(train_features[:, 0])):
                arr_dist += [euclidean_dist(test_features[i, :], train_features[j, :], comb)]
            arr_dist = list(np.argsort(arr_dist)[:k])
            for index in range(len(arr_dist)):
                arr_dist[index] = train_outcome[arr_dist[index]]
            test_outcome[i] = max(arr_dist, key=arr_dist.count)
        # If we got a better accuracy, change best combination to this combination
        if accuracy_score(test_real_outcome, test_outcome) * 100 > accuracy:
            accuracy = (accuracy_score(test_real_outcome, test_outcome) * 100)
            best_comb = comb
        test_outcome = test_data.values[:, 8]

    print(best_comb)
