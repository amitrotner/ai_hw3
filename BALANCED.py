import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier


if __name__ == "__main__":
    train_data = pd.read_csv("train.csv", sep=',', header=0)
    T = len([1 for value in list(train_data.values[:, 8]) if value == 1])
    list_of_index_to_drop = []
    for index, row in train_data.iterrows():
        if row['Outcome'] == 0:
            list_of_index_to_drop += [index]
            T -= 1
        if T == 0:
            break
    train_data = train_data.drop(train_data.index[index] for index in list_of_index_to_drop)
    train_features = train_data.values[:, 0:8]
    train_outcome = train_data.values[:, 8]
    decision_tree = DecisionTreeClassifier(criterion="entropy")
    decision_tree.fit(train_features, train_outcome)
    test_data = pd.read_csv("test.csv", sep=',', header=0)
    test_features = test_data.values[:, 0:8]
    test_real_outcome = test_data.values[:, 8]
    test_outcome = decision_tree.predict(test_features)
    print(confusion_matrix(test_real_outcome, test_outcome))
