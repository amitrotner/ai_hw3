import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    train_data = pd.read_csv("train.csv", sep=',', header=0)
    train_features = train_data.values[:, 0:8]
    train_outcome = train_data.values[:, 8]
    decision_tree = DecisionTreeClassifier(criterion="entropy", min_samples_split=9, class_weight={1: 4, 0: 1})
    decision_tree.fit(train_features, train_outcome)
    test_data = pd.read_csv("test.csv", sep=',', header=0)
    test_features = test_data.values[:, 0:8]
    test_real_outcome = test_data.values[:, 8]
    test_outcome = decision_tree.predict(test_features)
    print(confusion_matrix(test_real_outcome, test_outcome))
