import matplotlib.pyplot as plt
import talib
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def feature_frequencies(X):
    lst = [0] * len(X[0])
    for point in X:
        for i in range(len(point) - 1):
            if point[i] != 0:
                lst[i] += 1

    indices = np.argsort(lst)[::-1]
    lst.sort(reverse=True)
    print(indices)
    print(lst)

    plt.figure()
    plt.title("Feature frequencies")
    plt.bar(range(X.shape[1]), lst,
           color="r", align="center")
    plt.xticks(range(X.shape[1]), indices, rotation=30)


    # fig.autofmt_xdate()
    # plt.xticks(range(X.shape[1]), lst)
    # plt.xlim([-1, X.shape[1]])
    # plt.xticks(range(X.shape[1]), indices)
    # plt.xlim([-1, X.shape[1]])
    plt.show()

def feature_importances(X, y):
    forest = RandomForestClassifier(n_estimators=250)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", align="center")
    plt.xticks(range(X.shape[1]), indices, rotation=30)
    plt.xlim([-1, X.shape[1]])
    plt.show()
