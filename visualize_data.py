import matplotlib.pyplot as plt
import talib
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def plot_feature_frequencies(X):
    '''
        Given a numpy training dataset with features and labels,
        this function plots the frequencies of each feature on a histogram,
        in reverse sorted order.
    '''

    # Update frequencies of each feature
    lst = [0] * len(X[0])
    for point in X:
        for i in range(len(point) - 1):
            if point[i] != 0:
                lst[i] += 1

    # sort indices from greatest to least, for histogram x-axis labels
    indices = np.argsort(lst)[::-1]

    # Reverse sort frequencies for the same reason
    lst.sort(reverse=True)

    # plot the frequencies
    plt.figure()
    plt.title("Feature frequencies")
    plt.bar(range(X.shape[1]), lst,
           color="r", align="center")
    plt.xticks(range(X.shape[1]), indices, rotation=30)
    plt.show()

def plot_feature_importances(X, y):
    '''
        Plot a histogram of important features given a dataset's X_train and
        y_train data, in reverse sorted order.
    '''

    # initiate the random forest with 250 trees
    forest = RandomForestClassifier(n_estimators=250)

    # train the forest on train dataset
    forest.fit(X, y)

    # get all the importances and their indices
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot the feature importances of the forest
    plt.figure()
    plt.title(f"Feature importances with {forest.__class__.__name__}")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", align="center")
    plt.xticks(range(X.shape[1]), indices, rotation=30)
    plt.xlim([-1, X.shape[1]])
    plt.show()
