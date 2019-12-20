import sys

import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import argparse
import csv
import random
import math

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectFromModel, VarianceThreshold

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve

import matplotlib.pyplot as plt

from cv_util import run_tune_test, show_foldwise_scores

from featurize import featurize_data
from visualize_data import plot_feature_frequencies, plot_feature_importances

import warnings
# suppress warnings
warnings.simplefilter(action='ignore')


def main():

    # Create parser to get raw csv filename argument
    parser = argparse.ArgumentParser(
        description='Create training dataset from OHLCV data.'
    )

    # Add argument to parser for filename
    parser.add_argument('--f', required=True,
        help='enter filename of CSV with data in following format: '
            + 'Date,Open,High,Low,Close,Volume,OpenInt. The file should be '
            + 'in the same directory.')

    # Get the actual argument
    args = parser.parse_args()
    filename = args.f

    # Dump csv data into numpy array
    raw_data = np.genfromtxt(args.f, dtype = float, delimiter=",",
     names=["Date", "Open", "High", "Low", "Close", "Volume", "Label"])

    # Generate labels
    for i in range(len(raw_data) - 1):

        # Assuming we're day trading here, i.e. buy the open, sell the close on the SAME day
        difference = raw_data[i+1]['Close'] - raw_data[i+1]['Open']
        raw_data[i]['Label'] = sign(difference)

    # remove header and first day of trading
    data = raw_data[2:]

    # featurize the data
    data = featurize_data(data)

    accuracies = []
    iterate = range(1, 100, 10)

    # for j in range(50):
    # for i in range(50):
    # X_train, y_train, X_test, y_test = split_dataset(data)
    X, y = data[:,:-1], data[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=0.3)
    # plot_feature_frequencies(X)
    # plot_feature_importances(X, y)
    # clf = AdaBoostClassifier()
    # params = {"n_estimators": [i for i in range(10, 200, 10)],}
    #         "max_features": [0.01, 0.1, 'sqrt']} # number of features
    # test_scores = run_tune_test(clf, params, X, y)
    # show_foldwise_scores(test_scores)

    # scores = []
    # iterate = range(1, 1000, 10)
    # for i in iterate:

    #
    #     model = RandomForestClassifier(n_estimators=i)
    #     model.fit(X_train, y_train)
    #     y_pred = model.predict(X_test)
    #     accuracy = accuracy_score(y_test, y_pred)
    #     scores.append(accuracy)



    # accuracy = accuracy_score(y_test, y_pred)
    # print(accuracy)

    clfs = [
        AdaBoostClassifier(n_estimators=100),
        RandomForestClassifier(n_estimators=100),
        MLPClassifier(learning_rate_init=0.0001,
                        learning_rate='constant',
                        solver='sgd',
                        max_iter=1000,
                        early_stopping=True)
    ]
    plt.clf()
    for clf in clfs:
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1, drop_intermediate=False)
        name = clf.__class__.__name__
        plt.plot(fpr, tpr, label=name)

    # plt.plot(iterate, scores, label="Accuracy of RandomForest")
    # plt.ylim((0, 1.1))
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve Comparison of Classifiers")
    plt.show()

def sign(num):
    '''
        If a number is positive, return 1. Else, return -1.
    '''
    return 1 if num > 0 else -1


if __name__ == '__main__':
    main()
