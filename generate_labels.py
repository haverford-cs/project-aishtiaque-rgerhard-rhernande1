import numpy as np
import argparse
import csv
from rnd import Stock
import talib
import random
import math

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, VarianceThreshold

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve

import matplotlib.pyplot as plt

from cv_util import run_tune_test, show_foldwise_scores

import warnings
# suppress warnings
warnings.simplefilter(action='ignore')


def featurize_data(data):
    '''
        Takes in a CSV with the following columns:

        =======================================
        Date,Open,High,Low,Close,Volume,OpenInt
        =======================================

        and assigns a label (-1 or 1) for each price point given whether the price
        point is higher or lower than the previous one.
    '''
    # Initialize output array
    output = np.ones((len(data), 1))

    # Iterate over function to match candlestick patterns
    for func in talib.get_function_groups()['Pattern Recognition']:
        # Interpret strings as functions and call them on numeric data
        func_to_call = getattr(talib, func)
        feature_column = func_to_call(data['Open'], data['High'], data['Low'],
                                        data['Close']).reshape(len(data), 1)

        # Column join indicator output for candlestick pattern features
        output = np.concatenate((output, feature_column), axis=1)

    # Concatenate labels to final dataframe
    output = np.concatenate((
                output,
                data['Label'].reshape(len(data), 1)),
            axis=1)
    output = output[:,1:]

    return output


def split_dataset(data):
    '''
        Shuffles a given dataset and returns train and test subsets.

        Reference from: https://stackoverflow.com/questions/3674409/
        how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros
    '''
    # get randomized indices to divide the data into
    indices = np.random.permutation(data.shape[0])

    # split the indices into test and train subsets with an 80-20 split
    split_idx = math.floor(0.8*len(data))
    training_idx, test_idx = indices[:split_idx], indices[split_idx:]

    # get the training and testing partitions
    training, test = data[training_idx,:], data[test_idx,:]

    # return train_X, train_y (labels), test_X, test_y (labels)
    return training[:,:-1], training[:,-1], test[:,:-1], test[:,-1]


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
    # print(data_with_features.shape)
    # print(train_X.shape)
    # print(train_y.shape)
    # print(test_X.shape)
    # print(test_y.shape)

    accuracies = []
    iterate = range(1, 100, 10)

    # for j in range(50):
    # for i in range(50):
    X_train, y_train, X_test, y_test = split_dataset(data)
    X, y = data[:,:-1], data[:,-1]
    clf = RandomForestClassifier()
    params = {"n_estimators": [i for i in range(10, 200, 10)],
            "max_features": [0.01, 0.1, 'sqrt']} # number of features
    test_scores = run_tune_test(clf, params, X, y)
    show_foldwise_scores(test_scores)






    # sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    # sel = sel.fit(train_X)
    # train_X = sel.transform(train_X)
    # test_X = sel.transform(test_X)
    #
    # model = RandomForestClassifier(n_estimators=100)
    # model.fit(train_X, train_y)
    # y_pred = model.predict(test_X)
    # print(accuracy_score(test_y, y_pred))

    # clfs = [
    #     AdaBoostClassifier(n_estimators=10),
    #     RandomForestClassifier(n_estimators=10),
    #     MLPClassifier(learning_rate_init=0.001,
    #                     learning_rate='constant',
    #                     solver='sgd',
    #                     max_iter=1000,
    #                     early_stopping=True)
    # ]
    # plt.clf()
    # for clf in clfs:
    #     clf = clf.fit(train_X, train_y)
    #     y_pred = clf.predict(test_X)
    #     fpr, tpr, _ = roc_curve(test_y, y_pred, pos_label=1)
    #     name = clf.__class__.__name__
    #     plt.plot(fpr, tpr, label=name)
    #
    # plt.ylim((0, 1.1))
    # plt.legend()
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Comparison of ROC Curves")
    # plt.show()

def sign(num):
    '''
        If a number is positive, return 1. Else, return -1.
    '''
    return 1 if num > 0 else -1


if __name__ == '__main__':
    main()
