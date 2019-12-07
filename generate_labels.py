'''

'''
import numpy as np
import argparse
import csv
from rnd import Stock
import talib
import random
import math

from sklearn.ensemble import RandomForestClassifier


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
    for i,row in enumerate(raw_data):
        if i <= 1:
            continue
        difference = raw_data[i]['Close'] - raw_data[i-1]['Close']
        raw_data[i]['Label'] = sign(difference)

    # remove header and first day of trading
    data = raw_data[2:]

    # featurize the data
    data_with_features = featurize_data(data)
    train_X, train_y, test_X, test_y = split_dataset(data_with_features)
    # print(data_with_features.shape)
    # print(train_X.shape)
    # print(train_y.shape)
    # print(test_X.shape)
    # print(test_y.shape)

    # create and fit a random forest classifier for the given training dataset
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    correct = 0

    # get accuracy
    for i in range(len(y_pred)):
        if y_pred[i] == test_y[i]:
            correct += 1
    print(f"Accuracy: {(correct/len(y_pred)) * 100}%")


def sign(num):
    '''
        If a number is positive, return 1. Else, return -1.
    '''
    return 1 if num > 0 else -1


if __name__ == '__main__':
    main()
