'''
    Takes in a CSV with the following columns:

    =======================================
    Date,Open,High,Low,Close,Volume,OpenInt
    =======================================

    and assigns a label (-1 or 1) for each price point given whether the price
    point is higher or lower than the previous one.

    Outputs the labelled dataset to <filename>_with_labels.csv file.
'''
import numpy as np
import argparse
import csv
from rnd import Stock
import talib

def featurize_data(data):
    # Initialize output array
    output = np.ones((len(data), 1))

    # Iterate over function to match candlestick patterns
    for func in talib.get_function_groups()['Pattern Recognition']:
        # Interpret strings as functions and call them on numeric data
        func_to_call = getattr(talib, func)
        feature_column = func_to_call(data['Open'], data['High'], data['Low'], data['Close']).reshape(len(data),1)

        # Column join indicator output for candlestick pattern features
        output = np.concatenate((output, feature_column), axis=1)

    # Concatenate labels to final dataframe
    output = np.concatenate((output, data['Label'].reshape(len(data), 1)), axis=1)
    output = output[:,1:]

    for row in output:
        print(row)

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
    featurized_data = featurize_data(data)
    #
    # return Stock(featurized_data)

def sign(num):
    '''
        If a number is positive, return 1. Else, return -1.
    '''
    if num > 0:
        return 1
    else:
        return -1

if __name__ == '__main__':
    main()
