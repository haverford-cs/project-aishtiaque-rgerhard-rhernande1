import numpy as np
import talib


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
