import matplotlib.pyplot as plt
import talib


def show_feature_frequencies(X):
    lst = [0] * len(X[0])
    for point in X:
        for i in range(len(point) - 1):
            if point[i] != 0:
                lst[i] += 1
    plt.figure()
    plt.title("Feature importances")
    fig = plt.bar(talib.get_function_groups()['Pattern Recognition'], lst,
           color="r", align="center")

    # fig.autofmt_xdate()
    # plt.xticks(range(X.shape[1]), lst)
    # plt.xlim([-1, X.shape[1]])
    plt.show()
