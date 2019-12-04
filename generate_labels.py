'''
    Takes in a CSV with the following columns:

    =======================================
    Date,Open,High,Low,Close,Volume,OpenInt
    =======================================

    and assigns a label (-1 or 1) for each price point given whether the price
    point is higher or lower than the previous one.

    Outputs the labelled dataset to <filename>_with_labels.csv file.
'''


import argparse
import csv


def main():
    parser = argparse.ArgumentParser(
        description='Create training dataset from OHLCV data.'
    )
    parser.add_argument('--f', required=True,
        help='enter filename of CSV with data in following format: '
            + 'Date,Open,High,Low,Close,Volume,OpenInt. The file should be '
            + 'in the same directory.')

    args = parser.parse_args()
    filename = args.f

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        lst = []
        for row in reader:
            if not lst:
                lst.append(row)
            previous = lst[-1]
            price_change = float(row['Close']) - float(previous['Close'])
            row['label'] = sign(price_change)
            del row['OpenInt']
            lst.append(row)

    # we discard the first data point since it doesn't have a label.
    lst = lst[1:]

    # write the data points with labels to CSV
    with open(f"{filename}_with_labels.csv", 'w') as csvfile:
        fieldnames = ['Date', 'Open',
                'High', 'Low', 'Close', 'Volume', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in lst:
            writer.writerow(item)


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
