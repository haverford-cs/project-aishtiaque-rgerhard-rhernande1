import csv
from datetime import date, timedelta
import argparse
# import tensorflow as tf


class Stock:
    '''

    '''
    def __init__(self, data):
        self.data = data
        self.open = data['Open']
        self.high = data['High']
        self.low = data['Low']
        self.close = data['Close']
        self.volume = data['Volume']
        self.labels = data['Label']

def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))

# def main():
#     dataset = tf.data.experimental.make_csv_dataset('aapl.us.csv_with_labels.csv', batch_size=5, label_name='label')
#     show_batch(dataset)
    # parser = argparse.ArgumentParser(
    #     description='Find the price of a stock on the next stock day.'
    # )
    # parser.add_argument('--name', required=True,
    #     help='Ticker symbol for the stock you want')
    # parser.add_argument('--date', required=True,
    #     help='Enter date to predict in ISO format')
    #
    #
    # args = parser.parse_args()
    # name = args.name
    # date = args.date
    # with open("aapl.us.csv", "r") as f:
    #     reader = csv.DictReader(f)
    #     lst = []
    #     for row in reader:
    #         lst.append(row)
    # dataset = Stock(lst, date)
    # print(dataset.moving_avg(1))


#main()
