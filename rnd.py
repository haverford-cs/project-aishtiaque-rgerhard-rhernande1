import csv
from datetime import date, timedelta
import argparse
import tensorflow as tf


class Stock:
    '''
        Example input:
        [OrderedDict([('Date', '1984-09-07'), ('Open', '0.42388'),
        ('High', '0.42902'), ('Low', '0.41874'), ('Close', '0.42388'),
        ('Volume', '23220030'), ('OpenInt', '0')])]
    '''
    def __init__(self, data, current_date):
        self.data = {item['Date']: Day(item, i) for i,item in enumerate(data)}
        if current_date not in self.data:
            raise ArgumentError("Please enter a valid date within the date range.")
        self.current_date = current_date

    def moving_avg(self, days):
        """
            Takes in the number of days to compute the moving average over.

            Finds the index of self.current_date, then calculates the
            sum of closing prices from current_date - days to current_date.

            Returns the average if exists.
        """
        if days-1 > self.data[self.current_date].index:
            return 0
        else:
            running_total = 0
            start_index = self.data[self.current_date].index
            for i in range(start_index, start_index-days-1, -1):
                j = date.fromisoformat(self.current_date) - timedelta(days=i)
                j = j.isoformat()
                running_total += float(self.data[j].close)
            return running_total/days

    # def three_crows(self, moving_avg):
    #     if self.current_date

class Day:
    def __init__(self, d, index):
        self.date = d['Date']
        self.open = d['Open']
        self.high = d['High']
        self.low = d['Low']
        self.close = d['Close']
        self.volume = d['Volume']
        self.index = index

    def __repr__(self):
        output = f"<Date: {self.date}, Index: {self.index}>"
        return output

def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))


def main():
    dataset = tf.data.experimental.make_csv_dataset('aapl.us.csv_with_labels.csv', batch_size=5, label_name='label')
    show_batch(dataset)
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


main()
