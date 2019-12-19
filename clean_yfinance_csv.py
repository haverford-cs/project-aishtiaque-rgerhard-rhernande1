import csv

def clean_yfinance_csv(filename):
    lst = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            lst.append(row)
    with open(f"{filename}.output", "w") as file:
        writer = csv.writer(file)
        for i in range(len(lst)):
            if i == 0:
                lst[i].remove('Adj Close')
                lst[i].append("Label")
                writer.writerow(lst[i])
            else:
                del lst[i][4]
                lst[i].append(0)
                writer.writerow(lst[i])
