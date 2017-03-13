import sys
import csv
import numpy as np


# Returns (X, Y1, Y2)
# Where:
# X = the data set,
# Y1 = the time to eviction
# Y2 = whether or not an eviction happened
def load_input_file(input_file_name):
    with open(input_file_name, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        X = list()
        Y1 = list()
        Y2 = list()

        # Skip the header
        next(csv_reader, None)

        for row in csv_reader:
            X.append(map(lambda x: num(x), row[0:-2]))
            Y1.append(num(row[-2]))
            Y2.append(num(row[-1]))

        return np.array(X), np.array(Y1), np.array(Y2)


def load_input_file_modified(input_file_name):
    with open(input_file_name, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        X = list()
        Y1 = list()
        Y2 = list()

        # Skip the header
        next(csv_reader, None)

        for row in csv_reader:
            X.append(map(lambda x: num(x), row[0:1] + row[2:-6]))
            Y1.append(num(row[-2]))
            Y2.append(num(row[-1]))

        return np.array(X), np.array(Y1), np.array(Y2)


# Returns int or float from string depending on what's read in.
def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)
