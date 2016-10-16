# -*- coding: utf-8 -*-

from numpy import *
from collections import namedtuple
from csv import *
import csv


def load_data(filename):
    data_array = []
    label_array = []
    col_types = [int, str, str, float, float, str, str, float, float, float, float, float, float, float, float, float]
    with open('houses-2016-10-13.csv') as f:
        f_csv = csv.reader(f)
        headings = next(f_csv)
        for row in f_csv:
            row = tuple(convert(value) for convert, value in zip(col_types, row))
            line_array = list(row)
            line_array[2] = row[2].split('-')[0]
            line_array[5] = float(filter(str.isdigit, line_array[5]))
            line_array[6] = float(line_array[6][0])
            temp_array = []
            for i in range(12):
                temp_array.append(line_array[i+4])
            data_array.append(temp_array)
            label_array.append(line_array[3])
    return data_array, label_array


#linear regression
def stand_regression(data_array, label_array):
    x_matrix = mat(data_array)
    y_matrix = mat(label_array).T
    xTx = x_matrix.T * x_matrix
    if linalg.det(xTx) == 0.0:
        print 'This matrix is singular, can not do inverse'
        return
    ws = xTx.I * (x_matrix.T * y_matrix)
    return ws





if __name__ == '__main__':
    data, label = load_data('houses-2016-10-13.csv')
    u = stand_regression(data, label)
    print u