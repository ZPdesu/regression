# -*- coding: utf-8 -*-

from numpy import *
from collections import namedtuple
from csv import *
import csv
import json


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


#lwl regression
def lwl_regression(testpoint, data_array, label_array, k = 0.5):
    x_matrix = mat(data_array)
    y_matrix = mat(label_array).T
    m = shape(x_matrix)[0]
    weights = mat(eye(m))
    for j in range(m):
        diff_matrix = testpoint - x_matrix[j, :]
        #print diff_matrix
        weights[j, j] = exp((diff_matrix * diff_matrix.T / (-2.0 * k ** 2)))
        print exp((diff_matrix * diff_matrix.T/ (-2.0 * k ** 2))[0,0])
    xTx = x_matrix.T * (weights * x_matrix)
    if linalg.det(xTx) == 0.0:
        print 'This matrix is singular, can not do inverse'
        return
    ws = xTx.I * (x_matrix.T * (weights * y_matrix))
    return testpoint * ws


def savefile(data):
    with open(' std_re.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    data, label = load_data('houses-2016-10-13.csv')
    # linear regression
    #a = stand_regression(data, label)
    #savefile(a.tolist())

    #lwl regression
    b = lwl_regression(data[8], data, label, k = 1000)

    print b