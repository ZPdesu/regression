# -*- coding: utf-8 -*-

from numpy import *
from collections import namedtuple
from csv import *
import matplotlib.pyplot as plt
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


def normalization(data_array, label_array):
    x_matrix = mat(data_array)
    y_matrix = mat(label_array).T
    y_mean = mean(y_matrix, 0)
    y_matrix = y_matrix - y_mean
    x_mean = mean(x_matrix, 0)
    x_var = var(x_matrix, 0)
    x_matrix = (x_matrix - x_mean) / x_var
    return x_matrix, y_matrix


def rss_error(y_array, y_hatarray):
    return ((y_array - y_hatarray) ** 2).sum()


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
        weights[j, j] = exp((diff_matrix * diff_matrix.T / (-2.0 * k ** 2)))
    xTx = x_matrix.T * (weights * x_matrix)
    if linalg.det(xTx) == 0.0:
        print 'This matrix is singular, can not do inverse'
        return
    ws = xTx.I * (x_matrix.T * (weights * y_matrix))
    return testpoint * ws


#ridge_regression
def ridge_regression(x_matrix, y_matrix, lam=0.2):
    xTx = x_matrix.T * x_matrix
    denom = xTx + eye(shape(x_matrix)[1]) * lam
    if linalg.det(denom) == 0.0:
        print 'This matrix is singular, can not do inverse'
        return
    ws = denom.I * (x_matrix.T * y_matrix)
    return ws


def ridge_test(data_array, label_array):
    x_matrix, y_matrix = normalization(data_array, label_array)
    num_testpts = 30
    ws_matrix = zeros((num_testpts, shape(x_matrix)[1]))
    for i in range(num_testpts):
        ws = ridge_regression(x_matrix, y_matrix, exp(i-10))
        ws_matrix[i, :] = ws.T
        print ws.T
    return ws_matrix


def stage_wise(data_array, label_array, num_it=100):
    x_matrix, y_matrix = normalization(data_array, label_array)
    m, n = shape(x_matrix)
    return_matrix = zeros((num_it, n))
    ws = zeros((n, 1))
    ws_test = ws.copy()
    ws_max = ws.copy()
    eps = 1e-6
    for i in range(num_it):
        print ws.T
        lowest_error = inf
        for j in range(n):
            for sign in [-1, 1]:
                ws_test = ws.copy()
                ws_test[j] += eps * sign
                y_test = x_matrix * ws_test
                rssE = rss_error(y_matrix.A, y_test.A)
                if rssE < lowest_error:
                    lowest_error = rssE
                    ws_max = ws_test
            ws = ws_max.copy()
            return_matrix[i, :] = ws.T
        return return_matrix




def savefile(data):
    with open(' std_re.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    data, label = load_data('houses-2016-10-13.csv')
    # linear regression
    #a = stand_regression(data, label)
    #savefile(a.tolist())

    # lwl regression
    #b = lwl_regression(data[3], data, label, k = 100)

    # ridge_regression
    '''
    ridge_weights = ridge_test(data, label)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridge_weights)
    plt.show()
    '''

    print 1