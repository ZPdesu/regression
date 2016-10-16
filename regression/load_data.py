# -*- coding: utf-8 -*-

from numpy import *
from collections import namedtuple
from csv import *
import csv


def load_data(filename):
    data_matrix = []
    label_matrix = []
    col_types = [int, str, str, float, float, str, str, float, float, float, float, float, float, float, float, float]
    with open('houses-2016-10-13.csv') as f:
        f_csv = csv.reader(f)
        headings = next(f_csv)
        for row in f_csv:
            row = tuple(convert(value) for convert, value in zip(col_types, row))
            line_array = list(row)
            line_array[2] = row[2].split('-')[0]
            line_array[5] = float(filter(str.isdigit, line_array[5]))
            data_matrix.append(line_array[4:5])
            data_matrix.append(line_array[7:15])
            label_matrix.append(line_array[3])
    return data_matrix, label_matrix


if __name__ == '__main__':
    load_data('houses-2016-10-13.csv')