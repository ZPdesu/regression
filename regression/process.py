# -*- coding: utf-8 -*-
from numpy import *
import codecs

def a():
    return 3

if __name__ == '__main__':
    a = matrix([[89,56],[23,56]])
    b = a.tolist()
    b[0][0] = 0
    print eye(5)