import numpy as np
import pandas as pd
import pyprind as pp
import sys,csv

def counter(a):
    a = list(a)
    unique, counts = np.unique(a, return_counts=True)
    return unique, counts

def getUserQuesNumList(dataList):
    a = list(dataList)
    target = np.empty((0, 2))
    size = len(a)
    temp = [a[0], 1]
    for i in range(1, size):
        if a[i] == a[i - 1]:
            temp[1] += 1
        else:
            target = np.vstack((target, temp))
            temp = [a[i], 1]
    return np.vstack((target, temp))

def getUserQuesNumIndexList(dataList):
    a = list(dataList)
    target = np.empty((0, 3))
    size = len(a)
    temp = [a[0], 1, 0]
    for i in range(1, size):
        if a[i] == a[i - 1]:
            temp[1] += 1
        else:
            target = np.vstack((target, temp))
            temp = [a[i], 1 ,i]
    return np.vstack((target, temp))

def create_column_dict_and_set(data,columnName,st,order=True):
    column_ct = data[columnName]
    column_set_original = list(column_ct.unique())
    if order:
        column_set_original = sorted(column_set_original)
    size = len(column_set_original)
    column_dict = {value: key + 1 for key, value in enumerate(column_set_original)}
    column_dict[0] = 0
    column_set = [i + 1 for i in range(size)]

    with open(st.skillSetFile, 'w') as f:
        w = csv.writer(f)
        w.writerow(column_set)
    print ('==> save ',st.skillSetFile)
    with open(st.skillDictFile, 'w') as f:
        w = csv.writer(f)
        for key, val in column_dict.items():
            w.writerow([key, val])
    print ('==> save ',st.skillDictFile)
    return column_set, column_dict
