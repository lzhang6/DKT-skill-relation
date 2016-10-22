import numpy as np
import pandas as pd

#data_file       = './data/connectData_middle.csv'
correct_file    = "./result/correctdata_middle_result.csv"
uncorrect_file  = "./result/uncorrectdata_middle_result.csv"

#data        = pd.read_csv(data_file)
correctM    = pd.read_csv(correct_file)
uncorrectM  = pd.read_csv(uncorrect_file)

print (correctM - uncorrectM)

