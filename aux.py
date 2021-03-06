import numpy as np
import os,sys,csv
from code0_parameter import DATASETSIZE, CELLTYPE
import tensorflow as tf
import pandas as pd

def counter(a):
    a = list(a)
    unique, counts = np.unique(a, return_counts=True)
    return unique, counts

def create_column_dict_and_set(data,columnName,dp):
    setName = os.path.dirname(dp.csv_file_name)+"/"+columnName+"_set_"+ str(dp.dataSetSize)+".csv"
    dictName = os.path.dirname(dp.csv_file_name)+"/"+columnName+"_dict_"+ str(dp.dataSetSize)+".csv"
    column_ct = data[columnName]
    column_set_original = list(column_ct.unique())
    size = len(column_set_original)
    column_dict = {value: key + 1 for key, value in enumerate(column_set_original)}
    column_dict[0] = 0
    column_set = [i + 1 for i in range(size)]

    with open(setName, 'w') as f:
        w = csv.writer(f)
        w.writerow(column_set)
    print ('==> save ',setName)
    with open(dictName, 'w') as f:
        w = csv.writer(f)
        for key, val in column_dict.items():
            w.writerow([key, val])
    print ('==> save ',dictName)
    return column_set, column_dict

def stastic_SecNumber_UserNumber_SkillNumber(data,dp):
    secNumber = len(getUserQuesNumList(data['user_id']))
    userNumber = len(data['user_id'].unique())
    skillNumber = len(data['skill_id'].unique())

    secNumberStr   = "SecNumber         {:>10}\n".format(secNumber)
    userNumberStr  = "userNumber        {:>10}\n".format(userNumber)
    skillNumberStr = "skillNumber       {:>10}\n".format(skillNumber)

    logwrite([secNumberStr,userNumberStr,skillNumberStr],dp,True)
    return secNumber,userNumber,skillNumber

def mean_normalization(X_train, X_test):
    data = np.concatenate((X_train, X_test), axis=0)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std


def xavier_init(fan_in, fan_out, function):
    if function is tf.nn.sigmoid:
        low = -4.0 * np.sqrt(6.0 / (fan_in + fan_out))
        high = 4.0 * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
    elif function is tf.nn.tanh:
        low = -1 * np.sqrt(6.0 / (fan_in + fan_out))
        high = 1 * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


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

def connectStringfromList(klist):
    if type(klist)!=list:
        raise ValueError("only convert list")
    tmp = ''
    for i, v in enumerate(klist):
        if i == 0:
            tmp = klist[i]
        else:
            tmp = tmp + " " + klist[i]
    return tmp

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def get_num_step(dataset):
    u, c = counter(dataset['user_id'])
    return max(c)

def logwrite(strList,dp,prt=False):
    logfileName = "result/log_"+str(dp.dataSetType)+"_"+ str(dp.currentTime) + "_" + str(CELLTYPE) + "_" + str(DATASETSIZE) + ".txt"

    for item in strList:
        with open(logfileName, "a") as myfile:
            myfile.write(str(item))
        if prt:
            print(item)

def printConfigration(config, dp, train_numb, test_numb):
    l1 = "\n"+"-" * 15+" Configuration "+"-" * 15
    l11= "DataSet             {:>10}".format(dp.dataSetType)
    l2 = "RNN layers          {:>10}".format(config.num_layer)
    l3 = "cell type           {:>10}".format(config.cell_type)
    l4 = "hidden_size         {:>10}".format(config.hidden_size)

    if config.num_layer == 2:
        l41 = "hidden_size2        {:>10}".format(config.hidden_size_2)
        logwrite([l1,l11,l2,l3,l4,l41],dp=dp,prt=True)
    else:
        logwrite([l1,l11,l2,l3,l4],dp=dp,prt=True)
    l5 = "keep_prob           {:>10}".format(config.keep_prob)
    l6 = "num_steps           {:>10}".format(config.num_steps)
    l7 = "seq_width           {:>10}".format(len(dp.columnsName_to_index))
    l8 = "skill_num           {:>10}".format(config.skill_num)
    l9 = "skill_id_one_hot    {:>10}".format(dp.columns_max['skill_id'] + 1)
    l10 = "max_max_epoch       {:>10}".format(config.max_max_epoch)
    l11 = "batch_size          {:>10}".format(config.batch_size)
    l12 = "train student number{:>10}".format(train_numb)
    l13 = "test student number {:>10}".format(test_numb)
    l14 = "-" * 20+" End "+"-" * 20+"\n"
    logwrite([l5,l6,l7,l8,l9,l10,l11,l12,l13,l14],dp=dp,prt=True)

def defineResult():
    auc_train = pd.DataFrame({"c1": [], "c2": [], "c3": [], "c4": [], "c5": []}, dtype="float32")
    r2_train = pd.DataFrame({"c1": [], "c2": [], "c3": [], "c4": [], "c5": []}, dtype="float32")
    rmse_train = pd.DataFrame({"c1": [], "c2": [], "c3": [], "c4": [], "c5": []}, dtype="float32")

    auc_test = pd.DataFrame({"c1": [], "c2": [], "c3": [], "c4": [], "c5": []}, dtype="float32")
    r2_test = pd.DataFrame({"c1": [], "c2": [], "c3": [], "c4": [], "c5": []}, dtype="float32")
    rmse_test = pd.DataFrame({"c1": [], "c2": [], "c3": [], "c4": [], "c5": []}, dtype="float32")
    return auc_train,r2_train,rmse_train,auc_test,r2_test,rmse_test

def saveResult(dp,auc_train,rmse_train,r2_train,auc_test,rmse_test,r2_test,mean_result):
    print("==> save the result\t", str(dp.currentTime))
    auc_train.to_csv("result/"+str(dp.dataSetType)+"/auc_train_" + str(dp.currentTime) + ".csv")
    rmse_train.to_csv(
        "result/rmse_train_" + str(dp.currentTime) +  ".csv")
    r2_train.to_csv("result/"+str(dp.dataSetType)+"/r2_train_" + str(dp.currentTime) + ".csv")

    auc_test.to_csv("result/"+str(dp.dataSetType)+"/auc_test_" + str(dp.currentTime) + ".csv")
    rmse_test.to_csv("result/"+str(dp.dataSetType)+"/rmse_test_" + str(dp.currentTime) + ".csv")
    r2_test.to_csv("result/"+str(dp.dataSetType)+"/r2_test_" + str(dp.currentTime) + ".csv")

    mean_result.to_csv("result/"+str(dp.dataSetType)+"/Mean_" + str(dp.currentTime) + ".csv")
if __name__ == "__main__":
    pass
