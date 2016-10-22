import pyprind,os,csv,random
import numpy as np
import pandas as pd
import pyprind as pp
import sys,csv
import code0_parameter as code0

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


class setting(object):
    def __init__(self):
        tp = code0.DatasetParameter()
        self.DATASETSIZE        = tp.dataSetSize  # small:0.1m | middle : 1m | large : 5m+

        self.rawDataFile        = './data/SkillBuilderAndRemediationData.csv'
        self.processDataFile    = './data/processData_'+self.DATASETSIZE+'.csv'
        self.connectDataFile    = './data/connectData_'+self.DATASETSIZE+'.csv'
        self.convertTrainFile   = './data/skillBuild_train_'+self.DATASETSIZE+'.csv'
        self.convertTestFile    = './data/skillBuild_test_'+self.DATASETSIZE+'.csv'
        self.skillSetFile       = './data/skill_set_'+self.DATASETSIZE+'.csv'
        self.skillDictFile      = './data/skill_dict_'+self.DATASETSIZE+'.csv'


def get_process_data(st):
    if os.path.exists(st.processDataFile):
        data = pd.read_csv(st.processDataFile)
        print ("==> load processDataFile directly")
    else:
        try:
            data = pd.read_csv(st.rawDataFile)
        except:
            raise NameError("can't load " + st.rawDataFile + " pleace check your file")
        print (data.columns)
        data = data.astype(int)
        data = data[data['rem_assignment'] == 0]
        filtedColumnNameList = ['skill_id','user_id', 'correct']
        data = data[filtedColumnNameList].fillna(0)


        if st.DATASETSIZE =='small':
            data = data[0:100000]
        elif st.DATASETSIZE =='middle':
            data = data[0:1000000]
        else:
            pass
        #data.to_csv("./data/unprocess.csv",index=False)
        skill_set, skill_dict = create_column_dict_and_set(data=data,columnName='skill_id',st=st,order=True)

        for i in pyprind.prog_percent(range(len(data)),stream=sys.stdout,title='reorder skill_id to reduce max number'):
            data.loc[i, "skill_id"] = skill_dict[data.loc[i, "skill_id"]]
        print (data.columns)
        data.to_csv(st.processDataFile,index=False)
    return data

def get_connect_data(st):
    if os.path.exists(st.connectDataFile):
        data = pd.read_csv(st.connectDataFile)
        print ("==> load connectDataFile directly")
        return data
    else:
        data = get_process_data(st)
        u,c = counter(data['user_id'])

        userQuesNumIndexList = getUserQuesNumIndexList(data['user_id'])
        newdata = pd.DataFrame()

        for i in pyprind.prog_percent(range(len(u)),stream=sys.stdout,title='connect User'):
            for k in range(len(userQuesNumIndexList)):
                if userQuesNumIndexList[k,0] ==u[i]:
                    temp = data.iloc[int(userQuesNumIndexList[k,2]):int(userQuesNumIndexList[k,2]+userQuesNumIndexList[k,1])]
                    newdata = newdata.append(temp)
        newdata.reset_index(drop=True)
        newdata.to_csv(st.connectDataFile,index=False)
        print ('==> save the connect user file to: ',st.connectDataFile)
        return newdata

def get_convert_data(st):
    if os.path.exists(st.convertTrainFile) and os.path.exists(st.convertTestFile):
        print ('==> convert Train and Test data are ready')
        return
    else:
        data = get_connect_data(st)
        skillList = list(data['skill_id'])
        correctList = list(data['correct'])

        userMatrix = getUserQuesNumList(data['user_id'])
        tuple_rows = []
        startIndex = 0
        for i in pyprind.prog_percent(range(len(userMatrix)),stream=sys.stdout,title='convert data to siyuan code formate'):
            num = int(userMatrix[i][1])
            tup = ([num], list(skillList[startIndex:startIndex+num]), list(correctList[startIndex:startIndex+num]))
            tuple_rows.append(tup)
            startIndex += num

        random.shuffle(tuple_rows)
        print (len(tuple_rows))
        train = tuple_rows[:int(0.8*len(tuple_rows))]
        test  = tuple_rows[int(0.8*len(tuple_rows)):]
        print ('==> train user:\t',len(train),"\ttest user\t",len(test))

        with open(st.convertTrainFile, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(train)):
                writer.writerow(train[i][0])
                writer.writerow(train[i][1])
                writer.writerow(train[i][2])
        print ('==> save train data')
        with open(st.convertTestFile, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(test)):
                writer.writerow(test[i][0])
                writer.writerow(test[i][1])
                writer.writerow(test[i][2])
        print ('==> save test data')

if __name__ == "__main__":
    st = setting()
    get_convert_data(st)
