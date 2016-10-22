import pandas as pd
import numpy as np

def processResult(data):
    print (np.shape(data))
    if np.shape(data)[0]!=np.shape(data)[1]:
        data = data.iloc[:,2:]

    print (np.shape(data))
    tmp = (pd.read_csv('./data/skill_dict_middle.csv',header=None)).as_matrix()
    skill_dict = {sd[1]:sd[0] for sd in tmp}
    columnName = [ skill_dict[i] for i in range(0,len(skill_dict))]
    columnName.remove(0)
    newData = pd.DataFrame(data.as_matrix(),columns=columnName,index=columnName)
    return newData

if __name__ == "__main__":
    correctdata = pd.read_csv('./result/correct_middle_1.csv')
    correctdata = processResult(correctdata)
    correctdata.to_csv('./result/correctdata_middle_result.csv')

    uncorrectdata = pd.read_csv('./result/uncorrect_middle_0.csv')
    uncorrectdata = processResult(uncorrectdata)
    uncorrectdata.to_csv('./result/uncorrectdata_middle_result.csv')
