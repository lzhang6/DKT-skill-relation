import pyprind,os,csv,random,sys
import numpy as np
import pandas as pd
import aux
import code1_data as code1
import code0_parameter as code0

class simulateData(object):
    dataFileName = './data/sim_dataset.csv'
    labelFileName = './data/sim_label.csv'
    filename = './data/sim_problem_data.csv'

def loadSimulateData():
    dataset,labels = create_label_and_delete_last_one()
    return dataset,labels


def create_label_and_delete_last_one():
    sd = simulateData()
    dataFileName = sd.dataFileName
    labelFileName = sd.labelFileName
    filename = sd.filename

    if os.path.exists(dataFileName) and os.path.exists(labelFileName):
        dataset = pd.read_csv(dataFileName)
        labels = pd.read_csv(labelFileName)
        print('==> ', dataFileName, " exists,load directly")
        print('==> ', labelFileName, " exists,load directly")
        return dataset, labels

    data = pd.read_csv(filename)
    data.rename(columns={'Student': 'user_id','Correctness': 'correct'}, inplace=True)
    print (data.columns)

    userID_Quest_number_matrix = aux.getUserQuesNumList(data['user_id'])  # user_id: number of questions
    print("==> creat skill_id+label, last record of every user is deleted")
    print("==> delete user whose problem number is less than 2")
    row_size = len(data);
    index = 0
    kindex = 0
    dataset = pd.DataFrame()
    labels = pd.DataFrame()

    bar = pyprind.ProgPercent(row_size, stream=sys.stdout)
    while (index < row_size):
        id_number = userID_Quest_number_matrix[kindex, 1]
        if id_number > 2:
            dataTemp = data.loc[index:index + id_number - 2]
            labeTemp = pd.DataFrame({'user_id': int(data.loc[index, 'user_id']),
                                     'label_skill_id': data.loc[index + 1:index + id_number - 1, "skill_id"],
                                     'label_correct': data.loc[index + 1:index + id_number - 1, "correct"]})

            assert len(dataTemp) == len(labeTemp)
            dataset = dataset.append(dataTemp)
            labels = labels.append(labeTemp)
            del dataTemp, labeTemp
        bar.update(id_number)
        index += id_number
        kindex += 1
    dataset = dataset.reset_index(drop=True)
    labels = labels.reset_index(drop=True)

    dataset = add_cross_feature_to_dataset(dataset)
    if os.path.exists(dataFileName): os.remove(dataFileName)
    if os.path.exists(labelFileName): os.remove(labelFileName)
    dataset.to_csv(dataFileName,index=False)
    labels.to_csv(labelFileName,index=False)
    print("==> save ", dataFileName)
    print("==> save ", labelFileName)

    assert len(dataset) == len(labels), "dateset size\t" + str(len(dataset)) + "\tlabels size\t" + str(len(labels))
    return dataset, labels

def add_cross_feature_to_dataset(dataset):
    dp = code0.DatasetParameter()
    if len(dp.dataset_columns_for_cross_feature) == 0:
        print("==> no need to add cross feature to dataset")
        return dataset
    else:
        print("==> add cross feature to dataset")
        columns_max, columns_numb, _ = code1.get_columns_info(dataset)
        d_size = len(dataset)
        for item in dp.dataset_columns_for_cross_feature:
            print("==> add", aux.connectStringfromList(item))
            temp = []
            for i in pyprind.prog_percent(range(d_size),stream=sys.stdout, title=item):
                if len(item) == 2:
                    value = dataset.loc[i, item[0]] + dataset.loc[i, item[1]] * (columns_max[item[0]] + 1)
                elif len(item) == 3:
                    value = dataset.loc[i, item[0]] + dataset.loc[i, item[1]] * (columns_max[item[0]] + 1) + \
                            dataset.loc[i, item[2]] * (columns_max[item[0]] + 1) * (columns_max[item[1]] + 1)
                else:
                    raise ValueError('cross features only support 3 at most')
                temp.append(value)
            dataset[aux.connectStringfromList(item)] = temp
        return dataset

def runEpoch(session, m, students, eval_op,verbose=False):
    pred_prob = []
    actual_labels = []                   # use for whole comparasion
    iteration = int(len(students)/m.batch_size)

    for i_iter in pyprind.prog_percent(range(iteration)):
        #bar.update(m.batch_size)
        x = np.zeros((m.batch_size, m.num_steps, m.seq_width))

        target_id = np.array([],dtype=np.int32)
        target_correctness = []         # use for just a batch

        #load data for a batch
        # tuple formate
        # 0: user_id
        # 1: record_numb
        # 2: data
        # 3: Target_Id
        # 4: correctness
        for i_batch in range(m.batch_size):
            student = students[i_iter*m.batch_size+i_batch]
            record_num = student[1]
            #record_content_pd = student[2].reset_index(drop=True)
            record_content = student[2].as_matrix()
            skill_id = student[3]
            correctness = student[4]

            # construct data for training:
            # data ~ x
            # target_id ~ skill_id
            # target_correctness ~ correctness
            for i_recordNumb in range(record_num):
                if(i_recordNumb<m.num_steps):
                    x[i_batch, i_recordNumb,:] = record_content[i_recordNumb,:]

                    if skill_id[i_recordNumb] in m.skill_set:
                        temp =i_batch*m.num_steps*m.skill_num + i_recordNumb*m.skill_num + skill_id[i_recordNumb]
                    else:
                        temp = i_batch*m.num_steps + i_recordNumb*m.skill_num + 0
                    target_id = np.append(target_id,[[temp]])
                    target_correctness.append(int(correctness[i_recordNumb]))
                    actual_labels.append(int(correctness[i_recordNumb]))
                else:
                    break
        pred, _ = session.run([m.pred, eval_op],feed_dict={m.inputs: x,
                                                           m.target_id: target_id,
                                                           m.target_correctness: target_correctness})

        for p in pred:
            pred_prob.append(p)

    return pred_prob


if __name__=="__main__":
    loadSimulateData()
