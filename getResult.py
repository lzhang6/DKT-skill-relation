from __future__ import print_function

import code0_parameter as code0
import code1_data as code1
import code2_model as code2
import code3_runEpoch as code3
import aux,pyprind
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime

def getBatchSize(num):
    for i in [2,3,5,7,9,13]:
        if num%i==0:
            return i
    raise ('check the batch size of ',str(num))


def constructData(dp):
    skillSet = list(pd.read_csv('./data/skill_set_'+dp.dataSetSize+'.csv'))
    data_correct = pd.DataFrame({
        'user_id':skillSet,
        'skill_id':skillSet,
        'correct':1
    })
    data_uncorrect = pd.DataFrame({
        'user_id':skillSet,
        'skill_id':skillSet,
        'correct':0
    })
    data_correct = data_correct.astype(np.int32)
    data_uncorrect = data_uncorrect.astype(np.int32)

    data_correct = code1.add_cross_feature_to_dataset(data_correct,dp)
    data_uncorrect = code1.add_cross_feature_to_dataset(data_uncorrect,dp)
    print (data_correct.head())
    print (data_uncorrect.head())
    return data_correct,data_uncorrect

def runEpoch(session, m, students, eval_op,verbose=False):
    iteration = int(len(students)/m.batch_size)
    #for i_iter in pyprind.prog_percent(range(iteration)):
    for i_iter in range(iteration):
        x = np.zeros((m.batch_size, m.num_steps, m.seq_width))
        for i_batch in range(m.batch_size):
            student = students.loc[i_iter*m.batch_size+i_batch:i_iter*m.batch_size+i_batch]
            x[i_batch, 0,:] = student.as_matrix()
        #print ("-"*15,i_iter,"-"*15,np.shape(x),"-"*15)
        rslt,_ = session.run([m.rslt,eval_op],feed_dict={m.inputs: x})
        if i_iter == 0:
            rslts = rslt
        else:
            rslts = np.vstack((rslts,rslt))
    return rslts

def main(unused_args):
    dp = code0.DatasetParameter()
    data_correct,data_uncorrect = constructData(dp)
    skill_num = len(data_correct['skill_id'].unique()) + 1  # 0 for unlisted skill_id
    dp.skill_num = skill_num
    dp.skill_set = list(data_correct['skill_id'].unique())
    dp.columns_max, dp.columns_numb, dp.columnsName_to_index = code1.get_columns_info(data_correct)
    dp.seq_width = len(dp.columnsName_to_index)

    print("-" * 50, "\ndp.columns_max\n", dp.columns_max, "\n")
    print("-" * 50, "\ndp.columns_numb\n", dp.columns_numb, "\n")
    print("-" * 50, "\ndp.columnsName_to_index\n", dp.columnsName_to_index, "\n")

    checkpoint_file='skillBuilder.chk'

    run_config = code0.ModelParamsConfig(dp)
    run_config.num_steps = 1
    run_config.batch_size = getBatchSize(len(data_correct))
    run_config.skill_num = skill_num

    with tf.Graph().as_default(), tf.Session() as session:
        print("\n==> Load Testing model")
        saver = tf.train.Saver(tf.all_variables())

        with tf.variable_scope("model", reuse=True):
            mrun = code2.Model(is_training=False, config=run_config, dp=dp)
            saver.restore(session, checkpoint_file)
            session.run(tf.all_variables())
        rslt = runEpoch(session,mrun, data_correct, tf.no_op())
        print (rslt)

def runEpochTest(students):
    pred_prob = []
    actual_labels = []
    iteration = int(len(students)/10)

    for i_iter in pyprind.prog_percent(range(iteration)):
        x = np.zeros((10, 1, 4))
        for i_batch in range(10):
            student = students.loc[i_iter*10+i_batch:i_iter*10+i_batch]
            x[i_batch, 0,:] = student.as_matrix()

        print (x)
        print (np.shape(x))


if __name__=="__main__":
    dp = code0.DatasetParameter()
    data_correct,data_uncorrect = constructData(dp)
    #runEpochTest(data_correct)
    print (data_correct[:10])
