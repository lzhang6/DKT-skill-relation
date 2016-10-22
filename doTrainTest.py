""" Code of deep knowledge tracing-assistment 2014-2015 dataset
Reference:
    1. https://github.com/siyuanzhao/2016-EDM/
    2. https://www.tensorflow.org/versions/0.6.0/tutorials/recurrent/index.html
    3. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/ptb_word_lm.py
    4. https://github.com/Cospel/rbm-ae-tf

Run code:
    1. only set the hyperparameter in code0_params.py
    2. train your autoencoder parameters
       python trainWeights.py
    3. python doAll.py

Environment:
    1. ubuntu 14.04
    2. python3
    3. tensorflow : 0.10
    4. cuda 7.5
    5. GPU GTX1070 (8G)
    6. CPU i5-6600k
    7. RAM: 16G
"""
from __future__ import print_function

import code0_parameter as code0
import code1_data as code1
import code2_model as code2
import code3_runEpoch as code3
import aux
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import getResult

np.set_printoptions(threshold=np.inf)

def main(unused_args):
    dp = code0.DatasetParameter()
    dataset, labels = code1.load_data(dp)
    tuple_data = code1.convert_data_labels_to_tuples(dataset, labels)

    kp = dataset['skill_id'].unique()
    skill_num = max(kp) + 1  # 0 for unlisted skill_id
    dp.skill_num = skill_num
    dp.skill_set = list(dataset['skill_id'].unique())
    dp.columns_max, dp.columns_numb, dp.columnsName_to_index = code1.get_columns_info(dataset)
    dp.seq_width = len(dp.columnsName_to_index)

    print("-" * 50, "\ndp.columns_max\n", dp.columns_max, "\n")
    print("-" * 50, "\ndp.columns_numb\n", dp.columns_numb, "\n")
    print("-" * 50, "\ndp.columnsName_to_index\n", dp.columnsName_to_index, "\n")

    config = code0.ModelParamsConfig(dp)
    eval_config = code0.ModelParamsConfig(dp)

    #checkpoint_file='skillBuilder.chk'

    if dp.dataSetType=='kdd':
        config.num_steps = 2000
    else:
        config.num_steps = aux.get_num_step(dataset)

    eval_config.num_steps = config.num_steps
    eval_config.batch_size = 2

    config.skill_num = skill_num
    eval_config.skill_num = config.skill_num

    ##################################################################################
    rdp = code0.DatasetParameter()
    data_correct,data_uncorrect = getResult.constructData(rdp)
    rdp.skill_num = skill_num
    rdp.skill_set = dp.skill_set
    rdp.columns_max, rdp.columns_numb, rdp.columnsName_to_index = code1.get_columns_info(data_correct)
    rdp.seq_width = len(rdp.columnsName_to_index)

    run_config = code0.ModelParamsConfig(rdp)
    run_config.num_steps = config.num_steps
    run_config.batch_size = getResult.getBatchSize(len(data_correct))
    run_config.skill_num = skill_num
    ##################################################################################

    #auc_train,r2_train,rmse_train,auc_test,r2_test,rmse_test = aux.defineResult()
    CVname = ['c1']#auc_test.columns
    print (CVname)
    size = len(tuple_data)

    # write all the records to log file
    aux.printConfigration(config=config, dp=dp, train_numb=int(size * 0.8), test_numb=int(size * 0.2))
    str_cross_columns_list = ['-'.join(i) for i in dp.model_cross_columns]
    str_cross_columns = ','.join(str_cross_columns_list)
    aux.logwrite(["==> model_cross_columns\n" + str_cross_columns], dp,True)

    for index, cv_num_name in enumerate(CVname):
        aux.logwrite(["\nCross-validation: \t" + str(index + 1) + "/5"], dp,prt=True)
        timeStampe = datetime.datetime.now().strftime("%m-%d-%H:%M")
        aux.logwrite(["\ntime:\t" + timeStampe], dp)

        train_tuple_rows = tuple_data[:int(index * 0.2 * size)] + tuple_data[int((index + 1) * 0.2 * size):]
        test_tuple_rows = tuple_data[int(index * 0.2 * size): int((index + 1) * 0.2 * size)]

        with tf.Graph().as_default(), tf.Session() as session:
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            # training model
            print("\n==> Load Training model")
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = code2.Model(is_training=True, config=config, dp=dp)
            # testing model
            print("\n==> Load Testing model")
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mtest = code2.Model(is_training=False, config=eval_config, dp=dp)

            print("\n==> Load Running model")
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mrun = code2.Model(is_training=False, config=run_config, dp=rdp)

            tf.initialize_all_variables().run()

            saver = tf.train.Saver(tf.all_variables())

            print("==> begin to run epoch...")
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                rt = session.run(m.lr)
                rmse, auc, r2 = code3.run_epoch(session, m, train_tuple_rows, m.train_op, verbose=True)
                train_result = "==> %s cross-valuation: Train Epoch: %d\tLearning rate: %.3f\t rmse: %.3f \t auc: %.3f \t r2: %.3f" % (
                    cv_num_name, i + 1, rt, rmse, auc, r2)
                print(train_result)

                aux.logwrite('\n'+train_result, dp,False)

                display=5
                if ((i + 1) % display == 0):
                    print("-" * 80)
                    rmse, auc, r2 = code3.run_epoch(session, mtest, test_tuple_rows, tf.no_op())
                    test_result = "==> %s cross-valuation: Test Epoch: %d \t rmse: %.3f \t auc: %.3f \t r2: %.3f" % (
                        cv_num_name, (i + 1) / display, rmse, auc, r2)
                    print(test_result)

                    if (i+1==config.max_max_epoch):
                        #data_correct,data_uncorrect = getResult.constructData(dp)
                        rslt = getResult.runEpoch(session,mrun,data_correct,tf.no_op())
                        rslt2 = getResult.runEpoch(session,mrun,data_uncorrect,tf.no_op())

                        print (np.shape(rslt))
                        print (np.shape(rslt2))

                        """
                        rslt = np.reshape(rslt,(-1,np.shape(rslt)[-1]))
                        rslt2 = np.reshape(rslt,(-1,np.shape(rslt2)[-1]))
                        print ('==> after reshape')
                        print (np.shape(rslt))
                        print (np.shape(rslt2))
                        """
                        rsltF = pd.DataFrame(rslt)
                        rsltF2 = pd.DataFrame(rslt2)
                        rsltF.to_csv("./result/correct_"+dp.dataSetSize+"_1.csv")
                        rsltF2.to_csv("./result/uncorrect_"+dp.dataSetSize+"_0.csv")

                    print("=" * 80)
                    #aux.logwrite('\n'+test_result, dp,False)
    print("==> Finsih! whole process, save result and print\t" + dp.currentTime)

if __name__ == "__main__":
    tf.app.run()


