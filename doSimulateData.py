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
import simulateData

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
    run_config = code0.ModelParamsConfig(dp)

    #checkpoint_file='skillBuilder.chk'

    config.num_steps = aux.get_num_step(dataset)
    eval_config.num_steps = config.num_steps
    run_config.num_steps = config.num_steps
    eval_config.batch_size = 2
    run_config.batch_size = 50
    config.skill_num = skill_num
    eval_config.skill_num = config.skill_num
    run_config.skill_num = config.skill_num


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
                mrun = code2.Model(is_training=False, config=run_config, dp=dp)

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
                        run_dataset,run_lables = simulateData.create_label_and_delete_last_one()
                        run_tuple_data = code1.convert_data_labels_to_tuples(run_dataset,run_lables)
                        print ("length of run_tuple_data \t",len(run_tuple_data))
                        preds = simulateData.runEpoch(session,mrun,run_tuple_data,tf.no_op())

                        print (len(preds))
                        #print ("file result\t",preds)
                        result = pd.DataFrame({'result':list(preds)})
                        result.to_csv('./result/simulate_'+dp.dataSetSize+'_result.csv')
                    print("=" * 80)
                    #aux.logwrite('\n'+test_result, dp,False)
    print("==> Finsih! whole process, save result and print\t" + dp.currentTime)

if __name__ == "__main__":
    tf.app.run()


