import tensorflow as tf
import datetime

# Hyperparameter for all kinds of file
DATASETTYPE = 'assistment2009' #'assistment2009''
TARGETSIZE = 500         #kdd: 500, assistment 100
AUTOENCODER_ACT = 'tanh'                   # tanh, sigmoid


DATASETSIZE = "middle"    #'large | middle | small'
RNN_layer_number = 1     #'1|2'
CELLTYPE = "LSTM"         #"RNN | LSTM | GRU"

BASELINE = False


class DatasetParameter(object):
    def __init__(self):
        self.dataSetType  = DATASETTYPE          #"assistment2009 "
        self.dataSetSize  = DATASETSIZE          # 'small |large | middle'

        if self.dataSetType == "assistment2009":
            self.processedFileName = './data/connectData_'+self.dataSetSize+'.csv'
            self.filtedColumnNameList = ['skill_id',
                                         'user_id', 'correct']

        self.currentTime = datetime.datetime.now().strftime("%m-%d-%H:%M")
        self.dataset_columns_for_cross_feature = [['skill_id','correct']]
        self.model_cross_columns  = [['skill_id','correct']]                                 #"the continues data columns needed to consider"

        # need to change value
        self.columnsName_to_index = {}
        self.columns_max = {}
        self.columns_numb = {}
        self.seq_width =0
        self.skill_num = 0

    def convertCrossCoumnsToNameList(self,Flag=True):
        if Flag:
            mcu = self.dataset_columns_for_cross_feature
        else:
            mcu = self.model_cross_columns
        crossFeatureNameList = []
        if len(mcu)!=0:
            for index_ccl,crossColumnsList in enumerate(mcu):
                crossFeatureName=''
                if len(set(crossColumnsList))<=1:
                    raise ValueError("need two different feature at least ")

                for index_cc,crossColumn in enumerate(crossColumnsList):
                    if index_cc ==0:
                        crossFeatureName=crossColumn
                    else:
                        crossFeatureName=crossFeatureName+" "+crossColumn
                crossFeatureNameList.append(crossFeatureName)
        return crossFeatureNameList

class autoencoderParameter(object):
    def __init__(self):
        self.epoch_rbm = 10
        self.epoch_autoencoder = 10
        self.batch_size = 50
        self.num_steps = 100

class SAEParamsConfig(object):
    def __init__(self):
        self.learning_rate = 0.005
        self.min_lr = 0.0001
        self.lr_decay = 0.98
        self.layer_num = 1
        self.init_scale = 0.05
        self.target_size = TARGETSIZE
        self.max_max_epoch = 5
        self.display_step = 1

        self.batch_size = 300
        self.num_steps = 0               # need to resign value of time stampes
        self.seq_width = 0               # need to resign value

#Parameter for RNN
class ModelParamsConfig(object):
    def __init__(self,dp):
        self.num_steps = 0               # need to resign value of time stampes
        self.skill_num = 0               # need to resign value of skill number
        self.seq_width = 0               # need to resign value
        if dp.dataSetType=='kdd':
            self.batch_size = 5
        else:
            self.batch_size = 30
        if DATASETSIZE == 'small':
            self.max_max_epoch = 5
        else:
            self.max_max_epoch = 40
        self.num_layer = RNN_layer_number
        self.cell_type = CELLTYPE    #"RNN | LSTM | GRU"
        self.hidden_size = 200
        self.hidden_size_2 = 150

        self.init_scale = 0.05
        self.learning_rate = 0.05
        self.max_grad_norm = 4
        self.max_epoch = 5
        self.keep_prob = 0.6
        self.lr_decay = 0.9
        self.momentum = 0.95
        self.min_lr = 0.0001

if __name__ == "__main__":
    param_ass = DatasetParameter()
    print (param_ass.convertCrossCoumnsToNameList())
