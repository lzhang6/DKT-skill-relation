import tensorflow as tf
import numpy as np
import code0_parameter as code0
import code1_data as code1


class ONEHOTENCODERINPUT(object):
    def __init__(self, ap, dp, inputs,printControl=True):
        self.batch_size = batch_size = ap.batch_size
        self.num_steps = num_steps = ap.num_steps
        self.seq_width = seq_width = len(dp.columnsName_to_index)
        self.skill_num = dp.skill_num
        self.dp = dp
        self.ap = ap
        self.model_cross_columns = dp.model_cross_columns
        self.inputs = inputs
        self.printControl=printControl


        width_deep_width_dict = {"skill_id": dp.columns_max['skill_id'] + 1,"correct": dp.columns_max['correct'] + 1}

        self.data_skill_id = tf.to_int32(
            tf.slice(self.inputs, [0, 0, dp.columnsName_to_index['skill_id']], [-1, -1, 1]))
        self.data_skill_id_process = tf.to_float(tf.squeeze(
            tf.one_hot(indices=self.data_skill_id, depth=width_deep_width_dict['skill_id'], on_value=1.0, off_value=0.0,
                       axis=-1)))
        self.data_correct = tf.slice(self.inputs, [0, 0, dp.columnsName_to_index['correct']], [-1, -1, 1])


    def getSkillCorrectCrossFeature(self):
        TensorCrossFeatures = self._getCrossFeature(['skill_id correct'])
        if self.printControl: print("==> [Tensor Shape] skill_id and correct cross feature\t", TensorCrossFeatures.get_shape())
        return TensorCrossFeatures


    def _getCrossFeature(self, crossFeatureNameList):
        wide_length = 0
        for i, crossFeatureName in enumerate(crossFeatureNameList):  # crossFeatureName is a string'correct first_response_time'
            depthValue = int(self.dp.columns_max[crossFeatureName] + 1)
            wide_length += depthValue

            tmp_value = tf.to_int32(
                tf.slice(self.inputs, [0, 0, self.dp.columnsName_to_index[crossFeatureName]], [-1, -1, 1]))

            tmp_kk = tf.squeeze(tf.one_hot(indices=tmp_value, depth=depthValue, on_value=1.0, off_value=0.0, axis=-1))
            #tmp_kk = tf.one_hot(indices=tmp_value, depth=depthValue, on_value=1.0, off_value=0.0, axis=-1)
            tmp_kkp = tf.reshape(tmp_kk,[self.batch_size,-1,depthValue])
            tmp_value_ohe = tf.to_float(tmp_kkp)
            if self.printControl: print("==> [Tensor Shape] Cross Feature--", crossFeatureName, " width\t", depthValue)

            if i == 0:
                TensorCrossFeatures = tmp_value_ohe
            else:
                TensorCrossFeatures = tf.concat(2, [TensorCrossFeatures, tmp_value_ohe])
        # if no cross features, the return value is null
        return TensorCrossFeatures

if __name__ == "__main__":
    dp = code0.DatasetParameter()
    ap = code0.autoencoderParameter()

    dataset, labels = code1.load_data(dp)
    # tuple_data = code1.convert_data_labels_to_tuples(dataset, labels)

    skill_num = len(dataset['skill_id'].unique()) + 1  # 0 for unlisted skill_id
    dp.skill_num = skill_num
    dp.skill_set = list(dataset['skill_id'].unique())
    dp.columns_max, dp.columns_numb, dp.columnsName_to_index = code1.get_columns_info(dataset)
    dp.seq_width = len(dp.columnsName_to_index)

    print("columns_max\n", dp.columns_max)
    print("columns_numb\n", dp.columns_numb)
    print("columnsName_to_index\n", dp.columnsName_to_index)

    data = np.random.randint(low=0,high=2, size=())
    g =tf.Graph()
    with g.as_default():
        inputs = tf.placeholder(tf.float32, [ap.batch_size, ap.num_steps, len(dp.columnsName_to_index)])
        m = ONEHOTENCODERINPUT(ap=ap, dp=dp,inputs=inputs)

    with tf.Session(graph=g) as sess:
        print("-" * 60)
        m.getSkillCorrectCrossFeature()
        print("-" * 60)

