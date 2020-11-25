import tensorflow as tf
from tensorflow.keras import layers
from bert import BertModelLayer, params_from_pretrained_ckpt
import numpy as np
from params_flow.activations import gelu

"""
对比model_ber_bert版本：
1. 增加多种结构提取特征并使用胶囊网络融合
"""

tag_map = ['o', 'trigger', 'object', 'subject', 'time', 'location']

"""
简要版

每次输入一段文本进行训练
"""


def split(data):
    """
    :param data: 预测值
    :return:
    """
    rt = []
    tag = []
    cache = []
    o_c = 0
    for index, x in enumerate(data):
        cache.append(str(index))
        if x == 'O':
            o_c += 1
        if ('_S' in x and len(cache) > 1) or (x == 'O' and o_c == 1 and len(cache) > 1):
            if o_c != 1 or not (x == 'O' and o_c == 1 and len(cache) > 1):
                o_c = 0
            rt.append('_'.join(cache[:-1]))
            cache = [cache[-1]]
            if len(data[index - 1]) > 1:

                tag.append(data[index - 1][:-2])
            else:
                tag.append(data[index - 1])
    if len(cache) != 0:
        rt.append('_'.join(cache))
        if len(data[-1]) > 1:
            tag.append(data[-1][:-2])
        else:
            tag.append(data[-1])
    return rt, tag


class Ner(tf.keras.Model):

    def __init__(self, config):
        super(Ner, self).__init__()
        self.config = config
        model_dir = config['dir']

        bert_params = params_from_pretrained_ckpt(model_dir)
        bert_params['out_layer_ndxs'] = [_ for _ in range(bert_params['num_layers'])]
        bert_model = BertModelLayer.from_params(bert_params, name='bert')

        self.albert = bert_model

        self.at_dense = layers.Dense(1, name='at_dense')
        self.d_dense = layers.Dense(self.config['hiddent_size'], name='d_dense', activation=gelu)
        # self.bilstm = layers.Bidirectional(layers.LSTM(self.config['hiddent_size'], return_sequences=True),
        #                                    name='bilstm_ner')

        # 事件抽取
        self.cnn = layers.Conv1D(self.config['hiddent_size'], 5, name='cnn')

        self.dropout = layers.Dropout(0.5)

        self.tense = layers.Dense(self.config['tense_size'], name='tense')
        self.polarity = layers.Dense(self.config['polarity_size'], name='polarity')

    # @tf.function
    def __encode(self, text, mask, training=True):
        """
        第一层编码
        :param text: tf.int32 [None, seq_length]
        :param mask: tf.bool [None, seq_length]
        :param training: bool
        :return: [None, seq_length, albert_output_size]
        """

        em = self.albert([text, tf.ones(text.shape, dtype=tf.int32)],
                         tf.sequence_mask(mask, maxlen=self.config['seq_length']), training)
        em = tf.stack(em, axis=2)
        em_at = tf.expand_dims(tf.nn.softmax(tf.squeeze(self.at_dense(em), axis=-1), axis=-1), axis=-1)
        em_ = tf.reduce_sum(tf.multiply(em, em_at), axis=2)
        em_d = self.d_dense(em_)
        # em_d = self.bilstm(em_d)
        return em_d

    def __get_event_ft(self, content_ids_all, masks, entity_index_pl_all, istraining=False):
        """
        返回bert编码后的文本，以及文本上个字符的表征
        :param content_ids_all:
        :param masks:
        :param istraining:
        :return:
        """
        text_em = self.__encode(content_ids_all, masks, istraining)

        entity_index_pl_all = [tf.squeeze(_, axis=-1) for _ in tf.split(entity_index_pl_all, [1, 1], axis=-1)]

        entity_em_1 = []
        entity_em_2 = []
        for i in range(text_em.shape[0]):
            entity_em_1.append(tf.expand_dims(tf.nn.embedding_lookup(text_em[i], entity_index_pl_all[0][i]), axis=0))
            entity_em_2.append(tf.expand_dims(tf.nn.embedding_lookup(text_em[i], entity_index_pl_all[1][i]), axis=0))

        entity_em = tf.concat([tf.concat(entity_em_1, axis=0), tf.concat(entity_em_2, axis=0)], axis=-1)
        return text_em, entity_em, text_em[:, 0, :]

    def call(self, data):
        """

        :param text: 原文 tf.int32 [None, seq_length]
        :param mask: 文本实际长度mask tf.bool [None, seq_length]
        :param tag: 实体识别标签 tf.int32 [None, seq_length]
        :param entity_index_s2: 实体位置 句中索引-起始， 句中索引-结束 tf.int32 [entity_count, 2]
        :param content_type: 与entity_index_s2对应，代表对应数据的类型 tf.int32 [entity_count]
        :param entity_masks: 标识是否为相同实体， 例如0011, 前两个相同，后两个也相同 tf.int32 [entity_size]
        :param index_map: 实体相对位置关系
        :param entity_type: 实体类型 tf.int32 [entity_size]
        :param path_tag: 路径标签,每行分别是前缀路径以及下一个role的tag [node_size, 4+entity_size]
        :param training: 是否是训练 tf.bool
        :return:
        """
        content_path_all_dt, masks3, entity_index_pl_all_dt, entity_mask_all_dt, tense, polarity = data

        _, entity_em_, ft_ = self.__get_event_ft(content_path_all_dt, masks3, entity_index_pl_all_dt, True)
        #
        # ft = self.bilstm(entity_em_)[:,0,:]

        ft = tf.squeeze(self.cnn(entity_em_), axis=1)

        ft = tf.concat([ft_, ft], axis=-1)

        # ft = self.dropout(ft, True)

        ft_tense = self.tense(ft)
        ft_polarity = self.polarity(ft)

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(tense, depth=self.config['tense_size']),
                                                    logits=ft_tense)) \
               + tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(polarity, depth=self.config['polarity_size']),
                                                    logits=ft_polarity))
        loss = loss / 2

        # return loss, [tense_p, tense_tag], [polarity_p, polarity_tag]
        return loss

    def predict(self, content_path_all_dt, masks3, entity_index_pl_all_dt):
        _, entity_em_, ft_ = self.__get_event_ft(content_path_all_dt, masks3, entity_index_pl_all_dt, False)
        # ft = self.bilstm(entity_em_)[:, 0, :]
        ft = tf.squeeze(self.cnn(entity_em_), axis=1)

        ft = tf.concat([ft_, ft], axis=-1)

        # ft = self.dropout(ft, False)

        ft_tense = self.tense(ft)
        ft_polarity = self.polarity(ft)

        tense_p = tf.argmax(tf.nn.softmax(ft_tense, axis=-1), axis=-1)
        polarity_p = tf.argmax(tf.nn.softmax(ft_polarity, axis=-1), axis=-1)

        return np.array(tense_p, dtype=np.int32), np.array(polarity_p, dtype=np.int32)

    def predict_v2(self, content_path_all_dt, masks3, entity_index_pl_all_dt):
        _, entity_em_, ft_ = self.__get_event_ft(content_path_all_dt, masks3, entity_index_pl_all_dt, False)
        # ft = self.bilstm(entity_em_)[:, 0, :]
        ft = tf.squeeze(self.cnn(entity_em_), axis=1)

        ft = tf.concat([ft_, ft], axis=-1)

        # ft = self.dropout(ft, False)

        ft_tense = self.tense(ft)
        ft_polarity = self.polarity(ft)

        tense_p = ft_tense
        polarity_p = ft_polarity

        return np.array(tense_p), np.array(polarity_p)


"""
albert bilstn cnn Epoch 10, tense_f1: 0.3815，polarity_f1: 0.3841 *

albert cnn Epoch 5, tense_f1: 0.3700，polarity_f1: 0.3810 *

albert bilstm Epoch 8, tense_f1: 0.4056，polarity_f1: 0.3275 *

albert Epoch 6, tense_f1: 0.4049，polarity_f1: 0.3521 *

concat Epoch 4, tense_f1: 0.4118，polarity_f1: 0.4328 *

Epoch 8, tense_f1: 0.3970，polarity_f1: 0.3820 *
"""