import tensorflow as tf
from tensorflow.keras import layers
from bert import BertModelLayer, params_from_pretrained_ckpt, fetch_google_albert_model,albert_params
import numpy as np
import os
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
    for index,x in enumerate(data):
        cache.append(str(index))
        if x =='O':
            o_c += 1
        if ('_S' in x and len(cache) > 1) or (x == 'O' and o_c == 1 and len(cache)> 1):
            if o_c != 1 or not (x == 'O' and o_c == 1 and len(cache)> 1):
                o_c = 0
            rt.append('_'.join(cache[:-1]))
            cache = [cache[-1]]
            if len(data[index-1]) > 1:

                tag.append(data[index-1][:-2])
            else:
                tag.append(data[index-1])
    if len(cache)!=0:
        rt.append('_'.join(cache))
        if len(data[-1])>1:
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
        self.bilstm = layers.Bidirectional(layers.LSTM(self.config['hiddent_size'], return_sequences=True),
                                           name='bilstm_ner')

        # 实体识别
        self.ner_dense_start = layers.Dense(config['type_size'], name='ner_dense_start')
        self.ner_dense_end = layers.Dense(config['type_size'], name='ner_dense_end')

    # @tf.function
    def __encode(self, text, mask, training=True):
        """
        第一层编码
        :param text: tf.int32 [None, seq_length]
        :param mask: tf.bool [None, seq_length]
        :param training: bool
        :return: [None, seq_length, albert_output_size]
        """

        em = self.albert([text, tf.ones(text.shape, dtype=tf.int32)], tf.sequence_mask(mask, maxlen=self.config['seq_length']), training)
        em = tf.stack(em, axis=2)
        em_at = tf.expand_dims(tf.nn.softmax(tf.squeeze(self.at_dense(em), axis=-1), axis=-1), axis=-1)
        em_ = tf.reduce_sum(tf.multiply(em, em_at), axis=2)
        em_d = self.d_dense(em_)
        em_d = self.bilstm(em_d)
        return em_d

    # @tf.function
    def __get_ner_loss(self, ft_start, ft_end, tag_start, tag_end, masks):
        """
        实体识别部分的解码
        :param logits: 序列编码 tf.float32 [None, seq_length, albert_output_size]
        :param mask: 序列实际长度 tf.int32 [None]
        :param tag: 标签 tf.int32 [None, pos_size]
        :return:
        """
        tag_s = tf.one_hot(tag_start, depth=self.config['type_size'], axis=-1)
        loss_s = tf.nn.softmax_cross_entropy_with_logits(labels=tag_s, logits=ft_start)

        tag_e = tf.one_hot(tag_end, depth=self.config['type_size'], axis=-1)
        loss_e = tf.nn.softmax_cross_entropy_with_logits(labels=tag_e, logits=ft_end)

        masks_ = tf.sequence_mask(masks, maxlen=self.config['seq_length'])
        loss_s = tf.reduce_mean(tf.boolean_mask(loss_s, masks_))
        loss_e = tf.reduce_mean(tf.boolean_mask(loss_e, masks_))
        loss = (loss_e + loss_s) / 2
        return loss

    # @tf.function
    def __get_path_loss(self, ft, path_tag_all):
        path_tag_all = tf.cast(path_tag_all, dtype=tf.float32)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=ft, labels=path_tag_all)

        return tf.reduce_mean(loss)

    def __get_ner_ft(self, content_ids_all, masks, istraining=False):
        """
        返回bert编码后的文本，以及文本上个字符的表征
        :param content_ids_all:
        :param masks:
        :param istraining:
        :return:
        """
        text_em = self.__encode(content_ids_all, masks, istraining)

        ems_start_ft_ = self.ner_dense_start(text_em)

        ems_end_ft_ = self.ner_dense_end(text_em)

        return text_em, ems_start_ft_, ems_end_ft_


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
        content_ids_all, tag_start_all, tag_end_all, masks = data

        text_em, ems_start_ft_, ems_end_ft_ = self.__get_ner_ft(content_ids_all, masks, True)

        # 计算loss
        ner_loss = self.__get_ner_loss(ems_start_ft_, ems_end_ft_, tag_start_all, tag_end_all, masks)

        return ner_loss

    def predict_ner(self, content, mask):
        """
        目前仅适用单段落
        :param content:
        :param mask:
        :return:
        """
        _, ems_start_ft_, ems_end_ft_ = self.__get_ner_ft(content, mask)

        return np.array(ems_start_ft_).argmax(-1), np.array(ems_end_ft_).argmax(-1)

    def predict_ner_p(self, content, mask):
        """
        目前仅适用单段落
        :param content:
        :param mask:
        :return:
        """
        _, ems_start_ft_, ems_end_ft_ = self.__get_ner_ft(content, mask)

        return np.array(ems_start_ft_), np.array(ems_end_ft_)







