import os
from model_dt_v2 import *
from bert.tokenization.albert_tokenization import FullTokenizer
from tqdm import tqdm
from util.util import *
import os
import random

random.seed(0)

"""
对比v1版本：
1. 按顺序预测论元，全为空则为0
"""
def padding(value, padding, max_length):
    cls, sep, padding = padding
    length =len(value)
    if len(value)>max_length:
        value = value[:max_length]
        length = max_length
        value= [cls]+value+[sep]
    else:
        value = [cls]+value+[sep]+[padding for _ in range(max_length-len(value))]
    length +=2
    mask = [True for _ in range(length)]+[False for _ in range(max_length+2-length)]
    return value, mask, length-2

def padding2(value, padding, max_length):
    value = value[:max_length]
    if len(value)!=max_length:
        value = value + [padding for _ in range(max_length-len(value))]

    return value


def split_content(content, max_length, sep=['；', '。', '！']):
    max_length = max_length-2
    text = [[]]
    for index, c in enumerate(content):
        text[-1].append(c)
        if c in sep:
            text.append([])
    cache = [x for x in text if len(x)!=0]
    text = []
    for x in cache:
        if len(x)>max_length:
            size = len(x)//max_length+1
            length = len(x)//size+1
            for _ in range(size):
                text.append(x[_*length:(_+1)*length])
        else:
            text.append(x)
    return text


def process_text(content, vocab, padding_value, max_length=256):
    """
    处理文本
    :param content: 原文
    :param vocab: 词库
    :param padding_value: 填充值
    :return: text-分句并转为编码的文本， mask-各句子实际长度， sentebce_mask-实际句子数量，slength-每句的偏移值
    """
    text = []
    pindex = 0
    for index, c in enumerate(content):
        if c in ['；', '。', '！', '，']:
            text.append(content[pindex:index + 1])
            pindex = index + 1
    if pindex != len(content) and len([_ for _ in content[pindex:] if _!=' ']) != 0:
        text.append(content[pindex:])

    content = vocab.convert_tokens_to_ids([_ if _ in vocab.vocab else '[UNK]' for _ in content])
    content, mask, length = padding(content, padding_value, max_length-2)
    return content, mask, length

def find_all(string, sub):
    rt = []
    pi = 0
    sub = [_ for _ in sub]
    string = [_ for _ in string]
    for index, x in enumerate(string):
        if pi >=len(sub):
            print('')
        if sub[pi] == x:
            pi+=1
        else:
            if sub[0]==x:
                pi = 1
            else:
                pi = 0
        if pi==len(sub):
            rt.append(index-len(sub)+1)
            pi = 0
    return rt


def process_entity(group, max_length):
    """

    :param content: 原文
    :param elements: 四元组
    :param slength: 各句偏移值
    :param vocab: 词典
    :return: ner_tag-实体标签, entity_index-实体索引, entity_masks-标记相同实体, entity_map-实体列表
    """
    # 'trigger', 'object', 'subject', 'time', 'location'
    tag_start = [0 for _ in range(max_length-1)]
    tag_end = [0 for _ in range(max_length - 1)]
    type = ['o', 'trigger', 'object', 'subject', 'time', 'location']

    entity_index = []
    entity_map = []
    entity_type = []
    # flag = 0
    for element in group:
        for ei, e in enumerate(element):

            if isinstance(e[0], str) and e[0].strip()=='':
                continue
            if not isinstance(e[0], list):
                es = [e]
            else:
                es = [_ for _ in e if _[0].strip()!='']
                if len(es)==0:
                    continue
            for e in es:
                # 查找全部相同实体，修改tag
                index = e[1:]

                start = ei+1
                end = ei+1

                em = ' '.join([e[0], str(e[1]), str(e[2])])

                if em not in entity_map:
                    entity_map.append(em)
                    entity_type.append(ei+1)
                    tag_start[index[0]] = start
                    if index[1]>=len(tag_end):
                        print('')
                    tag_end[index[1]] = end

                    entity_index.append([index[0]+1,index[1]+1])

    tag_start = [0]+tag_start
    tag_end = [0]+tag_end

    return tag_start, tag_end, entity_index, entity_map, entity_type


def get_sptype_count(types, type):
    count = 0
    for x in types:
        if x==type:
            count+=1
    return count

def process_path_tag(group, entity_map, entity_type):
    """

    :param content: 原文
    :param elements: 四元组
    :param entity_map: 实体列表
    :return: 路径tag
    """
    # 生成路径tag
    # 'trigger', 'object', 'subject', 'time', 'location'
    tree = {"index": None, "child": {}}
    # group[0][2]=['', -1, -1]
    for element in group:
        cache = tree
        # element = [x for x in element if x[0]!='']
        for ei,e in enumerate(element):
            if isinstance(e[0], list):
                print('')

            em = ' '.join([e[0], str(e[1]), str(e[2])])
            if em not in cache['child']:
                if e[0] == '':
                    cache['child'][em] = {"index": -1, "type": ei+1,
                                          "child": {}}
                else:
                    cache['child'][em]={"index": entity_map.index(em),"type":entity_type[entity_map.index(em)], "child": {}}

            cache = cache['child'][em]


    # 广度遍历树，生成path_tag
    prefix_path = []
    cnode = [[prefix_path, tree]]
    path_tag = []
    pos_id = []
    while len(cnode) != 0:
        node = cnode.pop()
        prefix_path = node[0]
        node = node[1]
        next_node = [0 for __ in entity_map]

        for n in node['child']:
            if node['child'][n]['index']!=-1:
                next_node[node['child'][n]['index']] = 1
            prefix_path_ = [__ for __ in prefix_path]
            prefix_path_.append(node['child'][n]['index'])
            cnode.append([prefix_path_, node['child'][n]])

        path_tag.append([prefix_path, next_node])

    path_tag = [_ for _ in path_tag if len(_[0])!=5]
    return path_tag


def porcess_pos(pos_id, entity_type):
    pos_id = np.array(pos_id, dtype=np.int32)
    pos_id_ = [_ for _ in pos_id]
    # 根据类型排除
    entity_type_set = list(set(entity_type))
    pos_id = np.array(pos_id, dtype=np.int32)
    for x in set(entity_type_set):
        if x == 1:
            continue
        mask = np.array([0 if _ == x else 1 for _ in entity_type], dtype=np.int32)
        m = min((pos_id * (1 - mask)).tolist())

        m_ = (np.array(pos_id, dtype=np.int32) - (m - 1)) * (1 - mask)

        pos_id = pos_id * mask + m_

    mask = np.array([0 if _ == 1 else 1 for _ in entity_type], dtype=np.int32)
    pos_id = pos_id * mask

    pos_id = pos_id.tolist()
    return pos_id


def get_process_content(vocab, content, padding_value, max_length, tag):
    content_split = split_content(content, max_length)
    content_ids = []
    masks = []
    for _ in content_split:
        content_id, mask, length = process_text(_, vocab, padding_value, max_length)
        content_ids.append(content_id)
        masks.append(mask)
    return content_ids,masks, [tag for x in content_ids]


def test_ner_tag(content_test, tag_start_all, tag_end_all):
    entity_test = []
    for xi, x in enumerate(tag_start_all[:-1]):
        if x != 0:
            end = xi + 1
            for xi_ in range(xi + 1, len(tag_end_all)):
                end = xi_
                if tag_end_all[xi_] == x:
                    break
            entity_test.append(''.join(content_test[xi:end]))
    return entity_test

def get_raw(indexs, content):
    return ''.join([content[x] for x in indexs])


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    st = x_exp / x_exp_row_sum
    return st

def content_split_fn(content, entity_index_, path_tag, entity_type):
    insert_tag_path = ['[unused1]', '[unused2]','[unused3]','[unused4]','[unused5]'] # 前置路径的插入符号
    insert_tag = ['[unused6]','[unused7]','[unused8]','[unused9]','[unused10]'] # 候选集的插入符号
    content_ = ['[CLS]'] + list(content) + ['[SEP]']
    lindex = 0
    content_split = []
    content_type = []
    content_index = []
    for i_, _ in enumerate(entity_index_):
        if lindex != _[1]:
            content_split.append(content_[lindex:_[1]])
            content_type.append(-1)
            content_index.append(-1)
        content_split.append(content_[_[1]:_[2]])
        if _[0] in path_tag:
            content_type.append(insert_tag_path[entity_type[_[0]]-1])
        else:
            content_type.append(insert_tag[entity_type[_[0]]-1])
        content_index.append(i_)

        lindex = _[2]

    if (len(content_) - lindex) != 0:
        content_split.append(content_[lindex:])
        content_type.append(-1)

    return content_split, content_type, content_index


def get_process_data(entity_index_, content_split, content_type, content_index):
    content_path = []
    entity_index_pl_ = [[] for _ in entity_index_]
    for i_ in range(len(content_split)):
        if content_type[i_]!=-1:
            entity_index_pl_[entity_index_[content_index[i_]][0]].append(len(content_path))
            content_path.extend([content_type[i_]] + list(content_split[i_]) + [content_type[i_]])
            entity_index_pl_[entity_index_[content_index[i_]][0]].append(len(content_path) - 1)
        # if content_type[i_] == 0:
        #     entity_index_pl_[entity_index_[content_index[i_]][0]].append(len(content_path))
        #     content_path.extend(['#'] + list(content_split[i_]) + ['#'])
        #     entity_index_pl_[entity_index_[content_index[i_]][0]].append(len(content_path) - 1)
        # elif content_type[i_] == 1:
        #     entity_index_pl_[entity_index_[content_index[i_]][0]].append(len(content_path))
        #     content_path.extend(['[SEP]'] + list(content_split[i_]) + ['[SEP]'])
        #     entity_index_pl_[entity_index_[content_index[i_]][0]].append(len(content_path) - 1)
        else:
            content_path.extend(content_split[i_])
    return content_path, entity_index_pl_


def content_padding(content, entity_index_, path_tag, entity_type):
    content_split, content_type, content_index = content_split_fn(content, entity_index_, path_tag, entity_type)

    # 在前置路径上的左右插入 #，在候选集上插入 [SEP]
    content_path, entity_index_pl_ = get_process_data(entity_index_, content_split, content_type,
                                                      content_index)
    return content_path, entity_index_pl_


def get_norm_data(data):
    tag_index = ['trigger', 'subject', 'object', 'loc', 'time']
    pdata = []
    x = data

    content = x['sentence']
    group = []
    for _ in [x['event']]:
        group.append(
            [[''.join(_['trigger']['text']), _['trigger']['offset'], _['trigger']['offset'] + _['trigger']['length']],
             ['', -1, -1],
             ['', -1, -1],
             ['', -1, -1],
             ['', -1, -1]])
        for __ in _['arguments']:
            group[-1][tag_index.index(__['role'])] = [''.join(__['text']), __['offset'], __['offset'] + __['length']]
        group[-1] = {'events': group[-1]}
    return {'content': content, 'group': group}

def get_norm_data_v2(data):
    tag_index = ['trigger', 'subject', 'object', 'loc', 'time']
    pdata = []
    x = data

    content = x['sentence']
    group = []
    for _ in x['event']:
        group.append(
            [[''.join(_['trigger']['text']), _['trigger']['offset'], _['trigger']['offset'] + _['trigger']['length']],
             ['', -1, -1],
             ['', -1, -1],
             ['', -1, -1],
             ['', -1, -1]])
        for __ in _['arguments']:
            group[-1][tag_index.index(__['role'])] = [''.join(__['text']), __['offset'], __['offset'] + __['length']]
        group[-1] = {'events': group[-1]}
    return {'content': content, 'group': group}

def process_data(content, events):
    cache = []

    content = content.rstrip() + ' '

    _, _, entity_index, entity_map, entity_type = process_entity([events], len(content) + 2)

    entity_index_ = sorted([[i_] + _ for i_, _ in enumerate(entity_index)], key=lambda x: x[1])

    path_tag_dt = [entity_map.index(' '.join([_[0], str(_[1]), str(_[2])])) for _ in events if
                   _[0] != '']
    content_path_d, entity_index_pl_d = content_padding(content, entity_index_, path_tag_dt, entity_type)

    content_path_ids_dt = vocab.convert_tokens_to_ids(
        [_ if _ in vocab.vocab else '[UNK]' for _ in content_path_d])
    cache.append(padding2(content_path_ids_dt, 0, max_length))
    cache.append(len(content_path_ids_dt))
    cache.append(padding2(entity_index_pl_d, [0, 0], 5))
    cache.append(len(entity_index_pl_d))

    return cache

def get_dataset_by_id(path, batch_size, vocab, max_length):
    tense_map = ['过去', '将来', '其他', '现在']
    polarity_map = ['肯定', '可能', '否定']
    cache = read_json(path)
    data = [_ for _ in cache if len(_['content'])<=max_length*0.8]

    cache = []

    for xi, x in enumerate(data):
        content = x['content'].rstrip() + ' '
        for group_o in x['group']:
            events = group_o['event']
            cache.append([])
            cache[-1].append(tense_map.index(group_o['tense']))
            cache[-1].append(polarity_map.index(group_o['polarity']))

            _, _, entity_index, entity_map, entity_type = process_entity([events], len(content) + 2)

            entity_index_ = sorted([[i_] + _ for i_, _ in enumerate(entity_index)], key=lambda x: x[1])

            path_tag_dt = [entity_map.index(' '.join([_[0], str(_[1]), str(_[2])])) for _ in group_o['event'] if _[0] != '']
            content_path_d, entity_index_pl_d = content_padding(content, entity_index_, path_tag_dt, entity_type)

            content_path_ids_dt = vocab.convert_tokens_to_ids(
                [_ if _ in vocab.vocab else '[UNK]' for _ in content_path_d])
            cache[-1].append(padding2(content_path_ids_dt, 0, max_length))
            cache[-1].append(len(content_path_ids_dt))
            cache[-1].append(padding2(entity_index_pl_d, [0, 0], 5))
            cache[-1].append(len(entity_index_pl_d))

    data = cache
    random.shuffle(data)
    def get_dataset():
        size = len(data) // batch_size + 1

        for i in range(size):
            # try:
            pdata = data[i * batch_size:(i + 1) * batch_size]
            if len(pdata) != batch_size:
                continue

            content_path_all_dt = []  # 加了标致后的文本
            entity_index_pl_all_dt = []  # 实体的index

            masks3 = []
            entity_mask_all_dt = []

            tense = []
            polarity = []

            for xi, x in enumerate(pdata):
                tense_tag, polarity_tag, content_ids, masks, entity_index, entity_masks = x
                content_path_all_dt.append(content_ids)
                masks3.append(masks)
                entity_index_pl_all_dt.append(entity_index)
                entity_mask_all_dt.append(entity_masks)
                tense.append(tense_tag)
                polarity.append(polarity_tag)
            yield content_path_all_dt, masks3, entity_index_pl_all_dt, entity_mask_all_dt, tense, polarity

    return get_dataset

# vocab = FullTokenizer('./model/roeberta_zh_L-24_H-1024_A-16/vocab.txt')
# max_length = 256
# dense_map = ['过去','将来','其他','现在']
# polarity_map = ['肯定', '可能', '否定']
# for index, x in enumerate(get_dataset_by_id(['./data/process/train.json'],2,vocab, max_length)()):
#     print(index)
#     i = 1
#
# print('')


def get_data(fn, max_length=256, repeat=1, prefetch=512, shuffle=512, seed=0, count=0):
    dataset = tf.data.Dataset.from_generator(fn,
                                             (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32),
                                             (tf.TensorShape([None, max_length]), tf.TensorShape([None]),
                                              tf.TensorShape([None, 5, 2]), tf.TensorShape([None]),
                                              tf.TensorShape([None]), tf.TensorShape([None]))
                                             )

    dataset = dataset.repeat(count=repeat).prefetch(prefetch).shuffle(shuffle, seed)

    # count = 116848
    if count==0:
        # 统计数量
        init_call = None
        for i, x in enumerate(dataset):
            print(i)
            if init_call == None:
                init_call = x
                break
        for x in fn():
            count+=1
        # count = 116848
        return dataset, init_call, count
    else:
        return dataset



def conver_token_to_id(content, padding_value, max_length):
    content_split = split_content(content, max_length)
    content_ids = []
    masks = []
    for _ in content_split:
        content_id, mask, length = process_text(_, vocab, padding_value, max_length)
        content_ids.append(content_id)
        masks.append(mask)

    return content_ids, masks

def get_entity_and_type(ft_start, ft_end, content, sep):
    rt = []
    tag = []
    entity = []
    for index, _ in enumerate(ft_start[:-1]):
        if _ != 0:
            if _ in ft_end[index + 1:]:
                e = 1 + index + ft_end[index + 1:].index(_)
            else:
                e = index + len(ft_end[index + 1:])
            for c in sep:
                if c in content[index:e]:
                    e = index + content[index:e].index(c)
            rt.append([i for i in range(index, e)])
            tag.append(tag_map[_])

    for index, _ in enumerate(rt):
        if tag[index] == 'O':
            continue
        word = [content[c] for c in _]
        entity.append([word, _, tag[index]])

    return tag, entity

def get_entity_index(entity):
    rt = []
    entity_map = []
    entity_map_ = []
    entity_type = []
    for x in range(len(entity)):
        word = '-'.join([str(_) for _ in entity[x][0]])
        word_index = '-'.join([str(_) for _ in entity[x][0]+entity[x][1]])
        entity_type.append(tag_map.index(entity[x][-1]))
        rt.append([])
        rt[-1].extend(entity[x][1])
        entity_map.append(word)
        entity_map_.append(word_index)

    max_l = max([len(_) for _ in rt])
    rt = [padding2(_, 0, max_l) for _ in rt]
    return rt, entity_map, entity_map_, entity_type


def convert_to_tensor(data, dtype=tf.int32):
    return tf.convert_to_tensor(data, dtype=dtype)

def get_f1_e(p, t, isp=True):
    if isp:
        p = set([' '.join([__ for __ in _ if __!='']) for _ in p])
        t = set([' '.join([__[0] for __ in _ if __[0]!='']) for _ in t])
    else:
        p = set(p)
        t = set(t)

    P = len(p & t) / (len(p) + 1e-8)
    G = len(p & t) / (len(t) + 1e-8)

    f1 = (2 * P * G) / (P + G + 1e-8)
    return f1

def get_pos(content, index_x, index_y):
    content = content[min([index_x, index_y]):max([index_x, index_y])]
    return 0 if '。' in content else 1

def dev_step(raw, masks, entity_index, model):

    tense_p, polarity_p = model.predict_v2(convert_to_tensor(raw,  dtype=tf.int32), convert_to_tensor(masks,  dtype=tf.int32),
                                        convert_to_tensor(entity_index,  dtype=tf.int32))

    return tense_p, polarity_p

def get_avg(model, ckpt, init_call, paths):
    n2v = {}
    for path in paths:
        ckpt.restore(path)
        model.call(init_call)

        for v in model.trainable_variables:
            if v.name not in n2v:
                n2v[v.name] = v.numpy()
            else:
                n2v[v.name] = 0.95*n2v[v.name]+0.05*v.numpy()

    return n2v

def replace_v(model, n2v):
    for v in model.trainable_variables:
        v.assign(n2v[v.name])
    return model

if __name__=="__main__":
    rt = read_json(['./sub/all.json'])
    tense_map = ['过去', '将来', '其他', '现在']
    polarity_map = ['肯定', '可能', '否定']

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = {
        "model_name": "roberta_zh_large_model.ckpt",
        "dir": "./model/roeberta_zh_L-24_H-1024_A-16",
        # "model_name": "albert_model.ckpt",
        # "dir": "./model/albert_tiny_489k",
        "hiddent_size": 312,
        "hiddent_size2": 312 * 2,
        "dropout": 0.5,
        "seq_length": 256,
        'type_size': 6,
        'entity_size': 32,
        'batch_size': 4,
        'tense_size': 4,
        'polarity_size': 3
    }

    dir = config['dir']
    max_length = config['seq_length']
    batch_size = config['batch_size']

    load_f = 'roberta_dt_v2_1'

    model = Ner(config)

    vocab = FullTokenizer(dir + '/vocab.txt')

    train_dataset, init_call, count = get_data(
        get_dataset_by_id(['./data/train.json'], batch_size, vocab, max_length))

    ckpt = tf.train.Checkpoint(model=model)

    ckpt.restore(tf.train.latest_checkpoint('./model/tense&polarity/0/'))

    def get_time(content):
        index = []
        for x in '周日月年时分季代':
            if x in content:
                if x=='周':
                    index.append(content.index(x)+1)
                else:
                    index.append(content.index(x))
        if len(index)!=0:
            return content[:max(index)+1]
        else:
            return ''

    dev_data = rt
    # dev_data = read_json(['./data/process/dev.json'])
    tqdm_dev = tqdm(total=len(dev_data))
    save_data = []
    for i, data in enumerate(dev_data):
        tqdm_dev.update(1)
        event_save = data['event']
        data_ = get_norm_data_v2(data)
        events = [_['events'] for _ in data_['group']]
        events_p = []
        for i, x in enumerate(events):
            content_ids, masks, entity_index, _ = process_data(data_['content'], x)
            tense_p, polarity_p = dev_step([content_ids], [masks], [entity_index], model)
            tense_p = tense_p[0]
            polarity_p = polarity_p[0]
            es = event_save[i]
            es['tense'] = tense_p
            es['polarity'] = polarity_p
            es['trigger']['text'] = ''.join(es['trigger']['text'])
            for _ in es['arguments']:
                _['text'] = ''.join(_['text'])
            events_p.append(es)
        save_data.append({'sentence': data['sentence'], 'words': data['words'], 'event': events_p})
        # save_data.append({'sentence': content, 'events': p_group, 'event_tag':data['group']})

    ckpt.restore(tf.train.latest_checkpoint('./model/tense&polarity/1/'))

    dev_data = save_data
    # dev_data = read_json(['./data/process/dev.json'])
    tqdm_dev = tqdm(total=len(dev_data))
    save_data = []
    for i, data in enumerate(dev_data):
        tqdm_dev.update(1)
        event_save = data['event']
        data_ = get_norm_data_v2(data)
        events = [_['events'] for _ in data_['group']]
        events_p = []
        for i, x in enumerate(events):
            content_ids, masks, entity_index, _ = process_data(data_['content'], x)
            tense_p, polarity_p = dev_step([content_ids], [masks], [entity_index], model)
            tense_p = tense_p[0]
            polarity_p = polarity_p[0]
            es = event_save[i]
            es['tense'] = tense_map[np.argmax(es['tense']+tense_p)]
            es['polarity'] = polarity_map[np.argmax(es['polarity']+polarity_p)]
            es['trigger']['text'] = ''.join(es['trigger']['text'])
            for _ in es['arguments']:
                _['text'] = ''.join(_['text'])
            events_p.append(es)
        save_data.append({'sentence': data['sentence'], 'words': data['words'], 'events': events_p})
        # save_data.append({'sentence': content, 'events': p_group, 'event_tag':data['group']})

    save_data_ = {}
    for x in save_data:
        if x['sentence'] not in save_data_:
            save_data_[x['sentence']] = {'sentence':x['sentence'], 'words': x['words'], 'events': x['events']}
        else:
            save_data_[x['sentence']]['events'].extend(x['events'])


    test_raw = read_json('./data/raw/测试集/sentences.json')
    sub_data = []
    for x in test_raw:
        key = x['sentence']
        if key in save_data_:
            sub_data.append(save_data_[key])
        else:
            sub_data.append('None')

    print(len(sub_data))
    with open('./sub/sub.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(sub_data, ensure_ascii=False))