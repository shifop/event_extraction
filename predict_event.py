import os
from model_mcr import *
from bert.tokenization.albert_tokenization import FullTokenizer
from tqdm import tqdm
from util.util import *
import os
import random
from sklearn.metrics import f1_score

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
    insert_tag_path = '[unused1]' # 前置路径的插入符号
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
            content_type.append(insert_tag_path)
        else:
            content_type.append(entity_type[_[0]])
        content_index.append(i_)

        lindex = _[2]

    if (len(content_) - lindex) != 0:
        content_split.append(content_[lindex:])
        content_type.append(-1)
        content_index.append(-1)

    return content_split, content_type, content_index


def get_process_data(content_split, content_type):
    content_path = []
    tag_s = {}
    tag_e = {}
    for i_ in range(len(content_split)):
        if isinstance(content_type[i_], str):
            content_path.extend([content_type[i_]] + list(content_split[i_]) + [content_type[i_]])
        else:
            content_path.extend(content_split[i_])
        if content_type[i_] in [0,1,2,3,4,5,6]:
            tag_e[len(content_path)] = content_type[i_]
            tag_s[len(content_path)-len(content_split[i_])] = content_type[i_]

    start_tag = [0 for _ in range(len(content_path))]
    end_tag = [0 for _ in range(len(content_path))]
    for key in tag_s:
        start_tag[key] = tag_s[key]

    for key in tag_e:
        end_tag[key] = tag_e[key]
    return content_path, start_tag, end_tag


def content_padding(content, entity_index_, path_tag, entity_type):
    content_split, content_type, content_index = content_split_fn(content, entity_index_, path_tag, entity_type)

    # 在前置路径上的左右插入 #，在候选集上插入 [SEP]
    content_path, start_tag, end_tag = get_process_data(content_split, content_type)
    return content_path, start_tag, end_tag


def get_dataset_by_id(path, batch_size, vocab, max_length, isdev = False):
    cache = read_json(path)
    data = [_ for _ in cache if len(_['content'])<=max_length*0.8]

    cache = []
    for x in data:
        content = x['content'].rstrip() + ' '
        # if isdev and '中国家用电器协会针对电热水器涉水涉电' not in content:
        #     continue
        group = [_['event'] for _ in x['group']]
        for _ in group:
            cache.append([content, _])

    data = cache
    data_ = cache

    # random.shuffle(data)
    cache = []
    for xi, x in enumerate(data):
        content, group = x
        _, _, entity_index, entity_map, entity_type = process_entity([group], len(content) + 2)

        path_tag = entity_map.index(' '.join([group[0][0], str(group[0][1]), str(group[0][2])]))

        entity_index_ = sorted([[i_] + _ for i_, _ in enumerate(entity_index)], key=lambda x: x[1])

        # 路径预测
        content_path, start_tag, end_tag = content_padding(content, entity_index_, [path_tag], entity_type)

        content_ids = vocab.convert_tokens_to_ids(
            [_ if _ in vocab.vocab else '[UNK]' for _ in content_path])

        cache.append([content_path, len(content_ids), padding2(content_ids, 0, max_length),
                      padding2(start_tag, 0, max_length), padding2(end_tag, 0, max_length)])

    random.shuffle(cache)

    data = cache

    def get_dataset():
        size = len(data) // batch_size + 1
        data__ = data_

        for i in range(size):
            # try:
            pdata = data[i * batch_size:(i + 1) * batch_size]
            if len(pdata) != batch_size:
                continue

            content_ids_all = []
            tag_start_all = []
            tag_end_all = []
            masks = []
            raw = []

            for xi, x in enumerate(pdata):
                content_path, mask, content_ids, start_tag, end_tag = x

                raw.append(content_path)
                masks.append(mask)
                content_ids_all.append(content_ids)
                tag_start_all.append(start_tag)
                tag_end_all.append(end_tag)

            if isdev:
                yield raw, content_ids_all, tag_start_all, tag_end_all, masks
            else:
                yield content_ids_all, tag_start_all, tag_end_all, masks

    return get_dataset


def get_data(fn, max_length=256, repeat=1, prefetch=512, shuffle=512, seed=0, count=0):
    dataset = tf.data.Dataset.from_generator(fn,
                                             (tf.int32, tf.int32, tf.int32, tf.int32),
                                             (tf.TensorShape([None, max_length]), tf.TensorShape([None, max_length]),
                                              tf.TensorShape([None, max_length]), tf.TensorShape([None]))
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

def get_f1(p, tag):
    p = [int(_) for _ in p]
    tag = [int(_) for _ in tag]
    f1 = f1_score(tag, p, average='macro')
    return f1

@tf.function
def train_step(input, optimizers, index, model):
    """
    使用不同学习率
    :param text:
    :param tag:
    :param mask:
    :param optimizer:
    :param model:
    :return:
    """
    # 打开梯度记录管理器
    with tf.GradientTape() as tape:
        loss = model.call(input)

    # 使用梯度记录管理器求解全部参数的梯度
    grads = tape.gradient(loss, model.trainable_variables)
    # 使用梯度和优化器更新参数
    optimizers[0].apply_gradients(zip(grads[:index], model.trainable_variables[:index]))
    optimizers[1].apply_gradients(zip(grads[index:], model.trainable_variables[index:]))
    # 返回平均损失
    return loss


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

def get_f1_e(p, t):
    p = set(p)
    t = set(t)

    P = len(p & t) / (len(p) + 1e-8)
    G = len(p & t) / (len(t) + 1e-8)

    f1 = (2 * P * G) / (P + G + 1e-8)
    return f1

def get_pos(content, index_x, index_y):
    content = content[min([index_x, index_y]):max([index_x, index_y])]
    return 0 if '。' in content else 1

def dev_step(data, model):
    raw, content, masks = data

    p_start, p_end = model.predict_ner(convert_to_tensor(content), convert_to_tensor(masks))
    f1_all = []
    entitys = []
    for i in range(len(p_start)):
        _, entity = get_entity_and_type(list(p_start[i])[:masks[i]],
                                          list(p_end[i])[:masks[i]],
                                          raw[i],
                                          '')
        entitys.append(entity)

    return entitys

def get_avg(model, ckpt, init_call, paths):
    n2v = {}
    for path in tqdm(paths):
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


def get_dev_data(data, batch_size, vocab, max_length):
    cache = data

    data = []
    for x in cache:
        for _ in x['trigger']:
            data.append([{'sentence': x['sentence'], 'words': x['words'], 'trigger': _}, x['sentence'], _])

    cache = []
    for xi, x in enumerate(data):
        test_data, content, trigger = x

        content = content + ' '

        entity_index_ = [[0, trigger[1] + 1, trigger[1] + trigger[2] + 1]]

        # 路径预测
        content_path, _, _ = content_padding(content, entity_index_, [0], [1])

        content_ids = vocab.convert_tokens_to_ids(
            [_ if _ in vocab.vocab else '[UNK]' for _ in content_path])

        cache.append([test_data, content_path, len(content_ids), padding2(content_ids, 0, max_length)])

    data = cache

    def get_dataset():
        size = len(data) // batch_size + 1

        for i in range(size):
            # try:
            pdata = data[i * batch_size:(i + 1) * batch_size]
            if len(pdata) == 0:
                continue

            content_ids_all = []
            masks = []
            raw = []
            test_datas = []

            for xi, x in enumerate(pdata):
                test_data, content_path, mask, content_ids = x

                test_datas.append(test_data)
                raw.append(content_path)
                masks.append(mask)
                content_ids_all.append(content_ids)

            yield test_datas, raw, content_ids_all, masks

    return get_dataset


def clearn_trigger(sentence, data):
    """
    有介绍只选介绍
    有发布消息，去除发布
    """
    if len(data) == 1:
        return data
    if len(data) == 2:
        if data[0][0] == '介绍' or data[1][0] == '介绍':
            print('=============================================')
            print('触发规则，仅选择‘介绍’')
            print('原文：%s' % (sentence))
            print('%s %s' % (data[0][0], data[1][0]))

            if data[0][0] == '介绍':
                return [data[0]]
            else:
                return [data[1]]

        if ('举行' in sentence or '发布' in sentence) and (data[0][0] == '举行' or data[1][0] == '举行'):
            print('=============================================')
            print('触发规则，仅选择‘举行’')
            print('原文：%s' % (sentence))
            print('%s %s' % (data[0][0], data[1][0]))

            return [_ for _ in data if _[0] == '举行']

        if data[0][0] == '告诉' or data[1][0] == '告诉':
            print('=============================================')
            print('触发规则，仅选择‘告诉’')
            print('原文：%s' % (sentence))
            print('%s %s' % (data[0][0], data[1][0]))

            if data[0][0] == '告诉':
                return [data[0]]
            else:
                return [data[1]]

        if '发布消息' in sentence and (data[0][0] == '发布' or data[1][0] == '发布'):
            print('=============================================')
            print('触发规则，去除发布')
            print('原文：%s' % (sentence))
            print('%s %s' % (data[0][0], data[1][0]))

            return [_ for _ in data if _[0] != '发布']

        if '播放' in sentence and (data[0][0] == '播放' or data[1][0] == '播放'):
            print('=============================================')
            print('触发规则，去除播放')
            print('原文：%s' % (sentence))
            print('%s %s' % (data[0][0], data[1][0]))

            return [_ for _ in data if _[0] != '播放']

        if '发表' in sentence and ('讲话' in sentence or '文章' in sentence) and (data[0][0] == '发表' or data[1][0] == '发表'):
            print('=============================================')
            print('触发规则，仅选择‘发表’')
            print('原文：%s' % (sentence))
            print('%s %s' % (data[0][0], data[1][0]))

            return [_ for _ in data if _[0] == '发表']

        if data[0][0] in data[1][0]:
            print('=============================================')
            print('原文：%s' % (sentence))
            print('%s %s => %s' % (data[0][0], data[1][0], data[0][0]))
            return [data[0]]
        elif data[1][0] in data[0][0]:
            print('=============================================')
            print('原文：%s' % (sentence))
            print('%s %s => %s' % (data[0][0], data[1][0], data[1][0]))
            return [data[1]]
        else:
            print('=============================================')
            print('原文：%s' % (sentence))
            print('%s %s 不改变' % (data[0][0], data[1][0]))
            return data
    else:
        print('=============================================')
        print('原文：%s' % (sentence))
        print('数量：%d，不满足条件，不修改 %s' % (len(data), str([_[0] for _ in data])))
        return data


def clearn_trigger_v2(data):
    rt = []
    for x in data:
        if len(x[0]) > 4:
            print('%s => %s' % (x[0], x[0][:2]))
            rt.append([x[0][:2], x[1], 2])
        else:
            rt.append(x)

    return rt

if __name__ == '__main__':
    name = ['bert_base_event', 'roberta_large_event', 'albert_xlarge_event']
    gpu_id = '1'
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    for path in name:
        print('加载模型：./model/%s' % path)
        print('保持预测结果：./sub/%s' % (path + '.json'))
        config = read_json_v2('./model/%s/config.json' % (path))
        print('config:%s' % (json.dumps(config, ensure_ascii=False)))


        dir = config['dir']
        max_length = config['seq_length']
        batch_size = config['batch_size']

        model = Ner(config)

        vocab = FullTokenizer(dir+'/vocab.txt')

        train_dataset, init_call, count = get_data(
            get_dataset_by_id(['./data/train.json', './data/dev.json'], batch_size, vocab, max_length), max_length=max_length)

        dev_data = get_dataset_by_id(['./data/dev.json'], 1, vocab, max_length, True)

        ckpt = tf.train.Checkpoint(model=model)
        ckpt.restore(tf.train.latest_checkpoint('./model/'+path))


        data = read_json('./sub/%s.json'%(path.replace('event','trigger')))

        """
        清洗触发词
        
        1. 有重复，选短的
        
        2. 过长的选前2个
        """

        for x in data:
            x['trigger'] = clearn_trigger(x['sentence'], x['trigger'])

        for x in data:
            x['trigger'] = clearn_trigger_v2(x['trigger'])


        dev_data = get_dev_data(data, 4, vocab, max_length)

        test_data = []
        for data in dev_data():
            test_data.append(data)

        tag_map = ['o', 'trigger', 'subject', 'object', 'location', 'time']
        rt = []
        for x in tqdm(test_data):
            test_data_, raw, content_ids_all, masks = x
            rt_ = dev_step([raw, content_ids_all, masks], model)
            events = []
            for i, _ in enumerate(rt_):
                trigger_offset = test_data_[i]['trigger'][1]
                events.append([])
                for __ in _:
                    events[-1].append({'role':__[2], 'text': ''.join(__[0]), 'offset': __[1][0]-1 if __[1][0]-1<trigger_offset else __[1][0]-3, 'length':len(__[0])})

            sub_data = []
            for i in range(len(test_data_)):
                test_data__ = test_data_[i]
                c = {'sentence':test_data__['sentence'],
                                 'words':test_data__['words'],
                                 'event':{
                                     'trigger':{
                                         'text':test_data__['trigger'][0],
                                         'offset': test_data__['trigger'][1],
                                         'length':test_data__['trigger'][2]},
                                     'arguments': events[i]
                                 }
                                }
                sub_data.append(c)

            rt.extend(sub_data)


        # 清理部分重复论元以及去除tag
        rt_process = []
        for x in rt:
            arguments = []
            arguments_role = []
            Flag = False
            time_index = -1
            for _ in x['event']['arguments']:
                if _['role'] in arguments_role:
                    Flag = True
                    if _['role']=='time':
                        arguments[time_index] = _
                    continue

                if _['role']=='time':
                    time_index = len(arguments)
                arguments_role.append(_['role'])
                arguments.append(_)
            x['event']['arguments_'] = arguments
            if Flag:
                sentence = x['sentence']
                event = [x['event']['trigger']['text']]+[_['text']+' '+_['role'] for _ in x['event']['arguments_']]

                print(sentence)
                print(str(event))
                print('=================================')

        with open('./sub/%s.json'%(path), 'w', encoding='utf-8') as f:
            f.write(json.dumps(rt, ensure_ascii=False))