import json
import csv
# import pandas as pd

def read_json(paths):
    if not isinstance(paths, list):
        paths = [paths]
    data = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            data.extend(json.loads(f.read()))
    return data

def read_json_v2(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    return data

def read_csv(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data



def map_to_stock_variable_name(name, prefix="bert"):

    name_map={
        'bert/embeddings/word_embeddings/embeddings':'',
        'bert/embeddings/word_embeddings_projector/projector': 'bert/encoder/embedding_hidden_mapping_in/bias',
        'bert/embeddings/word_embeddings_projector/bias': 'bert/encoder/embedding_hidden_mapping_in/kernel',
        'bert/embeddings/token_type_embeddings/embeddings': '',
        'bert/embeddings/position_embeddings/embeddings': '',
        'bert/embeddings/LayerNorm/gamma': 'bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta',
        'bert/embeddings/LayerNorm/beta': 'bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma',
        'bert/encoder/layer_shared/attention/self/query/kernel': 'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel',
        'bert/encoder/layer_shared/attention/self/query/bias': 'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias',
        'bert/encoder/layer_shared/attention/self/key/kernel': 'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel',
        'bert/encoder/layer_shared/attention/self/key/bias': 'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias',
        'bert/encoder/layer_shared/attention/self/value/kernel': 'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel',
        'bert/encoder/layer_shared/attention/self/value/bias': 'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias',
        'bert/encoder/layer_shared/attention/output/dense/kernel': 'bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel',
        'bert/encoder/layer_shared/attention/output/dense/bias': 'bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias',
        'bert/encoder/layer_shared/attention/output/LayerNorm/gamma': '',
        'bert/encoder/layer_shared/attention/output/LayerNorm/beta': '',
        'bert/encoder/layer_shared/intermediate/kernel': '',
        'bert/encoder/layer_shared/intermediate/bias': '',
        'bert/encoder/layer_shared/output/LayerNorm/gamma': '',
        'bert/encoder/layer_shared/output/LayerNorm/beta':''

    }
    name = name.split(":")[0]
    ns   = name.split("/")
    pns  = prefix.split("/")

    if ns[:len(pns)] != pns:
        return None

    name = "/".join(["bert"] + ns[len(pns):])
    ns   = name.split("/")

    if ns[1] not in ["encoder", "embeddings"]:
        return None
    if ns[1] == "embeddings":
        if ns[2] == "LayerNorm":
            return name
        elif ns[2] == "word_embeddings_projector":
            ns[2] = "word_embeddings_2"
            if ns[3] == "projector":
                ns[3] = "embeddings"
                return "/".join(ns[:-1])
            return "/".join(ns)
        else:
            return "/".join(ns[:-1])
    if ns[1] == "encoder":
        if ns[3] == "intermediate":
            return "/".join(ns[:4] + ["dense"] + ns[4:])
        else:
            return name
    return None

if __name__=='__main__':
    data =read_csv('../data/sub/sub_v2.csv')
    # for x in data:
    #     if len(''.join(x))<10:
    #         print('')

    data2 = read_csv('../data/sub/事件抽取挑战赛_事件抽取挑战赛提交样例.csv')
    data2 = set([x[0] for x in data2][1:])
    cache = set()
    with open('../data/sub/sub_v3.csv','w',encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['id', 'trigger', 'object', 'subject', 'time', 'location'])

        save_data = []
        lid = ''
        for x in data:
            if len(''.join(x[2:]))!=0 and x[1]!='':
                if ' '.join(x[:2]) not in cache and len(x[1])<=3:
                    cache.add(' '.join(x[:2]))
                    save_data.append(x)

        data_1 = set([x[0] for x in save_data])
        for x in save_data:
            csv_writer.writerow(x)
        for x in data2:
            if x not in data_1:
                csv_writer.writerow([str(x)]+['恢复','深航','湖北其他城市的航班运力','',''])


    # df = pd.read_csv('../data/sub/sub_p.csv', skipinitialspace=True)
    # df.to_csv('../data/sub/sub_p2.csv', index=False, encoding='utf-8', sep=',')

    # data2 = read_csv('../data/sub/sub.csv')
    # data = read_csv('../data/sub/事件抽取挑战赛_事件抽取挑战赛提交样例 (2).csv')
    #
    # index = [x[0] for x in data2]
    # for x in data:
    #     if x[0] not in index:
    #         print(x[0])

    # print('')

    # with open('../data/sub/sub2.csv','w',encoding='utf-8') as f:
    #     f.write(','.join(['id', 'trigger', 'object', 'subject', 'time', 'location'])+'\n')
    #     for x in data:
    #         if len(''.join(x[2:]))!=0 and x[1]!='' and len(x[1])<3:
    #             f.write(','.join(x)+'\n')