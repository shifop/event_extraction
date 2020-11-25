import json
import csv
# import pandas as pd

def read_json(paths):
    data = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            data.extend(json.loads(f.read()))
    return data

def read_csv(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data


def check(content, data):
    for x in data:
        if x!=content and content in x:
            return True
    return False

if __name__=='__main__':
    data =read_csv('../data/sub/sub.csv')
    # for x in data:
    #     if len(''.join(x))<10:
    #         print('')

    data2 = read_csv('../data/sub/事件抽取挑战赛_事件抽取挑战赛提交样例.csv')
    data2 = set([x[0] for x in data2][1:])
    cache = []
    cache_set = set()
    with open('../data/sub/sub_v2.csv','w',encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['id', 'trigger', 'object', 'subject', 'time', 'location'])

        save_data = []
        lid = ''
        for x in data:
            if len(''.join(x[2:]))!=0 and x[1]!='':
                if x[0] !=lid and len(cache)!=0:
                    # 删除重复部分
                    cache_ = [' '.join([__ for __ in _[1:] if __!='']) for _ in cache]
                    cache = [_ for _ in cache if not check(' '.join([__ for __ in _[1:] if __!='']), cache_)]
                    save_data.extend(cache)
                    cache = []
                if ' '.join(x) not in cache_set:
                    cache_set.add(' '.join(x))
                    cache.append(x)
                lid = x[0]

        save_data.extend(cache)

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