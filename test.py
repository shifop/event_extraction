from util.util import read_json
import json

def get_e(x_str_, x_str):
    if len(x_str) != len(x_str):
        return False

    for i in range(len(x_str)):
        if x_str_[i] != x_str[i]:
            return False
    return True


if __name__=='__main__':
    data1 = read_json('./sub/sub.json')
    data2 = read_json('./sub/CreativityFair_d07ebe121f88fb80d41470b438bee041_trigger_and_event_and_dt_0920_roberta_large_alber_xlarge_bert_base_m_v2.json')

    error_ = []
    for i in range(len(data1)):
        events1 = data1[i]['events']
        events2 = data2[i]['events']

        for _ in events1:
            _['arguments'] = sorted(_['arguments'], key=lambda x:str(x))

        for _ in events2:
            _['arguments'] = sorted(_['arguments'], key=lambda x:str(x))

        events1 = sorted([[_['trigger'], _['arguments'], _['tense'], _['polarity']] for _ in events1], key=lambda x:str(x))
        events2 = sorted([[_['trigger'], _['arguments'], _['tense'], _['polarity']] for _ in events2], key=lambda x:str(x))


        x_str = json.dumps(events1, ensure_ascii=False)
        x_str_ = json.dumps(events2, ensure_ascii=False)

        if not get_e(x_str, x_str_):
            error_.append(i)

    print(len(error_))