from util.util import read_json
import json

def get_event_str(data):
    _ = data
    return str([_['trigger']['text']]+[__['text']+' '+__['role'] for __ in _['arguments']])


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


if __name__=='__main__':
    data1 = read_json('./sub/albert_xlarge_event.json')
    data2 = read_json('./sub/bert_base_event.json')
    data3 = read_json('./sub/roberta_large_event.json')
    test_raw = read_json('./data/raw/测试集/sentences.json')

    data_ = [data1, data2, data3]
    process_data = []
    for rt in data_:
        # 清除[unused1]
        for x in rt:
            arguments = []
            for _ in x['event']['arguments_']:
                _['text'] = _['text'].strip()
                if '[unused1]' in _['text']:
                    print('清除‘[unused1]’：%s' % (_['text']))
                    _['text'] = _['text'].split('[unused1]')[0].split('，')[0]
                    if _['role'] == 'location':
                        _['text'] = _['text'].split('的')[0]
                    if _['role'] == 'time':
                        _['text'] = get_time(_['text'].split('：')[0])

                    if _['role'] == 'time' and len(set(list('周日月年时分季代')) & set(list(_['text']))) == 0:
                        _['text'] = 'unused'
                    _['length'] = len(_['text'])
                    print('清除后：%s ：%s' % (_['text'], _['role']))
                if _['text'] != 'unused':
                    arguments.append(_)
            x['event']['arguments'] = arguments
            del x['event']['arguments_']

        save_data = []
        for i, data in enumerate(rt):
            event_save = [data['event']]
            save_data.append({'sentence': data['sentence'], 'words': data['words'], 'event': event_save})

        save_data_ = {}
        for x in save_data:
            for _ in x['event']:
                for __ in _['arguments']:
                    if __['role'] == 'location':
                        __['role'] = 'loc'

            if x['sentence'] not in save_data_:
                save_data_[x['sentence']] = {'sentence': x['sentence'], 'words': x['words'], 'events': x['event']}
            else:
                save_data_[x['sentence']]['events'].extend(x['event'])

        sub_data = []
        for x in test_raw:
            key = x['sentence']
            if key in save_data_:
                sub_data.append(save_data_[key])
            else:
                sub_data.append('None')

        process_data.append(sub_data)

    data_ = process_data
    # 投票选择论元
    rt = []
    error_ = []
    data1, data2, data3 = data_

    for i in range(len(data1)):
        data = [_[i] for _ in data_ if _[i] != 'None']
        if len(data) == 0:
            error_.append(i)
            rt.append('None')
            continue

        sentence = data[0]['sentence']
        words = data[0]['words']

        trigger = {}
        trigger_c = {}
        events = [_['events'] for _ in data]
        for event in events:
            for x in event:
                t = x['trigger']['text']
                if t not in trigger:
                    trigger[t] = {'subject': {}, 'object': {}, 'time': {}, 'loc': {}, 'trigger': x['trigger']}
                    trigger_c[t] = 1
                else:
                    trigger_c[t] += 1
                for _ in x['arguments']:
                    atext = _['text']
                    if atext not in trigger[t][_['role']]:
                        trigger[t][_['role']][atext] = []
                    trigger[t][_['role']][atext].append(_)

        trigger_c = [k for k in trigger_c if trigger_c[k] != 1]
        if len(trigger_c) == 0:
            error_.append(i)
            rt.append('None')
            continue
        nevent = []
        for key in trigger_c:
            cache = trigger[key]
            trigger_ = []
            for k in cache:
                if k in ['trigger', 'tense', 'polarity'] or len(cache[k]) == 0:
                    continue
                # 选择最多的
                ks = [_ for _ in cache[k]]
                ks_count = [len(cache[k][_]) for _ in cache[k]]
                trigger_.append(cache[k][ks[ks_count.index(max(ks_count))]][0])

            nevent.append({'trigger': cache['trigger'], 'arguments': trigger_})
        s = {'sentence': sentence, 'words': words, 'event': nevent}
        rt.append(s)

    for i, x in enumerate(rt):
        if x == 'None':
            rt[i] = {'sentence': test_raw[i]['sentence'], 'words': test_raw[i]['words'], 'event': []}

    print(len(rt))
    with open('./sub/all.json',  'w', encoding='utf-8') as f:
        f.write(json.dumps(rt, ensure_ascii=False))

