import json
import re



file = open('config', 'r', encoding='utf-8')
config = json.load(file)


def raw_tgt_TO_vocab_list_for_per_src(p1, p2, p3):
    total_list_train = []
    file = open(p1, 'r', encoding='utf-8')
    data = json.load(file) 
    for k, v in data.items():
        str_per = v['tgt'][0]
        pattern = re.compile(r'[\u4e00-\u9fff\s]+')
        matches = pattern.findall(str_per)
        result = ''.join(matches)
        per_list = result.split()
        total_list_train.append(per_list)

    total_list_dev = []
    file = open(p2, 'r', encoding='utf-8')
    data = json.load(file) 
    for k, v in data.items():
        str_per = v['tgt'][0]
        pattern = re.compile(r'[\u4e00-\u9fff\s]+')
        matches = pattern.findall(str_per)
        result = ''.join(matches)
        per_list = result.split()
        total_list_dev.append(per_list)

    total_list_test = []
    file = open(p3, 'r', encoding='utf-8')
    data = json.load(file) 
    for k, v in data.items():
        str_per = v['tgt'][0]
        pattern = re.compile(r'[\u4e00-\u9fff\s]+')
        matches = pattern.findall(str_per)
        result = ''.join(matches)
        per_list = result.split()
        total_list_test.append(per_list)

    return (total_list_train, total_list_dev, total_list_test)


def get_promt_for_per_str(train, dev, test, word_h_path):
    file = open(word_h_path, 'r', encoding='utf-8')
    data = json.load(file)

    total_train = []
    for line in train:
        str_list = []
        for word in line:
            if word in data:
                str_list.append(word)
        if len(str_list)==0:
            str_list.append('sorry')
        total_train.append(str_list)

    total_dev = []
    for line in dev:
        str_list = []
        for word in line:
            if word in data:
                str_list.append(word)
        if len(str_list)==0:
            str_list.append('sorry')
        total_dev.append(str_list)

    total_test = []
    for line in test:
        str_list = []
        for word in line:
            if word in data:
                str_list.append(word)
        if len(str_list)==0:
            str_list.append('sorry')
        total_test.append(str_list)

    return total_train, total_dev, total_test


def reset_src(p1, train):
    file = open(p1, 'r', encoding='utf-8')
    data = json.load(file)
    train_src = []
    for k, v in data.items():
        origin_src = v['src'].replace(' ', '')
        train_src.append(origin_src)   
    train_prompt = []
    for line in train:
        if line[0] == 'sorry':
            str = ' '
        else:
            str = ','.join(line)
        train_prompt.append(str)
    for i in range(len(train_prompt)):
        str = train_prompt[i]
        src = train_src[i]
        if str == ' ':
            continue
        else:
            train_src[i] = str + ',' + src

    return train_src


train, dev, test = raw_tgt_TO_vocab_list_for_per_src(
    config.rawpath.clothing_train, 
    config.rawpath.clothing_dev,
    config.rawpath.clothing_test
)

train, dev, test = get_promt_for_per_str(train, dev, test, config.path.h_gamma)

with open(config.path.prompt_per_train, 'w', encoding='utf-8') as file:
        json_str = json.dumps(train, ensure_ascii=False, indent=4)
        file.write(json_str)
    
with open(config.path.prompt_per_dev, 'w', encoding='utf-8') as file:
        json_str = json.dumps(dev, ensure_ascii=False, indent=4)
        file.write(json_str)

with open(config.path.prompt_per_test, 'w', encoding='utf-8') as file:
        json_str = json.dumps(test, ensure_ascii=False, indent=4)
        file.write(json_str)

train_src = reset_src(config.rawpath.clothing_train, train)
dev_src = reset_src(config.rawpath.clothing_dev, dev)
test_src = reset_src(config.rawpath.clothing_test, test)

with open(config.path.new_src_train, 'w', encoding='utf-8') as file:
        json_str = json.dumps(train_src, ensure_ascii=False, indent=4)
        file.write(json_str)

with open(config.path.new_src_dev, 'w', encoding='utf-8') as file:
        json_str = json.dumps(dev_src, ensure_ascii=False, indent=4)
        file.write(json_str)

with open(config.path.new_src_test, 'w', encoding='utf-8') as file:
        json_str = json.dumps(test_src, ensure_ascii=False, indent=4)
        file.write(json_str)