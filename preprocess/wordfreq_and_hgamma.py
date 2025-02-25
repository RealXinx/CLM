import json
import re
import jieba.posseg as pseg
from collections import Counter



file = open('config', 'r', encoding='utf-8')
config = json.load(file)


def raw_tgt_TO_vocab_list(p1, p2, p3):
    list = []

    file = open(p1, 'r', encoding='utf-8')
    data = json.load(file) 
    for k, v in data.items():
        list += v['tgt']
    
    file = open(p2, 'r', encoding='utf-8')
    data = json.load(file) 
    for k, v in data.items():
        list += v['tgt']
        
    file = open(p3, 'r', encoding='utf-8')
    data = json.load(file) 
    for k, v in data.items():
        list += v['tgt']
        
    str = ' '.join(list)
    pattern = re.compile(r'[\u4e00-\u9fff\s]+')
    matches = pattern.findall(str)
    result = ''.join(matches)
    
    return result.split()


def is_adjective_or_noun(vocab_list):
    adj = []
    n = []

    for word in vocab_list:
        word = pseg.cut(word)
        for w, flag in word:
            if flag.startswith('a'):
                adj.append(w)
            if flag.startswith('n'):
                n.append(w)
            else:
                continue
    
    return (adj, n)



def h_freq_word(file_pth):
    word_list = []
    with open(file_pth, 'r', encoding='utf-8') as file:
        for line in file:
            a = line.strip()
            a = a.split(' ')

            if int(a[1]) >= config.gamma:
                word_list.append(a[0])
    
    return word_list



vocab_list = raw_tgt_TO_vocab_list(
    config.rawpath.clothing_train, 
    config.rawpath.clothing_dev,
    config.rawpath.clothing_test
)

count_adj_list, count_n_list = is_adjective_or_noun(vocab_list)
adj_n_list = count_adj_list + count_n_list
word_freq = Counter(adj_n_list)

with open(config.path.word_all, 'w', encoding='utf-8') as file:
    for k, v in word_freq.items():
        file.write(k + ' ' + str(v) + '\n')

word_list = h_freq_word(config.path.word_all)

with open(config.path.h_gamma, 'w', encoding='utf-8') as file:
        json_str = json.dumps(word_list, ensure_ascii=False, indent=4)
        file.write(json_str)