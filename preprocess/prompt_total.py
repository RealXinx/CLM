import json



file = open('config', 'r', encoding='utf-8')
config = json.load(file)


def get_name_pp(pp_path1, pp_path2, pp_path3, org_path1, org_path2, org_path3):
    file = open(pp_path1, 'r', encoding='utf-8')
    prompt1 = json.load(file)

    file = open(pp_path2, 'r', encoding='utf-8')
    prompt2 = json.load(file)

    file = open(pp_path3, 'r', encoding='utf-8')
    prompt3 = json.load(file)

    file = open(org_path1, 'r', encoding='utf-8')
    origin1 = json.load(file)

    file = open(org_path2, 'r', encoding='utf-8')
    origin2 = json.load(file)

    file = open(org_path3, 'r', encoding='utf-8')
    origin3 = json.load(file)


    dic = {}
    i = 0
    for k, v in origin1.items():
        dic[k+'.jpg'] = prompt1[i]
        i+=1
    i = 0
    for k, v in origin2.items():
        dic[k+'.jpg'] = prompt2[i]
        i+=1
    i = 0
    for k, v in origin3.items():
        dic[k+'.jpg'] = prompt3[i]
        i+=1
    
    return dic

dic = get_name_pp(
    config.path.prompt_per_train,
    config.path.prompt_per_dev,
    config.path.prompt_per_test,
    config.rawpath.clothing_train, 
    config.rawpath.clothing_dev,
    config.rawpath.clothing_test)

with open(config.path.prompt_total, 'w', encoding='utf-8') as file:
        json_str = json.dumps(dic, ensure_ascii=False, indent=4)
        file.write(json_str)