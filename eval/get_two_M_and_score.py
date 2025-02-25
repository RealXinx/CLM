import json
import torch
import torch
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import precision_score, recall_score, f1_score

import CLM_model



# beta12 exchange
device = "cuda" if torch.cuda.is_available() else "cpu"

file = open('config', 'r', encoding='utf-8')
config = json.load(file)


Model_vi = CLM_model.ACLIP(config.vis.hidden_dim).to(device)
Model_vi.load_state_dict(torch.load(r"vis.pth", weights_only=True))


file = open(config.path.ids_chin_eng, 'r', encoding='utf-8')
dict = json.load(file)

file = open(config.path.prompt_total, 'r', encoding='utf-8')
prompt = json.load(file)

file = open(config.path.atrb_eng, 'r', encoding='utf-8')
atrb_eng = json.load(file)


def chin_ids(prompt, dict):
    label = torch.zeros(len(dict))
    for i in range(len(dict)):
        if dict[str(i)][0] in prompt:
            label[i] = 1

    return label


def transform(image):
    inputs = Model_vi.preprocess(text=atrb_eng, images=image, return_tensors="pt", padding=True)
    
    return inputs


def get_score(pred, true, th):
    mask = pred > th
    pred = mask.int().to('cpu').numpy()
    true = true.int().to('cpu').numpy()

    a = precision_score(true, pred, average=config.vis.score_average)*100
    b = recall_score(true, pred, average=config.vis.score_average)*100
    c = f1_score(true, pred, average=config.vis.score_average)*100

    print(a)
    print(b)
    print(c)

    return a,b,c


def get_ids_from_th(pred, th):
    mask = pred.sigmoid() > th
    mask = mask.int()
    
    return mask


def get_pred_beta1(ids):
    list = []
    for line in ids:
        promt = []
        for i in line:
            promt.append(dict[str(i.item())][0])

        promt = ",".join(promt)
        list.append(promt)

    return list


def get_pred_beta2(ids):
    list = []
    for line in ids:
        promt = []
        for i in range(len(line)):
            if line[i] >0:
                promt.append(dict[str(i)][0])

        promt = ",".join(promt)
        list.append(promt)

    return list


class MyDataset(Dataset):
    def __init__(self, path):
        super(Dataset, self).__init__() 
        self.root_dir = path
        self.image_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, i):
        image_name = self.image_files[i]
        image_promt = prompt.get(image_name)
        pp_label = chin_ids(image_promt, dict)
        
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        inputs = transform(image)
        
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'label': pp_label
        }

dataset = MyDataset(config.rawpath.img_test)
datas = DataLoader(dataset, batch_size=int(config.test.len/100), shuffle=False)


top_ids_1 = []
top_ids_2 = []

pre = 0
rec = 0
f1c = 0

for i, d in enumerate(datas):
    input_ids = d['input_ids'][0].to(device)
    attention_mask = d['attention_mask'][0].to(device)
    pixel_values = d['pixel_values'].to(device)
    label = d['label']

    x = {'input_ids':input_ids,
         'attention_mask':attention_mask,
         'pixel_values':pixel_values}

    pred = Model_vi(x)

    a,b,c = get_score(pred, label, 0.1)

    pre+=a
    rec+=b
    f1c+=c

    top_values, top_indices = torch.topk(pred.sigmoid(), k=4, dim=1)
    top_ids_1.append(top_indices)

    top_indices = get_ids_from_th(pred, 0.1)
    top_ids_2.append(top_indices)
    

top_ids_1 = torch.cat(top_ids_1, dim=0)
top_ids_2 = torch.cat(top_ids_2, dim=0)


print(pre/100)
print(rec/100)
print(f1c/100)

prompt_1 = get_pred_beta1(top_ids_1.cpu())
prompt_2 = get_pred_beta2(top_ids_2.cpu())

with open(config.test.prompt_per_tpk, 'w', encoding='utf-8') as file:
        json_str = json.dumps(prompt_1, ensure_ascii=False, indent=4)
        file.write(json_str)

with open(config.test.prompt_per_thr, 'w', encoding='utf-8') as file:
        json_str = json.dumps(prompt_2, ensure_ascii=False, indent=4)
        file.write(json_str)