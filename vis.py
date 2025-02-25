import torch
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import os

import CLM_model



device = "cuda" if torch.cuda.is_available() else "cpu"


file = open('config', 'r', encoding='utf-8')
config = json.load(file)


file = open(config.path.prompt_total, 'r', encoding='utf-8')
prompt = json.load(file)

file = open(config.path.ids_chin_eng, 'r', encoding='utf-8')
dict = json.load(file)

file = open(config.path.atrb_eng, 'r', encoding='utf-8')
atrb_eng = json.load(file)


def chin_ids(prompt, dict):
    label = torch.zeros(len(dict))
    
    for i in range(len(dict)):
        if dict[str(i)][0] in prompt:
            label[i] = 1 
    
    return label


def transform(image):
    inputs = Model.preprocess(text=atrb_eng, images=image, return_tensors="pt", padding=True)
    
    return inputs


def get_score(pred, true, th):
    mask = pred.sigmoid() > th
    
    pred = mask.int().to('cpu').numpy()
    true = true.int().to('cpu').numpy()

    print(precision_score(true, pred, average=config.vis.score_average)*100)
    print(recall_score(true, pred, average=config.vis.score_average)*100)
    print(f1_score(true, pred, average=config.vis.score_average)*100)


class MyDataset(Dataset):
    def __init__(self, root, transform):
        super(Dataset, self).__init__() 
        self.root_dir = root
        self.image_files = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, i):
        image_name = self.image_files[i]
        image_promt = prompt.get(image_name)
        pp_label = chin_ids(image_promt, dict)
        
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        inputs = self.transform(image)
        
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'label': pp_label
        }


dataset = MyDataset(config.rawpath.img_train, transform)
datas = DataLoader(dataset, batch_size=config.vis.batch_size, shuffle=False)


Model = CLM_model.ACLIP(config.vis.hidden_dim).to(device)

weight = torch.ones(len(dict)).to(device)
criterion = torch.nn.BCEWithLogitsLoss(weight=weight)

optimizer = torch.optim.Adam(Model.parameters(), lr=config.vis.lr)


for epoch in range(config.vis.epoch):
    for i, d in enumerate(datas):
        input_ids = d['input_ids'][0].to(device)
        attention_mask = d['attention_mask'][0].to(device)
        pixel_values = d['pixel_values'].to(device)
        label = d['label'].to(device)

        x = {'input_ids':input_ids,
             'attention_mask':attention_mask,
             'pixel_values':pixel_values}
        
        pred = Model(x)

        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        get_score(pred, label, 0.1)