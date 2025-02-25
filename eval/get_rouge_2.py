import torch
import utils
import json
import CLM_model

device = 'cuda'



file = open('config', 'r', encoding='utf-8')
config = json.load(file)

SRC_IDS = torch.load(config.test.src_token_thr, weights_only=True)
SRC_MSK = torch.load(config.test.src_atte_msk_thr, weights_only=True)

file = open(config.test.new_src_thr, 'r', encoding='utf-8')
SRC = json.load(file)
file = open(config.test.tgt, 'r', encoding='utf-8')
TGT = json.load(file)


Model_txt = CLM_model.Bart_Model(config.txt.tgt_max_len).to(device)
Model_txt.load_state_dict(torch.load(r'txt.pth', weights_only=True))


a = 0
b = 0
c = 0

for i in range(int(config.test.len/100)):
        begin = i*100
        end = (i+1)*100

        src_ids = SRC_IDS[begin:end].to(device)
        src_msk = SRC_MSK[begin:end].to(device)
        pred_ids = Model_txt.get_pred_token(src_ids, src_msk)
    
        src = SRC[begin:end]
        tgt = TGT[begin:end]
        s = utils.show_result(src, pred_ids, tgt)

        a +=s['rouge-1']
        b +=s['rouge-2']
        c +=s['rouge-l']


print(int(config.test.len/100))
print(int(config.test.len/100))
print(int(config.test.len/100))   