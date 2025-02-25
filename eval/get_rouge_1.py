import torch
import utils
import json
device = 'cuda:0'



file = open('config', 'r', encoding='utf-8')
config = json.load(file)


dataset = utils.Dataset(config.test.new_src_tpk, config.test.tgt)
SRC, TGT = dataset.get_data()


src_ids, src_atte_msk, tgt_ids, tgt_atte_msk = [], [], [], []

for i in range(int(config.test.len/100)):
    print(i)
    
    begin = i*100
    end   = (i+1)*100
    
    bsrc_ids, bsrc_atte_msk = utils.text_to_ids(SRC[begin:end], config.txt.src_max_len)
    btgt_ids, btgt_atte_msk = utils.text_to_ids(TGT[begin:end], config.txt.tgt_max_len)

    src_ids.append(bsrc_ids)
    src_atte_msk.append(bsrc_atte_msk)
    tgt_ids.append(btgt_ids)
    tgt_atte_msk.append(btgt_atte_msk)


src_ids = torch.cat(src_ids, dim=0)
src_atte_msk = torch.cat(src_atte_msk, dim=0)
tgt_ids = torch.cat(tgt_ids, dim=0)
tgt_atte_msk = torch.cat(tgt_atte_msk, dim=0)


torch.save(src_ids, config.test.src_token_tpk)
torch.save(src_atte_msk, config.test.src_atte_msk_tpk)
torch.save(tgt_ids, config.test.tgt_token_tpk)
torch.save(tgt_atte_msk, config.test.tgt_atte_msk_tpk)



dataset = utils.Dataset(config.test.new_src_thr, config.test.tgt)
SRC, TGT = dataset.get_data()


src_ids, src_atte_msk, tgt_ids, tgt_atte_msk = [], [], [], []

for i in range(int(config.test.len/100)):
    print(i)
    
    begin = i*100
    end   = (i+1)*100
    
    bsrc_ids, bsrc_atte_msk = utils.text_to_ids(SRC[begin:end], config.txt.src_max_len)
    btgt_ids, btgt_atte_msk = utils.text_to_ids(TGT[begin:end], config.txt.tgt_max_len)

    src_ids.append(bsrc_ids)
    src_atte_msk.append(bsrc_atte_msk)
    tgt_ids.append(btgt_ids)
    tgt_atte_msk.append(btgt_atte_msk)


src_ids = torch.cat(src_ids, dim=0)
src_atte_msk = torch.cat(src_atte_msk, dim=0)
tgt_ids = torch.cat(tgt_ids, dim=0)
tgt_atte_msk = torch.cat(tgt_atte_msk, dim=0)


torch.save(src_ids, config.test.src_token_thr)
torch.save(src_atte_msk, config.test.src_atte_msk_thr)
torch.save(tgt_ids, config.test.tgt_token_thr)
torch.save(tgt_atte_msk, config.test.tgt_atte_msk_thr)