import torch
import utils
import CLM_model
import json
device = 'cuda'



file = open('config', 'r', encoding='utf-8')
config = json.load(file)


src_ids = torch.load(config.path.train_src_token, weights_only=True)
src_atte_msk = torch.load(config.path.train_src_atte_msk, weights_only=True)
tgt_ids = torch.load(config.path.train_tgt_token, weights_only=True)
tgt_atte_msk = torch.load(config.path.train_tgt_atte_msk, weights_only=True)

srcs, tgts = utils.get_batch(src_ids, src_atte_msk, tgt_ids, tgt_atte_msk, config.txt.batch_size)


model = CLM_model.Bart_Model(config.txt.tgt_max_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.txt.lr)
criterion = torch.nn.CrossEntropyLoss(ignore_index=utils.tokenizer.pad_token_id)


for epoch in range(config.txt.epoch):
    for src, tgt in zip(srcs, tgts):
        src_ids, src_atte_msk = src[:, :config.txt.src_max_len].to(device), src[:, config.txt.src_max_len:].to(device)
        tgt_ids, tgt_atte_msk = tgt[:, :config.txt.tgt_max_len].to(device), tgt[:, config.txt.tgt_max_len:].to(device)
        pred = model(src_ids, src_atte_msk, tgt_ids, tgt_atte_msk)

        optimizer.zero_grad()
        loss = criterion(pred.view(-1, pred.shape[-1]), tgt_ids.view(-1))
        loss.backward()
        optimizer.step()
        print(loss)