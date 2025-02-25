import lawrouge
import torch
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


file = open('config', 'r', encoding='utf-8')
config = json.load(file)

tokenizer = AutoTokenizer.from_pretrained(config.download.bart)    


def text_to_ids(text, max_length):
    x = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
    )   
        
    ids = x.input_ids
    attention_mask = x.attention_mask

    return (ids, attention_mask)


def ids_to_text(decode_ids, target=None):
    output_str = tokenizer.batch_decode(decode_ids, skip_special_tokens=True)
    output_str = [s.replace(" ","") for s in output_str]
        
    if target is not None:
        rouge = lawrouge.Rouge()
        score = rouge.get_scores(output_str, target, avg=True)

        score = {'rouge-1': score['rouge-1']['f'], 'rouge-2': score['rouge-2']['f'], 'rouge-l': score['rouge-l']['f']}
        score = {key: value * 100 for key, value in score.items()}

        return (output_str, score)
        
    return output_str


def get_batch(src_ids, src_atte_msk, tgt_ids, tgt_atte_msk, batch_size):
    src = torch.cat([src_ids, src_atte_msk], dim=1)
    tgt = torch.cat([tgt_ids, tgt_atte_msk], dim=1)
    
    src = DataLoader(src, shuffle=False, batch_size=batch_size)
    tgt = DataLoader(tgt, shuffle=False, batch_size=batch_size)
    
    return (src, tgt)


def show_result(orig, pred_ids, tgt):
    pred, score = ids_to_text(pred_ids.to('cpu'), tgt)
    
    for i in range(len(tgt)):
        print(f'original:\n{orig[i]}')
        print(f'\npred:\n{pred[i]}')
        print(f'\ntarget:\n{tgt[i]}')
        print('\n')

    print(f'\nscore:')
    print(score)
    
    return score