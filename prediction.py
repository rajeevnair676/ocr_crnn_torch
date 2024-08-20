import torch
import config
import os
from data_loader import CustomDataset
from torch.utils.data import DataLoader
from itertools import groupby
import numpy as np

def ctc_decoder(predictions):
    '''
    input: given batch of predictions from text rec model
    output: return lists of raw extracted text

    '''
    text_list = []
    
    pred_indcies = torch.argmax(predictions, axis=2)
    
    for i in range(pred_indcies.shape[0]):
        ans = ""
        
        ## merge repeats
        merged_list = [k for k,_ in groupby(pred_indcies[i])]
        
        ## remove blanks
        for p in merged_list:
            if p != len(config.TOKENIZER.vocab):
                ans += config.TOKENIZER.vocab[int(p)]
        
        text_list.append(ans)
        
    return text_list

def collate_func(batch):
    # (B, [images, encodings, lengths])
    images, encodings, lengths = zip(*batch)
    images = torch.stack(images)
    encodings = torch.cat(encodings, dim=-1)
    lengths = torch.cat(lengths, dim=-1)
    return images, encodings, lengths

model = torch.load(os.path.join(config.OUTPUT_MODEL_PATH)).to(config.DEVICE)
# print(model)


test_data = CustomDataset(r'data\Synthetic_Rec_En_V2',
                          config.IMAGE_HEIGHT,
                          config.IMAGE_WIDTH,
                          config.TOKENIZER,
                          train=False,
                          test=False
                          )
test_loader = DataLoader(test_data, config.BATCH_SIZE, shuffle=True,collate_fn=collate_func)

for i,(image,label,lens) in enumerate(test_loader):
    print(image.shape)
    predictions = model(image.to(config.DEVICE))

    pred_dec = ctc_decoder(predictions)
    labels = config.TOKENIZER.decode1D(label,lens)

    # out = torch.argmax(predictions,dim=2).permute(1,0)
    # out_decode = config.TOKENIZER.batch_decode(out)
    print("Predicted:" ,pred_dec)
    # print("Target :",labels)
    print("Target :")
    for i in range(config.BATCH_SIZE):
        print(''.join(label for label in labels[i]))
