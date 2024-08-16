import fastwer
import torch
import torch.nn as nn
from itertools import groupby
import config

class WERMetric():
    def __init__(self):
        pass

    def __call__(self,inputs,target):
        cer = 0.0
        for input_,target_ in zip(inputs,target):
            cer+= fastwer.score_sent(''.join(input_),''.join(target_),char_level=True)
            # print(''.join(input_))
        cer/=len(inputs)

        return cer


class CharacterErrorRate(nn.Module):
    def __init__(self):
        super(CharacterErrorRate, self).__init__()

    def forward(self, predictions, targets):
        # Convert tensors to lists of strings
        predictions_str = [''.join(prediction) for prediction in predictions]
        targets_str = [''.join(target) for target in targets]

        # Compute character error rate using fastwer
        cer = fastwer.score(predictions_str, targets_str, char_level=True)

        return torch.tensor(cer, dtype=torch.float32)

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