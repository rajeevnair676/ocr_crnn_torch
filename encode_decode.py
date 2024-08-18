import os
import torch

class Tokenizer():
    def __init__(self,label_path,max_length):
        self.image_labels=[]

        with open(label_path,'r') as f:
            for line in f.readlines():
                self.image_labels.append(line.split('\t')[1].rstrip())

        self.vocab = sorted(set("".join(map(str, self.image_labels))))
        self.vocab.append('')
        self.vocab_size = len(self.vocab)
        
        self.idx2wrd  = {k:v for k,v in enumerate(self.vocab)}
        self.wrd2idx = {v:k for k,v in enumerate(self.vocab)}
        self.max_length = max_length

    def encode(self,text):
        encoded_text = [self.wrd2idx[char] for char in text]
        if len(encoded_text)>self.max_length:
            encoded_text = encoded_text[:self.max_length]
        return encoded_text
    
    def decode(self,input_ids):
        decoded_text = [self.idx2wrd[int(ids)] for ids in input_ids]
        return decoded_text
    
    def batch_encode(self, batch_text):
        total_text = []
        for txt in batch_text:
            encode_text = torch.LongTensor(self.encode(txt))
            total_text.append(encode_text)
        return torch.cat(total_text)
        # return torch.LongTensor([self.encode(txt) for txt in batch_text])
    
    def batch_decode(self, batch_input_ids):
        return [self.decode(ids) for ids in batch_input_ids]
    
    def decode1D(self, input_ids, input_lengths):
        st = 0
        batch_strs = []
        for length in input_lengths:
            batch_strs.append(self.decode(input_ids[st:st+length]))
            st = length
        return batch_strs



# print(img_labels[:5])
# tokens = Tokenizer(label_path,50)

# print(tokens.batch_encode(img_labels)[:5])