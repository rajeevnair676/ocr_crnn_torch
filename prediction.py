import torch
import config
import os
from data_loader import CustomDataset
from torch.utils.data import DataLoader
from itertools import groupby
import numpy as np
import cv2

def ctc_decoder(predictions):
    '''
    input: given batch of predictions from text rec model
    output: return lists of raw extracted text

    '''
    text_list = []
    pred_indcies = torch.argmax(predictions, dim=2).permute(1,0)
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
# model = torch.load(r'models\torch\OCR_CRNN_V8_D2_32_640.pt').to(config.DEVICE)
print(f"Using model '{config.OUTPUT_MODEL_PATH}' for predictions")


test_data = CustomDataset(r"E:\EFR\Datasets\OCR_CROPS_ENGLISH(0-4000)\images",
                          config.IMAGE_HEIGHT,
                          config.IMAGE_WIDTH,
                          config.TOKENIZER,
                          mode='test'
                          )
# test_loader = DataLoader(test_data, config.BATCH_SIZE, shuffle=True,collate_fn=collate_func)
test_loader = DataLoader(test_data, config.BATCH_SIZE, shuffle=True)

for i,(image) in enumerate(test_loader):
    print(image.shape)
    predictions = model(image.to(config.DEVICE))

    pred_dec = ctc_decoder(predictions)
    # labels = config.TOKENIZER.decode1D(label,lens)

    # out = torch.argmax(predictions,dim=2).permute(1,0)
    # out_decode = config.TOKENIZER.batch_decode(out)
    
    # print("Target :",labels)
    for i in range(config.BATCH_SIZE):
        img = image[i,:,:,:]
        img = img.permute(2,1,0).numpy()
        img = ((img[:,:,::-1]+1.0)*127.0).astype(np.uint8)
        print("Predicted:",pred_dec[i])
        cv2.imshow("Label",img)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
        print()
