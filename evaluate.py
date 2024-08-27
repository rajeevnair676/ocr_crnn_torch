import config
import os
import random
import torch
from metric import WERMetric,ctc_decoder
from data_loader import CustomDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


# test_data = r"E:\EFR\Datasets\OCR_CROPS_ENGLISH(0-4000)\images"

def collate_func(batch):
    # (B, [images, encodings, lengths])
    images, encodings, lengths = zip(*batch)
    images = torch.stack(images)
    encodings = torch.cat(encodings, dim=-1)
    lengths = torch.cat(lengths, dim=-1)
    return images, encodings, lengths

eval_dataset = CustomDataset(config.REAL_VAL_DATA_PATH, 
                              config.MODEL_INPUT_SHAPE[1], 
                              config.MODEL_INPUT_SHAPE[0], 
                              config.TOKENIZER,
                              mode='eval')

eval_loader = DataLoader(eval_dataset, config.BATCH_SIZE, shuffle=True, collate_fn=collate_func)

model = torch.load(os.path.join(config.OUTPUT_MODEL_PATH)).to(config.DEVICE)
# model = torch.load(r'models\torch\OCR_CRNN_V8_D2_32_640.pt').to(config.DEVICE)
print(f"Using model '{config.OUTPUT_MODEL_PATH}' for predictions")

cer_metric = WERMetric()
CER = 0.0

pbar = tqdm(enumerate(eval_loader),total=len(eval_loader))
for i,(image,label,lens) in pbar:
    predictions = model(image.to(config.DEVICE))

    pred_dec = ctc_decoder(predictions)
    labels = config.TOKENIZER.decode1D(label,lens)

    cer = cer_metric(pred_dec,labels)
    CER+= cer
    # out = torch.argmax(predictions,dim=2).permute(1,0)
    # out_decode = config.TOKENIZER.batch_decode(out)
    desc_ = f"CER: {round(CER/(i+1),2)}"
    pbar.set_description(desc_)

total_cer = CER/len(eval_loader)
print(f"CER for model:{round(total_cer,4)}")