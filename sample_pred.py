import config
import random
import torch
from metric import WERMetric
from data_loader import CustomDataset
from torch.utils.data import DataLoader


# pred = [[random.randint(0, 94) for _ in range(10)] for _ in range(20)]
# # print(pred)
# print(config.TOKENIZER.vocab[95])
print(config.TOKENIZER.vocab_size)
# # print(config.TOKENIZER.batch_decode(pred))

# def collate_func(batch):
#     # (B, [images, encodings, lengths])
#     images, encodings, lengths = zip(*batch)
#     images = torch.stack(images)
#     encodings = torch.cat(encodings, dim=-1)
#     lengths = torch.cat(lengths, dim=-1)
#     return images, encodings, lengths


# train_dataset = CustomDataset(config.DATA_PATH, config.MODEL_INPUT_SHAPE[1], config.MODEL_INPUT_SHAPE[0], config.TOKENIZER)
# train_loader = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=True, collate_fn=collate_func)


# print(next(iter(train_loader))[0].shape)