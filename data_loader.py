import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
from encode_decode import Tokenizer


class CustomDataset(Dataset):
    def __init__(self,data_path,input_height,input_width,tokenizer:Tokenizer,train=True,test=False):
        self.data_path = data_path
        self.input_height = input_height
        self.input_width = input_width
        self.tokenizer = tokenizer
        self.train=train
        self.test=test
        self.img_labels = []
        
        if self.test:
            self.img_names = [img for img in os.listdir(self.data_path)]

        else:
            if self.train:
                label_file = os.path.join(self.data_path,f'train/gt.txt')
            else:
                label_file = os.path.join(self.data_path,f'test/gt.txt')
            
            with open(label_file,'r') as f:
                for line in f.readlines():
                    self.img_labels.append((line.split('\t')[0],line.split('\t')[1].rstrip()))
            

    def __len__(self):
        if self.test:
            return len(os.listdir(self.data_path))
        else:
            return len(self.img_labels)

    def __getitem__(self, index):
        
        if self.test:
            #Give the images path direct in case of test
            img_name = self.img_names[index]
            images = cv2.imread(os.path.join(self.data_path,img_name))
            images = self.preprocess(images)
            return torch.FloatTensor(images)
        else:
            img_name, labels = self.img_labels[index]
            if self.train:
                images = cv2.imread(os.path.join(self.data_path,"train/images/"+img_name))
                images = self.preprocess(images)
                encodings = self.tokenizer.encode(labels)
                return torch.FloatTensor(images), torch.LongTensor(encodings), torch.LongTensor([len(encodings)])
            else:
                images = cv2.imread(os.path.join(self.data_path,"test/images/"+img_name))
                images = self.preprocess(images)
                encodings = self.tokenizer.encode(labels)
                return torch.FloatTensor(images),torch.LongTensor(encodings),torch.LongTensor([len(encodings)])

    
    def preprocess(self, img):        
        h, w, _ = img.shape
        ratio = w / h
        resized_w = int(self.input_height * ratio)
        if resized_w > self.input_width:
            resized_w = self.input_width
        img = cv2.resize(img, (resized_w, self.input_height))
        # img = img[:, :, ::-1] / 255   # BGR to RGB and normalize
        img = (img[:, :, ::-1].astype(np.float32)*1./127)-1
        # img = img[:, :, ::-1].astype(np.float32)*1./255  # BGR to RGB and normalize
        
        img = img.swapaxes(0, 2) # channels first (c, w, h)
        target = np.zeros((3, self.input_width, self.input_height))
        target[:, :resized_w, :] = img
        return target

def train_test_split(img_name, split_percent=0.1):
    a,b = [],[]
    # random.shuffle(images)
    for count,img in enumerate(img_name):
        if count % int(1/split_percent)==0:
            b.append(img)
        else:
            a.append(img)
    return a,b


# if __name__=="__main__":      
#     train_data = CustomDataset(DATA_PATH,MODEL_INPUT_SHAPE[1],MODEL_INPUT_SHAPE[0])
#     print(train_data)
#     dataloader = DataLoader(train_data, batch_size=4, shuffle=True)

    # for images, labels in dataloader:
    #     print(images.size(), labels)




