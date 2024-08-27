import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
from encode_decode import Tokenizer


class CustomDataset(Dataset):
    def __init__(self,data_path,input_height,input_width,tokenizer:Tokenizer,mode='train'):
        self.data_path = data_path
        self.input_height = input_height
        self.input_width = input_width
        self.tokenizer = tokenizer
        self.mode=mode
        self.img_labels = []
        
        if self.mode=='test':
            self.img_names = [img for img in os.listdir(self.data_path)]

        else:
            if self.mode=='train':
                label_file = os.path.join(self.data_path,f'train/gt.txt')
            elif self.mode=='val':
                label_file = os.path.join(self.data_path,f'test/gt.txt')
            elif self.mode=='eval':
                label_file = os.path.join(self.data_path,f'gt.txt')
            
            with open(label_file,'r') as f:
                for line in f.readlines():
                    self.img_labels.append((line.split('.jpg')[0],line.split('.jpg')[1].strip()))
            

    def __len__(self):
        if self.mode=='test':
            return len(os.listdir(self.data_path))
        else:
            return len(self.img_labels)

    def __getitem__(self, index):
        
        if self.mode=='test':
            #Give the images path direct in case of test
            img_name = self.img_names[index]
            images = cv2.imread(os.path.join(self.data_path,img_name))
            images = self.preprocess(images)
            return torch.FloatTensor(images)
        else:
            img_name, labels = self.img_labels[index]
            if self.mode=='train':
                images = cv2.imread(os.path.join(self.data_path,"train/images/"+img_name+".jpg"))
                images = self.preprocess(images)
                encodings = self.tokenizer.encode(labels)
                return torch.FloatTensor(images), torch.LongTensor(encodings), torch.LongTensor([len(encodings)])
            elif self.mode=='val':
                images = cv2.imread(os.path.join(self.data_path,"test/images/"+img_name+".jpg"))
                images = self.preprocess(images)
                encodings = self.tokenizer.encode(labels)
                return torch.FloatTensor(images),torch.LongTensor(encodings),torch.LongTensor([len(encodings)])
            elif self.mode=='eval':
                images = cv2.imread(os.path.join(self.data_path,"images/"+img_name+".jpg"))
                images = self.preprocess(images)
                encodings = self.tokenizer.encode(labels)
                return torch.FloatTensor(images),torch.LongTensor(encodings),torch.LongTensor([len(encodings)])

    
    # def preprocess(self, img):        
    #     h, w, _ = img.shape
    #     ratio = w / h
    #     resized_w = int(self.input_height * ratio)
    #     if resized_w > self.input_width:
    #         resized_w = self.input_width
    #     img = cv2.resize(img, (resized_w, self.input_height))
    #     # img = img[:, :, ::-1] / 255   # BGR to RGB and normalize
    #     img = (img[:, :, ::-1].astype(np.float32)*1./127)-1
    #     # img = img[:, :, ::-1].astype(np.float32)*1./255  # BGR to RGB and normalize
        
    #     img = img.swapaxes(0, 2) # channels first (c, w, h)
    #     target = np.zeros((3, self.input_width, self.input_height))
    #     target[:, :resized_w, :] = img
    #     return target
    
    def preprocess(self, img):
        h, w, _ = img.shape
        ratio = w / h
        resized_w = int(self.input_height * ratio)  
        # Resize the width if it exceeds the input width
        isBottomPad = False
        resized_h = self.input_height
        if resized_w > self.input_width:
            resized_w = self.input_width
            resized_h = int(h / w * self.input_width)
            isBottomPad = True
        # Resize image while preserving aspect ratio
        img = cv2.resize(img, (resized_w, resized_h))
        # Convert BGR to RGB and normalize
        img = (img[:, :, ::-1] / 127.0) - 1
        img = img.swapaxes(0, 2)#.swapaxes(1, 2) # channels first (c, h, w)
        # Prepare target tensor with appropriate dimensions
        target = np.zeros((3, self.input_width,self.input_height))
        # Place the resized image into the target tensor
        if isBottomPad:
            target[:,:,:resized_h] = img
        else:
            target[:,:resized_w,:] = img
        return target
    
    

# def train_test_split(img_name, split_percent=0.1):
#     a,b = [],[]
#     # random.shuffle(images)
#     for count,img in enumerate(img_name):
#         if count % int(1/split_percent)==0:
#             b.append(img)
#         else:
#             a.append(img)
#     return a,b
