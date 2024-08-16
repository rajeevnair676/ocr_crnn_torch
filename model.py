import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self,output):
        super().__init__()
        # total tokens 
        self.output=output
        self.maxpool1=torch.nn.MaxPool2d(2)
        self.maxpool2=torch.nn.MaxPool2d(2)
        self.cnn1=torch.nn.Conv2d(1,8,3,)
        self.cnn2=torch.nn.Conv2d(8,8,3)
        # final conv layer of 4 channels
        self.cnn3=torch.nn.Conv2d(8,4,3,) 
        # 2 layer gru with 32 units
        self.encgru=nn.GRU(40,32,2,batch_first=True,dropout=0.1)
        # for inputing one hot encoded digits
        self.emb=nn.Embedding(self.output,8)
        # 2 layer gru with 32 units
        self.decgru=nn.GRU(8,32,2,batch_first=True,dropout=0.1)
        # timeshared linear layer
        self.Linear=nn.Linear(32,self.output,bias=True,) 


    def forward(self,x,val):
        x=self.cnn1(x)
        x=nn.functional.relu(x)
        x=self.maxpool1(x)
        x=self.cnn2(x)
        x=nn.functional.relu(x)
        x=self.maxpool2(x)
        x=self.cnn3(x)
        x=nn.functional.relu(x)
        batch,channel,time,emb=x.shape
        # concatenating along the y axis
        x=x.permute(0,2,1,3).reshape(batch,time,emb*channel)
        # only hidden state is passed to decoder
        _,hidden=self.encgru(x)
        batch,time,embe=val.shape
        x=self.emb(val)
        x=nn.functional.relu(x)
        x=x.squeeze(2)
        x,_= self.decgru(x,hidden)
        x=nn.functional.relu(x)
        x=self.Linear(x.reshape(-1,32))
        return x
    
    def predict(self,x):
        t=[]
        x=self.cnn1(x)
        x=nn.functional.relu(x)
        x=self.maxpool1(x)
        x=self.cnn2(x)
        x=nn.functional.relu(x)
        x=self.maxpool2(x)
        x=self.cnn3(x)
        x=nn.functional.relu(x)
        batch,channel,time,emb=x.shape
        x=x.permute(0,2,1,3).reshape(batch,time,emb*channel)
        _,hidden=self.encgru(x)
        # <start> token index
        index=10
        pred=[index]
        # maximum length is less than 12
        for _ in range(12):
            x=self.emb(torch.tensor([[[index]]]))
            x=nn.functional.relu(x)
            x=x.squeeze(2)
            x,hidden= self.decgru(x,hidden)
            x=nn.functional.relu(x)
            x=self.Linear(x.reshape(-1,32))
            index=torch.argmax(x,-1)[0]
            pred.append(index.item())
            # if <end> token then break loop
            if index ==11:
                break
        return pred
# total 13 tokens are used, includes 10 digits + <start> + <end> + "."
model = EncoderDecoder(13)