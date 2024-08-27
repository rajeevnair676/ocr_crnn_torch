import torch
import torch.nn as nn
from torchsummary import summary


class BidirectionalLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(BidirectionalLSTM,self).__init__()
        self.lstm1 = nn.LSTM(input_size,hidden_size,bidirectional=True,dropout=0.2)
        self.linear = nn.Linear(hidden_size*2,output_size)

    def forward(self,x:torch.Tensor):
        rec_output,_ = self.lstm1(x)
        # print("x", rec_output.shape,type(rec_output))
        seq_length,batch_size,out_shape = rec_output.size()
        seq_length2 = rec_output.view(seq_length*batch_size,out_shape)
        out = self.linear(seq_length2)
        return out.view(seq_length,batch_size,-1)
    

class BidirectionalGRU(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(BidirectionalGRU,self).__init__()
        self.lstm1 = nn.GRU(input_size,hidden_size,bidirectional=True,dropout=0.1)
        self.linear = nn.Linear(hidden_size*2,output_size)

    def forward(self,x:torch.Tensor):
        rec_output,_ = self.lstm1(x)
        # print("x", rec_output.shape,type(rec_output))
        seq_length,batch_size,out_shape = rec_output.size()
        seq_length2 = rec_output.view(seq_length*batch_size,out_shape)
        out = self.linear(seq_length2)
        return out.view(seq_length,batch_size,-1)

    
class CRNNModel(nn.Module):
    def __init__(self,num_classes):
        super(CRNNModel,self).__init__()
        self.conv_layers = nn.Sequential(
                    nn.Conv2d(3,64,3,1,1),                         
                    nn.ReLU(),
                    nn.MaxPool2d(2,2),

                    nn.Conv2d(64,64,3,1,1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d((1,2),(1,2)),

                    nn.Conv2d(64,128,3,1,1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2),

                    nn.Conv2d(128,128,3,1,1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2),

                    nn.Conv2d(128,128,3,1,1),
                    nn.BatchNorm2d(128),                     
                    nn.ReLU(),
                    nn.MaxPool2d(2,(1,2)),

                    nn.Conv2d(128,128,3,1,1),
                    nn.BatchNorm2d(128),                     
                    nn.ReLU(),
        )
        self.lstm_layer = nn.Sequential(BidirectionalLSTM(128,256,256),
                                        BidirectionalLSTM(256,256,256))
        self.linear1 = nn.Linear(256,128)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(128,num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)
        # self.log_softmax = nn.Softmax(dim=2)

        # self._initialize_weights()

    def forward(self, x: torch.Tensor):
        x = self.conv_layers(x)
        x = torch.squeeze(x, dim=3)
        x = x.permute(2,0,1)
        x = self.lstm_layer(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        out = self.log_softmax(x)
        return out
    
    # def _initialize_weights(self) -> None:
    #     for module in self.modules():
    #         if isinstance(module, nn.Conv2d):
    #             nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
    #         elif isinstance(module, nn.BatchNorm2d):
    #             nn.init.constant_(module.weight, 1)
    #             nn.init.constant_(module.bias, 0)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# crnn = CRNNModel(96).to(DEVICE)
# summary(crnn, (3, 640,32)) 

# inp = torch.rand((50,2,128))
# bilstm = BidirectionalLSTM(128,64,64).to(DEVICE)

# out = bilstm(inp.to(DEVICE))
# print(out.shape)


# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# ├─Sequential: 1-1                        [-1, 128, 79, 1]          --
# |    └─Conv2d: 2-1                       [-1, 64, 640, 32]         1,792
# |    └─ReLU: 2-2                         [-1, 64, 640, 32]         --
# |    └─MaxPool2d: 2-3                    [-1, 64, 320, 16]         --
# |    └─Conv2d: 2-4                       [-1, 64, 320, 16]         36,928
# |    └─BatchNorm2d: 2-5                  [-1, 64, 320, 16]         128
# |    └─ReLU: 2-6                         [-1, 64, 320, 16]         --
# |    └─MaxPool2d: 2-7                    [-1, 64, 320, 8]          --
# |    └─Conv2d: 2-8                       [-1, 128, 320, 8]         73,856
# |    └─BatchNorm2d: 2-9                  [-1, 128, 320, 8]         256
# |    └─ReLU: 2-10                        [-1, 128, 320, 8]         --
# |    └─MaxPool2d: 2-11                   [-1, 128, 160, 4]         --
# |    └─Conv2d: 2-12                      [-1, 128, 160, 4]         147,584
# |    └─BatchNorm2d: 2-13                 [-1, 128, 160, 4]         256
# |    └─ReLU: 2-14                        [-1, 128, 160, 4]         --
# |    └─MaxPool2d: 2-15                   [-1, 128, 80, 2]          --
# |    └─Conv2d: 2-16                      [-1, 128, 80, 2]          147,584
# |    └─BatchNorm2d: 2-17                 [-1, 128, 80, 2]          256
# |    └─ReLU: 2-18                        [-1, 128, 80, 2]          --
# |    └─MaxPool2d: 2-19                   [-1, 128, 79, 1]          --
# |    └─Conv2d: 2-20                      [-1, 128, 79, 1]          147,584
# |    └─BatchNorm2d: 2-21                 [-1, 128, 79, 1]          256
# |    └─ReLU: 2-22                        [-1, 128, 79, 1]          --
# ├─Sequential: 1-2                        [-1, 2, 256]              --
# |    └─BidirectionalLSTM: 2-23           [-1, 2, 256]              --
# |    |    └─LSTM: 3-1                    [-1, 2, 512]              790,528
# |    |    └─Linear: 3-2                  [-1, 256]                 131,328
# |    └─BidirectionalLSTM: 2-24           [-1, 2, 256]              --
# |    |    └─LSTM: 3-3                    [-1, 2, 512]              1,052,672
# |    |    └─Linear: 3-4                  [-1, 256]                 131,328
# ├─Linear: 1-3                            [-1, 2, 128]              32,896
# ├─Dropout: 1-4                           [-1, 2, 128]              --
# ├─Linear: 1-5                            [-1, 2, 96]               12,384
# ├─LogSoftmax: 1-6                        [-1, 2, 96]               --
# ==========================================================================================
# Total params: 2,707,616
# Trainable params: 2,707,616
# ├─Dropout: 1-4                           [-1, 2, 128]              --
# ├─Linear: 1-5                            [-1, 2, 96]               12,384
# ├─LogSoftmax: 1-6                        [-1, 2, 96]               --
# ==========================================================================================
# ├─Dropout: 1-4                           [-1, 2, 128]              --
# ├─Linear: 1-5                            [-1, 2, 96]               12,384
# ├─LogSoftmax: 1-6                        [-1, 2, 96]               --
# ==========================================================================================
# ├─Dropout: 1-4                           [-1, 2, 128]              --
# ├─Linear: 1-5                            [-1, 2, 96]               12,384
# ├─LogSoftmax: 1-6                        [-1, 2, 96]               --
# ==========================================================================================
# ├─Dropout: 1-4                           [-1, 2, 128]              --
# ├─Linear: 1-5                            [-1, 2, 96]               12,384
# ├─LogSoftmax: 1-6                        [-1, 2, 96]               --
# ==========================================================================================
# ├─Dropout: 1-4                           [-1, 2, 128]              --
# ├─Linear: 1-5                            [-1, 2, 96]               12,384
# ├─LogSoftmax: 1-6                        [-1, 2, 96]               --
# ├─Linear: 1-5                            [-1, 2, 96]               12,384
# ├─LogSoftmax: 1-6                        [-1, 2, 96]               --
# ├─LogSoftmax: 1-6                        [-1, 2, 96]               --
