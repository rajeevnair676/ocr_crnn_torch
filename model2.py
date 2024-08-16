import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary


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

# inp = torch.rand((50,2,128))
# bilstm = BidirectionalLSTM(128,64,64).to(DEVICE)

# out = bilstm(inp.to(DEVICE))
# print(out.shape)


        
