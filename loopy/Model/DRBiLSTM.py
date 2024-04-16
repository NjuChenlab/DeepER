import torch
import torch.nn as nn
import torch.nn.functional as F


# 残差LSTM块
class resLSTMblock(nn.Module):
    def __init__(self,in_feature,out_feature,cellnum,bidirectional=True) -> None:
        super(resLSTMblock,self).__init__()
        #虚线残差和实线残差分别处理
        if in_feature != out_feature:
            self.solid = nn.Linear(in_feature,out_feature)
        else:
            self.solid = None

        self.lstm = nn.LSTM(in_feature,out_feature//2,cellnum,bidirectional=bidirectional)
        self.lstm.flatten_parameters()

    def forward(self,x):
        if self.solid == None:
            x_ = x
        else:
            x_ = self.solid(x)

        x,_ = self.lstm(x)
        x = F.relu(x) + x_
        return x
    

class DRBiLSTM(nn.Module):
    def __init__(self,in_feature,layers,blocknum,hidden,dropout=0.2,bidirectional=True) -> None:
        super(DRBiLSTM,self).__init__()

        self.in_feature,self.blocknum,self.layers,self.hidden,self.dropout,self.bidirectional = in_feature,blocknum,layers,hidden,dropout,bidirectional
        
        self.lstm1 = nn.LSTM(in_feature,hidden//2,2,bidirectional=self.bidirectional)
        self.lstm1.flatten_parameters() 

        self.resLSTMs = self.make_layers()

        self.fc = nn.Linear(self.hidden,2)
#         self.fc = nn.Sequential(
#             nn.Linear(self.hidden,self.hidden//2),
#             nn.Dropout(self.dropout),
#             nn.ReLU(),
#             nn.Linear(self.hidden//2,1)
#         )

    def make_layers(self):
        layers = []
        for i in range(self.blocknum):
            layers.append(resLSTMblock(self.hidden,self.hidden,self.layers,bidirectional=self.bidirectional))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = x.permute(1,0,2)
        x,_ = self.lstm1(x)
        x = self.resLSTMs(x)
        x = self.fc(x)
        x = x.permute(1,0,2)

        return F.sigmoid(x)[:,:,1]

class sp_DeepER(nn.Module):
    def __init__(self) -> None:
        super(sp_DeepER,self).__init__()
        self.lstm1 = nn.LSTM(4,32//2,2,bidirectional=True)
        self.lstm1.flatten_parameters()
        self.resLSTM1 = resLSTMblock(32,32,2,True)
        self.resLSTM2 = resLSTMblock(32,32,2,True)
        self.resLSTM3 = resLSTMblock(32,32,2,True)
        self.resLSTM4 = resLSTMblock(32,32,2,True)
        
        self.fc = nn.Linear(32,2)

    def forward(self,x):
        re = []
        x = x.permute(1,0,2)
        x,_ = self.lstm1(x)
        x = self.resLSTM1(x)
        re.append(F.sigmoid(self.fc(x).permute(1,0,2))[:,:,1].cpu().detach())
        x = self.resLSTM2(x)
        re.append(F.sigmoid(self.fc(x).permute(1,0,2))[:,:,1].cpu().detach())
        x = self.resLSTM3(x)
        re.append(F.sigmoid(self.fc(x).permute(1,0,2))[:,:,1].cpu().detach())
        x = self.resLSTM4(x)
        re.append(F.sigmoid(self.fc(x).permute(1,0,2))[:,:,1].cpu().detach())
        x = x.permute(1,0,2)
        return re


#测试
if __name__ =="__main__":
    model = DRBiLSTM(4,2,2,64)

    a = torch.ones((16,5000,4))
    b =model(a)
    print(b.shape)