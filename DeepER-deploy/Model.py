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
    default = [4,2,2,64]
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



class Model(nn.Module):
    default = []
    def __init__(self) -> None:
        super(Model,self).__init__()
        self.conv3 = nn.Sequential(
            ResCNNblock2(1,32,3),
            nn.Dropout(0.2),
            ResCNNblock2(32,64,3),
            nn.Dropout(0.2),
        )
        
        self.conv6 = nn.Sequential(
            ResCNNblock2(1,32,6),
            nn.Dropout(0.2),
            ResCNNblock2(32,64,6),
            nn.Dropout(0.2),

        )
        self.conv7 = nn.Sequential(
            ResCNNblock2(1,32,7),
            nn.Dropout(0.2),
            ResCNNblock2(32,64,7),
            nn.Dropout(0.2)
        )

        self.conv9 = nn.Sequential(
            ResCNNblock2(1,32,9),
            nn.Dropout(0.2),
            ResCNNblock2(32,64,9),
            nn.Dropout(0.2)

        )

        self.conv = nn.Conv1d(1,256,1)

        self.lstm1 = nn.LSTM(256,64,2,bidirectional=True,dropout=0.1)
        self.lstm2 = nn.LSTM(128,1,2,bidirectional=True,dropout=0.1)

        self.fc0 = nn.Sequential(
            nn.Linear(256,2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fc = nn.Sequential(
            nn.Linear(10000,6000),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(6000,5000),
        )
        

    def forward(self,x):
        bs = x.shape[0]
        x = x.unsqueeze(1)
        x3 = self.conv3(x)
        x6 = self.conv6(x)
        x7 = self.conv7(x)
        x9 = self.conv9(x)
        xr= self.conv(x)
       
        # print(x3.shape)
        # print(x6.shape)
        # print(x7.shape)
        # print(x9.shape)
        # print(xr.shape)

        x = torch.cat([x3,x6,x9,x7],dim=1)
        del x3,x6,x7,x9
        x = x.permute(2, 0, 1)
        x,_ = self.lstm1(x)
        x,_ = self.lstm2(x)
        x = x.permute(1, 2, 0)

        x = x.reshape(bs,x.shape[1]*x.shape[2])
        x = self.fc(x)
        return F.sigmoid(x)



class ResCNNblock1(nn.Module):
    def __init__(self,inchannel,outchannel,ker) -> None:
        super(ResCNNblock1,self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inchannel,outchannel,ker,1,ker-1),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(),
            nn.Conv1d(outchannel,outchannel,ker,1),
            nn.BatchNorm1d(outchannel)
        )

    def forward(self,x):
        x_ = x
        x = self.block(x)
        x = x + x_
        del x_
        return F.relu(x)

# 虚线残差
class ResCNNblock2(nn.Module):
    def __init__(self,inchannel,outchannel,ker) -> None:
        super(ResCNNblock2,self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inchannel,outchannel,ker,1,ker-1),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(),
            nn.Conv1d(outchannel,outchannel,ker,1),
            nn.BatchNorm1d(outchannel)
        )
        self.cut = nn.Conv1d(inchannel,outchannel,1,1)

    def forward(self,x):
        x_ = x
        x = self.block(x)
        x = x + self.cut(x_)
        del x_
        return F.relu(x)


#测试
if __name__ =="__main__":
    model = DRBiLSTM(4,2,4,64)

    a = torch.ones((16,5000,4))
    b =model(a)
    print(b.shape)