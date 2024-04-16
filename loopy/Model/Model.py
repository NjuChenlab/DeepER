'''
这个模块包括实现的一些模型，以及如果可能，模型结构作图。
除了transformer,我们提供3个模型,以TextCNN为核心的ResTextCNN,以RNN为核心的RTextCNN,以TCN为核心的TTextCNN。
我们预期TTextCNN拥有最快的速度。
我们最好之后提供变化的定义方式
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model,self).__init__()
        self.conv3 = nn.Sequential(
            ResCNNblock2(1,32,3),
            nn.Dropout(0.4),
            ResCNNblock2(32,64,3),
            nn.Dropout(0.4),
        )
        
        self.conv6 = nn.Sequential(
            ResCNNblock2(1,32,6),
            nn.Dropout(0.4),
            ResCNNblock2(32,64,6),
            nn.Dropout(0.4),

        )
        self.conv7 = nn.Sequential(
            ResCNNblock2(1,32,7),
            nn.Dropout(0.4),
            ResCNNblock2(32,64,7),
            nn.Dropout(0.4)
        )

        self.conv9 = nn.Sequential(
            ResCNNblock2(1,32,9),
            nn.Dropout(0.4),
            ResCNNblock2(32,64,9),
            nn.Dropout(0.4)

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
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

Res_Bidir_LSTM = {
	'name' : 'Res_Bidir_LSTM',
	'bidir' : True,
	'clip_val' : 10,
	'drop_prob' : 0,
	'n_epochs_hold' : 100,
	'n_layers' : 2,
	'learning_rate' : [0.0015],
	'weight_decay' : 0,
	'n_residual_layers' : 2,
	'n_highway_layers' : 2,
	'diag' : 'Architecure chosen is Residual Bidirectional LSTM',
	'save_file' : 'results_res_lstm1.txt'
}

Architecture = {
	'Res_Bidir_LSTM' : Res_Bidir_LSTM
}

# Choose what architecure you want here:
name_modle = 'Res_Bidir_LSTM'
arch = Architecture[name_modle]

# This will set the values according to that architecture
bidir = arch['bidir']
clip_val = arch['clip_val']
drop_prob = arch['drop_prob']
n_epochs_hold = arch['n_epochs_hold']
n_layers = arch['n_layers']
learning_rate = arch['learning_rate']
weight_decay = arch['weight_decay']
n_highway_layers = arch['n_highway_layers']
n_residual_layers = arch['n_residual_layers']

# These are for diagnostics
diag = arch['diag']
save_file = arch['save_file']

# This will stay common for all architectures:
n_classes = 2
n_input = 4
n_hidden = 32
batch_size = 64
n_epochs = 20

#train the model 
#checkpoint = ''
#is_train = True

#predict the genome-wide R-loops
checkpoint = './Checkpoints_window/' + name_modle + '/best_model.pth'
is_train = None



class Res_Bidir_LSTMModel(nn.Module):
    def __init__(self, n_input=n_input, n_hidden=n_hidden, n_layers=n_layers,
                 n_classes=n_classes, drop_prob=drop_prob):
        super(Res_Bidir_LSTMModel, self).__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.drop_prob = drop_prob
        self.n_input = n_input

        self.lstm1 = nn.LSTM(n_input, int(n_hidden/2), n_layers, bidirectional=True, dropout=self.drop_prob)
        self.lstm2 = nn.LSTM(n_hidden, int(n_hidden/2), n_layers, bidirectional=True, dropout=self.drop_prob)

        self.fc = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(drop_prob)

    def addResidualLayers(self, x, hidden):
        for i in range(n_residual_layers):
            mid = F.relu(x)
            self.lstm2.flatten_parameters()
            x, hidden2 = self.lstm2(mid, hidden)
            x = F.relu(x)
            x += mid
        return x

    def forward(self, x, hidden):
        x = x.permute(1, 0, 2)
        self.lstm1.flatten_parameters()
        x, hidden1 = self.lstm1(x, hidden)
        for i in range(n_highway_layers):
            x = self.addResidualLayers(x, hidden)
        x = self.dropout(x)
        # out = x[-1]
        # out = self.fc(out)
        out = self.fc(x)
        out = out.view(-1,2)
        out_train = out
        out = F.softmax(out,dim=1)

        return out, out_train

    def init_hidden(self, batch_size):
        ''' Initialize hidden state'''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # if (train_on_gpu):
        if (torch.cuda.is_available() ):
            hidden = (weight.new(2*self.n_layers, batch_size, int(self.n_hidden/2)).zero_().cuda(),
                weight.new(2*self.n_layers, batch_size, int(self.n_hidden/2)).zero_().cuda())
        else:
            hidden = (weight.new(2*self.n_layers, batch_size, int(self.n_hidden/2)).zero_(),
                weight.new(2*self.n_layers, batch_size, int(self.n_hidden/2)).zero_())

        return hidden

def init_weights(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)



class packnet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bilstm = Res_Bidir_LSTMModel()

    def load_(self,para):
        self.bilstm.load_state_dict(torch.load(para))

    def forward(self,x:torch.Tensor):
        re = []
        for i in range(x.shape[0]):
            a = x[i]
            a = a.unsqueeze(0)
            test_h = self.bilstm.init_hidden(len(a))
            inputs= a.cuda()
            # print(inputs.shape)  
            test_h = tuple([each.data for each in test_h])
            output,output_test= self.bilstm(inputs.float(), test_h)
            re.append(output[:,1])
        return torch.stack(re,dim=0)



class Model1(nn.Module):
    def __init__(self) -> None:
        super(Model1,self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv1d(1,32,3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool1d(3,3),
            nn.Conv1d(32,128,3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool1d(3,3),
            nn.Conv1d(128,256,3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool1d(3,3),
            nn.Conv1d(256,256,3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool1d(3,3),
            nn.Conv1d(256,256,3),
            nn.ReLU(),
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv1d(1,32,6),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool1d(3,3),
            nn.Conv1d(32,128,6),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool1d(3,3),
            nn.Conv1d(128,256,6),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool1d(3,3),
            nn.Conv1d(256,256,6),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool1d(3,3),
            nn.Conv1d(256,256,6),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(1,32,7),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool1d(3,3),
            nn.Conv1d(32,128,7),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool1d(3,3),
            nn.Conv1d(128,256,7),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool1d(3,3),
            nn.Conv1d(256,256,7),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool1d(3,3),
            nn.Conv1d(256,256,7),
            nn.ReLU(),
        )

        self.conv9 = nn.Sequential(
            nn.Conv1d(1,32,9),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool1d(3,3),
            nn.Conv1d(32,128,9),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool1d(3,3),
            nn.Conv1d(128,256,9),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool1d(3,3),
            nn.Conv1d(256,256,9),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool1d(3,3),
            nn.Conv1d(256,256,9),
            nn.ReLU(),
        )


        self.lstm1 = nn.LSTM(50,64,2,bidirectional=True)
        self.lstm2 = nn.LSTM(128,20,2,bidirectional=True)

        self.fc0 = nn.Sequential(
            nn.Linear(256,4),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fc = nn.Sequential(
            nn.Linear(10240,6000),
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
        # xr= self.conv(x)
       
        # print(x3.shape)
        # print(x6.shape)
        # print(x7.shape)
        # print(x9.shape)
        # print(xr.shape)

        x = torch.cat([x3,x6,x9,x7],dim=2)
        del x3,x6,x7,x9
        # print(x.shape)
        x = x.permute(1, 0,2)
        x,_ = self.lstm1(x)
        x,_ = self.lstm2(x)
        x = x.permute(1, 0, 2)
        
        x = x.reshape(bs,x.shape[1]*x.shape[2])
        
        x = self.fc(x)
        return F.sigmoid(x)

import torch
from torch import nn


n_classes = 2
n_input = 4
drop_prob = 0
batch_size = 128



def conv1d_5(inplanes, outplanes, stride=1):
    return nn.Conv1d(inplanes, outplanes, kernel_size=5, stride=stride,
                     padding=2, bias=False)
# 下采样block，下采样过程中使用resnet作为backbone
class Block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None,
            bn=False,drop=True):
        super(Block, self).__init__()
        self.bn = bn
        self.drop=drop
        self.conv1 = conv1d_5(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.drop1=nn.Dropout(0.2)
        self.conv2 = conv1d_5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.drop2 = nn.Dropout(0.2)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.bn:
            out = self.bn1(out)
        if self.drop:
            out=self.drop1(out)
        out = F.relu(out)

        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        if self.drop:
            out = self.drop2(out)
        out = F.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out
# 上采样block
class Decoder_block(nn.Module):

    def __init__(self, inplanes, outplanes, kernel_size=5, stride=5):
        super(Decoder_block, self).__init__()
        self.upsample = nn.ConvTranspose1d(in_channels=inplanes, out_channels=outplanes,
                           padding=0, kernel_size=kernel_size, stride=stride, bias=False)
        self.conv1 = conv1d_5(inplanes, outplanes)
        self.conv2 = conv1d_5(outplanes, outplanes)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        out = torch.cat((x1, x2), dim=1)

        out = self.conv1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = F.relu(out)

        return out


class Uresnet(nn.Module):
    def __init__(self, inplanes=4, layers=[2,2,2,2]):
        super(Uresnet, self).__init__()
        self.inplanes = inplanes
        self.encoder1 = self._make_encoder(Block, 32, layers[0], 5)
        self.encoder2 = self._make_encoder(Block, 64, layers[1], 5)
        self.encoder3 = self._make_encoder(Block, 128, layers[2], 5)
        self.encoder4 = self._make_encoder(Block, 256, layers[3], 4)
        self.decoder3 = Decoder_block(256, 128, stride=4, kernel_size=4)
        self.decoder2 = Decoder_block(128, 64)
        self.decoder1 = Decoder_block(64, 32)
        # 如果要修改输出的通道，在这里修改
        self.conv1x1 = nn.ConvTranspose1d(32, 2, kernel_size=5, stride=5, bias=False)

    def _make_encoder(self, block, planes, blocks, stride=1): #block,32,2,5
        downsample = None
        if self.inplanes != planes or stride != 1:
            downsample = nn.Conv1d(self.inplanes, planes, stride=stride, kernel_size=1, bias=False)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


    def forward(self, x):
        x=x.permute(0,2,1)
        down1 = self.encoder1(x)
        down2 = self.encoder2(down1)
        down3 = self.encoder3(down2)
        down4 = self.encoder4(down3)

        up3 = self.decoder3(down4, down3)
        up2 = self.decoder2(up3, down2)########attention
        up1 = self.decoder1(up2, down1)
        out = self.conv1x1(up1)[:,1,:]
        return F.sigmoid(out)


#测试
if __name__ =="__main__":
    model = Model1()
    a = torch.ones((16,5000))
    b =model(a)
    print(b.shape)