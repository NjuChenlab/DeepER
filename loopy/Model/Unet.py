import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dblock(nn.Module):
    def __init__(self,in_channel,out_channel,kersize,stride,padding,droprob=0.25,ifres=True) -> None:
        super(Conv1dblock,self).__init__()
        if ifres:
            if in_channel != out_channel: #虚线残差,暂时先不要虚线残差了
                self.resconnect = nn.Linear(in_channel,out_channel,bias=False)
            else:
                self.resconnect = True
        else:
            self.resconnect = False
        self.Conv1 = nn.Conv1d(in_channel,out_channel,kersize,stride,padding)
        self.drop1 = nn.Dropout(droprob)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.Conv2 = nn.Conv1d(out_channel,out_channel,kersize,stride,padding)
        self.drop2 = nn.Dropout(droprob)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self,x):
        if not isinstance(self.resconnect,bool):
            res = self.resconnect(x.permute(0,2,1)).permute(0,2,1)
        else:
            if self.resconnect:
                res = x
            else:
                res = None
        x = self.Conv1(x)
        x = self.drop1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.Conv2(x)
        x = self.drop2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if not isinstance(res,type(None)):
            x = x+res
        del res
        return x
    
class Encoderblock(nn.Module):
    def __init__(self,convnum,in_channel,out_channel,kersize,stride,padding,droprob=0.25) -> None:
        super(Encoderblock,self).__init__()
        self.Conv = nn.Conv1d(in_channel,out_channel,kersize,stride,padding)
        Blocklis = []
        for i in range(convnum):
            Blocklis.append(Conv1dblock(out_channel,out_channel,kersize,1,padding,droprob=droprob))
        self.block = nn.Sequential(*Blocklis)
        del Blocklis
        
    def forward(self,x):
        x = self.Conv(x)
        x = self.block(x)
        return x

class Decoderblock(nn.Module):
    def __init__(self,convnum,in_channel,out_channel,kersize,stride,padding,droprob=0.25) -> None:
        super(Decoderblock,self).__init__()
        self.TransConv = nn.ConvTranspose1d(in_channel,out_channel,kersize[0],stride,0)
        
        Blocklis = []
        for i in range(convnum):
            if i==0:
                Blocklis.append(Conv1dblock(out_channel*2,out_channel,kersize[1],1,padding,droprob=droprob))
            else:
                Blocklis.append(Conv1dblock(out_channel,out_channel,kersize[1],1,padding,droprob=droprob))

        self.block = nn.Sequential(*Blocklis)

    def forward(self,x,enc):
        x = self.TransConv(x)
        x = torch.cat((x,enc),dim=1)
        x = self.block(x)
        return x


class Unet(nn.Module):
    default = []
    def __init__(self,) -> None:
        super(Unet,self).__init__()
        self.Encoder1 = Encoderblock(2,4,32,5,5,2)
        self.Encoder2 = Encoderblock(2,32,64,5,5,2)
        self.Encoder3 = Encoderblock(2,64,128,5,5,2)
        self.Encoder4 = Encoderblock(2,128,256,5,4,2)
        self.Decoder1 = Decoderblock(2,256,128,(4,3),4,1)
        self.Decoder2 = Decoderblock(2,128,64,(5,3),5,1)
        self.Decoder3 = Decoderblock(2,64,32,(5,3),5,1)
        # self.Decoder4 = Decoderblock(2,32,4,(4,3),4,1)
        self.conv1x1 = nn.ConvTranspose1d(32, 1, kernel_size=5, stride=5, bias=False)


    def forward(self,x):
        x = x.permute(0,2,1)
        x1 = self.Encoder1(x)
        del x
        # print(x1.shape)
        x2 = self.Encoder2(x1)
        # print(x2.shape)
        x3 = self.Encoder3(x2)
        # print(x3.shape)
        x = self.Encoder4(x3)
        # print(x.shape)
        x = self.Decoder1(x,x3)
        # print(x.shape)
        x = self.Decoder2(x,x2)
        # print(x.shape)
        x = self.Decoder3(x,x1)
        # print(x.shape)
        x = self.conv1x1(x)
        # print(x.shape)

        return F.sigmoid(x[:,0,:])
    
if __name__=="__main__":
    
    model = Unet()

    x = torch.zeros((6,5000,4))
    x = model(x)
    print(x.shape)
        