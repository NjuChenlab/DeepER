import torch
from loopy import Train
from loopy.myData import DataBuffer
from loopy.Train import train_log
from random import choice
from torch import optim
from torch.utils.data import DataLoader
import os
#工具包
import loopy.utils as ut
#损失函数等
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DataParallel
import sys
from sklearn.model_selection import train_test_split

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")

#参数定义区
traindata = sys.argv[1]   #正例文件
extra_neg_data = sys.argv[2]
validata = sys.argv[3]
vali_neg_data = sys.argv[4]
logfile = "model.log"
parafile = sys.argv[5]
create_folder(os.path.dirname(parafile))

#监控的名称
namelis = [["loss","testloss"],["Point-acc","Ptestacc","Region-acc","Rtestacc"],["Point-Precisson","PtestPrecission","Point-Recall","PtestRecll"],
           ["Point-F1","PtestF1"],["Region-Precission","RtestPrecission","Region-Recall","RtestRecall"],["Region-F1","RtestF1"]]
colors = [[None,None],[None,None,None,None],[None,None,None,None],[None,None],[None,None,None,None],[None,None]]
colors = ut.randcolor_lis(colors)
printmode = "text"
shownum = 400
#初始损失
ini_lr = 0.0016
decay_rounds = 5
decay = 0.9
#
maxlen = 5000
epochs = 100
pvalue = 0.05
#评估参数
windowlen = 200
windowstep = 10
#采用的模型
from loopy.Model.DRBiLSTM import DRBiLSTM as Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#日志文件
train_log = train_log(
    namelis,
    logfile,
    colors=colors,
    mode=printmode,
    shownum=shownum,
    env="Console"
)
#加载数据
traindata = DataBuffer(
    traindata,
    maxlen,
    zoomrate=1,
    label_mode="si",
    tech_score=None,
    strand=True
)
traindata.addneg(
    path=extra_neg_data
)
if validata == "None":
    testdata = DataBuffer(
        validata,
        maxlen,
        zoomrate=1,
        label_mode="si",
        tech_score=None,
        strand=True
    )
    testdata.addneg(
        path=vali_neg_data
    )
else:
    traindata,testdata = train_test_split(traindata, test_size=0.22, random_state=32)

train_loader = DataLoader(
    dataset=traindata,
    batch_size=32,
    shuffle=True,
    num_workers=2
)
test_loader = DataLoader(
    dataset=testdata,
    batch_size=32,
    shuffle=True,
    num_workers=2
)


import torch.nn as nn
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
    else:
        ut.initialize_weights(m)

#实例化模型
model = Model(4,2,2,64).to(device)

readpara = r"para/just_tuning.pth"
if os.path.exists(parafile):
    model.load_state_dict(torch.load(parafile))
    print("read")
else:
    for name in model.modules():
        init_weights(name)
    pass

# model = DataParallel(model)
#模型工具定义
#优化器
import torch.nn.functional as F
optimizer = optim.Adam(model.parameters(),lr=ini_lr,betas=(0.9, 0.999), eps=1e-8,) #weight_decay=0.0001
scheduler = StepLR(optimizer, step_size=decay_rounds, gamma=decay)
#损失函数
crt = torch.nn.BCELoss().to(device) # weight=torch.Tensor([1.0,27.0])

class WeightedBCELoss(torch.nn.Module):
    def __init__(self,):
        super(WeightedBCELoss, self).__init__()

        self.bgweight = 1
        self.pvalue = 0.5

    def forward(self, input_, target):
        weights = self.makeweight(target)  # 设置为0的位置权重很低，为1的位置权重较高
        loss = (F.binary_cross_entropy(input_, target, weight=weights, reduction='mean') + F.binary_cross_entropy(target, input_, weight=weights, reduction='mean'))/2
        return loss
    
    def makeweight(self,label:torch.Tensor):
        weights = []
        for i in range(label.shape[0]):
            rloop = label[i]
            # 统计大于阈值的元素个数
            count = torch.sum(rloop > self.pvalue).item()
            # 计算大于阈值的比例
            ratio = count / rloop.numel()
            ratio = (1/ratio) - 1 if ratio >0 else 1 
            weights.append(rloop.detach()*ratio+1)
        return torch.stack(weights,dim=0)

#损失函数
lossfunc = WeightedBCELoss()
if __name__ == "__main__":
    Train.train(
        model,optimizer,scheduler,epochs,lossfunc,"hc",train_loader,train_log,device,pvalue,0,windowlen,windowstep,parafile,test_loader
    )
