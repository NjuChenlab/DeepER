import torch
import torch.nn as nn
import numpy as np
from random import choice

#编码
def coding(string,mode="dc",ebsize=1,step=1):
    '''
    mode:可以为\n
    hc:独热编码\n
    dc:labelcoding\n
    之后再加一个傅里叶编码
    '''
    drmap = {
        "N":0,
        "A":1,
        "C":2,
        "G":3,
        "T":4
    }
    htmap = {
        "N":[0,0,0,0],
        "A":[1,0,0,0],
        "C":[0,1,0,0],
        "G":[0,0,1,0],
        "T":[0,0,0,1]
    }
    cd = []
    for i in list(string):
        if mode=="hc":
            cd.append(htmap[i])
        elif mode=="dc":
            cd.append(drmap[i])
    if ebsize==1:
        return torch.from_numpy(np.array(cd))
    
    if step ==0 or step>ebsize:
        raise

    #如果还要嵌入
    eb = []
    j = 0
    while 1:
        a = cd[j:j+ebsize]
        if type(a[0]) == list:
            a = np.array([item for sublist in a for item in sublist])
        eb.append(a)
        j+=step
        if j+ebsize>len(cd):
            break
    return torch.from_numpy(np.array(cd))

#
def codeall(strings,mode="dc",ebsize=1,step=0):
    '''
    mode:可以为\n
    hc:独热编码\n
    dc:直接编码\n
    '''
    res = []
    for i in strings:
        res.append(coding(i,mode,ebsize,step))
    val= torch.tensor([item.cpu().detach().numpy() for item in res]).cuda()
    # return torch.from_numpy(np.array(res))
    return val

#滑动窗口
def slide(string,length,step=1):
    pass

#从正链获得负链或者负链获得正链
def rev_chain(chain:str):
    dic = {"A":"T","T":"A","G":"C","C":"G"}
    chain = chain[::-1]
    re = ""
    for i in range(len(chain)):
        try:
            re += dic(chain[i])
        except:
            raise ValueError("Unknown base"+chain[i])
    return re

#模型用函数
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.orthogonal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)

def set_parameters_as_trainable(model):
    for param in model.parameters():
        param.requires_grad = True

def make_Model_single_GPU(model:nn.Module):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    for module in model.modules():
        module.to(device)
        # print(module)
        # for name, param in module.named_parameters():
            # print(name)
    print("finish")

#随机颜色
def randcolor():
    colors = ["#3a30d9", "#2189c9", "#b8c63a", "#ce3235", "#bc0eaa", "#1d871e", "#a43255", "#2366a9", "#2d2414", "#696895", "#4609a6", "#f30be2", "#82de8c", "#65cf2d", "#0742da", "#aba26f", "#f8023f", "#7eeab2", "#681f31", "#bec3de", "#10aa11", "#5db2c6", "#8de172", "#79c2c0", "#313d05", "#ef6a69", "#fec480", "#d10956", "#4df132", "#68d206"]
    return choice(colors)

#随机颜色列表
def randcolor_lis(lisshape:list):
    '''
    :lisshape:要获得的随机颜色列表的形状
    '''
    re = []
    for i in lisshape:
        apd = []
        for j in i:
            apd.append(randcolor())
        re.append(apd)
    return re