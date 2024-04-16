'''
模型部署使用的模块
包含一个轻量级读取的类和轻量级预测的类
之后还要包括输出绘图等
'''
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import loopy.utils as ut

class quickloader(Dataset):
    def __init__(self,path,batch_size) -> None:
        '''
        用于轻量化读取fasta的部件,不提供任何其他信息
        现在还没加上不定长序列的处理，之后还得加上
        :path:fasta文件的路径
        :batch_size:并行运算序列的大小
        '''
        self.batch_size = batch_size
        self.obj = []
        with open(path,mode="r",encoding="utf-8") as f:
            for i in f.readlines():
                if i[0]==">":
                    self.obj.append([i[1:len(i)].rstrip("\n"),""])
                    continue
                else:
                    self.obj[-1][1] += i.rstrip("\n")
        
    def __getitem__(self,index):
        return self.obj[index][1],self.obj[index][0]



#定义Rloop区间的方式
def defregion(data01,cut_off):
    rloop = {}
    count,num = 0,0
    xregion,textlabel = [],[]

    for m in range(0,4800+1,10):
        n = m+200
        region = data01[m:n]
        #print(len(region))
        if sum(region)/len(region)>=cut_off:
            count += 1
            rloop[count]=[m,n]
    for m in range(1,count):
        x = rloop[m][0]
        y = rloop[m][1]
        z = rloop[m+1][0]
        h = rloop[m+1][1]
        if y>z:
            del rloop[m]
            del rloop[m+1]
            rloop[m+1] = [x,h]
    
    for value in rloop.values():
        xregion.append(value)
        textlabel.append(value[0])
        textlabel.append(value[1])
    return(xregion,textlabel)



#部署使用的类
class PreDict():
    def __init__(self,data,model,para,codemode,
                 regionfunc=defregion,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> None:
        
        loader = DataLoader(
            dataset=data,
            batch_size=64,
            shuffle=True,
            num_workers=1
        )
        self.func = regionfunc
        self.device = device
        self.codemode = codemode
        self.model = model().load_state_dict(torch.load(para))
        self.model.eval()
        self.re = {}
        for step,(batch_x,headinfo) in enumerate(loader):
            x = ut.codeall(batch_x,codemode).to(device).to(torch.float)
            pred = model(x)
            for i in range(pred.shape[0]):
                self.re[headinfo] = (batch_x[i],pred[i],regionfunc(pred[i]))
    
    #重新预测某一链的负链
    def get_rev(self,headinfo):
        chain = ut.rev_chain(self.re[headinfo][0])
        x = ut.codeall(chain,self.codemode)
        pred = self.model(x)
        res = (self.re[headinfo][0],pred[0],self.func(pred[0]))
        self.re[headinfo+"_rev"] = res
        return res
    
    #可视化某个序列结果
    def visualize(self,headinfo):
        pass

    #
    def getitem(self,headerinfo):
        return self.re[headerinfo]



