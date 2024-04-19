'''
这个模块包括主要训练模型的过程以及可视化训练过程的方式
现在也包括损失函数类
'''
import loopy.Evaluate as Evaluate
import loopy.myData as myData
import loopy.utils as ut
import time
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader
from IPython import display

# plt.rcParams['font.family'] = 'SimHei'

#可视化类
class train_log():
    '''
    训练过程的日志类，负责记录训练过程产生的loss，ACC等记录。\n
    '''
    def __init__(self,namelis,logfile,colors,mode="text",shownum=200,cols=2,canvasize=(24,18),env="Jupter") -> None:
        '''
        实例化一个类,指明要记录的对象
        namelis:记录的名字列表,格式应为[[a,b],[c]],在同一个子list中的变量绘制时在同一图片中\n
        logfile:日志文件路径\n
        colors:颜色列表\n
        mode:打印输出还是绘图输出("text","pic")\n
        shownum:显示的历史记录数量,若为0就全部显示\n
        cols:总的列数\n
        canvasize:画布大小\n
        env:在什么环境下运行,JupterNoteBook还是直接命令行|"Console"\n
        '''
        self.namelis = namelis
        self.env = env
        self.colors = colors
        self.mode = mode
        if self.mode=="Console":
            plt.ion()
        self.shownum = shownum
        self.cols = cols
        self.rows = len(self.namelis)//cols if len(self.namelis)%cols == 0 else (len(self.namelis)//cols)+1
        if mode == "pic":
            self.fig, self.ax = plt.subplots(nrows=self.rows,ncols=self.cols)
            self.lines = []
            self.fig.set_size_inches(canvasize[0],canvasize[1])
            for i in range(len(self.namelis)):
                self.ax[i//cols,i%cols].set_title("&".join(self.namelis[i]))  #标题
                if self.mode=="Console":
                    self.ax[i//cols,i%cols].set_autoscaley_on(True)
                    self.ax[i//cols,i%cols].grid()
        self.logfile = open(logfile,mode="w",encoding="utf-8")
        self.log = {}
        for i in self.namelis:
            for j in i:
                self.log[j] = [0]


    #设置log的值
    def setlog(self,value):
        '''
        按照namelis排列的增加的值,如果某个值为None则采用上一个值
        '''
        name =self.namelis
        if isinstance(name,str) and isinstance(value,float):
            self.log[name].append(value)
        elif isinstance(name,list) and isinstance(value,list):
            for i in range(len(name)):
                for j in range(len(name[i])):
                    if value[i][j] == None:
                        self.log[name[i][j]].append(self.log[name[i][j]][-1])
                    else:
                        self.log[name[i][j]].append(value[i][j])
        else:
            raise ValueError()

    #有一定问题，考虑输入形式
    def getlastlog(self,name):
        '''
        这里的list为[str,str]
        '''
        if isinstance(name,str):
            return self.log[name][-1]
        elif isinstance(name,list):
            re = []
            for i in range(len(name)):
                re.append(self.log[name[i]][-1])
            else:
                return re

    def getnumlog(self,name:str):
        '''
        shownum设置成0显示所有
        '''
        if self.shownum == 0:
            return self.log[name]
        else:
            if len(self.log[name]) < self.shownum:
                return self.log[name]
            else:
                return self.log[name][-self.shownum:-1] + [self.log[name][-1]]

    def getmin(self,name:str):
        return min(self.log[name])

    #获得上n次的平均
    def getlastavg(self,name,n):
        if len(self.log[name]) < n:
            return mean(self.log[name])
        else:
            return mean(self.log[name][-n:])

    #打印相关信息
    def printlog(self,run:int,minute,decimal=4):
        '''
        run:当前循环轮数
        decimal:保留小数位
        '''
        name = self.namelis
        print(f">>> run: {run}|time: {minute} >>>")
        for i in range(len(name)):
            string = "\t\t"
            for j in range(len(name[i])):
                string += f"{name[i][j]}:{round(self.getlastlog(name[i][j]),decimal)}\t"  
            else:
                print(string)

    #相关信息写入日志
    def writeself(self,run:int,minute,decimal=4):
        '''
        run:当前循环轮数
        '''
        name = self.namelis
        self.logfile.write(f">>> run: {run}|time: {minute} >>>\n")
        for i in range(len(name)):
            string = "\t\t"
            for j in range(len(name[i])):
                string += f"{name[i][j]}:{round(self.getlastlog(name[i][j]),decimal)}\t"  
            else:
                self.logfile.write(string+"\n")

    def writemore(self,message):
        self.logfile.write(message)
        print(message)

    def draw_frame(self,Epoch):
        self.ax[0,0].set_title(f"Epoch{Epoch}:"+";"+"&".join(self.namelis[0]))
        if self.env == "Jupter":
            for i in range(len(self.namelis)):
                for j in range(len(self.namelis[i])):
                    b = self.namelis[i][j]
                    a = self.getnumlog(b)
                    self.ax[i//self.cols,i%self.cols].plot(list(range(len(a))),a,color=self.colors[i][j],label=b)
                    self.ax[i//self.cols,i%self.cols].legend()
            display.display(self.fig)
            display.clear_output(wait=True)
        elif self.env == "Console":
            for i in range(len(self.namelis)):
                for j in range(len(self.namelis[i])):
                    b = self.namelis[i][j]
                    a = self.getnumlog(b)
                    self.ax[i//self.cols,i%self.cols].plot(list(range(len(a))),a,color=self.colors[i][j],label=b)
                    self.ax[i//self.cols,i%self.cols].relim()
                    self.ax[i//self.cols,i%self.cols].autoscale_view()
                    self.ax[i//self.cols,i%self.cols].legend()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def clean_frame(self):
        for i in range(len(self.namelis)):
            self.ax[i//self.cols,i%self.cols].cla()

    #绘制整个训练流程的某个指标变化
    def draw_index(self,target,sep=50):
        '''
        绘制整个训练流程的某个指标变化\n
        :target:要展示的目标
        :sep:截取间隔轮数
        '''
        self.clean_frame()
        if self.env == "Jupter":
            for i in range(len(self.namelis)):
                for j in range(len(self.namelis[i])):
                    b = self.namelis[i][j]
                    a = self.log[b][::sep]
                    self.ax[i//self.cols,i%self.cols].plot(list(range(len(a))),a,color=self.colors[i][j],label=b)
                    self.ax[i//self.cols,i%self.cols].legend()
            display.display(self.fig)
            display.clear_output(wait=True)
        elif self.env == "Console":
            for i in range(len(self.namelis)):
                for j in range(len(self.namelis)):
                    b = self.namelis[i][j]
                    a = self.log[b][::sep]
                    self.ax[i//self.cols,i%self.cols].plot(list(range(len(a))),a,color=self.colors[i][j],label=b)
                    self.ax[i//self.cols,i%self.cols].relim()
                    self.ax[i//self.cols,i%self.cols].autoscale_view()
                    self.ax[i//self.cols,i%self.cols].legend()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        


    #之后加入的方法    



    def flush(self):
        self.logfile.flush()

def mean(lis):
    return sum(lis)/len(lis)



#现在用模块化的方式
#之后有很多问题都需要改，一个是绘图的时机，另一个是评估的时机，还有训练本身的问题
def train(model:torch.nn.Module,optimizer,scheduler,epochs,lossfunc,codemode:str,traindata:DataLoader,log:train_log,device,pvalue,bgweight,windowlen,windowstep,parapath,testdata=None,checkrun=50,):
    '''
    :methodlis:每种东西的求法函数。
    :pvalue:判定是否为背景的p值
    :bgweight:背景权重，
    :checkrun:每多少轮就test一次
    :parapath:参数保存路径
    :judgefunc:生成各种评估参数的方法，给的参数是,pred,label,run
    '''
    model.train()
    starttime = time.time()
    run = 0
    for epoch in range(epochs):
        scheduler.step()
        log.writemore(f"Start Epoch {epoch}")
        log.writemore('\tlearning rate: ' + f"{optimizer.param_groups[0]['lr']}")
        for step,(batch_x,batch_y,chrinfo) in enumerate(traindata):
            run+=1

            x = ut.codeall(batch_x,codemode).to(device).to(torch.float)
            # labels = ut.makelabel(batch_y[0],batch_y[1],1000).to(device).to(torch.float)
            labels =batch_y.to(device).to(torch.float)

            #对label处理下，防止inf
            judge = labels
            labels = abs(labels-torch.tensor(0.0000001,dtype=torch.float, device=device))

            pred = model(x)
            loss = lossfunc(pred,labels).to(torch.float)
#             print(pred.shape,labels.shape)
#             weight = torch.full((labels.shape[0],labels.shape[1]), bgweight).to(device)
#             weight[labels>pvalue]=1-bgweight
#             loss = torch.mean(weight*loss)

            #梯度下降
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            endtime = time.time()
            minute = int((endtime-starttime)/60)

            #计算所有评估指标
            #之后这里可以修改为用户想要的方法
            #评估训练过程的准确性
            if log.mode=="text":
                p_acc,p_P,p_R,p_F1 = Evaluate.point_evaluate(pvalue,pred,judge,mode="train")
                if run>100*checkrun and run%checkrun==0:
                    r_acc,r_P,r_R,r_F1 = Evaluate.old_region_evaluate(pvalue,pred,judge,windowlen,windowstep,mode="train")
                else:
                    r_acc,r_P,r_R,r_F1 = None,None,None,None
            else:
                if run%checkrun == 0:
                    p_acc,p_P,p_R,p_F1 = Evaluate.point_evaluate(pvalue,pred,judge,mode="train")
                    if run>1000*checkrun and run %(checkrun*2) == 0:
                        r_acc,r_P,r_R,r_F1 = Evaluate.old_region_evaluate(pvalue,pred,judge,windowlen,windowstep,mode="train")
                    else:
                        r_acc,r_P,r_R,r_F1 = None,None,None,None
                else:
                    p_acc,p_P,p_R,p_F1,r_acc,r_P,r_R,r_F1 = None,None,None,None,None,None,None,None

            
            #评估测试集准确性
            if run%(checkrun*5) == 0 and run > 50*checkrun:
                if run%(20*checkrun)==0 and False:
                    tsloss,pointlevel,regionlevel = Evaluate.testprocess(testdata,model,codemode,lossfunc,device,pvalue,windowlen,windowstep)
                    ptaccuracy,ptP,ptR,ptF1 = pointlevel
                    rtaccuracy,rtP,rtR,rtF1 = regionlevel
                elif run%(10*checkrun)==0:
                    tsloss,pointlevel = Evaluate.testloss(testdata,model,codemode,lossfunc,device,pvalue)
                    ptaccuracy,ptP,ptR,ptF1 = pointlevel
                    rtaccuracy,rtP,rtR,rtF1 = None,None,None,None
                else:
                    tsloss,ptaccuracy,ptP,ptR,ptF1,rtaccuracy,rtP,rtR,rtF1 = None,None,None,None,None,None,None,None,None
            else:
                tsloss,ptaccuracy,ptP,ptR,ptF1,rtaccuracy,rtP,rtR,rtF1 = None,None,None,None,None,None,None,None,None
            tsloss,ptaccuracy,ptP,ptR,ptF1,rtaccuracy,rtP,rtR,rtF1 = None,None,None,None,None,None,None,None,None
            
            addlog = [[loss.item(),tsloss],[p_acc,ptaccuracy,r_acc,rtaccuracy],[p_P,ptP,p_R,ptR],[p_F1,ptF1],[r_P,rtP,r_R,rtR],[r_F1,rtF1]]
            log.setlog(addlog)


            if log.mode=="text":
                if run%checkrun == 0:
                    log.writeself(run,minute,4)
                    log.printlog(run,minute,4)
            else:
                if run%checkrun == 0:
                    log.writeself(run,minute,4)
                log.clean_frame()
                log.draw_frame(epoch)
            
            if run%checkrun == 0 and loss.item() < 0.9*log.getlastavg("loss",checkrun):
                torch.save(model.state_dict(),parapath)
                log.writemore("para saved")
        
        #每个Epoch评估测试集
        if epoch>0:
#             tsloss,pointlevel,regionlevel = Evaluate.testprocess(testdata,model,codemode,lossfunc,device,pvalue,windowlen,windowstep)
            tsloss,pointlevel = Evaluate.testloss(testdata,model,codemode,lossfunc,device,pvalue)
            ptaccuracy,ptP,ptR,ptF1 = pointlevel
            rtaccuracy,rtP,rtR,rtF1 = None,None,None,None
        else:
            tsloss,ptaccuracy,ptP,ptR,ptF1,rtaccuracy,rtP,rtR,rtF1 = None,None,None,None,None,None,None,None,None
        addlog = [[None,tsloss],[p_acc,ptaccuracy,r_acc,rtaccuracy],[p_P,ptP,p_R,ptR],[p_F1,ptF1],[r_P,rtP,r_R,rtR],[r_F1,rtF1]]
        log.setlog(addlog)
        torch.save(model.state_dict(),parapath+f".Epoch{epoch}.pkl")
        log.writemore("para saved")
        log.flush()