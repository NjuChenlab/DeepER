'''
这个模块包括模型评估以及作图可视化
现在的问题是还没有兼容用户自定义的方法
'''
import torch
import loopy.utils as ut
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import time
import numpy as np
from torch import optim
from random import randint
import psutil
from sklearn.model_selection import train_test_split

# plt.rcParams['font.family'] = 'SimHei'

def testloss(loader,model,codemode,lossfunc,device,pvalue):
    re = 0
    i = 0
    pTP,pFN,pTN,pFP = 0,0,0,0
    for step,(batch_x,batch_y,chrinfo) in enumerate(loader):
        i+=1
        x = ut.codeall(batch_x,codemode).to(device).to(torch.float)
        labels =batch_y.to(device).to(torch.float)
        #对label处理下，防止inf
        labels = abs(labels-torch.tensor(0.000001,dtype=torch.float, device=device))

        pred = model(x)
        if lossfunc != None:
            loss = lossfunc(labels,pred).to(torch.float)
            re += loss.item()

        pTP_,pFN_,pTN_,pFP_ = point_evaluate(pvalue,pred,labels,mode="test")
        pTP += pTP_
        pFN += pFN_
        pTN += pTN_
        pFP += pFP_
    p_accuracy = (pTP+pTN)/(pTP+pFN+pTN+pFP)
    try:
        pP = pTP/(pTP+pFP)
        pR = pTP/(pTP+pFN)
        pF1=2*(pP*pR)/(pP+pR)
    except:
        pP = 0
        pR = 0
        pF1= 0 
    return re/i,(p_accuracy,pP,pR,pF1)

def testprocess(loader,model,codemode,lossfunc,device,pvalue,windowlen,windowstep):
    '''
    之后还是得加权
    lossfunc:可为None，None时跳过求testloss
    '''
    re = 0
    i = 0
    pTP,pFN,pTN,pFP = 0,0,0,0
    rTP,rFN,rTN,rFP = 0,0,0,0
    model.eval()
    for step,(batch_x,batch_y,chrinfo) in enumerate(loader):
        i+=1
        x = ut.codeall(batch_x,codemode).to(device).to(torch.float)
        # x = ut.codeall(batch_x).to(device).to(torch.float)
        # labels = ut.makelabel(batch_y[0],batch_y[1],1000).to(device).to(torch.float)
        labels =batch_y.to(device).to(torch.float)
        #对label处理下，防止inf
        labels = abs(labels-torch.tensor(0.000001,dtype=torch.float, device=device))

        pred = model(x)
        if lossfunc != None:
            loss = lossfunc(pred,labels).to(torch.float)
            re +=loss.item()
        

        pTP_,pFN_,pTN_,pFP_ = point_evaluate(pvalue,pred,labels,mode="test")
        pTP += pTP_
        pFN += pFN_
        pTN += pTN_
        pFP += pFP_

        rTP_,rFN_,rTN_,rFP_ = old_region_evaluate(pvalue,pred,labels,windowlen,windowstep,mode="test")
        rTP += rTP_
        rFN += rFN_
        rTN += rTN_
        rFP += rFP_
    try:
        p_accuracy = (pTP+pTN)/(pTP+pFN+pTN+pFP)
    except:
        p_accuracy = 0
    try:
        r_accuracy = (rTP+rTN)/(rTP+rFN+rTN+rFP)
    except:
        r_accuracy = 0
    try:
        pP = pTP/(pTP+pFP)
        pR = pTP/(pTP+pFN)
        pF1=2*(pP*pR)/(pP+pR)
    except:
        pP = 0
        pR = 0
        pF1= 0 
    try:
        rP = rTP/(rTP+rFP)
        rR = rTP/(rTP+rFN)
        rF1=2*(rP*rR)/(rP+rR)
    except:
        rP = 0
        rR = 0
        rF1= 0 
    return re/i,(p_accuracy,pP,pR,pF1),(r_accuracy,rP,rR,rF1)

import numpy as np
from sklearn.metrics import confusion_matrix
# 点精度评估
def point_evaluate(pvalue,pred,label,mode="train"):
    '''
    输入两个张量,一个为模型预测结果,另一个为实际结果\n
    返回TP,FN,TN,FP\n
    mode参数为"skip" "train" "test"
    '''
    if mode == "skip":
        return 0,0,0,0
    mask1 = pred >= 1-pvalue
    mask0 = pred < 1-pvalue
    pred[mask1] = 1
    pred[mask0] = 0

    mask1 = label >= 0.99
    mask0 = label < 0.99
    label[mask1] = 1
    label[mask0] = 0
    confusion_matrix_result = confusion_matrix(label.cpu().detach().numpy().flatten(),pred.cpu().detach().numpy().flatten())
    try:
        TP = confusion_matrix_result[1, 1]  # 第1行和第1列表示正例
        FP = confusion_matrix_result[0, 1]  # 第0行和第1列表示负例预测为正例
        TN = confusion_matrix_result[0, 0]  # 第0行和第0列表示负例
        FN = confusion_matrix_result[1, 0]  # 第1行和第0列表示正例预测为负例
    except:
        TP = 0
        FP = 0
        TN = 0 
        FN = 1 # to avoid division by 0

    #计算P:准确率,R:召回率
    accuracy = (TP+TN)/(TP+FP+TN+FN)
    try:
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F1=2*(P*R)/(P+R)
    except:
        P = 0
        R = 0
        F1= 0 
    if mode == "train":
        return accuracy,P,R,F1
    if mode == "test":
        return TP,FN,TN,FP
    
# 区域精度评估,旧的师姐的评估方式,用于和旧模型比较
def old_region_evaluate(pvalue,pred,label,windowlen=200,step=20,mode="train"):
    '''
    输入两个张量,一个为模型预测结果,另一个为实际结果\n
    返回TP,FN,TN,FP\n
    mode参数为"skip" "train" "test"
    '''
    if mode=="skip":
        return 0,0,0,0
    TP,FN,TN,FP = 0,0,0,0
    judge = torch.abs(label-pred)
    for i in range(pred.shape[0]):
        a = pred[i]
        new = torch.zeros((a.shape[0],))
        for k in range(0,a.shape[0]-200+1,10):
            if torch.sum(a[k:k+200])/windowlen > 0.95:
                new[k:k+200] = 1
        b = label[i]
        j = judge[i]
        start = 0
        while 1:
            if start+windowlen < a.shape[0]:
                aa = new[start:start+windowlen]
                bb = b[start:start+windowlen]
                #核心评估方式
                count1 = torch.sum(bb > 1-pvalue).item()    #判断bb为正例还是负例，若一半为正就是正例
                count2 = torch.sum(aa > 1-pvalue).item()    #判断jj正确与否

                if count1 > (windowlen*0):
                    #正例
                    #判断是否分类正确
                    if count2 > (windowlen*0):
                        TP += 1
                    else:
                        FN += 1
                else:
                    #负例
                    #判断是否分类正确
                    if count2 <= (windowlen*0):
                        TN += 1
                    else:
                        FP += 1
                start += step
            else:
                break
    if mode == "train":
        accuracy = (TP+TN)/(TP+FN+TN+FP)
        try:
            P = TP/(TP+FP)
            R = TP/(TP+FN)
            F1=2*(P*R)/(P+R)
        except:
            P = 0
            R = 0
            F1= 0 
        return accuracy,P,R,F1
    if mode == "test":
        return TP,FN,TN,FP

#滑窗切分
def sliding_window_binarize(x, window_size, pvalue, stride=1):
    '''
    stride:步长
    window_size:窗口长度
    '''
    batch_size, seq_len = x.shape
    out_len = (seq_len - window_size) // stride + 1
    out = torch.zeros(batch_size, out_len, dtype=torch.long, device=x.device)
    
    for i in range(0, out_len*stride, stride):
        start = i 
        end = min(start + window_size, seq_len)
        window = x[:, start:end]
        
        above_thresh = window > (1 - pvalue)
        num_above = above_thresh.sum(dim=1)
        out[:, i//stride] = (num_above >= window_size/2).long()
    return out


#简单效果可视化部分
#还没兼容独热编码，之后兼容下独热编码
def seeresult(data,model,codemode,device):
    for step,(batch_x,batch_y,chrinfo) in enumerate(data):
        x = ut.codeall(batch_x,codemode).to(device).to(torch.float)
        # labels = ut.makelabel(batch_y[0],batch_y[1],1000).to(device).to(torch.float)
        labels =batch_y.to(device).to(torch.float)

        #对label处理下，防止inf
        labels = abs(labels-torch.tensor(0.0000001,dtype=torch.float, device=device)).cpu().detach()

        pred = model(x).cpu().detach()

        plt.figure(figsize=(30,40))
        for i in range(pred.shape[0]):
            a = pred[i]
            b = labels[i]
            plt.subplot(32+1,2,i+1)
            c = list(range(a.shape[0]))
            plt.plot(c,a,color="r")
            plt.plot(c,b,color="b")
        #时机成熟再保存
        # plt.savefig()
        plt.show()
        break

def seeresults(data,Model,Para,device,allcodemode,param):
    plt.figure(figsize=(30,40))

    
    for step,(batch_x,batch_y,chrinfo) in enumerate(data):
        # labels = ut.makelabel(batch_y[0],batch_y[1],1000).to(device).to(torch.float)
        labels =batch_y.to(device).to(torch.float)

        #对label处理下，防止inf
        labels = abs(labels-torch.tensor(0.0000001,dtype=torch.float, device=device)).cpu().detach()

        for i in range(len(Para)):
            # model = torch.load(Para[i],map_location=device)
            x = ut.codeall(batch_x,allcodemode[i]).to(device).to(torch.float)

            model = Model[i](*param[i]).to(device)
    #         model = DataParallel(model)#这里记得改回来
            model.load_state_dict(torch.load(Para[i]))
            pred = model(x).cpu().detach()

            for i in range(pred.shape[0]):
                a = pred[i]
                b = labels[i]
                plt.subplot(32+1,2,i+1)
                c = list(range(a.shape[0]))
                plt.plot(c,a)
                plt.plot(c,b,color="b")
        #时机成熟再保存
        # plt.savefig()
        plt.show()
        break

#绘图部分，绘图部分着重于对比不同模型间差异，要求函数可以加入不同模型的结果
#点精度的ROC曲线绘制
def get_p_roc(data,Model,para:list,allcodemode,names:list,device,param,savepath=None):
    '''
    获取模型评估结果
    :Model:模型类的列表(已弃用)
    '''
    # data.batch_size = 256
    # if len(Model) != len(para):
    #     raise ValueError("every model must have it`para")
    y_trues = []
    y_scores = []
    for i in range(len(para)):
        # model = torch.load(para[i],map_location=device)
        model = Model[i](*param[i]).to(device)
        model.load_state_dict(torch.load(para[i]))
        for step,(batch_x,batch_y,chrinfo) in enumerate(data):
            x = ut.codeall(batch_x,allcodemode[i]).to(device).to(torch.float)
            # labels = ut.makelabel(batch_y[0],batch_y[1],1000).to(device).to(torch.float)
            labels =batch_y.to(device).to(torch.float)

            #对label处理下，防止inf
            judge = labels
            labels = abs(labels-torch.tensor(0.0000001,dtype=torch.float, device=device)).cpu().detach()

            pred = model(x).cpu().detach()

            y_trues.append(judge.cpu().flatten())
            y_scores.append(pred.flatten())
            break
        model = None
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated() 
    
    plot_multi_roc(y_trues,y_scores,names,savepath)

def get_r_roc(data,Model,Para,allcodemode,names,device,windowlen,windowstep,pvalue,savepath=None):
    '''
    获取模型评估结果
    :Model:模型保存的参数路径
    '''
    # data.batch_size = 256
    # if len(Model) != len(para):
    #     raise ValueError("every model must have it`para")
    y_trues = []
    y_scores = []
    for i in range(len(Para)):
        # model = torch.load(Para[i],map_location=device)
        model = Model[i]().to(device)
        model.load_state_dict(torch.load(Para[i]))
        for step,(batch_x,batch_y,chrinfo) in enumerate(data):
            x = ut.codeall(batch_x,allcodemode[i]).to(device).to(torch.float)
            # labels = ut.makelabel(batch_y[0],batch_y[1],1000).to(device).to(torch.float)
            labels =batch_y.to(device).to(torch.float)

            #对label处理下，防止inf
            judge = labels
            labels = abs(labels-torch.tensor(0.0000001,dtype=torch.float, device=device)).cpu().detach()

            pred = model(x).cpu().detach()

            y_trues.append(sliding_window_binarize(judge.cpu(),windowlen,pvalue,stride=windowstep).flatten())
            y_scores.append(sliding_window_binarize(pred,windowlen,pvalue,stride=windowstep).flatten())
            break
        model = None
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated() 
    
    plot_multi_roc(y_trues,y_scores,names,savepath)

def plot_roc_curve(y_true, y_score, name):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    # 计算最佳cutoff值
    best_threshold_index = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_threshold_index]
    Youden_index = np.argmax(tpr - fpr)  # Only the first occurrence is returned.
    optimal_threshold = thresholds[Youden_index]

    print("Best cutoff value:", optimal_threshold)
    plt.plot(fpr, tpr, label=f'{name} (area = {roc_auc:.2f})')
    return roc_auc

#绘制ROC曲线的封装好的代码    
def plot_multi_roc(y_trues, y_scores, names,savepath=None):
    plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='Luck')
    
    aucs = []
    for i, name in enumerate(names):
        y_true = y_trues[i]
        y_score = y_scores[i]
        aucvalue = plot_roc_curve(y_true, y_score, name)
        aucs.append(aucvalue)
        
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    if savepath != None:
        plt.savefig(savepath)
    return aucs
from torch.nn.parallel import DataParallel

def CommonPlot(data,Model,Para,device,allcodemode,pvalue,windowlen,windowstep,param,savepath=None):
    '''
    常规评估方式的作图
    :model:包括模型类的List
    :para:参数的List
    '''
    categ = [["pACC","pPrecission","pRecall","pF1"],["rACC","rPrecission","rRecall","rF1"]]
    
    # if len(Model) != len(para):
    #     raise ValueError("every model must have it`para")
    stats = {

    }
    for i in range(len(Para)):
        # model = torch.load(Para[i],map_location=device)
        
        model = Model[i](*param[i]).to(device)
#         model = DataParallel(model)#这里记得改回来
        model.load_state_dict(torch.load(Para[i]))
#         _,a = testprocess_2(data,model,allcodemode[i],None,device,pvalue,windowlen,windowstep)
        _,a,b = testprocess(data,model,allcodemode[i],None,device,pvalue,windowlen,windowstep)
        stats[f"model{i}"] = [list(a),list(b)]
#         stats[f"model{i}"] = [list(a)]

        model = None
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated() 

    radar_plot(categ,stats,savepath=savepath)
    print(pvalue,stats)
    return stats

#雷达图
def radar_plot(names:list,data:dict,savepath=None):
    '''
    :names:网状图不同的标题
    :data:不同的数据列，形式如下
        {
            "m1":[[...],[...],...]
            "m2":[[...],[...],...]
        }
    '''
    fig, ax = plt.subplots(figsize=(18, 12), subplot_kw=dict(polar=True),nrows=1,ncols=len(names))
    #洗一下数据
    for i in range(len(names)):
        name = np.array(names[i])
        angles=np.linspace(0, 2*np.pi, len(name), endpoint=False).tolist()
        angles+=angles[:1]
        for k,v in data.items():
            dt = np.concatenate((np.array(v[i]),[np.array(v[i][0])]))
            ax[i].fill(angles, dt, alpha=0.4,label=k)
        ax[i].legend()
        ax[i].set_yticklabels([])
        ax[i].set_xticks(angles[:-1])
        ax[i].set_xticklabels(name)
        ax[i].set_title("-".join(name))
    if savepath != None:
        plt.savefig(savepath)
    plt.show()

#模型用时和内存占用对比函数
def timeplot(data,Model,Para,allcodemode,devicename:str=("gpu","cpu"),lossfunc=None,savepath=None):
    '''
    这个函数负责对比模型时间，内存/显存消耗
    :data:任意数据集loader
    :Model:模型类之列表
    :para:模型参数路径列表
    :devicename:评估使用的设备名称
    :lossfunc:如果为默认None，就不进行训练过程
    '''
    if devicename == "gpu":
        device = torch.device('cuda:0')
    elif devicename == "cpu":
        device = torch.device("cpu")
        process = psutil.Process()
        basecache = process.memory_info().rss/1024**2
    else:
        raise ValueError("devicename false")
    
    timelis = []
    cachelis = []
    n = 10

    for i in range(len(Para)):
        starttime =  time.time()
        # model = torch.load(Para[i],map_location=device)
        
        if lossfunc != None:
            optimizer = optim.Adam(model.parameters(),lr=0.00001,betas=(0.9, 0.999), eps=1e-8)

        model = Model[i]().to(device)
        model.load_state_dict(torch.load(Para[i]))
        j = 0
        for step,(batch_x,batch_y,chrinfo) in enumerate(data):
            j+=1
            x = ut.codeall(batch_x,allcodemode[i]).to(device).to(torch.float)
            # labels = ut.makelabel(batch_y[0],batch_y[1],1000).to(device).to(torch.float)
            labels =batch_y.to(device).to(torch.float)

            #对label处理下，防止inf
            judge = labels
            labels = abs(labels-torch.tensor(0.0000001,dtype=torch.float, device=device))
            pred = model(x)

            if lossfunc != None:
                loss = lossfunc(labels,pred).to(torch.float)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if j > n:
                break
        endtime = time.time()-starttime
        timelis.append(endtime)
        #处理一下
        if devicename=="cpu":
            cachelis.append((process.memory_info().rss/1024**2)-basecache)
        elif devicename=="gpu":
            cachelis.append(torch.cuda.memory_allocated()/1024**2)
        model = None
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
    
    fig, ax = plt.subplots(figsize=(18, 12), subplot_kw=dict(polar=True),ncols=2)
    ax[1].bar(list(range(len(timelis))),timelis)
    ax[1].set_title("time")
    ax[1].xticks(list(range(len(timelis))),[f"model{i}" for i in range(len(timelis))])
    ax[1].bar(list(range(len(cachelis))),cachelis)
    ax[1].set_title("time")
    ax[1].xticks(list(range(len(cachelis))),[f"model{i}" for i in range(len(cachelis))])
    fig.show()
    if savepath != None:
        fig.savefig(savepath)


def violinplot(dataset,Para,allcodemode,device,replication,pvalue,windowlen,windowstep,savepath=None):
    '''
    根据数据绘制小提琴图
    :
    :replication:获得的数据重复次数
    '''
    lis = [[] for i in range(len(Para)*2)]
    name = [f"model{i//2+1}-{'pF1' if i%2==0 else 'rF1'}" for i in range(len(Para)*2)]
    for i in range(replication):
        seed = randint(0,500)
        subdata,_ = train_test_split(dataset, test_size=0.2, random_state=seed)
        for j in range(len(Para)):
            model = torch.load(Para[i],map_location=device)
            # model = Model[j]().to(device)
            # model.load_state_dict(torch.load(para[j]))

            _,a,b = testprocess(subdata,model,allcodemode[j],None,device,pvalue,windowlen,windowstep)
            lis[j*2].append(a[-1])
            lis[j*2+1].append(b[-1])

            model = None
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated() 
        
    lis = [np.array(i) for i in lis]
    plt.violinplot(lis,showmeans=True)
    plt.xticks(range(1,len(lis)+1), name)
    if savepath != None:
        plt.savefig(savepath)
    plt.show()

#计算
def calculate_position_ratio(label, pred, pvalue):
    mask = (label > 1 - pvalue)
    region = pred[mask]
    num_less_than_pvalue = torch.sum(region < pvalue).item()
    num_between_pvalue_and_1_minus_pvalue = torch.sum((region >= pvalue) & (region <= 1 - pvalue)).item()
    num_greater_than_1_minus_pvalue = torch.sum(region > 1 - pvalue).item()
    
    total_positions = region.size(0)
    ratio_less_than_pvalue = num_less_than_pvalue / total_positions
    ratio_between_pvalue_and_1_minus_pvalue = num_between_pvalue_and_1_minus_pvalue / total_positions
    ratio_greater_than_1_minus_pvalue = num_greater_than_1_minus_pvalue / total_positions
    
    return {
        "num_less_than_pvalue": num_less_than_pvalue,
        "num_between_pvalue_and_1_minus_pvalue": num_between_pvalue_and_1_minus_pvalue,
        "num_greater_than_1_minus_pvalue": num_greater_than_1_minus_pvalue,
        "ratio_less_than_pvalue": ratio_less_than_pvalue,
        "ratio_between_pvalue_and_1_minus_pvalue": ratio_between_pvalue_and_1_minus_pvalue,
        "ratio_greater_than_1_minus_pvalue": ratio_greater_than_1_minus_pvalue
    }


#区分开数据的置信程度
def rankdata(pred,label,pvalue) -> int:
    mask = (label > 1 - pvalue)
    region = pred[mask]
    num_less_than_pvalue = torch.sum(region < pvalue).item()
    num_between_pvalue_and_1_minus_pvalue = torch.sum((region >= pvalue) & (region <= 1 - pvalue)).item()
    num_greater_than_1_minus_pvalue = torch.sum(region > 1 - pvalue).item()
    
    total_positions = region.size(0)
    ratio_less_than_pvalue = num_less_than_pvalue / total_positions
    ratio_between_pvalue_and_1_minus_pvalue = num_between_pvalue_and_1_minus_pvalue / total_positions
    ratio_greater_than_1_minus_pvalue = num_greater_than_1_minus_pvalue / total_positions

    my_list = [ratio_less_than_pvalue,ratio_between_pvalue_and_1_minus_pvalue,ratio_greater_than_1_minus_pvalue]

    return my_list.index(max(my_list))


#从数据集获取区分开的不同程度的数据,将区分开的数据写入文档，便于后续MEME分析等
def getrankdata(dataloader,para,codemode,pvalue,device,fileappendix="None"):
    '''
    :model:模型类
    :para:保存的训练好的参数路径
    :pvalue:置信度,现在的方法只区分高置信和低置信的数据
    '''
    message = {
        1:[],
        2:[],
        3:[]
    }
    model = torch.load(para,map_location=device)

    # model = model.load_state_dict(torch.load(para))
    for step,(batch_x,batch_y,chrinfo) in enumerate(dataloader):
        x = ut.codeall(batch_x,mode=codemode).to(device).to(torch.float)
        # labels = ut.makelabel(batch_y[0],batch_y[1],1000).to(device).to(torch.float)
        labels =batch_y.to(device).to(torch.float)

        #对label处理下，防止inf
        judge = labels
        labels = abs(labels-torch.tensor(0.0000001,dtype=torch.float, device=device)).cpu().detach()

        pred = model(x).cpu().detach()

        #循环处理每一个
        for i in range(judge.shape[0]):
            a = judge[i]
            b = pred[i]
            rank = rankdata(a,b,pvalue)
            message[rank].append(chrinfo[i])
    if fileappendix != None:
        #分别存储生成的序列
        pass
    return message
