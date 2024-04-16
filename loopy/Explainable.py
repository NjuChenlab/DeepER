'''
这个模块包括可解释性方法以及可视化
'''
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import loopy.utils as ut


def smoothgradcampp(model, input_data,pvalue=0.05,num_samples=50, stdev_spread=0.05, magnitude=False):
    stdev = stdev_spread * (input_data.max() - input_data.min())

    total_gradients = 0
    for i in range(num_samples):
        noise = torch.randn_like(input_data) * stdev
        noisy_input = input_data + noise
        noisy_input = Variable(noisy_input, requires_grad=True)

        output = model(noisy_input)

        #先支持对预测正例部分的反向传播
        summed_output = torch.sum(output[output > 1 - pvalue])
        summed_output.backward()

        gradients = noisy_input.grad
        if magnitude:
            gradients = gradients ** 2

        total_gradients += gradients

    mean_gradients = total_gradients / num_samples
    return mean_gradients


#利用某条数据绘制显著图
def saliency_map(input_data,model,lossfunc):
    pass


#利用某条数据的显著图绘制趋势图
def trend_map():

    pass
# 为自循环神经网络可解释性准备
def sp_deeper_explain(data,model,codemode,device):
    for step,(batch_x,batch_y,chrinfo) in enumerate(data):
        x = ut.codeall(batch_x,codemode).to(device).to(torch.float)
        # labels = ut.makelabel(batch_y[0],batch_y[1],1000).to(device).to(torch.float)
        labels =batch_y.to(device).to(torch.float)

        #对label处理下，防止inf
        labels = abs(labels-torch.tensor(0.0000001,dtype=torch.float, device=device)).cpu().detach()

        pred = model(x)

        fig, ax = plt.subplots(figsize=(18, 24),nrows=15,ncols=4)
        for i in range(4):
            a = pred[i]
            print(a.shape)
            c = list(range(a.shape[1]))
            for j in range(12):
                aa = a[j]
                b = labels[j]
                ax[(j//4)*5+i,j%4].plot(c,aa)
                ax[(j//4)*5+i,j%4].plot(c,b)
        fig.show()
        return 
