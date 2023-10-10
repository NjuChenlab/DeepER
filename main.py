import torch
import numpy as np
from loadDataset import get_region, get_dnaseq,get_dataset,get_label,get_dataset_all
from train import train
from torch import nn
from model import Res_Bidir_LSTMModel,init_weights
from Functions import plot, evaluate,evaluate_test
import config as cfg
import sys
import os
from sklearn.preprocessing import LabelBinarizer
from joblib import load,dump
from sparse_vector.sparse_vector import SparseVector
import random
import argparse
# LSTM Neural Network's internal structure
n_hidden = cfg.n_hidden
n_classes = cfg.n_classes
epochs = cfg.n_epochs
learning_rate = cfg.learning_rate
weight_decay = cfg.weight_decay
clip_val = cfg.clip_val
diag = cfg.diag

# Training
# check if GPU is available
#train_on_gpu = torch.cuda.is_available()
if (torch.cuda.is_available() ):
    print('Training on GPU')
else:
    print('GPU not available! Training on CPU. Try to keep n_epochs very small')

def main():
    if cfg.is_train:
        print('find region')
        posi_region, neg_region, for_label, rev_label =get_region(args.width,args.label,args.neg)
        print('posi_region,negi_region:', len(posi_region), len(neg_region))
        print('get dna sequence and length of sequence')
        dna, len_chrom = get_dnaseq(args.pkl_path)

        np.random.seed(10)
        np.random.shuffle(neg_region)

        random.seed(100)
        posi_region=posi_region+neg_region

        np.random.seed(20)
        np.random.shuffle(posi_region)

        print('divide train,test and val datasets')
        train_region = posi_region[0:int(len(posi_region) * 0.7)]
        print('train samples:', len(train_region))
        val_region = posi_region[int(len(posi_region)* 0.7):int(len(posi_region)*0.9)]
        print('validation samples:', len(val_region))
        test_region = posi_region[int(len(posi_region)*0.9):]
        print('test samples:', len(test_region))
        print('get train')
        X_train, y_train = get_dataset_all(train_region, dna, for_label, rev_label)
        print('get val')
        X_val, y_val = get_dataset_all(val_region, dna, for_label, rev_label)
        print('get test')
        X_test, y_test = get_dataset_all(test_region, dna, for_label, rev_label)

        y_all = np.append(y_train, y_test)
        y_all = np.append(y_all, y_val)
        one = np.sum(y_all.flatten())
        zero = len(posi_region) * args.width - one
        ratio = int(zero / one)
        print('ratio:', ratio)
        del y_all

        print("Some useful info to get an insight on dataset's shape and normalisation:")
        print("(X shape, y shape, every X's mean, every X's standard deviation)")
        print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))

        for lr in learning_rate:
            arch = cfg.arch
            if arch['name'] == 'LSTM1' or arch['name'] == 'LSTM2':
                net = LSTMModel()
            elif arch['name'] == 'Res_LSTM':
                net = Res_LSTMModel()
            elif arch['name'] == 'Res_Bidir_LSTM':
                net = Res_Bidir_LSTMModel()
            elif arch['name'] == 'Bidir_LSTM1' or arch['name'] == 'Bidir_LSTM2':
                net = Bidir_LSTMModel()
            else:
                print("Incorrect architecture chosen. Please check architecture given in config.py. Program will exit now! :() ")
                sys.exit()
            print(net)
            if cfg.checkpoint:
              if os.path.isfile(cfg.checkpoint):
                  print("=> loading checkpoint '{}'".format(cfg.checkpoint))
                  net = torch.load(cfg.checkpoint)

              else:
                  print("=> no checkpoint found at ...")
                  net.apply(init_weights)
            else:
              net.apply(init_weights)
            print(diag)
            opt =torch.optim.Adam(net.parameters(),lr=lr,weight_decay=cfg.weight_decay)
            number_gpu = 0
            torch.cuda.set_device(number_gpu)
            criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, ratio])).cuda(number_gpu)

            net = net.float()
            params = train(net, X_train, y_train, X_val, y_val, opt=opt, criterion=criterion, epochs=epochs, clip_val=clip_val, model_save=arch['name'])

            if not os.path.exists("img_window/" + arch['name']):
              os.makedirs("img_window/" + arch['name'])
            evaluate(params['best_model'], X_test, y_test, criterion, save_dir="img_window/" + arch['name'])
            plot(params['epochs'], params['train_loss'], params['test_loss'], 'loss', lr, save_dir="img_window/" + arch['name'])
            plot(params['epochs'], params['train_accuracy'], params['test_accuracy'], 'accuracy', lr,"img_window/" + arch['name'])
            plot(params['epochs'], params['train_f1score'], params['test_f1score'], 'F1', lr,save_dir="img_window/" + arch['name'])
            plot(params['epochs'], params['train_auc'], params['test_auc'], 'AUC', lr,save_dir="img_window/" + arch['name'])
            #plot(params['lr'], params['train_loss'], params['test_loss'], 'loss_lr', lr)
    else:
        for lr in learning_rate:
            arch = cfg.arch
            if arch['name'] == 'LSTM1' or arch['name'] == 'LSTM2':
                net = LSTMModel()
            elif arch['name'] == 'Res_LSTM':
                net = Res_LSTMModel()
            elif arch['name'] == 'Res_Bidir_LSTM':
                net = Res_Bidir_LSTMModel()
            elif arch['name'] == 'Bidir_LSTM1' or arch['name'] == 'Bidir_LSTM2':
                net = Bidir_LSTMModel()
            else:
                print(
                    "Incorrect architecture chosen. Please check architecture given in config.py. Program will exit now! :() ")
                sys.exit()
            print(net)
            if cfg.checkpoint:
                if os.path.isfile(cfg.checkpoint):
                    print("=> loading checkpoint '{}'".format(cfg.checkpoint))
                    net = torch.load(cfg.checkpoint,map_location='cuda:0')
                    print(net) 
                else:
                    print("=> no checkpoint found at ...")
                    net.apply(init_weights)
                    sys.exit()
            else:
                net.apply(init_weights)
                sys.exit()
            print(diag)


            number_gpu = 1
            torch.cuda.set_device(number_gpu)
            criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 27.0])).cuda(number_gpu)
            net = net.float()
            if (torch.cuda.is_available()):
                net.cuda()
            if not os.path.exists("predict/" + arch['name']):
                os.makedirs("predict/" + arch['name'])
            for_label, rev_label = get_label(args.label)
            print('dna')
            dna, len_chrom = get_dnaseq(args.pkl_path)

            width=args.width
            map_path=args.label
            for file in os.listdir(map_path):
                if file.split('_')[1]=='for':
                    strand="+"
                else:
                    strand="-"
                chroms = ['chrY']
              #  chroms=[f'chr{i}' for i in list(range(1,23))+['X','Y']]
                for chrom in chroms:
                    print(chrom +' is being predicted' )
                    length=len_chrom[chrom]
                    data_predict = []
                    stored= np.zeros((2,length))
                    for ele in range(0,length-width,width):
                        space = [ele,min(ele + width,length)]
                        data_predict.append([chrom,space[0],space[1],strand])
                    for i in range(0, len(data_predict), 1):
                        item=data_predict[i]
                        X_predict, y_predict = get_dataset(item,dna,for_label,rev_label)
                     #   print(X_predict.shape,y_predict.shape)
                        out, top_class = evaluate_test(net, X_predict, y_predict, criterion,
                                                          save_dir="predict/" + arch['name'])
                        if strand=="+":
                            prob1 = out.cpu().detach().numpy().flatten()
                            label = top_class.cpu().detach().numpy().flatten()
                        else:
                            prob1 = out.cpu().detach().numpy().flatten()[::-1]
                            label = top_class.cpu().detach().numpy().flatten()[::-1]
                        start = data_predict[i][1]
                        end = data_predict[i][2]
                        stored[0][start:end] = label
                        stored[1][start:end] = prob1

                    if not os.path.exists("predict/result/"):
                        os.makedirs("predict/result/")
                    if strand=="+":
                        dump(stored, f"predict/result/{chrom}_for")
                    else:
                        dump(stored, f"predict/result/{chrom}_rev")
            #evaluate(net, X_val, y_val, criterion, save_dir="predict/" + arch['name'])


# 创建一个 ArgumentParser 对象
parser = argparse.ArgumentParser(description='Utility for train model and predict genome-wide R-loops')
# 添加命令行参数
parser.add_argument('-width',dest='width', type=int, required=True,help='Divide the length of dataset interval')
parser.add_argument('-label',dest='label', type=str,  required=True,help='path to label file')
parser.add_argument('-neg',dest='neg', type=str, required=True,help='negative 5k interval file (.bed)')
parser.add_argument('-pkl_path',dest='pkl_path',type=str, required=True, help='the path of genome file (.pkl)')
# 解析命令行参数
args = parser.parse_args()

main()






