import torch
from torch import nn
import numpy as np
from test import test
from Functions import  getLRScheduler
import torch.nn.utils.clip_grad as clip_grad
import config as cfg
import os
import sklearn.metrics as metrics
from Functions import extract_batch_size

batch_size = cfg.batch_size

def train(net, X_train, y_train, X_test, y_test, opt, criterion, epochs=100, clip_val=15, model_save=''):
    print("\n\n********** Running training! ************\n\n")

    sched = getLRScheduler(optimizer=opt)
    #if (train_on_gpu):
    if (torch.cuda.is_available() ):
        net.cuda()

    train_losses = []
    net.train()

    best_accuracy = 0.0
    min_loss=100.0
    best_model = None
    epoch_train_losses = []
    epoch_train_acc = []
    epoch_test_losses = []
    epoch_test_acc = []
    epoch_train_f1score = []
    epoch_train_auc = []
    epoch_test_f1score = []
    epoch_test_auc = []
    params = {
        'best_model' : best_model,
        'epochs' : [],
        'train_loss' : [],
        'test_loss' : [],
        'lr' : [],
        'train_accuracy' : [],
        'test_accuracy' : [],
        'train_f1score':[],
        'test_f1score':[],
        'train_auc': [],
        'test_auc': [],
    }
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        train_losses = []
        step = 1

        h = net.init_hidden(batch_size)

        train_accuracy = 0
        train_len = len(X_train)
        train_f1score = 0
        train_auc = 0

        while step * batch_size <= train_len:
            batch_xs = extract_batch_size(X_train, step, batch_size)
            # batch_ys = one_hot_vector(extract_batch_size(y_train, step, batch_size))
            batch_ys = extract_batch_size(y_train, step, batch_size)

            inputs, targets = torch.from_numpy(batch_xs), torch.from_numpy(batch_ys.flatten('F'))#扁平函数flatten，F按列顺序
            # inputs, targets = torch.from_numpy(batch_xs), torch.from_numpy(batch_ys)
            #if (train_on_gpu):
            if (torch.cuda.is_available() ):
                inputs, targets = inputs.cuda(), targets.cuda()

            h = tuple([each.data for each in h])
            opt.zero_grad()

            output, output_train = net(inputs.float(), h)
            #print(output)
            # print("lenght of inputs is {} and target value is {}".format(inputs.size(), targets.size()))
            train_loss = criterion(output_train, targets.long())
            train_losses.append(train_loss.item())

            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == targets.view(*top_class.shape).long()
            train_accuracy += torch.mean(equals.type(torch.FloatTensor))
            equals = top_class
            train_f1score += metrics.f1_score(top_class.cpu(), targets.view(*top_class.shape).long().cpu(),average='macro')
            train_auc += metrics.roc_auc_score(targets.view(*top_class.shape).long().cpu(),output[:, 1].detach().cpu().numpy(), average='macro')
            train_loss.backward()
            #for name, param in net.named_parameters():
               # print(name, param.grad)
            clip_grad.clip_grad_norm_(net.parameters(), clip_val)
            opt.step()
            step += 1

        p = opt.param_groups[0]['lr']
        params['lr'].append(p)
        params['epochs'].append(epoch)
        sched.step()
        train_loss_avg = np.mean(train_losses)
        train_accuracy_avg = train_accuracy/(step-1)
        epoch_train_losses.append(train_loss_avg)
        epoch_train_acc.append(train_accuracy_avg)
        epoch_train_f1score.append(train_f1score / (step - 1))
        epoch_train_auc.append(train_auc / (step - 1))
        print("Epoch: {}/{}...".format(epoch + 1, epochs),
              ' ' * 16 + "Train Loss: {:.4f}".format(train_loss_avg),
              "Train accuracy: {:.4f}...".format(train_accuracy_avg))
        test_loss,test_f1score,test_auc,test_accuracy, best_accuracy, min_loss,best_model = test(net, X_test, y_test, criterion, best_accuracy,min_loss,best_model, test_batch=batch_size)
        epoch_test_losses.append(test_loss)
        epoch_test_acc.append(test_accuracy)
        epoch_test_f1score.append(test_f1score)
        epoch_test_auc.append(test_auc)
        if epoch % 10 == 0:
            if not os.path.exists("Checkpoints_window/" + model_save):
                os.makedirs("Checkpoints_window/" + model_save)
            torch.save(net, "Checkpoints_window/" + model_save + "/" + "model_{}.pth".format(str(epoch).zfill(3)))
            torch.save(best_model, "Checkpoints_window/" + model_save + "/" + "best_model.pth")
        if ((epoch+1) % 10 == 0):
            print("Epoch: {}/{}...".format(epoch + 1, epochs),
                  ' ' * 16 + "Test Loss: {:.4f}...".format(test_loss),
                  "Test accuracy: {:.4f}...".format(test_accuracy),
                  "Test F1: {:.4f}...".format(test_f1score))

    print('!!! Best accuracy is : {} !!!'.format(best_accuracy))
    print('!!! Best loss is : {} !!!'.format(min_loss))
    params['best_model'] = best_model
    params['train_loss'] = epoch_train_losses
    params['test_loss'] = epoch_test_losses
    params['train_accuracy'] = epoch_train_acc
    params['test_accuracy'] = epoch_test_acc
    params['train_f1score'] = epoch_train_f1score
    params['test_f1score'] = epoch_test_f1score
    params['train_auc'] = epoch_train_auc
    params['test_auc'] = epoch_test_auc
    return params
