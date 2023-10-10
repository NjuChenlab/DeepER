import torch
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import config as cfg
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib
import math
from joblib import load,dump

matplotlib.use('Agg')

LABELS = [0,1]
n_classes = cfg.n_classes
n_epochs_hold = cfg.n_epochs_hold
n_epochs_decay = cfg.batch_size - n_epochs_hold
epochs = cfg.n_epochs
# Define function to generate batches of a particular size

def extract_batch_size(_train, step, batch_size):
    shape = list(_train.shape)
    shape[0] = batch_size
    batch = np.empty(shape)

    for i in range(batch_size):
        index = ((step - 1) * batch_size + i) % len(_train)
        batch[i] = _train[index]

    return batch


def getLRScheduler(optimizer):
    def lambdaRule(epoch):
        lr_l = 1.0 - max(0, epoch - n_epochs_hold) / float(n_epochs_decay + 1)
        return lr_l

    schedular = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdaRule)
    #schedular = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    return schedular

def plot(x_arg, param_train, param_test, label, lr,save_dir='.'):
    plt.figure()
    plt.plot(x_arg, param_train, color='blue', label='train')
    plt.plot(x_arg, param_test, color='red', label='test')
    plt.legend()
    if (label == 'accuracy'):
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.title('Training and Test Accuracy', fontsize=20)
        plt.savefig(save_dir + '/' +'Accuracy_' + str(epochs) + str(lr) +'.pdf')
        #plt.show()
    elif (label == 'loss'):
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Training and Test Loss', fontsize=20)
        plt.savefig(save_dir + '/' +'Loss_' + str(epochs) + str(lr) + '.pdf')
        #plt.show()
    elif (label == 'F1'):
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('F1 score', fontsize=14)
        plt.title('Training and Test F1score', fontsize=20)
        plt.savefig(save_dir + '/' +'F1score_' + str(epochs) + str(lr) + '.pdf')
        #plt.show()
    elif (label == 'AUC'):
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('ROC AUC', fontsize=14)
        plt.title('Training and Test AUC', fontsize=20)
        plt.savefig(save_dir + '/' +'AUC' + str(epochs) + str(lr) + '.pdf')
        #plt.show()
    else:
        plt.xlabel('Learning rate', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Training loss and Test loss with learning rate', fontsize=20)
        plt.savefig(save_dir + '/' +'Loss_lr_' + str(epochs) + str(lr) + '.pdf')
        #plt.show()

def evaluate(net, X_test, y_test, criterion, save_dir='.'):
    torch.cuda.empty_cache()
    test_batch = len(X_test)
    net.eval()
    test_h = net.init_hidden(test_batch)
    inputs, targets = torch.from_numpy(X_test), torch.from_numpy(y_test.flatten('F'))
    if (torch.cuda.is_available() ):
            inputs, targets = inputs.cuda(), targets.cuda()

    test_h = tuple([each.data for each in test_h])
    output, output_test = net(inputs.float(), test_h)
    test_loss = criterion(output_test, targets.long())
    top_p, top_class = output.topk(1, dim=1)
    targets = targets.view(*top_class.shape).long()
    equals = top_class == targets

    if (torch.cuda.is_available() ):
            top_class, targets = top_class.cpu(), targets.cpu()
            output=output.cpu()
    precision, recall, thresholds = metrics.precision_recall_curve(targets, output[:, 1].detach().cpu().numpy())
    FRP, TRP, roc_thresholds = metrics.roc_curve(targets, output[:, 1].detach().cpu().numpy())
    test_accuracy = torch.mean(equals.type(torch.FloatTensor))
    test_f1score = metrics.f1_score(top_class, targets, average='macro')


    print("Final loss is: {}".format(test_loss.item()))
    print("Final accuracy is: {}". format(test_accuracy))
    print("Final f1 score is: {}".format(test_f1score))

    confusion_matrix = metrics.confusion_matrix(top_class, targets)
    print("---------Confusion Matrix--------")
    print(confusion_matrix)
    
    with open(save_dir+'/'+"printf.txt",'w') as fp:
      fp.write("Final loss is: {}\n".format(test_loss.item()))
      fp.write("Final accuracy is: {}\n". format(test_accuracy))
      fp.write("Final f1 score is: {}\n".format(test_f1score))
      fp.write("---------Confusion Matrix--------\n")
      fp.write(str(confusion_matrix))

    calculate_asei_rloop(confusion_matrix,titel='rloop_asei',save_dir=save_dir)
    normalized_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100
    plotPRcurve(precision, recall, save_dir)
    plotROCcurve(TRP, FRP, save_dir)
    plotConfusionMatrix(normalized_confusion_matrix, save_dir)
    f1 = 2 * recall * precision / (recall + precision)
    optimal_f1_index = np.argmax(f1)
    optimal_threshold = thresholds[optimal_f1_index]
    print("optimal_threshold:", optimal_threshold)
    f1 = f1[0:-1]
    plotf1curve(f1, thresholds, save_dir)

    return output[:,1],top_class


def plotPRcurve(precision,recall,save_dir):
    plt.figure()
    plt.plot(recall,precision)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(save_dir+"/"+"PR.pdf")

def plotROCcurve(TRP,FRP,save_dir):
    plt.figure()
    plt.plot(FRP,TRP)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig(save_dir+"/"+"ROC.pdf")

def plotf1curve(f1,thresholds,save_dir):
    plt.figure()
    plt.plot(thresholds,f1)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.xlabel('threshold')
    plt.ylabel('F1 score')
    plt.savefig(save_dir+"/"+"f1-threshold.pdf")

def evaluate_test(net, X_test, y_test, criterion, save_dir='.'):
    test_batch = len(X_test)
    net.eval()
    test_h = net.init_hidden(test_batch)
    inputs, targets = torch.from_numpy(X_test), torch.from_numpy(y_test.flatten('F'))
    if (torch.cuda.is_available() ):
            inputs, targets = inputs.cuda(), targets.cuda()

    test_h = tuple([each.data for each in test_h])
    output, output_test = net(inputs.float(), test_h)
    test_loss = criterion(output_test, targets.long())
    top_p, top_class = output.topk(1, dim=1)
    return output[:,1],top_class




def plotConfusionMatrix(normalized_confusion_matrix, save_dir='.'):
    plt.figure()
    plt.imshow(
        normalized_confusion_matrix,
        interpolation='nearest',
        cmap=plt.cm.rainbow
    )
    plt.title("Confusion matrix \n(normalised to % of total test data)")
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
    plt.ylabel('Pred label')
    plt.xlabel('True label')
    plt.savefig(save_dir+"/"+"confusion_matrix.pdf")
    #plt.show()

def calculate_asei_rloop(confusion_matrix,titel='',save_dir='.'):

    # titel = 'ophthalmologist_qw'
    # confusion_matrix = [[303,1,10],[2,359,17],[3,15,219]]   # ophthamology A

    # confusion_matrix = [[305, 1, 8], [2, 367, 9], [2, 13, 222]]   # ophthamology B
    # confusion_matrix = [[98, 5, 7], [2, 97, 6], [2, 6, 79]]   # ophthalmologist_lzw
    # confusion_matrix = [[75, 4, 31], [4, 76, 25], [16, 19, 52]]  # ophthalmologist_qw


    
    sum_matrix = sum(sum(i) for i in confusion_matrix)#所有元素相???        if iter_num == 0:
 
    #R = TP/(TP+FN)
    #P = TP/(TP+FP)
    #sen = TP/(TP+FN)
    #spe = TN/(TN+FP)

    P = sum(confusion_matrix[1])
    N = sum_matrix - P
    TP = confusion_matrix[1][1]
    FN = confusion_matrix[0][1]
    FP = confusion_matrix[1][0]
    TN = N - FN
    print('P,N',P,N)
    print(TP,FP,TN,FN)
    current_class_acc = (TP+TN)/(TP+TN+FP+FN)
    # current_class_acc = format(current_class_acc, '.3f')         # current_class_acc1 = ('%.2f' %current_class_acc)
    current_acc_ci_low = current_class_acc - 1.96 * math.sqrt((current_class_acc*(1-current_class_acc))/sum_matrix)
    # current_acc_ci_low = format(current_acc_ci_low, '.3f')
    current_acc_ci_up = current_class_acc + 1.96 * math.sqrt((current_class_acc*(1-current_class_acc))/sum_matrix)
    # current_acc_ci_up = format(current_acc_ci_up, '.3f')
    
    if TP+FN==0:
        current_class_sen=0
        current_sen_ci_low=0
        current_sen_ci_up=0
    else:
        current_class_sen = TP/(TP+FN)
        current_sen_ci_low = current_class_sen - 1.96 * math.sqrt((current_class_sen*(1-current_class_sen))/P)
        current_sen_ci_up = current_class_sen + 1.96 * math.sqrt((current_class_sen*(1-current_class_sen))/P)

    current_class_spe = TN/(TN+FP)
    current_spe_ci_low = current_class_spe - 1.96 * math.sqrt((current_class_spe*(1-current_class_spe))/N)
    current_spe_ci_up = current_class_spe + 1.96 * math.sqrt((current_class_spe*(1-current_class_spe))/N)

    result_value = [[current_class_acc,current_acc_ci_low,current_acc_ci_up],
                    [current_class_sen,current_sen_ci_low,current_sen_ci_up],
                    [current_class_spe,current_spe_ci_low,current_spe_ci_up]
                    ]
    for i in range(0,3):
        for j in range(0,3):
            result_value[i][j] = format(result_value[i][j], '.3f')
            # result_value[i][j] = ('%.3f' % result_value[i][j])
    #print(result_value)
    result_confusion_file = save_dir + '/' + titel + '_assi.txt'
    with open('./' + result_confusion_file, "a") as file_object:
        for i in result_value:
            file_object.writelines(str(i) + '\n')
        file_object.writelines('\n')
        file_object.close()
    # print(current_class_acc(current_acc_ci_low,current_acc_ci_up),current_class_spe,current_class_sen)`



