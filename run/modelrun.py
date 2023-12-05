import torch
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import model

def get_dataset_all(region, strand):
    region = list(region.upper())
    le = LabelBinarizer().fit(np.array([["A"], ["C"], ["T"], ["G"]]))
    variables = []
    if strand == "forward":
        hot_seq = np.transpose(le.transform(region))
    else:
        dnatable = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        rev = [dnatable[i] for i in region[::-1]]
        hot_seq = np.transpose(le.transform(rev))
    variables.append(hot_seq.transpose())
    return np.array(variables)


def rloop(data01, cut_off):
    rloop = {}
    count, num = 0, 0
    xregion, textlabel = [], []

    for m in range(0, 4810, 10):
        n = m + 200
        region = data01[m:n]
        if sum(region) / len(region) >= float(cut_off):
            count += 1
            rloop[count] = [m, n]
    for m in range(1, count):
        x = rloop[m][0]
        y = rloop[m][1]
        z = rloop[m + 1][0]
        h = rloop[m + 1][1]
        if y > z:
            del rloop[m]
            del rloop[m + 1]
            rloop[m + 1] = [x, h]

    for value in rloop.values():
        xregion.append(value)
        textlabel.append(value[0])
        textlabel.append(value[1])
    return (xregion, textlabel)


def visial(data, xregion, textlabel, position, name):
    contig = {
        "font.family":'serif',
        "font.size":15,
        "mathtext.fontset":'stix'
    }
    rcParams.update(contig)
    
    mpl.use('Agg')
    plt.suptitle("R-loop region prediction result - {}".format(name),fontsize=15,fontweight='bold')
    plt.xlabel("Position",fontsize=15,fontweight='bold')
    plt.ylabel("R-loop probability",fontsize=15,fontweight='bold')
    xline = np.array([i for i in range(0, 5000)])
    for x in textlabel:
        plt.text(x, 1.1, x, ha='center', va='bottom', fontsize=11)
    for i in xregion:
        plt.fill_between(xline[i[0]:i[1]], 0, 1.1, color="lightblue", alpha=0.5)
    plt.bar([i for i in range(0, 5000)], data, 1, color="blue",alpha=0.8)
    plt.savefig(position)
    plt.close()


net = torch.load('best_model.pth', map_location=torch.device('cpu'))
