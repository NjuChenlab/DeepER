import numpy as np
import os
from joblib import load
from sklearn.preprocessing import LabelBinarizer
from sparse_vector.sparse_vector import SparseVector

#divide negative and positive 5k intervals
def get_region(width,path,neg5kfile):
    ints_in=[]
    ints_out=[]
    for file in os.listdir(path):
       # print(file)
        if file.split('_')[1]=="for":
            strand="+"
        else:
            strand="-"
        file_path=path+file
        file_data=load(file_path)
        for key,value in file_data.items():
            chrom=key
        #    print(value)
        #    print(value.shape)
            for st in range(0,value.shape-width,width):
                interval=[st,min(st+width,value.shape)]
                if file_data[chrom][interval[0]:interval[1]].any():
                    if strand=='+':
                        ints_in.append([chrom,interval[0],interval[1],'+'])
                    else:
                        ints_in.append([chrom, interval[0], interval[1], '-'])
                else:
                    if strand=='+':
                        ints_out.append([chrom,interval[0],interval[1],'+'])
                    else:
                        ints_out.append([chrom, interval[0], interval[1], '-'])
    for_path = path + 'label_for'
    rev_path = path + 'label_rev'
    for_label = load(for_path)
    rev_label = load(rev_path)
    ints_out=[]
    with open(neg5kfile,'r') as file:
        for line in file:
            line=line.strip().split('\t')
            chr=line[0]
            start=line[1]
            end=line[2]
            strand=line[5]
            ints_out.append([chr, start, end, strand])
 #   print(ints_in)
    return ints_in,ints_out,for_label,rev_label


def get_dnaseq(path):
    sequence={}
    len_chrom={}
    for file in os.listdir(path):
        chrom=file.split('.')[0]
        file_path=path+file
        sequence.update({chrom:load(file_path)})
        len_chrom.update({chrom:int(len(load(file_path)))})
    return sequence,len_chrom

def get_dataset_all(region,seq,for_label,rev_label):
    #print(region)
    le = LabelBinarizer().fit(np.array([["A"], ["C"], ["T"], ["G"]]))
    tables=[]
    variables=[]
    for re in region:
        chrom=re[0]
        start=int(re[1])
        end=int(re[2])
        strand=re[3]
        if strand=="+":
            table = for_label[chrom][start:end]
            tables.append(table)
            hot_seq=np.transpose(le.transform(list(seq[chrom][start:end])))
        else:
            dnatable = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
            table=rev_label[chrom][start:end][::-1]
            tables.append(table)
            rev=[dnatable[i] for i in seq[chrom][start:end]][::-1]
            hot_seq=np.transpose(le.transform(rev))
        variables.append(hot_seq.transpose())
    return np.array(variables),np.array(tables)


def get_dataset(region,seq,for_label,rev_label):
    #print(region)
    le = LabelBinarizer().fit(np.array([["A"], ["C"], ["T"], ["G"]]))
    tables=[]
    variables=[]
    chrom=region[0]
    #print(chrom)
    start=int(region[1])
    end=int(region[2])
    strand=region[3]
    if strand=="+":
        table = for_label[chrom][start:end]
        #print(table)
        tables.append(table)
        hot_seq=np.transpose(le.transform(list(seq[chrom][start:end])))
    else:
        dnatable = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        table=rev_label[chrom][start:end][::-1]
        # print(table)
        tables.append(table)
        rev=[dnatable[i] for i in seq[chrom][start:end]][::-1]
        hot_seq=np.transpose(le.transform(rev))
    variables.append(hot_seq.transpose())
       # print(hot_seq)
    return np.array(variables),np.array(tables)

def get_label(path):
    for_path = path + 'label_for'
    rev_path = path + 'label_rev'
    for_label = load(for_path)
    rev_label = load(rev_path)
    return for_label,rev_label















